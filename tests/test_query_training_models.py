"""Actual query-loss updates and checkpoint resumption through tiny native models."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.finetune.vision import eomt_full_finetune, pmt_head_finetune
from equimo.serialization import load_weights, save_model
from equimo.vision.models import EoMT, PMT, PMTConfig, VisionTransformer
from equimo.vision.models.eomt import anneal_mask_state
from equimo.vision.models.pmd import anneal_pmt_mask_state
from equimo.vision.pmt_checkpoint import load_pmt_checkpoint, save_pmt_checkpoint
from equimo.vision.query_training import (
    QueryLossConfig,
    QueryTargets,
    prepare_query_loss,
    query_segmentation_loss,
    reduce_query_losses,
)


def _model(kind):
    backbone = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[2],
        reg_tokens=1,
        num_classes=0,
        dynamic_img_size=True,
        use_local_pos_embed=kind == "pmt",
        global_pos_embed_reg=True,
        key=jr.PRNGKey(61),
    )
    if kind == "eomt":
        return EoMT(backbone, 2, num_queries=3, num_blocks=1, key=jr.PRNGKey(62)), None
    config = PMTConfig(
        dim=8,
        num_heads=2,
        hidden_dim=8,
        taps=(0, 1),
        num_queries=3,
        num_blocks=1,
        norm_layer="batchnorm",
    )
    return eqx.nn.make_with_state(PMT)(
        backbone, 2, config=config, key=jr.PRNGKey(62), backbone_id="synthetic-test"
    )


def _equal(left, right):
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if eqx.is_array(x):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kind", ["eomt", "pmt"])
def test_query_loss_update_respects_plan_and_resumes_from_checkpoint(kind, tmp_path):
    model, state = _model(kind)
    plan = (
        eomt_full_finetune(model)
        if kind == "eomt"
        else pmt_head_finetune(model, state=state)
    )
    images = jr.normal(jr.PRNGKey(63), (3, 3, 16, 16))
    valid = jnp.array([True, True, False])
    masks = jnp.stack((jnp.indices((4, 4))[0] < 2, jnp.indices((4, 4))[0] >= 2))
    target = QueryTargets(
        jnp.array([0, 1]), masks, jnp.ones(2, bool), jnp.ones((4, 4), bool)
    )
    targets = jax.tree.map(lambda x: jnp.stack((x, x, x)), target)
    targets = eqx.tree_at(lambda t: t.example_valid, targets, valid)
    config = QueryLossConfig(matching_num_points=6, loss_num_points=5)

    def objective(trainable, state, key, step, batch_images):
        candidate = plan.combine(trainable)
        keys = jr.split(key, 6)
        if kind == "pmt":
            mask_state = anneal_pmt_mask_state(step, (0,), (4,))
            outputs, new_state = jax.vmap(
                lambda image, keep, k: candidate(
                    image,
                    state,
                    inference=False,
                    key=k,
                    mask_state=mask_state,
                    example_valid=keep,
                ),
                axis_name="pmt_batch",
                out_axes=(0, None),
            )(batch_images, valid, keys[:3])
        else:
            mask_state = anneal_mask_state(step, (0,), (4,))
            outputs = jax.vmap(
                lambda image, k: candidate(
                    image, inference=False, key=k, mask_state=mask_state
                )
            )(batch_images, keys[:3])
            new_state = None
        contexts = jax.vmap(
            lambda out, tgt, k: prepare_query_loss(out, tgt, key=k, config=config)
        )(outputs, targets, keys[3:])
        losses = jax.vmap(query_segmentation_loss)(outputs, targets, contexts)
        result = reduce_query_losses(losses)
        return result.total, (new_state, result)

    # Images and state stay explicit arguments; see the query-training docs.
    train_step = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
    key = jr.PRNGKey(64)
    (value, (updated_state, result)), gradient = train_step(
        plan.trainable, state, key, jnp.array(1), images
    )
    assert bool(jnp.isfinite(value)) and bool(result.is_valid)
    assert int(result.final.supervised_examples) == 2
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
    updated_params = eqx.apply_updates(
        plan.trainable, jax.tree.map(lambda g: -1e-3 * g, gradient)
    )
    updated = plan.combine(updated_params)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(updated_params),
            jax.tree.leaves(plan.trainable),
            strict=True,
        )
    )
    if kind == "pmt":
        _equal(updated.backbone, model.backbone)
        for lateral in model.head.lateral:
            assert int(updated_state.get(lateral.norm.count_index)) == 1
        path = save_pmt_checkpoint(
            tmp_path / "model",
            updated,
            state=updated_state,
            mask_state=anneal_pmt_mask_state(2, (0,), (4,)),
        )
        template, template_state = _model(kind)
        restored, restored_state, restored_mask = load_pmt_checkpoint(
            path, template, state=template_state
        )
        next_step = restored_mask.step
        restored_plan = pmt_head_finetune(restored, state=restored_state)
    else:
        assert not np.array_equal(
            updated.backbone.patch_embed.proj.weight,
            model.backbone.patch_embed.proj.weight,
        )
        path = save_model(
            tmp_path / "model", updated, model_config={"step": 2}, compression=False
        )
        template, _ = _model(kind)
        restored = load_weights(
            template, path=path, expected_model_config={"step": 2}, inference_mode=False
        )
        restored_state, next_step = None, jnp.array(2)
        restored_plan = eomt_full_finetune(restored)

    _equal(restored_plan.trainable, updated_params)
    next_key, _ = jr.split(key)
    uninterrupted = train_step(
        updated_params, updated_state, next_key, jnp.array(2), images
    )
    resumed = train_step(
        restored_plan.trainable, restored_state, next_key, next_step, images
    )
    _equal(uninterrupted, resumed)
