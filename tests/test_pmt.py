"""Frozen-feature PMD, normalization, and bound checkpoint behavior."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from equimo.finetune.vision import pmt_head_finetune
from equimo.utils import nearest_power_of_2_divisor
from equimo.vision.models import PMT, PMTConfig, PlainMaskDecoder, VisionTransformer
from equimo.vision.models.eomt import anneal_mask_state
from equimo.vision.models.pmd import (
    PMTMaskState,
    ReferenceBatchNorm,
    anneal_pmt_mask_state,
)
from equimo.vision.models.pmt import mask_free_pmt
from equimo.vision.pmt_checkpoint import (
    load_pmt_checkpoint,
    load_pmt_decoder,
    save_pmt_checkpoint,
    save_pmt_decoder,
)
from equimo.vision.segmentation import QuerySegmentationOutput


def _model(seed: int = 0, *, norm: str = "groupnorm") -> tuple[PMT, eqx.nn.State]:
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
        use_local_pos_embed=True,
        global_pos_embed_reg=True,
        key=jr.PRNGKey(seed),
    )
    config = PMTConfig.small(
        dim=8,
        num_heads=2,
        hidden_dim=8,
        taps=(0, 1),
        num_blocks=2,
        num_queries=2,
        norm_layer=norm,
    )
    return eqx.nn.make_with_state(PMT)(
        backbone, 3, key=jr.PRNGKey(1), config=config, backbone_id="tiny-base"
    )


def test_presets_and_group_count():
    small = PMTConfig.small()
    assert (small.dim, small.num_heads, small.hidden_dim, small.taps) == (
        384,
        6,
        384,
        (2, 5, 8, 11),
    )
    assert (PMTConfig.base().dim, PMTConfig.base().taps) == (768, (2, 5, 8, 11))
    assert (PMTConfig.large().dim, PMTConfig.large().taps) == (1024, (5, 11, 17, 23))
    assert PMTConfig.small(num_queries=200).num_queries == 200
    assert PMTConfig.small(norm_kwargs={"eps": 1e-4}).norm_kwargs == (("eps", 1e-4),)
    assert [nearest_power_of_2_divisor(dim, 32) for dim in (384, 768, 1024)] == [32] * 3
    assert nearest_power_of_2_divisor(24, 32) == 8
    decoder = PlainMaskDecoder(
        PMTConfig.small(
            dim=24,
            num_heads=2,
            hidden_dim=24,
            taps=(0, 1),
            num_queries=2,
            num_blocks=1,
        ),
        num_classes=3,
        num_prefix_tokens=2,
        patch_size=(8, 8),
        key=jr.PRNGKey(0),
    )
    assert decoder.lateral[0].norm.groups == 8
    with pytest.raises(ValueError, match="norm_layer"):
        PMTConfig(norm_layer="unknown")
    with pytest.raises(ValueError, match="normalization option"):
        PMTConfig(norm_kwargs={"momentum": 0.1})


def test_groupnorm_no_state_batch_independence_and_geometry():
    model, state = _model()
    assert not model.is_stateful()
    assert jtu.tree_leaves(state) == []
    assert model.head.lateral[0].norm.groups == 8
    key = jr.PRNGKey(7)
    first = jr.normal(jr.PRNGKey(3), (3, 16, 24))
    second = jr.normal(jr.PRNGKey(4), (3, 16, 24))
    alone = model(first, key=key)
    with_state, returned = model(first, state, key=key)
    np.testing.assert_array_equal(alone.final.mask_logits, with_state.final.mask_logits)
    assert returned is state
    batched = jax.vmap(lambda image: model(image, key=key))(jnp.stack((first, second)))
    np.testing.assert_allclose(
        alone.final.mask_logits, batched.final.mask_logits[0], atol=1e-6
    )
    assert alone.final.mask_logits.shape == (2, 4, 6)
    assert alone.final.class_logits.shape == (2, 4)

    assert len(alone.auxiliary) == 2
    features = model.encode(first)
    assert features.grid_size == (2, 3)
    assert features.prefix_tokens == ("cls", "register_0")
    assert all(level.shape == (8, 8) for level in features.levels)
    np.testing.assert_array_equal(model.encoder_features(first), features.levels[-1])
    np.testing.assert_array_equal(model.features(first, key=key), features.levels[-1])

    def final_mask_shape(prediction: QuerySegmentationOutput) -> tuple[int, ...]:
        return prediction.final.mask_logits.shape

    assert final_mask_shape(alone) == (2, 4, 6)
    cached = model.decode(features, key=key)
    np.testing.assert_array_equal(cached.final.mask_logits, alone.final.mask_logits)
    compiled = eqx.filter_jit(lambda image: model(image, key=key))(first)
    np.testing.assert_allclose(
        compiled.final.mask_logits, alone.final.mask_logits, atol=2e-5
    )


def test_cached_features_reject_a_different_encoder_with_the_same_name():
    original, _ = _model(seed=0)
    different, _ = _model(seed=99)
    image = jr.normal(jr.PRNGKey(17), (3, 16, 24))

    features = original.encode(image)

    assert features.backbone_id == different.backbone_id
    assert features.backbone_digest == original.backbone_digest
    assert features.backbone_digest != different.backbone_digest
    with pytest.raises(ValueError, match="encoder"):
        different.decode(features, key=jr.PRNGKey(18))


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm", "none"])
def test_alternative_lateral_norms_are_stateless(norm: str):
    model, state = _model(norm=norm)
    assert not model.is_stateful()
    assert jtu.tree_leaves(state) == []
    output = model(jnp.ones((3, 16, 24)), key=jr.PRNGKey(19))
    assert output.final.mask_logits.shape == (2, 4, 6)
    assert bool(jnp.all(jnp.isfinite(output.final.mask_logits)))


def test_reference_batchnorm_statistics_and_evaluation():
    norm, state = eqx.nn.make_with_state(ReferenceBatchNorm)(2)
    samples = jnp.asarray(
        [
            [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
            [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]],
        ]
    )
    output, updated = jax.vmap(
        lambda sample: norm(sample, state, inference=False),
        axis_name="pmt_batch",
        out_axes=(0, None),
    )(samples)
    data = np.asarray(samples).transpose(1, 0, 2).reshape(2, -1)
    mean = data.mean(axis=1)
    biased = data.var(axis=1)
    unbiased = data.var(axis=1, ddof=1)
    expected = (np.asarray(samples) - mean[None, :, None]) / np.sqrt(
        biased[None, :, None] + 1e-5
    )
    np.testing.assert_allclose(output, expected, atol=1e-6)
    np.testing.assert_allclose(updated.get(norm.mean_index), 0.1 * mean, atol=1e-6)
    np.testing.assert_allclose(
        updated.get(norm.variance_index), 0.9 + 0.1 * unbiased, atol=1e-6
    )
    assert int(updated.get(norm.count_index)) == 1
    evaluation, same = norm(samples[0], updated, inference=True)
    expected_eval = (np.asarray(samples[0]) - 0.1 * mean[:, None]) / np.sqrt(
        (0.9 + 0.1 * unbiased)[:, None] + 1e-5
    )
    np.testing.assert_allclose(evaluation, expected_eval, atol=1e-6)
    assert same is updated


def test_reference_batchnorm_counts_images_with_one_token_each():
    norm, state = eqx.nn.make_with_state(ReferenceBatchNorm)(2)
    samples = jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]])
    output, updated = jax.vmap(
        lambda sample: norm(sample, state, inference=False),
        axis_name="pmt_batch",
        out_axes=(0, None),
    )(samples)
    np.testing.assert_allclose(output[:, 0, 0], [-1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(updated.get(norm.mean_index), [0.2, 0.3])
    with pytest.raises(ValueError, match="at least two"):
        jax.vmap(
            lambda sample: norm(sample, state, inference=False),
            axis_name="pmt_batch",
            out_axes=(0, None),
        )(samples[:1])


def test_decoder_plan_gradients_and_frozen_encoder():
    model, _ = _model()
    plan = pmt_head_finetune(model)
    assert plan.report.trainable_params < plan.report.total_params
    assert all(path.startswith("head.") for path in plan.report.target_paths)
    assert any(
        "lateral" in path and ".norm." in path for path in plan.report.target_paths
    )
    assert any("blocks" in path for path in plan.report.target_paths)
    assert any("queries" in path for path in plan.report.target_paths)
    image = jr.normal(jr.PRNGKey(5), (3, 16, 16))
    before = model.encode(image)

    def loss(trainable):
        candidate = eqx.combine(trainable, plan.frozen)
        prediction = candidate(image, key=jr.PRNGKey(6)).final
        return jnp.sum(prediction.class_logits**2) + jnp.sum(prediction.mask_logits**2)

    value, gradient = eqx.filter_value_and_grad(loss)(plan.trainable)
    assert bool(jnp.isfinite(value))
    assert gradient.head.queries is not None
    assert gradient.head.lateral[0].norm.weight is not None
    updates = jtu.tree_map(lambda g: -1e-4 * g if eqx.is_array(g) else None, gradient)
    changed = eqx.combine(eqx.apply_updates(plan.trainable, updates), plan.frozen)
    assert not np.array_equal(changed.head.queries, model.head.queries)
    for old, new in zip(
        jtu.tree_leaves(model.backbone), jtu.tree_leaves(changed.backbone)
    ):
        if eqx.is_array(old):
            np.testing.assert_array_equal(old, new)
    after = changed.encode(image)
    for old, new in zip(before.levels, after.levels, strict=True):
        np.testing.assert_array_equal(old, new)


def test_mask_annealing_and_state_errors():
    model, _ = _model()
    state = anneal_pmt_mask_state(5, (0, 5), (5, 10))
    np.testing.assert_array_equal(state.probabilities, [0, 1])
    np.testing.assert_array_equal(
        state.probabilities,
        anneal_mask_state(5, (0, 5), (5, 10)).probabilities,
    )
    with pytest.raises(ValueError, match="terminal"):
        mask_free_pmt(model, state)
    terminal = anneal_pmt_mask_state(10, (0, 5), (5, 10))
    unmasked = mask_free_pmt(model, terminal)
    assert not unmasked.head.masked_attention
    prediction = unmasked(jnp.ones((3, 16, 16)))
    assert prediction.auxiliary == ()
    assert prediction.final.mask_logits.shape == (2, 4, 4)
    bn_model, _ = _model(norm="batchnorm")
    with pytest.raises(ValueError, match="state"):
        bn_model(jnp.ones((3, 16, 16)), key=jr.PRNGKey(1))
    with pytest.raises(ValueError, match="state"):
        pmt_head_finetune(bn_model)
    with pytest.raises(TypeError, match="state"):
        model(jnp.ones((3, 16, 16)), object(), key=jr.PRNGKey(1))


@pytest.mark.parametrize("norm", ["groupnorm", "batchnorm"])
def test_native_checkpoints_and_exact_base_rejection(tmp_path: Path, norm: str):
    model, state = _model(norm=norm)
    actual_state = state if norm == "batchnorm" else None
    if norm == "batchnorm":
        image = jr.normal(jr.PRNGKey(8), (2, 3, 16, 16))
        _, actual_state = jax.vmap(
            lambda sample: model(sample, state, key=jr.PRNGKey(9), inference=False),
            axis_name="pmt_batch",
            out_axes=(0, None),
        )(image)
    saved_mask_state = anneal_pmt_mask_state(5, (0, 5), (5, 10))
    full = save_pmt_checkpoint(
        tmp_path / "full", model, state=actual_state, mask_state=saved_mask_state
    )
    decoder = save_pmt_decoder(
        tmp_path / "decoder", model, state=actual_state, mask_state=saved_mask_state
    )
    random_template, random_state = _model(seed=99, norm=norm)
    restored, restored_state, mask_state = load_pmt_checkpoint(
        full, random_template, state=random_state if norm == "batchnorm" else None
    )
    assert restored.backbone_digest == model.backbone_digest
    np.testing.assert_array_equal(mask_state.probabilities, [0, 1])
    assert int(mask_state.step) == 5
    image = jr.normal(jr.PRNGKey(2), (3, 16, 16))
    original_output = model(image, actual_state, key=jr.PRNGKey(3))
    new_output = restored(image, restored_state, key=jr.PRNGKey(3))
    np.testing.assert_array_equal(
        original_output[0].final.mask_logits, new_output[0].final.mask_logits
    )
    cached = model.encode(image)
    restored_cached = restored.decode(cached, restored_state, key=jr.PRNGKey(3))
    np.testing.assert_array_equal(
        original_output[0].final.mask_logits,
        restored_cached[0].final.mask_logits,
    )
    matching_template, matching_state = _model(norm=norm)
    restored_decoder, decoder_state, decoded_mask_state = load_pmt_decoder(
        decoder,
        matching_template,
        state=matching_state if norm == "batchnorm" else None,
    )
    np.testing.assert_array_equal(decoded_mask_state.probabilities, [0, 1])
    assert int(decoded_mask_state.step) == 5
    decoded = restored_decoder(image, decoder_state, key=jr.PRNGKey(3))
    np.testing.assert_array_equal(
        original_output[0].final.mask_logits, decoded[0].final.mask_logits
    )
    with pytest.raises(ValueError, match="base identity"):
        load_pmt_decoder(
            decoder,
            random_template,
            state=random_state if norm == "batchnorm" else None,
        )
    wrong_norm, wrong_state = _model(
        norm="batchnorm" if norm == "groupnorm" else "groupnorm"
    )
    with pytest.raises(ValueError):
        load_pmt_decoder(
            decoder,
            wrong_norm,
            state=wrong_state if wrong_norm.is_stateful() else None,
        )
    wrong_ontology, ontology_state = eqx.nn.make_with_state(PMT)(
        matching_template.backbone,
        3,
        key=jr.PRNGKey(1),
        config=matching_template.head.config,
        backbone_id="tiny-base",
        class_ontology=("a", "b", "c"),
    )
    with pytest.raises(ValueError, match="configuration"):
        load_pmt_decoder(
            decoder,
            wrong_ontology,
            state=ontology_state if norm == "batchnorm" else None,
        )


def test_low_precision_and_direct_state_round_trip():
    model, state = _model()
    model = jtu.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        model,
    )
    image = jnp.ones((3, 16, 16), dtype=jnp.bfloat16)
    direct = model(image, key=jr.PRNGKey(2))
    paired, returned = model(image, state, key=jr.PRNGKey(2))
    assert direct.final.mask_logits.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(direct.final.mask_logits)))
    np.testing.assert_array_equal(direct.final.mask_logits, paired.final.mask_logits)
    assert returned is state
    with pytest.raises(ValueError, match="length"):
        model(image, key=jr.PRNGKey(2), mask_state=PMTMaskState(jnp.ones((1,))))
