"""Native EoMT and semantic/panoptic prediction contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import pytest

from equimo.finetune.vision import eomt_full_finetune
from equimo.serialization import load_weights, save_model
from equimo.vision.models import EoMT, VisionTransformer
from equimo.vision.models.eomt import EoMTMaskState, anneal_mask_state, mask_free_eomt
from equimo.vision.segmentation import (
    QuerySegmentationOutput,
    merge_semantic_crops,
    panoptic_predictions,
    resize_mask_logits,
    restore_panoptic_mask_logits,
    semantic_scores,
)


REFERENCE = Path(__file__).parent / "data" / "eomt_tiny_reference.npz"
PROVENANCE = REFERENCE.with_suffix(".json")


def _tiny_model(*, masked_attention: bool = True) -> EoMT:
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
        global_pos_embed_reg=True,
        eps=1e-6,
        act_layer="exactgelu",
        key=jr.PRNGKey(0),
    )
    return EoMT(
        backbone,
        3,
        num_queries=2,
        num_blocks=1,
        key=jr.PRNGKey(1),
        masked_attention=masked_attention,
    )


def _torch_name(path: str) -> str:
    if path == "queries":
        return "q.weight"
    if path == "backbone.cls_token":
        return "encoder.backbone.cls_token"
    if path == "backbone.reg_tokens":
        return "encoder.backbone.reg_token"
    if path == "backbone.global_pos_embed.weight":
        return "encoder.backbone.pos_embed"
    if path.startswith("backbone.blocks[0].blocks["):
        return (
            re.sub(
                r"backbone.blocks\[0\].blocks\[(\d+)\]",
                r"encoder.backbone.blocks.\1",
                path,
            )
            .replace(".prenorm.", ".norm1.")
            .replace(".norm.", ".norm2.")
        )
    if path.startswith("backbone."):
        return "encoder." + path
    if path.startswith("mask_head.layers["):
        return re.sub(
            r"mask_head.layers\[(\d+)\]",
            lambda match: "mask_head." + str(int(match.group(1)) * 2),
            path,
        )
    if path.startswith("upscale["):
        return re.sub(r"upscale\[(\d+)\]", r"upscale.\1", path)
    return path


def _load_reference_model(fixture) -> EoMT:
    consumed = set()

    def copy_author_tensor(path, leaf):
        if not eqx.is_array(leaf):
            return leaf
        own_name = jtu.keystr(path).lstrip(".")
        author_name = "state." + _torch_name(own_name)
        tensor = np.asarray(fixture[author_name])
        consumed.add(author_name)
        if own_name.endswith(".conv1.weight"):
            # PyTorch transposed convolution stores input/output channels and
            # spatial kernel orientation opposite to Equinox's convention.
            tensor = tensor.swapaxes(0, 1)[:, :, ::-1, ::-1]
        if tensor.shape != leaf.shape:
            assert tensor.size == leaf.size
            tensor = tensor.reshape(leaf.shape)
        return jnp.asarray(tensor, dtype=leaf.dtype)

    model = jtu.tree_map_with_path(copy_author_tensor, _tiny_model())
    author_params = {name for name in fixture.files if name.startswith("state.")} - {
        "state.attn_mask_probs",
        "state.encoder.pixel_mean",
        "state.encoder.pixel_std",
    }
    assert consumed == author_params
    return model


def test_pinned_author_query_outputs_and_attention_mask():
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance["author_revision"] == "7bd19ddd621c5c6adedcd260458a34783cd4a45f"
    with np.load(REFERENCE) as fixture:
        model = _load_reference_model(fixture)
        spatial = jnp.asarray(fixture["upscale_probe.input"])
        np.testing.assert_allclose(
            model.upscale[0].conv1(spatial),
            fixture["upscale_probe.conv1"],
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            model.upscale[0](spatial),
            fixture["upscale_probe.output"],
            rtol=2e-5,
            atol=2e-5,
        )
        image = jnp.asarray(fixture["image"])
        image = (image - jnp.asarray([0.485, 0.456, 0.406])[:, None, None]) / (
            jnp.asarray([0.229, 0.224, 0.225])[:, None, None]
        )

        attention = model._attention_mask(
            jnp.asarray(fixture["masked.mask.0"]),
            height=2,
            width=2,
            probability=jnp.asarray(1.0),
            key=jr.PRNGKey(7),
        )
        np.testing.assert_array_equal(attention, fixture["masked.attention.0"])

        for mode, candidate in (
            ("masked", model),
            ("unmasked", mask_free_eomt(model, EoMTMaskState(jnp.zeros(1)))),
        ):
            output = candidate(image, key=jr.PRNGKey(7), inference=True)
            predictions = (*output.auxiliary, output.final)
            trace = candidate.token_trace(image, key=jr.PRNGKey(7), inference=True)
            assert len(predictions) == (2 if mode == "masked" else 1)
            assert len(trace) == len(predictions)
            for index, prediction in enumerate(predictions):
                np.testing.assert_allclose(
                    trace[index],
                    fixture[f"{mode}.tokens.{index}"],
                    rtol=2e-5,
                    atol=2e-5,
                )
                np.testing.assert_allclose(
                    prediction.class_logits,
                    fixture[f"{mode}.class.{index}"],
                    rtol=2e-5,
                    atol=2e-5,
                )
                np.testing.assert_allclose(
                    prediction.mask_logits,
                    fixture[f"{mode}.mask.{index}"],
                    rtol=2e-5,
                    atol=2e-5,
                )


def test_default_query_insertion_and_rectangular_geometry():
    model = _tiny_model()
    image = jr.normal(jr.PRNGKey(3), (3, 16, 24))
    output = model(image, key=jr.PRNGKey(4))
    assert output.final.class_logits.shape == (2, 4)
    assert output.final.mask_logits.shape == (2, 4, 6)
    assert len(output.auxiliary) == 1
    joint = model.joint_features(image, key=jr.PRNGKey(4))
    assert joint.shape == (2 + 2 + 6, 8)
    np.testing.assert_array_equal(model.features(image, key=jr.PRNGKey(4)), joint)

    def final_mask_shape(prediction: QuerySegmentationOutput) -> tuple[int, ...]:
        return prediction.final.mask_logits.shape

    assert final_mask_shape(output) == (2, 4, 6)
    compiled = jax.jit(lambda x: model(x, key=jr.PRNGKey(4)).final.mask_logits)(image)
    np.testing.assert_allclose(compiled, output.final.mask_logits, atol=1e-6)


def test_rotary_queries_and_rectangular_patches():
    backbone = VisionTransformer(
        img_size=32,
        in_channels=3,
        dim=8,
        patch_size=16,
        num_heads=2,
        depths=[2, 2],
        reg_tokens=2,
        num_classes=0,
        dynamic_img_size=True,
        use_global_pos_embed=False,
        use_local_pos_embed=True,
        local_pos_embed_config_patch={
            "strategy": "period",
            "base": 100.0,
            "normalize_coords": "separate",
            "rescale_coords": 2.0,
            "dtype": jnp.float32,
            "periods_dtype": jnp.float32,
        },
        key=jr.PRNGKey(30),
    )
    model = EoMT(backbone, 3, num_queries=3, num_blocks=2, key=jr.PRNGKey(31))
    image = jr.normal(jr.PRNGKey(32), (3, 32, 48))
    tokens, height, width, rotary = backbone.prepare_tokens(
        image, key=jr.PRNGKey(33), inference=True
    )
    assert (height, width) == (2, 3)
    assert tokens.shape == (1 + 2 + 6, 8)
    assert rotary is not None
    assert rotary.cos.shape[0] == tokens.shape[0]
    output = model(image, key=jr.PRNGKey(34))
    assert output.final.mask_logits.shape == (3, 8, 12)
    assert len(output.auxiliary) == 2
    assert bool(jnp.all(jnp.isfinite(output.final.mask_logits)))


def test_rejects_unsupported_query_configuration():
    backbone = _tiny_model().backbone
    with pytest.raises(ValueError, match="between 1 and backbone depth"):
        EoMT(backbone, 3, num_blocks=3, key=jr.PRNGKey(4))
    with pytest.raises(ValueError, match="positive"):
        EoMT(backbone, 3, num_queries=0, key=jr.PRNGKey(4))
    with pytest.raises(ValueError, match="dynamic image-size"):
        EoMT(
            VisionTransformer(
                img_size=16,
                in_channels=3,
                dim=8,
                patch_size=8,
                num_heads=2,
                depths=[2],
                key=jr.PRNGKey(4),
            ),
            3,
            num_blocks=1,
            key=jr.PRNGKey(5),
        )
    with pytest.raises(ValueError, match="feature-only"):
        EoMT(
            VisionTransformer(
                img_size=16,
                in_channels=3,
                dim=8,
                patch_size=8,
                num_heads=2,
                depths=[2],
                num_classes=3,
                dynamic_img_size=True,
                key=jr.PRNGKey(4),
            ),
            3,
            num_blocks=1,
            key=jr.PRNGKey(5),
        )
    with pytest.raises(ValueError, match="explicit PRNG key"):
        _tiny_model()(jnp.zeros((3, 16, 16)))


def test_mask_schedule_and_terminal_transition():
    early = anneal_mask_state(0, (2,), (12,))
    middle = anneal_mask_state(7, (2,), (12,))
    terminal = anneal_mask_state(12, (2,), (12,))
    np.testing.assert_array_equal(early.probabilities, [1])
    np.testing.assert_allclose(middle.probabilities, [0.5**0.9], rtol=1e-6)
    np.testing.assert_array_equal(terminal.probabilities, [0])
    with pytest.raises(ValueError, match="terminal"):
        mask_free_eomt(_tiny_model(), middle)
    assert not mask_free_eomt(_tiny_model(), terminal).masked_attention
    with pytest.raises(ValueError, match="must follow"):
        anneal_mask_state(0, (0,), (0,))


def test_low_precision_logits_and_segmentation_adapters():
    model = jtu.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        _tiny_model(masked_attention=False),
    )
    output = model(jnp.zeros((3, 16, 16), dtype=jnp.bfloat16)).final
    assert output.class_logits.dtype == jnp.bfloat16
    assert output.mask_logits.dtype == jnp.bfloat16
    semantic = semantic_scores(output.mask_logits, output.class_logits)
    panoptic = panoptic_predictions(
        output.mask_logits, output.class_logits, stuff_classes=(0,), mask_threshold=0.5
    )
    assert semantic.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(semantic)))
    assert panoptic.class_ids.dtype == jnp.int32


def test_mask_polarity_and_per_query_removal():
    model = _tiny_model()
    logits = jnp.array(
        [
            [[2.0, -2.0], [2.0, -2.0]],
            [[-2.0, 2.0], [-2.0, 2.0]],
        ]
    )
    masked = model._attention_mask(
        logits, height=2, width=2, probability=jnp.asarray(1.0), key=jr.PRNGKey(4)
    )
    unmasked = model._attention_mask(
        logits, height=2, width=2, probability=jnp.asarray(0.0), key=jr.PRNGKey(4)
    )
    np.testing.assert_array_equal(masked[:2, 4:], [[1, 0, 1, 0], [0, 1, 0, 1]])
    assert bool(jnp.all(unmasked))


def test_semantic_reduction_and_resize_order():
    classes = jnp.array([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]])
    masks = jnp.array([[[2.0, 0.0]], [[-2.0, 4.0]]])
    actual = semantic_scores(masks, classes, output_size=(1, 3))
    expected = jnp.einsum(
        "qc,qhw->chw",
        jax.nn.softmax(classes, axis=-1)[:, :-1],
        jax.nn.sigmoid(resize_mask_logits(masks, (1, 3))),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    wrong_order = jax.image.resize(
        jax.nn.sigmoid(masks), (2, 1, 3), "linear", antialias=False
    )
    assert not np.allclose(
        wrong_order, jax.nn.sigmoid(resize_mask_logits(masks, (1, 3)))
    )


def test_panoptic_stuff_merge_thing_instances_and_void():
    masks = jnp.full((3, 2, 2), -10.0)
    masks = masks.at[0, 0, 0].set(10).at[1, 0, 1].set(10).at[2, 1, 0].set(10)
    classes = jnp.array([[10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 10.0, -10.0]])
    output = panoptic_predictions(masks, classes, stuff_classes=(0,))
    np.testing.assert_array_equal(output.class_ids, [[0, 0], [1, 2]])
    np.testing.assert_array_equal(output.segment_ids, [[0, 0], [1, -1]])
    compiled = jax.jit(lambda m, c: panoptic_predictions(m, c, stuff_classes=(0,)))(
        masks, classes
    )
    np.testing.assert_array_equal(compiled.segment_ids, output.segment_ids)


def test_panoptic_rejects_no_object_and_overlap():
    masks = jnp.array([[[10.0, 10.0]], [[10.0, -10.0]]])
    classes = jnp.array([[10.0, -10.0], [-10.0, 10.0]])
    output = panoptic_predictions(masks, classes, stuff_classes=())
    np.testing.assert_array_equal(output.class_ids, [[0, 0]])
    no_object = panoptic_predictions(
        masks, jnp.array([[-10.0, 10.0], [-10.0, 10.0]]), stuff_classes=()
    )
    np.testing.assert_array_equal(no_object.segment_ids, [[-1, -1]])

    # Query one wins a single pixel of its larger original region and is
    # rejected even though its class score exceeds the score threshold.
    overlap_masks = jnp.array([[[10.0, 10.0, -10.0]], [[2.0, 2.0, 10.0]]])
    overlap_classes = jnp.array([[10.0, -10.0, -10.0], [0.0, 2.0, -10.0]])
    rejected = panoptic_predictions(
        overlap_masks,
        overlap_classes,
        stuff_classes=(),
        mask_threshold=0.5,
    )
    np.testing.assert_array_equal(rejected.segment_ids, [[0, 0, -1]])


def test_semantic_crop_and_panoptic_geometry():
    crops = jnp.array([[[[1.0, 3.0]]], [[[5.0, 7.0]]]])
    merged = merge_semantic_crops(crops, ((0, 0), (0, 1)), (1, 3), (1, 3))
    np.testing.assert_allclose(merged, [[[1.0, 4.0, 7.0]]])
    uncovered = merge_semantic_crops(crops[:1], ((0, 0),), (1, 3), (1, 3))
    assert bool(jnp.isnan(uncovered[0, 0, 2]))
    mask = jnp.arange(4.0).reshape(1, 2, 2)
    restored = restore_panoptic_mask_logits(
        mask, padded_size=(4, 4), resized_size=(2, 4), output_size=(4, 8)
    )
    assert restored.shape == (1, 4, 8)


def test_full_encoder_plan_gradients_and_reload(tmp_path):
    terminal_state = anneal_mask_state(12, (2,), (12,))
    model = mask_free_eomt(_tiny_model(), terminal_state)
    plan = eomt_full_finetune(model)
    assert plan.report.trainable_params == plan.report.total_params
    paths = set(plan.report.target_paths)
    assert "queries" in paths
    assert any("patch_embed" in path for path in paths)
    assert any("upscale" in path for path in paths)
    image = jr.normal(jr.PRNGKey(9), (3, 16, 16))

    def loss(candidate):
        output = candidate(image, key=jr.PRNGKey(2), inference=False).final
        return jnp.sum(output.class_logits**2) + jnp.sum(output.mask_logits**2)

    gradients = eqx.filter_grad(loss)(model)
    for gradient in (
        gradients.backbone.patch_embed.proj.weight,
        gradients.backbone.blocks[0].blocks[0].attn.qkv.weight,
        gradients.queries,
        gradients.class_head.weight,
        gradients.mask_head.layers[0].weight,
        gradients.upscale[0].conv1.weight,
    ):
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.any(gradient != 0))

    optimizer = optax.sgd(1e-4)
    trainable_gradients = eqx.filter_grad(lambda params: loss(plan.combine(params)))(
        plan.trainable
    )
    updates, _ = optimizer.update(trainable_gradients, optimizer.init(plan.trainable))
    updated = plan.combine(eqx.apply_updates(plan.trainable, updates))
    assert not jnp.array_equal(
        updated.backbone.patch_embed.proj.weight, model.backbone.patch_embed.proj.weight
    )
    assert not jnp.array_equal(updated.queries, model.queries)
    assert not jnp.array_equal(updated.class_head.weight, model.class_head.weight)
    assert not jnp.array_equal(
        updated.upscale[0].conv1.weight, model.upscale[0].conv1.weight
    )

    checkpoint_config = {
        "num_classes": 3,
        "num_queries": 2,
        "num_blocks": 1,
        "masked_attention": False,
        "mask_step": int(terminal_state.step),
        "mask_probabilities": terminal_state.probabilities.tolist(),
    }
    path = save_model(
        tmp_path / "eomt",
        model,
        model_config=checkpoint_config,
        compression=False,
    )
    restored = load_weights(
        mask_free_eomt(_tiny_model(), terminal_state),
        path=path,
        expected_model_config=checkpoint_config,
    )
    np.testing.assert_allclose(
        restored(image, key=jr.PRNGKey(3)).final.mask_logits,
        model(image, key=jr.PRNGKey(3)).final.mask_logits,
        rtol=0,
        atol=0,
    )
