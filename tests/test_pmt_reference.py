"""Numerical PMD comparison with a pinned author implementation fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from equimo.core.layers.rotary import RotaryFactors
from equimo.vision.models.pmd import (
    PMTConfig,
    PMTFeatures,
    PlainMaskDecoder,
    ReferenceBatchNorm,
)


REFERENCE = Path(__file__).parent / "data" / "pmt_tiny_reference.npz"


def _reference_name(name: str) -> str:
    if name == "queries":
        return "decoder.q.weight"
    match = re.fullmatch(r"lateral\[(\d+)\]\.(.+)", name)
    if match:
        index, suffix = match.groups()
        if suffix == "scale":
            return f"decoder.scalers.{index}"
        if suffix.startswith("norm."):
            return f"norm.{index}.{suffix.removeprefix('norm.')}"
        layer, field = suffix.split(".")
        return f"decoder.lateral_projections.{index}.{'0' if layer == 'fc1' else '2'}.{field}"
    match = re.fullmatch(r"blocks\[(\d+)\]\.(.+)", name)
    if match:
        index, suffix = match.groups()
        suffix = (
            suffix.replace("prenorm.", "norm1.")
            .replace("norm.", "norm2.")
            .replace("attn.proj.", "attention.o_proj.")
            .replace("attn.query_bias", "attention.q_proj.bias")
            .replace("attn.value_bias", "attention.v_proj.bias")
            .replace("ls1.gamma", "layer_scale1.lambda1")
            .replace("ls2.gamma", "layer_scale2.lambda1")
            .replace("mlp.fc1.", "mlp.up_proj.")
            .replace("mlp.fc2.", "mlp.down_proj.")
        )
        return f"decoder.blocks.{index}.{suffix}"
    match = re.fullmatch(r"mask_head\.layers\[(\d+)\]\.(.+)", name)
    if match:
        return f"decoder.mask_head.{2 * int(match[1])}.{match[2]}"
    return "decoder." + re.sub(r"upscale\[(\d+)\]", r"upscale.\1", name)


def _load_decoder(fixture) -> tuple[PlainMaskDecoder, eqx.nn.State]:
    config = PMTConfig.small(
        dim=8,
        num_heads=2,
        hidden_dim=8,
        taps=(0, 1),
        num_queries=2,
        num_blocks=2,
        norm_layer="batchnorm",
    )
    decoder, state = eqx.nn.make_with_state(PlainMaskDecoder)(
        config,
        num_classes=3,
        num_prefix_tokens=2,
        patch_size=(8, 8),
        key=jr.PRNGKey(1),
    )
    consumed = set()

    def copy_author_tensor(path, leaf):
        if not eqx.is_array(leaf):
            return leaf
        name = jtu.keystr(path).lstrip(".")
        if name.endswith(".attn.qkv.weight"):
            prefix = _reference_name(name).removesuffix(".attn.qkv.weight")
            keys = [f"{prefix}.attention.{part}_proj.weight" for part in "qkv"]
            value = np.concatenate([fixture[key] for key in keys], axis=0)
            consumed.update(keys)
        else:
            key = _reference_name(name)
            value = np.asarray(fixture[key])
            consumed.add(key)
        if name.endswith(".conv1.weight"):
            value = value.swapaxes(0, 1)[:, :, ::-1, ::-1]
        if value.shape != leaf.shape:
            assert value.size == leaf.size
            value = value.reshape(leaf.shape)
        return jnp.asarray(value, dtype=leaf.dtype)

    decoder = jtu.tree_map_with_path(copy_author_tensor, decoder)
    for index, lateral in enumerate(decoder.lateral):
        norm = lateral.norm
        for state_index, suffix in (
            (norm.mean_index, "running_mean"),
            (norm.variance_index, "running_var"),
            (norm.count_index, "num_batches_tracked"),
        ):
            key = f"norm.{index}.{suffix}"
            state = state.set(state_index, jnp.asarray(fixture[key]))
            consumed.add(key)
    expected = {key for key in fixture.files if key.startswith(("decoder.", "norm."))}
    assert consumed == expected - {"decoder.attn_mask_probs"}
    return decoder, state


def test_author_sync_batchnorm_training_statistics():
    with np.load(REFERENCE) as fixture:
        norm, state = eqx.nn.make_with_state(ReferenceBatchNorm)(8)
        norm = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            norm,
            (
                jnp.asarray(fixture["norm.0.weight"]),
                jnp.asarray(fixture["norm.0.bias"]),
            ),
        )
        for index, suffix in (
            (norm.mean_index, "running_mean"),
            (norm.variance_index, "running_var"),
            (norm.count_index, "num_batches_tracked"),
        ):
            state = state.set(index, jnp.asarray(fixture[f"norm.0.{suffix}"]))
        values = jnp.asarray(fixture["batchnorm_train.input"])
        output, updated = jax.vmap(
            lambda sample: norm(sample, state, inference=False),
            axis_name="pmt_batch",
            out_axes=(0, None),
        )(values)
        np.testing.assert_allclose(
            output, fixture["batchnorm_train.output"], atol=2e-5, rtol=2e-5
        )
        for index, suffix in (
            (norm.mean_index, "running_mean"),
            (norm.variance_index, "running_var"),
            (norm.count_index, "num_batches_tracked"),
        ):
            np.testing.assert_allclose(
                updated.get(index), fixture[f"batchnorm_train.{suffix}"], atol=2e-6
            )


@pytest.mark.parametrize("masked", [True, False])
def test_author_decoder_numerical_parity(masked: bool):
    provenance = json.loads(REFERENCE.with_suffix(".json").read_text())
    assert provenance["author_revision"] == "0e803722aa5737a242b383dec1b90c2c66b86baa"
    mode = "masked" if masked else "unmasked"
    with np.load(REFERENCE) as fixture:
        decoder, state = _load_decoder(fixture)
        decoder = eqx.tree_at(lambda model: model.masked_attention, decoder, masked)
        levels = tuple(jnp.asarray(fixture[f"level.{index}"]) for index in range(2))
        cos = jnp.asarray(fixture["rope.cos"])
        sin = jnp.asarray(fixture["rope.sin"])
        rotary = RotaryFactors(
            sin=jnp.concatenate((jnp.zeros((2, 4)), sin)),
            cos=jnp.concatenate((jnp.ones((2, 4)), cos)),
            layout="split_half",
        )
        features = PMTFeatures(
            levels=levels,
            rotary=rotary,
            grid_size=(2, 3),
            patch_size=(8, 8),
            prefix_tokens=("cls", "register_0"),
            taps=(0, 1),
            positional_configuration=(),
            backbone_id="fixture",
            backbone_digest=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            input_view="native",
        )
        projected = []
        for index, (layer, level) in enumerate(
            zip(decoder.lateral, levels, strict=True)
        ):
            normalized, _ = layer.norm(level.T, state, inference=True)
            np.testing.assert_allclose(
                normalized.T, fixture[f"normalized.{index}"], atol=2e-6, rtol=2e-6
            )
            value, _ = layer(level, state, inference=True)
            np.testing.assert_allclose(
                value, fixture[f"lateral.{index}"], atol=2e-6, rtol=2e-6
            )
            projected.append(value)
        np.testing.assert_allclose(
            jnp.sum(jnp.stack(projected), axis=0),
            fixture["fused"],
            atol=3e-6,
            rtol=3e-6,
        )
        result, returned_state = decoder(features, state, key=jr.PRNGKey(12))
        assert returned_state is state
        predictions = (*result.auxiliary, result.final)
        assert len(predictions) == (3 if masked else 1)
        for index, prediction in enumerate(predictions):
            np.testing.assert_allclose(
                prediction.class_logits,
                fixture[f"{mode}.class.{index}"],
                atol=2e-4,
                rtol=2e-4,
            )
            np.testing.assert_allclose(
                prediction.mask_logits,
                fixture[f"{mode}.mask.{index}"],
                atol=2e-4,
                rtol=2e-4,
            )
