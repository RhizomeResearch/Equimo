import importlib.util
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.core.layers import with_attention_implementation
from equimo.language.models import (
    CausalTextTransformer,
    TextTransformerEncoder,
    sam31_text,
)
from equimo.registry import get_model_cls


KEY = jr.PRNGKey(0)
GPU = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="cuDNN attention requires a CUDA GPU"
)


def test_text_transformer_encoder_forward_and_features_shape():
    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=128,
        key=KEY,
    )
    ids = jnp.array([1, 2, 3, 0, 0])
    padding_mask = jnp.array([0, 0, 0, 1, 1])

    features = model.features(ids, padding_mask, key=KEY, inference=True)
    pooled = model(ids, padding_mask, key=KEY, inference=True)

    assert features.shape == (5, 16)
    assert pooled.shape == (16,)
    assert jnp.all(jnp.isfinite(features))
    assert jnp.all(jnp.isfinite(pooled))


def test_text_transformer_encoder_ignores_padded_token_ids():
    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=128,
        key=KEY,
    )
    ids_a = jnp.array([1, 2, 3, 4, 5])
    ids_b = jnp.array([1, 2, 3, 17, 18])
    padding_mask = jnp.array([0, 0, 0, 1, 1])

    features_a = model.features(ids_a, padding_mask, key=KEY, inference=True)
    features_b = model.features(ids_b, padding_mask, key=KEY, inference=True)
    pooled_a = model(ids_a, padding_mask, key=KEY, inference=True)
    pooled_b = model(ids_b, padding_mask, key=KEY, inference=True)

    assert jnp.allclose(features_a[:3], features_b[:3], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(pooled_a, pooled_b, rtol=1e-6, atol=1e-6)

    intermediates_a = model.intermediate_features(
        ids_a,
        padding_mask,
        key=KEY,
        inference=True,
        n_last_blocks=2,
    )
    intermediates_b = model.intermediate_features(
        ids_b,
        padding_mask,
        key=KEY,
        inference=True,
        n_last_blocks=2,
    )

    assert len(intermediates_a) == len(intermediates_b) == 2
    for intermediate_a, intermediate_b in zip(
        intermediates_a, intermediates_b, strict=True
    ):
        assert jnp.allclose(
            intermediate_a[:3], intermediate_b[:3], rtol=1e-6, atol=1e-6
        )


def test_language_model_registered_by_modality():
    assert (
        get_model_cls("text_transformer_encoder", modality="language")
        is TextTransformerEncoder
    )


def test_text_transformer_encoder_preserves_bfloat16_dtype():
    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=1,
        num_heads=2,
        vocab_size=32,
        key=KEY,
    )
    model = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        model,
    )
    ids = jnp.asarray([1, 2, 0, 0])
    padding_mask = jnp.asarray([0, 0, 1, 1])

    features = model.features(ids, padding_mask, key=KEY, inference=True)
    pooled = model(ids, padding_mask, key=KEY, inference=True)

    assert features.dtype == jnp.bfloat16
    assert pooled.dtype == jnp.bfloat16


def _causal_text_transformer(**overrides):
    cfg = dict(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=64,
        context_length=6,
        out_dim=8,
        act_layer="exactgelu",
        key=KEY,
    )
    cfg.update(overrides)
    return CausalTextTransformer(**cfg)


CAUSAL_IDS = jnp.array([5, 7, 11, 13, 0, 0])
CAUSAL_PADDING = jnp.array([False, False, False, False, True, True])


def test_causal_text_transformer_returns_per_token_features():
    model = _causal_text_transformer()

    features = model.features(CAUSAL_IDS, CAUSAL_PADDING)
    called = model(CAUSAL_IDS, CAUSAL_PADDING)

    assert features.shape == (6, 8)
    assert jnp.all(jnp.isfinite(features))
    assert jnp.array_equal(features, called)
    assert get_model_cls("causal_text_transformer", modality="language") is type(model)


def test_causal_text_transformer_without_projection_keeps_width():
    model = _causal_text_transformer(out_dim=None)

    assert model.features(CAUSAL_IDS, CAUSAL_PADDING).shape == (6, 16)


@pytest.mark.parametrize("changed_position", [1, 2, 3])
def test_causal_text_transformer_later_tokens_leave_earlier_outputs_bit_exact(
    changed_position,
):
    model = _causal_text_transformer()
    changed_ids = CAUSAL_IDS.at[changed_position:].add(17)

    original = np.asarray(model.features(CAUSAL_IDS, CAUSAL_PADDING))
    changed = np.asarray(model.features(changed_ids, CAUSAL_PADDING))

    np.testing.assert_array_equal(
        original[:changed_position], changed[:changed_position]
    )
    assert not np.array_equal(original[changed_position:], changed[changed_position:])


def test_causal_text_transformer_padded_ids_leave_valid_outputs_bit_exact():
    model = _causal_text_transformer()
    changed_ids = CAUSAL_IDS.at[4:].set(jnp.array([31, 47]))

    original = np.asarray(model.features(CAUSAL_IDS, CAUSAL_PADDING))
    changed = np.asarray(model.features(changed_ids, CAUSAL_PADDING))

    np.testing.assert_array_equal(original[:4], changed[:4])


def test_causal_text_transformer_treats_token_id_zero_as_a_real_token():
    model = _causal_text_transformer()
    ids = CAUSAL_IDS.at[2].set(0)
    unpadded = jnp.zeros_like(CAUSAL_PADDING)

    features = np.asarray(model.features(ids, unpadded))
    expected = np.asarray(
        model.features(ids, unpadded.at[3:].set(True))[:3],
    )

    np.testing.assert_array_equal(features[:3], expected)
    assert not np.array_equal(
        features[2], np.asarray(model.features(CAUSAL_IDS, unpadded))[2]
    )


def test_causal_text_transformer_jit_and_vmap_match_eager():
    model = _causal_text_transformer()
    eager = model.features(CAUSAL_IDS, CAUSAL_PADDING)

    def features(token_ids, padding_mask):
        return model.features(token_ids, padding_mask)

    compiled = jax.jit(features)(CAUSAL_IDS, CAUSAL_PADDING)
    batched = jax.jit(jax.vmap(features))(
        jnp.stack([CAUSAL_IDS, CAUSAL_IDS]),
        jnp.stack([CAUSAL_PADDING, CAUSAL_PADDING]),
    )

    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)
    assert batched.shape == (2, 6, 8)
    assert jnp.allclose(batched[0], eager, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(batched[1], eager, rtol=1e-6, atol=1e-6)


def test_causal_text_transformer_preserves_bfloat16_dtype():
    model = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        _causal_text_transformer(),
    )

    features = model.features(CAUSAL_IDS, CAUSAL_PADDING)

    assert features.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(features.astype(jnp.float32)))


def _bfloat16_causal_xla_and_cudnn():
    model = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        _causal_text_transformer(dim=128, depth=1, context_length=32, out_dim=16),
    )
    return model, with_attention_implementation(model, "cudnn")


CUDNN_IDS = jr.randint(jr.PRNGKey(1), (32,), 1, 64)
CUDNN_PADDING = jnp.arange(32) >= 20


@GPU
def test_cudnn_causal_text_transformer_matches_xla():
    weights = jr.normal(jr.PRNGKey(2), (32, 16), jnp.float32)

    def loss(model):
        features = model.features(CUDNN_IDS, CUDNN_PADDING)
        return jnp.sum(features.astype(jnp.float32) * weights)

    def evaluate(model):
        output = eqx.filter_jit(lambda m: m.features(CUDNN_IDS, CUDNN_PADDING))(model)
        return output, eqx.filter_jit(eqx.filter_grad(loss))(model)

    model, fused = _bfloat16_causal_xla_and_cudnn()
    expected_output, expected_gradients = evaluate(model)
    output, gradients = evaluate(fused)

    assert output.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        output.astype(jnp.float32),
        expected_output.astype(jnp.float32),
        rtol=0.05,
        atol=0.02,
    )
    for actual, expected in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(expected_gradients), strict=True
    ):
        actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
        assert jnp.linalg.norm(actual - expected) <= 0.03 * jnp.linalg.norm(expected)


@GPU
def test_cudnn_causal_text_transformer_later_tokens_leave_earlier_outputs_bit_exact():
    _, fused = _bfloat16_causal_xla_and_cudnn()
    features = eqx.filter_jit(
        lambda model, token_ids: model.features(token_ids, CUDNN_PADDING)
    )

    original = np.asarray(features(fused, CUDNN_IDS).astype(jnp.float32))
    changed = np.asarray(features(fused, CUDNN_IDS.at[10:].add(17)).astype(jnp.float32))

    np.testing.assert_array_equal(original[:10], changed[:10])
    assert not np.array_equal(original[10:20], changed[10:20])


def test_sam31_text_factory_builds_the_documented_configuration():
    model = eqx.filter_eval_shape(sam31_text)
    output = eqx.filter_eval_shape(
        lambda: sam31_text()(jnp.zeros(32, dtype=jnp.int32), jnp.zeros(32, dtype=bool))
    )
    parameters = sum(
        leaf.size
        for leaf in jax.tree.leaves(model)
        if isinstance(leaf, jax.ShapeDtypeStruct)
    )

    assert output.shape == (32, 256)
    assert model.position_embedding.shape == (32, 1024)
    assert model.token_embedding.weight.shape == (49408, 1024)
    assert len(model.transformer.blocks) == 24
    assert 350_000_000 < parameters < 358_000_000


def _tiny_reference():
    data = Path(__file__).parent / "data"
    config = json.loads((data / "sam3_tiny_reference.json").read_text())["config"]
    return config, np.load(data / "sam3_tiny_reference.npz")


def _tiny_reference_text_model(config, fixture):
    spec = importlib.util.spec_from_file_location(
        "sam31_converter",
        Path(__file__).resolve().parents[1] / "models" / "sam31.py",
    )
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    state = {
        name.removeprefix("state."): fixture[name]
        for name in fixture.files
        if name.startswith("state.")
    }
    template = CausalTextTransformer(
        dim=config["text_width"],
        mlp_ratio=4.0,
        depth=config["text_depth"],
        num_heads=config["text_heads"],
        vocab_size=config["num_merges"] + 514,
        context_length=config["context_length"],
        out_dim=config["dim"],
        act_layer="exactgelu",
        key=jr.PRNGKey(0),
    )
    return converter.apply_state(template, state, "text")


def test_causal_text_transformer_matches_the_reference_on_the_tiny_fixture():
    config, fixture = _tiny_reference()
    model = _tiny_reference_text_model(config, fixture)
    padding_mask = jnp.asarray(fixture["outputs.text.0"])
    token_embeddings = fixture["outputs.text.2"]
    embedding = np.asarray(model.token_embedding.weight)
    token_ids = np.argmin(
        np.sum(np.abs(embedding[None, None] - token_embeddings[..., None, :]), axis=-1),
        axis=-1,
    ).transpose(1, 0)

    outputs = jax.vmap(model)(jnp.asarray(token_ids), padding_mask)

    np.testing.assert_allclose(
        np.swapaxes(np.asarray(outputs), 0, 1),
        fixture["outputs.text.1"],
        rtol=1e-5,
        atol=1e-5,
    )
