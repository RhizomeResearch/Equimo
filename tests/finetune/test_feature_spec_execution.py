"""Executable FeatureSpec contract and cross-modality parity tests."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.audio.models import AudioSpectrogramTransformer
from equimo.language.models import TextTransformerEncoder
from equimo.tabular.models import TabPFN
from equimo.vision.models import VisionTransformer


class EchoFeatures(eqx.Module):
    def features(self, x, padding_mask=None):
        del padding_mask
        return x


class LayerFeatures(eqx.Module):
    def layers(self, x):
        return x, x + 2


@dataclass(frozen=True)
class _PromptConfig:
    prepend_to: str = "after_cls"


class PromptTokenFeatures(eqx.Module):
    num_base_prefix_tokens: int = eqx.field(static=True, default=1)
    num_prompt_tokens: int = eqx.field(static=True, default=2)
    config: _PromptConfig = eqx.field(static=True, default=_PromptConfig())

    def features(self, x):
        return x


@pytest.mark.parametrize(
    ("layout", "shape", "pooling", "expected_axes"),
    (
        ("BNC", (2, 3, 4), "mean_token", (1,)),
        ("BTC", (2, 3, 4), "mean_token", (1,)),
        ("BCT", (2, 4, 3), "mean_frame", (2,)),
        ("BCHW", (2, 3, 2, 2), "global_avg", (2, 3)),
        ("BC", (2, 4), None, ()),
    ),
)
def test_layouts_resolve_feature_and_sequence_axes(
    layout, shape, pooling, expected_axes
):
    x = jnp.arange(float(jnp.prod(jnp.asarray(shape)))).reshape(shape)
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout=layout,
        token_selection="all",
        pooling=pooling,
    )

    result = eqft.extract_features(EchoFeatures(), x, feature_spec=spec)
    expected = x if not expected_axes else jnp.mean(x, axis=expected_axes)

    assert jnp.allclose(result, expected)


def test_unbatched_layout_and_explicit_endpoint_path_are_supported():
    x = jnp.arange(12.0).reshape(3, 4)
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BTC",
        token_selection="all",
        pooling="mean_token",
    )

    result = eqft.extract_features(EchoFeatures(), x, feature_spec=spec)

    assert jnp.array_equal(result, jnp.mean(x, axis=0))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"output_layout": "BC", "token_selection": "cls", "pooling": None},
        {"output_layout": "BCHW", "token_selection": "all", "pooling": "cls"},
        {"output_layout": "BTC", "token_selection": "custom", "pooling": None},
        {
            "output_layout": "BTC",
            "token_selection": "last_valid",
            "pooling": "mean_token",
        },
        {
            "output_layout": "BTC",
            "token_selection": "all",
            "pooling": None,
            "mask_field": "padding_mask",
        },
    ),
)
def test_contradictory_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        eqft.FeatureSpec(endpoint="features", **kwargs)


def test_mask_is_padding_polarity_and_all_padding_returns_zero():
    x = jnp.asarray([[1.0, 2.0], [3.0, 6.0], [50.0, 100.0]])
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BTC",
        token_selection="all",
        pooling="mean_token",
        mask_field="padding_mask",
    )

    pooled = eqft.extract_features(
        EchoFeatures(), x, jnp.asarray([0, 0, 1]), feature_spec=spec
    )
    all_padding = eqft.extract_features(
        EchoFeatures(), x, jnp.ones((3,), dtype=jnp.int32), feature_spec=spec
    )

    assert jnp.array_equal(pooled, jnp.asarray([2.0, 4.0]))
    assert jnp.array_equal(all_padding, jnp.zeros((2,)))


def test_last_valid_and_mask_shape_validation():
    x = jnp.arange(12.0).reshape(3, 4)
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BTC",
        token_selection="last_valid",
        pooling=None,
        mask_field="padding_mask",
    )

    result = eqft.extract_features(
        EchoFeatures(), x, jnp.asarray([0, 0, 1]), feature_spec=spec
    )

    assert jnp.array_equal(result, x[1])
    with pytest.raises(ValueError, match="mask shape"):
        eqft.extract_features(EchoFeatures(), x, jnp.asarray([0, 1]), feature_spec=spec)


@pytest.mark.parametrize("exclude", (True, False))
def test_prompt_tokens_are_conditionally_excluded(exclude):
    x = jnp.arange(10.0).reshape(5, 2)
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BNC",
        token_selection="patches",
        pooling="mean_patch",
        exclude_prompt_tokens=exclude,
    )

    result = eqft.extract_features(PromptTokenFeatures(), x, feature_spec=spec)
    expected_tokens = x[3:] if exclude else jnp.concatenate([x[1:3], x[3:]])

    assert jnp.array_equal(result, jnp.mean(expected_tokens, axis=0))


@pytest.mark.parametrize("normalization", ("l2", "standardize"))
def test_normalization_applies_only_to_feature_axis(normalization):
    x = jnp.asarray([[3.0, 4.0], [1.0, 3.0]])
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BC",
        token_selection="all",
        pooling=None,
        normalize=normalization,
    )

    result = eqft.extract_features(EchoFeatures(), x, feature_spec=spec)

    if normalization == "l2":
        expected = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    else:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        expected = (x - mean) / jnp.sqrt(variance + 1e-6)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("method", ("last", "mean", "concat"))
def test_layer_aggregation_is_executable(method):
    x = jnp.arange(12.0).reshape(3, 4)
    spec = eqft.FeatureSpec(
        endpoint="layers",
        output_layout="BTC",
        token_selection="all",
        pooling="mean_token",
        layer_aggregation={"method": method},
    )

    result = eqft.extract_features(LayerFeatures(), x, feature_spec=spec)

    if method == "last":
        expected = jnp.mean(x + 2, axis=0)
    elif method == "mean":
        expected = jnp.mean(x + 1, axis=0)
    else:
        expected = jnp.concatenate([jnp.mean(x, axis=0), jnp.mean(x + 2, axis=0)])
    assert jnp.array_equal(result, expected)


def test_preprocessing_fingerprint_is_required_and_checked():
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BC",
        token_selection="all",
        pooling=None,
        preprocessing_fingerprint="sha256:expected",
    )
    x = jnp.ones((4,))

    with pytest.raises(ValueError, match="no observed fingerprint"):
        eqft.extract_features(EchoFeatures(), x, feature_spec=spec)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        eqft.extract_features(
            EchoFeatures(),
            x,
            feature_spec=spec,
            observed_preprocessing_fingerprint="sha256:other",
        )
    result = eqft.extract_features(
        EchoFeatures(),
        x,
        feature_spec=spec,
        observed_preprocessing_fingerprint="sha256:expected",
    )
    assert jnp.array_equal(result, x)


def test_feature_extractor_and_linear_probe_forward_the_spec():
    spec = eqft.FeatureSpec(
        endpoint="features",
        output_layout="BTC",
        token_selection="all",
        pooling="mean_token",
    )
    x = jnp.arange(12.0).reshape(3, 4)
    extractor = eqft.FeatureExtractor(EchoFeatures(), feature_spec=spec)
    probe = eqft.LinearProbe(EchoFeatures(), eqft.IdentityHead(), feature_spec=spec)

    expected = jnp.mean(x, axis=0)
    assert jnp.array_equal(eqx.filter_jit(extractor)(x), expected)
    assert jnp.array_equal(probe(x), expected)


def test_explicit_vision_spec_matches_native_readout():
    key = jr.PRNGKey(10)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[1],
        global_pool="avg",
        num_classes=0,
        key=key,
    )
    x = jr.normal(jr.PRNGKey(11), (3, 16, 16))
    spec = eqft.FeatureSpec("forward_features", "BNC", "all", "native")

    explicit = eqft.extract_features(model, x, feature_spec=spec, key=key)
    fallback = eqft.extract_features(model, x, pool="auto", key=key)

    assert jnp.allclose(explicit, fallback)


def test_explicit_audio_spec_matches_native_readout():
    key = jr.PRNGKey(12)
    model = AudioSpectrogramTransformer(
        input_fdim=16,
        input_tdim=16,
        dim=8,
        patch_size=8,
        fstride=8,
        tstride=8,
        num_heads=2,
        depths=[1],
        global_pool="token",
        num_classes=0,
        key=key,
    )
    x = jr.normal(jr.PRNGKey(13), (16, 16))
    spec = eqft.FeatureSpec("forward_features", "BNC", "all", "native")

    explicit = eqft.extract_features(model, x, feature_spec=spec, key=key)
    fallback = eqft.extract_features(model, x, pool="auto", key=key)

    assert jnp.allclose(explicit, fallback)


def test_explicit_language_spec_matches_native_readout():
    key = jr.PRNGKey(14)
    model = TextTransformerEncoder(
        dim=8,
        mlp_ratio=2.0,
        depth=1,
        num_heads=2,
        vocab_size=32,
        key=key,
    )
    ids = jnp.asarray([1, 2, 3, 4])
    padding_mask = jnp.asarray([0, 0, 0, 1])
    spec = eqft.FeatureSpec(
        "features",
        "BTC",
        "all",
        "mean_token",
        mask_field="padding_mask",
    )

    explicit = eqft.extract_features(
        model, ids, padding_mask, feature_spec=spec, key=key
    )
    native = model(ids, padding_mask, key=key, inference=True)

    assert jnp.allclose(explicit, native, rtol=1e-6, atol=1e-6)


def test_explicit_tabular_spec_preserves_native_prediction_rows():
    key = jr.PRNGKey(15)
    model = TabPFN(
        num_classes=4,
        dim=16,
        depths=(1, 1, 1),
        num_heads=(2, 2, 2),
        num_inducing_points=4,
        feature_group_size=3,
        num_cls_tokens=2,
        num_kv_heads_test=1,
        decoder_head_dim=8,
        decoder_num_heads=2,
        mlp_ratio=2.0,
        key=key,
    )
    x = jr.normal(jr.PRNGKey(16), (8, 4))
    y = jr.randint(jr.PRNGKey(17), (8,), 0, 4)
    n_train = 5
    spec = eqft.FeatureSpec("__call__", "BC", "all", None)

    explicit = eqft.extract_features(model, x, y, n_train, feature_spec=spec, key=key)
    native = model(x, y, n_train, key=key, inference=True)

    assert explicit.shape == (3, 4)
    assert jnp.allclose(explicit, native)
