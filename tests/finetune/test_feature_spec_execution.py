"""Executable FeatureSpec contract and cross-modality parity tests."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.audio.models import AudioSpectrogramTransformer
from equimo.language.models import TextTransformerEncoder
from equimo.tabular.models import TabPFN
from equimo.vision.models import VisionTransformer, dinov2_vits14_reg


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


class SpatialTokenFeatures(eqx.Module):
    patch_size: tuple[int, int] = eqx.field(static=True)
    prefix_tokens: tuple[str, ...] = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    pad: bool = eqx.field(static=True)

    def __init__(
        self,
        patch_size=(2, 2),
        prefix_tokens=("cls", "register_0"),
        *,
        pad=True,
    ):
        self.patch_size = patch_size
        self.prefix_tokens = prefix_tokens
        self.num_prefix_tokens = len(prefix_tokens)
        self.pad = pad

    def features(self, x):
        height, width = x.shape[-2:]
        patch_h, patch_w = self.patch_size
        pad_h = (-height) % patch_h
        pad_w = (-width) % patch_w
        if not self.pad and (pad_h or pad_w):
            raise ValueError("input must be divisible by the patch size")
        x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))
        patches = x[0, ::patch_h, ::patch_w].reshape(-1, 1)
        prefix = -jnp.arange(1, len(self.prefix_tokens) + 1, dtype=x.dtype)[:, None]
        return jnp.concatenate((prefix, patches), axis=0)

    def feature_metadata(self, x, *, endpoint, endpoint_options):
        del endpoint, endpoint_options
        height, width = x.shape[-2:]
        patch_h, patch_w = self.patch_size
        grid_h = (height + patch_h - 1) // patch_h if self.pad else height // patch_h
        grid_w = (width + patch_w - 1) // patch_w if self.pad else width // patch_w
        return {
            "input_size": (height, width),
            "patch_size": self.patch_size,
            "patch_padding": (
                grid_h * patch_h - height,
                grid_w * patch_w - width,
            ),
            "grid_size": (grid_h, grid_w),
            "prefix_tokens": self.prefix_tokens,
            "tokens_include_prefix": True,
            "endpoint_normalization": "none",
            "positional_configuration": (("kind", "coordinate-test"),),
        }


class IncorrectSpatialMetadata(SpatialTokenFeatures):
    def feature_metadata(self, x, *, endpoint, endpoint_options):
        metadata = super().feature_metadata(
            x,
            endpoint=endpoint,
            endpoint_options=endpoint_options,
        )
        return {**metadata, "grid_size": (1, 1)}


@pytest.fixture(scope="module")
def dinov2_vits14_reg_small():
    key = jr.PRNGKey(100)
    model = dinov2_vits14_reg(
        pretrained=False,
        img_size=28,
        depths=[1],
        key=key,
    )
    image = jr.normal(jr.PRNGKey(101), (3, 28, 28))
    return model, image, key


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


def test_dense_spatial_layout_derives_grid_metadata_from_declared_axes():
    x = jnp.arange(24.0).reshape(3, 2, 4)
    spec = eqft.FeatureSpec("features", "BCHW", "all", None, return_metadata=True)

    result = eqft.extract_features(EchoFeatures(), x, feature_spec=spec)

    assert jnp.array_equal(result.features, x)
    assert result.levels[0].input_size == (2, 4)
    assert result.levels[0].grid_size == (2, 4)
    assert result.levels[0].feature_width == 3


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


@pytest.mark.parametrize(
    "endpoint_options",
    (
        {"unknown": True},
        {"indices": ()},
        {"indices": (0,), "n_last_blocks": 1},
        {"n_last_blocks": 0},
        {"apply_norm": "yes"},
    ),
)
def test_invalid_endpoint_options_are_rejected(endpoint_options):
    with pytest.raises(ValueError):
        eqft.FeatureSpec(
            "intermediate_features",
            "BNC",
            "all",
            None,
            endpoint_options=endpoint_options,
        )


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


@pytest.mark.parametrize(
    "prefix_tokens",
    ((), ("cls",), ("cls", "register_0", "register_1", "register_2")),
)
def test_spatial_metadata_preserves_row_major_patch_order_and_prefixes(
    prefix_tokens,
):
    model = SpatialTokenFeatures(prefix_tokens=prefix_tokens)
    image = jnp.arange(24.0).reshape(1, 4, 6)
    spec = eqft.FeatureSpec(
        "features",
        "BNC",
        "patches",
        None,
        endpoint_options={},
        return_metadata=True,
    )

    result = eqft.extract_features(model, image, feature_spec=spec)

    assert isinstance(result, eqft.FeatureResult)
    assert jnp.array_equal(
        result.features[:, 0],
        jnp.asarray([0.0, 2.0, 4.0, 12.0, 14.0, 16.0]),
    )
    assert result.levels == (
        eqft.FeatureLevelMetadata(
            layer_index=None,
            feature_width=1,
            source_layout="BNC",
            endpoint_normalization="none",
            post_normalization="none",
            prefix_tokens=prefix_tokens,
            positional_configuration=(("kind", "coordinate-test"),),
            input_size=(4, 6),
            patch_size=(2, 2),
            patch_padding=(0, 0),
            grid_size=(2, 3),
            flatten_order="row-major",
        ),
    )


@pytest.mark.parametrize(
    ("shape", "patch_size", "expected_grid"),
    (
        ((256, 256), (16, 16), (16, 16)),
        ((512, 512), (16, 16), (32, 32)),
        ((512, 1024), (16, 16), (32, 64)),
        ((1024, 512), (16, 16), (64, 32)),
        ((518, 1022), (14, 14), (37, 73)),
    ),
)
def test_spatial_metadata_covers_square_rectangular_and_patch14_profiles(
    shape, patch_size, expected_grid
):
    model = SpatialTokenFeatures(patch_size=patch_size)
    image = jnp.zeros((3, *shape), dtype=jnp.bfloat16)
    spec = eqft.FeatureSpec("features", "BNC", "patches", None, return_metadata=True)

    result = eqft.extract_features(model, image, feature_spec=spec)

    assert result.features.dtype == jnp.bfloat16
    assert result.levels[0].grid_size == expected_grid
    assert result.features.shape == (expected_grid[0] * expected_grid[1], 1)


def test_spatial_metadata_reports_padding_and_rejects_unsupported_geometry():
    image = jnp.zeros((1, 5, 7))
    spec = eqft.FeatureSpec("features", "BNC", "patches", None, return_metadata=True)

    padded = eqft.extract_features(
        SpatialTokenFeatures(pad=True), image, feature_spec=spec
    )
    assert padded.levels[0].grid_size == (3, 4)
    assert padded.levels[0].patch_padding == (1, 1)

    with pytest.raises(ValueError, match="divisible"):
        eqft.extract_features(SpatialTokenFeatures(pad=False), image, feature_spec=spec)


def test_spatial_metadata_requires_a_model_contract_for_token_grids():
    spec = eqft.FeatureSpec("features", "BNC", "patches", None, return_metadata=True)
    with pytest.raises(ValueError, match="does not publish feature_metadata"):
        eqft.extract_features(EchoFeatures(), jnp.ones((4, 3)), feature_spec=spec)


def test_spatial_metadata_rejects_a_grid_that_disagrees_with_tokens():
    spec = eqft.FeatureSpec("features", "BNC", "patches", None, return_metadata=True)
    with pytest.raises(ValueError, match="token count"):
        eqft.extract_features(
            IncorrectSpatialMetadata(),
            jnp.ones((1, 4, 6)),
            feature_spec=spec,
        )


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


def test_dinov2_explicit_specs_match_normalized_native_outputs(
    dinov2_vits14_reg_small,
):
    model, image, key = dinov2_vits14_reg_small
    preprocessing_id = "preprocessing:sha256:dinov2-test"
    final_cls = eqft.FeatureSpec(
        endpoint="forward_features",
        output_layout="BNC",
        token_selection="cls",
        pooling=None,
        normalize="none",
        preprocessing_fingerprint=preprocessing_id,
    )
    final_patch_mean = eqft.FeatureSpec(
        endpoint="forward_features",
        output_layout="BNC",
        token_selection="patches",
        pooling="mean_patch",
        normalize="l2",
        exclude_prompt_tokens=True,
        preprocessing_fingerprint=preprocessing_id,
    )

    native = model.forward_features(image, key=key, inference=True)
    cls = eqft.extract_features(
        model,
        image,
        feature_spec=final_cls,
        observed_preprocessing_fingerprint=preprocessing_id,
        key=key,
        inference=True,
    )
    patch_mean = eqft.extract_features(
        model,
        image,
        feature_spec=final_patch_mean,
        observed_preprocessing_fingerprint=preprocessing_id,
        key=key,
        inference=True,
    )
    expected_patch_mean = jnp.mean(native["x_norm_patchtokens"], axis=0)
    expected_patch_mean /= jnp.maximum(
        jnp.linalg.norm(expected_patch_mean),
        1e-12,
    )

    assert cls.shape == (384,)
    assert patch_mean.shape == (384,)
    assert cls.dtype == native["x_norm_cls_token"].dtype
    assert patch_mean.dtype == native["x_norm_patchtokens"].dtype
    assert jnp.all(jnp.isfinite(cls))
    assert jnp.all(jnp.isfinite(patch_mean))
    assert jnp.allclose(cls, native["x_norm_cls_token"], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(patch_mean, expected_patch_mean, rtol=1e-6, atol=1e-6)


def test_dinov2_explicit_specs_support_jit_and_vmap(dinov2_vits14_reg_small):
    model, image, key = dinov2_vits14_reg_small
    preprocessing_id = "preprocessing:sha256:dinov2-test"
    spec = eqft.FeatureSpec(
        endpoint="forward_features",
        output_layout="BNC",
        token_selection="patches",
        pooling="mean_patch",
        normalize="l2",
        exclude_prompt_tokens=True,
        preprocessing_fingerprint=preprocessing_id,
    )

    def extract_one(sample, sample_key):
        return eqft.extract_features(
            model,
            sample,
            feature_spec=spec,
            observed_preprocessing_fingerprint=preprocessing_id,
            key=sample_key,
            inference=True,
        )

    eager = extract_one(image, key)
    compiled = jax.jit(extract_one)(image, key)
    batch = jnp.stack((image, image))
    keys = jr.split(key, 2)
    batched = jax.jit(jax.vmap(extract_one))(batch, keys)

    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)
    assert batched.shape == (2, 384)
    assert jnp.all(jnp.isfinite(batched))


def test_feature_result_supports_jit_vmap_and_gradients():
    model = SpatialTokenFeatures(prefix_tokens=("cls",))
    spec = eqft.FeatureSpec("features", "BNC", "patches", None, return_metadata=True)

    def extract_one(image):
        return eqft.extract_features(model, image, feature_spec=spec)

    image = jnp.arange(24.0).reshape(1, 4, 6)
    eager = extract_one(image)
    compiled = jax.jit(extract_one)(image)
    batched = jax.jit(jax.vmap(extract_one))(jnp.stack((image, image + 1)))
    gradient = jax.grad(lambda value: jnp.sum(extract_one(value).features))(image)

    assert jnp.array_equal(compiled.features, eager.features)
    assert compiled.levels == eager.levels
    assert batched.features.shape == (2, 6, 1)
    assert batched.levels == eager.levels
    assert gradient.shape == image.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_vit_separate_intermediates_preserve_indices_prefixes_and_norm():
    key = jr.PRNGKey(102)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=[2, 2],
        depths=[2, 2],
        reg_tokens=2,
        num_classes=0,
        key=key,
    )
    image = jr.normal(jr.PRNGKey(103), (3, 16, 16))
    spec = eqft.FeatureSpec(
        "intermediate_features",
        "BNC",
        "all",
        None,
        layer_aggregation={"method": "separate"},
        endpoint_options={"indices": (1, 2), "apply_norm": True},
        return_metadata=True,
    )

    result = eqft.extract_features(model, image, feature_spec=spec, key=key)
    expected = model.intermediate_features(
        image,
        key=key,
        inference=True,
        indices=(1, 2),
        apply_norm=True,
    )

    assert isinstance(result, eqft.FeatureResult)
    assert len(result.features) == 2
    assert all(
        jnp.allclose(actual, reference)
        for actual, reference in zip(result.features, expected, strict=True)
    )
    assert tuple(level.layer_index for level in result.levels) == (1, 2)
    assert all(level.feature_width == 8 for level in result.levels)
    assert all(
        level.endpoint_normalization == "encoder_final_norm" for level in result.levels
    )
    assert all(
        level.prefix_tokens == ("cls", "register_0", "register_1")
        for level in result.levels
    )
    assert all(level.grid_size == (2, 2) for level in result.levels)
    assert all(features.shape == (7, 8) for features in result.features)

    low_precision_model = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        model,
    )
    low_precision = eqx.filter_jit(
        lambda current_model, sample: eqft.extract_features(
            current_model,
            sample,
            feature_spec=spec,
            key=key,
        )
    )(low_precision_model, image.astype(jnp.bfloat16))
    assert all(level.dtype == jnp.bfloat16 for level in low_precision.features)


def test_feature_spec_endpoint_options_conflict_with_runtime_arguments():
    key = jr.PRNGKey(104)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[2],
        num_classes=0,
        key=key,
    )
    spec = eqft.FeatureSpec(
        "intermediate_features",
        "BNC",
        "all",
        None,
        layer_aggregation={"method": "last"},
        endpoint_options={"indices": (0,)},
    )

    with pytest.raises(ValueError, match="conflict with runtime arguments"):
        eqft.extract_features(
            model,
            jnp.ones((3, 16, 16)),
            feature_spec=spec,
            key=key,
            indices=(1,),
        )


def test_dinov2_explicit_spec_does_not_fall_back_from_invalid_endpoint(
    dinov2_vits14_reg_small,
):
    model, image, key = dinov2_vits14_reg_small
    spec = eqft.FeatureSpec(
        endpoint="missing_features",
        output_layout="BNC",
        token_selection="cls",
        pooling=None,
    )

    with pytest.raises(ValueError, match="missing component"):
        eqft.extract_features(model, image, feature_spec=spec, key=key)


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
