import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.serialization as serialization
from equimo.core.layers import (
    Attention,
    BlockChunk,
    Mlp,
    SwiGluFused,
    apply_rotary_qk,
)
from equimo.registry import get_model_cls
from equimo.timeseries import layers
from equimo.timeseries.layers import registry as timeseries_registry
from equimo.timeseries.layers.axis_attn import _time_rotary_factors
from equimo.timeseries.models import Forecast, T0, t0, t0_alpha
from equimo.timeseries.models.t0 import _T0_REGISTRY


KEY = jr.PRNGKey(0)


def _tiny(**overrides):
    cfg = dict(
        embed_dim=16,
        num_layers=3,
        num_heads=2,
        mlp_hidden_dim=32,
        patch_size=4,
        group_every_n=3,
        dropout=0.0,
    )
    cfg.update(overrides)
    return T0(**cfg, key=KEY)


def _inputs(time=7):
    values = jr.normal(jr.PRNGKey(1), (2, time))
    mask = jnp.zeros((2, time), dtype=jnp.int8)
    groups = jnp.zeros((2, time), dtype=jnp.int32)
    types = jnp.zeros((2, time), dtype=jnp.int32)
    return values, mask, groups, types


def test_forward_shape_finite_and_monotonic():
    output = _tiny()(*_inputs(), key=KEY, inference=True)
    assert output.shape == (2, 2, 4, 5)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.all(jnp.diff(output, axis=-1) >= 0))


def test_features_flatten_variates_and_patches():
    features = _tiny().features(*_inputs(), key=KEY, inference=True)
    assert features.shape == (4, 16)
    assert bool(jnp.all(jnp.isfinite(features)))


@pytest.mark.parametrize(
    ("context", "expected_shape"),
    [
        (jnp.arange(7, dtype=jnp.float32), (1, 5, 3)),
        (jnp.arange(14, dtype=jnp.float32).reshape(2, 7), (2, 5, 3)),
        (jnp.arange(28, dtype=jnp.float32).reshape(2, 2, 7), (2, 2, 5, 3)),
    ],
)
def test_predict_context_shapes_are_deterministic(context, expected_shape):
    model = _tiny()

    first = model.predict(context, horizon=5)
    second = model.predict(context, horizon=5)

    assert first.quantiles.shape == expected_shape
    assert first.quantiles.dtype == jnp.float32
    assert first.quantile_levels == (0.1, 0.5, 0.9)
    assert first.median.shape == expected_shape[:-1]
    assert bool(jnp.all(jnp.isfinite(first.quantiles)))
    assert jnp.array_equal(first.quantiles, second.quantiles)


def test_predict_supports_missing_values_and_future_covariates():
    model = _tiny()
    context = jnp.arange(14, dtype=jnp.float32).reshape(1, 2, 7)
    context = context.at[0, 0, 2].set(jnp.nan)
    future = jnp.arange(24, dtype=jnp.float32).reshape(1, 2, 12)
    future = future.at[0, 1, 9].set(jnp.nan)

    forecast = model.predict(
        context,
        horizon=5,
        quantiles=(0.25, 0.75),
        future_covariates=future,
    )

    assert forecast.quantiles.shape == (1, 2, 5, 2)
    assert forecast.quantile_levels == (0.25, 0.75)
    assert forecast.median.shape == (1, 2, 5)
    assert bool(jnp.all(jnp.isfinite(forecast.quantiles)))
    assert bool(jnp.all(jnp.diff(forecast.quantiles, axis=-1) >= 0))


def test_forecast_interpolates_median_when_not_requested():
    forecast = Forecast(
        quantiles=jnp.asarray([[[0.0, 2.0]]], dtype=jnp.float32),
        quantile_levels=(0.25, 0.75),
    )

    assert jnp.array_equal(forecast.median, jnp.asarray([[1.0]], dtype=jnp.float32))


def test_predict_validates_public_inputs():
    model = _tiny()
    context = jnp.arange(7, dtype=jnp.float32)

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        model.predict(context, horizon=0)
    with pytest.raises(ValueError, match="quantiles must be non-empty"):
        model.predict(context, horizon=1, quantiles=())
    with pytest.raises(ValueError, match="each quantile must be in"):
        model.predict(context, horizon=1, quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="sorted ascending without duplicates"):
        model.predict(context, horizon=1, quantiles=(0.5, 0.1))
    with pytest.raises(ValueError, match="sorted ascending without duplicates"):
        model.predict(context, horizon=1, quantiles=(0.5, 0.5))
    with pytest.raises(ValueError, match="context must be"):
        model.predict(jnp.zeros((1, 1, 1, 1)), horizon=1)
    with pytest.raises(ValueError, match="future_covariates must be"):
        model.predict(
            context,
            horizon=2,
            future_covariates=jnp.zeros((1, 1, 8)),
        )


def test_predict_rolls_out_beyond_native_horizon():
    model = _tiny(patch_size=512)

    forecast = model.predict(
        jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32),
        horizon=1025,
        quantiles=(0.5,),
    )

    assert forecast.quantiles.shape == (1, 1025, 1)
    assert bool(jnp.all(jnp.isfinite(forecast.quantiles)))


@pytest.mark.parametrize("seq_len", [3, 4])
def test_xpos_matches_upstream_rotation_and_split_scale_layout(seq_len):
    dim = 4
    q = jnp.arange(1, seq_len * dim + 1, dtype=jnp.float32).reshape(1, 1, seq_len, dim)
    k = q + 0.5
    factors = _time_rotary_factors(seq_len, dim)
    rotated_q, rotated_k = apply_rotary_qk(q, k, factors)

    positions = jnp.arange(seq_len, dtype=jnp.float32)
    frequencies = 1.0 / (10_000 ** (jnp.arange(0, dim, 2) / dim))
    angles = jnp.repeat(jnp.outer(positions, frequencies), 2, axis=-1)
    sin, cos = jnp.sin(angles), jnp.cos(angles)
    base = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
    power = (positions - (seq_len - 1) // 2) / 512.0
    half_scale = base[None] ** power[:, None]
    scale = jnp.concatenate((half_scale, half_scale), axis=-1)

    def rotate(x):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        paired = jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)
        return x * cos + paired * sin

    assert jnp.allclose(rotated_q, rotate(q) * scale)
    assert jnp.allclose(rotated_k, rotate(k) / scale)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_xpos_preserves_low_precision_dtype(dtype):
    values = jnp.ones((1, 1, 4, 4), dtype=dtype)

    q, k = apply_rotary_qk(values, values, _time_rotary_factors(4, 4))

    assert q.dtype == dtype
    assert k.dtype == dtype


def test_constructor_rejects_odd_head_dimension():
    with pytest.raises(ValueError, match="embed_dim / num_heads must be even"):
        _tiny(embed_dim=6, num_heads=2)


def test_constructor_rejects_empty_quantile_levels():
    with pytest.raises(ValueError, match="quantile_levels must be a non-empty"):
        _tiny(quantile_levels=())


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_constructor_rejects_invalid_quantile_levels(quantile):
    with pytest.raises(ValueError, match="each quantile must be in"):
        _tiny(quantile_levels=(0.5, quantile))


def test_constructor_sorts_valid_quantile_levels():
    model = _tiny(quantile_levels=(0.9, 0.1, 0.5))
    assert model.quantile_levels == (0.1, 0.5, 0.9)


def test_reuses_equimo_blocks_and_t0_pattern():
    model = _tiny()
    assert isinstance(model.blocks[0], BlockChunk)
    assert isinstance(model.patch_encoder.projection.mlp, Mlp)
    assert isinstance(model.blocks[0].blocks[0].attn.attention, Attention)
    assert isinstance(model.blocks[0].blocks[0].ffn, SwiGluFused)
    assert [block.attention_type for block in model.blocks[0].blocks] == [
        "time",
        "time",
        "group",
    ]


def test_factory_and_registry():
    kwargs = dict(
        embed_dim=16,
        num_layers=3,
        num_heads=2,
        mlp_hidden_dim=32,
        patch_size=4,
        key=KEY,
    )
    assert isinstance(t0(**kwargs), T0)
    assert isinstance(t0_alpha(**kwargs), T0)
    assert get_model_cls("t0", modality="timeseries") is T0
    base_cfg, variant_cfg = _T0_REGISTRY["t0_alpha"]
    assert base_cfg | variant_cfg == {
        "embed_dim": 512,
        "num_layers": 24,
        "num_heads": 8,
        "mlp_hidden_dim": 2048,
        "patch_size": 32,
        "group_every_n": 3,
        "dropout": 0.1,
        "quantile_levels": (0.1, 0.25, 0.5, 0.75, 0.9),
    }


def test_t0_pretrained_alias_loads_t0_alpha(monkeypatch):
    cfg = {
        "embed_dim": 16,
        "num_layers": 3,
        "num_heads": 2,
        "mlp_hidden_dim": 32,
        "patch_size": 4,
        "group_every_n": 3,
        "dropout": 0.0,
    }
    loaded = {}

    def fake_load_weights(model, *, identifier, inference_mode):
        loaded.update(identifier=identifier, inference_mode=inference_mode)
        return model

    monkeypatch.setitem(_T0_REGISTRY, "t0", (cfg, {}))
    monkeypatch.setattr(serialization, "load_weights", fake_load_weights)

    model = t0(pretrained=True, key=KEY)

    assert isinstance(model, T0)
    assert loaded == {"identifier": "t0_alpha", "inference_mode": True}


@pytest.mark.parametrize("factory", [t0, t0_alpha])
def test_pretrained_variants_reject_overridden_configs(factory):
    with pytest.raises(ValueError, match="do not accept configuration overrides"):
        factory(pretrained=True, embed_dim=16)


def test_timeseries_layer_registry(monkeypatch):
    assert layers.get_layer("axisattention") is layers.AxisAttention
    assert layers.get_layer("AXISATTENTION") is layers.AxisAttention
    assert layers.get_layer("patchencoder") is layers.PatchEncoder
    assert layers.get_layer("residualmlp") is layers.ResidualMlp
    assert layers.get_layer("t0block") is layers.T0Block
    assert layers.get_layer("mlp") is Mlp
    assert layers.get_layer(layers.AxisAttention) is layers.AxisAttention

    class CustomLayer(layers.ResidualMlp):
        pass

    layers.register_layer("custom_t0_layer")(CustomLayer)
    assert layers.get_layer("custom_t0_layer") is CustomLayer
    with pytest.raises(ValueError):
        layers.register_layer("custom_t0_layer")(CustomLayer)

    class ReplacementLayer(layers.ResidualMlp):
        pass

    layers.register_layer("custom_t0_layer", force=True)(ReplacementLayer)
    assert layers.get_layer("custom_t0_layer") is ReplacementLayer

    with pytest.raises(
        ValueError,
        match="not found in the time-series layer scope",
    ) as error:
        layers.get_layer("__missing_timeseries_layer__")
    assert "axisattention" in str(error.value)
    assert "mlp" in str(error.value)

    class LocalMlp(layers.ResidualMlp):
        pass

    monkeypatch.setitem(timeseries_registry._LAYER_REGISTRY, "mlp", LocalMlp)
    assert layers.get_layer("mlp") is LocalMlp
