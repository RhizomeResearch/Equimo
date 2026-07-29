import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.core.layers import Attention, BlockChunk, Mlp, SwiGluFused
from equimo.registry import get_model_cls
from equimo.time_series import layers
from equimo.time_series.models import T0, t0, t0_alpha
from equimo.time_series.layers.axis_attn import _xpos
from equimo.time_series.models.t0 import _T0_REGISTRY


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


def test_xpos_matches_upstream_split_scale_layout():
    q = jnp.ones((1, 1, 3, 4), dtype=jnp.float32)
    rotated_q, _ = _xpos(q, q)
    base = (jnp.arange(0, 4, 2) + 0.4 * 4) / (1.4 * 4)
    half_scale = base ** (-1 / 512)
    expected = jnp.concatenate((half_scale, half_scale))
    assert jnp.allclose(rotated_q[0, 0, 0], expected)


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
    assert get_model_cls("t0", modality="time_series") is T0
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


def test_pretrained_variants_reject_unsupported_or_overridden_configs():
    with pytest.raises(ValueError, match="Supported T0 pretrained variants: t0_alpha"):
        t0(pretrained=True)
    with pytest.raises(ValueError, match="do not accept configuration overrides"):
        t0_alpha(pretrained=True, embed_dim=16)


def test_time_series_layer_registry():
    assert layers.get_layer("axisattention") is layers.AxisAttention
    assert layers.get_layer("patchencoder") is layers.PatchEncoder
    assert layers.get_layer("residualmlp") is layers.ResidualMlp
    assert layers.get_layer("t0block") is layers.T0Block
    assert layers.get_layer("mlp") is Mlp

    class CustomLayer(layers.ResidualMlp):
        pass

    layers.register_layer("custom_t0_layer")(CustomLayer)
    assert layers.get_layer("custom_t0_layer") is CustomLayer
    with pytest.raises(ValueError):
        layers.register_layer("custom_t0_layer")(CustomLayer)
