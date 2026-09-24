"""Tests for equimo.vision.layers.squeeze_excite."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.vision.layers.squeeze_excite import (
    EffectiveSEModule,
    SEModule,
    get_se,
)

KEY = jr.PRNGKey(0)
IN_CHANNELS = 32
H, W = 8, 8
LOW_PRECISION = pytest.mark.parametrize(
    "dtype", (jnp.bfloat16, jnp.float16), ids=("bfloat16", "float16")
)


def _cast_floating(module, dtype):
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
        module,
    )


# ---------------------------------------------------------------------------
# SEModule
# ---------------------------------------------------------------------------


class TestSEModule:
    def test_output_shape(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)

    def test_output_finite(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(se(x)))

    def test_output_dtype_preserved_float32(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.float32)
        assert se(x).dtype == jnp.float32

    @LOW_PRECISION
    def test_low_precision_output_finite(self, dtype):
        se = _cast_floating(SEModule(IN_CHANNELS, key=KEY), dtype)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(dtype)
        out = se(x)
        assert jnp.all(jnp.isfinite(out))

    def test_channel_attention_effect(self):
        """Output must differ from input (attention was actually applied)."""
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert not jnp.allclose(se(x), x)

    def test_custom_rd_ratio(self):
        se = SEModule(IN_CHANNELS, rd_ratio=0.25, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)

    def test_use_norm(self):
        import equinox as eqx

        se = SEModule(IN_CHANNELS, use_norm=True, key=KEY)
        assert not isinstance(se.norm, eqx.nn.Identity)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(se(x)))

    def test_custom_act_layer(self):
        se = SEModule(IN_CHANNELS, act_layer=jax.nn.hard_sigmoid, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)
        assert jnp.all(jnp.isfinite(se(x)))

    def test_kwargs_ignored(self):
        """Extra kwargs must be silently accepted (registry call compatibility)."""
        se = SEModule(IN_CHANNELS, key=KEY, unknown_kwarg=True)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)


# ---------------------------------------------------------------------------
# EffectiveSEModule
# ---------------------------------------------------------------------------


class TestEffectiveSEModule:
    def test_output_shape(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)

    def test_output_finite(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(se(x)))

    @LOW_PRECISION
    def test_low_precision_output_finite(self, dtype):
        se = _cast_floating(EffectiveSEModule(IN_CHANNELS, key=KEY), dtype)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(dtype)
        out = se(x)
        assert jnp.all(jnp.isfinite(out))

    def test_channel_attention_effect(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert not jnp.allclose(se(x), x)

    def test_custom_act_layer(self):
        se = EffectiveSEModule(IN_CHANNELS, act_layer=jax.nn.sigmoid, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)
        assert jnp.all(jnp.isfinite(se(x)))

    def test_kwargs_ignored(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY, unknown_kwarg=42)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)


# ---------------------------------------------------------------------------
# get_se
# ---------------------------------------------------------------------------


class TestGetSe:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("semodule", SEModule),
            ("effectivesemodule", EffectiveSEModule),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_se(name) is expected

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_se("nonexistent_se")
