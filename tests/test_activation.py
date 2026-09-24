"""Tests for equimo.core.layers.activation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.core.layers.activation import _ACT_REGISTRY, get_act, register_act

KEY = jr.PRNGKey(0)
X = jnp.ones((4, 8), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# get_act
# ---------------------------------------------------------------------------


class TestGetAct:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("relu", jax.nn.relu),
            ("gelu", jax.nn.gelu),
            ("silu", jax.nn.silu),
            ("elu", jax.nn.elu),
            ("sigmoid", jax.nn.sigmoid),
            ("hard_sigmoid", jax.nn.hard_sigmoid),
            ("hard_swish", jax.nn.hard_swish),
            ("softmax", jax.nn.softmax),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_act(name) is expected

    def test_callable_passthrough(self):
        fn = lambda x: x
        assert get_act(fn) is fn

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown activation string"):
            get_act("nonexistent_act")

    def test_all_builtins_callable(self):
        builtins = [
            "relu",
            "gelu",
            "exactgelu",
            "silu",
            "elu",
            "sigmoid",
            "hard_sigmoid",
            "hard_swish",
            "softmax",
        ]
        for name in builtins:
            fn = get_act(name)
            out = fn(X)
            assert out.shape == X.shape, f"{name} changed shape"
            assert jnp.all(jnp.isfinite(out)), f"{name} produced non-finite output"


# ---------------------------------------------------------------------------
# register_act
# ---------------------------------------------------------------------------


class TestRegisterAct:
    def test_register_default_name(self):
        @register_act()
        def my_custom_act(x):
            return x * 2

        assert "my_custom_act" in _ACT_REGISTRY
        assert get_act("my_custom_act") is my_custom_act

    def test_register_custom_name(self):
        @register_act(name="MySwish")
        def another_act(x):
            return x * jax.nn.sigmoid(x)

        assert "myswish" in _ACT_REGISTRY
        assert get_act("myswish") is another_act

    def test_register_duplicate_raises(self):
        @register_act(name="dup_act_1")
        def act_a(x):
            return x

        with pytest.raises(ValueError, match="already registered"):

            @register_act(name="dup_act_1")
            def act_b(x):
                return x

    def test_register_returns_original_function(self):
        def raw_fn(x):
            return x + 1

        decorated = register_act(name="reg_returns_test")(raw_fn)
        assert decorated is raw_fn
