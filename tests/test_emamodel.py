import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from equimo.core import update_ema_model


class _EmaState(eqx.Module):
    scalar: jax.Array
    vector: jax.Array
    metadata: str = eqx.field(static=True)


def test_update_ema_model_returns_computed_midpoint():
    ema_model = _EmaState(jnp.array(2.0), jnp.array([2.0, 6.0]), "ema")
    current_model = _EmaState(jnp.array(4.0), jnp.array([4.0, 2.0]), "ema")

    result = update_ema_model(
        ema_model, eqx.filter(current_model, eqx.is_array), decay=0.5
    )

    assert jnp.allclose(result.scalar, 3.0)
    assert jnp.allclose(result.vector, jnp.array([3.0, 4.0]))
    assert result.metadata == "ema"


@pytest.mark.parametrize(("decay", "use_current"), [(0.0, True), (1.0, False)])
def test_update_ema_model_decay_endpoints(decay, use_current):
    ema_model = _EmaState(jnp.array(2.0), jnp.array([1.0, 3.0]), "ema")
    current_model = _EmaState(jnp.array(4.0), jnp.array([5.0, 7.0]), "ema")
    expected = current_model if use_current else ema_model

    result = update_ema_model(
        ema_model, eqx.filter(current_model, eqx.is_array), decay=decay
    )

    assert jnp.allclose(result.scalar, expected.scalar)
    assert jnp.allclose(result.vector, expected.vector)
    assert result.metadata == "ema"


def test_update_ema_model_updates_multiple_leaves_through_jit():
    ema_model = _EmaState(jnp.array(-2.0), jnp.array([0.0, 4.0, 8.0]), "ema")
    current_model = _EmaState(jnp.array(2.0), jnp.array([4.0, 8.0, 12.0]), "ema")

    result = update_ema_model(
        ema_model, eqx.filter(current_model, eqx.is_array), decay=0.25
    )

    assert jnp.allclose(result.scalar, 1.0)
    assert jnp.allclose(result.vector, jnp.array([3.0, 7.0, 11.0]))
    assert result.metadata == "ema"
