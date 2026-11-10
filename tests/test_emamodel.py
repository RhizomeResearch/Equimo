from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from equimo.core import update_ema_model
from equimo.core.layers.dropout import DropPathAdd
from equimo.core.parameters import buffer


class _EmaState(eqx.Module):
    scalar: jax.Array
    vector: jax.Array
    metadata: str = eqx.field(static=True)


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


class _EmaStateWithConstants(eqx.Module):
    weight: jax.Array
    index: jax.Array
    basis: jax.Array = buffer()


def test_update_ema_model_copies_integer_and_buffer_leaves():
    ema_model = _EmaStateWithConstants(
        jnp.array([0.0]), jnp.array([1, 2]), jnp.array([1.0, 1.0])
    )
    current_model = _EmaStateWithConstants(
        jnp.array([4.0]), jnp.array([3, 4]), jnp.array([2.0, 2.0])
    )

    result = update_ema_model(
        ema_model, eqx.filter(current_model, eqx.is_array), decay=0.5
    )

    assert jnp.allclose(result.weight, 2.0)
    assert result.index.dtype == jnp.int32
    assert jnp.array_equal(result.index, current_model.index)
    assert jnp.array_equal(result.basis, current_model.basis)


class _EmaStateWithNonArrayLeaves(eqx.Module):
    weight: jax.Array
    drop_path: DropPathAdd
    activation: Callable


def test_update_ema_model_keeps_non_array_leaves():
    ema_model = _EmaStateWithNonArrayLeaves(
        jnp.array([0.0]), DropPathAdd(0.1), jax.nn.relu
    )
    current_model = _EmaStateWithNonArrayLeaves(
        jnp.array([4.0]), DropPathAdd(0.2), jax.nn.gelu
    )

    result = update_ema_model(
        ema_model, eqx.filter(current_model, eqx.is_array), decay=0.5
    )

    assert jnp.allclose(result.weight, jnp.array([2.0]))
    assert result.drop_path.p == 0.1
    assert result.activation is jax.nn.relu
