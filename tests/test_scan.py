"""Tests for state-space scan operations."""

import jax.numpy as jnp
import jax.random as jr

from equimo.core.ops import ssd


def _ssd_reference(x, a, b, c, initial_state):
    state = initial_state
    outputs = []
    for index in range(x.shape[0]):
        state = jnp.exp(a[index])[:, None, None] * state + jnp.einsum(
            "hp,hn->hpn", x[index], b[index]
        )
        outputs.append(jnp.einsum("hn,hpn->hp", c[index], state))
    return jnp.stack(outputs), state


def test_ssd_matches_recurrent_reference_with_initial_state():
    keys = jr.split(jr.PRNGKey(0), 5)
    x = jr.normal(keys[0], (6, 2, 3))
    a = -jr.uniform(keys[1], (6, 2), minval=0.01, maxval=0.2)
    b = jr.normal(keys[2], (6, 2, 4))
    c = jr.normal(keys[3], (6, 2, 4))
    initial_state = jr.normal(keys[4], (2, 3, 4))

    actual, final_state = ssd(
        x,
        a,
        b,
        c,
        chunk_size=2,
        initial_states=initial_state,
    )
    expected, expected_final_state = _ssd_reference(x, a, b, c, initial_state)

    assert actual.shape == x.shape
    assert final_state.shape == initial_state.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(final_state, expected_final_state, rtol=1e-5, atol=1e-5)


def test_ssd_preserves_low_precision_dtype():
    x = jnp.ones((4, 1, 2), dtype=jnp.bfloat16)
    a = jnp.full((4, 1), -0.1, dtype=jnp.bfloat16)
    b = jnp.ones((4, 1, 3), dtype=jnp.bfloat16)
    c = jnp.ones((4, 1, 3), dtype=jnp.bfloat16)

    output, final_state = ssd(x, a, b, c, chunk_size=2)

    assert output.dtype == jnp.bfloat16
    assert final_state.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(output))
