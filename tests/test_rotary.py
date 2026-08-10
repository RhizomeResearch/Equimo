"""Tests for shared rotary-factor generation and application."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx

from equimo.core.layers.rotary import (
    RotaryFactors,
    apply_rotary,
    apply_rotary_qk,
    insert_rotary_identity,
    make_1d_rotary_factors,
)
from equimo.tabular.layers.attention import _apply_rope


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("split_half", [-3.0, -4.0, 1.0, 2.0]),
        ("interleaved", [-2.0, 1.0, -4.0, 3.0]),
    ],
)
def test_rotary_layouts_rotate_expected_feature_pairs(layout, expected):
    factors = RotaryFactors(
        sin=jnp.ones((1, 4)),
        cos=jnp.zeros((1, 4)),
        layout=layout,
    )
    actual = apply_rotary(jnp.array([[1.0, 2.0, 3.0, 4.0]]), factors)

    assert jnp.array_equal(actual, jnp.array([expected]))


def test_scaled_factors_rotate_before_reciprocal_qk_scaling():
    factors = RotaryFactors(
        sin=jnp.ones((1, 4)),
        cos=jnp.zeros((1, 4)),
        layout="interleaved",
        scale=jnp.array([[2.0, 3.0, 4.0, 5.0]]),
    )
    values = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])

    q, k = apply_rotary_qk(values, values, factors)
    rotated = jnp.array([[[-2.0, 1.0, -4.0, 3.0]]])

    assert jnp.array_equal(q, rotated * factors.scale)
    assert jnp.allclose(k, rotated / factors.scale)
    with pytest.raises(ValueError, match="scaled factors"):
        apply_rotary(values, factors)


def test_rotary_qk_preserves_each_input_dtype():
    factors = RotaryFactors(
        sin=jnp.ones((2, 4), dtype=jnp.float32),
        cos=jnp.zeros((2, 4), dtype=jnp.float32),
        layout="split_half",
    )
    q = jnp.ones((1, 2, 4), dtype=jnp.bfloat16)
    k = jnp.ones((1, 2, 4), dtype=jnp.float16)

    rotated_q, rotated_k = jax.jit(apply_rotary_qk)(q, k, factors)

    assert rotated_q.dtype == q.dtype
    assert rotated_k.dtype == k.dtype


def test_insert_rotary_identity_preserves_layout_and_scale():
    factors = RotaryFactors(
        sin=jnp.full((2, 4), 0.5),
        cos=jnp.full((2, 4), 0.25),
        layout="interleaved",
        scale=jnp.full((2, 4), 2.0),
    )

    actual = insert_rotary_identity(factors, index=1, count=2)

    assert actual.layout == factors.layout
    assert jnp.array_equal(actual.sin[1:3], jnp.zeros((2, 4)))
    assert jnp.array_equal(actual.cos[1:3], jnp.ones((2, 4)))
    assert actual.scale is not None
    assert jnp.array_equal(actual.scale[1:3], jnp.ones((2, 4)))


def test_make_1d_factors_isolated_by_theta_and_sequence_length():
    short = make_1d_rotary_factors(3, 4, theta=100.0)
    wide_base = make_1d_rotary_factors(3, 4, theta=100_000.0)
    long = make_1d_rotary_factors(7, 4, theta=100.0)
    short_again = make_1d_rotary_factors(3, 4, theta=100.0)

    assert not jnp.array_equal(short.sin, wide_base.sin)
    assert long.sin.shape == (7, 4)
    assert jnp.array_equal(short.sin, short_again.sin)


def test_tabular_rope_bypasses_equinox_theta_blind_cache():
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    narrow = eqx.nn.RotaryPositionalEmbedding(4, theta=100.0)
    wide = eqx.nn.RotaryPositionalEmbedding(4, theta=100_000.0)

    narrow_output = _apply_rope(narrow, x)
    wide_output = _apply_rope(wide, x)

    assert not jnp.array_equal(narrow_output, wide_output)


def test_rotary_qk_requires_full_sequence_factors():
    factors = make_1d_rotary_factors(2, 4)
    q = jnp.ones((1, 3, 4))

    with pytest.raises(ValueError, match="sequence length"):
        apply_rotary_qk(q, q, factors)
