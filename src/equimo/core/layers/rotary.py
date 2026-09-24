"""Shared rotary-position factor generation and application."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

RotaryLayout = Literal["split_half", "interleaved"]


class RotaryFactors(eqx.Module):
    """Runtime rotary factors with an explicit feature-pairing convention.

    ``sin`` and ``cos`` have shape ``(sequence, feature)``. When ``scale`` is
    present, :func:`apply_rotary_qk` applies it after rotation to queries and
    applies its reciprocal to keys.
    """

    sin: jax.Array
    cos: jax.Array
    scale: jax.Array | None
    layout: RotaryLayout = eqx.field(static=True)

    def __init__(
        self,
        sin: jax.Array,
        cos: jax.Array,
        layout: RotaryLayout,
        scale: jax.Array | None = None,
    ) -> None:
        sin = jnp.asarray(sin)
        cos = jnp.asarray(cos)
        if layout not in ("split_half", "interleaved"):
            raise ValueError(
                f"layout must be either 'split_half' or 'interleaved'; got {layout!r}."
            )
        if sin.ndim != 2 or cos.ndim != 2:
            raise ValueError("sin and cos must have shape (sequence, feature).")
        if sin.shape != cos.shape:
            raise ValueError(
                f"sin and cos shapes must match; got {sin.shape} and {cos.shape}."
            )
        if sin.shape[-1] == 0 or sin.shape[-1] % 2 != 0:
            raise ValueError("rotary feature dimension must be positive and even.")
        if sin.dtype != cos.dtype:
            raise ValueError(
                f"sin and cos dtypes must match; got {sin.dtype} and {cos.dtype}."
            )
        if scale is not None:
            scale = jnp.asarray(scale)
            if scale.shape != sin.shape:
                raise ValueError(
                    "scale must match sin/cos shape; got "
                    f"{scale.shape} and {sin.shape}."
                )

        self.sin = sin
        self.cos = cos
        self.scale = scale
        self.layout = layout

    @property
    def sequence_length(self) -> int:
        return self.sin.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.sin.shape[1]


def _rotate_split_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _rotate_interleaved(x: jax.Array) -> jax.Array:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)


def _broadcast_factor(
    factor: jax.Array,
    x: jax.Array,
    sequence_axis: int,
) -> jax.Array:
    axis = sequence_axis % x.ndim
    if axis == x.ndim - 1:
        raise ValueError("sequence_axis cannot refer to the feature dimension.")
    shape = [1] * x.ndim
    shape[axis] = factor.shape[0]
    shape[-1] = factor.shape[1]
    return factor.reshape(shape)


def _apply_rotary_unscaled(
    x: jax.Array,
    factors: RotaryFactors,
    *,
    sequence_axis: int,
) -> jax.Array:
    axis = sequence_axis % x.ndim
    if x.shape[-1] != factors.feature_dim:
        raise ValueError(
            "rotary feature dimension must match the input; got "
            f"{factors.feature_dim} and {x.shape[-1]}."
        )
    if x.shape[axis] != factors.sequence_length:
        raise ValueError(
            "rotary sequence length must match the input; got "
            f"{factors.sequence_length} and {x.shape[axis]}."
        )

    x = x.astype(factors.sin.dtype)
    sin = _broadcast_factor(factors.sin, x, axis)
    cos = _broadcast_factor(factors.cos, x, axis)
    rotated = (
        _rotate_split_half(x)
        if factors.layout == "split_half"
        else _rotate_interleaved(x)
    )
    return (x * cos) + (rotated * sin)


def apply_rotary(
    x: jax.Array,
    factors: RotaryFactors,
    *,
    sequence_axis: int = -2,
) -> jax.Array:
    """Apply unscaled rotary factors and preserve the input dtype."""

    if factors.scale is not None:
        raise ValueError("scaled factors must be applied to Q/K together.")
    dtype = x.dtype
    return _apply_rotary_unscaled(
        x,
        factors,
        sequence_axis=sequence_axis,
    ).astype(dtype)


def apply_rotary_qk(
    q: jax.Array,
    k: jax.Array,
    factors: RotaryFactors,
) -> tuple[jax.Array, jax.Array]:
    """Apply rotary factors to Q/K and preserve their respective dtypes."""

    q_dtype, k_dtype = q.dtype, k.dtype
    q = _apply_rotary_unscaled(q, factors, sequence_axis=-2)
    k = _apply_rotary_unscaled(k, factors, sequence_axis=-2)
    if factors.scale is not None:
        scale = _broadcast_factor(
            factors.scale.astype(factors.sin.dtype),
            q,
            -2,
        )
        q = q * scale
        k = k / scale
    return q.astype(q_dtype), k.astype(k_dtype)


def make_1d_rotary_factors(
    sequence_length: int,
    feature_dim: int,
    *,
    theta: float = 10_000.0,
    layout: RotaryLayout = "split_half",
    dtype: jnp.dtype = jnp.float32,
) -> RotaryFactors:
    """Generate deterministic one-dimensional rotary factors without caching."""

    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative.")
    if feature_dim <= 0 or feature_dim % 2 != 0:
        raise ValueError("feature_dim must be positive and even.")
    if theta <= 0:
        raise ValueError("theta must be positive.")

    frequencies = 1.0 / (
        theta ** (jnp.arange(0, feature_dim, 2, dtype=jnp.float32) / float(feature_dim))
    )
    positions = jnp.arange(sequence_length, dtype=jnp.float32)
    angles = jnp.outer(positions, frequencies)
    if layout == "split_half":
        angles = jnp.tile(angles, (1, 2))
    elif layout == "interleaved":
        angles = jnp.repeat(angles, 2, axis=-1)
    else:
        raise ValueError(
            f"layout must be either 'split_half' or 'interleaved'; got {layout!r}."
        )
    dtype = jnp.dtype(dtype)
    return RotaryFactors(
        sin=jnp.sin(angles).astype(dtype),
        cos=jnp.cos(angles).astype(dtype),
        layout=layout,
    )


def insert_rotary_identity(
    factors: RotaryFactors,
    *,
    index: int,
    count: int,
) -> RotaryFactors:
    """Insert unrotated sequence rows, including unit reciprocal scales."""

    if not 0 <= index <= factors.sequence_length:
        raise ValueError(
            f"index must be in [0, {factors.sequence_length}]; got {index}."
        )
    if count < 0:
        raise ValueError("count must be non-negative.")

    shape = (count, factors.feature_dim)
    sin = jnp.concatenate(
        (factors.sin[:index], jnp.zeros(shape, factors.sin.dtype), factors.sin[index:]),
        axis=0,
    )
    cos = jnp.concatenate(
        (factors.cos[:index], jnp.ones(shape, factors.cos.dtype), factors.cos[index:]),
        axis=0,
    )
    scale = None
    if factors.scale is not None:
        scale = jnp.concatenate(
            (
                factors.scale[:index],
                jnp.ones(shape, factors.scale.dtype),
                factors.scale[index:],
            ),
            axis=0,
        )
    return RotaryFactors(sin=sin, cos=cos, layout=factors.layout, scale=scale)


__all__ = [
    "RotaryFactors",
    "RotaryLayout",
    "apply_rotary",
    "apply_rotary_qk",
    "insert_rotary_identity",
    "make_1d_rotary_factors",
]
