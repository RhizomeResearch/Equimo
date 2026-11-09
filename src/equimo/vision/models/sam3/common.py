"""Building blocks shared by the SAM 3.1 encoder, detector and tracker."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.core.layers.attention import (
    CrossAttention,
    _apply_module_last_dim,
    _fused_attention,
)
from equimo.core.layers.rotary import RotaryFactors, apply_rotary
from equimo.vision.layers.posemb import VisionRoPE


class MLP(eqx.Module):
    """Linear layers with an activation between them, applied over the last axis."""

    layers: tuple[eqx.nn.Linear, ...]
    act: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
        act_layer: str | Callable = "relu",
    ):
        sizes = [in_features, *[hidden_features] * (num_layers - 1), out_features]
        self.layers = tuple(
            eqx.nn.Linear(size_in, size_out, key=subkey)
            for size_in, size_out, subkey in zip(
                sizes[:-1], sizes[1:], jr.split(key, num_layers)
            )
        )
        self.act = get_act(act_layer)

    def __call__(self, x: Float[Array, "... in_features"]) -> Array:
        for layer in self.layers[:-1]:
            x = self.act(_apply_module_last_dim(layer, x))
        return _apply_module_last_dim(self.layers[-1], x)


def layer_norm(norm: eqx.nn.LayerNorm, x: jax.Array) -> jax.Array:
    """Apply ``norm`` over the last axis in float32 and return the input dtype."""
    return _apply_module_last_dim(norm, x.astype(jnp.float32)).astype(x.dtype)


def group_norm(norm: eqx.nn.GroupNorm, x: jax.Array) -> jax.Array:
    """Apply ``norm`` to a ``(C, H, W)`` map in float32, returning the input dtype."""
    return norm(x.astype(jnp.float32)).astype(x.dtype)


class PositionedAttention(CrossAttention):
    """Cross-attention whose keys may differ from its values.

    DETR-style attention adds position encodings to queries and keys only, and
    SAM's RoPE rotates queries and keys after projection, neither of which the
    base class can express. ``context`` holds the keys and ``values`` defaults
    to it. ``key_mask`` uses ``True`` for keys to attend to; ``bias`` is an
    additive logit bias broadcasting to ``(heads, query, key)``. The layer has
    no dropout, so ``key`` and ``inference`` are accepted only for the endpoint
    convention.
    """

    def __call__(
        self,
        x: Float[Array, "query dim"],
        context: Float[Array, "key kv_dim"],
        *,
        values: Float[Array, "key kv_dim"] | None = None,
        rotary: RotaryFactors | None = None,
        key_rotary: RotaryFactors | None = None,
        key_mask: jax.Array | None = None,
        bias: jax.Array | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Float[Array, "query dim"]:
        q = self._heads(self.q, x)
        k = self._heads(self.k, context)
        v = self._heads(self.v, context if values is None else values)
        if rotary is not None:
            q = apply_rotary(q, rotary)
        if key_rotary is not None:
            k = apply_rotary(k, key_rotary)

        allowed = None
        if key_mask is not None:
            allowed = jnp.broadcast_to(
                jnp.asarray(key_mask, dtype=bool),
                (self.num_heads, x.shape[0], context.shape[0]),
            )
        if self.implementation == "cudnn":
            out = _fused_attention(q, k, v, allowed, bias=bias)
        else:
            logits = jnp.einsum("hqd,hkd->hqk", q, k).astype(jnp.float32) / jnp.sqrt(
                self.head_dim
            )
            if bias is not None:
                logits = logits + jnp.asarray(bias, jnp.float32)
            weights = jax.nn.softmax(logits, axis=-1, where=allowed).astype(x.dtype)
            out = jnp.einsum("hqk,hkd->hqd", weights, v)
        out = jnp.swapaxes(out, 0, 1).reshape(x.shape[0], self.internal_dim)
        return jax.vmap(self.proj)(out)


def exchanged_rope_factors(rope: VisionRoPE, grid: int) -> RotaryFactors:
    """Return ``rope``'s factors for a square grid, rotating width before height.

    ``VisionRoPE`` lists height angles first. On a square grid, exchanging the two
    grid axes of its table gives the width-first order SAM rotates in. The table
    follows from static configuration, so it is evaluated while tracing.
    """

    def exchange(table: jax.Array) -> jax.Array:
        return table.reshape(grid, grid, -1).swapaxes(0, 1).reshape(grid * grid, -1)

    with jax.ensure_compile_time_eval():
        factors = rope.get_factors(H=grid, W=grid)
        return RotaryFactors(
            exchange(factors.sin), exchange(factors.cos), layout=factors.layout
        )


def sine_coordinates(
    coords: Float[Array, "..."],
    num_features: int,
    *,
    temperature: float = 10000.0,
) -> Float[Array, "... num_features"]:
    """Encode coordinates in ``[0, 1]`` with sines and cosines, in float32.

    Even features are sines and odd features are cosines, as in DETR.
    """
    coords = coords.astype(jnp.float32) * 2.0 * jnp.pi
    index = jnp.arange(num_features, dtype=jnp.float32)
    denominators = temperature ** (2.0 * jnp.floor(index / 2.0) / num_features)
    scaled = coords[..., None] / denominators
    return jnp.where(index % 2 == 0, jnp.sin(scaled), jnp.cos(scaled))


def sine_position_encoding(
    height: int,
    width: int,
    num_features: int,
    *,
    temperature: float = 10000.0,
) -> Float[Array, "channels height width"]:
    """DETR sine encoding of a grid, ``(2 * num_features, H, W)``, in float32.

    Coordinates are normalized to ``(0, 1]``. The channels hold the y encoding
    first, then x. The table follows from the shape, so it is evaluated while
    tracing.
    """
    with jax.ensure_compile_time_eval():
        eps = 1e-6
        ys = jnp.arange(1, height + 1, dtype=jnp.float32) / (height + eps)
        xs = jnp.arange(1, width + 1, dtype=jnp.float32) / (width + eps)
        pos_y = sine_coordinates(ys, num_features, temperature=temperature)
        pos_x = sine_coordinates(xs, num_features, temperature=temperature)
        pos_y = jnp.broadcast_to(pos_y[:, None, :], (height, width, num_features))
        pos_x = jnp.broadcast_to(pos_x[None, :, :], (height, width, num_features))
        return jnp.moveaxis(jnp.concatenate((pos_y, pos_x), axis=-1), -1, 0)


def inverse_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Return ``logit(x)`` in float32 with the logit range the reference clamps to.

    ``x`` and ``1 - x`` are limited to at least 1e-3, which caps the result at
    about 6.9 and keeps it finite at 0 and 1.
    """
    x = jnp.clip(x.astype(jnp.float32), 0.0, 1.0)
    return jnp.log(jnp.maximum(x, 1e-3) / jnp.maximum(1.0 - x, 1e-3))


def flatten_map(x: Float[Array, "channels height width"]) -> jax.Array:
    """Turn a ``(C, H, W)`` map into row-major ``(H * W, C)`` tokens."""
    channels, height, width = x.shape
    return jnp.moveaxis(x, 0, -1).reshape(height * width, channels)


def unflatten_map(x: Float[Array, "tokens channels"], grid: int) -> jax.Array:
    """Turn row-major tokens of a square grid back into a ``(C, H, W)`` map."""
    return jnp.moveaxis(x.reshape(grid, grid, x.shape[-1]), -1, 0)


__all__: list[str] = []
