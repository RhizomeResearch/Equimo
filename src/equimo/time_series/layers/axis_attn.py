# ty: ignore[invalid-assignment]

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from equimo.core.layers import Attention, RMSNormGated

from .registry import register_layer


def _linear(layer: eqx.nn.Linear, x: Array) -> Array:
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


def _rotate_half(x: Array) -> Array:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)


def _xpos(q: Array, k: Array) -> tuple[Array, Array]:
    """Exact rotary-embedding-torch 0.8.x RoPE + XPos."""
    dim, seq_len = q.shape[-1], q.shape[-2]
    positions = jnp.arange(seq_len, dtype=q.dtype)
    frequencies = 1.0 / (10_000 ** (jnp.arange(0, dim, 2) / dim))
    angles = jnp.repeat(jnp.outer(positions, frequencies), 2, axis=-1)
    base = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
    power = (positions - (seq_len - 1) // 2) / 512.0
    half_scale = base[None] ** power[:, None]
    scale = jnp.concatenate((half_scale, half_scale), axis=-1)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return (
        (q * cos + _rotate_half(q) * sin) * scale,
        (k * cos + _rotate_half(k) * sin) / scale,
    )


@register_layer()
class AxisAttention(eqx.Module):
    """T0 axis routing and XPos around Equimo's shared attention parameters."""

    attention: Attention
    attention_type: str = eqx.field(static=True)

    def __init__(self, dim, num_heads, dropout, attention_type, *, key):
        self.attention = Attention(
            dim,
            num_heads,
            qk_norm=True,
            norm_layer=RMSNormGated,
            eps=1e-8,
            attn_drop=dropout,
            proj_drop=dropout,
            key=key,
        )
        self.attention_type = attention_type

    def __call__(self, x, mask, *, key, inference=None):
        if self.attention_type == "group":
            x = jnp.swapaxes(x, 0, 1)
        key1, key2 = jr.split(key)
        attn = self.attention
        qkv = _linear(attn.qkv, x).reshape(
            *x.shape[:-1], 3, attn.num_heads, attn.head_dim
        )
        q, k, v = jnp.moveaxis(qkv, -3, 0)
        q, k, v = (jnp.swapaxes(value, -3, -2) for value in (q, k, v))
        q, k = attn.q_norm(q), attn.k_norm(k)  # ty: ignore[call-non-callable]
        if self.attention_type == "time":
            q, k = _xpos(q, k)
        weights = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(attn.head_dim)
        weights = jnp.where(mask[..., None, :, :], weights, -jnp.inf)
        weights = jnp.nan_to_num(
            jax.nn.softmax(weights.astype(jnp.float32), axis=-1)
        ).astype(x.dtype)
        weights = attn.attn_drop(weights, key=key1, inference=inference)
        x = jnp.einsum("...hqk,...hkd->...hqd", weights, v)
        x = jnp.swapaxes(x, -3, -2).reshape(*x.shape[:-3], x.shape[-2], -1)
        x = _linear(attn.proj, x)
        x = attn.proj_drop(x, key=key2, inference=inference)
        return jnp.swapaxes(x, 0, 1) if self.attention_type == "group" else x
