# ty: ignore[invalid-assignment]

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from equimo.core.layers import Attention, RMSNormGated
from equimo.core.layers.attention import rope_apply_interleaved

from .registry import register_layer


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
        rope_apply_interleaved(q, sin, cos) * scale,
        rope_apply_interleaved(k, sin, cos) / scale,
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
        x = self.attention(
            x,
            mask=mask,
            qk_transform=_xpos if self.attention_type == "time" else None,
            key=key,
            inference=inference,
        )
        return jnp.swapaxes(x, 0, 1) if self.attention_type == "group" else x
