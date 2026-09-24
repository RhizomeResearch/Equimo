# Adapted from tfc-t0 and modified for JAX/Equinox; see NOTICE.


import equinox as eqx
import jax.numpy as jnp

from equimo.core.layers import (
    Attention,
    RMSNormGated,
    RotaryFactors,
    make_1d_rotary_factors,
)

from .registry import register_layer


def _time_rotary_factors(seq_len: int, dim: int) -> RotaryFactors:
    """Build T0's exact interleaved RoPE and reciprocal XPos factors."""
    factors = make_1d_rotary_factors(
        seq_len,
        dim,
        theta=10_000.0,
        layout="interleaved",
        dtype=jnp.float32,
    )
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    base = (jnp.arange(0, dim, 2, dtype=jnp.float32) + 0.4 * dim) / (1.4 * dim)
    power = (positions - (seq_len - 1) // 2) / 512.0
    half_scale = base[None] ** power[:, None]
    scale = jnp.concatenate((half_scale, half_scale), axis=-1)
    return RotaryFactors(
        sin=factors.sin,
        cos=factors.cos,
        layout=factors.layout,
        scale=scale,
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
        rotary = (
            _time_rotary_factors(x.shape[-2], self.attention.head_dim)
            if self.attention_type == "time"
            else None
        )
        x = self.attention(
            x,
            mask=mask,
            rotary=rotary,
            key=key,
            inference=inference,
        )
        return jnp.swapaxes(x, 0, 1) if self.attention_type == "group" else x
