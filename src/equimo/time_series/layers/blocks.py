# ty: ignore[invalid-assignment]
# Adapted from tfc-t0 and modified for JAX/Equinox; see NOTICE.
from typing import cast

import equinox as eqx
import jax.random as jr

from equimo.core.layers import RMSNormGated, SwiGluFused

from .axis_attn import AxisAttention
from .registry import get_layer, register_layer


@register_layer()
class T0Block(eqx.Module):
    attn_norm: RMSNormGated
    attn: AxisAttention
    ffn_norm: RMSNormGated
    ffn: SwiGluFused
    ffn_dropout: eqx.nn.Dropout
    attention_type: str = eqx.field(static=True)

    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        dropout,
        attention_type,
        *,
        attention_layer="axisattention",
        key,
        drop_path=0.0,
    ):
        del drop_path
        key1, key2 = jr.split(key)
        attention_layer = cast(type[AxisAttention], get_layer(attention_layer))
        self.attn_norm = RMSNormGated(dim, eps=1e-8)
        self.attn = attention_layer(dim, num_heads, dropout, attention_type, key=key1)
        self.ffn_norm = RMSNormGated(dim, eps=1e-8)
        # SwiGluFused applies the LLaMA 2/3 contraction internally.
        self.ffn = SwiGluFused(
            dim, hidden_dim=hidden_dim * 3 // 2, dropout_rate=0.0, key=key2
        )
        self.ffn_dropout = eqx.nn.Dropout(dropout)
        self.attention_type = attention_type

    def __call__(
        self,
        x,
        *,
        time_attn_mask,
        group_attn_mask,
        key,
        inference=None,
    ):
        key1, key2, key3 = jr.split(key, 3)
        mask = time_attn_mask if self.attention_type == "time" else group_attn_mask
        x += self.attn(self.attn_norm(x), mask, key=key1, inference=inference)
        shape = x.shape
        flat = self.ffn_norm(x).reshape(-1, shape[-1])
        out = self.ffn(flat, key=key2, inference=inference).reshape(shape)
        return x + self.ffn_dropout(out, key=key3, inference=inference)
