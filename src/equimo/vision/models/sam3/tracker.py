"""Memory encoder and memory attention of the SAM 3.1 tracker."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.layers.attention import CrossAttention, _fused_attention
from equimo.core.layers.norm import LayerNorm2d
from equimo.core.layers.rotary import RotaryFactors, apply_rotary
from equimo.vision.layers.convolution import ConvNeXtBlock
from equimo.vision.layers.posemb import VisionRoPE
from equimo.vision.models.sam3.common import (
    MLP,
    PositionedAttention,
    exchanged_rope_factors,
    layer_norm,
)
from equimo.vision.models.sam3.interactive import MultiplexMaskDecoder, PromptEncoder

# Dense float32 logits against 52,144 keys take about 8.6 GB per layer.
_ATTENTION_CHUNK_QUERIES = 1024
_MEMORY_STRIDE = 16
_FUSER_DEPTH = 2


def _chunked_attention(
    q: Float[Array, "heads query head_dim"],
    k: Float[Array, "heads key head_dim"],
    v: Float[Array, "heads key head_dim"],
    key_lengths: Int[Array, ""],
) -> Float[Array, "heads query head_dim"]:
    """XLA attention over packed keys, one block of queries at a time."""
    heads, num_queries, head_dim = q.shape
    num_chunks = -(-num_queries // _ATTENTION_CHUNK_QUERIES)
    padded_queries = num_chunks * _ATTENTION_CHUNK_QUERIES
    chunks = jnp.pad(q, ((0, 0), (0, padded_queries - num_queries), (0, 0)))
    chunks = jnp.moveaxis(
        chunks.reshape(heads, num_chunks, _ATTENTION_CHUNK_QUERIES, head_dim), 1, 0
    )
    allowed = (jnp.arange(k.shape[1]) < key_lengths)[None, None, :]

    def one_chunk(chunk: jax.Array) -> jax.Array:
        logits = jnp.einsum("hqd,hkd->hqk", chunk, k).astype(jnp.float32) / jnp.sqrt(
            head_dim
        )
        weights = jax.nn.softmax(logits, axis=-1, where=allowed).astype(v.dtype)
        return jnp.einsum("hqk,hkd->hqd", weights, v)

    out = jnp.moveaxis(jax.lax.map(one_chunk, chunks), 0, 1)
    return out.reshape(heads, padded_queries, head_dim)[:, :num_queries]


def _attend_packed_memory(
    attention: CrossAttention,
    queries: Float[Array, "query dim"],
    keys: Float[Array, "key dim"],
    values: Float[Array, "key dim"],
    rotary: RotaryFactors,
    key_rotary: RotaryFactors,
    key_lengths: Int[Array, ""],
) -> Float[Array, "query dim"]:
    """Attend to memory whose valid keys are packed first.

    cuDNN takes the key count directly. Without it, queries run in blocks,
    because float32 logits against every key do not fit in memory.
    """
    q = apply_rotary(attention._heads(attention.q, queries), rotary)
    k = apply_rotary(attention._heads(attention.k, keys), key_rotary)
    v = attention._heads(attention.v, values)
    if attention.implementation == "cudnn":
        out = _fused_attention(q, k, v, key_value_seq_lengths=key_lengths)
    else:
        out = _chunked_attention(q, k, v, key_lengths)
    out = jnp.swapaxes(out, 0, 1).reshape(queries.shape[0], attention.internal_dim)
    return jax.vmap(attention.proj)(out)


def _memory_key_rotary(
    factors: RotaryFactors,
    num_keys: int,
    num_spatial_keys: Int[Array, ""],
    grid: int,
) -> RotaryFactors:
    """Repeat ``factors`` per memory frame and leave object pointers unrotated.

    Spatial keys come first, each frame being one grid of tokens. Everything
    after them gets the identity rotation.
    """
    positions = jnp.arange(num_keys)
    table_index = positions % (grid * grid)
    spatial = (positions < num_spatial_keys)[:, None]
    return RotaryFactors(
        sin=jnp.where(spatial, factors.sin[table_index], 0).astype(factors.sin.dtype),
        cos=jnp.where(spatial, factors.cos[table_index], 1).astype(factors.cos.dtype),
        layout=factors.layout,
    )


class MemoryEncoder(eqx.Module):
    """Fuse a frame's slot masks with its features into a spatial memory."""

    mask_convs: tuple[eqx.nn.Conv2d, ...]
    mask_norms: tuple[LayerNorm2d, ...]
    mask_proj: eqx.nn.Conv2d
    feature_proj: eqx.nn.Conv2d
    fuser: tuple[ConvNeXtBlock, ...]

    def __init__(self, dim: int, in_channels: int, *, key: PRNGKeyArray):
        key_masks, key_mask_proj, key_feature_proj, key_fuser = jr.split(key, 4)
        widths = [min(in_channels * 2**i, dim) for i in range(5)]
        self.mask_convs = tuple(
            eqx.nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, key=subkey)
            for width_in, width_out, subkey in zip(
                widths[:-1], widths[1:], jr.split(key_masks, 4)
            )
        )
        self.mask_norms = tuple(LayerNorm2d(width) for width in widths[1:])
        self.mask_proj = eqx.nn.Conv2d(widths[-1], dim, 1, key=key_mask_proj)
        self.feature_proj = eqx.nn.Conv2d(dim, dim, 1, key=key_feature_proj)
        self.fuser = tuple(
            ConvNeXtBlock(dim, act_layer="exactgelu", init_values=1e-6, key=subkey)
            for subkey in jr.split(key_fuser, _FUSER_DEPTH)
        )

    def __call__(
        self,
        features: Float[Array, "dim grid grid"],
        masks: Float[Array, "channels height width"],
    ) -> Float[Array, "dim grid grid"]:
        """Return the memory for slot masks and conditioning flags in ``masks``."""
        grid = features.shape[-1]
        target = _MEMORY_STRIDE * grid
        x = jax.image.resize(
            masks.astype(jnp.float32),
            (masks.shape[0], target, target),
            method="bilinear",
            antialias=True,
        ).astype(features.dtype)
        for conv, norm in zip(self.mask_convs, self.mask_norms):
            x = jax.nn.gelu(norm(conv(x)), approximate=False)
        out = self.feature_proj(features) + self.mask_proj(x)
        for block in self.fuser:
            out = block(out, inference=True)
        return out


class MemoryAttentionBlock(eqx.Module):
    """Pre-norm block of RoPE self-attention and packed-memory cross-attention."""

    norm1: eqx.nn.LayerNorm
    attn: PositionedAttention
    norm2: eqx.nn.LayerNorm
    cross_attn: CrossAttention
    norm3: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(
        self, dim: int, num_heads: int, ffn_features: int, *, key: PRNGKeyArray
    ):
        key_attn, key_cross, key_mlp = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = PositionedAttention(dim, num_heads, key=key_attn)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, key=key_cross)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.mlp = MLP(dim, ffn_features, dim, 2, key=key_mlp)

    def __call__(
        self,
        x: Float[Array, "tokens dim"],
        memory_keys: Float[Array, "keys dim"],
        memory_values: Float[Array, "keys dim"],
        rotary: RotaryFactors,
        key_rotary: RotaryFactors,
        key_lengths: Int[Array, ""],
    ) -> Float[Array, "tokens dim"]:
        normed = layer_norm(self.norm1, x)
        x = x + self.attn(normed, normed, rotary=rotary, key_rotary=rotary)
        x = x + _attend_packed_memory(
            self.cross_attn,
            layer_norm(self.norm2, x),
            memory_keys,
            memory_values,
            rotary,
            key_rotary,
            key_lengths,
        )
        return x + self.mlp(layer_norm(self.norm3, x))


class MemoryAttention(eqx.Module):
    """Condition the current frame's tokens on spatial memories and object pointers."""

    blocks: tuple[MemoryAttentionBlock, ...]
    norm: eqx.nn.LayerNorm
    rope: VisionRoPE

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        depth: int,
        grid: int,
        *,
        key: PRNGKeyArray,
    ):
        self.blocks = tuple(
            MemoryAttentionBlock(dim, num_heads, ffn_features, key=block_key)
            for block_key in jr.split(key, depth)
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.rope = VisionRoPE(
            "mode",
            dim=dim,
            num_heads=num_heads,
            freqs_for="lang",
            theta=10000,
            pt_seq_len=grid,
        )

    def __call__(
        self,
        x: Float[Array, "tokens dim"],
        memory: Float[Array, "keys dim"],
        memory_pos: Float[Array, "keys dim"],
        *,
        num_spatial_keys: Int[Array, ""],
        key_lengths: Int[Array, ""],
    ) -> Float[Array, "tokens dim"]:
        """Attend to ``memory``, whose valid keys are packed at the front.

        The first ``num_spatial_keys`` keys are spatial tokens of the memory
        frames, followed by object pointers, up to ``key_lengths`` valid keys.
        """
        grid = int(x.shape[0] ** 0.5)
        rotary = exchanged_rope_factors(self.rope, grid)
        key_rotary = _memory_key_rotary(rotary, memory.shape[0], num_spatial_keys, grid)
        keys = memory + memory_pos
        for block in self.blocks:
            x = block(x, keys, memory, rotary, key_rotary, key_lengths)
        return layer_norm(self.norm, x)


class Sam3Tracker(eqx.Module):
    """Modules of the memory-based tracker."""

    prompt_encoder: PromptEncoder
    memory_encoder: MemoryEncoder
    memory_attention: MemoryAttention
    mask_decoder: MultiplexMaskDecoder

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        *,
        key: PRNGKeyArray,
        memory_depth: int,
        memory_heads: int,
        decoder_depth: int,
        num_slots: int,
        grid: int,
        image_size: int,
    ):
        key_prompt, key_memory_encoder, key_attention, key_decoder = jr.split(key, 4)
        self.prompt_encoder = PromptEncoder(dim, grid, image_size, key=key_prompt)
        self.memory_encoder = MemoryEncoder(dim, 2 * num_slots, key=key_memory_encoder)
        self.memory_attention = MemoryAttention(
            dim, memory_heads, ffn_features, memory_depth, grid, key=key_attention
        )
        self.mask_decoder = MultiplexMaskDecoder(
            dim, num_heads, ffn_features, decoder_depth, num_slots, key=key_decoder
        )


__all__: list[str] = []
