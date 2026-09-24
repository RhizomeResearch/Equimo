"""Shared ViT-style token and positional-embedding builders.

These free functions return components that model classes assign to their own
existing fields. They must never wrap components in a submodule: nesting would
change ``jtu.keystr`` paths and break every saved checkpoint signature.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from equimo.vision.layers.patch import PatchEmbedding
from equimo.vision.layers.posemb import (
    CompositeVisionRoPE,
    LearnedPosEmbed,
    VisionRoPE,
)


class TokenEmbeddings(NamedTuple):
    """Components and counts for a ViT-style token embedding stack."""

    patch_embed: "PatchEmbedding"
    num_patches: int
    cls_token: jax.Array | None
    reg_tokens: jax.Array | None
    mask_token: jax.Array | None
    num_prefix_tokens: int
    num_embedded_prefix_tokens: int
    embed_len: int
    global_pos_embed: "LearnedPosEmbed | None"


def build_token_embeddings(
    *,
    img_size: int,
    in_channels: int,
    dim: int,
    patch_size: int,
    class_token: bool,
    reg_tokens: int,
    use_mask_token: bool,
    dynamic_img_size: bool,
    dynamic_img_pad: bool,
    global_pos_embed_cls: bool,
    global_pos_embed_reg: bool,
    use_global_pos_embed: bool,
    interpolate_antialias: bool,
    embed_size: int,
    key_patchemb: PRNGKeyArray,
    key_posemb: PRNGKeyArray,
    key_cls: PRNGKeyArray,
    key_reg: PRNGKeyArray,
) -> TokenEmbeddings:
    """Build the patch embedding, prefix tokens, and learned position table."""

    num_prefix_tokens = (1 if class_token else 0) + reg_tokens

    patch_embed = PatchEmbedding(
        in_channels=in_channels,
        embed_dim=dim,
        patch_size=patch_size,
        img_size=img_size,
        flatten=not dynamic_img_size,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        key=key_patchemb,
    )
    num_patches = patch_embed.num_patches
    assert num_patches is not None
    cls_token = jr.normal(key_cls, (1, dim)) if class_token else None
    reg_tokens_array = jr.normal(key_reg, (reg_tokens, dim)) if reg_tokens > 0 else None
    mask_token = jnp.zeros((1, dim)) if use_mask_token else None

    num_embedded_prefix_tokens = 0
    if not global_pos_embed_cls:
        embed_len = num_patches
    elif global_pos_embed_reg:
        embed_len = num_patches + num_prefix_tokens
        num_embedded_prefix_tokens += num_prefix_tokens
    else:
        num_embedded_prefix_tokens += 1
        embed_len = num_patches + 1

    if use_global_pos_embed:
        global_pos_embed = LearnedPosEmbed(
            weight=jr.normal(key_posemb, (embed_len, dim)),
            dim=dim,
            embed_size=embed_size,
            num_prefix_tokens=num_prefix_tokens,
            num_embedded_prefix_tokens=num_embedded_prefix_tokens,
            global_pos_embed_cls=global_pos_embed_cls,
            global_pos_embed_reg=global_pos_embed_reg,
            antialias=interpolate_antialias,
        )
    else:
        global_pos_embed = None

    return TokenEmbeddings(
        patch_embed=patch_embed,
        num_patches=num_patches,
        cls_token=cls_token,
        reg_tokens=reg_tokens_array,
        mask_token=mask_token,
        num_prefix_tokens=num_prefix_tokens,
        num_embedded_prefix_tokens=num_embedded_prefix_tokens,
        embed_len=embed_len,
        global_pos_embed=global_pos_embed,
    )


def build_local_rope(
    *,
    dim: int,
    num_heads: int | list[int],
    use_local_pos_embed: bool,
    class_token: bool,
    local_pos_embed_reg: bool,
    num_prefix_tokens: int,
    num_reg_tokens: int,
    config_patch: dict,
    config_reg: dict,
    static_heads_error: str,
) -> "CompositeVisionRoPE | None":
    """Build the composite patch/register RoPE, or ``None`` when disabled."""

    if not use_local_pos_embed:
        return None
    if not isinstance(num_heads, int):
        raise ValueError(static_heads_error)
    patch_rope = VisionRoPE(
        dim=dim,
        num_heads=num_heads,
        **config_patch,
    )
    n_prefix = (1 if class_token else 0) if local_pos_embed_reg else num_prefix_tokens
    n_reg = num_reg_tokens if local_pos_embed_reg else 0
    reg_rope = (
        VisionRoPE(
            dim=dim,
            num_heads=num_heads,
            **config_reg,
        )
        if n_reg > 0
        else None
    )
    return CompositeVisionRoPE(
        patch_rope,
        reg_rope=reg_rope,
        num_prefix_tokens=n_prefix,
        num_registers=n_reg,
    )
