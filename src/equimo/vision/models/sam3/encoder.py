"""ViTDet trunk and tri-neck of SAM 3.1."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core.layers.ffn import Mlp
from equimo.core.layers.rotary import RotaryFactors
from equimo.vision.layers.posemb import VisionRoPE
from equimo.vision.models.sam3.common import (
    PositionedAttention,
    exchanged_rope_factors,
    flatten_map,
    layer_norm,
)


class ImageFeatures(eqx.Module):
    """Feature pyramids of one image, each ordered from high to low resolution."""

    detector: tuple[jax.Array, jax.Array, jax.Array]
    interactive: tuple[jax.Array, jax.Array, jax.Array]
    propagation: tuple[jax.Array, jax.Array, jax.Array]


class ViTDetBlock(eqx.Module):
    """Pre-norm transformer block that attends globally or within square windows.

    The block partitions its own windows and computes its own RoPE grid, so that
    every block sits in one flat list and global and windowed blocks can use
    different grids.
    """

    norm1: eqx.nn.LayerNorm
    attn: PositionedAttention
    norm2: eqx.nn.LayerNorm
    mlp: Mlp
    rope: VisionRoPE
    window_size: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_features: int,
        *,
        key: PRNGKeyArray,
        window_size: int,
        rope_grid: int,
    ):
        key_attn, key_mlp = jr.split(key)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = PositionedAttention(dim, num_heads, key=key_attn)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=mlp_features, act_layer="exactgelu", key=key_mlp)
        self.rope = VisionRoPE(
            "mode",
            dim=dim,
            num_heads=num_heads,
            freqs_for="lang",
            theta=10000,
            pt_seq_len=rope_grid,
        )
        self.window_size = window_size

    def _forward(
        self, x: Float[Array, "tokens dim"], rotary: RotaryFactors
    ) -> Float[Array, "tokens dim"]:
        normed = layer_norm(self.norm1, x)
        x = x + self.attn(normed, normed, rotary=rotary, key_rotary=rotary)
        return x + self.mlp(layer_norm(self.norm2, x), None, inference=True)

    def __call__(
        self, x: Float[Array, "dim grid grid"]
    ) -> Float[Array, "dim grid grid"]:
        dim, grid, _ = x.shape
        if self.window_size == 0:
            rotary = exchanged_rope_factors(self.rope, grid)
            out = self._forward(flatten_map(x), rotary)
            return jnp.moveaxis(out.reshape(grid, grid, dim), -1, 0)

        rotary = exchanged_rope_factors(self.rope, self.window_size)
        tokens = rearrange(
            x,
            "c (m h) (n w) -> (m n) (h w) c",
            h=self.window_size,
            w=self.window_size,
        )
        out = jax.vmap(self._forward, in_axes=(0, None))(tokens, rotary)
        return rearrange(
            out,
            "(m n) (h w) c -> c (m h) (n w)",
            m=grid // self.window_size,
            n=grid // self.window_size,
            h=self.window_size,
            w=self.window_size,
        )


class ViTDetTrunk(eqx.Module):
    """Vision transformer with windowed blocks and a few global blocks.

    The learned position table covers one window's worth of tokens and is tiled
    over the grid at run time, so fine-tuning updates a single copy of it.
    """

    patch_embed: eqx.nn.Conv2d
    pos_embed: Float[Array, "tokens embed_dim"]
    ln_pre: eqx.nn.LayerNorm
    blocks: tuple[ViTDetBlock, ...]
    patch_size: int = eqx.field(static=True)
    table_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        key: PRNGKeyArray,
        in_channels: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_features: int,
        window_size: int,
        global_blocks: tuple[int, ...],
    ):
        grid = img_size // patch_size
        if img_size % patch_size != 0 or grid % window_size != 0:
            raise ValueError(
                f"img_size {img_size} must divide into patches of {patch_size} and "
                f"then into windows of {window_size} tokens."
            )
        key_patch, key_pos, *key_blocks = jr.split(key, depth + 2)
        self.patch_embed = eqx.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=False,
            key=key_patch,
        )
        self.pos_embed = 0.02 * jr.normal(key_pos, (window_size**2, embed_dim))
        self.ln_pre = eqx.nn.LayerNorm(embed_dim)
        self.blocks = tuple(
            ViTDetBlock(
                embed_dim,
                num_heads,
                mlp_features,
                window_size=0 if index in global_blocks else window_size,
                rope_grid=window_size,
                key=block_key,
            )
            for index, block_key in enumerate(key_blocks)
        )
        self.patch_size = patch_size
        self.table_size = window_size

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "embed_dim grid grid"]:
        patches = self.patch_embed(x)
        dim, grid, _ = patches.shape
        table = self.pos_embed.astype(patches.dtype).reshape(
            self.table_size, self.table_size, dim
        )
        repeats = grid // self.table_size
        table = jnp.tile(table, (repeats, repeats, 1))
        tokens = flatten_map(patches) + table.reshape(grid * grid, dim)
        tokens = layer_norm(self.ln_pre, tokens)
        out = jnp.moveaxis(tokens.reshape(grid, grid, dim), -1, 0)
        for block in self.blocks:
            out = block(out)
        return out


def _upsample_twofold(
    conv: eqx.nn.ConvTranspose2d, x: Float[Array, "channels height width"]
) -> jax.Array:
    """Apply a 2x2 stride-2 transposed convolution as a per-pixel matmul.

    The windows do not overlap, so this equals the transposed convolution while
    avoiding the bfloat16 cost and export gaps of lowering it as a convolution
    (see ``eomt.ScaleBlock``).
    """
    assert conv.bias is not None
    _, height, width = x.shape
    # Equinox correlates the stored kernel with the dilated input.
    weight = conv.weight[:, :, ::-1, ::-1]
    accumulation_dtype = jnp.promote_types(jnp.result_type(weight, x), jnp.float32)
    upscaled = jnp.einsum(
        "oiab,ihw->ohawb", weight, x, preferred_element_type=accumulation_dtype
    )
    upscaled = upscaled.reshape(-1, 2 * height, 2 * width) + conv.bias
    return upscaled.astype(x.dtype)


class PyramidLevel(eqx.Module):
    """One output level of a simple feature pyramid."""

    upsample: tuple[eqx.nn.ConvTranspose2d, ...]
    proj: eqx.nn.Conv2d
    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        *,
        key: PRNGKeyArray,
    ):
        num_upsample = {4: 2, 2: 1, 1: 0}[scale]
        key_proj, key_conv, *key_up = jr.split(key, 2 + num_upsample)
        widths = [in_channels // 2**i for i in range(num_upsample + 1)]
        self.upsample = tuple(
            eqx.nn.ConvTranspose2d(width_in, width_out, 2, stride=2, key=subkey)
            for width_in, width_out, subkey in zip(widths[:-1], widths[1:], key_up)
        )
        self.proj = eqx.nn.Conv2d(widths[-1], out_channels, 1, key=key_proj)
        self.conv = eqx.nn.Conv2d(
            out_channels, out_channels, 3, padding=1, key=key_conv
        )

    def __call__(self, x: Float[Array, "channels grid grid"]) -> jax.Array:
        for index, conv in enumerate(self.upsample):
            if index > 0:
                x = jax.nn.gelu(x, approximate=False)
            x = _upsample_twofold(conv, x)
        return self.conv(self.proj(x))


class FeaturePyramid(eqx.Module):
    """Simple feature pyramid at scale factors 4, 2 and 1 of the trunk's map."""

    levels: tuple[PyramidLevel, PyramidLevel, PyramidLevel]

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        key4, key2, key1 = jr.split(key, 3)
        self.levels = (
            PyramidLevel(in_channels, out_channels, 4, key=key4),
            PyramidLevel(in_channels, out_channels, 2, key=key2),
            PyramidLevel(in_channels, out_channels, 1, key=key1),
        )

    def __call__(
        self, x: Float[Array, "channels grid grid"]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        high, middle, low = self.levels
        return high(x), middle(x), low(x)


class TriNeck(eqx.Module):
    """One feature pyramid each for the detector, interactive head and tracker."""

    detector: FeaturePyramid
    interactive: FeaturePyramid
    propagation: FeaturePyramid

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        key_detector, key_interactive, key_propagation = jr.split(key, 3)
        self.detector = FeaturePyramid(in_channels, out_channels, key=key_detector)
        self.interactive = FeaturePyramid(
            in_channels, out_channels, key=key_interactive
        )
        self.propagation = FeaturePyramid(
            in_channels, out_channels, key=key_propagation
        )

    def __call__(self, x: Float[Array, "channels grid grid"]) -> ImageFeatures:
        return ImageFeatures(
            detector=self.detector(x),
            interactive=self.interactive(x),
            propagation=self.propagation(x),
        )


__all__ = ["ImageFeatures"]
