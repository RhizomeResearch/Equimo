# ty: ignore[invalid-assignment]
# ty: ignore[call-non-callable]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]
# ty: ignore[invalid-argument-type]
__all__ = [
    "VisionTransformer",
    # Standard ViT presets
    "vit_tiny_patch16_224",
    "vit_tiny_patch32_224",
    "vit_small_patch16_224",
    "vit_small_patch32_224",
    "vit_base_patch16_224",
    "vit_base_patch32_224",
    "vit_large_patch16_224",
    "vit_large_patch32_224",
    "vit_huge_patch14_224",
    "vit_huge_patch16_224",
    # DINOv2
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "dinov2_vitl14",
    "dinov2_vitl14_reg",
    "dinov2_vitg14",
    "dinov2_vitg14_reg",
    # DINOv3
    "dinov3_vits16_pretrain_lvd1689m",
    "dinov3_vits16plus_pretrain_lvd1689m",
    "dinov3_vitb16_pretrain_lvd1689m",
    "dinov3_vitl16_pretrain_lvd1689m",
    "dinov3_vith16plus_pretrain_lvd1689m",
    "dinov3_vit7b16_pretrain_lvd1689m",
    "dinov3_vitl16_pretrain_sat493m",
    "dinov3_vit7b16_pretrain_sat493m",
    # LingBot-Vision
    "lingbot_vits16",
    "lingbot_vitb16",
    "lingbot_vitl16",
    "lingbot_vitg16",
    # EUPE
    "eupe_vitt16",
    "eupe_vits16",
    "eupe_vitb16",
    # SigLIP2
    "siglip2_vitb16_224",
    "siglip2_vitb16_256",
    "siglip2_vitb16_384",
    "siglip2_vitb16_512",
    "siglip2_vitb32_256",
    "siglip2_vitl16_256",
    "siglip2_vitl16_384",
    "siglip2_vitl16_512",
    "siglip2_vitso400m14_224",
    "siglip2_vitso400m14_378",
    "siglip2_vitso400m16_256",
    "siglip2_vitso400m16_384",
    "siglip2_vitso400m16_512",
    "siglip2_vitgiantopt16_256",
    "siglip2_vitgiantopt16_384",
    # TIPS
    "tips_vits14_hr",
    "tips_vitb14_hr",
    "tips_vitl14_hr",
    "tips_vitso400m14_hr",
    "tips_vitg14_lr",
    "tips_vitg14_hr",
    # ViT-5
    "vit5_small",
    "vit5_base",
    "vit5_large",
    "vit5_xlarge",
]

from typing import Callable, Literal, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core._prng import split_for_mode
from equimo.core.intermediates import intermediate_indices
from equimo.core.layers.activation import get_act
from equimo.vision.layers.attention import (
    get_attn,
    get_attn_block,
)
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.generic import (
    BlockChunk,
    count_chunk_blocks,
    make_transformer_block_chunk,
)
from equimo.core.layers.norm import get_norm
from equimo.core.layers.rotary import RotaryFactors
from equimo.vision.layers.patch import PatchEmbedding
from equimo.vision.models._embedding import build_local_rope, build_token_embeddings
from equimo.vision.layers.posemb import LearnedPosEmbed, CompositeVisionRoPE
from equimo.registry import register_model
from equimo.utils import make_drop_path_schedule, pool_sd, to_list
from equimo.core.factory import build_model_variant


@register_model("vit", modality="vision")
class VisionTransformer(eqx.Module):
    """Vision Transformer (ViT) implementation.

    A transformer architecture for image processing that divides input images into patches,
    processes them through transformer blocks, and includes options for class tokens,
    registration tokens, and various pooling strategies.

    Attributes:
        patch_embed: Patch embedding layer
        global_pos_embed: Model-level positional embedding applied after patching (e.g. APE)
        local_pos_embed: Block-level positional embedding passed to each attention block (e.g. RoPE)
        cls_token: Class token for classification (optional)
        reg_tokens: Registration tokens for alignment (optional)
        blocks: List of transformer blocks
        pos_drop: Positional dropout layer
        norm: Normalization layer
        head: Classification head
        dim: Model dimension
        num_patches: Number of image patches
        global_pool: Global pooling strategy
        num_reg_tokens: Number of registration tokens
        num_prefix_tokens: Total number of prefix tokens
        num_embedded_prefix_tokens: Number of embedded prefix tokens
        global_pos_embed_cls: Whether the class token receives global positional embedding
        global_pos_embed_reg: Whether reg tokens receive global positional embedding
        local_pos_embed_reg: Whether reg tokens receive local positional embedding (RoPE)
        embed_len: Total embedding length
        dynamic_img_size: Whether to support dynamic image sizes
        antialias: Whether to use antialiasing in interpolation
    """

    patch_embed: PatchEmbedding
    global_pos_embed: LearnedPosEmbed | None
    local_pos_embed: CompositeVisionRoPE | None
    cls_token: jax.Array | None
    reg_tokens: jax.Array | None
    mask_token: jax.Array | None
    blocks: Tuple[BlockChunk, ...]
    pos_drop: eqx.nn.Dropout
    norm: eqx.Module
    local_cls_norm: eqx.Module | None
    head: eqx.Module

    dim: int = eqx.field(static=True)
    embed_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    global_pool: str = eqx.field(static=True)
    num_reg_tokens: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    num_embedded_prefix_tokens: int = eqx.field(static=True)
    global_pos_embed_cls: bool = eqx.field(static=True)
    global_pos_embed_reg: bool = eqx.field(static=True)
    local_pos_embed_reg: bool = eqx.field(static=True)
    embed_len: int = eqx.field(static=True)
    dynamic_img_size: bool = eqx.field(static=True)
    antialias: bool = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        dim: int,
        patch_size: int,
        num_heads: int | list[int],
        depths: list[int],
        *,
        key: PRNGKeyArray,
        use_mask_token: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        class_token: bool = True,
        global_pos_embed_cls: bool = True,
        global_pos_embed_reg: bool = False,
        local_pos_embed_reg: bool = False,
        reg_tokens: int = 4,
        use_global_pos_embed: bool = True,
        use_local_pos_embed: bool = False,
        local_pos_embed_config_patch: dict = {
            "strategy": "period",
            "base": 100.0,
            "normalize_coords": "separate",
            "dtype": jnp.float32,
        },
        local_pos_embed_config_reg: dict = {
            "strategy": "period",
            "base": 100.0,
            "normalize_coords": "separate",
            "dtype": jnp.float32,
        },
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: str | type[eqx.Module] = "attentionblock",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: str | Callable = "gelu",
        attn_layer: str | type[eqx.Module] = "attention",
        ffn_layer: str | type[eqx.Module] = "mlp",
        ffn_bias: bool = True,
        ffn_kwargs: dict = {},
        norm_layer: str | type[eqx.Module] = "layernorm",
        untie_global_and_local_cls_norm: bool = False,
        init_values: float | None = None,
        global_pool: Literal[
            "", "token", "cls_patch_mean", "avg", "avgmax", "max"
        ] = "avg",
        num_classes: int | None = 1000,
        interpolate_antialias: bool = False,
        eps: float = 1e-5,
        **kwargs,
    ):
        depth = sum(depths)
        key_patchemb, key_posemb, key_cls, key_reg, key_head, *block_subkeys = jr.split(
            key, 5 + len(depths)
        )
        self.dim = dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.dynamic_img_size = dynamic_img_size
        self.antialias = interpolate_antialias
        self.global_pos_embed_cls = global_pos_embed_cls
        self.global_pos_embed_reg = global_pos_embed_reg
        self.local_pos_embed_reg = local_pos_embed_reg
        self.global_pool = global_pool
        self.embed_size = img_size // patch_size

        block = get_attn_block(block)
        attn_layer = get_attn(attn_layer)
        ffn_layer = get_ffn(ffn_layer)
        norm_layer = get_norm(norm_layer)
        act_layer = get_act(act_layer)

        embeddings = build_token_embeddings(
            img_size=img_size,
            in_channels=in_channels,
            dim=dim,
            patch_size=patch_size,
            class_token=class_token,
            reg_tokens=reg_tokens,
            use_mask_token=use_mask_token,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            global_pos_embed_cls=global_pos_embed_cls,
            global_pos_embed_reg=global_pos_embed_reg,
            use_global_pos_embed=use_global_pos_embed,
            interpolate_antialias=interpolate_antialias,
            embed_size=self.embed_size,
            key_patchemb=key_patchemb,
            key_posemb=key_posemb,
            key_cls=key_cls,
            key_reg=key_reg,
        )
        self.patch_embed = embeddings.patch_embed
        self.num_patches = embeddings.num_patches
        self.cls_token = embeddings.cls_token
        self.reg_tokens = embeddings.reg_tokens
        self.mask_token = embeddings.mask_token
        self.num_embedded_prefix_tokens = embeddings.num_embedded_prefix_tokens
        self.embed_len = embeddings.embed_len
        self.global_pos_embed = embeddings.global_pos_embed

        self.local_pos_embed = build_local_rope(
            dim=dim,
            num_heads=num_heads,
            use_local_pos_embed=use_local_pos_embed,
            class_token=class_token,
            local_pos_embed_reg=local_pos_embed_reg,
            num_prefix_tokens=self.num_prefix_tokens,
            num_reg_tokens=self.num_reg_tokens,
            config_patch=local_pos_embed_config_patch,
            config_reg=local_pos_embed_config_reg,
            static_heads_error=(
                "Local pos embedding (RoPE) currently requires a static number of heads."
            ),
        )
        self.pos_drop = eqx.nn.Dropout(pos_drop_rate)

        dpr = make_drop_path_schedule(
            drop_path_rate, [depth], uniform=drop_path_uniform
        )

        n_chunks = len(depths)
        dims = to_list(dim, n_chunks)
        num_heads = to_list(num_heads, n_chunks)
        attn_layer = to_list(attn_layer, n_chunks)
        self.blocks = tuple(
            chunk
            for i, depth in enumerate(depths)
            if (
                chunk := make_transformer_block_chunk(
                    depth=depths[i],
                    dim=dims[i],
                    num_heads=num_heads[i],
                    block=block,
                    attn_layer=attn_layer[i],
                    ffn_layer=ffn_layer,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    act_layer=act_layer,
                    ffn_bias=ffn_bias,
                    ffn_kwargs=ffn_kwargs,
                    norm_layer=norm_layer,
                    eps=eps,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    init_values=init_values,
                    key=block_subkeys[i],
                )
            )
            is not None
        )

        self.norm = norm_layer(dim, eps=eps)

        # WARNING: This has no effect in the code.
        # This norm layer is created to hold some training-only norm layer of Dinov3
        self.local_cls_norm = (
            norm_layer(dim, eps=eps) if untie_global_and_local_cls_norm else None
        )

        head_in_features = 2 * dim if global_pool == "cls_patch_mean" else dim
        self.head = (
            eqx.nn.Linear(head_in_features, num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]] = None,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        """Extract features from input image.

        Args:
            x: Input image tensor
            inference: Whether to run stochastic layers in inference mode;
                True disables dropout and drop-path.
            key: PRNG key for random operations
            mask: optional binary mask of the size of the input after patch embedding

        Returns:
            Processed feature tensor
        """
        key_pos = split_for_mode(key, len(self.blocks) + 1, inference=inference)[0]
        x, H, W, rotary = self._prepare_tokens(
            x,
            key=key_pos,
            mask=mask,
            inference=inference,
        )
        return self._run_blocks(
            (x, H, W, rotary),
            key=key,
            inference=inference,
            **kwargs,
        )

    def intermediate_features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]] = None,
        inference: Optional[bool] = None,
        indices: Sequence[int] | None = None,
        n_last_blocks: int | None = None,
        apply_norm: bool = False,
        **kwargs,
    ) -> tuple[Float[Array, "seqlen dim"], ...]:
        """Return selected token outputs, optionally after the final norm."""

        total = count_chunk_blocks(self.blocks)
        wanted = intermediate_indices(
            total, indices=indices, n_last_blocks=n_last_blocks
        )
        key_pos, *block_subkeys = split_for_mode(
            key, len(self.blocks) + 1, inference=inference
        )
        x, H, W, rotary = self._prepare_tokens(
            x,
            key=key_pos,
            mask=mask,
            inference=inference,
        )

        outputs = []
        offset = 0
        for blk, key_block in zip(self.blocks, block_subkeys):
            if self.local_pos_embed is not None and not inference:
                key_pos, key_rope = jr.split(key_pos, 2)
                rotary = self.local_pos_embed.get_factors(
                    H=H, W=W, inference=inference, key=key_rope
                )
            blocks = blk.blocks
            n_blocks = 0 if blocks is None else len(blocks)
            local_indices = tuple(
                i - offset for i in sorted(wanted) if offset <= i < offset + n_blocks
            )
            if local_indices:
                x, chunk_outputs = blk.intermediate_features(
                    x,
                    rotary=rotary,
                    inference=inference,
                    key=key_block,
                    indices=local_indices,
                    **kwargs,
                )
                outputs.extend(
                    jax.vmap(self.norm)(output) if apply_norm else output
                    for output in chunk_outputs
                )
            else:
                x = blk(
                    x,
                    rotary=rotary,
                    inference=inference,
                    key=key_block,
                    **kwargs,
                )
            offset += n_blocks

        return tuple(outputs)

    def feature_metadata(
        self,
        x: Float[Array, "channels height width"],
        *args,
        endpoint: str,
        endpoint_options: dict,
    ) -> dict:
        """Describe token geometry for explicit feature extraction contracts."""

        del args
        if x.ndim != 3:
            raise ValueError(
                "VisionTransformer feature metadata requires one CHW image."
            )
        input_size = (int(x.shape[-2]), int(x.shape[-1]))
        raw_patch_size = self.patch_embed.patch_size
        patch_size = (
            (raw_patch_size, raw_patch_size)
            if isinstance(raw_patch_size, int)
            else (int(raw_patch_size[0]), int(raw_patch_size[1]))
        )
        grid_size = self.patch_embed.dynamic_feat_size(input_size)
        patch_padding = (
            grid_size[0] * patch_size[0] - input_size[0],
            grid_size[1] * patch_size[1] - input_size[1],
        )
        prefix_tokens = (("cls",) if self.cls_token is not None else ()) + tuple(
            f"register_{index}" for index in range(self.num_reg_tokens)
        )

        layer_indices: tuple[int, ...] = ()
        if endpoint == "intermediate_features":
            total = count_chunk_blocks(self.blocks)
            layer_indices = tuple(
                sorted(
                    intermediate_indices(
                        total,
                        indices=endpoint_options.get("indices"),
                        n_last_blocks=endpoint_options.get("n_last_blocks"),
                    )
                )
            )

        endpoint_normalization = "none"
        if endpoint == "forward_features" or endpoint_options.get("apply_norm", False):
            endpoint_normalization = "encoder_final_norm"

        return {
            "input_size": input_size,
            "patch_size": patch_size,
            "patch_padding": patch_padding,
            "grid_size": grid_size,
            "prefix_tokens": prefix_tokens,
            "tokens_include_prefix": endpoint != "forward_features",
            "layer_indices": layer_indices,
            "endpoint_normalization": endpoint_normalization,
            "positional_configuration": self._feature_position_configuration(),
        }

    def _feature_position_configuration(self) -> tuple[tuple[str, str], ...]:
        configuration = [
            (
                "global.type",
                "none"
                if self.global_pos_embed is None
                else type(self.global_pos_embed).__name__,
            ),
            (
                "local.type",
                "none"
                if self.local_pos_embed is None
                else type(self.local_pos_embed).__name__,
            ),
        ]
        if self.global_pos_embed is not None:
            configuration.extend(
                (
                    ("global.embed_size", str(self.global_pos_embed.embed_size)),
                    ("global.antialias", str(self.global_pos_embed.antialias)),
                )
            )
        if self.local_pos_embed is not None:
            patch_rope = self.local_pos_embed.patch_rope
            configuration.extend(
                (
                    ("local.strategy", str(patch_rope.strategy)),
                    ("local.normalize_coords", str(patch_rope.normalize_coords)),
                    ("local.shift_coords", str(patch_rope.shift_coords)),
                    ("local.jitter_coords", str(patch_rope.jitter_coords)),
                    ("local.rescale_coords", str(patch_rope.rescale_coords)),
                    ("local.dtype", str(patch_rope.dtype)),
                )
            )
        return tuple(configuration)

    def _prepare_tokens(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]],
        inference: Optional[bool],
    ):
        x = self.patch_embed(x)

        if mask is not None:
            assert self.mask_token is not None, (
                "To use masked forward, init the model with `use_mask_token=True`."
            )
            if self.dynamic_img_size:
                mask = rearrange(mask, "h w -> 1 h w")
                value = rearrange(self.mask_token, "1 c -> c 1 1")
            else:
                mask = rearrange(mask, "h w -> (h w) 1")
                value = self.mask_token
            x = jnp.where(mask, x, value.astype(x.dtype))

        H = W = self.embed_size
        if self.dynamic_img_size:
            _, H, W = x.shape

        if self.global_pos_embed is not None:
            x = self.global_pos_embed(
                x,
                cls_token=self.cls_token,
                reg_tokens=self.reg_tokens,
                dynamic_img_size=self.dynamic_img_size,
            )
        else:
            prefix = [t for t in (self.cls_token, self.reg_tokens) if t is not None]
            if self.dynamic_img_size:
                x = rearrange(x, "c h w -> (h w) c")
            x = jnp.concatenate([*prefix, x], axis=0) if prefix else x

        rotary = None
        if self.local_pos_embed is not None and inference:
            rotary = self.local_pos_embed.get_factors(
                H=H, W=W, inference=inference, key=key
            )
        return x, H, W, rotary

    def prepare_tokens(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: PRNGKeyArray,
        inference: bool,
    ) -> tuple[jax.Array, int, int, RotaryFactors | None]:
        """Prepare image tokens and spatial rotary factors for block execution.

        The token order is class, registers, patches. The returned rotary
        factors follow that same order and may be extended with query rows.
        """
        tokens, height, width, rotary = self._prepare_tokens(
            x, key=key, mask=None, inference=inference
        )
        if rotary is None and self.local_pos_embed is not None:
            rotary = self.local_pos_embed.get_factors(
                H=height, W=width, inference=inference, key=key
            )
        return tokens, height, width, rotary

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks in execution order."""
        return self._num_block_layers()

    def block_at(self, index: int) -> eqx.Module:
        """Return one logical transformer block, including chunked models."""
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"Transformer block index {index} is out of range.")
        for chunk in self.blocks:
            blocks = chunk.blocks or ()
            size = len(blocks)
            if index < size:
                return blocks[index]
            index -= size
        raise AssertionError("Block inventory disagrees with num_blocks.")

    def _run_blocks(
        self,
        prepared,
        *,
        key: PRNGKeyArray,
        inference: Optional[bool],
        token_transform: Callable | None = None,
        token_transform_key: PRNGKeyArray | None = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        """Run prepared tokens, optionally transforming them before each layer."""

        x, H, W, rotary = prepared
        key_pos, *block_subkeys = split_for_mode(
            key, len(self.blocks) + 1, inference=inference
        )
        num_transform_keys = max(self._num_block_layers(), 1)
        transform_subkeys = (
            (None,) * num_transform_keys
            if token_transform_key is None
            else jr.split(token_transform_key, num_transform_keys)
        )

        if not self.blocks and token_transform is not None:
            x, _ = token_transform(
                x,
                rotary,
                0,
                H,
                W,
                transform_subkeys[0],
                inference,
            )
            return x

        layer_index = 0
        for blk, key_block in zip(self.blocks, block_subkeys, strict=True):
            if self.local_pos_embed is not None and not inference:
                key_pos, key_rope = jr.split(key_pos, 2)
                rotary = self.local_pos_embed.get_factors(
                    H=H, W=W, inference=inference, key=key_rope
                )

            if token_transform is None:
                x = blk(
                    x,
                    rotary=rotary,
                    inference=inference,
                    key=key_block,
                    **kwargs,
                )
                continue

            blocks = blk.blocks
            num_blocks = 0 if blocks is None else len(blocks)
            key_down, *layer_subkeys = split_for_mode(
                key_block, num_blocks + 2, inference=inference
            )
            x = blk.posemb(x)
            if not blk.downsample_last and blk.downsample is not None:
                x = (
                    blk.downsample(x, inference=inference, key=key_down)
                    if blk.downsampler_needs_key
                    else blk.downsample(x)
                )
            if blocks is not None:
                for block, layer_key in zip(blocks, layer_subkeys, strict=False):
                    x, rotary = token_transform(
                        x,
                        rotary,
                        layer_index,
                        H,
                        W,
                        transform_subkeys[layer_index],
                        inference,
                    )
                    x = block(
                        x,
                        rotary=rotary,
                        inference=inference,
                        key=layer_key,
                        **kwargs,
                    )
                    layer_index += 1
            if blk.downsample_last and blk.downsample is not None:
                x = (
                    blk.downsample(x, inference=inference, key=key_down)
                    if blk.downsampler_needs_key
                    else blk.downsample(x)
                )

        return x

    def _num_block_layers(self) -> int:
        """Return the number of logical transformer layers across block chunks."""

        return count_chunk_blocks(self.blocks)

    def forward_features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """Process features and return intermediate representations.

        Args:
            x: Input image tensor
            inference: Whether to run stochastic layers in inference mode;
                True disables dropout and drop-path.
            key: PRNG key for random operations

        Returns:
            Dictionary containing:
                - x_norm_cls_token: Normalized class token
                - x_norm_reg_tokens: Normalized registration tokens
                - x_norm_patchtokens: Normalized patch tokens
                - x_prenorm: Pre-normalized features
        """
        x = self.features(x, inference=inference, key=key, **kwargs)
        x_norm = jax.vmap(self.norm)(x)
        cls_offset = 1 if self.cls_token is not None else 0
        reg_start = cls_offset
        reg_end = reg_start + self.num_reg_tokens

        return {
            "x_norm_cls_token": x_norm[0] if self.cls_token is not None else None,
            "x_norm_reg_tokens": x_norm[reg_start:reg_end],
            "x_norm_patchtokens": x_norm[reg_end:],
            "x_prenorm": x,
        }

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:  # noqa: F821
        """Process input image through the full network.

        Args:
            x: Input image tensor
            inference: Whether to run stochastic layers in inference mode;
                True disables dropout and drop-path.
            key: PRNG key for random operations

        Returns:
            Classification logits
        """
        x = self.features(x, inference=inference, key=key, **kwargs)
        x = jax.vmap(self.norm)(x)
        x = pool_sd(
            x,
            num_prefix_tokens=self.num_prefix_tokens,
            pool_type=self.global_pool,
            reduce_include_prefix=False,
        )

        x = self.head(x)

        return x


_VIT_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "num_classes": 1000,
    "reg_tokens": 0,
    "use_mask_token": False,
    "dynamic_img_size": False,
    "act_layer": "gelu",
}
_DINOV2_BASE_CFG: dict = {
    "img_size": 518,
    "in_channels": 3,
    "patch_size": 14,
    "num_classes": 0,
    "use_mask_token": True,
    "init_values": 1e-5,
    "eps": 1e-6,
    "dynamic_img_size": False,
    "act_layer": "exactgelu",
}
_DINOV3_LOCAL_ROPE_CFG: dict = {
    "strategy": "period",
    "base": 100.0,
    "normalize_coords": "separate",
    "rescale_coords": 2.0,
    "dtype": jnp.float32,
    "periods_dtype": jnp.float32,
}
_DINOV3_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": True,
    "use_global_pos_embed": False,
    "use_local_pos_embed": True,
    "local_pos_embed_config_patch": _DINOV3_LOCAL_ROPE_CFG,
    "reg_tokens": 4,
    "init_values": 1e-5,
    "eps": 1e-5,
    "dynamic_img_size": True,
    "act_layer": "exactgelu",
}
_LINGBOT_BASE_CFG: dict = {
    "img_size": 512,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": True,
    "use_global_pos_embed": False,
    "use_local_pos_embed": True,
    "local_pos_embed_config_patch": _DINOV3_LOCAL_ROPE_CFG,
    "reg_tokens": 4,
    "init_values": 1e-5,
    "eps": 1e-5,
    "dynamic_img_size": True,
    "act_layer": "exactgelu",
    "attn_layer": "maskedkeyattention",
    "global_pool": "token",
}
_EUPE_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": True,
    "use_global_pos_embed": False,
    "use_local_pos_embed": True,
    "local_pos_embed_config": {
        "strategy": "period",
        "base": 100.0,
        "normalize_coords": "separate",
        "rescale_coords": 2.0,
    },
    "reg_tokens": 4,
    "init_values": 1e-5,
    "eps": 1e-5,
    "dynamic_img_size": True,
    "act_layer": "exactgelu",
}
_SIGLIP2_BASE_CFG: dict = {
    "img_size": 384,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": False,
    "reg_tokens": 0,
    "class_token": False,
    "global_pos_embed_cls": False,
    "init_values": None,
    "eps": 1e-6,
    "dynamic_img_size": False,
    "act_layer": "gelu",
}
_TIPS_BASE_CFG: dict = {
    "in_channels": 3,
    "patch_size": 14,
    "num_classes": 0,
    "use_mask_token": True,
    "reg_tokens": 1,
    "init_values": 1e-5,
    "eps": 1e-6,
    "dynamic_img_size": False,
    "act_layer": "exactgelu",
}
_VIT5_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 1000,
    "reg_tokens": 4,
    "class_token": True,
    "global_pos_embed_cls": False,  # APE on patches only (not CLS/reg)
    "global_pos_embed_reg": False,  # APE on patches only (not CLS/reg)
    "local_pos_embed_reg": True,  # registers get their own RoPE via CompositeVisionRoPE
    "use_mask_token": False,
    "dynamic_img_size": False,
    "use_global_pos_embed": True,  # learned APE
    "use_local_pos_embed": True,  # RoPE
    "local_pos_embed_config_patch": {
        "strategy": "mode",
        "freqs_for": "lang",
        "theta": 10000,
        "pt_seq_len": 14,  # = 224 // 16
    },
    "local_pos_embed_config_reg": {
        "strategy": "mode",
        "freqs_for": "lang",
        "theta": 100,  # reg_theta in PyTorch
        "pt_seq_len": 2,  # = int(sqrt(4))
    },
    "act_layer": "gelu",
    "norm_layer": "rmsnorm",
    "eps": 1e-6,
    "qkv_bias": False,
    "qk_norm": True,
    "init_values": 1e-4,  # layer scale
}


_VIT_REGISTRY: dict[str, tuple[dict, dict]] = {
    # Standard ViT (Dosovitskiy et al. + DeiT-III Ti/S)
    "vit_tiny_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 192, "patch_size": 16, "num_heads": [3], "depths": [12]},
    ),
    "vit_tiny_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 192, "patch_size": 32, "num_heads": [3], "depths": [12]},
    ),
    "vit_small_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 384, "patch_size": 16, "num_heads": [6], "depths": [12]},
    ),
    "vit_small_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 384, "patch_size": 32, "num_heads": [6], "depths": [12]},
    ),
    "vit_base_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 768, "patch_size": 16, "num_heads": [12], "depths": [12]},
    ),
    "vit_base_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 768, "patch_size": 32, "num_heads": [12], "depths": [12]},
    ),
    "vit_large_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 1024, "patch_size": 16, "num_heads": [16], "depths": [24]},
    ),
    "vit_large_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 1024, "patch_size": 32, "num_heads": [16], "depths": [24]},
    ),
    "vit_huge_patch14_224": (
        _VIT_BASE_CFG,
        {"dim": 1280, "patch_size": 14, "num_heads": [16], "depths": [32]},
    ),
    "vit_huge_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 1280, "patch_size": 16, "num_heads": [16], "depths": [32]},
    ),
    # DINOv2
    "dinov2_vits14": (
        _DINOV2_BASE_CFG,
        {"dim": 384, "num_heads": [6], "depths": [12], "reg_tokens": 0},
    ),
    "dinov2_vits14_reg": (
        _DINOV2_BASE_CFG,
        {"dim": 384, "num_heads": [6], "depths": [12], "reg_tokens": 4},
    ),
    "dinov2_vitb14": (
        _DINOV2_BASE_CFG,
        {"dim": 768, "num_heads": [12], "depths": [12], "reg_tokens": 0},
    ),
    "dinov2_vitb14_reg": (
        _DINOV2_BASE_CFG,
        {"dim": 768, "num_heads": [12], "depths": [12], "reg_tokens": 4},
    ),
    "dinov2_vitl14": (
        _DINOV2_BASE_CFG,
        {"dim": 1024, "num_heads": [16], "depths": [24], "reg_tokens": 0},
    ),
    "dinov2_vitl14_reg": (
        _DINOV2_BASE_CFG,
        {"dim": 1024, "num_heads": [16], "depths": [24], "reg_tokens": 4},
    ),
    "dinov2_vitg14": (
        _DINOV2_BASE_CFG,
        {
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "reg_tokens": 0,
            "ffn_layer": "swiglufused",
        },
    ),
    "dinov2_vitg14_reg": (
        _DINOV2_BASE_CFG,
        {
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "reg_tokens": 4,
            "ffn_layer": "swiglufused",
        },
    ),
    # DINOv3 (LVD-1689M)
    "dinov3_vits16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {"dim": 384, "num_heads": 6, "depths": [12]},
    ),
    "dinov3_vits16plus_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 384,
            "num_heads": 6,
            "depths": [12],
            "mlp_ratio": 6.0,
            "ffn_layer": "swiglu",
        },
    ),
    "dinov3_vitb16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {"dim": 768, "num_heads": 12, "depths": [12]},
    ),
    "dinov3_vitl16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {"dim": 1024, "num_heads": 16, "depths": [24]},
    ),
    "dinov3_vith16plus_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 1280,
            "num_heads": 20,
            "depths": [32],
            "mlp_ratio": 6.0,
            "ffn_layer": "swiglu",
        },
    ),
    "dinov3_vit7b16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 4096,
            "num_heads": 32,
            "depths": [40],
            "mlp_ratio": 3.0,
            "untie_global_and_local_cls_norm": True,
            "ffn_layer": "swiglu",
            "ffn_kwargs": {"align_to": 64},
            "qkv_bias": False,
        },
    ),
    # DINOv3 (SAT-493M)
    "dinov3_vitl16_pretrain_sat493m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 1024,
            "num_heads": 16,
            "depths": [24],
            "untie_global_and_local_cls_norm": True,
        },
    ),
    "dinov3_vit7b16_pretrain_sat493m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 4096,
            "num_heads": 32,
            "depths": [40],
            "mlp_ratio": 3.0,
            "untie_global_and_local_cls_norm": True,
            "ffn_layer": "swiglu",
            "ffn_kwargs": {"align_to": 64},
            "qkv_bias": False,
        },
    ),
    # LingBot-Vision
    "lingbot_vits16": (
        _LINGBOT_BASE_CFG,
        {"dim": 384, "num_heads": 6, "depths": [12]},
    ),
    "lingbot_vitb16": (
        _LINGBOT_BASE_CFG,
        {"dim": 768, "num_heads": 12, "depths": [12]},
    ),
    "lingbot_vitl16": (
        _LINGBOT_BASE_CFG,
        {"dim": 1024, "num_heads": 16, "depths": [24]},
    ),
    "lingbot_vitg16": (
        _LINGBOT_BASE_CFG,
        {
            "dim": 1536,
            "num_heads": 24,
            "depths": [40],
            "ffn_layer": "swiglu",
            "qkv_bias": False,
        },
    ),
    # EUPE
    "eupe_vitt16": (
        _EUPE_BASE_CFG,
        {"dim": 192, "num_heads": 3, "depths": [12]},
    ),
    "eupe_vits16": (
        _EUPE_BASE_CFG,
        {"dim": 384, "num_heads": 6, "depths": [12]},
    ),
    "eupe_vitb16": (
        _EUPE_BASE_CFG,
        {"dim": 768, "num_heads": 12, "depths": [12]},
    ),
    # SigLIP2
    "siglip2_vitb16_224": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 224, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb16_256": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 256, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb16_384": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 384, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb16_512": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 512, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb32_256": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 256,
            "patch_size": 32,
            "dim": 768,
            "num_heads": [12],
            "depths": [12],
        },
    ),
    "siglip2_vitl16_256": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 256, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "siglip2_vitl16_384": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 384, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "siglip2_vitl16_512": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 512, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "siglip2_vitso400m14_224": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 224,
            "patch_size": 14,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m14_378": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 378,
            "patch_size": 14,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m16_256": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 256,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m16_384": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 384,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m16_512": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 512,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitgiantopt16_256": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 256, "dim": 1536, "num_heads": [16], "depths": [40]},
    ),
    "siglip2_vitgiantopt16_384": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 384, "dim": 1536, "num_heads": [16], "depths": [40]},
    ),
    # TIPS
    "tips_vits14_hr": (
        _TIPS_BASE_CFG,
        {"img_size": 448, "dim": 384, "num_heads": [6], "depths": [12]},
    ),
    "tips_vitb14_hr": (
        _TIPS_BASE_CFG,
        {"img_size": 448, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "tips_vitl14_hr": (
        _TIPS_BASE_CFG,
        {"img_size": 448, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "tips_vitso400m14_hr": (
        _TIPS_BASE_CFG,
        {
            "img_size": 448,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "tips_vitg14_lr": (
        _TIPS_BASE_CFG,
        {
            "img_size": 224,
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "ffn_layer": "swiglufused",
        },
    ),
    "tips_vitg14_hr": (
        _TIPS_BASE_CFG,
        {
            "img_size": 448,
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "ffn_layer": "swiglufused",
        },
    ),
    "vit5_small": (
        _VIT5_BASE_CFG,
        {"dim": 384, "num_heads": 6, "depths": [12]},
    ),
    "vit5_base": (
        _VIT5_BASE_CFG,
        {"dim": 768, "num_heads": 12, "depths": [12]},
    ),
    "vit5_large": (
        _VIT5_BASE_CFG,
        {"dim": 1024, "num_heads": 16, "depths": [24]},
    ),
    "vit5_xlarge": (
        _VIT5_BASE_CFG,
        {"dim": 1152, "num_heads": 16, "depths": [28]},
    ),
}


def _catalog_model_variants():
    """Derive the representative catalog entry from the ViT authority."""
    from equimo.catalog import (
        ModelInput,
        ModelProvenance,
        ModelVariant,
        PretrainedWeights,
    )

    variant = "dinov2_vits14_reg"
    base_cfg, variant_cfg = _VIT_REGISTRY[variant]
    cfg = base_cfg | variant_cfg
    return (
        ModelVariant(
            key=f"vision/{variant}",
            modality="vision",
            family="dinov2",
            variant=variant,
            model_registry_key="vit",
            constructor=f"{__name__}.{variant}",
            inputs=(
                ModelInput(
                    name="x",
                    shape=(cfg["in_channels"], cfg["img_size"], cfg["img_size"]),
                    axes=("channels", "height", "width"),
                    dtype="float32",
                    description=(
                        "One image; checkpoint-specific normalization is "
                        "caller-managed."
                    ),
                ),
            ),
            pretrained=PretrainedWeights(available=True, identifier=variant),
            provenance=ModelProvenance(
                conversion="models/torch_models.py",
                reference=(
                    "tests/data/reference_provenance.json#"
                    "dinov2_vits14_reg_reference.npz"
                ),
            ),
            notes=("Checkpoint availability does not trigger automatic downloading.",),
            field_status=(
                ("inputs", "complete"),
                ("pretrained", "complete"),
                ("provenance", "complete"),
                ("notes", "complete"),
            ),
        ),
    )


def _build_vit(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> VisionTransformer:
    """Construct a :class:`VisionTransformer` from the unified registry and
    optionally load pretrained weights.

    Args:
        variant: A key in :data:`_VIT_REGISTRY`.
        pretrained: If ``True``, download and deserialise the pretrained
            checkpoint from the default repository.
        inference_mode: Passed to :func:`equimo.serialization.load_weights` when
            *pretrained* is ``True``.  Defaults to ``True``.
        key: PRNG key for parameter initialisation.  Defaults to
            ``PRNGKey(42)`` when ``None``.
        **overrides: Extra keyword arguments merged into the model config,
            overriding stored defaults (e.g. ``num_classes=10``).

    Returns:
        A :class:`VisionTransformer` instance.

    Raises:
        KeyError: If *variant* is not found in the registry.
    """
    if pretrained and variant.startswith("lingbot_"):
        raise ValueError(
            "LingBot-Vision checkpoints require local conversion; construct with "
            "pretrained=False and load the converted archive with load_weights(path=...)."
        )
    return build_model_variant(
        VisionTransformer,
        _VIT_REGISTRY,
        variant,
        pretrained=pretrained,
        inference_mode=inference_mode,
        key=key,
        **overrides,
    )


def vit_tiny_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-Ti/16 — 192-dim, 3 heads, 12 blocks, patch 16, 224*224."""
    return _build_vit("vit_tiny_patch16_224", **kwargs)


def vit_tiny_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-Ti/32 — 192-dim, 3 heads, 12 blocks, patch 32, 224*224."""
    return _build_vit("vit_tiny_patch32_224", **kwargs)


def vit_small_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-S/16 — 384-dim, 6 heads, 12 blocks, patch 16, 224*224."""
    return _build_vit("vit_small_patch16_224", **kwargs)


def vit_small_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-S/32 — 384-dim, 6 heads, 12 blocks, patch 32, 224*224."""
    return _build_vit("vit_small_patch32_224", **kwargs)


def vit_base_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-B/16 — 768-dim, 12 heads, 12 blocks, patch 16, 224*224."""
    return _build_vit("vit_base_patch16_224", **kwargs)


def vit_base_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-B/32 — 768-dim, 12 heads, 12 blocks, patch 32, 224*224."""
    return _build_vit("vit_base_patch32_224", **kwargs)


def vit_large_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-L/16 — 1024-dim, 16 heads, 24 blocks, patch 16, 224*224."""
    return _build_vit("vit_large_patch16_224", **kwargs)


def vit_large_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-L/32 — 1024-dim, 16 heads, 24 blocks, patch 32, 224*224."""
    return _build_vit("vit_large_patch32_224", **kwargs)


def vit_huge_patch14_224(**kwargs) -> VisionTransformer:
    """ViT-H/14 — 1280-dim, 16 heads, 32 blocks, patch 14, 224*224."""
    return _build_vit("vit_huge_patch14_224", **kwargs)


def vit_huge_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-H/16 — 1280-dim, 16 heads, 32 blocks, patch 16, 224*224."""
    return _build_vit("vit_huge_patch16_224", **kwargs)


def dinov2_vits14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-S/14 (no register tokens)."""
    return _build_vit("dinov2_vits14", pretrained=pretrained, **kwargs)


def dinov2_vits14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-S/14 with 4 register tokens."""
    return _build_vit("dinov2_vits14_reg", pretrained=pretrained, **kwargs)


def dinov2_vitb14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-B/14 (no register tokens)."""
    return _build_vit("dinov2_vitb14", pretrained=pretrained, **kwargs)


def dinov2_vitb14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-B/14 with 4 register tokens."""
    return _build_vit("dinov2_vitb14_reg", pretrained=pretrained, **kwargs)


def dinov2_vitl14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-L/14 (no register tokens)."""
    return _build_vit("dinov2_vitl14", pretrained=pretrained, **kwargs)


def dinov2_vitl14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-L/14 with 4 register tokens."""
    return _build_vit("dinov2_vitl14_reg", pretrained=pretrained, **kwargs)


def dinov2_vitg14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-g/14 (no register tokens)."""
    return _build_vit("dinov2_vitg14", pretrained=pretrained, **kwargs)


def dinov2_vitg14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-g/14 with 4 register tokens."""
    return _build_vit("dinov2_vitg14_reg", pretrained=pretrained, **kwargs)


def dinov3_vits16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-S/16 (LVD-1689M)."""
    return _build_vit(
        "dinov3_vits16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vits16plus_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-S/16+ with SwiGLU FFN (LVD-1689M)."""
    return _build_vit(
        "dinov3_vits16plus_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vitb16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-B/16 (LVD-1689M)."""
    return _build_vit(
        "dinov3_vitb16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vitl16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-L/16 (LVD-1689M)."""
    return _build_vit(
        "dinov3_vitl16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vith16plus_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-H/16+ with SwiGLU FFN (LVD-1689M)."""
    return _build_vit(
        "dinov3_vith16plus_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vit7b16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-7B/16 with SwiGLU FFN (LVD-1689M)."""
    return _build_vit(
        "dinov3_vit7b16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vitl16_pretrain_sat493m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-L/16 (SAT-493M)."""
    return _build_vit("dinov3_vitl16_pretrain_sat493m", pretrained=pretrained, **kwargs)


def dinov3_vit7b16_pretrain_sat493m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-7B/16 with SwiGLU FFN (SAT-493M)."""
    return _build_vit(
        "dinov3_vit7b16_pretrain_sat493m", pretrained=pretrained, **kwargs
    )


def lingbot_vits16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """LingBot-Vision Small backbone with 16-pixel patches."""
    return _build_vit("lingbot_vits16", pretrained=pretrained, **kwargs)


def lingbot_vitb16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """LingBot-Vision Base backbone with 16-pixel patches."""
    return _build_vit("lingbot_vitb16", pretrained=pretrained, **kwargs)


def lingbot_vitl16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """LingBot-Vision Large backbone with 16-pixel patches."""
    return _build_vit("lingbot_vitl16", pretrained=pretrained, **kwargs)


def lingbot_vitg16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """LingBot-Vision Giant backbone with SwiGLU feed-forward layers."""
    return _build_vit("lingbot_vitg16", pretrained=pretrained, **kwargs)


def eupe_vitt16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """EUPE ViT-T/16 (vit_tiny)."""
    return _build_vit("eupe_vitt16", pretrained=pretrained, **kwargs)


def eupe_vits16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """EUPE ViT-S/16 (vit_small)."""
    return _build_vit("eupe_vits16", pretrained=pretrained, **kwargs)


def eupe_vitb16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """EUPE ViT-B/16 (vit_base)."""
    return _build_vit("eupe_vitb16", pretrained=pretrained, **kwargs)


def siglip2_vitb16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 224*224."""
    return _build_vit("siglip2_vitb16_224", pretrained=pretrained, **kwargs)


def siglip2_vitb16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 256*256."""
    return _build_vit("siglip2_vitb16_256", pretrained=pretrained, **kwargs)


def siglip2_vitb16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 384*384."""
    return _build_vit("siglip2_vitb16_384", pretrained=pretrained, **kwargs)


def siglip2_vitb16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 512*512."""
    return _build_vit("siglip2_vitb16_512", pretrained=pretrained, **kwargs)


def siglip2_vitb32_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/32 at 256*256."""
    return _build_vit("siglip2_vitb32_256", pretrained=pretrained, **kwargs)


def siglip2_vitl16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-L/16 at 256*256."""
    return _build_vit("siglip2_vitl16_256", pretrained=pretrained, **kwargs)


def siglip2_vitl16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-L/16 at 384*384."""
    return _build_vit("siglip2_vitl16_384", pretrained=pretrained, **kwargs)


def siglip2_vitl16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-L/16 at 512*512."""
    return _build_vit("siglip2_vitl16_512", pretrained=pretrained, **kwargs)


def siglip2_vitso400m14_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/14 at 224*224."""
    return _build_vit("siglip2_vitso400m14_224", pretrained=pretrained, **kwargs)


def siglip2_vitso400m14_378(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/14 at 378*378."""
    return _build_vit("siglip2_vitso400m14_378", pretrained=pretrained, **kwargs)


def siglip2_vitso400m16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/16 at 256*256."""
    return _build_vit("siglip2_vitso400m16_256", pretrained=pretrained, **kwargs)


def siglip2_vitso400m16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/16 at 384*384."""
    return _build_vit("siglip2_vitso400m16_384", pretrained=pretrained, **kwargs)


def siglip2_vitso400m16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/16 at 512*512."""
    return _build_vit("siglip2_vitso400m16_512", pretrained=pretrained, **kwargs)


def siglip2_vitgiantopt16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-giantopt/16 at 256*256."""
    return _build_vit("siglip2_vitgiantopt16_256", pretrained=pretrained, **kwargs)


def siglip2_vitgiantopt16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-giantopt/16 at 384*384."""
    return _build_vit("siglip2_vitgiantopt16_384", pretrained=pretrained, **kwargs)


def tips_vits14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-S/14 high-res (448*448)."""
    return _build_vit("tips_vits14_hr", pretrained=pretrained, **kwargs)


def tips_vitb14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-B/14 high-res (448*448)."""
    return _build_vit("tips_vitb14_hr", pretrained=pretrained, **kwargs)


def tips_vitl14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-L/14 high-res (448*448)."""
    return _build_vit("tips_vitl14_hr", pretrained=pretrained, **kwargs)


def tips_vitso400m14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-SO400M/14 high-res (448*448)."""
    return _build_vit("tips_vitso400m14_hr", pretrained=pretrained, **kwargs)


def tips_vitg14_lr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-g/14 low-res (224*224)."""
    return _build_vit("tips_vitg14_lr", pretrained=pretrained, **kwargs)


def tips_vitg14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-g/14 high-res (448*448)."""
    return _build_vit("tips_vitg14_hr", pretrained=pretrained, **kwargs)


def vit5_small(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT5-S/16 — 384-dim, 6 heads, 12 blocks, 4 registers, RoPE + APE."""
    return _build_vit("vit5_small", pretrained=pretrained, **kwargs)


def vit5_base(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT5-B/16 — 768-dim, 12 heads, 12 blocks, 4 registers, RoPE + APE."""
    return _build_vit("vit5_base", pretrained=pretrained, **kwargs)


def vit5_large(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT5-L/16 — 1024-dim, 16 heads, 24 blocks, 4 registers, RoPE + APE."""
    return _build_vit("vit5_large", pretrained=pretrained, **kwargs)


def vit5_xlarge(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT5-XL/16 — 1152-dim, 16 heads, 28 blocks, 4 registers, RoPE + APE."""
    return _build_vit("vit5_xlarge", pretrained=pretrained, **kwargs)
