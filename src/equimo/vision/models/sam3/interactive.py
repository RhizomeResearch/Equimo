"""SAM 2-style prompt encoder and mask decoders of SAM 3.1."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from equimo.core.layers.attention import CrossAttention
from equimo.core.layers.norm import LayerNorm2d
from equimo.core.parameters import buffer
from equimo.vision.models.sam3.common import (
    MLP,
    PositionedAttention,
    flatten_map,
    layer_norm,
    unflatten_map,
)
from equimo.vision.models.sam3.encoder import ImageFeatures, _upsample_twofold

_STABILITY_OFFSET = 0.05
_STABILITY_THRESHOLD = 0.98
_NO_OBJECT_LOGIT = -1024.0
_INTERACTIVE_MASK_TOKENS = 4
_MASKS_PER_SLOT = 3
_MAX_INTERACTIVE_POINTS = 16
_MASK_IN_CHANNELS = 16


class InteractivePrompt(eqx.Module):
    """Point, box and mask prompt with fixed shapes.

    Points and boxes are in input pixels. Point labels are 1 (positive), 0
    (negative) or -1 (padding). The box is ``(x1, y1, x2, y2)``.
    """

    points: Float[Array, "points 2"]
    point_labels: Int[Array, " points"]
    box: Float[Array, " 4"]
    has_box: Bool[Array, ""]
    mask_input: Float[Array, "mask_height mask_width"]
    has_mask: Bool[Array, ""]

    @classmethod
    def create(
        cls,
        *,
        points: jax.Array | None = None,
        point_labels: jax.Array | None = None,
        box: jax.Array | None = None,
        mask_input: jax.Array | None = None,
        mask_size: int = 288,
        max_points: int = _MAX_INTERACTIVE_POINTS,
    ) -> InteractivePrompt:
        """Pad points to ``max_points``; absent prompts are marked as such."""
        if points is None:
            points = jnp.zeros((0, 2), jnp.float32)
            point_labels = jnp.zeros((0,), jnp.int32)
        assert point_labels is not None
        missing = max_points - points.shape[0]
        if missing < 0:
            raise ValueError(
                f"Got {points.shape[0]} points, more than the maximum {max_points}."
            )
        points = jnp.pad(points, ((0, missing), (0, 0)))
        point_labels = jnp.pad(
            jnp.asarray(point_labels, jnp.int32), (0, missing), constant_values=-1
        )
        has_box = box is not None
        has_mask = mask_input is not None
        return cls(
            points=points,
            point_labels=point_labels,
            box=jnp.zeros(4, jnp.float32) if box is None else box,
            has_box=jnp.asarray(has_box),
            mask_input=(
                jnp.zeros((mask_size, mask_size), jnp.float32)
                if mask_input is None
                else mask_input
            ),
            has_mask=jnp.asarray(has_mask),
        )


class InteractiveOutput(eqx.Module):
    """Four candidate masks with their scores and the selected index."""

    mask_logits: Float[Array, "masks height width"]
    iou: Float[Array, " masks"]
    object_score_logit: Float[Array, ""]
    selected: Int[Array, ""]


class MultiplexOutput(eqx.Module):
    """Per-slot outputs of the multiplex decoder for one bucket."""

    mask_logits: Float[Array, "slots masks height width"]
    iou: Float[Array, "slots masks"]
    object_score_logits: Float[Array, " slots"]
    mask_tokens: Float[Array, "slots masks dim"]


class PromptEncoder(eqx.Module):
    """Encode points, boxes and low-resolution masks as decoder prompts."""

    fourier_matrix: Float[Array, "2 fourier"] = buffer()
    point_embeddings: Float[Array, "4 dim"]
    not_a_point: Float[Array, " dim"]
    no_mask_embed: Float[Array, " dim"]
    mask_convs: tuple[eqx.nn.Conv2d, eqx.nn.Conv2d, eqx.nn.Conv2d]
    mask_norms: tuple[LayerNorm2d, LayerNorm2d]
    grid: int = eqx.field(static=True)
    image_size: int = eqx.field(static=True)

    def __init__(self, dim: int, grid: int, image_size: int, *, key: PRNGKeyArray):
        key_fourier, key_points, key_pad, key_mask, key_c1, key_c2, key_c3 = jr.split(
            key, 7
        )
        self.fourier_matrix = jr.normal(key_fourier, (2, dim // 2))
        self.point_embeddings = jr.normal(key_points, (4, dim))
        self.not_a_point = jr.normal(key_pad, (dim,))
        self.no_mask_embed = jr.normal(key_mask, (dim,))
        half = _MASK_IN_CHANNELS // 4
        self.mask_convs = (
            eqx.nn.Conv2d(1, half, 2, stride=2, key=key_c1),
            eqx.nn.Conv2d(half, _MASK_IN_CHANNELS, 2, stride=2, key=key_c2),
            eqx.nn.Conv2d(_MASK_IN_CHANNELS, dim, 1, key=key_c3),
        )
        self.mask_norms = (LayerNorm2d(half), LayerNorm2d(_MASK_IN_CHANNELS))
        self.grid = grid
        self.image_size = image_size

    def _fourier(self, coords: Float[Array, "... 2"]) -> Float[Array, "... dim"]:
        matrix = jax.lax.stop_gradient(self.fourier_matrix).astype(jnp.float32)
        angles = 2.0 * jnp.pi * (2.0 * coords.astype(jnp.float32) - 1.0) @ matrix
        return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)

    def dense_position_encoding(self) -> Float[Array, "dim grid grid"]:
        """Fourier encoding of the feature grid's cell centers."""
        centers = (jnp.arange(self.grid, dtype=jnp.float32) + 0.5) / self.grid
        ys, xs = jnp.meshgrid(centers, centers, indexing="ij")
        encoded = self._fourier(jnp.stack((xs, ys), axis=-1))
        return jnp.moveaxis(encoded, -1, 0).astype(self.no_mask_embed.dtype)

    def _downscale_mask(
        self, mask: Float[Array, "mask_height mask_width"]
    ) -> Float[Array, "dim grid grid"]:
        x = mask[None].astype(self.no_mask_embed.dtype)
        for conv, norm in zip(self.mask_convs[:2], self.mask_norms):
            x = jax.nn.gelu(norm(conv(x)), approximate=False)
        return self.mask_convs[2](x)

    def empty(
        self,
    ) -> tuple[Float[Array, "1 dim"], Float[Array, "dim grid grid"]]:
        """Sparse and dense prompts of a decoder call without any user prompt."""
        dense = jnp.broadcast_to(
            self.no_mask_embed[:, None, None],
            (self.no_mask_embed.shape[0], self.grid, self.grid),
        )
        return self.not_a_point[None], dense

    def __call__(
        self, prompt: InteractivePrompt
    ) -> tuple[
        Float[Array, "tokens dim"],
        Bool[Array, " tokens"],
        Float[Array, "dim grid grid"],
    ]:
        """Return sparse tokens, the mask of tokens in use, and the dense prompt.

        The token layout is fixed: the points, the two box corners, then the
        padding point that is added when no box is given. Tokens out of use are
        masked from attention, so they have no effect on the output.
        """
        dtype = self.no_mask_embed.dtype
        corners = prompt.box.reshape(2, 2)
        coords = jnp.concatenate(
            (prompt.points, corners, jnp.zeros((1, 2), prompt.points.dtype)), axis=0
        )
        labels = jnp.concatenate(
            (prompt.point_labels, jnp.asarray([2, 3, -1], prompt.point_labels.dtype))
        )
        valid = jnp.concatenate(
            (
                prompt.point_labels != -1,
                jnp.stack((prompt.has_box, prompt.has_box, ~prompt.has_box)),
            )
        )
        position = self._fourier((coords + 0.5) / self.image_size).astype(dtype)
        learned = self.point_embeddings[jnp.maximum(labels, 0)]
        tokens = jnp.where(
            (labels == -1)[:, None], self.not_a_point, position + learned
        ).astype(dtype)

        no_mask = jnp.broadcast_to(
            self.no_mask_embed[:, None, None],
            (self.no_mask_embed.shape[0], self.grid, self.grid),
        )
        dense = jnp.where(
            prompt.has_mask, self._downscale_mask(prompt.mask_input), no_mask
        )
        return tokens, valid, dense


class TwoWayBlock(eqx.Module):
    """Post-norm block updating prompt tokens from the image and back."""

    attn: CrossAttention
    norm1: eqx.nn.LayerNorm
    cross_attn: tuple[PositionedAttention, PositionedAttention]
    norm2: eqx.nn.LayerNorm
    mlp: MLP
    norm3: eqx.nn.LayerNorm
    norm4: eqx.nn.LayerNorm

    def __init__(
        self, dim: int, num_heads: int, ffn_features: int, *, key: PRNGKeyArray
    ):
        key_attn, key_to_image, key_to_token, key_mlp = jr.split(key, 4)
        self.attn = CrossAttention(dim, num_heads, key=key_attn)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.cross_attn = (
            PositionedAttention(
                dim, num_heads, internal_dim=dim // 2, key=key_to_image
            ),
            PositionedAttention(
                dim, num_heads, internal_dim=dim // 2, key=key_to_token
            ),
        )
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = MLP(dim, ffn_features, dim, 2, key=key_mlp)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.norm4 = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        tokens: Float[Array, "tokens dim"],
        image: Float[Array, "keys dim"],
        image_pos: Float[Array, "keys dim"],
        valid: Bool[Array, " tokens"],
    ) -> tuple[Float[Array, "tokens dim"], Float[Array, "keys dim"]]:
        tokens = layer_norm(
            self.norm1, tokens + self.attn(tokens, tokens, key_mask=valid)
        )
        token_to_image, image_to_token = self.cross_attn
        tokens = layer_norm(
            self.norm2,
            tokens + token_to_image(tokens, image + image_pos, values=image),
        )
        tokens = layer_norm(self.norm3, tokens + self.mlp(tokens))
        attended = image_to_token(image + image_pos, tokens, key_mask=valid)
        return tokens, layer_norm(self.norm4, image + attended)


class TwoWayTransformer(eqx.Module):
    """Transformer shared by the interactive and multiplex mask decoders."""

    blocks: tuple[TwoWayBlock, ...]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        self.blocks = tuple(
            TwoWayBlock(dim, num_heads, ffn_features, key=block_key)
            for block_key in jr.split(key, depth)
        )
        self.norm = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        tokens: Float[Array, "tokens dim"],
        image: Float[Array, "keys dim"],
        image_pos: Float[Array, "keys dim"],
        valid: Bool[Array, " tokens"],
    ) -> tuple[Float[Array, "tokens dim"], Float[Array, "keys dim"]]:
        for block in self.blocks:
            tokens, image = block(tokens, image, image_pos, valid)
        return layer_norm(self.norm, tokens), image


class MaskUpsampler(eqx.Module):
    """Upsample decoder features fourfold, fusing the two high-resolution levels."""

    conv_s0: eqx.nn.Conv2d
    conv_s1: eqx.nn.Conv2d
    deconvs: tuple[eqx.nn.ConvTranspose2d, eqx.nn.ConvTranspose2d]
    norms: tuple[LayerNorm2d, LayerNorm2d]

    def __init__(self, dim: int, *, key: PRNGKeyArray):
        key_s0, key_s1, key_d0, key_d1 = jr.split(key, 4)
        self.conv_s0 = eqx.nn.Conv2d(dim, dim // 8, 1, key=key_s0)
        self.conv_s1 = eqx.nn.Conv2d(dim, dim // 4, 1, key=key_s1)
        self.deconvs = (
            eqx.nn.ConvTranspose2d(dim, dim // 4, 2, stride=2, key=key_d0),
            eqx.nn.ConvTranspose2d(dim // 4, dim // 8, 2, stride=2, key=key_d1),
        )
        self.norms = (LayerNorm2d(dim // 4), LayerNorm2d(dim // 8))

    def __call__(
        self,
        image: Float[Array, "dim grid grid"],
        high_res: tuple[jax.Array, jax.Array],
    ) -> Float[Array, "pixel height width"]:
        high, middle = high_res
        x = _upsample_twofold(self.deconvs[0], image) + self.conv_s1(middle)
        x = jax.nn.gelu(self.norms[0](x), approximate=False)
        x = _upsample_twofold(self.deconvs[1], x) + self.conv_s0(high)
        return jax.nn.gelu(self.norms[1](x), approximate=False)


class MaskDecoder(eqx.Module):
    """SAM 2-style decoder with four mask tokens, IoU and object scores."""

    transformer: TwoWayTransformer
    upsampler: MaskUpsampler
    mask_tokens: Float[Array, "masks dim"]
    iou_token: Float[Array, " dim"]
    object_token: Float[Array, " dim"]
    mask_embed: MLP
    iou_head: MLP
    object_head: MLP

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        (
            key_transformer,
            key_upsampler,
            key_masks,
            key_iou,
            key_object,
            key_embed,
            key_iou_head,
            key_object_head,
        ) = jr.split(key, 8)
        self.transformer = TwoWayTransformer(
            dim, num_heads, ffn_features, depth, key=key_transformer
        )
        self.upsampler = MaskUpsampler(dim, key=key_upsampler)
        self.mask_tokens = jr.normal(key_masks, (_INTERACTIVE_MASK_TOKENS, dim))
        self.iou_token = jr.normal(key_iou, (dim,))
        self.object_token = jr.normal(key_object, (dim,))
        self.mask_embed = MLP(dim, dim, dim // 8, 3, key=key_embed)
        self.iou_head = MLP(dim, dim, _INTERACTIVE_MASK_TOKENS, 3, key=key_iou_head)
        self.object_head = MLP(dim, dim, 1, 3, key=key_object_head)

    def __call__(
        self,
        image: Float[Array, "dim grid grid"],
        image_pos: Float[Array, "dim grid grid"],
        sparse: Float[Array, "tokens dim"],
        sparse_valid: Bool[Array, " tokens"],
        dense: Float[Array, "dim grid grid"],
        high_res: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return mask logits (float32), predicted IoU and the object score logit."""
        dtype = image.dtype
        grid = image.shape[-1]
        tokens = jnp.concatenate(
            (
                self.iou_token[None],
                self.object_token[None],
                self.mask_tokens,
                sparse,
            ),
            axis=0,
        ).astype(dtype)
        valid = jnp.concatenate(
            (jnp.ones(2 + _INTERACTIVE_MASK_TOKENS, bool), sparse_valid)
        )
        tokens, attended = self.transformer(
            tokens, flatten_map(image + dense), flatten_map(image_pos), valid
        )
        pixels = self.upsampler(unflatten_map(attended, grid), high_res)
        masks = jnp.einsum(
            "qc,chw->qhw",
            self.mask_embed(tokens[2 : 2 + _INTERACTIVE_MASK_TOKENS]).astype(
                jnp.float32
            ),
            pixels.astype(jnp.float32),
        )
        return masks, self.iou_head(tokens[0]), self.object_head(tokens[1])[0]


class MultiplexMaskDecoder(eqx.Module):
    """Decode several objects in one pass, with five tokens per slot.

    Each slot holds three mask tokens, an IoU token and an object-score token.
    A slot's tokens carry that slot's valid or invalid embedding, which is how
    the shared decoder keeps objects apart.
    """

    transformer: TwoWayTransformer
    upsampler: MaskUpsampler
    mask_tokens: Float[Array, "masks dim"]
    iou_token: Float[Array, " dim"]
    object_token: Float[Array, " dim"]
    valid_embed: Float[Array, "slots dim"]
    invalid_embed: Float[Array, "slots dim"]
    mask_embed: MLP
    iou_head: MLP
    object_head: MLP
    num_slots: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        depth: int,
        num_slots: int,
        *,
        key: PRNGKeyArray,
    ):
        (
            key_transformer,
            key_upsampler,
            key_masks,
            key_iou,
            key_object,
            key_valid,
            key_invalid,
            key_embed,
            key_iou_head,
            key_object_head,
        ) = jr.split(key, 10)
        self.transformer = TwoWayTransformer(
            dim, num_heads, ffn_features, depth, key=key_transformer
        )
        self.upsampler = MaskUpsampler(dim, key=key_upsampler)
        self.mask_tokens = jr.normal(key_masks, (_MASKS_PER_SLOT, dim))
        self.iou_token = jr.normal(key_iou, (dim,))
        self.object_token = jr.normal(key_object, (dim,))
        self.valid_embed = jr.normal(key_valid, (num_slots, dim))
        self.invalid_embed = jr.normal(key_invalid, (num_slots, dim))
        self.mask_embed = MLP(dim, dim, dim // 8, 3, key=key_embed)
        self.iou_head = MLP(dim, dim, _MASKS_PER_SLOT, 3, key=key_iou_head)
        self.object_head = MLP(dim, dim, 1, 3, key=key_object_head)
        self.num_slots = num_slots

    def __call__(
        self,
        image: Float[Array, "dim grid grid"],
        image_pos: Float[Array, "dim grid grid"],
        sparse: Float[Array, "tokens dim"],
        dense: Float[Array, "dim grid grid"],
        high_res: tuple[jax.Array, jax.Array],
        slot_valid: Bool[Array, " slots"],
    ) -> MultiplexOutput:
        dtype = image.dtype
        grid = image.shape[-1]
        per_slot = _MASKS_PER_SLOT + 2
        slot_tokens = jnp.concatenate(
            (self.mask_tokens, self.iou_token[None], self.object_token[None]), axis=0
        )
        slot_embed = jnp.where(
            slot_valid[:, None], self.valid_embed, self.invalid_embed
        )
        tokens = (slot_tokens[None] + slot_embed[:, None]).reshape(
            self.num_slots * per_slot, slot_tokens.shape[-1]
        )
        tokens = jnp.concatenate((tokens, sparse), axis=0).astype(dtype)
        valid = jnp.ones(tokens.shape[0], bool)
        tokens, attended = self.transformer(
            tokens, flatten_map(image + dense), flatten_map(image_pos), valid
        )
        pixels = self.upsampler(unflatten_map(attended, grid), high_res)

        slots = tokens[: self.num_slots * per_slot].reshape(
            self.num_slots, per_slot, tokens.shape[-1]
        )
        mask_tokens = slots[:, :_MASKS_PER_SLOT]
        object_scores = self.object_head(slots[:, -1])[:, 0]
        masks = jnp.einsum(
            "smc,chw->smhw",
            self.mask_embed(mask_tokens).astype(jnp.float32),
            pixels.astype(jnp.float32),
        )
        masks = jnp.where(
            (object_scores > 0)[:, None, None, None], masks, _NO_OBJECT_LOGIT
        )
        return MultiplexOutput(
            mask_logits=masks,
            iou=self.iou_head(slots[:, _MASKS_PER_SLOT]),
            object_score_logits=object_scores,
            mask_tokens=mask_tokens,
        )


def _select_mask(
    mask_logits: Float[Array, "masks height width"],
    iou: Float[Array, " masks"],
    num_labels: Int[Array, ""],
) -> Int[Array, ""]:
    """Pick the output mask from the number of point labels.

    With at most one label the best multimask output is selected. With more, the
    single-mask output is selected unless it is unstable.
    """
    best_multimask = jnp.argmax(iou[1:].astype(jnp.float32)) + 1
    single = mask_logits[0]
    above = jnp.sum(single > _STABILITY_OFFSET, axis=(-2, -1))
    outer = jnp.sum(single > -_STABILITY_OFFSET, axis=(-2, -1))
    stability = jnp.where(outer > 0, above / jnp.maximum(outer, 1), 1.0)
    use_single = (num_labels > 1) & (stability >= _STABILITY_THRESHOLD)
    return jnp.where(use_single, 0, best_multimask)


class InteractiveHead(eqx.Module):
    """Segment one object in one image from points, a box or a mask."""

    prompt_encoder: PromptEncoder
    mask_decoder: MaskDecoder
    no_memory_embed: Float[Array, " dim"]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        depth: int,
        grid: int,
        image_size: int,
        *,
        key: PRNGKeyArray,
    ):
        key_prompt, key_decoder, key_memory = jr.split(key, 3)
        self.prompt_encoder = PromptEncoder(dim, grid, image_size, key=key_prompt)
        self.mask_decoder = MaskDecoder(
            dim, num_heads, ffn_features, depth, key=key_decoder
        )
        self.no_memory_embed = jr.normal(key_memory, (dim,))

    def __call__(
        self, features: ImageFeatures, prompt: InteractivePrompt
    ) -> InteractiveOutput:
        high, middle, low = features.interactive
        image = low + self.no_memory_embed[:, None, None].astype(low.dtype)
        sparse, valid, dense = self.prompt_encoder(prompt)
        masks, iou, object_score = self.mask_decoder(
            image,
            self.prompt_encoder.dense_position_encoding(),
            sparse,
            valid,
            dense,
            (high, middle),
        )
        num_labels = jnp.sum(prompt.point_labels != -1) + 2 * prompt.has_box
        return InteractiveOutput(
            mask_logits=masks,
            iou=iou,
            object_score_logit=object_score,
            selected=_select_mask(masks, iou, num_labels),
        )


__all__ = ["InteractiveOutput", "InteractivePrompt"]
