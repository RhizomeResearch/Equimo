"""SAM 2-style prompt encoder and mask decoders of SAM 3.1."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from equimo.core.layers.attention import CrossAttention
from equimo.core.layers.ffn import LinearStack, Mlp
from equimo.core.layers.norm import LayerNorm2d, layer_norm
from equimo.core.parameters import buffer
from equimo.vision.layers.convolution import upsample_twofold
from equimo.vision.layers.posemb import dense_fourier_encoding, fourier_encoding
from equimo.vision.models.sam3.encoder import ImageFeatures

_STABILITY_OFFSET = 0.05
_STABILITY_THRESHOLD = 0.98
_NO_OBJECT_LOGIT = -1024.0
_INTERACTIVE_MASK_TOKENS = 4
_MASKS_PER_SLOT = 3
_MAX_INTERACTIVE_POINTS = 16
_SOURCE_PIXELS_IMAGE_SIZE = 1008
_MASK_IN_CHANNELS = 16


class InteractivePrompt(eqx.Module):
    """Point, box and mask prompt with fixed shapes.

    Points and boxes are in input pixels, or in the source image's pixels when
    ``source_size`` is given to :meth:`create`. Point labels are 1 (positive), 0
    (negative) or -1 (padding), and default to positive. The box is
    ``(x1, y1, x2, y2)``.
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
        source_size: tuple[int, int] | None = None,
        image_size: int = _SOURCE_PIXELS_IMAGE_SIZE,
    ) -> InteractivePrompt:
        """Pad points to ``max_points``; absent prompts are marked as such.

        With ``source_size`` ``(height, width)``, points and the box are scaled
        from those pixels to the model's ``image_size``.
        """
        if points is None:
            points = jnp.zeros((0, 2), jnp.float32)
            point_labels = jnp.zeros((0,), jnp.int32)
        elif point_labels is None:
            point_labels = jnp.ones(points.shape[0], jnp.int32)
        point_labels = jnp.asarray(point_labels, jnp.int32)
        point_labels = eqx.error_if(
            point_labels,
            jnp.any((point_labels < -1) | (point_labels > 1)),
            "Point labels must be 1 (positive), 0 (negative) or -1 (padding); "
            "pass a box through `box`.",
        )
        if source_size is not None:
            height, width = source_size
            scale = jnp.asarray([image_size / width, image_size / height], jnp.float32)
            points = jnp.asarray(points, jnp.float32) * scale
            if box is not None:
                box = jnp.asarray(box, jnp.float32) * jnp.tile(scale, 2)
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


class ImagePositionEncoding(eqx.Module):
    """Fourier position encoding of the tracker's feature grid."""

    fourier_matrix: Float[Array, "2 fourier"] = buffer()
    grid: int = eqx.field(static=True)
    image_size: int = eqx.field(static=True)

    def __init__(self, dim: int, grid: int, image_size: int, *, key: PRNGKeyArray):
        self.fourier_matrix = jr.normal(key, (2, dim // 2))
        self.grid = grid
        self.image_size = image_size

    def __call__(self, dtype: jnp.dtype) -> Float[Array, "dim grid grid"]:
        return dense_fourier_encoding(self.fourier_matrix, self.grid, dtype)


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

    def dense_position_encoding(self) -> Float[Array, "dim grid grid"]:
        """Fourier encoding of the feature grid's cell centers."""
        return dense_fourier_encoding(
            self.fourier_matrix, self.grid, self.no_mask_embed.dtype
        )

    def _downscale_mask(
        self, mask: Float[Array, "mask_height mask_width"]
    ) -> Float[Array, "dim grid grid"]:
        x = mask[None].astype(self.no_mask_embed.dtype)
        for conv, norm in zip(self.mask_convs[:2], self.mask_norms):
            x = jax.nn.gelu(norm(conv(x)), approximate=False)
        return self.mask_convs[2](x)

    def __call__(
        self, prompt: InteractivePrompt
    ) -> tuple[
        Float[Array, "tokens dim"],
        Bool[Array, " tokens"],
        Float[Array, "dim grid grid"],
    ]:
        """Return sparse tokens, the mask of tokens in use, and the dense prompt.

        The token layout is fixed: the points, the two box corners, then the
        padding point that is always appended. Without any point or box, the
        first point slot also holds a padding point, because the reference
        encodes one empty point in that case. Tokens out of use are masked from
        attention, so they have no effect on the output.
        """
        dtype = self.no_mask_embed.dtype
        corners = prompt.box.reshape(2, 2)
        coords = jnp.concatenate(
            (prompt.points, corners, jnp.zeros((1, 2), prompt.points.dtype)), axis=0
        )
        labels = jnp.concatenate(
            (prompt.point_labels, jnp.asarray([2, 3, -1], prompt.point_labels.dtype))
        )
        has_point = prompt.point_labels != -1
        empty_prompt = ~(jnp.any(has_point) | prompt.has_box)
        first_slot = jnp.zeros(has_point.shape, bool).at[0].set(True)
        valid = jnp.concatenate(
            (
                has_point | (first_slot & empty_prompt),
                jnp.stack((prompt.has_box, prompt.has_box, jnp.asarray(True))),
            )
        )
        position = fourier_encoding(
            self.fourier_matrix, (coords + 0.5) / self.image_size
        ).astype(dtype)
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
    """Post-norm block updating prompt tokens from the image and back.

    Position encodings go on the queries and the keys of each attention. The
    first block's self-attention takes no position encoding and adds no residual.
    """

    attn: CrossAttention
    norm1: eqx.nn.LayerNorm
    cross_attn: tuple[CrossAttention, CrossAttention]
    norm2: eqx.nn.LayerNorm
    mlp: Mlp
    norm3: eqx.nn.LayerNorm
    norm4: eqx.nn.LayerNorm
    use_token_position: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        *,
        key: PRNGKeyArray,
        use_token_position: bool,
    ):
        key_attn, key_to_image, key_to_token, key_mlp = jr.split(key, 4)
        self.attn = CrossAttention(dim, num_heads, key=key_attn)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.cross_attn = (
            CrossAttention(dim, num_heads, internal_dim=dim // 2, key=key_to_image),
            CrossAttention(dim, num_heads, internal_dim=dim // 2, key=key_to_token),
        )
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=ffn_features, act_layer="relu", key=key_mlp)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.norm4 = eqx.nn.LayerNorm(dim)
        self.use_token_position = use_token_position

    def __call__(
        self,
        tokens: Float[Array, "tokens dim"],
        image: Float[Array, "keys dim"],
        token_pos: Float[Array, "tokens dim"],
        image_pos: Float[Array, "keys dim"],
        valid: Bool[Array, " tokens"],
    ) -> tuple[Float[Array, "tokens dim"], Float[Array, "keys dim"]]:
        if self.use_token_position:
            positioned = tokens + token_pos
            tokens = layer_norm(
                self.norm1,
                tokens
                + self.attn(positioned, positioned, values=tokens, key_mask=valid),
            )
        else:
            tokens = layer_norm(self.norm1, self.attn(tokens, tokens, key_mask=valid))
        token_to_image, image_to_token = self.cross_attn
        tokens = layer_norm(
            self.norm2,
            tokens
            + token_to_image(tokens + token_pos, image + image_pos, values=image),
        )
        tokens = layer_norm(self.norm3, tokens + self.mlp(tokens, None, inference=True))
        attended = image_to_token(
            image + image_pos, tokens + token_pos, values=tokens, key_mask=valid
        )
        return tokens, layer_norm(self.norm4, image + attended)


class TwoWayTransformer(eqx.Module):
    """Transformer shared by the interactive and multiplex mask decoders."""

    blocks: tuple[TwoWayBlock, ...]
    cross_attn: CrossAttention
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
        key_blocks, key_final = jr.split(key)
        self.blocks = tuple(
            TwoWayBlock(
                dim,
                num_heads,
                ffn_features,
                use_token_position=index > 0,
                key=block_key,
            )
            for index, block_key in enumerate(jr.split(key_blocks, depth))
        )
        self.cross_attn = CrossAttention(
            dim, num_heads, internal_dim=dim // 2, key=key_final
        )
        self.norm = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        tokens: Float[Array, "tokens dim"],
        image: Float[Array, "keys dim"],
        image_pos: Float[Array, "keys dim"],
        valid: Bool[Array, " tokens"],
    ) -> tuple[Float[Array, "tokens dim"], Float[Array, "keys dim"]]:
        """Return the updated tokens and image; the input tokens are the positions."""
        token_pos = tokens
        for block in self.blocks:
            tokens, image = block(tokens, image, token_pos, image_pos, valid)
        attended = self.cross_attn(tokens + token_pos, image + image_pos, values=image)
        return layer_norm(self.norm, tokens + attended), image


class MaskUpsampler(eqx.Module):
    """Upsample decoder features fourfold, fusing the two high-resolution levels."""

    conv_s0: eqx.nn.Conv2d
    conv_s1: eqx.nn.Conv2d
    deconvs: tuple[eqx.nn.ConvTranspose2d, eqx.nn.ConvTranspose2d]
    norm: LayerNorm2d

    def __init__(self, dim: int, *, key: PRNGKeyArray):
        key_s0, key_s1, key_d0, key_d1 = jr.split(key, 4)
        self.conv_s0 = eqx.nn.Conv2d(dim, dim // 8, 1, key=key_s0)
        self.conv_s1 = eqx.nn.Conv2d(dim, dim // 4, 1, key=key_s1)
        self.deconvs = (
            eqx.nn.ConvTranspose2d(dim, dim // 4, 2, stride=2, key=key_d0),
            eqx.nn.ConvTranspose2d(dim // 4, dim // 8, 2, stride=2, key=key_d1),
        )
        self.norm = LayerNorm2d(dim // 4)

    def project(
        self, high_res: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Project the two high-resolution levels to the upsampler's widths."""
        high, middle = high_res
        return self.conv_s0(high), self.conv_s1(middle)

    def __call__(
        self,
        image: Float[Array, "dim grid grid"],
        high_res: tuple[jax.Array, jax.Array],
    ) -> Float[Array, "pixel height width"]:
        """Upsample ``image``, adding levels already passed through :meth:`project`."""
        high, middle = high_res
        x = upsample_twofold(self.deconvs[0], image) + middle
        x = jax.nn.gelu(self.norm(x), approximate=False)
        x = upsample_twofold(self.deconvs[1], x) + high
        return jax.nn.gelu(x, approximate=False)


class MaskDecoder(eqx.Module):
    """SAM 2-style decoder with four mask tokens, IoU and object scores."""

    transformer: TwoWayTransformer
    upsampler: MaskUpsampler
    mask_tokens: Float[Array, "masks dim"]
    iou_token: Float[Array, " dim"]
    object_token: Float[Array, " dim"]
    mask_embed: tuple[LinearStack, ...]
    iou_head: LinearStack
    object_head: LinearStack

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
        self.mask_embed = tuple(
            LinearStack(dim, dim, dim // 8, 3, key=subkey)
            for subkey in jr.split(key_embed, _INTERACTIVE_MASK_TOKENS)
        )
        self.iou_head = LinearStack(
            dim, dim, _INTERACTIVE_MASK_TOKENS, 3, key=key_iou_head
        )
        self.object_head = LinearStack(dim, dim, 1, 3, key=key_object_head)

    def __call__(
        self,
        image: Float[Array, "dim grid grid"],
        image_pos: Float[Array, "dim grid grid"],
        sparse: Float[Array, "tokens dim"],
        sparse_valid: Bool[Array, " tokens"],
        dense: Float[Array, "dim grid grid"],
        high_res: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return mask logits (float32), IoU, the object score and the mask tokens."""
        dtype = image.dtype
        grid = image.shape[-1]
        tokens = jnp.concatenate(
            (
                self.object_token[None],
                self.iou_token[None],
                self.mask_tokens,
                sparse,
            ),
            axis=0,
        ).astype(dtype)
        valid = jnp.concatenate(
            (jnp.ones(2 + _INTERACTIVE_MASK_TOKENS, bool), sparse_valid)
        )
        tokens, attended = self.transformer(
            tokens,
            rearrange(image + dense, "c h w -> (h w) c"),
            rearrange(image_pos, "c h w -> (h w) c"),
            valid,
        )
        pixels = self.upsampler(
            rearrange(attended, "(h w) c -> c h w", h=grid, w=grid), high_res
        )
        mask_tokens = tokens[2 : 2 + _INTERACTIVE_MASK_TOKENS]
        embeddings = jnp.stack(
            [mlp(token) for mlp, token in zip(self.mask_embed, mask_tokens)]
        )
        masks = jnp.einsum(
            "qc,chw->qhw",
            embeddings.astype(jnp.float32),
            pixels.astype(jnp.float32),
        )
        return (
            masks,
            self.iou_head(tokens[1]),
            self.object_head(tokens[0])[0],
            mask_tokens,
        )


class MultiplexMaskDecoder(eqx.Module):
    """Decode several objects in one pass, without sparse or dense prompts.

    Each slot has an object-score token, an IoU token and three mask tokens,
    all with their own weights. A slot's mask tokens carry that slot's valid or
    invalid embedding, which is how the decoder keeps objects apart. The three
    per-slot mask embeddings are shared across slots.
    """

    transformer: TwoWayTransformer
    upsampler: MaskUpsampler
    mask_tokens: Float[Array, "tokens dim"]
    iou_token: Float[Array, "slots dim"]
    object_token: Float[Array, "slots dim"]
    valid_embed: Float[Array, "slots dim"]
    invalid_embed: Float[Array, "slots dim"]
    mask_embed: tuple[LinearStack, ...]
    iou_head: LinearStack
    object_head: LinearStack
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
        self.mask_tokens = jr.normal(key_masks, (num_slots * _MASKS_PER_SLOT, dim))
        self.iou_token = jr.normal(key_iou, (num_slots, dim))
        self.object_token = jr.normal(key_object, (num_slots, dim))
        self.valid_embed = jr.normal(key_valid, (num_slots, dim))
        self.invalid_embed = jr.normal(key_invalid, (num_slots, dim))
        self.mask_embed = tuple(
            LinearStack(dim, dim, dim // 8, 3, key=subkey)
            for subkey in jr.split(key_embed, _MASKS_PER_SLOT)
        )
        self.iou_head = LinearStack(dim, dim, _MASKS_PER_SLOT, 3, key=key_iou_head)
        self.object_head = LinearStack(dim, dim, 1, 3, key=key_object_head)
        self.num_slots = num_slots

    def __call__(
        self,
        image: Float[Array, "dim grid grid"],
        image_pos: Float[Array, "dim grid grid"],
        high_res: tuple[jax.Array, jax.Array],
        slot_valid: Bool[Array, " slots"],
    ) -> MultiplexOutput:
        dtype = image.dtype
        grid = image.shape[-1]
        slot_embed = jnp.where(
            slot_valid[:, None], self.valid_embed, self.invalid_embed
        )
        mask_tokens = (
            self.mask_tokens.reshape(self.num_slots, _MASKS_PER_SLOT, -1)
            + slot_embed[:, None, :]
        )
        tokens = jnp.concatenate(
            (
                self.object_token,
                self.iou_token,
                mask_tokens.reshape(-1, mask_tokens.shape[-1]),
            ),
            axis=0,
        ).astype(dtype)
        tokens, attended = self.transformer(
            tokens,
            rearrange(image, "c h w -> (h w) c"),
            rearrange(image_pos, "c h w -> (h w) c"),
            jnp.ones(tokens.shape[0], bool),
        )
        pixels = self.upsampler(
            rearrange(attended, "(h w) c -> c h w", h=grid, w=grid), high_res
        )

        object_tokens = tokens[: self.num_slots]
        iou_tokens = tokens[self.num_slots : 2 * self.num_slots]
        slots = tokens[2 * self.num_slots :].reshape(
            self.num_slots, _MASKS_PER_SLOT, tokens.shape[-1]
        )
        embeddings = jnp.stack(
            [mlp(slots[:, index]) for index, mlp in enumerate(self.mask_embed)],
            axis=1,
        )
        object_scores = self.object_head(object_tokens)[:, 0]
        masks = jnp.einsum(
            "smc,chw->smhw",
            embeddings.astype(jnp.float32),
            pixels.astype(jnp.float32),
        )
        masks = jnp.where(
            (object_scores > 0)[:, None, None, None], masks, _NO_OBJECT_LOGIT
        )
        return MultiplexOutput(
            mask_logits=masks,
            iou=self.iou_head(iou_tokens),
            object_score_logits=object_scores,
            mask_tokens=slots,
        )


def _select_mask(
    mask_logits: Float[Array, "masks height width"],
    iou: Float[Array, " masks"],
    num_labels: Int[Array, ""],
    has_mask: Bool[Array, ""],
) -> Int[Array, ""]:
    """Pick the output mask from the prompt.

    With at most one label and no mask prompt, the best multimask output is
    selected. Otherwise the single-mask output is selected unless it is
    unstable, in which case the best multimask output is selected.
    """
    best_multimask = jnp.argmax(iou[1:].astype(jnp.float32)) + 1
    single = mask_logits[0]
    above = jnp.sum(single > _STABILITY_OFFSET, axis=(-2, -1))
    outer = jnp.sum(single > -_STABILITY_OFFSET, axis=(-2, -1))
    stability = jnp.where(outer > 0, above / jnp.maximum(outer, 1), 1.0)
    use_single = ((num_labels > 1) | has_mask) & (stability >= _STABILITY_THRESHOLD)
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
        return self.predict_with_tokens(features, prompt)[0]

    def predict_with_tokens(
        self, features: ImageFeatures, prompt: InteractivePrompt
    ) -> tuple[InteractiveOutput, Float[Array, "masks dim"]]:
        """Return the output and the mask tokens the tracker builds pointers from."""
        high, middle, low = features.interactive
        image = low + self.no_memory_embed[:, None, None].astype(low.dtype)
        sparse, valid, dense = self.prompt_encoder(prompt)
        masks, iou, object_score, tokens = self.mask_decoder(
            image,
            self.prompt_encoder.dense_position_encoding(),
            sparse,
            valid,
            dense,
            self.mask_decoder.upsampler.project((high, middle)),
        )
        num_labels = jnp.sum(prompt.point_labels != -1) + 2 * prompt.has_box
        output = InteractiveOutput(
            mask_logits=masks,
            iou=iou,
            object_score_logit=object_score,
            selected=_select_mask(masks, iou, num_labels, prompt.has_mask),
        )
        return output, tokens


__all__ = ["ImagePositionEncoding", "InteractiveOutput", "InteractivePrompt"]
