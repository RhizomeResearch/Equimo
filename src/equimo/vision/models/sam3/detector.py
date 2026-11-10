"""Concept detector of SAM 3.1: prompt encoding, fusion, DETR decoder and masks."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from equimo.core.layers.attention import CrossAttention
from equimo.core.layers.ffn import LinearStack, Mlp
from equimo.core.layers.norm import layer_norm
from equimo.core.ops import inverse_sigmoid
from equimo.vision.layers.posemb import sine_coordinates, sine_position_encoding
from equimo.vision.models.sam3.encoder import ImageFeatures
from equimo.vision.ops import grid_sample, roi_align

_EXEMPLAR_TEXT = "<text placeholder>"
_EXEMPLAR_POOL_SIZE = 7
_SCORE_CLAMP = 12.0
_BOX_BIAS_SCALE = 8.0


def _pad_to(x: jax.Array, length: int, fill: float | bool | int) -> jax.Array:
    """Pad the first axis of ``x`` to ``length`` with ``fill``."""
    missing = length - x.shape[0]
    if missing < 0:
        raise ValueError(f"Got {x.shape[0]} entries, more than the maximum {length}.")
    pad_width = ((0, missing), *((0, 0) for _ in x.shape[1:]))
    return jnp.pad(x, pad_width, constant_values=fill)


class ConceptPrompt(eqx.Module):
    """Text and exemplar prompt with fixed shapes; padding masks use True for padding.

    Boxes are normalized cxcywh and points normalized xy; labels are 1 for positive
    and 0 for negative. The reference has no text-free path: for exemplars alone
    it encodes the string ``"<text placeholder>"``, which :func:`encode_concept`
    does for you.
    """

    text_features: Float[Array, "text dim"]
    text_padding_mask: Bool[Array, " text"]
    boxes: Float[Array, "boxes 4"]
    box_labels: Int[Array, " boxes"]
    box_padding_mask: Bool[Array, " boxes"]
    points: Float[Array, "points 2"]
    point_labels: Int[Array, " points"]
    point_padding_mask: Bool[Array, " points"]

    @classmethod
    def create(
        cls,
        *,
        text_features: jax.Array | None = None,
        text_padding_mask: jax.Array | None = None,
        boxes: jax.Array | None = None,
        box_labels: jax.Array | None = None,
        points: jax.Array | None = None,
        point_labels: jax.Array | None = None,
        dim: int = 256,
        max_text_tokens: int = 32,
        max_boxes: int = 16,
        max_points: int = 16,
        source_size: tuple[int, int] | None = None,
    ) -> ConceptPrompt:
        """Pad the given prompts to the maxima; absent prompts are fully padded.

        With ``source_size`` ``(height, width)``, ``boxes`` are ``(x1, y1, x2, y2)``
        and ``points`` ``(x, y)`` in those pixels instead of normalized.
        """
        if source_size is not None:
            boxes, points = _exemplars_from_source_pixels(boxes, points, source_size)
        if text_features is None:
            text_features = jnp.zeros((0, dim), jnp.float32)
            text_padding_mask = jnp.zeros((0,), bool)
        elif text_padding_mask is None:
            text_padding_mask = jnp.zeros(text_features.shape[0], bool)
        if boxes is None:
            boxes = jnp.zeros((0, 4), jnp.float32)
            box_labels = jnp.zeros((0,), jnp.int32)
        elif box_labels is None:
            box_labels = jnp.ones(boxes.shape[0], jnp.int32)
        if points is None:
            points = jnp.zeros((0, 2), jnp.float32)
            point_labels = jnp.zeros((0,), jnp.int32)
        elif point_labels is None:
            point_labels = jnp.ones(points.shape[0], jnp.int32)

        assert text_padding_mask is not None
        assert box_labels is not None
        assert point_labels is not None
        return cls(
            text_features=_pad_to(text_features, max_text_tokens, 0.0),
            text_padding_mask=_pad_to(
                jnp.asarray(text_padding_mask, bool), max_text_tokens, True
            ),
            boxes=_pad_to(boxes, max_boxes, 0.0),
            box_labels=_pad_to(jnp.asarray(box_labels, jnp.int32), max_boxes, 0),
            box_padding_mask=_pad_to(jnp.zeros(boxes.shape[0], bool), max_boxes, True),
            points=_pad_to(points, max_points, 0.0),
            point_labels=_pad_to(jnp.asarray(point_labels, jnp.int32), max_points, 0),
            point_padding_mask=_pad_to(
                jnp.zeros(points.shape[0], bool), max_points, True
            ),
        )

    def without_exemplars(self) -> ConceptPrompt:
        """Return the prompt with its boxes and points padded out."""
        return eqx.tree_at(
            lambda p: (
                p.boxes,
                p.box_labels,
                p.box_padding_mask,
                p.points,
                p.point_labels,
                p.point_padding_mask,
            ),
            self,
            (
                jnp.zeros_like(self.boxes),
                jnp.zeros_like(self.box_labels),
                jnp.ones_like(self.box_padding_mask),
                jnp.zeros_like(self.points),
                jnp.zeros_like(self.point_labels),
                jnp.ones_like(self.point_padding_mask),
            ),
        )


def _exemplars_from_source_pixels(
    boxes: jax.Array | None,
    points: jax.Array | None,
    source_size: tuple[int, int],
) -> tuple[jax.Array | None, jax.Array | None]:
    """Convert corner boxes and points in source pixels to normalized exemplars."""
    height, width = source_size
    scale = jnp.asarray([width, height], jnp.float32)
    if boxes is not None:
        boxes = jnp.asarray(boxes, jnp.float32)
        low, high = boxes[..., :2], boxes[..., 2:]
        boxes = jnp.concatenate(((low + high) / 2, high - low), axis=-1) / jnp.tile(
            scale, 2
        )
    if points is not None:
        points = jnp.asarray(points, jnp.float32) / scale
    return boxes, points


def encode_concept(
    text_model: Any,
    tokenizer: Any,
    concept: str | None = None,
    *,
    boxes: jax.Array | None = None,
    box_labels: jax.Array | None = None,
    points: jax.Array | None = None,
    point_labels: jax.Array | None = None,
    source_size: tuple[int, int] | None = None,
) -> ConceptPrompt:
    """Build a :class:`ConceptPrompt` from a concept, exemplars or both.

    ``text_model`` and ``tokenizer`` are the ``sam31_text`` model and a
    ``ClipBpeTokenizer``. With exemplars and no concept, the reference's
    placeholder text is encoded, as the reference does.
    """
    if concept is None and boxes is None and points is None:
        raise ValueError("Give a concept, exemplars or both.")
    token_ids, padding_mask = tokenizer.encode(
        [_EXEMPLAR_TEXT if concept is None else concept]
    )
    token_ids = jnp.asarray(token_ids[0], jnp.int32)
    padding_mask = jnp.asarray(padding_mask[0])
    return ConceptPrompt.create(
        text_features=text_model.features(token_ids, padding_mask),
        text_padding_mask=padding_mask,
        boxes=boxes,
        box_labels=box_labels,
        points=points,
        point_labels=point_labels,
        source_size=source_size,
    )


class DetectorOutput(eqx.Module):
    """Detector predictions for one image."""

    logits: Float[Array, " queries"]
    presence_logit: Float[Array, ""]
    boxes: Float[Array, "queries 4"]
    mask_logits: Float[Array, "queries height width"]
    layer_boxes: Float[Array, "layers queries 4"]
    layer_logits: Float[Array, "layers queries"]


def joint_logit(class_logit: jax.Array, presence_logit: jax.Array) -> jax.Array:
    """Return the logit of ``sigmoid(class) * sigmoid(presence)`` in float32.

    The product passes through :func:`inverse_sigmoid`, so the joint logit stays
    within about +/- 6.9.
    """
    class_probability = jax.nn.sigmoid(class_logit.astype(jnp.float32))
    presence_probability = jax.nn.sigmoid(presence_logit.astype(jnp.float32))
    return inverse_sigmoid(class_probability * presence_probability)


class GeometryBlock(eqx.Module):
    """Pre-norm exemplar block: self-attention, image cross-attention, feed-forward."""

    norm1: eqx.nn.LayerNorm
    attn: CrossAttention
    norm2: eqx.nn.LayerNorm
    cross_attn: CrossAttention
    norm3: eqx.nn.LayerNorm
    mlp: Mlp

    def __init__(
        self, dim: int, num_heads: int, ffn_features: int, *, key: PRNGKeyArray
    ):
        key_attn, key_cross, key_mlp = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, key=key_attn)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, key=key_cross)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=ffn_features, act_layer="relu", key=key_mlp)

    def __call__(
        self,
        tokens: Float[Array, "tokens dim"],
        valid: Bool[Array, " tokens"],
        image: Float[Array, "keys dim"],
        image_pos: Float[Array, "keys dim"],
    ) -> Float[Array, "tokens dim"]:
        normed = layer_norm(self.norm1, tokens)
        tokens = tokens + self.attn(normed, normed, key_mask=valid)
        normed = layer_norm(self.norm2, tokens)
        tokens = tokens + self.cross_attn(normed, image + image_pos, values=image)
        return tokens + self.mlp(layer_norm(self.norm3, tokens), None, inference=True)


class GeometryEncoder(eqx.Module):
    """Encode exemplar points and boxes as prompt tokens, followed by a class token.

    Each exemplar sums a linear projection of its coordinates, features sampled
    from the normalized image and its sine position encoding.
    """

    point_direct: eqx.nn.Linear
    point_pool: eqx.nn.Linear
    point_position: eqx.nn.Linear
    box_direct: eqx.nn.Linear
    box_pool: eqx.nn.Conv2d
    box_position: eqx.nn.Linear
    label_embed: Float[Array, "2 dim"]
    cls_token: Float[Array, " dim"]
    image_norm: eqx.nn.LayerNorm
    final_proj: eqx.nn.Linear
    final_norm: eqx.nn.LayerNorm
    blocks: tuple[GeometryBlock, ...]
    norm: eqx.nn.LayerNorm
    position_features: int = eqx.field(static=True)

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
            key_point_direct,
            key_point_pool,
            key_point_position,
            key_box_direct,
            key_box_pool,
            key_box_position,
            key_label,
            key_cls,
            key_final,
            *key_blocks,
        ) = jr.split(key, depth + 9)
        self.position_features = dim // 2
        self.point_direct = eqx.nn.Linear(2, dim, key=key_point_direct)
        self.point_pool = eqx.nn.Linear(dim, dim, key=key_point_pool)
        self.point_position = eqx.nn.Linear(dim, dim, key=key_point_position)
        self.box_direct = eqx.nn.Linear(4, dim, key=key_box_direct)
        self.box_pool = eqx.nn.Conv2d(dim, dim, _EXEMPLAR_POOL_SIZE, key=key_box_pool)
        self.box_position = eqx.nn.Linear(dim + 2, dim, key=key_box_position)
        self.label_embed = jr.normal(key_label, (2, dim))
        self.cls_token = jr.normal(key_cls, (dim,))
        self.image_norm = eqx.nn.LayerNorm(dim)
        self.final_proj = eqx.nn.Linear(dim, dim, key=key_final)
        self.final_norm = eqx.nn.LayerNorm(dim)
        self.blocks = tuple(
            GeometryBlock(dim, num_heads, ffn_features, key=block_key)
            for block_key in key_blocks
        )
        self.norm = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        image_tokens: Float[Array, "keys dim"],
        image_pos: Float[Array, "keys dim"],
        grid: int,
        prompt: ConceptPrompt,
    ) -> tuple[Float[Array, "tokens dim"], Bool[Array, " tokens"]]:
        """Return the tokens of points, boxes and the class token, then the valid mask.

        The layout is fixed: ``max_points`` point slots, ``max_boxes`` box slots
        and the class token, so padded slots stay inside the sequence.
        """
        dtype = image_tokens.dtype
        image_map = rearrange(
            layer_norm(self.image_norm, image_tokens),
            "(h w) c -> c h w",
            h=grid,
            w=grid,
        )

        points = prompt.points.astype(jnp.float32)
        x, y = points[:, 0], points[:, 1]
        position = jnp.concatenate(
            (
                sine_coordinates(x, self.position_features),
                sine_coordinates(y, self.position_features),
            ),
            axis=-1,
        )
        point_tokens = (
            jax.vmap(self.point_direct)(points.astype(dtype))
            + jax.vmap(self.point_pool)(grid_sample(image_map, points * 2.0 - 1.0))
            + jax.vmap(self.point_position)(position.astype(dtype))
            + self.label_embed[prompt.point_labels].astype(dtype)
        )

        boxes = prompt.boxes.astype(jnp.float32)
        center, size = boxes[:, :2], boxes[:, 2:]
        corners = jnp.concatenate((center - size / 2, center + size / 2), axis=-1)
        pooled = roi_align(image_map, corners * grid, _EXEMPLAR_POOL_SIZE)
        position = jnp.concatenate(
            (
                sine_coordinates(center[:, 1], self.position_features),
                sine_coordinates(center[:, 0], self.position_features),
                size[:, 1:],
                size[:, :1],
            ),
            axis=-1,
        )
        box_tokens = (
            jax.vmap(self.box_direct)(boxes.astype(dtype))
            + jax.vmap(self.box_pool)(pooled).reshape(pooled.shape[0], -1)
            + jax.vmap(self.box_position)(position.astype(dtype))
            + self.label_embed[prompt.box_labels].astype(dtype)
        )

        tokens = jnp.concatenate(
            (point_tokens, box_tokens, self.cls_token[None].astype(dtype)), axis=0
        )
        valid = jnp.concatenate(
            (
                ~prompt.point_padding_mask,
                ~prompt.box_padding_mask,
                jnp.ones(1, bool),
            )
        )
        tokens = layer_norm(self.final_norm, jax.vmap(self.final_proj)(tokens))
        for block in self.blocks:
            tokens = block(tokens, valid, image_tokens, image_pos)
        return layer_norm(self.norm, tokens), valid


class FusionBlock(eqx.Module):
    """Pre-norm block conditioning image tokens on the prompt tokens."""

    norm1: eqx.nn.LayerNorm
    attn: CrossAttention
    norm2: eqx.nn.LayerNorm
    cross_attn: CrossAttention
    norm3: eqx.nn.LayerNorm
    mlp: Mlp

    def __init__(
        self, dim: int, num_heads: int, ffn_features: int, *, key: PRNGKeyArray
    ):
        key_attn, key_cross, key_mlp = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, key=key_attn)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, key=key_cross)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=ffn_features, act_layer="relu", key=key_mlp)

    def __call__(
        self,
        x: Float[Array, "tokens dim"],
        pos: Float[Array, "tokens dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
    ) -> Float[Array, "tokens dim"]:
        normed = layer_norm(self.norm1, x)
        positioned = normed + pos
        x = x + self.attn(positioned, positioned, values=normed)
        x = x + self.cross_attn(
            layer_norm(self.norm2, x), prompt, key_mask=prompt_valid
        )
        return x + self.mlp(layer_norm(self.norm3, x), None, inference=True)


class FusionEncoder(eqx.Module):
    """Stack of fusion blocks over the image tokens."""

    blocks: tuple[FusionBlock, ...]

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
            FusionBlock(dim, num_heads, ffn_features, key=block_key)
            for block_key in jr.split(key, depth)
        )

    def __call__(
        self,
        x: Float[Array, "tokens dim"],
        pos: Float[Array, "tokens dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
    ) -> Float[Array, "tokens dim"]:
        for block in self.blocks:
            x = block(x, pos, prompt, prompt_valid)
        return x


class BoxRelativeBias(eqx.Module):
    """Per-head attention bias from each cell's offset to the box edges.

    Offsets to the two edges along an axis go through a signed log scale and a
    small network per axis; the bias of a cell is the sum of both axes.
    """

    embed_x: LinearStack
    embed_y: LinearStack

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        key_x, key_y = jr.split(key)
        self.embed_x = LinearStack(2, dim, num_heads, 2, key=key_x)
        self.embed_y = LinearStack(2, dim, num_heads, 2, key=key_y)

    def __call__(
        self, boxes: Float[Array, "queries 4"], grid: int
    ) -> Float[Array, "heads queries keys"]:
        with jax.ensure_compile_time_eval():
            cells = jnp.arange(grid, dtype=jnp.float32) / grid

        def log_scaled(delta: jax.Array) -> jax.Array:
            scaled = delta * _BOX_BIAS_SCALE
            return jnp.sign(scaled) * jnp.log2(jnp.abs(scaled) + 1.0) / 3.0

        boxes = boxes.astype(jnp.float32)
        half = boxes[:, 2:] / 2.0
        x_edges = jnp.stack(
            (boxes[:, 0] - half[:, 0], boxes[:, 0] + half[:, 0]), axis=-1
        )
        y_edges = jnp.stack(
            (boxes[:, 1] - half[:, 1], boxes[:, 1] + half[:, 1]), axis=-1
        )
        offsets_x = log_scaled(cells[None, :, None] - x_edges[:, None, :])
        offsets_y = log_scaled(cells[None, :, None] - y_edges[:, None, :])
        bias_x = self.embed_x(offsets_x.astype(self.embed_x.layers[0].weight.dtype))
        bias_y = self.embed_y(offsets_y.astype(self.embed_y.layers[0].weight.dtype))
        bias = bias_y[:, :, None, :] + bias_x[:, None, :, :]
        queries, _, _, heads = bias.shape
        return jnp.moveaxis(bias.reshape(queries, grid * grid, heads), -1, 0).astype(
            jnp.float32
        )


class DecoderBlock(eqx.Module):
    """Post-norm DETR layer: self, prompt and image attention, then feed-forward."""

    attn: CrossAttention
    norm1: eqx.nn.LayerNorm
    cross_attn: tuple[CrossAttention, CrossAttention]
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    mlp: Mlp
    norm4: eqx.nn.LayerNorm

    def __init__(
        self, dim: int, num_heads: int, ffn_features: int, *, key: PRNGKeyArray
    ):
        key_attn, key_prompt, key_image, key_mlp = jr.split(key, 4)
        self.attn = CrossAttention(dim, num_heads, key=key_attn)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.cross_attn = (
            CrossAttention(dim, num_heads, key=key_prompt),
            CrossAttention(dim, num_heads, key=key_image),
        )
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=ffn_features, act_layer="relu", key=key_mlp)
        self.norm4 = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        tokens: Float[Array, "tokens dim"],
        position: Float[Array, "tokens dim"],
        memory: Float[Array, "keys dim"],
        memory_pos: Float[Array, "keys dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
        bias: Float[Array, "heads tokens keys"],
    ) -> Float[Array, "tokens dim"]:
        positioned = tokens + position
        tokens = layer_norm(
            self.norm1, tokens + self.attn(positioned, positioned, values=tokens)
        )
        prompt_attn, image_attn = self.cross_attn
        tokens = layer_norm(
            self.norm2,
            tokens + prompt_attn(tokens + position, prompt, key_mask=prompt_valid),
        )
        tokens = layer_norm(
            self.norm3,
            tokens
            + image_attn(
                tokens + position, memory + memory_pos, values=memory, bias=bias
            ),
        )
        feed_forward = self.mlp(
            tokens.astype(jnp.float32), None, inference=True
        ).astype(tokens.dtype)
        return layer_norm(self.norm4, tokens + feed_forward)


class ConceptDecoder(eqx.Module):
    """DETR decoder with a presence token and iterative box refinement."""

    queries: Float[Array, "queries dim"]
    reference_boxes: Float[Array, "queries 4"]
    presence_token: Float[Array, " dim"]
    position_proj: LinearStack
    relative_bias: BoxRelativeBias
    blocks: tuple[DecoderBlock, ...]
    norm: eqx.nn.LayerNorm
    presence_norm: eqx.nn.LayerNorm
    box_head: LinearStack
    presence_head: LinearStack
    position_features: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        depth: int,
        num_queries: int,
        *,
        key: PRNGKeyArray,
    ):
        (
            key_queries,
            key_boxes,
            key_presence,
            key_position,
            key_bias,
            key_box_head,
            key_presence_head,
            *key_blocks,
        ) = jr.split(key, depth + 7)
        self.queries = jr.normal(key_queries, (num_queries, dim))
        self.reference_boxes = jr.normal(key_boxes, (num_queries, 4))
        self.presence_token = jr.normal(key_presence, (dim,))
        self.position_features = dim // 2
        self.position_proj = LinearStack(2 * dim, dim, dim, 2, key=key_position)
        self.relative_bias = BoxRelativeBias(dim, num_heads, key=key_bias)
        self.blocks = tuple(
            DecoderBlock(dim, num_heads, ffn_features, key=block_key)
            for block_key in key_blocks
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.presence_norm = eqx.nn.LayerNorm(dim)
        self.box_head = LinearStack(dim, dim, 4, 3, key=key_box_head)
        self.presence_head = LinearStack(dim, dim, 1, 3, key=key_presence_head)

    def _box_position(self, boxes: Float[Array, "queries 4"]) -> jax.Array:
        """Sine encoding of a box, ordered y, x, width, height, then projected."""
        center_x, center_y, width, height = jnp.split(boxes, 4, axis=-1)
        encoded = sine_coordinates(
            jnp.concatenate((center_y, center_x, width, height), axis=-1),
            self.position_features,
        )
        return self.position_proj(
            encoded.reshape(encoded.shape[0], -1).astype(self.queries.dtype)
        )

    def __call__(
        self,
        memory: Float[Array, "keys dim"],
        memory_pos: Float[Array, "keys dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
        grid: int,
    ) -> tuple[
        Float[Array, "layers queries dim"],
        Float[Array, "layers queries 4"],
        Float[Array, " layers"],
    ]:
        dtype = memory.dtype
        tokens = jnp.concatenate(
            (self.presence_token[None].astype(dtype), self.queries.astype(dtype)),
            axis=0,
        )
        boxes = jax.nn.sigmoid(self.reference_boxes.astype(jnp.float32))
        layer_tokens, layer_boxes, layer_presence = [], [], []
        for block in self.blocks:
            position = self._box_position(boxes).astype(dtype)
            # The presence token has no box, so it gets no position and no bias.
            position = jnp.concatenate(
                (jnp.zeros((1, position.shape[-1]), dtype), position), axis=0
            )
            bias = self.relative_bias(boxes, grid)
            bias = jnp.concatenate(
                (jnp.zeros((bias.shape[0], 1, bias.shape[-1]), bias.dtype), bias),
                axis=1,
            )
            tokens = block(
                tokens, position, memory, memory_pos, prompt, prompt_valid, bias
            )
            queries = layer_norm(self.norm, tokens[1:])
            delta = self.box_head(queries).astype(jnp.float32)
            new_boxes = jax.nn.sigmoid(inverse_sigmoid(boxes) + delta)
            # The next layer refines the boxes without changing this layer's own.
            boxes = jax.lax.stop_gradient(new_boxes)
            layer_tokens.append(queries)
            layer_boxes.append(new_boxes)
            presence = layer_norm(self.presence_norm, tokens[0])
            layer_presence.append(self.presence_head(presence)[0])
        return (
            jnp.stack(layer_tokens),
            jnp.stack(layer_boxes),
            jnp.stack(layer_presence),
        )


class DotProductScoring(eqx.Module):
    """Score queries by their dot product with the pooled, refined prompt."""

    prompt_mlp: LinearStack
    prompt_norm: eqx.nn.LayerNorm
    query_proj: eqx.nn.Linear
    prompt_proj: eqx.nn.Linear

    def __init__(self, dim: int, ffn_features: int, *, key: PRNGKeyArray):
        key_mlp, key_query, key_prompt = jr.split(key, 3)
        self.prompt_mlp = LinearStack(dim, ffn_features, dim, 2, key=key_mlp)
        self.prompt_norm = eqx.nn.LayerNorm(dim)
        self.query_proj = eqx.nn.Linear(dim, dim, key=key_query)
        self.prompt_proj = eqx.nn.Linear(dim, dim, key=key_prompt)

    def __call__(
        self,
        queries: Float[Array, "queries dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
    ) -> Float[Array, " queries"]:
        refined = layer_norm(self.prompt_norm, prompt + self.prompt_mlp(prompt))
        weights = prompt_valid.astype(jnp.float32)
        pooled = jnp.sum(refined.astype(jnp.float32) * weights[:, None], axis=0) / (
            jnp.maximum(weights.sum(), 1.0)
        )
        projected_prompt = self.prompt_proj(pooled.astype(prompt.dtype))
        projected_queries = jax.vmap(self.query_proj)(queries)
        scores = jnp.sum(
            projected_queries.astype(jnp.float32)
            * projected_prompt.astype(jnp.float32),
            axis=-1,
        )
        scale = jnp.sqrt(jnp.asarray(queries.shape[-1], jnp.float32))
        return jnp.clip(scores / scale, -_SCORE_CLAMP, _SCORE_CLAMP)


class PixelStage(eqx.Module):
    """3x3 convolution, group norm and ReLU."""

    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm

    def __init__(self, channels: int, *, key: PRNGKeyArray):
        self.conv = eqx.nn.Conv2d(channels, channels, 3, padding=1, key=key)
        self.norm = eqx.nn.GroupNorm(8, channels)

    def __call__(self, x: Float[Array, "channels height width"]) -> jax.Array:
        x = self.conv(x)
        return jax.nn.relu(self.norm(x.astype(jnp.float32)).astype(x.dtype))


def _upsample_nearest(x: Float[Array, "channels height width"]) -> jax.Array:
    channels, height, width = x.shape
    return jax.image.resize(x, (channels, 2 * height, 2 * width), method="nearest")


class SegmentationHead(eqx.Module):
    """Pixel decoder over the fused memory, dotted with per-query mask embeddings.

    The fused memory replaces the lowest pyramid level. It is upsampled twice,
    adding the middle and then the high level, so only two pixel stages run.
    """

    norm: eqx.nn.LayerNorm
    cross_attn: CrossAttention
    stages: tuple[PixelStage, PixelStage]
    instance_proj: eqx.nn.Conv2d
    mask_embed: LinearStack

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        key_cross, key_mask, key_middle, key_high, key_instance = jr.split(key, 5)
        self.norm = eqx.nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, key=key_cross)
        self.stages = (
            PixelStage(dim, key=key_middle),
            PixelStage(dim, key=key_high),
        )
        self.instance_proj = eqx.nn.Conv2d(dim, dim, 1, key=key_instance)
        self.mask_embed = LinearStack(dim, dim, dim, 3, key=key_mask)

    def pixel_embedding(
        self,
        pyramid: tuple[jax.Array, jax.Array, jax.Array],
        memory: Float[Array, "keys dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
    ) -> Float[Array, "dim height width"]:
        high, middle, low = pyramid
        grid = low.shape[-1]
        attended = memory + self.cross_attn(
            layer_norm(self.norm, memory), prompt, key_mask=prompt_valid
        )
        x = rearrange(attended, "(h w) c -> c h w", h=grid, w=grid)
        x = self.stages[0](_upsample_nearest(x) + middle)
        return self.stages[1](_upsample_nearest(x) + high)

    def __call__(
        self,
        pyramid: tuple[jax.Array, jax.Array, jax.Array],
        memory: Float[Array, "keys dim"],
        queries: Float[Array, "queries dim"],
        prompt: Float[Array, "prompt dim"],
        prompt_valid: Bool[Array, " prompt"],
    ) -> Float[Array, "queries height width"]:
        pixels = self.instance_proj(
            self.pixel_embedding(pyramid, memory, prompt, prompt_valid)
        )
        return jnp.einsum(
            "qc,chw->qhw",
            self.mask_embed(queries).astype(jnp.float32),
            pixels.astype(jnp.float32),
        )


class DetectorHead(eqx.Module):
    """Parts of the detector trained by head-only fine-tuning."""

    fusion: FusionEncoder
    decoder: ConceptDecoder
    scoring: DotProductScoring
    segmentation: SegmentationHead

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        fusion_depth: int,
        decoder_depth: int,
        num_queries: int,
        *,
        key: PRNGKeyArray,
    ):
        key_fusion, key_decoder, key_scoring, key_segmentation = jr.split(key, 4)
        self.fusion = FusionEncoder(
            dim, num_heads, ffn_features, fusion_depth, key=key_fusion
        )
        self.decoder = ConceptDecoder(
            dim, num_heads, ffn_features, decoder_depth, num_queries, key=key_decoder
        )
        self.scoring = DotProductScoring(dim, ffn_features, key=key_scoring)
        self.segmentation = SegmentationHead(dim, num_heads, key=key_segmentation)


class ConceptDetector(eqx.Module):
    """Find every instance of a prompted concept in one image."""

    geometry: GeometryEncoder
    head: DetectorHead

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_features: int,
        *,
        key: PRNGKeyArray,
        geometry_depth: int,
        fusion_depth: int,
        decoder_depth: int,
        num_queries: int,
    ):
        if dim % 8 != 0:
            raise ValueError(f"dim ({dim}) must be divisible by 8.")
        key_geometry, key_head = jr.split(key)
        self.geometry = GeometryEncoder(
            dim, num_heads, ffn_features, geometry_depth, key=key_geometry
        )
        self.head = DetectorHead(
            dim,
            num_heads,
            ffn_features,
            fusion_depth,
            decoder_depth,
            num_queries,
            key=key_head,
        )

    def __call__(
        self, features: ImageFeatures, prompt: ConceptPrompt
    ) -> DetectorOutput:
        high, middle, low = features.detector
        dtype = low.dtype
        dim, grid, _ = low.shape
        pos = rearrange(
            sine_position_encoding(grid, grid, dim // 2), "c h w -> (h w) c"
        ).astype(dtype)
        image_tokens = rearrange(low, "c h w -> (h w) c")

        geometry_tokens, geometry_valid = self.geometry(image_tokens, pos, grid, prompt)
        prompt_tokens = jnp.concatenate(
            (prompt.text_features.astype(dtype), geometry_tokens), axis=0
        )
        prompt_valid = jnp.concatenate((~prompt.text_padding_mask, geometry_valid))

        memory = self.head.fusion(image_tokens, pos, prompt_tokens, prompt_valid)
        layer_tokens, layer_boxes, layer_presence = self.head.decoder(
            memory, pos, prompt_tokens, prompt_valid, grid
        )
        layer_scores = jax.vmap(self.head.scoring, in_axes=(0, None, None))(
            layer_tokens, prompt_tokens, prompt_valid
        )
        layer_logits = joint_logit(layer_scores, layer_presence[:, None])
        mask_logits = self.head.segmentation(
            (high, middle, low), memory, layer_tokens[-1], prompt_tokens, prompt_valid
        )
        return DetectorOutput(
            logits=layer_logits[-1].astype(dtype),
            presence_logit=layer_presence[-1].astype(dtype),
            boxes=layer_boxes[-1].astype(dtype),
            mask_logits=mask_logits,
            layer_boxes=layer_boxes.astype(dtype),
            layer_logits=layer_logits.astype(dtype),
        )


__all__ = ["ConceptPrompt", "DetectorOutput"]
