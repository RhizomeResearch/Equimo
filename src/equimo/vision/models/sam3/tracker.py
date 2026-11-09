"""Memory encoder, memory attention and device tracking steps of SAM 3.1."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int, Int8, PRNGKeyArray

from equimo.core.layers.attention import CrossAttention, _fused_attention
from equimo.core.layers.norm import LayerNorm2d
from equimo.core.layers.rotary import RotaryFactors, apply_rotary
from equimo.vision.layers.convolution import ConvNeXtBlock
from equimo.vision.layers.posemb import VisionRoPE
from equimo.vision.models.sam3.common import (
    MLP,
    PositionedAttention,
    exchanged_rope_factors,
    flatten_map,
    layer_norm,
    sine_coordinates,
    sine_position_encoding,
    unflatten_map,
)
from equimo.vision.models.sam3.decoding import (
    greedy_iom_nms,
    iom_matrix,
    iou_matrix,
    remove_overlaps,
)
from equimo.vision.models.sam3.detector import DetectorOutput
from equimo.vision.models.sam3.encoder import ImageFeatures
from equimo.vision.ops import resize_bilinear
from equimo.vision.models.sam3.interactive import (
    _NO_OBJECT_LOGIT,
    ImagePositionEncoding,
    InteractiveHead,
    InteractivePrompt,
    MultiplexMaskDecoder,
)

# Dense float32 logits against 52,144 keys take about 8.6 GB per layer.
_ATTENTION_CHUNK_QUERIES = 1024
_MEMORY_STRIDE = 16
_MEMORY_DOWNSAMPLINGS = 4
_MASK_SCALE = 4
_MEMORY_INPUT_POSITION_SCALE = 0.1

SLOT_EMPTY = 0
SLOT_ACTIVE = 1
SLOT_REMOVED = 2

_NUM_COND_FRAMES = 4
_NUM_RECENT_FRAMES = 6
_NUM_POINTER_FRAMES = 15
_NUM_TIME_EMBEDDINGS = _NUM_RECENT_FRAMES + 1
_POINTER_DISTANCE_FRAMES = 16

_DETECTION_KEEP_PROBABILITY = 0.4
_NMS_IOM = 0.1
_BOUNDARY_MARGIN = 0.025
_MATCH_IOM = 0.1
_HIGH_IOM = 0.5
_NEW_OBJECT_PROBABILITY = 0.65
_CONFIDENT_DETECTION_PROBABILITY = 0.8
_CONFIDENT_TRACK_PROBABILITY = 0.8

_OVERLAP_LOSER_LOGIT = -10.0
_MIN_KEPT_AREA_FRACTION = 0.3


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

    ``queries`` and ``keys`` are already projected. cuDNN takes the key count
    directly. Without it, queries run in blocks, because float32 logits against
    every key do not fit in memory.
    """

    def split_heads(tokens: jax.Array) -> jax.Array:
        separated = tokens.reshape(
            tokens.shape[0], attention.num_heads, attention.head_dim
        )
        return jnp.swapaxes(separated, 0, 1)

    q = apply_rotary(split_heads(queries), rotary)
    k = apply_rotary(split_heads(keys), key_rotary)
    v = split_heads(jax.vmap(attention.v)(values))
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

    def __init__(
        self,
        dim: int,
        in_channels: int,
        start_channels: int,
        fuser_depth: int,
        *,
        key: PRNGKeyArray,
    ):
        key_masks, key_mask_proj, key_feature_proj, key_fuser = jr.split(key, 4)
        widths = [
            in_channels,
            *[start_channels * 4**i for i in range(1, _MEMORY_DOWNSAMPLINGS + 1)],
        ]
        self.mask_convs = tuple(
            eqx.nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, key=subkey)
            for width_in, width_out, subkey in zip(
                widths[:-1], widths[1:], jr.split(key_masks, _MEMORY_DOWNSAMPLINGS)
            )
        )
        self.mask_norms = tuple(LayerNorm2d(width) for width in widths[1:])
        self.mask_proj = eqx.nn.Conv2d(widths[-1], dim, 1, key=key_mask_proj)
        self.feature_proj = eqx.nn.Conv2d(dim, dim, 1, key=key_feature_proj)
        self.fuser = tuple(
            ConvNeXtBlock(dim, act_layer="exactgelu", init_values=1e-6, key=subkey)
            for subkey in jr.split(key_fuser, fuser_depth)
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
    """Pre-norm block of RoPE self-attention and packed-memory cross-attention.

    The cross-attention's queries add the current image's features, and its keys
    add the stored image features, each through their own projection.
    """

    norm1: eqx.nn.LayerNorm
    attn: PositionedAttention
    norm2: eqx.nn.LayerNorm
    cross_attn: CrossAttention
    image_q: eqx.nn.Linear
    image_k: eqx.nn.Linear
    norm3: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(
        self, dim: int, num_heads: int, ffn_features: int, *, key: PRNGKeyArray
    ):
        key_attn, key_cross, key_image_q, key_image_k, key_mlp = jr.split(key, 5)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = PositionedAttention(dim, num_heads, key=key_attn)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, key=key_cross)
        self.image_q = eqx.nn.Linear(dim, dim, key=key_image_q)
        self.image_k = eqx.nn.Linear(dim, dim, key=key_image_k)
        self.norm3 = eqx.nn.LayerNorm(dim)
        self.mlp = MLP(dim, ffn_features, dim, 2, act_layer="exactgelu", key=key_mlp)

    def __call__(
        self,
        x: Float[Array, "tokens dim"],
        image: Float[Array, "tokens dim"],
        memory_image: Float[Array, "keys dim"],
        memory: Float[Array, "keys dim"],
        memory_image_pos: Float[Array, "keys dim"],
        rotary: RotaryFactors,
        key_rotary: RotaryFactors,
        key_lengths: Int[Array, ""],
    ) -> Float[Array, "tokens dim"]:
        normed = layer_norm(self.norm1, x)
        x = x + self.attn(normed, normed, rotary=rotary, key_rotary=rotary)
        queries = jax.vmap(self.image_q)(image) + jax.vmap(self.cross_attn.q)(
            layer_norm(self.norm2, x)
        )
        keys = (
            jax.vmap(self.image_k)(memory_image)
            + jax.vmap(self.cross_attn.k)(memory)
            + memory_image_pos
        )
        x = x + _attend_packed_memory(
            self.cross_attn, queries, keys, memory, rotary, key_rotary, key_lengths
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
        x_pos: Float[Array, "tokens dim"],
        memory_image: Float[Array, "keys dim"],
        memory: Float[Array, "keys dim"],
        memory_image_pos: Float[Array, "keys dim"],
        *,
        num_spatial_keys: Int[Array, ""],
        key_lengths: Int[Array, ""],
    ) -> Float[Array, "tokens dim"]:
        """Attend to memory whose valid keys are packed at the front.

        ``memory_image`` holds the stored image features of each memory frame,
        ``memory`` the matching memory and ``memory_image_pos`` their position
        encodings. The first ``num_spatial_keys`` entries are spatial tokens of
        the memory frames. The object pointers follow, with zero image features
        and their temporal encoding as position, up to ``key_lengths`` valid
        entries.
        """
        grid = int(x.shape[0] ** 0.5)
        rotary = exchanged_rope_factors(self.rope, grid)
        key_rotary = _memory_key_rotary(rotary, memory.shape[0], num_spatial_keys, grid)
        out = x + _MEMORY_INPUT_POSITION_SCALE * x_pos.astype(x.dtype)
        for block in self.blocks:
            out = block(
                out,
                x,
                memory_image,
                memory,
                memory_image_pos,
                rotary,
                key_rotary,
                key_lengths,
            )
        return layer_norm(self.norm, out)


class TrackerState(eqx.Module):
    """Device state of the tracker, stacked over buckets of slots.

    Frame fields hold ``-1`` when the entry is absent. Ring buffers are indexed
    by the frame index modulo their length.
    """

    bucket_valid: Bool[Array, " buckets"]
    slot_status: Int8[Array, "buckets slots"]
    cond_memory: Float[Array, "buckets cond dim grid grid"]
    cond_features: Float[Array, "buckets cond dim grid grid"]
    cond_pointers: Float[Array, "buckets cond slots dim"]
    cond_frame: Int[Array, "buckets cond"]
    recent_memory: Float[Array, "buckets recent dim grid grid"]
    recent_features: Float[Array, "buckets recent dim grid grid"]
    recent_frame: Int[Array, "buckets recent"]
    recent_is_cond: Bool[Array, "buckets recent"]
    pointers: Float[Array, "buckets pointer slots dim"]
    pointer_frame: Int[Array, "buckets pointer"]
    last_masks: Float[Array, "buckets slots height width"]
    last_scores: Float[Array, "buckets slots"]


class TrackOutput(eqx.Module):
    """One propagated frame: for each slot, the best of its three masks."""

    mask_logits: Float[Array, "buckets slots height width"]
    object_score_logits: Float[Array, "buckets slots"]
    iou: Float[Array, "buckets slots"]
    pointers: Float[Array, "buckets slots dim"]


class Association(eqx.Module):
    """Detections matched against every slot, with ``tracks = buckets * slots``.

    Track axes are flattened over buckets. Only kept detections and active
    slots have overlaps; every other pair is 0.
    """

    keep: Bool[Array, " queries"]
    iom: Float[Array, "queries tracks"]
    matches: Bool[Array, "queries tracks"]
    high_iom: Bool[Array, "queries tracks"]
    ambiguous: Bool[Array, "queries tracks"]
    new_detection: Bool[Array, " queries"]
    confident_detection: Bool[Array, " queries"]
    confident_track: Bool[Array, " tracks"]
    track_iou: Float[Array, "tracks tracks"]


class CommitPlan(eqx.Module):
    """What the host decided for one frame.

    ``state_group`` assigns each bucket to the state it was created with.
    ``reencode`` must list every bucket of a state that received objects.
    ``source_size`` is the clip's frame size, static so that it compiles once.
    ``decoded`` marks added slots whose ``new_masks`` entry is already the
    object's mask logits, with its pointer and score in ``decoded_pointers`` and
    ``decoded_scores``; other added slots are decoded from a mask prompt.
    """

    frame_index: Int[Array, ""]
    add: Bool[Array, "buckets slots"]
    new_masks: Float[Array, "buckets slots height width"]
    decoded: Bool[Array, "buckets slots"]
    decoded_pointers: Float[Array, "buckets slots dim"]
    decoded_scores: Float[Array, "buckets slots"]
    recondition: Bool[Array, "buckets slots"]
    recondition_masks: Float[Array, "buckets slots height width"]
    suppress: Bool[Array, "buckets slots"]
    remove: Bool[Array, "buckets slots"]
    release: Bool[Array, " buckets"]
    reencode: Bool[Array, " buckets"]
    state_group: Int[Array, " buckets"]
    object_ids: Int[Array, "buckets slots"]
    source_size: tuple[int, int] = eqx.field(static=True)

    @classmethod
    def empty(
        cls,
        state: TrackerState,
        frame_index: Int[Array, ""],
        source_size: tuple[int, int],
    ) -> CommitPlan:
        """Return a plan that changes no object, with all fields sized for ``state``."""
        no_masks = jnp.zeros_like(state.last_masks)
        no_slots = jnp.zeros_like(state.slot_status, dtype=bool)
        no_buckets = jnp.zeros_like(state.bucket_valid)
        return cls(
            frame_index=jnp.asarray(frame_index, jnp.int32),
            add=no_slots,
            new_masks=no_masks,
            decoded=no_slots,
            decoded_pointers=jnp.zeros_like(state.pointers[:, 0]),
            decoded_scores=jnp.zeros_like(state.last_scores),
            recondition=no_slots,
            recondition_masks=no_masks,
            suppress=no_slots,
            remove=no_slots,
            release=no_buckets,
            reencode=no_buckets,
            state_group=jnp.arange(state.slot_status.shape[0], dtype=jnp.int32),
            object_ids=jnp.zeros_like(state.slot_status, dtype=jnp.int32),
            source_size=source_size,
        )


def _empty_buckets(template: TrackerState, num_buckets: int) -> TrackerState:
    """Return ``num_buckets`` cleared buckets shaped like those of ``template``."""
    zeros = jax.tree.map(
        lambda leaf: jnp.zeros((num_buckets, *leaf.shape[1:]), leaf.dtype), template
    )
    no_frames = eqx.tree_at(
        lambda s: (s.cond_frame, s.recent_frame, s.pointer_frame),
        zeros,
        replace_fn=lambda leaf: leaf - 1,
    )
    return eqx.tree_at(
        lambda s: s.last_masks,
        no_frames,
        replace_fn=lambda leaf: leaf + _NO_OBJECT_LOGIT,
    )


def pad_tracker_buckets(state: TrackerState, num_buckets: int) -> TrackerState:
    """Grow the bucket axis to the power of two ``num_buckets``, with cleared buckets."""
    current = state.slot_status.shape[0]
    if num_buckets <= current or num_buckets & (num_buckets - 1):
        raise ValueError(
            f"num_buckets must be a power of two above {current}, got {num_buckets}."
        )
    return jax.tree.map(
        lambda leaf, extra: jnp.concatenate((leaf, extra), axis=0),
        state,
        _empty_buckets(state, num_buckets - current),
    )


def _where_bucket(flags: jax.Array, a: TrackerState, b: TrackerState) -> TrackerState:
    def select(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.where(flags.reshape(-1, *(1,) * (x.ndim - 1)), x, y)

    return jax.tree.map(select, a, b)


def _selected_memory_frames(
    bucket: TrackerState,
) -> tuple[Bool[Array, " cond"], Bool[Array, " recent"], Bool[Array, " pointer"]]:
    """Tell which stored frames one bucket reads.

    All stored conditioning frames are read. The ring buffers supply the other
    frames, so a conditioning frame that is no longer among the stored ones
    still counts as a recent frame.
    """

    def is_selected_cond(frames: jax.Array) -> jax.Array:
        return jnp.any(frames[:, None] == bucket.cond_frame[None, :], axis=-1)

    cond = bucket.cond_frame >= 0
    recent = (bucket.recent_frame >= 0) & ~(
        bucket.recent_is_cond & is_selected_cond(bucket.recent_frame)
    )
    pointer = (bucket.pointer_frame >= 0) & ~is_selected_cond(bucket.pointer_frame)
    return cond, recent, pointer


def _shrunk_slots(
    logits: Float[Array, "tracks height width"], order: Int[Array, " tracks"]
) -> Bool[Array, " tracks"]:
    """Mark the tracks left with under 30% of their area after per-pixel overlap."""
    ranked = logits[order] > 0
    kept = remove_overlaps(ranked)
    before = ranked.reshape(ranked.shape[0], -1).sum(axis=-1)
    after = kept.reshape(kept.shape[0], -1).sum(axis=-1)
    shrunk = (before > 0) & (after < _MIN_KEPT_AREA_FRACTION * before)
    return jnp.zeros(order.shape[0], bool).at[order].set(shrunk)


def _keep_per_group(
    masks: Bool[Array, "tracks height width"],
    order: Int[Array, " tracks"],
    groups: Int[Array, " tracks"],
    num_groups: int,
) -> Bool[Array, "tracks height width"]:
    """Give each pixel to the highest-priority mask of each group."""
    ranked = masks[order]
    ranked_groups = groups[order]

    def visit(kept: jax.Array, group: jax.Array) -> tuple[jax.Array, None]:
        in_group = (ranked_groups == group)[:, None, None]
        return kept | remove_overlaps(ranked & in_group), None

    kept, _ = jax.lax.scan(
        visit, jnp.zeros_like(ranked), jnp.arange(num_groups, dtype=groups.dtype)
    )
    return jnp.zeros_like(kept).at[order].set(kept)


def _priority_order(
    object_score_logits: Float[Array, "..."], object_ids: Int[Array, "..."]
) -> Int[Array, " tracks"]:
    """Rank objects by decreasing tracker probability, ties to the lower ID."""
    probabilities = jax.nn.sigmoid(object_score_logits.astype(jnp.float32)).ravel()
    return jnp.lexsort((object_ids.ravel(), -probabilities))


def associate(
    detections: DetectorOutput, tracks: TrackOutput, state: TrackerState
) -> Association:
    """Compute the fixed-shape overlaps and flags the host policy decides from."""
    probabilities = jax.nn.sigmoid(detections.logits.astype(jnp.float32))
    detection_masks = detections.mask_logits > 0
    after_nms = greedy_iom_nms(
        detection_masks,
        probabilities,
        probabilities > _DETECTION_KEEP_PROBABILITY,
        _NMS_IOM,
    )
    centers = detections.boxes[..., :2].astype(jnp.float32)
    inside = jnp.all(
        (centers > _BOUNDARY_MARGIN) & (centers < 1.0 - _BOUNDARY_MARGIN), axis=-1
    )
    keep = after_nms & inside

    active = (state.slot_status == SLOT_ACTIVE).ravel()
    height, width = tracks.mask_logits.shape[-2:]
    track_masks = (tracks.mask_logits.reshape(-1, height, width) > 0) & active[
        :, None, None
    ]
    iom = jnp.where(
        keep[:, None] & active[None, :],
        iom_matrix(detection_masks, track_masks),
        0.0,
    )
    matches = iom >= _MATCH_IOM
    high_iom = iom >= _HIGH_IOM
    per_detection = jnp.sum(high_iom, axis=1)[:, None]
    per_track = jnp.sum(high_iom, axis=0)[None, :]
    new_detection = (
        (probabilities >= _NEW_OBJECT_PROBABILITY)
        & ~jnp.any(matches, axis=1)
        & (keep | ~jnp.any(active))
    )
    track_probabilities = jax.nn.sigmoid(
        tracks.object_score_logits.astype(jnp.float32)
    ).ravel()
    return Association(
        keep=keep,
        iom=iom,
        matches=matches,
        high_iom=high_iom,
        ambiguous=high_iom & ((per_detection > 1) | (per_track > 1)),
        new_detection=new_detection,
        confident_detection=probabilities >= _CONFIDENT_DETECTION_PROBABILITY,
        confident_track=track_probabilities > _CONFIDENT_TRACK_PROBABILITY,
        track_iou=iou_matrix(track_masks, track_masks),
    )


class Sam3Tracker(eqx.Module):
    """Modules of the memory-based tracker and its device steps."""

    image_position: ImagePositionEncoding
    memory_encoder: MemoryEncoder
    memory_attention: MemoryAttention
    mask_decoder: MultiplexMaskDecoder
    pointer_proj: MLP
    pointer_time_proj: eqx.nn.Linear
    memory_time_embed: Float[Array, "times dim"]

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
        memory_start_channels: int,
        memory_fuser_depth: int,
    ):
        (
            key_position,
            key_memory_encoder,
            key_attention,
            key_decoder,
            key_pointer,
            key_pointer_time,
            key_time_embed,
        ) = jr.split(key, 7)
        self.image_position = ImagePositionEncoding(
            dim, grid, image_size, key=key_position
        )
        self.memory_encoder = MemoryEncoder(
            dim,
            2 * num_slots,
            memory_start_channels,
            memory_fuser_depth,
            key=key_memory_encoder,
        )
        self.memory_attention = MemoryAttention(
            dim, memory_heads, ffn_features, memory_depth, grid, key=key_attention
        )
        self.mask_decoder = MultiplexMaskDecoder(
            dim, num_heads, ffn_features, decoder_depth, num_slots, key=key_decoder
        )
        self.pointer_proj = MLP(dim, dim, dim, 3, key=key_pointer)
        self.pointer_time_proj = eqx.nn.Linear(dim, dim, key=key_pointer_time)
        self.memory_time_embed = jr.normal(key_time_embed, (_NUM_TIME_EMBEDDINGS, dim))

    def empty_state(self, num_buckets: int = 1) -> TrackerState:
        """Return a state of ``num_buckets`` cleared buckets."""
        dim = self.pointer_time_proj.out_features
        dtype = self.memory_time_embed.dtype
        grid = self.image_position.grid
        slots = self.mask_decoder.num_slots
        mask_size = _MASK_SCALE * grid
        template = TrackerState(
            bucket_valid=jnp.zeros((0,), bool),
            slot_status=jnp.zeros((0, slots), jnp.int8),
            cond_memory=jnp.zeros((0, _NUM_COND_FRAMES, dim, grid, grid), dtype),
            cond_features=jnp.zeros((0, _NUM_COND_FRAMES, dim, grid, grid), dtype),
            cond_pointers=jnp.zeros((0, _NUM_COND_FRAMES, slots, dim), dtype),
            cond_frame=jnp.zeros((0, _NUM_COND_FRAMES), jnp.int32),
            recent_memory=jnp.zeros((0, _NUM_RECENT_FRAMES, dim, grid, grid), dtype),
            recent_features=jnp.zeros((0, _NUM_RECENT_FRAMES, dim, grid, grid), dtype),
            recent_frame=jnp.zeros((0, _NUM_RECENT_FRAMES), jnp.int32),
            recent_is_cond=jnp.zeros((0, _NUM_RECENT_FRAMES), bool),
            pointers=jnp.zeros((0, _NUM_POINTER_FRAMES, slots, dim), dtype),
            pointer_frame=jnp.zeros((0, _NUM_POINTER_FRAMES), jnp.int32),
            last_masks=jnp.zeros((0, slots, mask_size, mask_size), jnp.float32),
            last_scores=jnp.zeros((0, slots), jnp.float32),
        )
        return _empty_buckets(template, num_buckets)

    def _memory_bank(
        self,
        bucket: TrackerState,
        frame_index: Int[Array, ""],
        pointer_divisor: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, Int[Array, ""], Int[Array, ""]]:
        """Pack one bucket's memory, valid entries first.

        Returns the stored image features, the memory, their position encodings,
        the number of valid spatial keys and the number of all valid keys.
        """
        dtype = bucket.cond_memory.dtype
        dim, grid = bucket.cond_memory.shape[-3], bucket.cond_memory.shape[-1]
        cond, recent, pointer = _selected_memory_frames(bucket)

        # Closer conditioning frames take lower indices, closer recent frames higher.
        cond_time = jnp.clip(frame_index - bucket.cond_frame, 0, _NUM_RECENT_FRAMES)
        recent_time = jnp.clip(
            _NUM_RECENT_FRAMES - (frame_index - bucket.recent_frame),
            0,
            _NUM_RECENT_FRAMES,
        )
        time_embed = self.memory_time_embed[jnp.concatenate((cond_time, recent_time))]
        position = sine_position_encoding(grid, grid, dim // 2).astype(dtype)
        features = jnp.concatenate((bucket.cond_features, bucket.recent_features))
        memory = jnp.concatenate((bucket.cond_memory, bucket.recent_memory))
        spatial_position = position + time_embed[:, :, None, None]

        distance = (
            frame_index - jnp.concatenate((bucket.cond_frame, bucket.pointer_frame))
        ).astype(jnp.float32)
        time_encoding = sine_coordinates(distance / pointer_divisor, dim).astype(dtype)
        pointer_time = jax.vmap(self.pointer_time_proj)(time_encoding)
        object_pointers = jnp.concatenate((bucket.cond_pointers, bucket.pointers))
        pointer_position = jnp.broadcast_to(
            pointer_time[:, None, :], object_pointers.shape
        )

        frame_valid = jnp.concatenate((cond, recent))
        pointer_valid = jnp.concatenate((cond, pointer))
        slots = object_pointers.shape[1]
        valid = jnp.concatenate(
            (
                jnp.repeat(frame_valid, grid * grid),
                jnp.repeat(pointer_valid, slots),
            )
        )
        order = jnp.argsort(~valid, stable=True)

        def pack(spatial: jax.Array, pointers: jax.Array) -> jax.Array:
            tokens = jax.vmap(flatten_map)(spatial).reshape(-1, dim)
            return jnp.concatenate((tokens, pointers.reshape(-1, dim)))[order]

        return (
            pack(features, jnp.zeros_like(object_pointers)),
            pack(memory, object_pointers),
            pack(spatial_position, pointer_position),
            grid * grid * jnp.sum(frame_valid),
            jnp.sum(valid),
        )

    def propagate(
        self,
        state: TrackerState,
        features: ImageFeatures,
        frame_index: Int[Array, ""],
        *,
        num_frames: int | None = None,
    ) -> TrackOutput:
        """Track every bucket into the frame whose features are given."""
        high, middle, low = features.propagation
        grid = low.shape[-1]
        known_frames = (
            _POINTER_DISTANCE_FRAMES
            if num_frames is None
            else min(num_frames, _POINTER_DISTANCE_FRAMES)
        )
        pointer_divisor = max(known_frames - 1, 1)
        tokens = flatten_map(low)
        token_pos = flatten_map(
            sine_position_encoding(grid, grid, low.shape[0] // 2)
        ).astype(low.dtype)
        image_pos = self.image_position(low.dtype)
        high_res = self.mask_decoder.upsampler.project((high, middle))

        def one_bucket(bucket: TrackerState) -> TrackOutput:
            (
                memory_image,
                memory,
                memory_image_pos,
                num_spatial_keys,
                key_lengths,
            ) = self._memory_bank(bucket, frame_index, pointer_divisor)
            attended = self.memory_attention(
                tokens,
                token_pos,
                memory_image,
                memory,
                memory_image_pos,
                num_spatial_keys=num_spatial_keys,
                key_lengths=key_lengths,
            )
            active = bucket.slot_status == SLOT_ACTIVE
            out = self.mask_decoder(
                unflatten_map(attended, grid), image_pos, high_res, active
            )
            best = jnp.argmax(out.iou.astype(jnp.float32), axis=-1)[:, None]
            tokens_best = jnp.take_along_axis(out.mask_tokens, best[..., None], axis=1)[
                :, 0
            ]
            pointers = self.pointer_proj(tokens_best)
            return TrackOutput(
                mask_logits=jnp.take_along_axis(
                    out.mask_logits, best[..., None, None], axis=1
                )[:, 0],
                object_score_logits=out.object_score_logits,
                iou=jnp.take_along_axis(out.iou, best, axis=1)[:, 0],
                pointers=jnp.where(active[:, None], pointers, 0).astype(pointers.dtype),
            )

        return jax.vmap(one_bucket)(state)

    def _encode_memory(
        self,
        low: Float[Array, "dim grid grid"],
        mask_logits: Float[Array, "slots height width"],
        conditioning: Bool[Array, " slots"],
    ) -> Float[Array, "dim grid grid"]:
        signal = jax.nn.sigmoid(mask_logits.astype(jnp.float32))
        flags = jnp.broadcast_to(conditioning[:, None, None], signal.shape)
        return self.memory_encoder(
            low, jnp.concatenate((signal, flags.astype(jnp.float32)), axis=0)
        )

    def decode_object(
        self,
        interactive: InteractiveHead,
        features: ImageFeatures,
        prompt: InteractivePrompt,
    ) -> tuple[Float[Array, "height width"], Float[Array, " dim"], Float[Array, ""]]:
        """Return the prompted object's mask logits, pointer and object score."""
        output, tokens = interactive.predict_with_tokens(features, prompt)
        return (
            output.mask_logits[output.selected],
            self.pointer_proj(tokens[output.selected]),
            output.object_score_logit,
        )

    def _new_objects(
        self,
        interactive: InteractiveHead,
        features: ImageFeatures,
        masks: Float[Array, "buckets slots height width"],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Turn detector masks into conditioning masks, pointers and object scores."""

        def one_object(mask: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            prompt = InteractivePrompt.create(mask_input=mask, mask_size=mask.shape[-1])
            return self.decode_object(interactive, features, prompt)

        return jax.vmap(jax.vmap(one_object))(masks)

    def _reencode_buckets(
        self,
        low: Float[Array, "dim grid grid"],
        logits: Float[Array, "buckets slots height width"],
        scores: Float[Array, "buckets slots"],
        object_ids: Int[Array, "buckets slots"],
        conditioning: Bool[Array, "buckets slots"],
        plan: CommitPlan,
    ) -> Float[Array, "buckets dim grid grid"]:
        """Encode memory from masks made non-overlapping at the source frame size."""
        buckets, slots = logits.shape[:2]
        resized = resize_bilinear(
            logits.reshape(-1, *logits.shape[2:]), plan.source_size
        )
        kept = _keep_per_group(
            resized > 0,
            _priority_order(scores, object_ids),
            jnp.repeat(plan.state_group, slots),
            buckets,
        )
        clamped = jnp.where(
            kept, resized, jnp.minimum(resized, _OVERLAP_LOSER_LOGIT)
        ).reshape(buckets, slots, *plan.source_size)
        return jax.vmap(self._encode_memory, in_axes=(None, 0, 0))(
            low, clamped, conditioning
        )

    def commit(
        self,
        interactive: InteractiveHead,
        state: TrackerState,
        features: ImageFeatures,
        tracks: TrackOutput,
        plan: CommitPlan,
    ) -> TrackerState:
        """Apply the host's plan to one frame and store its memory.

        Released buckets are cleared first, so their slots can take new objects
        in the same step. Objects are only added to empty slots, so a removed
        slot stays removed until its bucket is released.
        """
        low = features.propagation[2]
        num_buckets = state.slot_status.shape[0]
        image_size = self.image_position.image_size
        state = _where_bucket(plan.release, _empty_buckets(state, num_buckets), state)
        occupied = state.slot_status == SLOT_ACTIVE
        add = plan.add & (state.slot_status == SLOT_EMPTY)
        recondition = plan.recondition & occupied
        remove = plan.remove & occupied
        tracked = occupied & ~remove
        no_object = jnp.asarray(_NO_OBJECT_LOGIT, jnp.float32)

        logits = jnp.where(
            recondition[..., None, None], plan.recondition_masks, tracks.mask_logits
        )
        logits = jnp.where(
            (plan.suppress & occupied)[..., None, None], no_object, logits
        )
        logits = jnp.where(tracked[..., None, None], logits, no_object)

        height, width = logits.shape[-2:]
        shrunk = _shrunk_slots(
            logits.reshape(-1, height, width),
            _priority_order(tracks.object_score_logits, plan.object_ids),
        ).reshape(tracked.shape)
        memory_logits = jnp.where(
            shrunk[..., None, None],
            jnp.minimum(logits, _OVERLAP_LOSER_LOGIT),
            logits,
        )
        upsampled = resize_bilinear(memory_logits, (image_size, image_size))
        memory = jax.vmap(self._encode_memory, in_axes=(None, 0, 0))(
            low, upsampled, recondition
        )

        def no_new_objects() -> tuple[jax.Array, jax.Array, jax.Array]:
            return (
                jnp.zeros_like(logits),
                jnp.zeros_like(tracks.pointers),
                jnp.zeros_like(tracks.object_score_logits),
            )

        new_logits, new_pointers, new_scores = jax.lax.cond(
            jnp.any(add & ~plan.decoded),
            lambda: self._new_objects(interactive, features, plan.new_masks),
            no_new_objects,
        )
        new_logits = jnp.where(
            plan.decoded[..., None, None], plan.new_masks, new_logits
        )
        new_pointers = jnp.where(
            plan.decoded[..., None], plan.decoded_pointers, new_pointers
        )
        new_scores = jnp.where(plan.decoded, plan.decoded_scores, new_scores)
        final_logits = jnp.where(add[..., None, None], new_logits, logits)
        final_scores = jnp.where(add, new_scores, tracks.object_score_logits)
        final_active = tracked | add
        final_pointers = jnp.where(add[..., None], new_pointers, tracks.pointers)
        final_pointers = jnp.where(final_active[..., None], final_pointers, 0).astype(
            tracks.pointers.dtype
        )
        conditioning = add | recondition

        reencode = plan.reencode | jnp.any(add, axis=-1)
        memory = jax.lax.cond(
            jnp.any(reencode),
            lambda: jnp.where(
                reencode.reshape(-1, 1, 1, 1),
                self._reencode_buckets(
                    low,
                    final_logits,
                    final_scores,
                    plan.object_ids,
                    conditioning,
                    plan,
                ),
                memory,
            ),
            lambda: memory,
        )

        status = jnp.where(add, SLOT_ACTIVE, state.slot_status)
        status = jnp.where(remove, SLOT_REMOVED, status).astype(jnp.int8)
        valid = state.bucket_valid | jnp.any(add, axis=-1)
        frame = plan.frame_index

        def write(
            bucket: TrackerState,
            bucket_memory: jax.Array,
            pointers: jax.Array,
            is_cond: Bool[Array, ""],
            bucket_valid: Bool[Array, ""],
            bucket_status: jax.Array,
            bucket_logits: jax.Array,
            bucket_scores: jax.Array,
        ) -> TrackerState:
            recent_slot = frame % _NUM_RECENT_FRAMES
            pointer_slot = frame % _NUM_POINTER_FRAMES
            # A frame committed twice replaces its own conditioning entry.
            same_frame = bucket.cond_frame == frame
            cond_slot = jnp.where(
                jnp.any(same_frame),
                jnp.argmax(same_frame),
                jnp.argmin(bucket.cond_frame),
            )
            store_cond = bucket_valid & is_cond

            def stored(
                buffer: jax.Array, slot: jax.Array, value, write_it
            ) -> jax.Array:
                return buffer.at[slot].set(jnp.where(write_it, value, buffer[slot]))

            shown = bucket_status == SLOT_ACTIVE
            return TrackerState(
                bucket_valid=bucket_valid,
                slot_status=bucket_status,
                cond_memory=stored(
                    bucket.cond_memory, cond_slot, bucket_memory, store_cond
                ),
                cond_features=stored(bucket.cond_features, cond_slot, low, store_cond),
                cond_pointers=stored(
                    bucket.cond_pointers, cond_slot, pointers, store_cond
                ),
                cond_frame=stored(bucket.cond_frame, cond_slot, frame, store_cond),
                recent_memory=stored(
                    bucket.recent_memory, recent_slot, bucket_memory, bucket_valid
                ),
                recent_features=stored(
                    bucket.recent_features, recent_slot, low, bucket_valid
                ),
                recent_frame=stored(
                    bucket.recent_frame, recent_slot, frame, bucket_valid
                ),
                # A frame committed twice stays conditioning, or it would be read twice.
                recent_is_cond=stored(
                    bucket.recent_is_cond,
                    recent_slot,
                    is_cond | jnp.any(same_frame),
                    bucket_valid,
                ),
                pointers=stored(bucket.pointers, pointer_slot, pointers, bucket_valid),
                pointer_frame=stored(
                    bucket.pointer_frame, pointer_slot, frame, bucket_valid
                ),
                last_masks=jnp.where(
                    shown[:, None, None], bucket_logits, _NO_OBJECT_LOGIT
                ),
                last_scores=jnp.where(shown, bucket_scores, 0.0),
            )

        return jax.vmap(write)(
            state,
            memory,
            final_pointers,
            jnp.any(conditioning, axis=-1),
            valid,
            status,
            final_logits,
            final_scores,
        )


__all__ = [
    "Association",
    "CommitPlan",
    "TrackOutput",
    "TrackerState",
    "associate",
    "pad_tracker_buckets",
]
