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
    sine_position_encoding,
    unflatten_map,
)
from equimo.vision.models.sam3.decoding import (
    greedy_iom_nms,
    iom_matrix,
    iou_matrix,
)
from equimo.vision.models.sam3.detector import DetectorOutput
from equimo.vision.models.sam3.encoder import ImageFeatures
from equimo.vision.ops import resize_bilinear, resize_bilinear_antialias
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

# Thresholds of ``facebookresearch/sam3`` at ``f1647e8``, under ``sam3/model/``:
# ``sam3_multiplex_base.py:run_backbone_and_detection`` keeps detections and
# ``sam3_video_base.py:_suppress_detections_close_to_boundary`` filters boxes;
# ``model_builder.py:det_nms_thresh`` sets the NMS threshold.
_DETECTION_KEEP_PROBABILITY = 0.4
_NMS_IOM = 0.1
_BOUNDARY_MARGIN = 0.025
# ``sam3_video_base.py:_associate_det_trk_compilable``.
_MATCH_IOM = 0.1
_HIGH_IOM = 0.5
_NEW_OBJECT_PROBABILITY = 0.65
_CONFIDENT_DETECTION_PROBABILITY = 0.8
# ``sam3_multiplex_base.py:_recondition_masklets``.
_CONFIDENT_TRACK_PROBABILITY = 0.8

# ``video_tracking_multiplex_demo.py:_suppress_object_pw_area_shrinkage`` and
# ``video_tracking_multiplex.py:_apply_non_overlapping_constraints``.
_OVERLAP_LOSER_LOGIT = -10.0
_MIN_KEPT_AREA_FRACTION = 0.3
# ``video_tracking_multiplex.py:_use_mask_as_output`` and
# ``video_tracking_multiplex_demo.py:add_new_masks``.
_MASK_INPUT_SCORE_LOGIT = 10.0
_VIDEO_MASK_LOGIT = 1024.0


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
    slots have overlaps; every other pair is 0. Rows and columns with two or
    more overlaps of at least 0.5 are zeroed for ``matches`` and
    ``unique_high_iom``, but not for ``iom`` and ``high_iom``.
    ``reconditioned_iou`` and ``reconditioned_nonempty`` hold what the track
    masks would give after reconditioning, which only applies to some frames.
    """

    keep: Bool[Array, " queries"]
    iom: Float[Array, "queries tracks"]
    matches: Bool[Array, "queries tracks"]
    high_iom: Bool[Array, "queries tracks"]
    unique_high_iom: Bool[Array, "queries tracks"]
    new_detection: Bool[Array, " queries"]
    confident_detection: Bool[Array, " queries"]
    confident_track: Bool[Array, " tracks"]
    track_nonempty: Bool[Array, " tracks"]
    track_iou: Float[Array, "tracks tracks"]
    reconditioned_iou: Float[Array, "tracks tracks"]
    reconditioned_nonempty: Bool[Array, " tracks"]


class CommitPlan(eqx.Module):
    """What the host decided for one frame.

    ``state_group`` assigns each bucket to the state it was created with.
    ``reencode`` must list every bucket of a state that received objects, and
    ``new_state`` the buckets of a state created on this frame, which store it
    as their conditioning frame. ``conditioning`` flags the slots born on this
    frame, added to or reconditioned on it. ``source_size`` is the clip's frame
    size, static so that it compiles once.
    """

    frame_index: Int[Array, ""]
    add: Bool[Array, "buckets slots"]
    new_masks: Float[Array, "buckets slots height width"]
    recondition: Bool[Array, "buckets slots"]
    recondition_masks: Float[Array, "buckets slots height width"]
    suppress: Bool[Array, "buckets slots"]
    remove: Bool[Array, "buckets slots"]
    release: Bool[Array, " buckets"]
    reencode: Bool[Array, " buckets"]
    new_state: Bool[Array, " buckets"]
    conditioning: Bool[Array, "buckets slots"]
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
            recondition=no_slots,
            recondition_masks=no_masks,
            suppress=no_slots,
            remove=no_slots,
            release=no_buckets,
            reencode=no_buckets,
            new_state=no_buckets,
            conditioning=no_slots,
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


def remove_tracker_slot(
    state: TrackerState,
    bucket: int,
    slot: int,
    released_buckets: tuple[int, ...],
) -> TrackerState:
    """Mark one slot as removed and clear the buckets left without objects."""
    status = state.slot_status.at[bucket, slot].set(SLOT_REMOVED)
    removed = eqx.tree_at(lambda s: s.slot_status, state, status)
    if not released_buckets:
        return removed
    release = (
        jnp.zeros(status.shape[0], bool).at[jnp.asarray(released_buckets)].set(True)
    )
    return _where_bucket(release, _empty_buckets(removed, status.shape[0]), removed)


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


def _pointer_time_encoding(
    distances: Float[Array, "..."], dim: int
) -> Float[Array, "... dim"]:
    """Encode frame distances, sines first and cosines second, without a 2 pi scale."""
    half = dim // 2
    index = jnp.arange(half, dtype=jnp.float32)
    denominators = 10000.0 ** (2.0 * jnp.floor(index / 2.0) / half)
    scaled = distances.astype(jnp.float32)[..., None] / denominators
    return jnp.concatenate((jnp.sin(scaled), jnp.cos(scaled)), axis=-1)


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


def _suppress_area_shrinkage(
    logits: Float[Array, "buckets slots height width"],
    present: Bool[Array, "buckets slots"],
    object_ids: Int[Array, "buckets slots"],
) -> Float[Array, "buckets slots height width"]:
    """Clamp the masks that lose most of their area to a per-pixel overlap.

    Overlap is resolved over every object at once, by highest logit with ties
    to the lower ID. An object keeping under 30% of its area, or starting
    empty, is clamped everywhere. Nothing changes for a single object.
    """
    flat = logits.reshape(-1, *logits.shape[-2:])
    rows = present.ravel()
    order = jnp.lexsort((object_ids.ravel(), ~rows))
    ranked = flat[order]
    winner = jnp.argmax(ranked, axis=0)
    wins = winner == jnp.arange(order.shape[0])[:, None, None]
    after_overlap = jnp.where(wins, ranked, jnp.minimum(ranked, _OVERLAP_LOSER_LOGIT))
    before = jnp.sum(ranked > 0, axis=(-2, -1))
    after = jnp.sum(after_overlap > 0, axis=(-2, -1))
    kept = after / jnp.maximum(before, 1) >= _MIN_KEPT_AREA_FRACTION
    kept = kept | (jnp.sum(rows) <= 1)
    clamped = jnp.where(
        kept[:, None, None], ranked, jnp.minimum(ranked, _OVERLAP_LOSER_LOGIT)
    )
    restored = jnp.zeros_like(clamped).at[order].set(clamped)
    return restored.reshape(logits.shape)


def _non_overlapping_per_state(
    logits: Float[Array, "tracks height width"],
    groups: Int[Array, " tracks"],
    num_groups: int,
) -> Float[Array, "tracks height width"]:
    """Clamp, in each state, every mask that is not the highest at a pixel.

    Ties go to the lower track index, which is the earlier object.
    """
    rows = jnp.arange(logits.shape[0])[:, None, None]

    def visit(keep: jax.Array, group: jax.Array) -> tuple[jax.Array, None]:
        in_group = (groups == group)[:, None, None]
        winner = jnp.argmax(jnp.where(in_group, logits, -jnp.inf), axis=0)
        return keep | ((winner == rows) & in_group), None

    keep, _ = jax.lax.scan(
        visit, jnp.zeros_like(logits, bool), jnp.arange(num_groups, dtype=groups.dtype)
    )
    return jnp.where(keep, logits, jnp.minimum(logits, _OVERLAP_LOSER_LOGIT))


def _masks_through_source_size(
    masks: Bool[Array, "buckets slots height width"],
    selected: Bool[Array, "buckets slots"],
    state_group: Int[Array, " buckets"],
    source_size: tuple[int, int],
    mask_size: int,
) -> Float[Array, "buckets slots mask_size mask_size"]:
    """Turn input masks into low-resolution logits through the clip's frame size.

    Masks given to a state together are made disjoint at the source size: a
    pixel claimed by another mask of the same state is empty for every mask.
    """
    buckets, slots = selected.shape
    rows = masks.reshape(-1, *masks.shape[-2:])
    groups = jnp.repeat(state_group, slots)
    source = resize_bilinear_antialias(rows.astype(jnp.float32), source_size) > 0.5
    source = source & selected.ravel()[:, None, None]
    per_state = jax.ops.segment_sum(
        source.astype(jnp.int32), groups, num_segments=buckets
    )
    others = per_state[groups] > source.astype(jnp.int32)
    logits = jnp.where(source & ~others, _VIDEO_MASK_LOGIT, -_VIDEO_MASK_LOGIT)
    low = resize_bilinear_antialias(logits, (mask_size, mask_size))
    return low.reshape(buckets, slots, mask_size, mask_size)


def _store_frame(
    bucket: TrackerState,
    frame: Int[Array, ""],
    low: Float[Array, "dim grid grid"],
    memory: jax.Array,
    pointers: jax.Array,
    is_new_state: Bool[Array, ""],
    valid: Bool[Array, ""],
    status: jax.Array,
    logits: jax.Array,
    scores: jax.Array,
) -> TrackerState:
    """Write one frame's memory, pointers and masks into a bucket."""
    recent_slot = frame % _NUM_RECENT_FRAMES
    pointer_slot = frame % _NUM_POINTER_FRAMES
    # A frame committed twice replaces its own conditioning entry.
    same_frame = bucket.cond_frame == frame
    cond_slot = jnp.where(
        jnp.any(same_frame), jnp.argmax(same_frame), jnp.argmin(bucket.cond_frame)
    )
    is_cond = is_new_state | jnp.any(same_frame)
    store_cond = valid & is_cond

    def stored(buffer: jax.Array, slot: jax.Array, value, write_it) -> jax.Array:
        return buffer.at[slot].set(jnp.where(write_it, value, buffer[slot]))

    return TrackerState(
        bucket_valid=valid,
        slot_status=status,
        cond_memory=stored(bucket.cond_memory, cond_slot, memory, store_cond),
        cond_features=stored(bucket.cond_features, cond_slot, low, store_cond),
        cond_pointers=stored(bucket.cond_pointers, cond_slot, pointers, store_cond),
        cond_frame=stored(bucket.cond_frame, cond_slot, frame, store_cond),
        recent_memory=stored(bucket.recent_memory, recent_slot, memory, valid),
        recent_features=stored(bucket.recent_features, recent_slot, low, valid),
        recent_frame=stored(bucket.recent_frame, recent_slot, frame, valid),
        # A frame committed twice stays conditioning, or it would be read twice.
        recent_is_cond=stored(bucket.recent_is_cond, recent_slot, is_cond, valid),
        pointers=stored(bucket.pointers, pointer_slot, pointers, valid),
        pointer_frame=stored(bucket.pointer_frame, pointer_slot, frame, valid),
        last_masks=logits,
        last_scores=scores,
    )


def detection_keep(
    detections: DetectorOutput,
) -> tuple[Bool[Array, " queries"], Bool[Array, " queries"]]:
    """Return which detections pass NMS, and which also pass the boundary filter."""
    probabilities = jax.nn.sigmoid(detections.logits.astype(jnp.float32))
    after_nms = greedy_iom_nms(
        detections.mask_logits > 0,
        probabilities,
        probabilities > _DETECTION_KEEP_PROBABILITY,
        _NMS_IOM,
    )
    centers = detections.boxes[..., :2].astype(jnp.float32)
    inside = jnp.all(
        (centers > _BOUNDARY_MARGIN) & (centers < 1.0 - _BOUNDARY_MARGIN), axis=-1
    )
    return after_nms, after_nms & inside


def associate_tracks(
    detections: DetectorOutput,
    after_nms: Bool[Array, " queries"],
    keep: Bool[Array, " queries"],
    tracks: TrackOutput,
    state: TrackerState,
) -> Association:
    """Compute the overlaps and flags for detections already filtered by ``keep``."""
    probabilities = jax.nn.sigmoid(detections.logits.astype(jnp.float32))
    detection_masks = detections.mask_logits > 0
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
    high_iom = iom >= _HIGH_IOM
    ambiguous_detection = jnp.sum(high_iom, axis=1) > 1
    ambiguous_track = jnp.sum(high_iom, axis=0) > 1
    unique_iom = jnp.where(
        ambiguous_detection[:, None] | ambiguous_track[None, :], 0.0, iom
    )
    unique_high_iom = unique_iom >= _HIGH_IOM
    new_detection = (
        (probabilities >= _NEW_OBJECT_PROBABILITY)
        & ~jnp.any(iom >= _MATCH_IOM, axis=1)
        & (keep | (after_nms & ~jnp.any(active)))
    )
    confident_detection = (
        (probabilities >= _CONFIDENT_DETECTION_PROBABILITY) & keep & ~new_detection
    )
    track_probabilities = jax.nn.sigmoid(
        tracks.object_score_logits.astype(jnp.float32)
    ).ravel()
    confident_track = track_probabilities > _CONFIDENT_TRACK_PROBABILITY

    eligible = unique_high_iom & confident_detection[:, None]
    reconditioned = jnp.any(eligible, axis=0) & confident_track
    chosen = detection_masks[jnp.argmax(eligible, axis=0)]
    reconditioned_masks = jnp.where(reconditioned[:, None, None], chosen, track_masks)
    return Association(
        keep=keep,
        iom=iom,
        matches=unique_iom >= _MATCH_IOM,
        high_iom=high_iom,
        unique_high_iom=unique_high_iom,
        new_detection=new_detection,
        confident_detection=confident_detection,
        confident_track=confident_track,
        track_nonempty=jnp.any(track_masks, axis=(-2, -1)),
        track_iou=iou_matrix(track_masks, track_masks),
        reconditioned_iou=iou_matrix(reconditioned_masks, reconditioned_masks),
        reconditioned_nonempty=jnp.any(reconditioned_masks, axis=(-2, -1)),
    )


def associate(
    detections: DetectorOutput, tracks: TrackOutput, state: TrackerState
) -> Association:
    """Compute the fixed-shape overlaps and flags the host policy decides from."""
    after_nms, keep = detection_keep(detections)
    return associate_tracks(detections, after_nms, keep, tracks, state)


class Sam3Tracker(eqx.Module):
    """Modules of the memory-based tracker and its device steps."""

    image_position: ImagePositionEncoding
    memory_encoder: MemoryEncoder
    memory_attention: MemoryAttention
    mask_decoder: MultiplexMaskDecoder
    pointer_proj: MLP
    pointer_time_proj: eqx.nn.Linear
    memory_time_embed: Float[Array, "times dim"]
    interactive_pointer_proj: MLP
    no_object_pointer: eqx.nn.Linear
    no_object_memory: Float[Array, "slots dim"]
    interactive_mask_downsample: eqx.nn.Conv2d

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
            key_interactive_pointer,
            key_no_object_pointer,
            key_no_object_memory,
            key_interactive_downsample,
        ) = jr.split(key, 11)
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
        self.interactive_pointer_proj = MLP(
            dim, dim, dim, 3, key=key_interactive_pointer
        )
        self.no_object_pointer = eqx.nn.Linear(dim, dim, key=key_no_object_pointer)
        self.no_object_memory = jr.normal(key_no_object_memory, (num_slots, dim))
        self.interactive_mask_downsample = eqx.nn.Conv2d(
            1, 1, kernel_size=4, stride=4, key=key_interactive_downsample
        )

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

        # Conditioning frames within reach count down from the last entry, and
        # further ones share it. Recent frames count up with their distance.
        cond_distance = frame_index - bucket.cond_frame
        cond_time = jnp.where(
            (cond_distance > 0) & (cond_distance < _NUM_TIME_EMBEDDINGS),
            _NUM_RECENT_FRAMES - cond_distance,
            _NUM_RECENT_FRAMES,
        )
        recent_time = jnp.clip(
            frame_index - bucket.recent_frame - 1, 0, _NUM_RECENT_FRAMES
        )
        time_embed = self.memory_time_embed[jnp.concatenate((cond_time, recent_time))]
        position = sine_position_encoding(grid, grid, dim // 2).astype(dtype)
        features = jnp.concatenate((bucket.cond_features, bucket.recent_features))
        memory = jnp.concatenate((bucket.cond_memory, bucket.recent_memory))
        spatial_position = position + time_embed[:, :, None, None]

        distance = (
            frame_index - jnp.concatenate((bucket.cond_frame, bucket.pointer_frame))
        ).astype(jnp.float32)
        time_encoding = _pointer_time_encoding(distance / pointer_divisor, dim).astype(
            dtype
        )
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
            pointers = jnp.where(
                (out.object_score_logits > 0)[:, None],
                pointers,
                jax.vmap(self.no_object_pointer)(pointers),
            )
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
        present: Bool[Array, " slots"],
        conditioning: Bool[Array, " slots"],
        appearing: Bool[Array, " slots"],
    ) -> Float[Array, "dim grid grid"]:
        """Encode one bucket. Absent slots get a zero mask channel and flag."""
        signal = 2.0 * jax.nn.sigmoid(mask_logits.astype(jnp.float32)) - 1.0
        signal = jnp.where(present[:, None, None], signal, 0.0)
        flags = jnp.broadcast_to(conditioning[:, None, None], signal.shape)
        memory = self.memory_encoder(
            low, jnp.concatenate((signal, flags.astype(jnp.float32)), axis=0)
        )
        absent = jnp.sum(~appearing[:, None] * self.no_object_memory, axis=0)
        return memory + absent[:, None, None].astype(memory.dtype)

    def decode_object(
        self,
        interactive: InteractiveHead,
        features: ImageFeatures,
        prompt: InteractivePrompt,
    ) -> tuple[Float[Array, "height width"], Float[Array, " dim"], Float[Array, ""]]:
        """Return the prompted object's mask logits, pointer and object score."""
        output, tokens = interactive.predict_with_tokens(features, prompt)
        score = output.object_score_logit
        appearing = score > 0
        # Multimask prompts point from the selected token; every other prompt
        # keeps the single-mask token, even if stability selected another mask.
        num_labels = jnp.sum(prompt.point_labels != -1) + 2 * prompt.has_box
        multimask = (num_labels <= 1) & ~prompt.has_mask
        token = tokens[jnp.where(multimask, output.selected, 0)]
        pointer = self.interactive_pointer_proj(token)
        return (
            jnp.where(appearing, output.mask_logits[output.selected], _NO_OBJECT_LOGIT),
            jnp.where(appearing, pointer, self.no_object_pointer(pointer)),
            score,
        )

    def _input_mask_outputs(
        self,
        interactive: InteractiveHead,
        features: ImageFeatures,
        masks: Bool[Array, "buckets slots height width"],
        dtype: jnp.dtype,
    ) -> tuple[Float[Array, "buckets slots dim"], Float[Array, "buckets slots"]]:
        """Return the pointer and object score of each input mask.

        The score only says whether the mask has any pixel.
        """

        def one_mask(mask: jax.Array) -> tuple[jax.Array, jax.Array]:
            prompt_mask = self.interactive_mask_downsample(mask[None].astype(dtype))[0]
            prompt = InteractivePrompt.create(
                mask_input=prompt_mask, mask_size=prompt_mask.shape[-1]
            )
            _, pointer, _ = self.decode_object(interactive, features, prompt)
            appearing = jnp.any(mask)
            score = jnp.where(
                appearing, _MASK_INPUT_SCORE_LOGIT, -_MASK_INPUT_SCORE_LOGIT
            )
            return (
                jnp.where(appearing, pointer, self.no_object_pointer(pointer)),
                score.astype(jnp.float32),
            )

        return jax.vmap(jax.vmap(one_mask))(masks)

    def commit(
        self,
        interactive: InteractiveHead,
        state: TrackerState,
        features: ImageFeatures,
        tracks: TrackOutput,
        plan: CommitPlan,
    ) -> TrackerState:
        """Apply the host's plan to one frame and store its memory.

        Objects removed on this frame stay in the frame's memory and masks, and
        released buckets are cleared afterwards. Only empty slots take new
        objects, so a removed slot stays removed until its bucket is released.
        """
        low = features.propagation[2]
        grid = low.shape[-1]
        mask_size = _MASK_SCALE * grid
        memory_size = _MEMORY_STRIDE * grid
        image_size = self.image_position.image_size
        occupied = state.slot_status == SLOT_ACTIVE
        add = plan.add & (state.slot_status == SLOT_EMPTY)
        recondition = plan.recondition & occupied
        remove = plan.remove & occupied
        present = occupied | add
        no_object = jnp.asarray(_NO_OBJECT_LOGIT, jnp.float32)

        keep_tracked = (plan.recondition_masks > 0) == (tracks.mask_logits > 0)
        reconditioned_logits = jnp.where(
            keep_tracked, tracks.mask_logits, plan.recondition_masks
        )
        logits = jnp.where(
            recondition[..., None, None], reconditioned_logits, tracks.mask_logits
        )
        logits = jnp.where(
            (plan.suppress & occupied)[..., None, None], _OVERLAP_LOSER_LOGIT, logits
        )
        logits = jnp.where(occupied[..., None, None], logits, no_object)

        memory_logits = _suppress_area_shrinkage(
            resize_bilinear(logits, (memory_size, memory_size)),
            occupied,
            plan.object_ids,
        )
        appearing = occupied & jnp.any(memory_logits > 0, axis=(-2, -1))
        memory = jax.vmap(self._encode_memory, in_axes=(None, 0, 0, 0, 0))(
            low, memory_logits, occupied, plan.conditioning, appearing
        )

        def binary_at_memory_size(masks: jax.Array) -> jax.Array:
            return resize_bilinear(masks, (memory_size, memory_size)) > 0

        selected = add | recondition

        def selected_outputs() -> tuple[jax.Array, jax.Array]:
            return self._input_mask_outputs(
                interactive,
                features,
                jnp.where(
                    add[..., None, None],
                    binary_at_memory_size(plan.new_masks),
                    binary_at_memory_size(plan.recondition_masks),
                ),
                low.dtype,
            )

        def no_selected_outputs() -> tuple[jax.Array, jax.Array]:
            return (
                jnp.zeros_like(tracks.pointers),
                jnp.zeros_like(tracks.object_score_logits, dtype=jnp.float32),
            )

        new_pointers, new_scores = jax.lax.cond(
            jnp.any(selected), selected_outputs, no_selected_outputs
        )

        def reencoded() -> tuple[jax.Array, jax.Array]:
            new_low = _masks_through_source_size(
                binary_at_memory_size(plan.new_masks),
                add,
                plan.state_group,
                plan.source_size,
                mask_size,
            )
            recondition_low = _masks_through_source_size(
                binary_at_memory_size(plan.recondition_masks),
                recondition,
                plan.state_group,
                plan.source_size,
                mask_size,
            )
            stored = jnp.where(occupied[..., None, None], tracks.mask_logits, no_object)
            stored = jnp.where(recondition[..., None, None], recondition_low, stored)
            stored = jnp.where(add[..., None, None], new_low, stored)

            buckets, slots = stored.shape[:2]
            high = _non_overlapping_per_state(
                resize_bilinear(stored, (image_size, image_size)).reshape(
                    buckets * slots, image_size, image_size
                ),
                jnp.repeat(plan.state_group, slots),
                buckets,
            ).reshape(buckets, slots, image_size, image_size)
            scores = jnp.where(selected, new_scores, tracks.object_score_logits)
            encoded = jax.vmap(self._encode_memory, in_axes=(None, 0, 0, 0, 0))(
                low, high, present, plan.conditioning, present & (scores > 0)
            )
            return encoded, new_low

        def no_reencoding() -> tuple[jax.Array, jax.Array]:
            return memory, jnp.zeros_like(logits)

        reencoded_memory, new_low = jax.lax.cond(
            jnp.any(plan.reencode), reencoded, no_reencoding
        )
        memory = jnp.where(plan.reencode.reshape(-1, 1, 1, 1), reencoded_memory, memory)

        final_logits = jnp.where(add[..., None, None], new_low, logits)
        final_scores = jnp.where(
            present,
            jnp.where(selected, new_scores, tracks.object_score_logits),
            0.0,
        ).astype(jnp.float32)
        final_pointers = jnp.where(selected[..., None], new_pointers, tracks.pointers)
        final_pointers = jnp.where(present[..., None], final_pointers, 0).astype(
            tracks.pointers.dtype
        )
        status = jnp.where(add, SLOT_ACTIVE, state.slot_status)
        status = jnp.where(remove, SLOT_REMOVED, status).astype(jnp.int8)

        written = jax.vmap(_store_frame, in_axes=(0, None, None, 0, 0, 0, 0, 0, 0, 0))(
            state,
            plan.frame_index,
            low,
            memory,
            final_pointers,
            plan.new_state,
            state.bucket_valid | jnp.any(add, axis=-1),
            status,
            final_logits,
            final_scores,
        )
        cleared = _where_bucket(
            plan.release, _empty_buckets(written, status.shape[0]), written
        )
        # The driver still shows the objects of a released bucket on this frame.
        return eqx.tree_at(
            lambda s: (s.last_masks, s.last_scores),
            cleared,
            (written.last_masks, written.last_scores),
        )

    def commit_point_object(
        self,
        interactive: InteractiveHead,
        state: TrackerState,
        features: ImageFeatures,
        prompt: InteractivePrompt,
        frame_index: Int[Array, ""],
        source_size: tuple[int, int],
    ) -> TrackerState:
        """Start an object from a point prompt, in slot 0 of bucket 0.

        Its mask passes through the clip's frame size, and the frame becomes a
        conditioning frame with the object's real object score.
        """
        low = features.propagation[2]
        mask_size = _MASK_SCALE * low.shape[-1]
        image_size = self.image_position.image_size
        mask, pointer, score = self.decode_object(interactive, features, prompt)
        low_mask = resize_bilinear_antialias(
            resize_bilinear(mask, source_size), (mask_size, mask_size)
        )
        active = jnp.zeros_like(state.slot_status[0], dtype=bool).at[0].set(True)
        logits = (
            jnp.full_like(state.last_masks[0], _NO_OBJECT_LOGIT).at[0].set(low_mask)
        )
        high = resize_bilinear(logits, (image_size, image_size))
        memory = self._encode_memory(low, high, active, active, active & (score > 0))
        scores = jnp.zeros_like(state.last_scores[0]).at[0].set(score)
        pointers = jnp.zeros_like(state.pointers[0, 0]).at[0].set(pointer)
        status = jnp.full_like(state.slot_status[0], SLOT_EMPTY).at[0].set(SLOT_ACTIVE)
        return jax.tree.map(
            lambda leaf, first: leaf.at[0].set(first),
            state,
            _store_frame(
                jax.tree.map(lambda leaf: leaf[0], state),
                frame_index,
                low,
                memory,
                pointers,
                jnp.asarray(True),
                jnp.asarray(True),
                status,
                logits,
                scores,
            ),
        )

    def commit_point_step(
        self,
        state: TrackerState,
        features: ImageFeatures,
        tracks: TrackOutput,
        frame_index: Int[Array, ""],
    ) -> TrackerState:
        """Store a propagated frame of a point track, from its own masks and scores."""
        low = features.propagation[2]
        image_size = self.image_position.image_size
        occupied = state.slot_status == SLOT_ACTIVE
        high = resize_bilinear(tracks.mask_logits, (image_size, image_size))
        memory = jax.vmap(self._encode_memory, in_axes=(None, 0, 0, 0, 0))(
            low,
            high,
            occupied,
            jnp.zeros_like(occupied),
            occupied & (tracks.object_score_logits > 0),
        )
        return jax.vmap(_store_frame, in_axes=(0, None, None, 0, 0, 0, 0, 0, 0, 0))(
            state,
            frame_index,
            low,
            memory,
            tracks.pointers,
            jnp.zeros_like(state.bucket_valid),
            state.bucket_valid,
            state.slot_status,
            jnp.where(occupied[..., None, None], tracks.mask_logits, _NO_OBJECT_LOGIT),
            jnp.where(occupied, tracks.object_score_logits, 0.0).astype(jnp.float32),
        )


__all__ = [
    "Association",
    "CommitPlan",
    "TrackOutput",
    "TrackerState",
    "associate",
    "associate_tracks",
    "detection_keep",
    "pad_tracker_buckets",
    "remove_tracker_slot",
]
