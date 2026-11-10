"""Host-side video tracking for SAM 3.1 (experimental).

The module holds the association policy, which only uses NumPy so that it can be
replayed on CPU from recorded device outputs, and the driver that runs the
device steps around it. It is imported explicitly and never by ``import equimo``.

Each rule below is the behavior of ``facebookresearch/sam3`` at ``f1647e8``, with
its location; paths are relative to ``sam3/model/``. For each frame the policy
applies them in this order:

- Hot-start (``sam3_multiplex_base.py:_process_hotstart_gpu``): a non-empty track
  with no IoM of at least 0.5 counts as unmatched. An object at most 14 frames
  old is removed after 8 such frames in total. Unambiguous overlaps of at least 0.1
  on the same detection count for both objects, once per detection, and an object
  younger than another that strictly predates it is removed at 8.
- Reconditioning (``sam3_multiplex_base.py:_recondition_masklets``): on frames
  where ``t % 16 == 0``, a track whose probability is above 0.8 takes the mask of
  a kept detection of probability at least 0.8 that it matches uniquely with an
  IoM of at least 0.5. Objects removed on this frame also take part, the frame
  stays non-conditioning, and detections that are new are not eligible.
- Occlusion (``sam3_multiplex_base.py:_suppress_overlapping_based_on_recent_occlusion``
  and ``_get_objects_to_suppress_based_on_most_recently_occluded``): after
  reconditioning, of two tracks with an IoU of at least 0.7, the one with the
  strictly later occlusion frame is suppressed if the other was occluded before.
  An object removed on this frame counts as occluded at frame 100000.
  A mask without pixels and a suppressed object both record this frame as occluded.
- New objects (``sam3_video_base.py:get_new_det_gpu_ids`` and
  ``_drop_new_det_with_obj_limit``): a detection without a raw IoM of at least 0.1
  is new. They keep detection order, kept detections first, unless the object
  count before this frame's removals is over ``max_objects``, when the highest
  scores are kept.
- States (``sam3_multiplex_base.py:_tracker_add_new_objects`` and
  ``multiplex_utils.py:MultiplexState``): new objects of one frame join the state
  with the fewest empty slots that can hold them all, the first created on ties,
  filling its buckets in order. Otherwise they start a new state on free buckets,
  growing the bucket count in powers of two. Removed slots are never reused in a
  live bucket, and a bucket without live objects is released after the frame's
  memory is written.
- Confirmation (``sam3_multiplex_base.py:update_masklet_confirmation_status``): an
  object is confirmed after 3 consecutive passes matched by a kept detection or
  created, counting both passes of the prompt frame.

Emission (``sam3_multiplex_tracking.py:propagate_in_video`` and
``_postprocess_output_batched``): an object is hidden when it is unconfirmed and
still live two frames later, so removed objects stay visible up to the frame they
are removed on. Removed objects are emitted on that frame with a score of -1e4.
Objects without pixels are dropped before per-pixel overlaps are resolved by
decreasing tracker probability (the detection probability on a birth frame), ties
to the lower ID, so an object can end up shown with an empty mask.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float
from loguru import logger

from equimo.vision.models.sam3.decoding import _SINGLE_FRAME_NMS_THRESHOLD
from equimo.vision.models.sam3.detector import ConceptPrompt, DetectorOutput
from equimo.vision.models.sam3.encoder import ImageFeatures
from equimo.vision.models.sam3.interactive import InteractivePrompt
from equimo.vision.models.sam3.model import Sam3
from equimo.vision.ops import resize_bilinear
from equimo.vision.segmentation import boxes_from_masks, non_greedy_iom_nms
from equimo.vision.models.sam3.tracker import (
    _MASK_SCALE,
    _NEW_OBJECT_PROBABILITY,
    _POINTER_DISTANCE_FRAMES,
    SLOT_ACTIVE,
    SLOT_EMPTY,
    SLOT_REMOVED,
    Association,
    CommitPlan,
    TrackerState,
    TrackOutput,
    detection_keep,
    pad_tracker_buckets,
    remove_tracker_slot,
)

_HOT_START_FRAMES = 15
_HOT_START_UNMATCHED_FRAMES = 8
_HOT_START_DUPLICATE_FRAMES = 8
_RECONDITION_PERIOD = 16
_OCCLUSION_IOU = 0.7
_CONFIRMATION_FRAMES = 3
# A frame is final once this many later frames are planned, because a frame only
# hides objects that are still unconfirmed by then.
_EMISSION_LAG = _CONFIRMATION_FRAMES - 1
_NO_OBJECT = -1
_ALWAYS_OCCLUDED = 100000
_REMOVED_SCORE = -1e4


def _next_power_of_two(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


@dataclass(frozen=True)
class FramePrediction:
    """Objects shown on one frame, at the clip's source size.

    ``boxes`` are ``(x_min, y_min, x_max, y_max)`` pixels of each mask before
    per-pixel overlaps are resolved. ``scores`` are detection probabilities at
    birth, the object-score probability of a point track, or -1e4 for an object
    removed on this frame.
    """

    frame_index: int
    object_ids: np.ndarray
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class FrameObservation:
    """Device outputs the policy decides from, over ``tracks = buckets * slots``.

    ``matches`` and ``unique_high_iom`` exclude ambiguous rows and columns;
    ``high_iom`` does not. See :class:`Association` for the other fields.
    """

    frame_index: int
    keep: np.ndarray
    matches: np.ndarray
    high_iom: np.ndarray
    unique_high_iom: np.ndarray
    new_detection: np.ndarray
    confident_detection: np.ndarray
    confident_track: np.ndarray
    track_nonempty: np.ndarray
    track_iou: np.ndarray
    reconditioned_iou: np.ndarray
    reconditioned_nonempty: np.ndarray
    detection_probabilities: np.ndarray

    @classmethod
    def from_device(
        cls,
        frame_index: int,
        association: Association,
        detection_probabilities: jax.Array,
    ) -> FrameObservation:
        """Copy one frame's device outputs to the host."""
        return cls(
            frame_index=frame_index,
            keep=np.asarray(association.keep),
            matches=np.asarray(association.matches),
            high_iom=np.asarray(association.high_iom),
            unique_high_iom=np.asarray(association.unique_high_iom),
            new_detection=np.asarray(association.new_detection),
            confident_detection=np.asarray(association.confident_detection),
            confident_track=np.asarray(association.confident_track),
            track_nonempty=np.asarray(association.track_nonempty),
            track_iou=np.asarray(association.track_iou),
            reconditioned_iou=np.asarray(association.reconditioned_iou),
            reconditioned_nonempty=np.asarray(association.reconditioned_nonempty),
            detection_probabilities=np.asarray(detection_probabilities),
        )


@dataclass
class TrackedObject:
    """What the host remembers about one object."""

    object_id: int
    bucket: int
    slot: int
    birth_frame: int
    birth_score: float
    unmatched_frames: int = 0
    matched_streak: int = 0
    last_occluded_frame: int = -1
    confirmed: bool = False


@dataclass(frozen=True)
class NewObject:
    """An object created from detector query ``query`` in a slot."""

    object_id: int
    query: int
    bucket: int
    slot: int
    score: float


@dataclass(frozen=True)
class Reconditioning:
    """An object whose mask is replaced by that of detector query ``query``."""

    object_id: int
    query: int


@dataclass(frozen=True)
class FrameDecision:
    """Everything the policy decided for one frame; the decision trace."""

    frame_index: int
    added: tuple[NewObject, ...] = ()
    removed_unmatched: tuple[int, ...] = ()
    removed_duplicate: tuple[int, ...] = ()
    reconditioned: tuple[Reconditioning, ...] = ()
    suppressed: tuple[int, ...] = ()
    confirmed: tuple[int, ...] = ()
    dropped_queries: tuple[int, ...] = ()
    released_buckets: tuple[int, ...] = ()
    reencode_buckets: tuple[int, ...] = ()
    new_state_buckets: tuple[int, ...] = ()
    num_buckets: int = 1


class AssociationPolicy:
    """Host decisions of the tracker, independent of the device."""

    def __init__(self, *, num_slots: int, max_objects: int):
        if max_objects < 1:
            raise ValueError(f"max_objects must be positive, got {max_objects}.")
        self.num_slots = num_slots
        self.max_objects = max_objects
        self.max_buckets = _next_power_of_two(max_objects)
        self.slot_status = np.full((1, num_slots), SLOT_EMPTY, np.int8)
        self.slot_object = np.full((1, num_slots), _NO_OBJECT, np.int32)
        self.bucket_valid = np.zeros(1, bool)
        self.state_group = np.zeros(1, np.int32)
        self.objects: dict[int, TrackedObject] = {}
        self._duplicate_frames: dict[tuple[int, int], int] = {}
        self._point_object_ids: set[int] = set()
        self.caller_removed: set[int] = set()
        self._next_object_id = 0
        self._state_order: list[int] = []

    @property
    def num_buckets(self) -> int:
        return self.slot_status.shape[0]

    def reserve_object_id(self, object_id: int | None = None) -> int:
        """Return an ID for an object tracked outside the policy."""
        if object_id is None:
            object_id = self._next_object_id
        elif object_id in self.objects or object_id in self._point_object_ids:
            raise ValueError(
                f"Object {object_id} already exists, and refining an existing "
                "object with points on video is not supported."
            )
        self._point_object_ids.add(object_id)
        self._next_object_id = max(self._next_object_id, object_id + 1)
        return object_id

    def remove_object(self, object_id: int) -> tuple[int, ...]:
        """Stop tracking a live object and return the buckets this releases."""
        obj = self.objects.get(object_id)
        if obj is None or self.slot_status[obj.bucket, obj.slot] != SLOT_ACTIVE:
            raise ValueError(f"Object {object_id} is not being tracked.")
        self.slot_status[obj.bucket, obj.slot] = SLOT_REMOVED
        self.caller_removed.add(object_id)
        return self._release_buckets()

    def _track_index(self, obj: TrackedObject) -> int:
        return obj.bucket * self.num_slots + obj.slot

    def live_objects(self) -> list[TrackedObject]:
        """Return the objects in active slots, by increasing object ID."""
        active = np.argwhere(self.slot_status == SLOT_ACTIVE)
        objects = [self.objects[int(self.slot_object[b, s])] for b, s in active]
        return sorted(objects, key=lambda obj: obj.object_id)

    def plan_prompt_frame(
        self, candidates: np.ndarray, keep: np.ndarray, probabilities: np.ndarray
    ) -> FrameDecision:
        """Create the objects of the prompt frame's first pass."""
        added, dropped, reencode, new_state = self._add_objects(
            candidates, keep, probabilities, frame=0, objects_before=0
        )
        return FrameDecision(
            frame_index=0,
            added=added,
            dropped_queries=dropped,
            reencode_buckets=reencode,
            new_state_buckets=new_state,
            num_buckets=self.num_buckets,
        )

    def plan(self, observation: FrameObservation) -> FrameDecision:
        """Apply the policy to one propagated frame and update its metadata."""
        frame = observation.frame_index
        live = self.live_objects()
        unmatched = self._hot_start_unmatched(live, observation)
        duplicate = self._hot_start_duplicates(live, observation) - unmatched
        removed = unmatched | duplicate

        reconditioned = self._reconditioning(live, observation)
        suppressed = self._occlusion_suppression(
            live, removed, reconditioned, observation
        )

        for object_id in removed:
            obj = self.objects[object_id]
            self.slot_status[obj.bucket, obj.slot] = SLOT_REMOVED

        added, dropped, reencode, new_state = self._add_objects(
            observation.new_detection,
            observation.keep,
            observation.detection_probabilities,
            frame=frame,
            objects_before=len(live),
        )
        released = self._release_buckets()
        confirmed = self._confirm(
            [obj for obj in live if obj.object_id not in removed], observation
        )
        return FrameDecision(
            frame_index=frame,
            added=added,
            removed_unmatched=tuple(sorted(unmatched)),
            removed_duplicate=tuple(sorted(duplicate)),
            reconditioned=reconditioned,
            suppressed=suppressed,
            confirmed=confirmed,
            dropped_queries=dropped,
            released_buckets=released,
            reencode_buckets=reencode,
            new_state_buckets=new_state,
            num_buckets=self.num_buckets,
        )

    def _in_hot_start(self, obj: TrackedObject, frame: int) -> bool:
        return frame - obj.birth_frame < _HOT_START_FRAMES

    def _hot_start_unmatched(
        self, live: list[TrackedObject], observation: FrameObservation
    ) -> set[int]:
        removed = set()
        for obj in live:
            track = self._track_index(obj)
            if (
                observation.track_nonempty[track]
                and not observation.high_iom[:, track].any()
            ):
                obj.unmatched_frames += 1
            if (
                self._in_hot_start(obj, observation.frame_index)
                and obj.unmatched_frames >= _HOT_START_UNMATCHED_FRAMES
            ):
                removed.add(obj.object_id)
        return removed

    def _hot_start_duplicates(
        self, live: list[TrackedObject], observation: FrameObservation
    ) -> set[int]:
        """Count overlaps on one detection per pair, and return the objects to remove."""
        frame = observation.frame_index
        for detection in range(observation.matches.shape[0]):
            sharing = [
                obj
                for obj in live
                if observation.matches[detection, self._track_index(obj)]
            ]
            if len(sharing) < 2:
                continue
            for index, older in enumerate(sharing):
                for younger in sharing[index + 1 :]:
                    if older.birth_frame < younger.birth_frame:
                        pair = (older.object_id, younger.object_id)
                        self._duplicate_frames[pair] = (
                            self._duplicate_frames.get(pair, 0) + 1
                        )
        return {
            younger
            for (_, younger), count in self._duplicate_frames.items()
            if count >= _HOT_START_DUPLICATE_FRAMES
            and younger in {obj.object_id for obj in live}
            and self._in_hot_start(self.objects[younger], frame)
        }

    def _reconditioning(
        self, live: list[TrackedObject], observation: FrameObservation
    ) -> tuple[Reconditioning, ...]:
        if observation.frame_index % _RECONDITION_PERIOD != 0:
            return ()
        eligible = (
            observation.unique_high_iom & observation.confident_detection[:, None]
        )
        chosen = []
        for obj in live:
            track = self._track_index(obj)
            queries = np.flatnonzero(eligible[:, track])
            if observation.confident_track[track] and queries.size > 0:
                chosen.append(Reconditioning(obj.object_id, int(queries[-1])))
        return tuple(chosen)

    def _occlusion_suppression(
        self,
        live: list[TrackedObject],
        removed: set[int],
        reconditioned: tuple[Reconditioning, ...],
        observation: FrameObservation,
    ) -> tuple[int, ...]:
        if reconditioned:
            track_iou = observation.reconditioned_iou
            nonempty = observation.reconditioned_nonempty
        else:
            track_iou = observation.track_iou
            nonempty = observation.track_nonempty

        def last_occluded(obj: TrackedObject) -> int:
            return (
                _ALWAYS_OCCLUDED
                if obj.object_id in removed
                else obj.last_occluded_frame
            )

        suppressed = set()
        for index, first in enumerate(live):
            for second in live[index + 1 :]:
                iou = track_iou[self._track_index(first), self._track_index(second)]
                if iou < _OCCLUSION_IOU:
                    continue
                first_time, second_time = last_occluded(first), last_occluded(second)
                if first_time > second_time and second_time > -1:
                    suppressed.add(first.object_id)
                elif second_time > first_time and first_time > -1:
                    suppressed.add(second.object_id)

        frame = observation.frame_index
        for obj in live:
            if obj.object_id in suppressed or not nonempty[self._track_index(obj)]:
                obj.last_occluded_frame = frame
        return tuple(sorted(suppressed))

    def _add_objects(
        self,
        candidates: np.ndarray,
        keep: np.ndarray,
        probabilities: np.ndarray,
        *,
        frame: int,
        objects_before: int,
    ) -> tuple[
        tuple[NewObject, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ]:
        """Place new detections and return them, the rest, and the affected buckets."""
        queries = np.flatnonzero(candidates)
        ordered = [
            *(int(query) for query in queries if keep[query]),
            *(int(query) for query in queries if not keep[query]),
        ]
        capacity = max(self.max_objects - objects_before, 0)
        if len(ordered) > capacity:
            scores = probabilities[ordered]
            by_score = np.argsort(scores)[::-1][:capacity]
            chosen = [ordered[index] for index in by_score]
        else:
            chosen = ordered
        placements, new_state = self._place(len(chosen))
        chosen = chosen[: len(placements)]

        added = []
        for query, (bucket, slot) in zip(chosen, placements):
            object_id = self._next_object_id
            self._next_object_id += 1
            score = float(probabilities[query])
            self.objects[object_id] = TrackedObject(
                object_id=object_id,
                bucket=bucket,
                slot=slot,
                birth_frame=frame,
                birth_score=score,
                matched_streak=1,
            )
            self.slot_status[bucket, slot] = SLOT_ACTIVE
            self.slot_object[bucket, slot] = object_id
            added.append(NewObject(object_id, query, bucket, slot, score))

        kept_queries = {new.query for new in added}
        dropped = tuple(query for query in ordered if query not in kept_queries)
        if dropped:
            logger.warning(
                f"Frame {frame}: dropped {len(dropped)} new objects: "
                f"max_objects is {self.max_objects} and at most "
                f"{self.max_buckets} buckets are available."
            )
        groups = {int(self.state_group[new.bucket]) for new in added}
        reencode = tuple(
            int(bucket)
            for bucket in np.flatnonzero(self.bucket_valid)
            if int(self.state_group[bucket]) in groups
        )
        return tuple(added), dropped, reencode, new_state

    def _place(self, count: int) -> tuple[list[tuple[int, int]], tuple[int, ...]]:
        """Choose ``count`` slots, joining a state or starting a new one.

        Also returns the buckets of a state started here.
        """
        if count == 0:
            return [], ()
        best_group = None
        best_available = None
        for group in self._state_order:
            available = sum(
                int((self.slot_status[bucket] == SLOT_EMPTY).sum())
                for bucket in self._group_buckets(group)
            )
            if available >= count and (
                best_available is None or available < best_available
            ):
                best_group, best_available = group, available
        if best_group is None:
            return self._start_state(count)
        slots = [
            (bucket, int(slot))
            for bucket in self._group_buckets(best_group)
            for slot in np.flatnonzero(self.slot_status[bucket] == SLOT_EMPTY)
        ]
        return slots[:count], ()

    def _group_buckets(self, group: int) -> list[int]:
        return [
            int(bucket)
            for bucket in np.flatnonzero(
                self.bucket_valid & (self.state_group == group)
            )
        ]

    def _start_state(self, count: int) -> tuple[list[tuple[int, int]], tuple[int, ...]]:
        """Open free buckets for one batch of objects, as one state."""
        needed = -(-count // self.num_slots)
        free = np.flatnonzero(~self.bucket_valid)
        if free.size < needed:
            used = self.num_buckets - free.size
            self._grow(min(_next_power_of_two(used + needed), self.max_buckets))
            free = np.flatnonzero(~self.bucket_valid)

        buckets = [int(bucket) for bucket in free[:needed]]
        if not buckets:
            return [], ()
        taken = set(self.state_group[self.bucket_valid].tolist())
        group = next(g for g in range(self.num_buckets) if g not in taken)
        self.bucket_valid[buckets] = True
        self.state_group[buckets] = group
        self._state_order.append(group)
        slots = [(b, s) for b in buckets for s in range(self.num_slots)]
        return slots[:count], tuple(buckets)

    def _grow(self, num_buckets: int) -> None:
        extra = num_buckets - self.num_buckets
        if extra <= 0:
            return
        self.slot_status = np.concatenate(
            (self.slot_status, np.full((extra, self.num_slots), SLOT_EMPTY, np.int8))
        )
        self.slot_object = np.concatenate(
            (self.slot_object, np.full((extra, self.num_slots), _NO_OBJECT, np.int32))
        )
        self.bucket_valid = np.concatenate((self.bucket_valid, np.zeros(extra, bool)))
        self.state_group = np.concatenate((self.state_group, np.zeros(extra, np.int32)))

    def _release_buckets(self) -> tuple[int, ...]:
        live = (self.slot_status == SLOT_ACTIVE).any(axis=1)
        released = np.flatnonzero(self.bucket_valid & ~live)
        self.bucket_valid[released] = False
        self.slot_status[released] = SLOT_EMPTY
        live_groups = set(self.state_group[self.bucket_valid].tolist())
        self._state_order = [g for g in self._state_order if g in live_groups]
        return tuple(int(bucket) for bucket in released)

    def _confirm(
        self, remaining: list[TrackedObject], observation: FrameObservation
    ) -> tuple[int, ...]:
        confirmed = []
        for obj in remaining:
            matched = bool(observation.matches[:, self._track_index(obj)].any())
            obj.matched_streak = obj.matched_streak + 1 if matched else 0
            if not obj.confirmed and obj.matched_streak >= _CONFIRMATION_FRAMES:
                obj.confirmed = True
                confirmed.append(obj.object_id)
        return tuple(confirmed)


def _encode_image(model: Sam3, frame: jax.Array) -> ImageFeatures:
    return model.encode_image(frame)


def _encode_and_detect(
    model: Sam3, frame: jax.Array, prompt: ConceptPrompt
) -> tuple[ImageFeatures, DetectorOutput]:
    features = model.encode_image(frame)
    return features, model.detect(features, prompt)


def _prompt_frame_objects(
    detections: DetectorOutput,
) -> tuple[Bool[Array, " queries"], Bool[Array, " queries"], Float[Array, " queries"]]:
    """Select the prompt frame's new objects with the single-frame NMS.

    Also returns which detections pass NMS and the boundary filter, which the
    order of the new objects follows.
    """
    probabilities = jax.nn.sigmoid(detections.logits.astype(jnp.float32))
    order = jnp.argsort(-probabilities, stable=True)
    ranked = (probabilities[order] >= _NEW_OBJECT_PROBABILITY) & non_greedy_iom_nms(
        detections.mask_logits[order] > 0, _SINGLE_FRAME_NMS_THRESHOLD
    )
    _, keep = detection_keep(detections)
    return jnp.zeros_like(ranked).at[order].set(ranked), keep, probabilities


def _propagate(
    model: Sam3,
    state: TrackerState,
    features: ImageFeatures,
    frame_index: jax.Array,
    num_frames: int | None,
) -> TrackOutput:
    return model.propagate(state, features, frame_index, num_frames=num_frames)


def _associate(
    model: Sam3,
    detections: DetectorOutput,
    tracks: TrackOutput,
    state: TrackerState,
) -> tuple[Association, jax.Array]:
    return (
        model.associate(detections, tracks, state),
        jax.nn.sigmoid(detections.logits.astype(jnp.float32)),
    )


def _commit(
    model: Sam3,
    state: TrackerState,
    features: ImageFeatures,
    tracks: TrackOutput,
    plan: CommitPlan,
) -> TrackerState:
    return model.commit(state, features, tracks, plan)


def _commit_point_object(
    model: Sam3,
    state: TrackerState,
    features: ImageFeatures,
    prompt: InteractivePrompt,
    frame_index: jax.Array,
    source_size: tuple[int, int],
) -> TrackerState:
    return model.tracker.commit_point_object(
        model.interactive, state, features, prompt, frame_index, source_size
    )


def _commit_point_step(
    model: Sam3,
    state: TrackerState,
    features: ImageFeatures,
    tracks: TrackOutput,
    frame_index: jax.Array,
) -> TrackerState:
    return model.tracker.commit_point_step(state, features, tracks, frame_index)


def _stored_tracks(state: TrackerState) -> TrackOutput:
    """Return the prompt frame's stored output, in place of propagating into it."""
    return TrackOutput(
        mask_logits=state.last_masks,
        object_score_logits=state.last_scores.astype(state.pointers.dtype),
        iou=jnp.zeros_like(state.last_scores, dtype=state.pointers.dtype),
        pointers=state.pointers[:, 0],
    )


def _pad_tracks(tracks: TrackOutput, num_buckets: int) -> TrackOutput:
    def pad(leaf: jax.Array) -> jax.Array:
        extra = jnp.zeros((num_buckets - leaf.shape[0], *leaf.shape[1:]), leaf.dtype)
        return jnp.concatenate((leaf, extra), axis=0)

    return jax.tree.map(pad, tracks)


def _render_prediction(
    frame_index: int,
    object_ids: np.ndarray,
    mask_logits: np.ndarray,
    priorities: np.ndarray,
    scores: np.ndarray,
    source_size: tuple[int, int],
) -> FramePrediction:
    """Upsample masks to the source size and give each pixel to one object.

    Objects without any pixel are omitted before overlaps are resolved, so an
    object can lose all its pixels and still be returned, with an empty mask.
    Overlaps go to the highest ``priorities`` value, ties to the lower ID.
    """
    empty = FramePrediction(
        frame_index=frame_index,
        object_ids=np.zeros(0, np.int32),
        masks=np.zeros((0, *source_size), bool),
        boxes=np.zeros((0, 4), np.float32),
        scores=np.zeros(0, np.float32),
    )
    if object_ids.shape[0] == 0:
        return empty
    order = np.argsort(object_ids, kind="stable")
    object_ids = object_ids[order]
    resized = resize_bilinear(jnp.asarray(mask_logits[order], jnp.float32), source_size)
    masks = np.asarray(resized > 0)
    has_area = masks.any(axis=(-2, -1))
    if not has_area.any():
        return empty
    masks = masks[has_area]
    boxes = np.asarray(boxes_from_masks(jnp.asarray(masks)))
    kept_priorities = priorities[order][has_area].astype(np.float32)
    if masks.shape[0] > 1:
        foreground = np.where(masks, kept_priorities[:, None, None], 0.0)
        winner = np.argmax(foreground, axis=0)
        rows = np.arange(masks.shape[0])[:, None, None]
        masks = masks & (winner == rows) & (foreground > 0)
    return FramePrediction(
        frame_index=frame_index,
        object_ids=object_ids[has_area],
        masks=masks,
        boxes=boxes,
        scores=scores[order][has_area],
    )


@dataclass(frozen=True)
class _FrameRecord:
    """One planned frame, kept until it is final."""

    frame_index: int
    object_ids: np.ndarray
    mask_logits: np.ndarray
    priorities: np.ndarray
    scores: np.ndarray


class _DeviceSteps:
    """Jitted device steps, which compile once per bucket count."""

    def __init__(self) -> None:
        self.encode_image = eqx.filter_jit(_encode_image)
        self.encode_and_detect = eqx.filter_jit(_encode_and_detect)
        self.prompt_frame_objects = eqx.filter_jit(_prompt_frame_objects)
        self.propagate = eqx.filter_jit(_propagate)
        self.associate = eqx.filter_jit(_associate)
        self.commit = eqx.filter_jit(_commit)
        self.commit_point_object = eqx.filter_jit(_commit_point_object)
        self.commit_point_step = eqx.filter_jit(_commit_point_step)


class PointTrack:
    """A new object added with points, tracked on its own.

    ``first_prediction`` is the object on the frame it was added on.
    """

    def __init__(
        self,
        tracker: Sam3VideoTracker,
        object_id: int,
        frame_index: int,
        features: ImageFeatures,
        prompt: InteractivePrompt,
    ):
        self.object_id = object_id
        self._tracker = tracker
        self._state = tracker._steps.commit_point_object(
            tracker.model,
            tracker.model.tracker.empty_state(1),
            features,
            prompt,
            jnp.asarray(frame_index, jnp.int32),
            tracker.source_size,
        )
        self._score = float(jax.nn.sigmoid(self._state.last_scores[0, 0]))
        self._next_frame = frame_index + 1
        self.first_prediction = self._predict(frame_index)

    def _predict(self, frame_index: int) -> FramePrediction:
        score = self._state.last_scores[0, :1].astype(jnp.float32)
        return _render_prediction(
            frame_index,
            np.asarray([self.object_id], np.int32),
            np.asarray(self._state.last_masks[0, :1]),
            np.asarray(jax.nn.sigmoid(score)),
            np.asarray([self._score], np.float32),
            self._tracker.source_size,
        )

    def push(self, frame: Float[Array, "channels height width"]) -> FramePrediction:
        """Track the object into the next frame."""
        tracker = self._tracker
        steps = tracker._steps
        frame_index = self._next_frame
        features = steps.encode_image(tracker.model, frame)
        index = jnp.asarray(frame_index, jnp.int32)
        tracks = steps.propagate(
            tracker.model, self._state, features, index, tracker._pointer_frames
        )
        self._state = steps.commit_point_step(
            tracker.model, self._state, features, tracks, index
        )
        self._next_frame = frame_index + 1
        return self._predict(frame_index)


class Sam3VideoTracker:
    """Detect and track the objects of a concept through frames (experimental).

    Frames are preprocessed by the caller to the model's input size.
    ``source_size`` is the ``(height, width)`` of the original clip: memory is
    re-encoded and predictions are returned at that size. ``num_frames`` is the
    clip length, or ``None`` when streaming.

    Each frame returned by :meth:`push_frame` is two frames behind the latest
    one, and :meth:`finish` returns the rest. :attr:`decisions` holds one
    :class:`FrameDecision` per planned pass, including new objects that were
    dropped because of ``max_objects``.
    """

    def __init__(
        self,
        model: Sam3,
        *,
        num_frames: int | None,
        source_size: tuple[int, int],
        max_objects: int = 16,
    ):
        self.model = model
        self.num_frames = num_frames
        # Longer clips share a pointer divisor, so they share a compiled step too.
        self._pointer_frames = (
            _POINTER_DISTANCE_FRAMES
            if num_frames is None
            else min(num_frames, _POINTER_DISTANCE_FRAMES)
        )
        self.source_size = source_size
        self.policy = AssociationPolicy(
            num_slots=model.tracker.mask_decoder.num_slots, max_objects=max_objects
        )
        self.decisions: list[FrameDecision] = []
        self._steps = _DeviceSteps()
        self._mask_size = _MASK_SCALE * model.tracker.image_position.grid
        self._state = model.tracker.empty_state(1)
        self._prompt: ConceptPrompt | None = None
        self._later_frames_prompt: ConceptPrompt | None = None
        self._next_frame = 0
        self._pending: deque[_FrameRecord] = deque()
        self._finished = False

    def set_prompt(self, prompt: ConceptPrompt) -> None:
        """Set the concept to detect, before the first frame is pushed.

        Exemplars apply to the prompt frame only, and the text to every frame,
        as in the reference (``sam3_multiplex_tracking.py:_run_single_frame_inference``).
        """
        if self._next_frame > 0:
            raise ValueError(
                "The prompt can only be set before the first frame; create a new "
                "tracker to change it."
            )
        self._prompt = prompt
        self._later_frames_prompt = prompt.without_exemplars()

    def remove_object(self, object_id: int) -> None:
        """Stop tracking a live object and hide it from frames not yet returned.

        Frames already returned keep it. A later detection of the same object
        becomes a new object with a new ID, as in the reference.
        """
        released = self.policy.remove_object(object_id)
        obj = self.policy.objects[object_id]
        self._state = remove_tracker_slot(self._state, obj.bucket, obj.slot, released)

    def push_frame(
        self, frame: Float[Array, "channels height width"]
    ) -> list[FramePrediction]:
        """Process one frame and return the frames that became final."""
        if self._prompt is None:
            raise ValueError("Call set_prompt before pushing the first frame.")
        if self._finished:
            raise ValueError("The clip is finished; create a new tracker.")
        if self.num_frames is not None and self._next_frame >= self.num_frames:
            raise ValueError(f"The clip only has {self.num_frames} frames.")

        steps = self._steps
        frame_index = self._next_frame
        prompt = self._prompt if frame_index == 0 else self._later_frames_prompt
        features, detections = steps.encode_and_detect(self.model, frame, prompt)
        index = jnp.asarray(frame_index, jnp.int32)
        if frame_index == 0:
            self._first_pass_of_prompt_frame(features, detections)
            tracks = _stored_tracks(self._state)
        else:
            tracks = steps.propagate(
                self.model, self._state, features, index, self._pointer_frames
            )

        association, probabilities = steps.associate(
            self.model, detections, tracks, self._state
        )
        decision = self.policy.plan(
            FrameObservation.from_device(frame_index, association, probabilities)
        )
        self._apply(decision, features, detections, tracks)

        self._record(decision, detections)
        self._next_frame = frame_index + 1
        return self._emit_final_frames(frame_index)

    def finish(self) -> list[FramePrediction]:
        """Return the remaining frames, once no frame will follow them."""
        self._finished = True
        predictions = [self._render(record) for record in self._pending]
        self._pending.clear()
        return predictions

    def add_object(
        self,
        frame_index: int,
        frame: Float[Array, "channels height width"],
        points: jax.Array | np.ndarray | Sequence[Sequence[float]],
        labels: jax.Array | np.ndarray | Sequence[int],
        *,
        object_id: int | None = None,
    ) -> PointTrack:
        """Start tracking a new object from points on a frame.

        Points are ``(x, y)`` in input pixels, with labels 1 (positive) and 0
        (negative). The object is tracked on its own, over the frames given to
        the returned :class:`PointTrack`. Naming an existing object raises an
        error: refining objects with points on video is not supported.
        """
        new_id = self.policy.reserve_object_id(object_id)
        features = self._steps.encode_image(self.model, frame)
        prompt = InteractivePrompt.create(
            points=jnp.asarray(points, jnp.float32),
            point_labels=jnp.asarray(labels, jnp.int32),
            mask_size=self._mask_size,
        )
        return PointTrack(self, new_id, frame_index, features, prompt)

    def _first_pass_of_prompt_frame(
        self, features: ImageFeatures, detections: DetectorOutput
    ) -> None:
        candidates, keep, probabilities = self._steps.prompt_frame_objects(detections)
        decision = self.policy.plan_prompt_frame(
            np.asarray(candidates), np.asarray(keep), np.asarray(probabilities)
        )
        if decision.added:
            self._apply(decision, features, detections, _stored_tracks(self._state))
        else:
            self.decisions.append(decision)

    def _apply(
        self,
        decision: FrameDecision,
        features: ImageFeatures,
        detections: DetectorOutput,
        tracks: TrackOutput,
    ) -> None:
        if decision.num_buckets > self._state.slot_status.shape[0]:
            self._state = pad_tracker_buckets(self._state, decision.num_buckets)
            tracks = _pad_tracks(tracks, decision.num_buckets)
        plan = self._commit_plan(decision, detections)
        self._state = self._steps.commit(
            self.model, self._state, features, tracks, plan
        )
        self.decisions.append(decision)

    def _commit_plan(
        self, decision: FrameDecision, detections: DetectorOutput
    ) -> CommitPlan:
        """Turn a decision into per-slot arrays."""
        state = self._state
        policy = self.policy
        shape = state.slot_status.shape
        add = np.zeros(shape, bool)
        recondition = np.zeros(shape, bool)
        suppress = np.zeros(shape, bool)
        remove = np.zeros(shape, bool)
        conditioning = np.zeros(shape, bool)

        for object_id in (*decision.removed_unmatched, *decision.removed_duplicate):
            obj = policy.objects[object_id]
            remove[obj.bucket, obj.slot] = True
        for object_id in decision.suppressed:
            obj = policy.objects[object_id]
            suppress[obj.bucket, obj.slot] = True

        def placed_masks(
            targets: list[tuple[int, int]], queries: list[int]
        ) -> jax.Array:
            masks = jnp.zeros_like(state.last_masks)
            if not targets:
                return masks
            buckets, slots = zip(*targets)
            return masks.at[jnp.asarray(buckets), jnp.asarray(slots)].set(
                detections.mask_logits[jnp.asarray(queries)]
            )

        new_targets = [(new.bucket, new.slot) for new in decision.added]
        for bucket, slot in new_targets:
            add[bucket, slot] = True
        recondition_targets = []
        for item in decision.reconditioned:
            obj = policy.objects[item.object_id]
            recondition[obj.bucket, obj.slot] = True
            recondition_targets.append((obj.bucket, obj.slot))
        for obj in policy.live_objects():
            if obj.birth_frame == decision.frame_index:
                conditioning[obj.bucket, obj.slot] = True
        conditioning = conditioning | add | recondition

        release = np.zeros(shape[0], bool)
        release[list(decision.released_buckets)] = True
        reencode = np.zeros(shape[0], bool)
        reencode[list(decision.reencode_buckets)] = True
        new_state = np.zeros(shape[0], bool)
        new_state[list(decision.new_state_buckets)] = True
        object_ids = np.maximum(policy.slot_object, 0).astype(np.int32)

        empty = CommitPlan.empty(
            state, jnp.asarray(decision.frame_index, jnp.int32), self.source_size
        )
        return eqx.tree_at(
            lambda p: (
                p.add,
                p.new_masks,
                p.recondition,
                p.recondition_masks,
                p.suppress,
                p.remove,
                p.release,
                p.reencode,
                p.new_state,
                p.conditioning,
                p.state_group,
                p.object_ids,
            ),
            empty,
            (
                jnp.asarray(add),
                placed_masks(new_targets, [new.query for new in decision.added]),
                jnp.asarray(recondition),
                placed_masks(
                    recondition_targets, [item.query for item in decision.reconditioned]
                ),
                jnp.asarray(suppress),
                jnp.asarray(remove),
                jnp.asarray(release),
                jnp.asarray(reencode),
                jnp.asarray(new_state),
                jnp.asarray(conditioning),
                jnp.asarray(policy.state_group),
                jnp.asarray(object_ids),
            ),
        )

    def _record(self, decision: FrameDecision, detections: DetectorOutput) -> None:
        policy = self.policy
        removed = {*decision.removed_unmatched, *decision.removed_duplicate}
        birth_queries = {new.object_id: new.query for new in decision.added}
        objects = sorted(
            [*policy.live_objects(), *(policy.objects[i] for i in removed)],
            key=lambda obj: obj.object_id,
        )
        buckets = jnp.asarray([obj.bucket for obj in objects], jnp.int32)
        slots = jnp.asarray([obj.slot for obj in objects], jnp.int32)
        mask_logits = np.array(self._state.last_masks[buckets, slots])
        priorities = np.asarray(
            jax.nn.sigmoid(self._state.last_scores[buckets, slots].astype(jnp.float32))
        ).copy()
        scores = np.zeros(len(objects), np.float32)
        for index, obj in enumerate(objects):
            scores[index] = obj.birth_score
            query = birth_queries.get(obj.object_id)
            if query is not None:
                mask_logits[index] = np.asarray(detections.mask_logits[query])
                priorities[index] = obj.birth_score
            if obj.object_id in removed:
                scores[index] = _REMOVED_SCORE
        self._pending.append(
            _FrameRecord(
                frame_index=decision.frame_index,
                object_ids=np.asarray([obj.object_id for obj in objects], np.int32),
                mask_logits=mask_logits,
                priorities=priorities,
                scores=scores,
            )
        )

    def _emit_final_frames(self, latest_frame: int) -> list[FramePrediction]:
        predictions = []
        while (
            self._pending
            and self._pending[0].frame_index <= latest_frame - _EMISSION_LAG
        ):
            predictions.append(self._render(self._pending.popleft()))
        return predictions

    def _render(self, record: _FrameRecord) -> FramePrediction:
        live = {obj.object_id for obj in self.policy.live_objects()}
        objects = self.policy.objects
        shown = np.asarray(
            [
                (objects[int(i)].confirmed or int(i) not in live)
                and int(i) not in self.policy.caller_removed
                for i in record.object_ids
            ],
            bool,
        )
        return _render_prediction(
            record.frame_index,
            record.object_ids[shown],
            record.mask_logits[shown],
            record.priorities[shown],
            record.scores[shown],
            self.source_size,
        )


__all__ = [
    "AssociationPolicy",
    "FrameDecision",
    "FrameObservation",
    "FramePrediction",
    "NewObject",
    "PointTrack",
    "Reconditioning",
    "Sam3VideoTracker",
    "TrackedObject",
]
