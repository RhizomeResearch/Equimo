"""SAM 3.1 association policy and video driver on tiny random-weight models."""

import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.vision.models.sam3 import (
    ConceptPrompt,
    DetectorOutput,
    Sam3,
)
from equimo.vision.models.sam3.tracker import (
    SLOT_ACTIVE,
    SLOT_EMPTY,
    SLOT_REMOVED,
    Association,
)
from equimo.vision.models.sam3.video import (
    AssociationPolicy,
    FrameObservation,
    FramePrediction,
    Reconditioning,
    Sam3VideoTracker,
    _render_prediction,
)
from equimo.vision.models.sam3.video import _commit as video_commit
from equimo.vision.models.sam3.video import _propagate as video_propagate

NUM_SLOTS = 2
NUM_QUERIES = 4
DIM = 16
GRID = 4
MASK_SIZE = 4 * GRID
SOURCE_SIZE = (40, 60)
TEXT = jr.normal(jr.PRNGKey(2), (5, DIM))
FRAME = jr.normal(jr.PRNGKey(1), (3, 56, 56))
OTHER_FRAME = jr.normal(jr.PRNGKey(3), (3, 56, 56))
PROMPT = ConceptPrompt.create(
    text_features=TEXT, dim=DIM, max_text_tokens=8, max_boxes=2, max_points=2
)


def _observation(policy: AssociationPolicy, frame: int, **fields) -> FrameObservation:
    tracks = policy.num_buckets * policy.num_slots
    defaults = {
        "keep": np.ones(NUM_QUERIES, bool),
        "matches": np.zeros((NUM_QUERIES, tracks), bool),
        "high_iom": np.zeros((NUM_QUERIES, tracks), bool),
        "unique_high_iom": np.zeros((NUM_QUERIES, tracks), bool),
        "new_detection": np.zeros(NUM_QUERIES, bool),
        "confident_detection": np.zeros(NUM_QUERIES, bool),
        "confident_track": np.zeros(tracks, bool),
        "track_nonempty": np.ones(tracks, bool),
        "track_iou": np.zeros((tracks, tracks), np.float32),
        "reconditioned_iou": np.zeros((tracks, tracks), np.float32),
        "reconditioned_nonempty": np.ones(tracks, bool),
        "detection_probabilities": np.zeros(NUM_QUERIES, np.float32),
    }
    return FrameObservation(frame_index=frame, **{**defaults, **fields})


def _policy(max_objects: int = 8, num_slots: int = NUM_SLOTS) -> AssociationPolicy:
    return AssociationPolicy(num_slots=num_slots, max_objects=max_objects)


def _start_objects(policy: AssociationPolicy, *queries: int) -> None:
    keep = np.zeros(NUM_QUERIES, bool)
    keep[list(queries)] = True
    policy.plan_prompt_frame(
        keep, np.ones(NUM_QUERIES, bool), np.full(NUM_QUERIES, 0.9, np.float32)
    )


def _matched(policy: AssociationPolicy, *tracks: int) -> np.ndarray:
    """Match each track to its own detection."""
    size = policy.num_buckets * policy.num_slots
    high_iom = np.zeros((NUM_QUERIES, size), bool)
    for query, track in enumerate(tracks):
        high_iom[query, track] = True
    return high_iom


def test_prompt_frame_keeps_the_highest_scores_over_the_object_limit():
    policy = _policy(max_objects=2)
    probabilities = np.asarray([0.7, 0.9, 0.8, 0.3], np.float32)
    decision = policy.plan_prompt_frame(
        np.ones(NUM_QUERIES, bool), np.ones(NUM_QUERIES, bool), probabilities
    )

    assert [new.query for new in decision.added] == [1, 2]
    assert decision.dropped_queries == (0, 3)
    assert decision.confirmed == ()
    assert policy.objects[0].birth_score == pytest.approx(0.9)
    assert not policy.objects[1].confirmed


def test_new_objects_keep_kept_detections_first_when_under_the_limit():
    policy = _policy()
    candidates = np.asarray([True, True, True, False])
    keep = np.asarray([False, True, False, True])
    decision = policy.plan_prompt_frame(
        candidates, keep, np.asarray([0.7, 0.9, 0.8, 0.3], np.float32)
    )
    assert [new.query for new in decision.added] == [1, 0, 2]


def test_object_limit_counts_objects_removed_on_the_same_frame():
    policy = _policy(max_objects=2)
    _start_objects(policy, 0, 1)
    for frame in range(7):
        policy.plan(_observation(policy, frame, high_iom=_matched(policy, 0)))
    new = np.zeros(NUM_QUERIES, bool)
    new[2] = True
    decision = policy.plan(
        _observation(
            policy,
            7,
            high_iom=_matched(policy, 0),
            new_detection=new,
            detection_probabilities=np.full(NUM_QUERIES, 0.9, np.float32),
        )
    )
    assert decision.removed_unmatched == (1,)
    assert decision.added == ()
    assert decision.dropped_queries == (2,)


def test_object_is_confirmed_after_three_consecutive_matched_frames():
    policy = _policy()
    new = np.zeros(NUM_QUERIES, bool)
    new[0] = True
    probabilities = np.full(NUM_QUERIES, 0.9, np.float32)
    decision = policy.plan(
        _observation(
            policy, 1, new_detection=new, detection_probabilities=probabilities
        )
    )
    object_id = decision.added[0].object_id
    assert not policy.objects[object_id].confirmed

    # A miss restarts the count that the birth frame began.
    assert policy.plan(_observation(policy, 2)).confirmed == ()
    for frame in (3, 4):
        decision = policy.plan(_observation(policy, frame, matches=_matched(policy, 0)))
        assert decision.confirmed == ()
    decision = policy.plan(_observation(policy, 5, matches=_matched(policy, 0)))
    assert decision.confirmed == (object_id,)


def test_hot_start_removes_unmatched_objects_only_inside_the_window():
    policy = _policy()
    _start_objects(policy, 0, 1)
    for frame in range(7):
        assert policy.plan(_observation(policy, frame)).removed_unmatched == ()
    decision = policy.plan(_observation(policy, 7))
    assert decision.removed_unmatched == (0, 1)

    late = _policy()
    _start_objects(late, 0)
    for frame in range(15):
        late.plan(_observation(late, frame, high_iom=_matched(late, 0)))
    for frame in range(15, 40):
        assert late.plan(_observation(late, frame)).removed_unmatched == ()


def test_hot_start_counts_only_tracks_with_pixels_as_unmatched():
    policy = _policy()
    _start_objects(policy, 0)
    for frame in range(10):
        decision = policy.plan(
            _observation(policy, frame, track_nonempty=np.zeros(2, bool))
        )
        assert decision.removed_unmatched == ()


def _sharing_observation(policy: AssociationPolicy, frame: int, detections: int = 1):
    matches = np.zeros((NUM_QUERIES, policy.num_slots), bool)
    matches[:detections, :2] = True
    return _observation(
        policy,
        frame,
        matches=matches,
        high_iom=np.ones((NUM_QUERIES, policy.num_slots), bool),
    )


def test_hot_start_removes_the_younger_of_two_objects_sharing_a_detection():
    policy = _policy()
    _start_objects(policy, 0)
    new = np.zeros(NUM_QUERIES, bool)
    new[1] = True
    probabilities = np.full(NUM_QUERIES, 0.9, np.float32)
    younger = policy.plan(
        _observation(
            policy, 1, new_detection=new, detection_probabilities=probabilities
        )
    ).added[0]

    for frame in range(2, 9):
        assert policy.plan(_sharing_observation(policy, frame)).removed_duplicate == ()
    decision = policy.plan(_sharing_observation(policy, 9))
    assert decision.removed_duplicate == (younger.object_id,)
    assert decision.removed_unmatched == ()


def test_hot_start_counts_each_shared_detection():
    policy = _policy()
    _start_objects(policy, 0)
    new = np.zeros(NUM_QUERIES, bool)
    new[1] = True
    policy.plan(
        _observation(
            policy,
            1,
            new_detection=new,
            detection_probabilities=np.full(NUM_QUERIES, 0.9, np.float32),
        )
    )
    for frame in range(2, 5):
        assert (
            policy.plan(_sharing_observation(policy, frame, 2)).removed_duplicate == ()
        )
    assert policy.plan(_sharing_observation(policy, 5, 2)).removed_duplicate == (1,)


def test_objects_of_the_same_frame_are_never_duplicates():
    policy = _policy()
    _start_objects(policy, 0, 1)
    for frame in range(12):
        assert policy.plan(_sharing_observation(policy, frame)).removed_duplicate == ()


def _recondition_fields(policy: AssociationPolicy) -> dict[str, np.ndarray]:
    confident_detection = np.zeros(NUM_QUERIES, bool)
    confident_detection[0] = True
    return {
        "high_iom": _matched(policy, 0),
        "unique_high_iom": _matched(policy, 0),
        "confident_detection": confident_detection,
        "confident_track": np.ones(policy.num_slots, bool),
    }


@pytest.mark.parametrize(
    "frame, override, expected",
    [
        (16, {}, (Reconditioning(0, 0),)),
        (0, {}, (Reconditioning(0, 0),)),
        (17, {}, ()),
        (16, {"confident_track": np.zeros(NUM_SLOTS, bool)}, ()),
        (16, {"confident_detection": np.zeros(NUM_QUERIES, bool)}, ()),
    ],
)
def test_reconditioning_needs_a_period_frame_and_confident_outputs(
    frame, override, expected
):
    policy = _policy()
    _start_objects(policy, 0)
    fields = {**_recondition_fields(policy), **override}
    assert policy.plan(_observation(policy, frame, **fields)).reconditioned == expected


def test_reconditioning_skips_ambiguous_matches():
    policy = _policy()
    _start_objects(policy, 0)
    fields = _recondition_fields(policy)
    fields["unique_high_iom"] = np.zeros_like(fields["unique_high_iom"])
    assert policy.plan(_observation(policy, 16, **fields)).reconditioned == ()


def _occlude(policy: AssociationPolicy, frame: int, *tracks: int) -> None:
    nonempty = np.ones(policy.num_buckets * policy.num_slots, bool)
    nonempty[list(tracks)] = False
    policy.plan(_observation(policy, frame, track_nonempty=nonempty))


def test_occlusion_suppresses_the_more_recently_occluded_object():
    policy = _policy()
    _start_objects(policy, 0, 1)
    track_iou = np.asarray([[1.0, 0.75], [0.75, 1.0]], np.float32)

    assert policy.plan(_observation(policy, 1, track_iou=track_iou)).suppressed == ()
    _occlude(policy, 2, 0)
    # The other object was never occluded, so it cannot suppress anything.
    assert policy.plan(_observation(policy, 3, track_iou=track_iou)).suppressed == ()
    _occlude(policy, 4, 1)
    assert policy.plan(_observation(policy, 5, track_iou=track_iou)).suppressed == (1,)

    below = np.asarray([[1.0, 0.65], [0.65, 1.0]], np.float32)
    assert policy.plan(_observation(policy, 6, track_iou=below)).suppressed == ()


def test_occlusion_ties_suppress_nobody():
    policy = _policy()
    _start_objects(policy, 0, 1)
    _occlude(policy, 1, 0, 1)
    track_iou = np.asarray([[1.0, 0.8], [0.8, 1.0]], np.float32)
    assert policy.plan(_observation(policy, 2, track_iou=track_iou)).suppressed == ()


def test_occlusion_reads_the_masks_after_reconditioning():
    policy = _policy()
    _start_objects(policy, 0, 1)
    _occlude(policy, 1, 0)
    _occlude(policy, 2, 1)
    fields = _recondition_fields(policy)
    fields["confident_track"] = np.ones(2, bool)
    overlapping = np.asarray([[1.0, 0.8], [0.8, 1.0]], np.float32)
    decision = policy.plan(
        _observation(policy, 16, reconditioned_iou=overlapping, **fields)
    )
    assert decision.reconditioned == (Reconditioning(0, 0),)
    assert decision.suppressed == (1,)


def test_objects_removed_this_frame_are_suppressed_first_when_they_overlap():
    policy = _policy()
    _start_objects(policy, 0)
    new = np.zeros(NUM_QUERIES, bool)
    new[1] = True
    policy.plan(
        _observation(
            policy,
            1,
            new_detection=new,
            detection_probabilities=np.full(NUM_QUERIES, 0.9, np.float32),
        )
    )
    _occlude(policy, 2, 0)
    track_iou = np.asarray([[1.0, 0.8], [0.8, 1.0]], np.float32)
    for frame in range(3, 10):
        policy.plan(_sharing_observation(policy, frame))
    observation = _sharing_observation(policy, 10)
    observation = FrameObservation(
        **{
            **observation.__dict__,
            "track_iou": track_iou,
            "track_nonempty": np.ones(2, bool),
        }
    )
    decision = policy.plan(observation)
    assert decision.removed_duplicate == (1,)
    assert decision.suppressed == (1,)


def test_new_objects_join_the_tightest_live_bucket_that_holds_the_batch():
    policy = _policy(num_slots=4)
    _start_objects(policy, 0, 1)
    new = np.zeros(NUM_QUERIES, bool)
    new[2:] = True
    probabilities = np.full(NUM_QUERIES, 0.9, np.float32)
    decision = policy.plan(
        _observation(
            policy,
            1,
            new_detection=new,
            detection_probabilities=probabilities,
            high_iom=_matched(policy, 0, 1),
        )
    )
    assert [(new.bucket, new.slot) for new in decision.added] == [(0, 2), (0, 3)]
    assert decision.reencode_buckets == (0,)
    assert decision.num_buckets == 1


def test_removed_slots_are_not_reused_and_a_large_batch_starts_a_new_state():
    policy = _policy()
    _start_objects(policy, 0, 1)
    for frame in range(8):
        policy.plan(_observation(policy, frame, high_iom=_matched(policy, 1)))
    assert policy.slot_status[0].tolist() == [SLOT_REMOVED, SLOT_ACTIVE]

    new = np.zeros(NUM_QUERIES, bool)
    new[:3] = True
    probabilities = np.full(NUM_QUERIES, 0.9, np.float32)
    decision = policy.plan(
        _observation(
            policy,
            8,
            new_detection=new,
            detection_probabilities=probabilities,
            high_iom=_matched(policy, 1),
        )
    )
    assert decision.num_buckets == 4
    assert [(new.bucket, new.slot) for new in decision.added] == [
        (1, 0),
        (1, 1),
        (2, 0),
    ]
    assert policy.state_group[1] == policy.state_group[2]
    assert policy.state_group[1] != policy.state_group[0]
    assert decision.reencode_buckets == (1, 2)
    assert decision.new_state_buckets == (1, 2)


def test_released_buckets_are_reused_and_get_a_fresh_state_group():
    policy = _policy()
    _start_objects(policy, 0, 1, 2)
    assert policy.num_buckets == 2
    for frame in range(7):
        policy.plan(_observation(policy, frame))
    decision = policy.plan(_observation(policy, 7))
    assert decision.released_buckets == (0, 1)
    assert not policy.bucket_valid.any()
    assert (policy.slot_status == SLOT_EMPTY).all()

    new = np.zeros(NUM_QUERIES, bool)
    new[0] = True
    decision = policy.plan(
        _observation(
            policy,
            8,
            new_detection=new,
            detection_probabilities=np.full(NUM_QUERIES, 0.9, np.float32),
        )
    )
    assert decision.num_buckets == 2
    assert decision.added[0].bucket == 0
    assert policy.slot_status[0, 0] == SLOT_ACTIVE


def test_new_objects_join_the_fullest_state_that_holds_them():
    policy = _policy(max_objects=16, num_slots=4)
    new = np.zeros(NUM_QUERIES, bool)
    new[0] = True
    probabilities = np.full(NUM_QUERIES, 0.9, np.float32)
    keep = np.ones(NUM_QUERIES, bool)

    policy.plan_prompt_frame(new, keep, probabilities)
    both = np.ones(NUM_QUERIES, bool)
    decision = policy.plan(
        _observation(
            policy,
            1,
            new_detection=both,
            detection_probabilities=probabilities,
            high_iom=np.ones((NUM_QUERIES, policy.num_slots), bool),
        )
    )
    assert [(new.bucket, new.slot) for new in decision.added] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert decision.reencode_buckets == (1,)

    decision = policy.plan(
        _observation(
            policy,
            2,
            new_detection=new,
            detection_probabilities=probabilities,
            high_iom=np.ones(
                (NUM_QUERIES, policy.num_buckets * policy.num_slots), bool
            ),
        )
    )
    assert [(item.bucket, item.slot) for item in decision.added] == [(0, 1)]
    assert decision.new_state_buckets == ()


def test_objects_over_the_cap_are_dropped_and_reported():
    policy = _policy(max_objects=2)
    _start_objects(policy, 0, 1)
    assert policy.max_buckets == 2
    for frame in range(8):
        policy.plan(_observation(policy, frame, high_iom=_matched(policy, 1)))

    new = np.zeros(NUM_QUERIES, bool)
    new[:3] = True
    decision = policy.plan(
        _observation(
            policy,
            8,
            new_detection=new,
            detection_probabilities=np.asarray([0.9, 0.8, 0.7, 0.1], np.float32),
            high_iom=_matched(policy, 1),
        )
    )
    assert [new.query for new in decision.added] == [0]
    assert decision.dropped_queries == (1, 2)
    assert decision.num_buckets == 2


def test_point_objects_get_unused_ids_and_existing_ones_are_refused():
    policy = _policy()
    _start_objects(policy, 0)
    assert policy.reserve_object_id() == 1
    assert policy.reserve_object_id(7) == 7
    with pytest.raises(ValueError, match="not supported"):
        policy.reserve_object_id(0)
    with pytest.raises(ValueError, match="not supported"):
        policy.reserve_object_id(7)

    new = np.zeros(NUM_QUERIES, bool)
    new[0] = True
    decision = policy.plan(
        _observation(
            policy,
            1,
            new_detection=new,
            detection_probabilities=np.full(NUM_QUERIES, 0.9, np.float32),
            high_iom=_matched(policy, 0),
        )
    )
    assert decision.added[0].object_id == 8


def test_rendering_gives_pixels_to_the_higher_priority_then_the_lower_id():
    logits = np.full((3, MASK_SIZE, MASK_SIZE), -1.0, np.float32)
    logits[0, :8] = 1.0
    logits[1, :] = 1.0
    prediction = _render_prediction(
        4,
        np.asarray([3, 5, 9], np.int32),
        logits,
        np.asarray([0.7, 0.7, 0.9], np.float32),
        np.asarray([0.6, 0.7, 0.8], np.float32),
        (MASK_SIZE, MASK_SIZE),
    )
    # Object 9 has no pixel, so it is omitted; the tie goes to the lower ID.
    assert prediction.object_ids.tolist() == [3, 5]
    assert prediction.scores.tolist() == pytest.approx([0.6, 0.7])
    assert prediction.masks[0, :8].all() and not prediction.masks[1, :8].any()
    assert prediction.masks[1, 8:].all()
    assert prediction.boxes[1].tolist() == [0.0, 0.0, 15.0, 15.0]


def test_rendering_keeps_an_object_that_lost_all_its_pixels():
    logits = np.full((2, MASK_SIZE, MASK_SIZE), -1.0, np.float32)
    logits[:, :8] = 1.0
    prediction = _render_prediction(
        0,
        np.asarray([1, 2], np.int32),
        logits,
        np.asarray([0.9, 0.2], np.float32),
        np.asarray([0.9, 0.2], np.float32),
        (MASK_SIZE, MASK_SIZE),
    )
    assert prediction.object_ids.tolist() == [1, 2]
    assert not prediction.masks[1].any()
    assert prediction.boxes[1].tolist() == prediction.boxes[0].tolist()


def test_importing_equimo_does_not_import_the_video_module():
    code = (
        "import sys, equimo, equimo.vision.models.sam3;"
        "assert 'equimo.vision.models.sam3.video' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture(scope="module")
def model() -> Sam3:
    return Sam3(
        key=jr.PRNGKey(0),
        in_channels=3,
        img_size=56,
        patch_size=14,
        embed_dim=DIM,
        depth=2,
        num_heads=2,
        mlp_features=2 * DIM,
        window_size=2,
        global_blocks=(1,),
        dim=DIM,
        head_num_heads=2,
        ffn_features=2 * DIM,
        geometry_depth=1,
        fusion_depth=1,
        decoder_depth=2,
        num_queries=NUM_QUERIES,
        two_way_depth=1,
        memory_depth=1,
        memory_heads=2,
        memory_start_channels=1,
        memory_fuser_depth=1,
        num_slots=NUM_SLOTS,
    )


def _scripted_detections(probabilities: list[float]) -> DetectorOutput:
    """Detections of disjoint rows of the mask, at the given probabilities."""
    logits = jnp.log(jnp.asarray(probabilities) / (1 - jnp.asarray(probabilities)))
    masks = -jnp.ones((NUM_QUERIES, MASK_SIZE, MASK_SIZE))
    for query in range(NUM_QUERIES):
        masks = masks.at[query, 4 * query : 4 * query + 3].set(5.0)
    return DetectorOutput(
        logits=logits,
        presence_logit=jnp.asarray(3.0),
        boxes=jnp.full((NUM_QUERIES, 4), 0.5),
        mask_logits=masks,
        layer_boxes=jnp.zeros((1, NUM_QUERIES, 4)),
        layer_logits=logits[None],
    )


def _script_detector(driver: Sam3VideoTracker, frames: dict[int, list[float]]) -> None:
    """Replace the detector's output for each frame, keeping the real image encoder."""
    encode = driver._steps.encode_and_detect
    no_detection = [0.05] * NUM_QUERIES

    def scripted(model, frame, prompt):
        features, _ = encode(model, frame, prompt)
        return features, _scripted_detections(
            frames.get(driver._next_frame, no_detection)
        )

    driver._steps.encode_and_detect = scripted


def _script_new_detections(driver: Sam3VideoTracker, new: dict[int, list[int]]) -> None:
    """Have ``associate`` report no match, and the given new detections per frame."""
    real = driver._steps.associate

    def scripted(model, detections, tracks, state):
        tracks_total = state.slot_status.size
        _, probabilities = real(model, detections, tracks, state)
        flags = np.zeros(NUM_QUERIES, bool)
        flags[new.get(driver._next_frame, [])] = True
        no_pairs = jnp.zeros((NUM_QUERIES, tracks_total), bool)
        return (
            Association(
                keep=jnp.ones(NUM_QUERIES, bool),
                iom=jnp.zeros((NUM_QUERIES, tracks_total)),
                matches=no_pairs,
                high_iom=no_pairs,
                unique_high_iom=no_pairs,
                new_detection=jnp.asarray(flags),
                confident_detection=jnp.zeros(NUM_QUERIES, bool),
                confident_track=jnp.zeros(tracks_total, bool),
                track_nonempty=jnp.ones(tracks_total, bool),
                track_iou=jnp.zeros((tracks_total, tracks_total)),
                reconditioned_iou=jnp.zeros((tracks_total, tracks_total)),
                reconditioned_nonempty=jnp.ones(tracks_total, bool),
            ),
            probabilities,
        )

    driver._steps.associate = scripted


def _capture_shown_objects(monkeypatch) -> list[tuple[int, list[int]]]:
    """Record which objects each emitted frame shows, whatever their masks hold."""
    shown: list[tuple[int, list[int]]] = []

    def capture(frame_index, object_ids, mask_logits, priorities, scores, size):
        shown.append((frame_index, object_ids.tolist()))
        return FramePrediction(
            frame_index, object_ids, mask_logits, np.zeros(0), scores
        )

    monkeypatch.setattr("equimo.vision.models.sam3.video._render_prediction", capture)
    return shown


def test_driver_runs_a_clip_with_bucket_growth_and_release(model, monkeypatch):
    traced_buckets = []

    def counted_commit(*args):
        traced_buckets.append(args[1].slot_status.shape[0])
        return video_commit(*args)

    monkeypatch.setattr("equimo.vision.models.sam3.video._commit", counted_commit)
    num_frames = 12
    driver = Sam3VideoTracker(
        model, num_frames=num_frames, source_size=SOURCE_SIZE, max_objects=3
    )
    driver.set_prompt(PROMPT)
    _script_detector(driver, {0: [0.9, 0.8, 0.7, 0.1], 9: [0.9, 0.2, 0.2, 0.2]})
    _script_new_detections(driver, {9: [0]})

    predictions = []
    for frame_index in range(num_frames):
        frame = FRAME if frame_index % 2 == 0 else OTHER_FRAME
        predictions += driver.push_frame(frame)
    predictions += driver.finish()

    assert [item.frame_index for item in predictions] == list(range(num_frames))
    for item in predictions:
        count = item.object_ids.shape[0]
        assert item.masks.shape == (count, *SOURCE_SIZE)
        assert item.masks.dtype == bool
        assert item.boxes.shape == (count, 4)
        assert item.scores.shape == (count,)

    trace = driver.decisions
    assert [new.object_id for new in trace[0].added] == [0, 1, 2]
    assert trace[0].num_buckets == 2
    removal = next(decision for decision in trace if decision.removed_unmatched)
    assert removal.frame_index == 7
    assert removal.removed_unmatched == (0, 1, 2)
    assert removal.released_buckets == (0, 1)
    reuse = next(decision for decision in trace if decision.frame_index == 9)
    assert reuse.added[0].object_id == 3
    assert (reuse.added[0].bucket, reuse.num_buckets) == (0, 2)

    # The only compilation of commit is at the grown bucket count.
    assert traced_buckets == [2]
    assert driver._state.slot_status.shape[0] == 2
    assert driver._state.bucket_valid.tolist() == [True, False]


def test_prompt_frame_runs_both_passes_and_keeps_one_conditioning_read(model):
    driver = Sam3VideoTracker(model, num_frames=4, source_size=SOURCE_SIZE)
    driver.set_prompt(PROMPT)
    _script_detector(driver, {0: [0.9, 0.2, 0.2, 0.2]})
    driver.push_frame(FRAME)

    assert len(driver.decisions) == 2
    assert [new.object_id for new in driver.decisions[0].added] == [0]
    state = driver._state
    assert state.cond_frame[0].tolist() == [0, -1, -1, -1]
    bucket = jax.tree.map(lambda leaf: leaf[0], state)
    _, _, _, spatial, _ = model.tracker._memory_bank(
        bucket, jnp.asarray(1, jnp.int32), 15
    )
    assert spatial == GRID * GRID


def test_new_objects_are_hidden_until_confirmed_two_frames_later(model, monkeypatch):
    shown = _capture_shown_objects(monkeypatch)
    driver = Sam3VideoTracker(model, num_frames=6, source_size=SOURCE_SIZE)
    driver.set_prompt(PROMPT)
    _script_detector(driver, {1: [0.9, 0.2, 0.2, 0.2]})

    real = driver._steps.associate

    def matched_after_birth(model, detections, tracks, state):
        association, probabilities = real(model, detections, tracks, state)
        new = driver._next_frame == 1
        pairs = np.zeros(association.matches.shape, bool)
        if driver._next_frame in (2, 3):
            pairs[0, 0] = True
        return (
            eqx.tree_at(
                lambda a: (a.new_detection, a.matches, a.high_iom),
                association,
                (
                    jnp.zeros(NUM_QUERIES, bool).at[0].set(new),
                    jnp.asarray(pairs),
                    jnp.asarray(pairs),
                ),
            ),
            probabilities,
        )

    driver._steps.associate = matched_after_birth
    for _ in range(4):
        driver.push_frame(FRAME)
    driver.finish()

    assert shown == [(0, []), (1, [0]), (2, [0]), (3, [0])]


def test_finish_clamps_the_look_ahead_to_the_last_frame(model, monkeypatch):
    shown = _capture_shown_objects(monkeypatch)
    driver = Sam3VideoTracker(model, num_frames=2, source_size=SOURCE_SIZE)
    driver.set_prompt(PROMPT)
    _script_detector(driver, {0: [0.9, 0.2, 0.2, 0.2]})
    assert driver.push_frame(FRAME) == []
    assert driver.push_frame(FRAME) == []
    assert len(driver.finish()) == 2
    assert shown == [(0, []), (1, [])]


def test_driver_runs_real_detection_and_association_on_random_weights(model):
    driver = Sam3VideoTracker(model, num_frames=5, source_size=SOURCE_SIZE)
    driver.set_prompt(PROMPT)
    predictions = []
    for _ in range(5):
        predictions += driver.push_frame(FRAME)
    predictions += driver.finish()
    assert [item.frame_index for item in predictions] == list(range(5))
    for item in predictions:
        assert np.isfinite(item.scores).all()


def test_driver_runs_in_bfloat16(model):
    low_precision = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        model,
    )
    prompt = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        PROMPT,
    )
    driver = Sam3VideoTracker(low_precision, num_frames=3, source_size=SOURCE_SIZE)
    driver.set_prompt(prompt)
    _script_detector(driver, {0: [0.9, 0.2, 0.2, 0.2]})
    predictions = []
    for _ in range(3):
        predictions += driver.push_frame(FRAME.astype(jnp.bfloat16))
    predictions += driver.finish()
    assert [item.frame_index for item in predictions] == [0, 1, 2]
    assert driver._state.last_masks.dtype == jnp.float32
    assert driver._state.last_scores.dtype == jnp.float32
    assert driver._state.cond_memory.dtype == jnp.bfloat16


def test_clips_of_at_least_sixteen_frames_share_one_propagate_compilation(
    model, monkeypatch
):
    traced = []

    def counted_propagate(*args):
        traced.append(args[4])
        return video_propagate(*args)

    monkeypatch.setattr("equimo.vision.models.sam3.video._propagate", counted_propagate)
    for num_frames in (16, 40, None):
        driver = Sam3VideoTracker(model, num_frames=num_frames, source_size=SOURCE_SIZE)
        driver.set_prompt(PROMPT)
        _script_detector(driver, {0: [0.9, 0.2, 0.2, 0.2]})
        _script_new_detections(driver, {})
        for _ in range(2):
            driver.push_frame(FRAME)
    assert traced == [16]


def test_exemplars_reach_only_the_prompt_frame(model):
    driver = Sam3VideoTracker(model, num_frames=3, source_size=SOURCE_SIZE)
    driver.set_prompt(
        ConceptPrompt.create(
            text_features=TEXT,
            boxes=jnp.asarray([[0.5, 0.5, 0.2, 0.2]]),
            points=jnp.asarray([[0.5, 0.5]]),
            max_boxes=2,
            max_points=2,
        )
    )
    real = driver._steps.encode_and_detect
    box_prompted = []
    point_prompted = []
    text_lengths = []

    def recording(model, frame, prompt):
        box_prompted.append(not bool(prompt.box_padding_mask.all()))
        point_prompted.append(not bool(prompt.point_padding_mask.all()))
        text_lengths.append(int((~prompt.text_padding_mask).sum()))
        return real(model, frame, prompt)

    driver._steps.encode_and_detect = recording
    for _ in range(3):
        driver.push_frame(FRAME)
    assert box_prompted == [True, False, False]
    assert point_prompted == [True, False, False]
    assert text_lengths == [TEXT.shape[0]] * 3


def _driver_with_three_objects(model) -> Sam3VideoTracker:
    """Three objects on frame 0, all confirmed on frame 2, and a new one on frame 3."""
    driver = Sam3VideoTracker(
        model, num_frames=6, source_size=SOURCE_SIZE, max_objects=3
    )
    driver.set_prompt(PROMPT)
    _script_detector(driver, {0: [0.9, 0.8, 0.7, 0.1], 3: [0.9, 0.2, 0.2, 0.2]})
    real = driver._steps.associate

    def matched(model, detections, tracks, state):
        association, probabilities = real(model, detections, tracks, state)
        pairs = np.zeros(association.matches.shape, bool)
        np.fill_diagonal(pairs, True)
        return (
            eqx.tree_at(
                lambda a: (a.new_detection, a.matches, a.high_iom),
                association,
                (
                    jnp.zeros(NUM_QUERIES, bool).at[0].set(driver._next_frame == 3),
                    jnp.asarray(pairs),
                    jnp.asarray(pairs),
                ),
            ),
            probabilities,
        )

    driver._steps.associate = matched
    return driver


def test_removed_object_leaves_the_pending_frames_and_its_bucket(model, monkeypatch):
    traced_buckets = []

    def counted_commit(*args):
        traced_buckets.append(args[1].slot_status.shape[0])
        return video_commit(*args)

    monkeypatch.setattr("equimo.vision.models.sam3.video._commit", counted_commit)
    shown = _capture_shown_objects(monkeypatch)
    driver = _driver_with_three_objects(model)
    assert driver.push_frame(FRAME) == []
    assert driver.decisions[0].added[2].object_id == 2
    assert driver.decisions[0].added[2].bucket == 1

    driver.remove_object(2)

    assert driver.policy.bucket_valid.tolist() == [True, False]
    assert driver._state.bucket_valid.tolist() == [True, False]
    assert driver._state.slot_status[1].tolist() == [SLOT_EMPTY] * NUM_SLOTS
    for _ in range(5):
        driver.push_frame(OTHER_FRAME)
    driver.finish()

    assert all(2 not in objects for _, objects in shown)
    assert [objects for frame, objects in shown if frame == 0] == [[0, 1]]
    assert traced_buckets == [2]


def test_removed_object_stays_on_frames_already_returned(model, monkeypatch):
    shown = _capture_shown_objects(monkeypatch)
    driver = _driver_with_three_objects(model)
    returned = []
    for _ in range(3):
        returned += driver.push_frame(FRAME)
    assert [item.frame_index for item in returned] == [0]
    assert shown == [(0, [0, 1, 2])]

    driver.remove_object(1)
    for _ in range(3):
        driver.push_frame(OTHER_FRAME)
    driver.finish()

    assert all(1 not in objects for frame, objects in shown if frame > 0)


def test_redetecting_a_removed_object_starts_a_new_identity(model):
    driver = _driver_with_three_objects(model)
    driver.push_frame(FRAME)
    driver.remove_object(0)
    driver.push_frame(OTHER_FRAME)
    driver.push_frame(FRAME)
    driver.push_frame(OTHER_FRAME)

    redetected = next(item for item in driver.decisions if item.frame_index == 3)
    assert [new.object_id for new in redetected.added] == [3]


def test_removing_an_object_that_is_not_tracked_raises(model):
    driver = _driver_with_three_objects(model)
    driver.push_frame(FRAME)
    with pytest.raises(ValueError, match="Object 7 is not being tracked"):
        driver.remove_object(7)
    driver.remove_object(0)
    with pytest.raises(ValueError, match="Object 0 is not being tracked"):
        driver.remove_object(0)


def test_compiles_once_per_bucket_count_while_objects_change(model, monkeypatch):
    traced_buckets = []

    def counted_commit(*args):
        traced_buckets.append(args[1].slot_status.shape[0])
        return video_commit(*args)

    monkeypatch.setattr("equimo.vision.models.sam3.video._commit", counted_commit)
    driver = Sam3VideoTracker(
        model, num_frames=6, source_size=SOURCE_SIZE, max_objects=4
    )
    driver.set_prompt(PROMPT)
    _script_detector(
        driver,
        {
            0: [0.9, 0.2, 0.2, 0.2],
            2: [0.9, 0.2, 0.2, 0.2],
            4: [0.9, 0.8, 0.7, 0.2],
        },
    )
    _script_new_detections(driver, {2: [0], 4: [0, 1, 2]})
    for _ in range(6):
        driver.push_frame(FRAME)
    assert traced_buckets == [1, 2]


def test_new_object_by_points_is_tracked_alone(model):
    driver = Sam3VideoTracker(model, num_frames=6, source_size=SOURCE_SIZE)
    driver.set_prompt(PROMPT)
    _script_detector(driver, {0: [0.9, 0.2, 0.2, 0.2]})
    _script_new_detections(driver, {})
    driver.push_frame(FRAME)
    before = driver._state

    points = np.asarray([[10.0, 20.0], [40.0, 30.0]])
    track = driver.add_object(1, OTHER_FRAME, points, [1, 0])
    assert track.object_id == 1
    assert track.first_prediction.frame_index == 1
    assert track.first_prediction.masks.shape[1:] == SOURCE_SIZE

    prediction = track.push(FRAME)
    assert prediction.frame_index == 2
    assert prediction.masks.shape[1:] == SOURCE_SIZE
    assert set(prediction.object_ids.tolist()) <= {1}

    # The point track never touches the main tracker's state.
    assert jnp.array_equal(before.last_masks, driver._state.last_masks)
    assert track._state.slot_status[0].tolist() == [SLOT_ACTIVE, SLOT_EMPTY]

    with pytest.raises(ValueError, match="not supported"):
        driver.add_object(1, OTHER_FRAME, points, [1, 0], object_id=0)


def test_point_object_starts_as_a_conditioning_frame(model):
    driver = Sam3VideoTracker(model, num_frames=3, source_size=SOURCE_SIZE)
    points = np.asarray([[10.0, 20.0], [40.0, 30.0]])

    track = driver.add_object(0, FRAME, points, [1, 0])
    assert track._state.recent_is_cond[0, 0]
    assert track._state.cond_frame[0].tolist() == [0, -1, -1, -1]
    assert track._state.slot_status[0].tolist() == [SLOT_ACTIVE, SLOT_EMPTY]


def test_driver_requires_a_prompt_and_rejects_frames_past_the_end(model):
    driver = Sam3VideoTracker(model, num_frames=1, source_size=SOURCE_SIZE)
    with pytest.raises(ValueError, match="set_prompt"):
        driver.push_frame(FRAME)
    driver.set_prompt(PROMPT)
    driver.push_frame(FRAME)
    with pytest.raises(ValueError, match="only has 1"):
        driver.push_frame(FRAME)


def test_driver_accepts_a_prompt_only_before_the_first_frame(model):
    driver = Sam3VideoTracker(model, num_frames=2, source_size=SOURCE_SIZE)
    driver.set_prompt(PROMPT)
    driver.set_prompt(PROMPT)
    driver.push_frame(FRAME)
    with pytest.raises(ValueError, match="before the first frame"):
        driver.set_prompt(PROMPT)
