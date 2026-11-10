# equimo.vision.models.sam3

## Overview

SAM 3.1 in JAX and Equinox: a ViTDet image encoder with three feature pyramids, a DETR-based concept detector with a
presence score, a SAM 2-style interactive head, and the tracker's prompt encoder, memory encoder, memory attention,
multiplex mask decoder and device tracking steps, plus a host-side video driver. Every endpoint is unbatched and batched
with `vmap`, except that the tracker steps carry their own bucket axis. Text arrives as features from the separate
`sam31_text` model in `equimo.language`. Only the converted weights are under the SAM License; the code stays under
Equimo's license.

## Interface

- `Sam3(*, key, in_channels, img_size, patch_size, embed_dim, depth, num_heads, mlp_features, window_size, global_blocks, dim, head_num_heads, ffn_features, geometry_depth, fusion_depth, decoder_depth, num_queries, two_way_depth, memory_depth, memory_heads, num_slots, memory_start_channels, memory_fuser_depth)`,
  registered as `"sam3"` for `modality="vision"`, with fields `trunk`, `neck`, `detector`, `interactive`, `tracker`.
  `[model.py:Sam3]`
- `sam31_multiplex(pretrained=False, inference_mode=True, **kwargs)`: the published configuration from `_SAM3_REGISTRY`.
  `[model.py:sam31_multiplex]`
- Endpoints, all taking `key=None, inference=True`:
  - `features(x (C, H, W)) -> (dim, G, G)`: the detector pyramid's low-resolution level; the `FeatureSpec` endpoint.
    `[model.py:Sam3.features]`
  - `encode_image(x, *, checkpoint_trunk=False) -> ImageFeatures`: `detector`, `interactive`, `propagation`, each three
    maps at 4G, 2G and G. `checkpoint_trunk` rematerializes the trunk blocks under autodiff, for fine-tuning through the
    trunk. `[model.py:Sam3.encode_image]`, `[encoder.py:ImageFeatures]`
  - `detect(features, ConceptPrompt) -> DetectorOutput`: `logits (Q)` (joint logit), `presence_logit`, `boxes (Q, 4)`
    (normalized cxcywh), `mask_logits (Q, 4G, 4G)` (float32), `layer_boxes (L, Q, 4)`, `layer_logits (L, Q)`.
    `[model.py:Sam3.detect]`, `[detector.py:DetectorOutput]`
  - `predict_interactive(features, InteractivePrompt) -> InteractiveOutput`: `mask_logits (4, 4G, 4G)` (float32),
    `iou (4)`, `object_score_logit`, `selected`. `[model.py:Sam3.predict_interactive]`,
    `[interactive.py:InteractiveOutput]`
- `ConceptPrompt.create(*, text_features, text_padding_mask, boxes, box_labels, points, point_labels, dim=256, max_text_tokens=32, max_boxes=16, max_points=16, source_size=None)`
  pads shorter prompts and raises `ValueError` for longer ones; absent prompts are fully padded. Padding masks use
  `True` for padding. With `source_size` `(height, width)`, boxes are xyxy and points xy in those pixels instead of
  normalized. `without_exemplars()` pads the boxes and points out. To match the reference on exemplars alone, pass the
  features of the text `"<text placeholder>"` rather than none. `[detector.py:ConceptPrompt]`, `[detector.py:_pad_to]`
- `encode_concept(text_model, tokenizer, concept=None, *, boxes, box_labels, points, point_labels, source_size) -> ConceptPrompt`
  tokenizes and encodes the concept, or that placeholder text for exemplars alone, and raises `ValueError` with neither.
  `text_model` and `tokenizer` are arguments so that `equimo.vision` does not import `equimo.language`.
  `[detector.py:encode_concept]`
- `InteractivePrompt.create(*, points, point_labels, box, mask_input, mask_size=288, max_points=16, source_size=None, image_size=1008)`
  pads points with label -1 and raises `ValueError` for more than `max_points`. Points and boxes are in input pixels, or
  in the pixels of `source_size` `(height, width)`, which scale to `image_size`. Point labels default to positive;
  values other than 1, 0 and -1 raise, also under `jit`. `[interactive.py:InteractivePrompt]`
- `decode_concepts(out, *, output_size, max_detections=None, threshold=0.5) -> ConceptDetections`: `keep`, `masks`,
  `boxes` (pixels, xyxy), `scores`, `query_index`, `presence`. `max_detections=None` keeps every query. A query is kept
  at `threshold` or above unless `non_greedy_iom_nms` suppresses it at 0.1; that NMS runs over every query, so one below
  `threshold` can still suppress. Boxes are taken before overlaps are resolved, then each pixel goes to the
  highest-scoring mask. `[decoding.py:decode_concepts]`, `[decoding.py:ConceptDetections]`,
  `[../../segmentation.py:non_greedy_iom_nms]`
- Tracker modules on `Sam3.tracker`: `image_position(dtype)` (the Fourier position map),
  `memory_encoder(features, masks (2 * num_slots, H, W))`,
  `memory_attention(x, x_pos, memory_image, memory, memory_image_pos, *, num_spatial_keys, key_lengths)`, and
  `mask_decoder(image, image_pos, high_res, slot_valid) -> MultiplexOutput`, where `high_res` already went through
  `mask_decoder.upsampler.project`. `[tracker.py:Sam3Tracker]`, `[tracker.py:MemoryEncoder]`,
  `[tracker.py:MemoryAttention]`, `[interactive.py:ImagePositionEncoding]`, `[interactive.py:MultiplexMaskDecoder]`
- Experimental tracking steps on `Sam3`:
  `propagate(state, features, frame_index, key=None, inference=True, *, num_frames=None) -> TrackOutput`,
  `associate(detections, tracks, state) -> Association` and `commit(state, features, tracks, plan) -> TrackerState`.
  `Sam3.tracker.empty_state(num_buckets=1)` builds a state and `pad_tracker_buckets(state, num_buckets)` grows it to a
  larger power of two. `CommitPlan.empty(state, frame_index, source_size)` builds a plan that changes no object, and
  `commit` with it still writes the frame's memory, ring buffers and `last_masks`.
  `Sam3.tracker.decode_object(interactive, features, prompt)` returns a prompted object's mask logits, pointer and
  object score. `Sam3.tracker.commit_point_object(interactive, state, features, prompt, frame_index, source_size)`
  starts a point-prompted object in slot 0 of bucket 0, and
  `Sam3.tracker.commit_point_step(state, features, tracks, frame_index)` stores a propagated frame from its own masks
  and scores. `remove_tracker_slot(state, bucket, slot, released_buckets)` marks a slot removed and clears the buckets
  it releases. `[model.py:Sam3]`, `[tracker.py:Sam3Tracker]`, `[tracker.py:CommitPlan.empty]`,
  `[tracker.py:Sam3Tracker.commit_point_object]`, `[tracker.py:Sam3Tracker.commit_point_step]`,
  `[tracker.py:pad_tracker_buckets]`, `[tracker.py:remove_tracker_slot]`
- Experimental host driver in `equimo.vision.models.sam3.video`, imported explicitly:
  `Sam3VideoTracker(model, *, num_frames, source_size, max_objects=16)` with `set_prompt(prompt)` (only before the first
  frame; later calls raise `ValueError`), `push_frame(frame) -> list[FramePrediction]`,
  `finish() -> list[FramePrediction]`, `remove_object(object_id)` and
  `add_object(frame_index, frame, points, labels, *, object_id=None) -> PointTrack`. Frames are at the model's input
  size and `source_size` is the clip's `(height, width)`. `decisions` holds one `FrameDecision` per planned pass.
  `remove_object` raises `ValueError` for an object that is not being tracked, and hides the object from frames not yet
  returned. `push_frame` raises `ValueError` before `set_prompt`, after `finish` and past `num_frames`; `add_object`
  raises `ValueError` for an ID already in use. `[video.py:Sam3VideoTracker]`, `[video.py:Sam3VideoTracker.push_frame]`,
  `[video.py:AssociationPolicy.reserve_object_id]`
- `PointTrack`: `object_id`, `first_prediction` (the frame it was added on) and `push(frame) -> FramePrediction`. Its
  `scores` hold, on every frame, the sigmoid of its object-score logit on the frame it was added on.
  `[video.py:PointTrack]`
- `FramePrediction`: `frame_index`, `object_ids (K)`, `masks (K, H, W)` (bool, source size), `boxes (K, 4)` (pixels,
  xyxy, before overlaps are resolved), `scores (K)` (for detected objects, detection probability at birth, or -1e4 on
  the frame an object is removed). `[video.py:FramePrediction]`
- `AssociationPolicy(*, num_slots, max_objects)`: NumPy-only decisions, with
  `plan_prompt_frame(candidates, keep, probabilities)` and `plan(FrameObservation) -> FrameDecision`, replayable from
  recorded device outputs. `objects` maps each ID to a `TrackedObject` (`object_id`, `bucket`, `slot`, `birth_frame`,
  `birth_score`, `unmatched_frames`, `matched_streak`, `last_occluded_frame`, `confirmed`). Also `num_buckets`,
  `reserve_object_id(object_id=None)` (an ID for an object tracked outside the policy),
  `remove_object(object_id) -> released_buckets` and `live_objects()` (active slots, by increasing ID).
  `[video.py:AssociationPolicy]`, `[video.py:TrackedObject]`, `[video.py:AssociationPolicy.reserve_object_id]`,
  `[video.py:AssociationPolicy.remove_object]`, `[video.py:AssociationPolicy.live_objects]`
- `FrameObservation`, the NumPy input of `plan`: `frame_index`, `detection_probabilities` and every `Association` field
  except `iom`. `from_device(frame_index, association, detection_probabilities)` copies them to the host.
  `[video.py:FrameObservation]`
- `FrameDecision`, the decision trace of one planned pass: `frame_index`, `added` (`NewObject`: `object_id`, `query`,
  `bucket`, `slot`, `score`), `removed_unmatched`, `removed_duplicate`, `reconditioned` (`Reconditioning`: `object_id`,
  `query`), `suppressed`, `confirmed`, `dropped_queries`, `released_buckets`, `reencode_buckets`, `new_state_buckets`
  and `num_buckets`. `[video.py:FrameDecision]`, `[video.py:NewObject]`, `[video.py:Reconditioning]`

Experimental pytrees, for `B` buckets of `S` slots, `T = B * S` tracks, grid `G`, `Q` queries and width `dim`; sizes are
for the published configuration in bfloat16 (`G = 72`, `S = 16`, `dim = 256`, about 59 MB per bucket):

| `TrackerState` field               | Shape                        | Dtype        | Content                                                                 |
| ---------------------------------- | ---------------------------- | ------------ | ----------------------------------------------------------------------- |
| `bucket_valid`                     | `(B)`                        | bool         | bucket in use                                                           |
| `slot_status`                      | `(B, S)`                     | int8         | `SLOT_EMPTY`, `SLOT_ACTIVE` or `SLOT_REMOVED`                           |
| `cond_memory`, `cond_features`     | `(B, 4, dim, G, G)` each     | model dtype  | memory (attention values and keys) and features (keys), 10.6 MB each    |
| `cond_pointers`, `cond_frame`      | `(B, 4, S, dim)`, `(B, 4)`   | model, int32 | object pointers and frame index, `-1` when absent                       |
| `recent_memory`, `recent_features` | `(B, 6, dim, G, G)` each     | model dtype  | ring buffer indexed by `frame % 6`, 15.9 MB each                        |
| `recent_frame`, `recent_is_cond`   | `(B, 6)`                     | int32, bool  | frame index, `-1` when absent, and whether that frame was conditioning  |
| `pointers`, `pointer_frame`        | `(B, 15, S, dim)`, `(B, 15)` | model, int32 | ring buffer of object pointers indexed by `frame % 15`                  |
| `last_masks`, `last_scores`        | `(B, S, 4G, 4G)`, `(B, S)`   | float32      | latest mask logits and object score logits, -1024 and 0 for other slots |

- `MultiplexOutput`, for one bucket: `mask_logits (S, 3, 4G, 4G)`, `iou (S, 3)`, `object_score_logits (S)` and
  `mask_tokens (S, 3, dim)`, with three candidate masks per slot. `[interactive.py:MultiplexOutput]`
- `TrackOutput`: `mask_logits (B, S, 4G, 4G)` (float32), `object_score_logits (B, S)`, `iou (B, S)` and
  `pointers (B, S, dim)`, for each slot's mask with the highest predicted IoU. `[tracker.py:TrackOutput]`
- `Association`: `keep (Q)`, `iom`, `matches`, `high_iom` and `unique_high_iom` `(Q, T)`, `new_detection` and
  `confident_detection` `(Q)`, `confident_track` and `track_nonempty` `(T)`, `track_iou` and `reconditioned_iou`
  `(T, T)`, and `reconditioned_nonempty (T)`. Track axes are flattened over buckets, and the detection axis has no
  bucket axis. Only kept detections and active slots have overlaps; every other pair is 0. Rows and columns with two or
  more overlaps of at least 0.5 are zeroed for `matches` and `unique_high_iom`, but not for `iom` and `high_iom`.
  `reconditioned_iou` and `reconditioned_nonempty` hold what the track masks would give after reconditioning.
  `[tracker.py:Association]`, `[tracker.py:associate_tracks]`
- `CommitPlan`: `frame_index` (scalar), `add`, `recondition`, `suppress`, `remove`, `conditioning` and `object_ids`
  `(B, S)`, `new_masks` and `recondition_masks` `(B, S, 4G, 4G)`, `release`, `reencode`, `new_state` and `state_group`
  `(B)`, and the static `source_size`. `state_group` assigns each bucket to the state it was created with. The host must
  list in `reencode` every bucket of a state that received objects and in `new_state` the buckets of a state created on
  this frame, which store it as their conditioning frame; `conditioning` flags the slots born on this frame, added to or
  reconditioned on it. `[tracker.py:CommitPlan]`
- `detection_keep(detections) -> (after_nms, keep)` and
  `associate_tracks(detections, after_nms, keep, tracks, state) -> Association` expose the filter and the association
  given a recorded keep mask. They, `remove_tracker_slot` and the module-level `associate` are in `tracker.__all__` but
  not exported by the package, so import them from `equimo.vision.models.sam3.tracker`. `[tracker.py:detection_keep]`,
  `[tracker.py:associate_tracks]`, `[tracker.py:__all__]`, `[__init__.py:__all__]`

## Invariants

- Self-attention parents are named `attn` and cross-attention parents `cross_attn`, with `q`, `k`, `v` and `proj` linear
  children; a block with two cross-attentions holds them in a tuple named `cross_attn`. Both parents carry the
  `attention` tags, so `attention.proj` selects cross-attention projections too. Attention blocks sit in lists named
  `blocks`, and the parts trained by head-only fine-tuning sit under `detector.head`. `[detector.py:DetectorHead]`,
  `[encoder.py:ViTDetBlock]`, `[interactive.py:TwoWayTransformer]`, `[../../../core/tags.py:_add_attention_tags]`
- The trunk's position table covers one window and is tiled at run time; `img_size` must divide into patches and then
  windows. Each `ViTDetBlock` partitions its own windows, so the blocks stay one flat list. `[encoder.py:ViTDetTrunk]`,
  `[encoder.py:ViTDetBlock]`
- `dim` must be divisible by 8, or building the detector raises `ValueError`. `[detector.py:ConceptDetector]`
- RoPE tables are computed while tracing and rotate width before height, by exchanging `VisionRoPE`'s grid axes.
  `[../../layers/posemb.py:VisionRoPE.width_first_factors]`
- The two random Fourier matrices are `buffer()` fields read through `stop_gradient`. `[interactive.py:PromptEncoder]`
- Prompt tokens out of use are masked from attention, so their contents have no effect on the output: padded exemplars,
  padded points and the box corners when there is no box. The appended padding point takes part, and an empty prompt
  also encodes one padding point in the first slot. `[interactive.py:PromptEncoder]`, `[detector.py:GeometryEncoder]`
- Attention softmax on the XLA path, RoPE rotation, layer norms, the DETR feed-forward, and mask logits run in float32;
  outputs return in the model dtype, except mask logits, which stay float32. cuDNN attention runs in its bfloat16 or
  float16 inputs and raises `ValueError` for any other dtype. `[../../../core/layers/attention.py:CrossAttention]`,
  `[../../../core/layers/norm.py:layer_norm]`, `[detector.py:DecoderBlock]`,
  `[../../../core/layers/attention.py:_fused_attention]`
- The joint logit is `inverse_sigmoid(sigmoid(class) * sigmoid(presence))`; `inverse_sigmoid` clamps to 1e-3, so it
  stays within about +/-6.9. `decode_concepts` does not apply presence again. `[detector.py:joint_logit]`,
  `[decoding.py:decode_concepts]`
- Mask selection counts point labels including the two box corners: with at most one and no mask prompt, the best of the
  three multimask outputs by predicted IoU; otherwise the single-mask output unless its stability score is below 0.98.
  `[interactive.py:_select_mask]`
- A single frame uses `non_greedy_iom_nms` at 0.1, and tracking uses `greedy_iom_nms`. `[decoding.py:decode_concepts]`,
  `[tracker.py:detection_keep]`
- The multiplex decoder sets the mask logits of a slot whose object score is not positive to -1024.
  `[interactive.py:MultiplexMaskDecoder]`
- Memory attention expects the valid keys packed first: spatial keys of each frame, then object pointers, then unused
  keys. RoPE repeats per memory frame and leaves pointers unrotated. Without cuDNN, queries run in blocks of 1,024,
  because float32 logits against every key would not fit in memory. `[tracker.py:MemoryAttention]`,
  `[tracker.py:_memory_key_rotary]`, `[tracker.py:_chunked_attention]`
- `MemoryEncoder` resizes masks to 16 times the feature grid in float32 before downsampling.
  `[tracker.py:MemoryEncoder]`
- A slot goes from empty to active and from active to removed. Objects are only added to empty slots, so a removed slot
  stays removed until its bucket is released. `[tracker.py:Sam3Tracker.commit]`
- A released bucket is cleared after this frame's write, so objects removed on it stay in its memory and masks on that
  frame, and the next frame sees a bucket from `empty_state`. Its `last_masks` and `last_scores` stay until the next
  commit, for the driver to show the frame. `[tracker.py:Sam3Tracker.commit]`, `[tracker.py:_empty_buckets]`
- `recent_*` and `pointers` are written at `frame % length` for every valid bucket, so with consecutive frames they hold
  the last 6 and the last 15 frames. The conditioning buffer keeps the 4 latest conditioning frames; a frame committed
  twice replaces its own entry and stays marked as conditioning in the ring buffer. Only a state created on a frame gets
  a conditioning entry, so a bucket normally has one. A frame is read once: a conditioning frame still in the buffer is
  skipped in the ring buffers, and one that left the buffer counts as a recent frame.
  `[tracker.py:_selected_memory_frames]`
- Spatial keys add a memory frame's stored features and its stored memory, each through its own projection, to position
  and temporal embeddings; values are the stored memory. Queries add the current image's features through another
  projection. Pointer tokens are both keys and values, with zero image features and a temporal encoding added to the
  keys. `[tracker.py:Sam3Tracker._memory_bank]`, `[tracker.py:MemoryAttentionBlock]`
- Frame indices are traced, and every step keeps its shapes while objects change, so `propagate`, `associate` and
  `commit` compile once per bucket count, and `commit` once per `source_size`. `[tracker.py:CommitPlan]`
- `num_frames` only sets the divisor of the pointer temporal encoding, `min(num_frames, 16) - 1` and at least 1. `None`
  means at least 16 frames, and the driver passes this capped value, so clips of 16 frames or more share one compiled
  `propagate`. `[tracker.py:Sam3Tracker.propagate]`, `[video.py:Sam3VideoTracker]`
- `commit` applies reconditioning and suppression, including to objects removed on this frame, clamps masks that lose
  over 70% of their area to a per-pixel overlap over all objects (or that start empty) to -10 for memory only, and
  encodes memory with `2 * sigmoid - 1`, 0 for absent slots, and `no_object_memory` for every slot that does not appear.
  The memory of a bucket listed in `reencode` is instead encoded from the consolidated output, which starts from the
  propagated `tracks.mask_logits`, so suppression and the area clamp do not reach it: new and reconditioned masks
  replace their rows after passing through `source_size` (`resize_bilinear_antialias`), other rows keep their
  low-resolution masks, overlaps are resolved per `state_group` at the image size, and a slot appears when its score is
  positive. `last_masks` keeps the reconditioned and suppressed logits either way, with the new masks in added slots.
  `[tracker.py:Sam3Tracker.commit]`, `[tracker.py:_suppress_area_shrinkage]`, `[tracker.py:_non_overlapping_per_state]`,
  `[tracker.py:_masks_through_source_size]`
- `associate` drops detections at probability 0.4 or below, runs greedy IoM NMS at 0.1, then the 0.025 boundary filter,
  in that order. `new_detection` needs probability 0.65, no raw overlap of at least 0.1, and `keep`, except that the
  boundary filter is ignored when no slot is active. `[tracker.py:associate]`, `[tracker.py:associate_tracks]`
- The policy's thresholds, order of decisions and reference locations are in the `video.py` module docstring. The object
  cap counts objects before this frame's removals. New objects go to the state with the fewest empty slots that holds
  the whole batch, or else to free buckets as a new state. The bucket count grows in powers of two up to
  `next_pow2(max_objects)`. `[video.py:AssociationPolicy]`
- `video.py` is never imported by `import equimo`. The policy uses only NumPy, so it replays on CPU from recorded device
  outputs. `[video.py:FrameObservation]`
- The prompt frame is always frame 0, so `set_prompt` raises once a frame has been pushed. It runs twice: alone
  (single-frame NMS, `max_objects`, 0.65), then as a tracked frame that reuses the stored output and runs the policy and
  `commit` as any other frame. Frame 0 counts as a reconditioning frame. Exemplars apply to that frame only, and the
  text to every frame. `[video.py:Sam3VideoTracker.set_prompt]`, `[video.py:Sam3VideoTracker.push_frame]`
- Once frame `t` is planned, frame `t - 2` is final; `finish` emits the rest with the look-ahead clamped to the last
  frame. An object is hidden while unconfirmed and live by then, and later removals do not hide frames already shown.
  Objects with no pixel on a frame are omitted before overlaps are resolved, so an object can be shown with an empty
  mask. `[video.py:Sam3VideoTracker._render]`, `[video.py:_render_prediction]`
- A `PointTrack` runs on its own single-bucket state and never touches the main tracker's device state. The interactive
  head's point result, through the source frame size, is its conditioning frame; later frames store memory from their
  own masks and object scores. Its ID comes from the policy's counter, which prevents ID collisions, so objects detected
  later take the IDs after it. `[video.py:PointTrack]`, `[video.py:AssociationPolicy.reserve_object_id]`

## Side Effects

- Importing the package registers `"sam3"` in the model registry. `[model.py:Sam3]`
- `sam31_multiplex(pretrained=True)` downloads through `equimo.serialization.load_weights`. `[model.py:sam31_multiplex]`
- `AssociationPolicy` logs a warning when new objects are dropped. `[video.py:AssociationPolicy]`

## Verification

- Run: `devenv shell -- uv run pytest tests/test_sam3.py tests/test_sam3_video.py -q`
- Full-weight parity: `devenv shell -- uv run pytest -m reference_parity tests/test_sam3_reference.py -q`. It needs the
  converted archives in `~/.cache/equimo/sam31`, and skips without them unless `EQUIMO_REQUIRE_REFERENCE_CACHE=1` makes
  a missing archive fail; the video tests also need the hosted video fixture.
- cuDNN attention needs a CUDA machine; see "Environment and Commands" in `AGENTS.md` for running the GPU-only tests.
