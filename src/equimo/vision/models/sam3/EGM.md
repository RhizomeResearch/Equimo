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
  - `encode_image(x) -> ImageFeatures`: `detector`, `interactive`, `propagation`, each three maps at 4G, 2G and G.
    `[model.py:Sam3.encode_image]`, `[encoder.py:ImageFeatures]`
  - `detect(features, ConceptPrompt) -> DetectorOutput`: `logits (Q)` (joint logit), `presence_logit`, `boxes (Q, 4)`
    (normalized cxcywh), `mask_logits (Q, 4G, 4G)` (float32), `layer_boxes (L, Q, 4)`, `layer_logits (L, Q)`.
    `[model.py:Sam3.detect]`, `[detector.py:DetectorOutput]`
  - `predict_interactive(features, InteractivePrompt) -> InteractiveOutput`: `mask_logits (4, 4G, 4G)` (float32),
    `iou (4)`, `object_score_logit`, `selected`. `[model.py:Sam3.predict_interactive]`,
    `[interactive.py:InteractiveOutput]`
- `ConceptPrompt.create(*, text_features, text_padding_mask, boxes, box_labels, points, point_labels, dim=256, max_text_tokens=32, max_boxes=16, max_points=16)`
  pads shorter prompts; absent prompts are fully padded. Padding masks use `True` for padding.
  `[detector.py:ConceptPrompt]`
- `InteractivePrompt.create(*, points, point_labels, box, mask_input, mask_size=288, max_points=16)` pads points with
  label -1; points and boxes are in input pixels. `[interactive.py:InteractivePrompt]`
- `decode_concepts(out, *, output_size, max_detections, threshold=0.5) -> ConceptDetections`: `keep`, `masks`, `boxes`
  (pixels, xyxy), `scores`, `query_index`, `presence`. `[decoding.py:decode_concepts]`,
  `[decoding.py:ConceptDetections]`
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
  larger power of two. `CommitPlan.empty(state, frame_index, source_size)` builds a plan that changes nothing.
  `Sam3.tracker.decode_object(interactive, features, prompt)` returns a prompted object's mask logits, pointer and
  object score. `[model.py:Sam3]`, `[tracker.py:Sam3Tracker]`, `[tracker.py:pad_tracker_buckets]`
- Experimental host driver in `equimo.vision.models.sam3.video`, imported explicitly:
  `Sam3VideoTracker(model, *, num_frames, source_size, max_objects=16)` with `set_prompt(prompt)` (only before the first
  frame; later calls raise `ValueError`), `push_frame(frame) -> list[FramePrediction]`,
  `finish() -> list[FramePrediction]` and
  `add_object(frame_index, frame, points, labels, *, object_id=None) -> PointTrack`. Frames are at the model's input
  size and `source_size` is the clip's `(height, width)`. `decisions` holds one `FrameDecision` per planned pass.
  `[video.py:Sam3VideoTracker]`
- `PointTrack`: `object_id`, `first_prediction` (the frame it was added on) and `push(frame) -> FramePrediction`.
  `[video.py:PointTrack]`
- `FramePrediction`: `frame_index`, `object_ids (K)`, `masks (K, H, W)` (bool, source size), `boxes (K, 4)` (pixels,
  xyxy, before overlaps are resolved), `scores (K)` (detection probability at birth). `[video.py:FramePrediction]`
- `AssociationPolicy(*, num_slots, max_objects)`: NumPy-only decisions, with `plan_prompt_frame(keep, probabilities)`
  and `plan(FrameObservation) -> FrameDecision`, replayable from recorded device outputs. `objects` maps each ID to a
  `TrackedObject` (`object_id`, `bucket`, `slot`, `birth_frame`, `birth_score`, `unmatched_frames`, `matched_streak`,
  `last_occluded_frame`, `confirmed`). `[video.py:AssociationPolicy]`, `[video.py:TrackedObject]`
- `FrameDecision`, the decision trace of one planned pass: `frame_index`, `added` (`NewObject`: `object_id`, `query`,
  `bucket`, `slot`, `score`), `removed_unmatched`, `removed_duplicate`, `reconditioned` (`Reconditioning`: `object_id`,
  `query`), `suppressed`, `confirmed`, `dropped_queries`, `released_buckets`, `reencode_buckets` and `num_buckets`.
  `[video.py:FrameDecision]`, `[video.py:NewObject]`, `[video.py:Reconditioning]`

Experimental pytrees, for `B` buckets of `S` slots, `T = B * S` tracks, grid `G`, `Q` queries and width `dim`; sizes are
for the published configuration in bfloat16 (`G = 72`, `S = 16`, `dim = 256`, about 59 MB per bucket):

| `TrackerState` field               | Shape                        | Dtype        | Content                                                                 |
| ---------------------------------- | ---------------------------- | ------------ | ----------------------------------------------------------------------- |
| `bucket_valid`                     | `(B)`                        | bool         | bucket in use                                                           |
| `slot_status`                      | `(B, S)`                     | int8         | `SLOT_EMPTY`, `SLOT_ACTIVE` or `SLOT_REMOVED`                           |
| `cond_memory`, `cond_features`     | `(B, 4, dim, G, G)` each     | model dtype  | memory (attention values) and features (spatial keys), 10.6 MB each     |
| `cond_pointers`, `cond_frame`      | `(B, 4, S, dim)`, `(B, 4)`   | model, int32 | object pointers and frame index, `-1` when absent                       |
| `recent_memory`, `recent_features` | `(B, 6, dim, G, G)` each     | model dtype  | ring buffer indexed by `frame % 6`, 15.9 MB each                        |
| `recent_frame`, `recent_is_cond`   | `(B, 6)`                     | int32, bool  | frame index, `-1` when absent, and whether that frame was conditioning  |
| `pointers`, `pointer_frame`        | `(B, 15, S, dim)`, `(B, 15)` | model, int32 | ring buffer of object pointers indexed by `frame % 15`                  |
| `last_masks`, `last_scores`        | `(B, S, 4G, 4G)`, `(B, S)`   | float32      | latest mask logits and object score logits, -1024 and 0 for other slots |

- `TrackOutput`: `mask_logits (B, S, 4G, 4G)` (float32), `object_score_logits (B, S)`, `iou (B, S)` and
  `pointers (B, S, dim)`, for each slot's mask with the highest predicted IoU. `[tracker.py:TrackOutput]`
- `Association`: `keep (Q)`, `iom`, `matches`, `high_iom` and `ambiguous` `(Q, T)`, `new_detection` and
  `confident_detection` `(Q)`, `confident_track (T)`, `track_iou (T, T)`. Track axes are flattened over buckets, and the
  detection axis has no bucket axis. `[tracker.py:Association]`
- `CommitPlan`: `frame_index` (scalar), `add`, `decoded`, `recondition`, `suppress`, `remove` and `object_ids` `(B, S)`,
  `new_masks` and `recondition_masks` `(B, S, 4G, 4G)`, `decoded_pointers (B, S, dim)`, `decoded_scores (B, S)`,
  `release`, `reencode` and `state_group` `(B)`, and the static `source_size`. An added slot marked `decoded` takes its
  mask, pointer and score as given; other added slots are decoded from `new_masks` as a mask prompt.
  `[tracker.py:CommitPlan]`

## Invariants

- Self-attention parents are named `attn` and cross-attention parents `cross_attn`, with `q`, `k`, `v` and `proj` linear
  children; a block with two cross-attentions holds them in a tuple named `cross_attn`. Both parents carry the
  `attention` tags, so `attention.proj` selects cross-attention projections too. Blocks sit in lists named `blocks`, and
  the parts trained by head-only fine-tuning sit under `detector.head`. `[detector.py:DetectorHead]`,
  `[encoder.py:ViTDetBlock]`
- The trunk's position table covers one window and is tiled at run time; `img_size` must divide into patches and then
  windows. Each `ViTDetBlock` partitions its own windows, so the blocks stay one flat list. `[encoder.py:ViTDetTrunk]`,
  `[encoder.py:ViTDetBlock]`
- RoPE tables are computed while tracing and rotate width before height, by exchanging `VisionRoPE`'s grid axes.
  `[common.py:exchanged_rope_factors]`
- The two random Fourier matrices are `buffer()` fields read through `stop_gradient`. `[interactive.py:PromptEncoder]`
- Prompt tokens out of use are masked from attention, so their contents have no effect on the output: padded exemplars,
  padded points and the box corners when there is no box. The appended padding point takes part, and an empty prompt
  also encodes one padding point in the first slot. `[interactive.py:PromptEncoder]`, `[detector.py:GeometryEncoder]`
- Attention softmax, RoPE rotation, layer norms, the DETR feed-forward, and mask logits run in float32; outputs return
  in the model dtype, except mask logits, which stay float32. `[common.py:PositionedAttention]`,
  `[common.py:layer_norm]`, `[detector.py:DecoderBlock]`
- The joint logit is `inverse_sigmoid(sigmoid(class) * sigmoid(presence))`; `inverse_sigmoid` clamps to 1e-3, so it
  stays within about +/-6.9. `decode_concepts` does not apply presence again. `[detector.py:joint_logit]`,
  `[decoding.py:decode_concepts]`
- Mask selection counts point labels including the two box corners: with at most one and no mask prompt, the best of the
  three multimask outputs by predicted IoU; otherwise the single-mask output unless its stability score is below 0.98.
  `[interactive.py:_select_mask]`
- `single_frame_nms` measures overlap as intersection over minimum, and a suppressed mask still suppresses later ones;
  `greedy_iom_nms` uses intersection over minimum, and only kept masks suppress. Ranking uses a stable sort.
  `[decoding.py:single_frame_nms]`, `[decoding.py:greedy_iom_nms]`
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
- A released bucket is cleared before the rest of the plan applies, so it equals a bucket from `empty_state` and can
  take new objects in the same `commit`. `[tracker.py:Sam3Tracker.commit]`, `[tracker.py:_empty_buckets]`
- `recent_*` and `pointers` are written at `frame % length` for every valid bucket, so with consecutive frames they hold
  the last 6 and the last 15 frames. The conditioning buffer keeps the 4 latest conditioning frames; a frame committed
  twice replaces its own entry and stays marked as conditioning in the ring buffer. A frame is read once: a conditioning
  frame still in the buffer is skipped in the ring buffers, and one that left the buffer counts as a recent frame.
  `[tracker.py:_selected_memory_frames]`
- Spatial keys are the stored features of a memory frame plus position and temporal embeddings; values are the stored
  memory. Pointer tokens are both, with a temporal encoding added to the keys. `[tracker.py:Sam3Tracker._memory_bank]`
- Frame indices are traced, and every step keeps its shapes while objects change, so `propagate`, `associate` and
  `commit` compile once per bucket count, and `commit` once per `source_size`. `[tracker.py:CommitPlan]`
- `num_frames` only sets the divisor of the pointer temporal encoding, `min(num_frames, 16) - 1` and at least 1. `None`
  means at least 16 frames. `[tracker.py:Sam3Tracker.propagate]`
- `commit` applies reconditioning and suppression, marks tracks left with under 30% of their area by per-pixel overlap
  (higher probability first, ties to the lower object ID) and clamps those masks to -10 for memory only, then encodes
  memory. A bucket with new objects, or listed in `reencode`, is encoded again from masks made non-overlapping per
  `state_group` at `source_size`. `[tracker.py:Sam3Tracker.commit]`, `[tracker.py:_shrunk_slots]`,
  `[tracker.py:_keep_per_group]`
- `associate` drops detections at probability 0.4 or below, runs greedy IoM NMS at 0.1, then the 0.025 boundary filter,
  in that order. `new_detection` needs probability 0.65 and no match, and ignores `keep` only when no slot is active.
  `[tracker.py:associate]`
- The policy's thresholds and order of decisions are in the `video.py` module docstring. New objects go to the live
  bucket with the fewest empty slots that holds the whole batch, or else to free buckets as a new state. The bucket
  count grows in powers of two up to `next_pow2(max_objects)`. `[video.py:AssociationPolicy]`
- `video.py` is never imported by `import equimo`. The policy uses only NumPy, so it replays on CPU from recorded device
  outputs. `[video.py:FrameObservation]`
- The prompt frame is always frame 0, so `set_prompt` raises once a frame has been pushed. It runs twice: alone
  (single-frame NMS, `max_objects`, 0.65, nothing hidden), then as a tracked frame that reuses the stored output and
  commits only if the policy changes something. Frame 0 counts as a reconditioning frame.
  `[video.py:Sam3VideoTracker.set_prompt]`, `[video.py:Sam3VideoTracker.push_frame]`
- Once frame `t` is planned, frame `t - 2` is final; `finish` emits the rest with the look-ahead clamped to the last
  frame. An object is shown from the first frame it was confirmed by two frames later, and later removals do not hide
  frames already shown. Objects with no pixel on a frame are omitted from it. `[video.py:Sam3VideoTracker._render]`
- A `PointTrack` runs on its own single-bucket state, never touches the main tracker's device state, and takes the
  interactive head's point result as its conditioning frame. Its ID comes from the policy's counter, which prevents ID
  collisions, so objects detected later take the IDs after it. `[video.py:PointTrack]`,
  `[video.py:AssociationPolicy.reserve_object_id]`

## Side Effects

- Importing the package registers `"sam3"` in the model registry. `[model.py:Sam3]`
- `sam31_multiplex(pretrained=True)` downloads through `equimo.serialization.load_weights`. `[model.py:sam31_multiplex]`
- `AssociationPolicy` logs a warning when new objects are dropped. `[video.py:AssociationPolicy]`

## Verification

- Run: `devenv shell -- uv run pytest tests/test_sam3.py tests/test_sam3_video.py -q`
- cuDNN attention needs a CUDA machine; see "Environment and Commands" in `AGENTS.md` for running the GPU-only tests.
