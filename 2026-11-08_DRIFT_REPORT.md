# SAM 3.1 drift report

Compares `docs/designs/sam3.md` and every `EGM.md` with the working copy. Items the design schedules after the reference
comparison (fixtures, archives, digests, licensing files, user documentation) are not reported, since they are not due
yet. Line numbers refer to the working copy. Each item ends with the suggested direction of the fix.

Contracts the SAM 3.1 work touches were compared with the code for those parts. The others went through automated checks
only: every `[file.py:symbol]` citation resolves to a real symbol, and modules and `__all__` names are mentioned in the
nearest contract.

## docs/designs/sam3.md

- **Float32 attention path is selected differently.** The design uses the chunked XLA path "when the dtype is float32"
  (lines 474-476). `tracker.py:115-118` selects on `implementation` instead: float32 with `"cudnn"` raises `ValueError`
  in `_fused_attention`, and bfloat16 with `"xla"` is chunked too. The contract already describes the code. Fix: update
  the design.
- **cuDNN is not the default for memory attention.** The requirements say memory attention uses "cuDNN in bfloat16"
  (lines 816-817). Every attention module defaults to `implementation="xla"`, and neither `Sam3` nor the driver switches
  it; callers must use `with_attention_implementation`. Fix: state in the design that it is opt-in.
- **The prompt frame is always frame 0.** `set_prompt` "applies from the next pushed frame" (line 274) and the prompt
  frame is processed twice (line 436). `video.py:771` runs the first pass only when `frame_index == 0`, and
  `video.py:248,254` hard-code frame 0. A later `set_prompt` changes detection but runs no prompt-frame pass, and no
  test covers it. Fix: decide between rejecting a prompt after frame 0 and implementing a later prompt frame.
- **A point object changes later object IDs in the main tracker.** The design says the main tracker's "state and outputs
  are not changed" (line 457). `add_object` reserves an ID from the policy's counter (`video.py:223-234,820`), so
  objects the detector adds later get different IDs than they would have. Fix: limit the design's claim to device state
  and masks, since sharing the counter is what prevents ID collisions.
- **`Sam3VideoTracker` has a required argument the interface omits.** `source_size` is keyword-required
  (`video.py:729-736`), but line 273 lists only `num_frames` and `max_objects`. Fix: update the design.
- **`add_object` and `PointTrack` have extra public members.** Line 277 omits `object_id=None` (`video.py:811`), without
  which the "points that name an existing object" rule at line 458 cannot be reached. Line 278 omits `first_prediction`
  (`video.py:678`). Fix: update the design.
- **`Sam3.propagate` and the factories have extra parameters.** `propagate` takes `num_frames` (`model.py:171-172`),
  which the rule at line 407 needs, but line 259 omits it. `sam31_multiplex` and `sam31_text` also take `inference_mode`
  (lines 253 and 266). Fix: update the design.
- **Behavior in the driver that the design does not state.** Objects with no pixel on a frame are omitted
  (`video.py:598`), and the prompt frame's second pass commits only if the policy changes something (`video.py:788`).
  Both are in the contract. Fix: add them to the design.
- **`Association` has no bucket axis.** Lines 298-299 give all four pytrees a leading bucket axis. `Association`
  flattens tracks over buckets and has no bucket axis (`tracker.py:323-338`), and `CommitPlan.frame_index` is a scalar.
  Fix: update the design.
- **`last_masks` and `last_scores` are not used for association.** Line 396 says they are "for association". `associate`
  reads `tracks` and `slot_status` only (`tracker.py:497-547`). The fields feed rendering and the stored prompt-frame
  output (`video.py:549,945`). Fix: update the design.
- **Packed keys are described too broadly.** Line 471 says `propagate` and `commit` pack keys, but `commit` runs no
  memory attention and only `propagate` calls `_memory_bank` (`tracker.py:701`). Lines 472 and 901 give
  `key_value_seq_lengths` the shape `[batch]`, while the code broadcasts it to `q.shape[:-3]` and the tracker passes a
  scalar under `vmap` (`tracker.py:116`). Fix: update the design.
- **`PositionedAttention` is missing from the design.** The `CrossAttention` specified at lines 463-469 cannot express
  separate keys and values or RoPE, so the trunk, detector, two-way transformer and memory attention use a subclass
  (`common.py:64`). Memory attention also bypasses both call paths through `_attend_packed_memory`
  (`tracker.py:98-120`). Fix: update the design.
- **Helpers are in a different file.** Lines 950-952 place the sine position encoding and the transposed RoPE factors in
  `encoder.py`; they are in `common.py:117,135,152`. Fix: update the design.
- **`__all__` exports more than specified.** Line 947 lists the model, factory, typed outputs, prompts and
  `decode_concepts`. `sam3/__init__.py:28,30` also exports `_SAM3_REGISTRY` and `pad_tracker_buckets`, and
  `vision/models/__init__.py:19-21` copies them into `equimo.vision.models`. The tests read the registry by attribute,
  and `equimo.language` does not export its registry. Fix: remove `_SAM3_REGISTRY` from `__all__`, and add
  `pad_tracker_buckets` to the design.
- **`video.py` is not NumPy-only.** Lines 233 and 969 describe the module and driver as NumPy. It imports JAX, Equinox
  and the model (`video.py:41-69`) and renders with `jax.image.resize` (`video.py:585`). Only `AssociationPolicy` is
  NumPy-only, as the contract says. Fix: update the design.
- **Reference citations are coarser than specified.** Lines 445-447 promise a reference location for each threshold and
  tie rule, in the appendix and in the policy docstrings, and line 969 asks for citations with the constants. The
  appendix gives file-level paths only (lines 854-857), `video.py` names two files once, and the thresholds in
  `tracker.py:58-68` have no citation. Fix: decide between adding them during the reference comparison and relaxing the
  design.
- **`core/tags.py` changed outside the plan.** `_add_attention_tags` gained `cross_attn` parents and tuple entries, and
  `tests/finetune/test_tags.py` gained a test, but the plan lists neither. No existing model uses a `cross_attn` path.
  Fix: add both to the plan.
- **The LoRA case is narrower than the journey.** CUJ 5 targets "attention projections" (line 167), and line 301 names
  `q`, `k`, `v` and `proj`. `_lora_encoder_and_detector` in `tests/cases/finetune_cases.py` selects `attention.proj`
  only. Fix: extend the case, or state in the design that only `proj` is targeted.

## src/equimo/vision/models/sam3/EGM.md

- Line 156 says a `PointTrack` "never touches the main tracker's state". That holds for device state, but `add_object`
  advances the policy's object ID counter.
- The interface does not say that the prompt-frame pass only happens on frame 0, so `set_prompt` reads as usable at any
  frame (lines 50-53, 150-152).
- `FrameDecision`'s fields, which form the decision trace the design's video parity criterion compares, are not listed.
  `NewObject`, `Reconditioning` and `TrackedObject` are exported from `video.py` and not mentioned.

## src/equimo/EGM.md

- The design lists this contract as modified alongside `vision/EGM.md` and `vision/models/EGM.md` (line 1076), and only
  those two were updated.
- The vision summary (lines 144-146) does not mention SAM 3.1 or `equimo.vision.ops`.
- The language summary (lines 147-148) names only `TextTransformerEncoder` and the SentencePiece tokenizer, not
  `CausalTextTransformer`, `sam31_text` or `ClipBpeTokenizer`.
- The optional-dependency list (lines 187-189) omits the `clip` extra.

## src/equimo/core/EGM.md

- `tags.py` now gives `cross_attn` parents and their tuple entries the attention tags, and the contract does not mention
  this rule. Only `sam3/EGM.md` states it, as a property of that model.
- `infer_block_depth` is in `tags.py`'s `__all__` and is not mentioned.

## src/equimo/vision/models/EGM.md

- The invariant at line 123 says all forward endpoints take one `[C, H, W]` image. `Sam3.detect`, `predict_interactive`,
  `propagate`, `associate` and `commit` take features, prompts or tracker state.
- The overview (lines 5-7) lists the architecture families without the promptable segmentation family.

## src/equimo/vision/EGM.md

No drift detected in the parts the SAM 3.1 work touches.

## src/equimo/core/layers/EGM.md

No drift detected in the parts the SAM 3.1 work touches.

## src/equimo/language/EGM.md

No drift detected in the parts the SAM 3.1 work touches.

## src/equimo/finetune/EGM.md

- `_serialization_codecs.py`, the private module behind `serialization.py`'s built-in codecs, is not mentioned. Found by
  the automated checks; the rest of this contract was not compared line by line.

## src/equimo/finetune/peft/EGM.md

- `_compat.py` exports eight helpers in `__all__` that are not mentioned: `bundle_get_path`, `is_head_path`,
  `is_linear_like`, `linear_bias`, `linear_from_state`, `linear_state`, `linear_weight` and
  `validate_logical_parameter_schema`. Found by the automated checks; the rest of this contract was not compared line by
  line.

## src/equimo/vision/layers/EGM.md

- `__init__.py` re-exports nine core registry helpers that are not mentioned (`register_act`, `get_dropout`,
  `register_dropout`, `get_ffn`, `register_ffn`, `get_mixer`, `register_mixer`, `get_norm`, `register_norm`). Found by
  the automated checks; the rest of this contract was not compared line by line.

## src/equimo/conversion/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/core/ops/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/core/implicit/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/vision/query_training/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/finetune/language/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/finetune/vision/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/finetune/audio/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/audio/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/audio/models/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/audio/layers/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/tabular/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/tabular/models/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/tabular/layers/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/timeseries/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/timeseries/models/EGM.md

No drift detected by the automated checks; not compared line by line.

## src/equimo/timeseries/layers/EGM.md

No drift detected by the automated checks; not compared line by line.
