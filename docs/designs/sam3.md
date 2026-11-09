# SAM 3.1 in Equimo

- **Author:** clementpoiret
- **Status:** Accepted
- **Date:** 2026-11-07

______________________________________________________________________

## Overview

This design adds SAM 3.1 (Meta, arXiv 2511.16719, checkpoint `facebook/sam3.1`, file `sam3.1_multiplex.pt`) to the core
`equimo` package as JAX/Equinox code, with converted pretrained weights hosted in `poiretclement/equimo`. It covers
image and video prediction:

- **Concept segmentation on images:** find every instance of a concept given a text prompt, box exemplars, or both, with
  a presence score that says whether the concept appears at all.
- **Interactive segmentation on images:** points, boxes and mask inputs, with multimask output.
- **Video:** text- or exemplar-prompted detection followed by tracking, forward in time, with the detector-to-tracker
  association policy that decides when objects are added, kept, suppressed or removed. Tracking supports more than 16
  objects through several multiplex buckets. Points on a video frame can add a new object.

The code is written for Equimo and stays under Equimo's `MIT AND Apache-2.0` license. Only the converted weights are
under the SAM License, as with DINOv3. This design was written from reading the reference implementation
(`facebookresearch/sam3` at `f1647e8`) and records behavior, not code. The implementation is drafted from this design,
Equimo's code, the SAM 3 paper and the SAM 3.1 release post; the reference is then read to find and fix behavior
differences, and each fix is logged in a provenance note. The reference also generates the test fixtures.

The work is delivered in two milestones with separate acceptance: **Image** (encoders, detector, interactive head,
tracker modules, archives, fine-tuning) and **Video** (tracking state, device steps, association policy, driver).

## The Problem

SAM 3.1 is only available as PyTorch code that is hard to use for research outside Meta's own pipeline:

- **It cannot be transformed.** The reference keeps a bfloat16 autocast context open for the life of the predictor, and
  the vision encoder's MLP casts to bfloat16 and refuses to run when gradients are enabled. Fine-tuning or distilling
  the encoder requires patching the reference first. None of it composes with `jax.jit`, `jax.vmap` or `jax.grad`.
- **It assumes one deployment shape.** The video predictor builder moves modules to CUDA unconditionally, expects
  FlashAttention 3, and wraps tracking in a stateful session API (`start_session`, `add_prompt`, `propagate_in_video`)
  that owns video decoding, device placement and object bookkeeping. A researcher who wants the detector's logits, the
  tracker's memory, or one tracking step on frames they already hold has to work around that API.
- **Equimo has no promptable or open-vocabulary segmentation model.** Equimo's segmentation models (EoMT, PMT) are
  closed-vocabulary. Users who want open-vocabulary detection, interactive masks or video object tracking in JAX have no
  option in Equimo today.

The users are the author's own research (inference, fine-tuning, distillation) and anyone who wants SAM 3.1 in JAX. They
need modules that follow Equimo's conventions (unbatched modules batched with `vmap`, explicit PRNG keys, bfloat16
support, Equimo archives with pinned digests) and that produce the same outputs as the reference, so that results
obtained with Equimo can be compared with results obtained with Meta's code.

## Goals

Image milestone:

- **Image concept segmentation.** Given a preprocessed 1008 x 1008 image and a text prompt (as token ids), box
  exemplars, or both, return per-query masks, boxes, scores and the presence score.
- **Image interactive segmentation.** Given points, boxes or a low-resolution mask input, return masks, predicted IoU
  scores and the selected multimask output.
- **Fine-tuning.** LoRA and head-only fine-tuning work through `equimo.finetune` on the image path, in bfloat16, at
  batch size 1 on a 16 GB GPU (RTX 4080), with opt-in trunk rematerialization if the measurement requires it.
- **Pretrained weights with license compliance.** Converted weights are hosted in `poiretclement/equimo` with pinned
  SHA-256 digests. The SAM License ships with the weights on Hugging Face and in this repository under
  `LICENSES/pretrained/`, and the license tables and NOTICE files list it.

Video milestone:

- **Video tracking.** Given frames the caller already decoded, a host-side driver runs detection, tracking and the
  association policy and returns per-frame masks with stable object IDs. Prompts are text for every frame, box exemplars
  for the prompt frame only, as in the reference, and points on a frame to add a new object. Tracking is forward-only
  and causal. The number of tracked objects is capped by a configurable `max_objects` (default 16, as in the reference
  3.1 builder) and can be raised above 16.

Both milestones:

- **Transformable device code.** Device steps trace under `jax.jit`, `jax.vmap` and `jax.grad`. Gradients flow through
  features, logits, masks and scores; thresholds, NMS and selection pass none. Changing the set of tracked objects does
  not trigger recompilation unless the bucket count grows.
- **Output parity with the reference**, measured against the reference patched to run in float32, with criteria fixed
  before any measurement:
  - tiny random-weight fixtures: `allclose` with `rtol` and `atol` 1e-5, as for existing Equimo models;
  - full weights on a real image: mean absolute error of at most 5e-4 per component output, with the measured values
    recorded in the fixture's provenance;
  - full weights on a small video: the same decision trace (objects added, suppressed, removed, reconditioned,
    confirmed) and the same object IDs on every frame, with per-object mask IoU of at least 0.99.
- **Bfloat16 support.** The image and video pipelines run in bfloat16. Compared with Equimo's float32 output on the same
  inputs: on images, mask IoU of at least 0.97 per object, score differences of at most 0.02, and the same detections at
  probability 0.5 or more; on video, the identical decision trace and object IDs on every frame and a median per-object
  mask IoU of at least 0.97 on each frame (see "Per-object bfloat16 mask IoU on video" for why the median).
- **Code license unchanged.** The code stays under `MIT AND Apache-2.0`. A provenance note in `docs/licensing/` records
  how the code was written, including each change made after reading the reference and the test that showed the
  difference.
- **Tests without the gated weights.** CI covers every component with tiny fixtures generated by the reference from
  random weights. Full-weight parity runs in the scheduled reference job.

## Non-Goals

- **Image and video preprocessing.** Decoding, resizing to 1008 x 1008 and normalization belong in justdata (the
  author's data-loading library, `../justdata`). Equimo's parity tests feed tensors the reference already preprocessed.
  Equimo keeps the decoding of model outputs (score thresholds, box rescaling, mask upsampling).
- **Video decoding** (MP4 files, JPEG folders).
- **The reference session API** (`handle_request`, start, add, propagate, remove, reset and close session).
- **Refining an existing object with points on video.** Deferred; see Future Improvements.
- **Reverse propagation and conditioning on later frames.**
- **Changing the concept prompt after the first frame.** `set_prompt` raises once a frame has been pushed.
- **Float32 inference on full-resolution video** as a supported use. Float32 is used for parity tests on short clips.
- **Multi-GPU object placement**, all-gathers, CPU offloading, `torch.compile` warm-up, FlashAttention 3 and the
  `use_rope_real` variant.
- **Triton post-processing:** connected components, hole filling and sprinkle removal, which the 3.1 builder disables.
- **Training:** Hydra configs, SAM 3 losses and recipes, dropout and drop-path, and full fine-tuning on a 16 GB GPU.
- **SA-Co evaluation**, `sam3/agent`, and visualization utilities.
- **The SAM 3 (November 2025) checkpoint** and its per-object tracker. The 3.1 checkpoint already includes a detector
  and an image-level interactive head.
- **Following upstream after `f1647e8`.** Later upstream changes are a separate piece of work.
- **A separate `equimo-sam3` wheel**, and similar splits for other families whose weights carry restrictive licenses.

## Open Questions

- **Checkpoint contents (resolved).** `sam3.1_multiplex.pt` is a bare state dictionary of 1,623 tensors, SHA-256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`. It stores the ViT RoPE tables (`freqs_cis`) and
  the unused `text_projection`, and has no stale tracker backbone keys. The converter classifies every key and fails on
  unknown ones.
- **Antialiased resize.** The memory encoder resizes 1008 to 1152 with torch's `antialias=True`. For upsampling this is
  expected to equal plain bilinear interpolation; an operator fixture confirms it.

Risks with a known fallback (memory on a 16 GB GPU, the reference GELU, the test clip) are in Risks and Mitigations.

## Critical User Journeys (CUJs)

### CUJ 1: Find every instance of a concept in an image

- **Given**: a 1008 x 1008 image preprocessed by the caller (for example with justdata), the `sam31_multiplex` vision
  model and the `sam31_text` text model loaded in bfloat16, and the `clip` extra installed for the tokenizer.
- **When**: the caller tokenizes "yellow school bus", encodes it with the text model, encodes the image, runs the
  detector, and decodes the result at the original image size.
- **Then**: the caller gets one mask, box, score and object index per detected bus, and the presence score. In float32
  the kept set and masks match the reference predictor run on the same image.

### CUJ 2: Click to segment an object in an image

- **Given**: the same image features as CUJ 1.
- **When**: the caller passes one positive point (or a box, or a previous low-resolution mask) to the interactive head.
- **Then**: the caller gets four mask logits, their predicted IoU, an object score and the selected index. The selection
  counts point labels, including the two box corners: with at most one label, the best of the three multimask outputs by
  predicted IoU is selected; with more, the single-mask output is selected unless its stability score is below 0.98, in
  which case the best multimask output is selected.

### CUJ 3: Track every instance of a concept through a video

- **Given**: frames the caller decoded and preprocessed, the clip length, a text prompt, and the default
  `max_objects=16`.
- **When**: the caller creates the video driver and feeds frames one at a time.
- **Then**: from the third frame on, each call returns the masks, boxes, object IDs and scores for the frame two steps
  earlier. New objects appear, duplicates and lost objects are removed during their first 15 frames, and the device code
  recompiles only when the bucket count reaches a new power of two.

### CUJ 4: Add a missed object with clicks on a video

- **Given**: a video the driver has processed, and an object the detector missed on frame 40.
- **When**: the caller adds a positive and a negative point for a new object on frame 40, then supplies frames 41
  onward.
- **Then**: the new object gets an ID, is segmented on frame 40 by the interactive head, and is tracked on its own over
  the supplied frames, as the reference's partial propagation does. Outputs already returned for other objects do not
  change.

### CUJ 5: Fine-tune the detector with LoRA

- **Given**: a labeled segmentation dataset and the `sam31_multiplex` model in bfloat16.
- **When**: the caller builds a LoRA plan with `equimo.finetune` over the encoder and detector attention projections, or
  a head-only plan over the detector heads, and trains on a 16 GB GPU at batch size 1.
- **Then**: gradients are finite, only the selected parameters change, and the plan saves and reloads as a delta.

## Technical Plan (Architecture)

### Terminology

- **Slot:** one object position inside a bucket. A slot is empty, active or removed.
- **Bucket:** 16 slots that the multiplex tracker processes in one forward pass; memory attention runs once per bucket.
- **State:** the reference's group of buckets created together from one batch of new objects. It matters for memory
  re-encoding, which the reference runs per state.
- **Conditioning frame:** a frame a state stores as a prompt. It is created when a state is created, by the detector or
  by points. Objects added to an existing state, and reconditioned objects, only set flags in the memory of an ordinary
  frame.
- **Hidden:** an object present in the state that is left out of a frame's output, because it is unconfirmed and still
  live two frames later.
- **IoM (intersection over minimum):** mask intersection divided by the smaller mask's area. The video policy uses it
  where SAM 2 used IoU.
- **Hot-start:** an object's first 15 frames, during which the policy may remove it as unmatched or as a duplicate.
- **Presence score:** a per-image logit saying whether the prompted concept appears at all; the 3.1 detector folds it
  into each query's joint score.
- **Identity hash:** the hash Equimo records for a module's static configuration (`_legacy_layouts.py`); changing a
  static field changes it and invalidates saved fine-tuning deltas.
- **TIPS, T0:** existing Equimo families. TIPS ships its text encoder as a separate language archive; T0 was translated
  from an Apache-2.0 upstream.

### Overview of the pieces

The work adds one vision model, one language model, a tokenizer, a few generic layers and operations, a host-side video
driver, and the conversion and licensing material. The text encoder is a separate language model with its own archive,
so the vision model takes text features, not token ids.

```text
                 caller (justdata or own code): decode, resize to 1008, normalize with mean = std = 0.5
                                   |
         text ---------------------+------------------------------- image / frame [3, 1008, 1008]
          |                                                                   |
   ClipBpeTokenizer (host)                                         Sam3.encode_image  [jit]
          |                                                    ViTDet trunk -> tri-neck: detector,
   CausalTextTransformer [jit]                                 interactive and propagation levels
   (sam31_text archive)                                                       |
          |  text features [32, 256] + padding mask     +---------------------+--------------------+
          +-------------------------------------------> |                     |                    |
                                                  Sam3.detect [jit]   Sam3.predict_interactive  video only:
                                                  geometry encoder,   [jit] prompt encoder,     Sam3.propagate [jit]
                                                  fusion encoder,     mask decoder               memory attention,
                                                  DETR decoder with                              multiplex decoder
                                                  presence, scoring,                                   |
                                                  segmentation head                         Sam3.associate [jit]
                                                       |                                    IoM, NMS, flags
                                         decode_concepts [jit]                                         |
                                         (image path)                                   host policy (NumPy):
                                                                                        counters, removals,
                                                                                        new IDs, buckets,
                                                                                        confirmation
                                                                                               |
                                                                                   Sam3.commit [jit]
                                                                                   memory encoder, slot
                                                                                   writes -> new state
```

### Placement

- **Names.** Family `sam31`; vision archive and factory `sam31_multiplex`; text archive and factory `sam31_text`;
  registry name `"sam3"`, class `Sam3`; registry name `"causal_text_transformer"`, class `CausalTextTransformer`.
- `equimo.vision.models.sam3`: a subpackage with the `Sam3` model registered as `"sam3"` with `modality="vision"`, its
  submodules, the device-side tracking steps, image decoding, and a lazily imported `video` module holding the host
  driver and the policy (the policy uses only NumPy; the module is never imported by `import equimo`).
- `equimo.language`: `CausalTextTransformer`, a CLIP-style text transformer registered as `"causal_text_transformer"`
  with `modality="language"`, and `ClipBpeTokenizer`.
- `equimo.core.layers.attention`: `CrossAttention`, a multi-head attention with separate query and key-value inputs
  (needed by the fusion encoder, DETR decoder, geometry encoder, segmentation head and two-way transformer), and
  per-batch key lengths and an additive bias in `_fused_attention`.
- `equimo.vision.ops` (new module, imported explicitly): `roi_align` and bilinear `grid_sample`. Resizing uses
  `jax.image.resize` directly.
- `VisionRoPE` is reused unchanged.

The `Sam3` vision archive is about 519M parameters (2.1 GB in float32): trunk 446M, tri-neck 21M, detector heads 33M,
tracker 20M. The text archive is about 354M parameters (1.4 GB). Image-only users load the tracker's 20M parameters too;
splitting it out would save under 4%.

### Public interface

All model endpoints take `key: PRNGKeyArray | None = None` and `inference: bool = True` per `AGENTS.md`, run unbatched,
and are batched with `vmap`. Shapes below are for the 1008 x 1008 input; tiny test configurations scale them.

```text
sam31_multiplex(pretrained: bool = False, inference_mode: bool = True, **kwargs) -> Sam3   # registered "sam3", modality "vision"
Sam3 fields: trunk, neck, detector, interactive, tracker
  features(x[3, H, W]) -> [256, 72, 72]                    # detector-level map; the FeatureSpec endpoint
  encode_image(x[3, H, W], *, checkpoint_trunk: bool = False) -> ImageFeatures
  detect(features: ImageFeatures, prompt: ConceptPrompt) -> DetectorOutput
  predict_interactive(features: ImageFeatures, prompt: InteractivePrompt) -> InteractiveOutput
  propagate(state: TrackerState, features: ImageFeatures, frame_index: int32[], *, num_frames: int | None = None) -> TrackOutput   # experimental
  associate(detections: DetectorOutput, tracks: TrackOutput, state: TrackerState) -> Association   # experimental
  commit(state, features, tracks, plan: CommitPlan) -> TrackerState                                # experimental

decode_concepts(out: DetectorOutput, *, output_size: tuple[int, int], max_detections: int | None = None,
                threshold: float = 0.5) -> ConceptDetections

encode_concept(text_model, tokenizer, concept: str | None = None, *, boxes=None, box_labels=None, points=None,
               point_labels=None, source_size: tuple[int, int] | None = None) -> ConceptPrompt

sam31_text(pretrained: bool = False, inference_mode: bool = True, **kwargs) -> CausalTextTransformer   # registered "causal_text_transformer"
  __call__(token_ids[32] int32, padding_mask[32] bool) -> [32, 256]      # per-token features
  features(token_ids, padding_mask) -> [32, 256]                         # the FeatureSpec endpoint

ClipBpeTokenizer(context_length: int = 32)
  encode(texts: Sequence[str]) -> (token_ids int32 [N, 32], padding_mask bool [N, 32])

Sam3VideoTracker(model: Sam3, *, num_frames: int | None, source_size: tuple[int, int], max_objects: int = 16)   # experimental, host
  set_prompt(prompt: ConceptPrompt) -> None             # before the first frame; raises afterwards
  remove_object(object_id: int) -> None                 # also hides the object from frames not yet returned
  push_frame(frame[3, H, W]) -> list[FramePrediction]  # the frames that became final
  finish() -> list[FramePrediction]                     # remaining frames at the end of the clip
  add_object(frame_index: int, frame[3, H, W], points, labels, *, object_id: int | None = None) -> PointTrack
PointTrack: object_id: int; first_prediction: FramePrediction; push(frame) -> FramePrediction  # the new object only
```

Typed inputs and outputs are Equinox modules with fixed shapes; every padding mask uses True for padding, as
`equimo.language` does, and modules turn it into a key mask internally:

- `ImageFeatures`: `detector`, `interactive` and `propagation`, each a tuple of three maps (`[256, 288, 288]`,
  `[256, 144, 144]`, `[256, 72, 72]`). Sine position encodings are computed from the shapes where they are used.
- `ConceptPrompt` (static maxima: 32 text tokens, 16 exemplar boxes, 16 exemplar points): `text_features [32, 256]`,
  `text_padding_mask [32]`, `boxes [16, 4]` (normalized cxcywh), `box_labels [16]` (1 positive, 0 negative),
  `box_padding_mask [16]`, `points [16, 2]` (normalized), `point_labels [16]`, `point_padding_mask [16]`. A `create`
  class method pads shorter inputs to the maxima, and with `source_size` reads boxes and points in those pixels.
- `DetectorOutput`: `logits [200]` (joint logit), `presence_logit []`, `boxes [200, 4]` (normalized cxcywh),
  `mask_logits [200, 288, 288]` (float32), and per-decoder-layer `layer_boxes [6, 200, 4]` and `layer_logits [6, 200]`.
- `ConceptDetections`: `keep [max_detections]`, `masks [max_detections, H, W]`, `boxes [max_detections, 4]` (pixels,
  xyxy), `scores [max_detections]`, `query_index [max_detections]`, `presence []`.
- `InteractivePrompt` (static maximum: 16 points): `points [16, 2]` in input pixels, `point_labels [16]` (1 positive, 0
  negative, -1 padding; `create` defaults them to positive and rejects other values), `box [4]` (pixels, xyxy) with
  `has_box []`, `mask_input [288, 288]` with `has_mask []`. `create` scales coordinates from `source_size`, when given.
- `InteractiveOutput`: `mask_logits [4, 288, 288]`, `iou [4]`, `object_score_logit []`, `selected []`.
- `FramePrediction` (NumPy): `frame_index`, `object_ids [K]`, `masks [K, H, W]` (bool), `boxes [K, 4]`, `scores [K]`.
- `TrackerState` and `TrackOutput` are pytrees with a leading bucket axis, and so are the per-slot and per-bucket fields
  of `CommitPlan`, whose `frame_index` is a scalar. `Association` flattens tracks over buckets and has no bucket axis.
  Their field lists and per-field sizes live in the sam3 `EGM.md`, because they are experimental.

Module naming fixes checkpoint paths and fine-tuning tags: self-attention parents are named `attn` and cross-attention
parents `cross_attn`, each with `q`, `k`, `v` and `proj` linear children; detector heads sit under a `head` path part;
transformer blocks sit in lists named `blocks`. The two random Fourier matrices of the prompt encoders are declared with
`equimo.core.parameters.buffer()` and read through `stop_gradient`.

### Vision encoder and neck

- **Trunk.** A dedicated ViTDet encoder: bias-free 14 x 14 patch embedding, a learned 24 x 24 absolute position table
  tiled 3 x 3 at run time (the cls row of the reference table is dropped at conversion), `ln_pre`, 32 pre-norm blocks of
  width 1024 with 16 heads and MLP width 4,736 with exact GELU, no final norm. Blocks 7, 15, 23 and 31 attend globally;
  the others attend within 24 x 24 windows. A dedicated `ViTDetBlock` computes its own window partition and RoPE grid,
  and the blocks sit in one flat `blocks` list so that paths match the reference's `blocks.{i}`. The position table
  stays 24 x 24 so its trained size and fine-tuning behavior match the reference.
- **RoPE.** `VisionRoPE` with the mode strategy (`freqs_for="lang"`, `theta=10000`, `rope_dim=32`, `pt_seq_len=24`)
  already produces the reference's frequencies, interleaved pairs and interpolation. The one difference is axis order:
  the reference rotates by width first, then height. On a square grid that is `VisionRoPE`'s table with the two grid
  axes exchanged, so the encoder transposes the computed factors from `(H, W)` to `(W, H)` order before flattening.
  `VisionRoPE` itself does not change, so no identity hash changes. Windowed blocks use a 24 x 24 grid; global blocks
  use 72 x 72 with positions scaled by 24/72. Tables are computed while tracing, not stored.
- **Tri-neck.** Three simple feature pyramids of the same structure over the block-31 map (scale factors 4, 2 and 1,
  giving 288, 144 and 72 pixel grids, 256 channels), one for the detector, one for the interactive head and one for
  propagation. Their sine position encodings (DETR style, 128 features per axis, normalized) are computed while tracing.
  Transposed-convolution weights are flipped at conversion, as for EoMT.
- **Output.** `encode_image` returns `ImageFeatures`, so that the detector, interactive head and tracker share one
  encoder pass.

### Text encoder and tokenizer

- **Tokenizer.** `ClipBpeTokenizer` is adapted from OpenAI CLIP's MIT-licensed `simple_tokenizer.py`: lower-casing and
  `ftfy` cleaning, byte-level BPE with the CLIP vocabulary (49,408 tokens), start and end tokens, padding with 0 and
  truncation to 32 tokens with the end token kept. Token id 0 is a real CLIP token, so the tokenizer returns a padding
  mask next to the ids. The MIT vocabulary file (about 1.3 MB gzipped) ships as package data, so there is no download.
  It needs a new optional extra, `clip`, with `regex` and `ftfy` (a version range).
- **Model.** `CausalTextTransformer` builds on `language/models.py:TransformerEncoderStack` with a causal `attn_mask`:
  token embedding, learned position table (32 x 1024), 24 pre-norm blocks (16 heads, MLP 4,096, exact GELU), `ln_final`,
  then a per-token projection to 256. It returns per-token features. The reference's unused `text_projection` is not
  converted.
- **Why not `TextTransformerEncoder`.** It pools, uses a fixed sinusoidal table and has no causal mask. Generalizing it
  would need three options that change its contract (learned positions, causal mask, per-token output); a separate class
  is clearer.

### Concept detector (image path and per-frame video detection)

`Sam3.detect(features, prompt)` returns a `DetectorOutput` with 200 queries:

- **Prompt.** The geometry encoder pools box features with `roi_align` (7 x 7, adaptive sampling, `aligned=False`) and
  point features with bilinear `grid_sample`, adds sine encodings and label embeddings, and runs 3 pre-norm layers with
  cross-attention to the image. The reference has no text-free path: its 3.1 predictor encodes the text
  `<text placeholder>` for exemplars alone (with a score of 0.52 on one box where an empty text gives 22 detections on
  the same image), so `encode_concept` encodes that text when it receives only exemplars.
- **Fusion encoder.** 6 pre-norm layers over the 72 x 72 detector features: self-attention with position on queries and
  keys, cross-attention to the prompt (text and geometry tokens) with a key mask, ReLU feed-forward.
- **Decoder.** 6 post-norm DETR layers with 200 learned queries plus the presence token, iterative box refinement from
  learned reference points, the log-scale box relative-position bias in image cross-attention, and text cross-attention.
  Its feed-forward runs in float32, as the reference forces.
- **Scoring.** Dot-product scoring against the mean-pooled prompt, scaled by `1 / sqrt(dim)` and clamped to +/-12, then
  the joint logit: `sigmoid(class) * sigmoid(presence)` through `inverse_sigmoid`, whose 1e-3 clamp keeps it within
  about +/-6.9. The presence logit is also returned.
- **Segmentation head.** Prompt cross-attention over the encoder memory, a pixel decoder that runs 2 stages on the
  encoder memory and the middle and high pyramid levels (nearest upsampling, 3 x 3 convolution, group norm, ReLU) and
  per-query mask embeddings, giving 288 x 288 mask logits and a one-channel semantic map.
- **Image decoding.** `decode_concepts` reproduces the reference predictor on a single image: probabilities from the
  joint logit, the single-frame mask NMS, new objects at probability 0.5 or more, boxes from mask extents, and per-pixel
  non-overlap by score. It takes a static output size and a static `max_detections`, by default every query; it gathers
  the kept entries at 288 x 288 before upsampling them bilinearly to the output size and thresholding at 0, so memory
  scales with `max_detections`, not 200. It compiles once per output size. It does not multiply by the presence
  probability a second time.

### Interactive head (images and new objects on video)

`Sam3.predict_interactive(features, prompt)` runs the SAM 2-style prompt encoder and mask decoder on the interactive
pyramid, after adding the learned no-memory embedding:

- Points are in input-pixel coordinates; boxes become two corner points; mask inputs are 288 x 288 logits. The point
  shift, label codes and padding rules are in the appendix. A padding point is always appended, and an empty prompt also
  encodes one padding point in the first slot.
- Outputs are fixed-shape: all 4 mask tokens' masks (288 x 288), predicted IoU, the object score, and the selected
  index. Selection follows CUJ 2. Both branches are computed and selected with `where`, so the shape never depends on
  the prompt.
- No 3.1 pipeline sends box prompts to this head, so box prompts are checked at module level against the reference
  prompt encoder and mask decoder.

### Video: device state

The device state is one `TrackerState` pytree stacked over `B` buckets of 16 slots. `max_objects` caps live objects. A
bucket whose slots are all removed or empty is released, and its index is reused with cleared state. Removed slots are
not reused within a live bucket, as in the reference, so `B` can exceed `max_objects / 16`. `B` is static, grows in
powers of two and never exceeds `next_pow2(max_objects)`; when no bucket index is free, new objects are dropped and
reported, like objects over the cap. `B` never shrinks, to avoid recompiling. The reference only ever reads a fixed
window of past frames, so fixed buffers reproduce it:

| Field                              | Shape (per bucket)       | Dtype        | Content                                                              |
| ---------------------------------- | ------------------------ | ------------ | -------------------------------------------------------------------- |
| `bucket_valid`                     | `[]`                     | bool         | bucket in use                                                        |
| `slot_status`                      | `[16]`                   | int8         | empty, active or removed                                             |
| `cond_memory`, `cond_features`     | `[4, 256, 72, 72]` each  | model dtype  | memory and propagation features of the 4 latest conditioning frames  |
| `cond_pointers`, `cond_frame`      | `[4, 16, 256]`, `[4]`    | model, int32 | object pointers and frame index (-1 when absent)                     |
| `recent_memory`, `recent_features` | `[6, 256, 72, 72]` each  | model dtype  | ring buffer for frames t-1 to t-6                                    |
| `recent_frame`, `recent_is_cond`   | `[6]`                    | int32, bool  | frame index (-1 when absent) and whether that frame was conditioning |
| `pointers`, `pointer_frame`        | `[15, 16, 256]`, `[15]`  | model, int32 | ring buffer of pointers for frames t-1 to t-15                       |
| `last_masks`, `last_scores`        | `[16, 288, 288]`, `[16]` | float32      | latest output per slot, for rendering and the prompt frame's reuse   |

In bfloat16 a bucket takes about 59 MB: memory and features 53 MB, pointers 0.2 MB, `last_masks` 5.3 MB. At 16 buckets
that is 0.94 GB. Host-side metadata (object IDs, slot of each object, the grouping of buckets into states, birth frame
and score, hot-start counters, last occlusion frame, confirmation status) stays in NumPy in the driver. Frame indices
are traced integers, so a new frame never recompiles.

Selection reproduces the reference: the latest 4 conditioning frames (in forward-only tracking, the reference's
"closest" rule selects exactly those); conditioning frames that are not selected but fall in t-1 to t-6 act as recent
frames; pointers come from the selected conditioning frames plus t-1 to t-15. A state only has a conditioning frame when
it is created, so a bucket normally holds one. The spatial temporal embedding is indexed as the reference does: a
conditioning frame at distance d uses `6 - d` while `0 < d < 7` and the last entry otherwise, a recent frame uses
`d - 1`. The pointer temporal encoding divides the frame distance by `min(num_frames, 16) - 1`, with sines first and
cosines second and no 2 pi scale. `num_frames=None` (streaming, length unknown) is treated as at least 16 frames; a
one-frame clip uses a divisor of 1.

### Video: per-frame flow

For frame t the driver runs:

1. **Encode and detect** (`encode_image`, `detect`): independent of tracker state, so frames can be batched with `vmap`.
1. **Propagate** (`Sam3.propagate`): for every bucket, memory attention (4 pre-norm layers, 8 heads of 32,
   self-attention with RoPE over 72 x 72, cross-attention whose keys are the selected memory frames plus pointer tokens,
   RoPE repeated per memory frame and not applied to pointers; queries add the current frame's features and keys add the
   stored frame features and position encodings, each through its own projection, and the inputs add `0.1` times their
   sine position), then the multiplex mask decoder (16 slots x 3 masks, without sparse or dense prompts, with per-slot
   valid and invalid embeddings on the mask tokens), giving per slot 288 x 288 mask logits, the object score, IoU
   predictions and the pointer. An object scored as absent uses `no_object_pointer` applied to its projected pointer.
   Empty slots are decoded and ignored, and their zero pointers stay among the keys, as in the reference.
1. **Associate** (`Sam3.associate`, on device): the detection keep mask after the batched NMS and the boundary filter,
   the IoM matrix between kept detections and tracks, with ambiguous rows and columns zeroed for the match and
   high-confidence flags but not for the track-matched flag, the new-detection flag, and the pairwise track IoU used for
   occlusion on the tracks and on the masks as reconditioning would leave them, all with fixed shapes.
1. **Plan** (host, NumPy): hot-start counters and removals, reconditioning choices, occlusion suppression, the
   new-object cap and IDs, best-fit bucket assignment, bucket release, and confirmation, with the reference thresholds
   (appendix). The result is a fixed-shape `CommitPlan` of per-slot arrays.
1. **Commit** (`Sam3.commit`): applies reconditioning and suppression to the mask logits, including objects removed on
   this frame, resizes them to 16 times the grid, runs per-pixel area-shrinkage suppression across all objects in ID
   order, and encodes memory (32-channel input: 16 mask slots of `2 * sigmoid - 1` and 16 conditioning flags, 0 for
   absent slots, plus `no_object_memory` for every slot that does not appear). Added objects and reconditioned ones go
   through the interactive head for their pointers and a +/-10 object score, from the binary mask at 16 times the grid,
   downsampled by `interactive_mask_downsample`. Every bucket of a state that received objects is re-encoded from the
   consolidated output: new masks pass through the source frame size, existing rows keep the propagated low-resolution
   masks, everything is resized to the image size, overlaps are resolved per state by highest logit (ties to the earlier
   object), and flags mark the new objects. It then marks removed slots and advances the ring buffers. A bucket left
   without live objects is cleared after the write. The source frame size is static, so `commit` compiles once per clip
   size and bucket count.
1. **Emit**: once frame t is planned, frame t-2 is final, because a frame hides only objects that are still unconfirmed
   and live two frames later, and hot-start removals never hide earlier frames. The reference's 15-frame hot-start
   buffer and its groups of 16 change only latency, not content, so the driver does not reproduce them. At the end of
   the clip, the remaining frames are emitted with the look-ahead clamped to the last frame. An object on its birth
   frame renders from its detector mask. Objects with no pixel are omitted before per-pixel overlaps are resolved, by
   the highest tracker probability (the detection probability on a birth frame), so an object can remain with an empty
   mask. An object removed on frame t is shown on t with a score of -1e4.

The prompt frame is always frame 0, so `set_prompt` raises once a frame has been pushed. It is processed twice, as in
the reference: once on its own (the single-frame NMS, the `max_objects` cap, the 0.65 new-object threshold, nothing
hidden) and once as the first propagated frame, where the stored conditioning output is reused and the policy and
`commit` run as on any frame.

`tracker.py` provides a function that pads a state's bucket axis to a larger power of two; `video.py` decides when to
call it.

### Video: association policy

The policy runs on the host in NumPy so that it can be replayed on CPU from recorded device outputs. Its thresholds, tie
rules and the behaviors reproduced on purpose are listed in the appendix, each with its reference location. The decision
rules are written in prose in the policy module's docstrings with the same citations. Where the reference sorts unstably
(keep order, NMS order), Equimo sorts stably, the cap selection is the same NumPy reversal of an ascending sort, and the
fixture clips are chosen so that no ties occur and no score lies within 1e-3 of a threshold it is compared with.

### Video: points on a new object

`Sam3VideoTracker.add_object` covers the case the reference supports without defects: a new object added with points on
frame f. The interactive head runs on frame f, its mask passes through the source frame size and becomes the
conditioning frame of a new single-bucket state with the object's real object score, and the returned `PointTrack` runs
`propagate` and stores each frame's memory from the propagated masks for that bucket only, over the frames the caller
supplies from f+1, without detection or association. This matches the reference's partial propagation after a point
prompt, and the reference is the oracle for its tests. The main tracker's device state and masks are not changed. The
object's ID comes from the main tracker's counter, which prevents ID collisions, so objects the detector adds later take
the IDs after it. Merging the point track back into the main tracker is part of the deferred refinement work. Points
that name an existing object, through `object_id`, raise an error that says refinement on video is not supported.

### Attention and numerics

- **`CrossAttention`** (`core/layers/attention.py`): `q`, `k`, `v` and `proj` linear fields, `dim`, `num_heads`,
  `internal_dim` (the projected width, which SAM's two-way transformer halves), `kv_dim`, and a static `implementation`.
  It is called as `(x, context, *, key_mask=None, bias=None, key=None, inference=True)`, where `key_mask` uses True for
  keys to attend to and `bias` is an additive logit bias (the DETR box bias). It is not registered as an
  `AttentionBlock` layer. `equinox.nn.MultiheadAttention` is not reused because it has no float32 softmax with masks
  through `where`, no zeroing of fully masked rows and no cuDNN path; the tabular `CrossAttention` is not reused because
  it has no bias or mask and uses TabPFN's scaling.
- **`PositionedAttention`** (`vision/models/sam3/common.py`): a `CrossAttention` subclass used in the trunk, detector,
  two-way transformer and memory self-attention. Its keys may differ from its values, because DETR-style attention adds
  position encodings to queries and keys only, and it rotates queries and keys with RoPE after projection. Memory
  cross-attention is a plain `CrossAttention` whose projections are called from `_attend_packed_memory` in `tracker.py`,
  for the packed keys below.
- **Packed keys.** Memory attention has up to 52,144 keys per bucket, and the count changes from frame to frame. A dense
  mask would take about 2.2 GB per call, so `propagate` packs the valid keys first (a gather with static size) and
  passes per-bucket key lengths. `_fused_attention` gains `key_value_seq_lengths`, broadcast to the leading shape
  `q.shape[:-3]` (a scalar under the tracker's `vmap`), for the unmasked case, where cuDNN accepts it without a mask,
  and zeroes the output rows of a batch entry whose length is 0.
- **Float32.** cuDNN rejects float32, and dense float32 logits for one memory-attention layer take about 8.6 GB, so
  memory attention runs a query-chunked XLA path (`lax.map` over blocks of 1,024 queries) whenever `implementation` is
  `"xla"`, in every dtype. A float32 model must stay on `"xla"`. The path is private to the tracker.
- **Precision rules.** Attention softmax in float32 with masks through `where`; the DETR feed-forward in float32; RoPE
  rotation in float32; mask logits in float32 before resizing; layer norms in float32. Outputs return in the model
  dtype.
- **Batching.** All modules are unbatched; images and frames are batched with `vmap`, buckets inside the tracker with
  `vmap`.

### Weights, conversion and archives

- A maintainer script, `models/sam31.py`, reads `sam3.1_multiplex.pt` on CPU with `torch.load(weights_only=True)` after
  checking its SHA-256, classifies every key (converted, computed and dropped, or unused and dropped), fails on unknown
  keys and on any model parameter without a source, naming them, and writes two float32 archives: `sam31_multiplex` (the
  vision model) and `sam31_text` (the text model). Bfloat16 users cast after loading.
- The archives are always `sam31_multiplex` and `sam31_text`; the SAM 3 paper describes further training of both
  encoders after their Perception Encoder initialization, so neither is a reused standalone model.
- Digests are pinned in `src/equimo/_pretrained.py` with a new repository revision after upload. Uploading needs the
  author's approval.

### Reference harness

Fixtures come from the reference, never from Equimo output, generated on the RTX 4080 in a separate pinned environment
(`models/sam31_reference.py`):

- **Float32 reference:** autocast and TF32 off, `use_fa3=False`, `use_rope_real=False`, the math attention kernel where
  memory attention requests FlashAttention 2 (which rejects float32), the ViT MLP replaced by plain fc1, exact GELU and
  fc2, and stored memory kept in float32. Memory attention runs in blocks of 1,024 queries and the detector one frame at
  a time, because dense float32 logits and the builder's chunks of 16 frames do not fit in 16 GB. Forward hooks save
  each component's outputs under Equimo names, with provenance.
- **Bfloat16 reference:** the shipped pipeline with `use_fa3=False`, for a reported comparison only.
- **`torch.compile` is off in every run,** because torchvision compiles `roi_align` on GPU whatever the reference's
  options, and eager kernels are reproducible.
- **Tiny fixtures:** a small random-weight configuration of the reference covering windowed and global blocks, RoPE
  interpolation, a small multiplex count and a clip longer than the memory windows, committed under `tests/data/`.
  Random weights give no detection above any threshold and no usable track mask, so the tiny clip feeds the association
  policy a fixed script of disjoint per-object masks and scores; the real detector and tracker still run and are
  recorded.
- **Margins and coverage:** at every threshold comparison the harness asserts that the compared value is at least 1e-3
  from the threshold, and on the full-weight clip it asserts that every decision type occurs (add, hot-start removal,
  reconditioning, confirmation). Both are clip-selection criteria; synthetic policy fixtures are added only if no
  redistributable clip covers a decision type. No clip tried triggers occlusion suppression, which needs two tracks that
  both went empty and then land on one object, so the scripted tiny fixture `sam3_tiny_occlusion_reference` covers it,
  replayed through the driver.

| Check                        | Inputs                                                              | Criterion                                                                                 |
| ---------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| component parity (CI)        | tiny random weights                                                 | `allclose` 1e-5                                                                           |
| component parity (scheduled) | full weights, real image                                            | mean absolute error at most 5e-4 per output                                               |
| image decoding (scheduled)   | full weights, real image                                            | same kept set; mask IoU at least 0.99                                                     |
| video (scheduled)            | full weights, a non-square clip of about 48 frames, not 1008 x 1008 | same decision trace and IDs; mask IoU at least 0.99                                       |
| new object by points (CI)    | tiny random weights                                                 | `allclose` 1e-5 against the reference's partial propagation                               |
| bfloat16 images (scheduled)  | Equimo bfloat16 vs float32                                          | mask IoU at least 0.97; score difference at most 0.02; same detections at probability 0.5 |
| bfloat16 video (scheduled)   | Equimo bfloat16 vs float32                                          | identical decisions and IDs; each frame's median per-object mask IoU at least 0.97        |

### Fine-tuning

LoRA and head-only plans use the existing `equimo.finetune` machinery on the image path. Head-only means the detector's
fusion encoder, decoder, scoring and segmentation head with the trunk, neck and text model frozen. The port has no
dropout or drop-path, so LoRA and head-only fine-tuning run deterministically; dropout has no parameters and can be
added later without changing archives. Whether LoRA through the frozen 32-block trunk fits in 16 GB at 1008 x 1008 is
measured: with XLA attention it runs out of memory, trying to allocate 15.4 GB on an RTX 4080, and fits with
`encode_image(x, checkpoint_trunk=True)` (7.3 GB peak); with cuDNN attention the peak is 8.4 GB without
rematerialization and 2.2 GB with it. Layer-wise learning-rate decay is not extended to the ViTDet trunk.

### Licensing

- "SAM Materials" is defined by the SAM License (the models, software and documentation Meta distributes under it). In
  the author's reading, this design and code written from it are not SAM Materials; only the weights are.
- The SAM License (last updated 2025-11-19) is copied to `LICENSES/pretrained/` and listed in `pyproject.toml`
  `license-files`, in both license tables, and next to both archives on Hugging Face.
- The tokenizer and its vocabulary are adapted from OpenAI's MIT-licensed CLIP release, not from the SAM repository; the
  root `NOTICE` records it.
- `docs/licensing/sam3-provenance.md` records how the code was written: a first draft from this design, Equimo, the SAM
  3 paper and the SAM 3.1 release post, then fixes made after reading the reference, each logged as the behavior that
  differed and the test that caught it.
- The model docstrings and documentation point users to the SAM License's acknowledgement and trade-control terms for
  the weights.

## Alternatives Considered

Decisions were made by the author during design; each entry gives the trade-off that decided it.

### Licensing and process

#### Host only a converter, not the weights

- **Pros**: no redistribution of SAM Materials from Equimo's repository.
- **Cons**: every user must obtain gated access and run a 3.5 GB conversion with PyTorch installed.
- **Reason for Rejection**: Rejected: hosting with the SAM License next to the weights costs one license file and one
  table row, while a converter-only release costs every user a gated download and a PyTorch install. Hosting weights
  does not change the code license, as the existing restrictively licensed weights show.

#### Translate the reference directly into the main package, as for T0

- **Pros**: a direct route to parity; the structure maps one to one onto the reference.
- **Cons**: T0's upstream is Apache-2.0, while code translated from the SAM 3 source would, in the author's reading, be
  a derivative of SAM Materials and put every `equimo` user and redistributor under the SAM License (trade controls,
  termination with deletion, terms Meta may change).
- **Reason for Rejection**: Rejected: a draft written from the design and then corrected against the reference keeps the
  code license, at the cost of a slower path to parity and a logged list of fixes.

#### A separate `equimo-sam3` wheel for SAM-derived code

- **Pros**: confines any SAM-licensed code to an opt-in distribution.
- **Cons**: a second package to version, test and release; an optional extra alone cannot include or exclude code.
- **Reason for Rejection**: Rejected: in the author's reading no code is SAM Material, so a second package would add
  release work without changing any license.

#### Clean room with a spec writer and implementers who never read the reference

- **Pros**: independence that does not depend on a process record.
- **Cons**: two roles and a recorded question channel for every component; per-component parity and copied thresholds
  make independence hard to claim anyway.
- **Reason for Rejection**: Rejected: the draft-then-compare process gives the same parity with one person, and the
  provenance note records each fix made after reading the reference.

#### Compare only end-to-end outputs

- **Pros**: fewer fixtures.
- **Cons**: a mismatch in a ten-stage pipeline cannot be located.
- **Reason for Rejection**: Rejected: per-component fixtures cost a few megabytes and locate each mismatch to one
  module; comparing activations does not affect how the code is licensed.

### Scope

#### Add the SAM 3 (November 2025) checkpoint for images

- **Pros**: matches `Sam3Processor` and the SAM 3 image predictor directly.
- **Cons**: a second 3.4 GB checkpoint, a second tracker, and a second set of fixtures.
- **Reason for Rejection**: Rejected: the 3.1 checkpoint already contains a detector and an image-level interactive
  head, so the second checkpoint doubles conversion and fixtures for no new capability.

#### Preprocessing in Equimo (`preprocess_sam3_image`)

- **Pros**: one package from file to masks; resize parity owned in one place.
- **Cons**: duplicates justdata's purpose; four different resize paths in the reference would each need a matching
  option.
- **Reason for Rejection**: Rejected: preprocessing belongs in justdata, and Equimo's tests can feed tensors the
  reference preprocessed, so Equimo needs none of the four resize paths.

#### A public `lax.scan` over frames for prompt-only tracking

- **Pros**: training through time without user-written loops.
- **Cons**: a public prompt schema to maintain, a second path to keep at parity, and outputs that differ from the
  driver's once detection is involved.
- **Reason for Rejection**: Rejected: no goal needs training through time; a test that runs `propagate` and `commit`
  under `lax.scan` keeps the option open without a second public path.

#### Only 16 objects (one bucket) at first

- **Pros**: a simpler state with no recompilation.
- **Cons**: does not reproduce the reference at its default.
- **Reason for Rejection**: Rejected: bucket growth is needed for parity even at 16 objects, because removed slots are
  never reused, so a one-bucket state would diverge from the reference on any clip with removals.

#### Refine existing objects with points on video now, fixing the reference's defects

- **Pros**: click-to-correct on video.
- **Cons**: about 53 MB of float32 window masks per bucket, a `refine` step and a memory rebuild, with no reference
  oracle; the invariant that would test the memory rebuild cannot hold, because masks decoded alongside other objects
  differ from those of an object tracked alone.
- **Reason for Rejection**: Deferred: adding new objects with points has a reference oracle and stays; refinement can be
  added later without breaking a stable API, because the tracking state is experimental.

#### Reproduce the reference's refinement defects

- **Pros**: identical outputs everywhere.
- **Cons**: knowingly ships a loss of spatial memory on refinement and a failure on a second refinement of one frame.
- **Reason for Rejection**: Rejected: shipping known defects to gain parity on one interaction is worse than not
  offering the interaction.

### Model structure

#### One `track_step` per frame

- **Pros**: one function to jit and document.
- **Cons**: the reference interleaves propagation, a host decision, and memory encoding that depends on it, so one step
  cannot reproduce it without moving the policy onto the device.
- **Reason for Rejection**: Rejected: splitting the frame into propagate, associate, a host plan and commit costs one
  host synchronization per frame, and in exchange the policy can be replayed on CPU from recorded device outputs.

#### Extend `VisionTransformer` instead of a dedicated ViTDet encoder

- **Pros**: reuses EoMT and PMT plumbing.
- **Cons**: per-block window and global attention with different RoPE grids changes the invariants and signatures of
  about 130 presets.
- **Reason for Rejection**: Rejected: a dedicated encoder costs one new block class and leaves 130 presets unchanged.

#### Reuse `WindowedSequence` for the trunk's windowed blocks

- **Pros**: existing window partitioning code.
- **Cons**: it builds its own blocks and passes no extra arguments, so it cannot pass per-grid RoPE factors, and it
  nests block paths so they no longer match the reference's `blocks.{i}`.
- **Reason for Rejection**: Rejected: a `ViTDetBlock` that partitions its own windows costs a few lines of reshaping and
  keeps checkpoint paths flat.

#### A new RoPE class

- **Pros**: shared layers stay untouched.
- **Cons**: `VisionRoPE` already computes the same frequencies, pairing and interpolation, so a copy would differ only
  in axis order.
- **Reason for Rejection**: Rejected: transposing `VisionRoPE`'s factor table costs one reshape and avoids a second
  class to maintain.

#### An axis-order field on `VisionRoPE`

- **Pros**: makes the axis order explicit.
- **Cons**: a new static field changes the identity hashes recorded for every existing model that uses `VisionRoPE`.
- **Reason for Rejection**: Rejected: the transpose gives the same table without invalidating existing saved deltas.

#### Text encoder in one archive with the vision model

- **Pros**: one archive, as the author's archive rule suggests for parts trained together; token-id input.
- **Cons**: text would be re-encoded on every call unless a separate endpoint is added, and the text tower could not be
  used on its own.
- **Reason for Rejection**: Rejected: a separate language archive, as TIPS does, costs a second download and keeps text
  features reusable across frames and models.

#### Text encoder inside the SAM 3.1 package

- **Pros**: everything SAM-specific in one place.
- **Cons**: either the first import from `equimo.vision` into `equimo.language`, or a text model under `equimo.vision`.
- **Reason for Rejection**: Rejected: a generic causal text model in `equimo.language` respects the modality boundary at
  the cost of one more registered family.

#### Generalize `TextTransformerEncoder`

- **Pros**: one text class in `equimo.language`.
- **Cons**: three options (learned positions, causal mask, per-token output) that change its contract.
- **Reason for Rejection**: Rejected: a separate `CausalTextTransformer` built on the shared `TransformerEncoderStack`
  reuses the blocks without changing an existing contract.

#### Download the tokenizer vocabulary

- **Pros**: a smaller wheel by about 1.3 MB.
- **Cons**: new download and digest code (the existing tokenizer download has no digest check) and a network call before
  the first tokenization.
- **Reason for Rejection**: Rejected: shipping the MIT vocabulary as package data costs 1.3 MB of wheel size and removes
  the download path entirely.

#### Pre-tile the absolute position table at conversion

- **Pros**: the existing `LearnedPosEmbed` works as-is.
- **Cons**: stores 5,184 rows instead of 576, so fine-tuning would update nine untied copies.
- **Reason for Rejection**: Rejected: tiling the 24 x 24 table at run time costs one tile operation per call and keeps
  fine-tuning behavior identical to the reference.

### Video state and attention

#### A host list of per-bucket jitted calls

- **Pros**: the simplest state.
- **Cons**: one dispatch per bucket per step.
- **Reason for Rejection**: Rejected: one stacked state vmapped over buckets costs a bucket axis on every field and
  removes per-bucket dispatch.

#### A static 16-bucket state

- **Pros**: never recompiles.
- **Cons**: always pays 16 memory-attention passes and 0.94 GB of buffers.
- **Reason for Rejection**: Rejected: growing the bucket count in powers of two costs at most five recompiles per clip
  and keeps the common one- or two-bucket case small.

#### Keep every frame's memory, as the reference does

- **Pros**: mirrors the reference's storage directly.
- **Cons**: memory grows with clip length, and shapes would change every frame.
- **Reason for Rejection**: Rejected: the reference reads only a fixed window, so fixed ring buffers reproduce it with
  constant memory and shapes.

#### A dense attention mask for memory attention

- **Pros**: uses `_fused_attention` unchanged.
- **Cons**: about 2.2 GB of mask per call.
- **Reason for Rejection**: Rejected: packed keys with per-bucket lengths cost one gather per call and no mask memory.

#### Reproduce the 15-frame hot-start buffer

- **Pros**: the same latency as the reference.
- **Cons**: 15 to 30 frames of delay with no effect on content in the 3.1 path.
- **Reason for Rejection**: Rejected: emitting frame f once frame f+2 is planned gives the same content with a two-frame
  delay.

### Decoding, weights and verification

#### Decode images like `Sam3Processor`

- **Pros**: the documented SAM 3 image API.
- **Cons**: on the 3.1 detector it multiplies the presence probability in twice; the reference never does this.
- **Reason for Rejection**: Rejected: matching the 3.1 predictor on a single image keeps parity with the code path the
  reference actually runs, at the cost of differing from the documented SAM 3 API.

#### NMS on the host in NumPy

- **Pros**: mirrors the reference's Python loop.
- **Cons**: image decoding could not be batched or jitted.
- **Reason for Rejection**: Rejected: fixed-shape NMS on device costs a 200 x 200 overlap matrix and keeps decoding
  jittable and batchable.

#### A bfloat16 archive

- **Pros**: half the download.
- **Cons**: the loaded dtype must match the template, so it would need its own identifier and digest.
- **Reason for Rejection**: Rejected: one float32 archive cast after loading costs 1 GB more download and avoids a
  second identifier per model.

#### Reuse `convert_params_from_torch`

- **Pros**: existing helper.
- **Cons**: it needs a CUDA-built torch model and drops buffers and complex values; the reference builder calls
  `.cuda()`.
- **Reason for Rejection**: Rejected: a state-dict-driven converter runs on CPU and sees every key, which the key
  classification needs.

#### Tolerances of 1e-5 everywhere

- **Pros**: one rule.
- **Cons**: 1e-5 per element is unlikely to hold for a 32-layer encoder at full resolution on different kernels.
- **Reason for Rejection**: Rejected: tiered tolerances (1e-5 on tiny fixtures, a mean absolute error of at most 5e-4 on
  full weights, decision trace and mask IoU on video) keep the strict bound where it can hold.

#### Tolerances derived from a measured noise floor

- **Pros**: tolerances with a measured basis.
- **Cons**: a measurement step before any test can be written, and a tolerance chosen after seeing the error.
- **Reason for Rejection**: Rejected: a bound fixed in advance with measured values recorded gives the same information
  without choosing the tolerance from the result.

#### Per-object bfloat16 mask IoU on video

- **Pros**: a per-object bound on every frame catches an error on a single object, and matches the float32 video
  criterion and the bfloat16 image criterion.
- **Cons**: bfloat16 alone breaks it on real footage. On the 48-frame fixture clip, Equimo bfloat16 against Equimo
  float32 falls below 0.97 on 3 frames (lowest about 0.94), and the reference's own bfloat16 pipeline against its
  float32 output does so on 2 frames (lowest 0.938), mostly on small objects, whose masks change most from a few
  boundary pixels. Comparing Equimo bfloat16 with the reference's bfloat16 is worse (6 frames, lowest 0.913), because
  two sets of rounding errors add up.
- **Reason for Rejection**: Rejected, measured: the clip's decisions are identical in bfloat16 and float32, so the
  criterion now requires the identical trace on every frame and bounds each frame's median object at 0.97 (lowest frame
  median measured: 0.986 to 0.992 across GPU runs, which vary slightly). The tradeoff is that one object per frame can
  drift further than 0.97 unnoticed as long as the decisions and the other objects hold, and the bound was changed after
  seeing the error, unlike the tolerances above. The float32 criterion is unchanged.

#### Generate fixtures on CPU with a shim

- **Pros**: no GPU needed.
- **Cons**: the reference hard-codes CUDA in several constructors and forward passes.
- **Reason for Rejection**: Rejected: generating on the RTX 4080 in a separate pinned environment costs a GPU run and
  avoids patching the reference's device handling.

## Risks and Mitigations

| Risk                                   | Impact                                                     | Status              | Mitigation or fallback                                                                                                                                               |
| -------------------------------------- | ---------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Video does not fit a 16 GB GPU         | bfloat16 tracking at several buckets fails on the RTX 4080 | measured            | fits with cuDNN attention at 4 buckets (4.5 GB peak); XLA attention runs out of memory at 4 buckets (6.1 GB at 2). Fallback: cuDNN attention, lower `max_objects`    |
| LoRA does not fit a 16 GB GPU          | the fine-tuning goal fails at 1008 x 1008                  | measured            | fits in 7.3 GB with trunk rematerialization on XLA attention and 2.2 GB on cuDNN (50 steps, finite loss); XLA attention without rematerialization runs out of memory |
| Reference GELU differs from the config | the fused ViT MLP may use tanh GELU in the shipped model   | open                | the float32 reference is patched to exact GELU; the bfloat16 comparison shows whether the shipped model differs                                                      |
| No suitable redistributable test clip  | the full-weight video fixture cannot be hosted             | clip chosen         | a CC0 clip with two or more objects of one concept, an occlusion and every decision type but occlusion suppression; the fallback's scripted fixture covers that one  |
| Near-threshold decision flips          | parity fails on numerically legitimate differences         | mitigated by design | the harness asserts a 1e-3 margin at every threshold comparison                                                                                                      |
| Licensing position challenged          | code would fall under the SAM License                      | author's reading    | rollback: remove the family and its digests (no other family depends on it); record in the provenance note whether legal review happens before release               |
| Pickled checkpoint                     | loading untrusted pickles runs code                        | mitigated by design | `torch.load(weights_only=True)` after checking a pinned SHA-256 of the source file                                                                                   |
| Upstream drift                         | later upstream fixes change behavior                       | accepted            | the reference is pinned at `f1647e8`; the tracking state, steps and driver are experimental                                                                          |

## Non-Functional Requirements

- **Performance and memory.** Video tracking state is about 59 MB per bucket in bfloat16. Memory attention uses packed
  keys (no dense mask). It runs on XLA by default and on cuDNN in bfloat16 when the caller opts in with
  `with_attention_implementation`. Memory and latency are measured on an RTX 4080 and recorded in `docs/performance.md`.
- **Compilation.** Device steps compile once per bucket count (1, 2, 4, ...); `commit` also once per source frame size.
  Frame indices and object changes within a bucket count never recompile.
- **Compatibility.** No existing public signature, checkpoint or identity hash changes: `VisionRoPE` is reused without a
  new field, `_fused_attention` gains two optional arguments, and the new extra is optional.
- **Security and licensing.** Archives load only through `load_weights` with pinned digests and checkpoint limits. The
  tokenizer vocabulary ships in the package. The SAM License travels with the weights in the repository and on Hugging
  Face.
- **Rollout.** The image endpoints (`features`, `encode_image`, `detect`, `predict_interactive`, `decode_concepts`) and
  the text model become stable once full-weight parity passes. `TrackerState`, `CommitPlan`, `propagate`, `associate`,
  `commit` and the video driver are experimental. Rolling back means removing the family and its archives' digests; no
  other family depends on it.

## Additional Resources

- SAM 3 paper: [arXiv 2511.16719](https://arxiv.org/abs/2511.16719).
- Reference implementation:
  [`facebookresearch/sam3` at `f1647e8d0c1d280991f4d1b7d974c2704537fe66`](https://github.com/facebookresearch/sam3/tree/f1647e8d0c1d280991f4d1b7d974c2704537fe66).
- Weights: [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1), file `sam3.1_multiplex.pt` (gated).
- SAM License, last updated 2025-11-19, as distributed with the weights on Hugging Face.
- SAM 3.1 release post: the ["Update March 27, 2026" section](https://ai.meta.com/blog/segment-anything-model-3/) at the
  top of Meta AI's SAM 3 blog post (object multiplexing, up to 16 objects per forward pass).
- jax2onnx RoPE fix: [enpasos/jax2onnx#291](https://github.com/enpasos/jax2onnx/pull/291).

## Future Improvements

- Refine existing objects with points on video, with the reference's two defects fixed (lost spatial memory when an
  object moves to its own bucket, failure on a second refinement of one frame), and merge point tracks into the main
  tracker.
- A shared recent-frame feature ring across buckets, if measured memory requires it.
- Follow upstream changes after `f1647e8`.
- A public scan over frames if video fine-tuning becomes a goal.
- Full fine-tuning with rematerialization on larger GPUs.

## Appendix: Reference behavior

Values come from the 3.1 builder and class defaults at `sam3@f1647e8`. Paths are relative to the reference repository:
the builder is `sam3/model_builder.py`, the video policy is in `sam3/model/sam3_video_base.py` and
`sam3/model/sam3_multiplex_base.py`, multiplex tracking is in `sam3/model/sam3_multiplex_tracking.py`, and the
interactive mask decoder is `sam3/sam/mask_decoder.py`.

### Association thresholds

Locations are relative to `sam3/model/`.

| Setting                      | Value                                                                                                              | Gates                                                                                                  | Reference                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| detection probability        | greater than 0.4                                                                                                   | detection kept                                                                                         | `sam3_multiplex_base.py:run_backbone_and_detection`                                       |
| NMS overlap (video path)     | IoM greater than 0.1                                                                                               | lower-scored detection suppressed                                                                      | `model_builder.py:det_nms_thresh`                                                         |
| boundary margin              | 0.025                                                                                                              | box center must be strictly inside 0.025 to 0.975                                                      | `sam3_video_base.py:_suppress_detections_close_to_boundary`                               |
| association                  | IoM at least 0.1, after ambiguous rows and columns are zeroed                                                      | detection matches a track, counts duplicates and confirmation                                          | `sam3_video_base.py:_associate_det_trk_compilable`                                        |
| new detection                | raw IoM below 0.1 with every track                                                                                 | detection can become an object                                                                         | `sam3_video_base.py:_associate_det_trk_compilable`                                        |
| track matched                | raw IoM at least 0.5                                                                                               | a non-empty track that has it is matched                                                               | `sam3_video_base.py:_associate_det_trk_compilable`                                        |
| new object                   | probability at least 0.65 (0.5 on a single image)                                                                  | a kept new detection becomes an object; only the boundary filter is ignored when nothing is tracked    | `sam3_multiplex_base.py:_associate_det_trk`                                               |
| ambiguity                    | IoM at least 0.5                                                                                                   | rows and columns with two or more of these are zeroed                                                  | `sam3_video_base.py:_associate_det_trk_compilable`                                        |
| reconditioning               | frames where t % 16 == 0; detection probability at least 0.8, unique IoM at least 0.5, track probability above 0.8 | detection mask replaces the track mask; frame stays non-conditioning                                   | `sam3_multiplex_base.py:_recondition_masklets`, `sam3_video_base.py:_create_cpu_metadata` |
| hot-start window             | 15 frames                                                                                                          | removals apply while an object is at most 14 frames old                                                | `sam3_multiplex_base.py:_process_hotstart_gpu`                                            |
| hot-start unmatched          | 8 frames or more                                                                                                   | object removed                                                                                         | `sam3_multiplex_base.py:_process_hotstart_gpu`                                            |
| hot-start duplicate          | 8 or more, counting one per shared detection                                                                       | younger of two objects sharing a detection removed, if the older strictly predates it                  | `sam3_multiplex_base.py:_process_hotstart_gpu`                                            |
| occlusion suppression        | IoU at least 0.7, after reconditioning                                                                             | strictly more recently occluded object suppressed, if the other was occluded before                    | `sam3_multiplex_base.py:_get_objects_to_suppress_based_on_most_recently_occluded`         |
| area shrinkage (memory only) | keeps less than 30% of its area, or has none                                                                       | mask clamped to -10 before memory encoding, over all objects in ID order; nothing changes for a single | `video_tracking_multiplex_demo.py:_suppress_object_pw_area_shrinkage`                     |
| confirmation                 | 3 consecutive passes matched or created                                                                            | object shown                                                                                           | `sam3_multiplex_base.py:update_masklet_confirmation_status`                               |
| object cap                   | `max_objects`, default 16, against the count before this frame's removals                                          | highest-scoring new objects kept when over the cap, otherwise all                                      | `sam3_video_base.py:get_new_det_gpu_ids`, `_drop_new_det_with_obj_limit`                  |

### Other reference rules

- **Single-frame NMS** (image decoding and the prompt frame's first pass): overlap is intersection over minimum, as in
  the batched video-path NMS, threshold 0.1, and a suppressed detection still suppresses later ones.
- **Mask logits** of a slot whose object score is not positive are set to -1024.
- **Scoring clamps:** the dot-product score is scaled by `1 / sqrt(dim)` (1/16 at the 256-wide model) and clamped to
  +/-12 before the joint logit.
- **Interactive prompts:** points are shifted by 0.5 pixel; labels are 1 (positive), 0 (negative), -1 (padding), 2 and 3
  (box corners); a padding point is always appended, because 3.1 sends boxes as corner points with the box argument
  unset.
- **Stability score:** `count(logit > 0.05) / count(logit > -0.05)` on the single-mask output, threshold 0.98. The
  pointer comes from the selected token for multimask prompts and from the first token otherwise
  (`sam/mask_decoder.py:MaskDecoder.forward`).
- **New object order:** kept detections first, in detection order, unless over the cap
  (`sam3_video_base.py: _drop_new_det_with_obj_limit`).
- **State placement:** the state with the fewest empty slots that holds all new objects, the first created on ties;
  buckets fill in order (`sam3_multiplex_base.py:_tracker_add_new_objects`, `multiplex_utils.py:MultiplexState`).
- **Emission:** hidden if unconfirmed and live two frames later; objects removed on the frame are shown with a score of
  -1e4; overlaps go to the highest tracker probability, or detection probability on a birth frame, ties to the lower ID,
  and objects without pixels are dropped first (`sam3_multiplex_tracking.py:propagate_in_video`,
  `_postprocess_output_batched`, `video_tracking_multiplex_demo.py:_apply_object_wise_non_overlapping_constraints`).
- **Reproduced on purpose:** hot-start removals are not hidden retroactively; the boundary filter runs after NMS; the
  emitted score is the detection probability at birth.

## Detailed Implementation Plan

Paths are relative to the repository root; reference paths are written `sam3@f1647e8:<path>`. Each file is marked Create
or Modify. No file names, identifiers, tests or documentation carry the step numbers or milestone labels used below;
they name what the code does.

### Shared core and vision building blocks

- `src/equimo/core/layers/attention.py` (Modify)
  - `_fused_attention` accepts optional `key_value_seq_lengths`, broadcast to `q.shape[:-3]`, when no mask is given, so
    packed keys reach cuDNN without a dense mask, and zeroes the output rows where the length is 0. It also accepts an
    optional `bias`, so the DETR box bias reaches cuDNN. Padding to multiples of 64 stays as is.
  - `CrossAttention` as specified in "Attention and numerics", with `with_attention_implementation` support.
- `src/equimo/core/layers/__init__.py` (Modify): export `CrossAttention`.
- `src/equimo/core/layers/EGM.md` (Modify): document per-batch key lengths and `CrossAttention`.
- `tests/test_core_attention.py` (Modify): packed keys with lengths equal a dense masked call (XLA, CPU); a zero length
  returns zeros; `CrossAttention` matches a NumPy reference on small inputs, including fully masked rows returning zeros
  and an additive bias; the GPU-only cuDNN tests gain a packed-key case with per-bucket lengths under `vmap`.
- `src/equimo/core/tags.py` (Modify): `cross_attn` parents and their tuple entries get the attention tags, so the
  `attention.*` tags select cross-attention layers.
- `tests/finetune/test_tags.py` (Modify): those tags, including the tuple entries.
- `src/equimo/core/EGM.md` (Modify): the tag rule.
- `src/equimo/vision/ops.py` (Create):
  `roi_align(features[C, H, W], boxes[N, 4], output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)` with
  boxes as xyxy in feature pixels, matching torchvision's CUDA kernel exactly, which reads the border for samples
  outside the map (adaptive sampling with a static maximum of samples per bin and masking: at most 11 x 11 on a 72 x 72
  map with 7 x 7 bins); `resize_bilinear(x[..., H, W], size)` for mask logits, at torch's float32 sample positions;
  `grid_sample(features[C, H, W], points[N, 2])` with points in [-1, 1], bilinear, `align_corners=False` and zero
  padding, built on `jax.scipy.ndimage.map_coordinates`. Imported explicitly; `src/equimo/vision/__init__.py` is not
  changed.
- `tests/test_vision_ops.py` (Create): each operation against small torch and torchvision fixtures, including boxes on
  the map border and sub-pixel boxes, with `allclose` 1e-5; `jax.image.resize` against torch's bilinear and antialiased
  resize at the sizes the model uses (custom resizing code is added only if this fails).
- `tests/data/vision_ops_reference.npz` (Create): those fixtures; provenance entry with
  `checkpoint: "none (operator fixture)"`.

### Language

- `src/equimo/language/models.py` (Modify): `CausalTextTransformer`, registered as `"causal_text_transformer"` with
  `modality="language"`, built on `TransformerEncoderStack` with a causal `attn_mask`, a learned position table, a final
  norm and an optional per-token output projection; `__call__` and `features` take `token_ids` and `padding_mask` (True
  = padding). A `_CAUSAL_TEXT_REGISTRY` and a same-named factory `sam31_text(pretrained=...)` built through
  `core/factory.py:build_model_variant`.
- `src/equimo/language/tokenizers.py` (Modify): `ClipBpeTokenizer`, adapted from OpenAI CLIP's MIT `simple_tokenizer.py`
  (CLIP text cleaning, byte-level BPE, 32-token context with the end token kept on truncation), returning ids and a
  padding mask; guarded by a `_require_clip()` that names the `clip` extra in its error. It reads the vocabulary from
  package data, or from a `path` argument.
- `src/equimo/language/data/bpe_simple_vocab_16e6.txt.gz` (Create): the MIT CLIP vocabulary, as package data.
- `src/equimo/language/__init__.py` (Modify): export both.
- `src/equimo/language/EGM.md` (Modify): the new model, factory, tokenizer, extra and padding convention.
- `tests/test_language.py` (Modify): tiny `CausalTextTransformer` parity against a reference-generated fixture
  (`allclose` 1e-5), causal masking (changing a later token leaves earlier outputs bit-exact), jit, vmap and bfloat16.
- `tests/test_language_tokenizers.py` (Modify): with `require_extra("ftfy", "clip")` and
  `require_extra("regex", "clip")`, token ids for a prompt corpus (lower and upper case, accents, emoji, HTML entities,
  prompts longer than 32 tokens) equal the expected ids exactly; the missing-extra error.
- `tests/data/tiny_clip_bpe.txt.gz` (Create): the first N merges of the CLIP vocabulary plus the special tokens.
- `tests/data/clip_tokenizer_reference.json` (Create): expected ids, produced by the reference tokenizer run by the
  harness on that same tiny file.

### SAM 3.1 vision package (`src/equimo/vision/models/sam3/`, Create)

- `__init__.py`: `__all__` with `Sam3`, `sam31_multiplex`, the typed outputs, prompts, `decode_concepts` and
  `pad_tracker_buckets`. It does not import `video`.
- `common.py`: one `MLP` module with `layers.{i}` linear fields and an activation argument, used by every SAM head;
  `PositionedAttention`; the sine position encoding and the transposed `VisionRoPE` factors, both computed while
  tracing.
- `encoder.py`: `ViTDetBlock` and the ViTDet trunk (flat `blocks` list, run-time tiled position table), the tri-neck
  with flipped transposed-convolution weights, and `ImageFeatures`.
- `detector.py`: `ConceptPrompt`, the geometry encoder (using `equimo.vision.ops`), the fusion encoder, the DETR decoder
  with presence token, box refinement and box relative-position bias, dot-product scoring and joint logit, the
  segmentation head, and `DetectorOutput`.
- `interactive.py`: the prompt encoder (random Fourier matrices as `buffer()`), the two-way transformer, the SAM 2-style
  mask decoder with stability selection, and the multiplex mask decoder (16 slots, valid and invalid embeddings) that
  shares the transformer code; `InteractivePrompt` and `InteractiveOutput`.
- `tracker.py`: the memory encoder (32-channel downsampler and a fuser of
  `vision/layers/convolution.py:ConvNeXtBlock(act_layer="exactgelu", init_values=1e-6)`), memory attention with packed
  keys and the private query-chunked float32 path, pointer projections and temporal encodings, `TrackerState`,
  `TrackOutput`, `Association`, `CommitPlan`, the step functions behind `propagate`, `associate` and `commit`, and the
  function that pads the bucket axis.
- `decoding.py`: fixed-shape IoM and IoU matrices, both NMS variants, `decode_concepts`, boxes from mask extents, and
  per-pixel non-overlap.
- `model.py`: the `Sam3` module (fields `trunk`, `neck`, `detector`, `interactive`, `tracker`), registered as `"sam3"`
  with `modality="vision"`, with the endpoints in "Public interface"; `_SAM3_REGISTRY` and the factory
  `sam31_multiplex(pretrained=...)`.
- `video.py`: `Sam3VideoTracker` and `PointTrack`, which drive the jitted steps from the host, and `AssociationPolicy`
  (NumPy only), holding the thresholds as module constants with reference citations in docstrings, object metadata,
  bucket assignment, bucket release and the decision to grow.
- `EGM.md`: interface, the experimental pytrees' field lists and per-field sizes (`TrackerState`, `TrackOutput`,
  `Association`, `CommitPlan`), the emission timeline, state invariants (slot status transitions, ring-buffer contents,
  traced frame indices, bucket release), license scope of the weights, and the verification command.

### Registration, conversion and archives

- `src/equimo/_pretrained.py` (Modify): digests for `sam31_multiplex` and `sam31_text`, and the new repository revision,
  after upload.
- `models/sam31.py` (Create): the converter, with a positional variant (`multiplex` or `text`), `--checkpoint`,
  `--source-dir`, `--output-dir` and `--dry-run` (printing `checkpoint=` and `output=` lines), run in the `reference`
  dependency group. It checks the source file's SHA-256, loads it with `torch.load(weights_only=True)` on CPU,
  classifies every key (converted, computed and dropped, or unused and dropped), fails on unknown keys and while any
  model parameter has no source, splits the packed attention projections, drops the position table's cls row, converts
  transposed convolutions with `swapaxes(0, 1)` and a spatial flip (forward through `eqx.nn.ConvTranspose2d`), and
  writes the archive with a conversion record. The torch-to-Equimo key mapping lives here and is imported by the tests
  through the existing `models` import pattern.
- `models/sam31_reference.py` (Create): the reference harness, run in its own environment. Builds the float32 reference,
  the bfloat16 reference with `use_fa3=False`, and the tiny random-weight configuration (a configuration table shared
  with the tests); registers forward hooks that save outputs; records decision traces from the policy; asserts the 1e-3
  threshold margin and no score ties at sorting sites; and writes npz files with keys `state.<torch key>` and
  `outputs.<equimo path>`, plus provenance.
- `models/sam31_reference_requirements.txt` (Create): pinned versions for that environment (the reference at its full
  40-character commit, its torch and torchvision, NumPy below 2), separate from `uv.lock` because of the NumPy conflict.
- `models/prefetch_reference_checkpoints.py` (Modify): also fetch the hosted video fixture through the test helper, so
  the scheduled job can run offline. It already reads archive identifiers from provenance.
- `tests/test_conversion_cli.py` (Modify): add `sam31.py` to `SCRIPTS` and `HELP_SCRIPTS`.

### Tests and fixtures

- `tests/test_sam3.py` (Create):
  - tiny-configuration parity per component (trunk, neck, text features into the detector, geometry encoder, fusion
    encoder, decoder layers, scoring, segmentation head, prompt encoder, both mask decoders, memory encoder, memory
    attention) with `allclose` 1e-5;
  - decoding on tiny outputs, including both NMS variants, bit-exact keep masks;
  - jit, vmap and gradient checks, and a bfloat16 run with finite outputs;
  - tracker invariants: one compilation per bucket count while objects are added and removed, slot status never returns
    from removed, ring buffers hold the right frame indices, released buckets are reused with cleared state;
  - `propagate` and `commit` run under `lax.scan` with a fixed plan and match the step-by-step loop bit-exactly;
  - a new object by points: `allclose` 1e-5 against the reference's partial propagation on the tiny clip.
- `tests/test_sam3_video.py` (Create): CPU-only replay of the policy. The reference's recorded per-frame device outputs
  for the tiny clip are fed to Equimo's policy, which must produce the identical decision trace; and Equimo's tiny
  network driven by the reference's decisions must reproduce the reference outputs (`allclose` 1e-5).
- `tests/data/sam3_tiny_reference.npz`, `tests/data/sam3_tiny_reference.json` (Create): tiny fixtures and the decision
  trace in the JSON companion, following the EoMT pattern, within 3 MB like the existing largest fixture.
- `tests/data/sam3_tiny_occlusion_reference.npz` and its JSON (Create): a scripted clip for the occlusion suppression
  the full-weight clip does not trigger.
- `tests/cases/sam31_video_fixture.py` (Create): downloads and reads the hosted video fixture, with its digest pinned.
- `tests/data/sam31_image_reference.npz` (Create): full-weight component outputs and text features on one real image,
  stored as fixed seeded index subsets of each output with the indices inside the npz; the preprocessed image is stored
  as the reference's resized uint8 image. It also holds the reference's detections for one box exemplar with no text, on
  a box whose every threshold comparison clears the 1e-3 margin.
- `tests/cases/reference_cases.py` (Modify): a `REFERENCE_CASES` entry for `sam31_image_reference` with
  `mean_absolute_error` 5e-4, needing both archives.
- `tests/test_reference_parity.py` (Modify): the family sets gain `("sam31", "vision")` and `("sam31", "text")`.
- `tests/test_sam3_reference.py` (Create): marked `reference_parity`, outside `REFERENCE_CASES`: image decoding on the
  real image and the full-weight video check, reading a hosted fixture archive downloaded by a named helper with a
  digest pinned in `_pretrained.py`'s style.
- `tests/data/reference_provenance.json` (Modify): entries for every new `.npz`, with the fixed tolerances and the
  measured values; `models/validate_references.py` needs no change, because every new `.npz` in `tests/data` gets an
  entry and JSON companions are not listed.
- `tests/data/README.md` (Modify): the SAM reference environment and the hosted video fixture.
- `tests/cases/pretrained_cases.py` (Modify): `"sam31": "sam31_"` in `PRETRAINED_FAMILY_PREFIXES` and representative
  cases.
- `tests/test_pretrained_coverage.py` (Modify): add the SAM 3.1 vision and text families.
- `tests/cases/model_variant_cases.py` (Modify): the SAM 3.1 and causal-text registries, so both archives map to named
  factories.
- `tests/cases/model_cases.py` (Modify): tiny `Sam3` and `CausalTextTransformer` cases appended at the end of
  `MODEL_CASES` (seeds follow position), with their `features` endpoints, which bring them under the output manifest,
  feature contract, PRNG-free inference and ONNX checks. Both stay in the ONNX matrix.
- `tests/test_checkpoint_signature_stability.py` (Modify): `BUILDERS` entries for tiny `Sam3` and
  `CausalTextTransformer`.
- `tests/data/model_output_references.json` and `tests/data/model_structure_signatures.json` (Modify): regenerated with
  `uv run python models/generate_model_contract_references.py` and
  `uv run python tests/test_checkpoint_signature_stability.py`; the diff must touch only the new keys.
- `tests/cases/finetune_cases.py` (Modify): a LoRA case over the encoder and detector attention projections and a
  head-only case on the tiny `Sam3`.
- `tests/finetune/test_real_model_strategy_coverage.py` (Modify): the asserted set of model names gains `"sam3"`.
- `tests/test_optional_dependencies.py` (Modify): the `clip` extra error.
- `tests/test_package_metadata.py` and `tests/test_distribution.py` (Modify): the SAM License snapshot in the expected
  license files; the provenance note in the source distribution.
- `tests/ci_shards.py` (Modify): add `tests/test_sam3.py` and `tests/test_sam3_video.py` to the model shard.
- `.gitlab-ci.yml` (Modify): a `test:extra:clip` job like `test:extra:language`, running the tokenizer tests with the
  `clip` extra.

### Licensing, metadata and documentation

- `LICENSES/pretrained/sam-License.md` (Create): the SAM License snapshot (last updated 2025-11-19).
- `LICENSES/pretrained/README.md` and `models/huggingface/README.md` (Modify): a `sam31` row in both license tables with
  all eight columns (family, prefix `sam31_`, count 2, license, snapshot, source, checked date, current license).
- `models/huggingface/NOTICE` (Modify): attribution and conversion notice for SAM 3.1.
- `NOTICE` (Modify): entries for the adapted CLIP tokenizer and vocabulary (MIT) and the SAM 3.1 weights.
- `pyproject.toml` (Modify): the snapshot in `license-files` after `lingbot` and before `siglip2`; a `clip` extra with
  `regex` and `ftfy` (a range starting at 6.1); `regex.**` and `ftfy.**` in `allowed-unresolved-imports`; the provenance
  note in `[tool.uv.build-backend] source-include`. The vocabulary under `src/equimo/language/data/` is included by the
  build backend like any package file; the distribution tests' wheel inventory gains it.
- `uv.lock` (Modify): relocked with the devenv-provided uv.
- `docs/licensing/sam3-provenance.md` (Create): how the code was written, the sources read before and after the first
  draft, whether legal review happened before release, and a log of each reference-guided fix (behavior, test,
  revision).
- `docs/sam3.md` (Create): image, interactive and video usage, the coordinate and label conventions, the 32-token text
  limit, the emission lag, `max_objects`, recompilation on bucket growth, the experimental tier, and the SAM License
  terms that apply to users of the weights.
- `docs/usage.md` (Modify): a short pointer to `docs/sam3.md`.
- `docs/stability.md` (Modify): the image endpoints and text model as stable once full-weight parity passes;
  `TrackerState`, `CommitPlan`, `propagate`, `associate`, `commit` and the video driver as experimental.
- `docs/testing.md` (Modify): the hosted video fixture and the SAM reference environment.
- `docs/performance.md` (Modify): measured memory and latency on the RTX 4080 for image inference, video tracking at 1,
  2 and 4 buckets, and LoRA fine-tuning, in bfloat16 with XLA and cuDNN attention.
- `README.md` and `CHANGELOG.md` (Modify): the new family, extra and documentation.
- `src/equimo/EGM.md`, `src/equimo/vision/EGM.md`, `src/equimo/vision/models/EGM.md` (Modify): the new subpackage,
  operations module and family, and the `features` endpoint returning the detector-level map.

The model catalog (`docs/model_catalog.md`, `src/equimo/catalog.py`) is deliberately representative and is not changed.

### Order of work and verification

`pytest` passes after every step: a family is registered in the same step as its `ModelCase`, output references and
structure signatures. Until digests are pinned, scheduled tests read archives through a local archive-path override that
only those tests use.

Drafting, from this design, Equimo's code, the SAM 3 paper and the SAM 3.1 release post, without reading the reference:

1. **Shared blocks.** `core/layers/attention.py`, `core/layers/__init__.py`, `core/layers/EGM.md`, `vision/ops.py` and
   their tests against NumPy expectations. Check: new tests pass; existing attention and RoPE tests pass unchanged.
1. **Text model and tokenizer.** The language files, package data, `pyproject.toml` extra, `uv.lock`, the CI job and the
   optional-dependency test; `causal_text_transformer` registered with its model case, references and signature. Check:
   causal masking bit-exact, jit, vmap, bfloat16.
1. **Image modules and tracker modules.** `common.py`, `encoder.py`, `detector.py`, `interactive.py`, `decoding.py`, the
   tracker's memory encoder, memory attention and multiplex decoder, `model.py`; `sam3` registered with its model case,
   references, signature and fine-tuning cases. Check: shapes, jit, vmap, gradients and finite bfloat16 outputs.
1. **Tracking state and steps.** The rest of `tracker.py`. Check: one compilation per bucket count, slot and ring-buffer
   invariants, the scan test.
1. **Policy and driver.** `video.py`. Check: the driver runs a random-weight clip end to end with growth and release.
1. **Converter key classification.** `models/sam31.py` without archive writing. The key names, shapes and digest of
   `sam3.1_multiplex.pt` are read for this mapping; the reference code is not. Check: every checkpoint key is
   classified; the dry run prints the expected lines. The drafted model does not cover the checkpoint (149 unmapped keys
   and 62 parameters without a source), so a run fails and lists them until the reference comparison below corrects the
   model.

Comparison with the reference starts here. Image milestone:

1. **Reference harness.** `models/sam31_reference.py`, its requirements file, operator and tiny fixtures. Check: tiny
   fixtures regenerate byte-identically twice.
1. **Tiny component parity.** Component and operator tests against fixtures; each fix logged in the provenance note.
   Check: `allclose` 1e-5 everywhere, tokenizer ids exact.
1. **Archives and full-weight image parity.** Archive writing in `models/sam31.py`; the image reference case and
   `tests/test_sam3_reference.py` image part. Check: mean absolute error at most 5e-4 per output; image decoding kept
   set and IoU; bfloat16 image criteria; LoRA and head-only 50 steps in 16 GB with finite loss; memory measured.
1. **Release of the image path.** Upload with approval, digests and pretrained inventory edits (`_pretrained.py`,
   pretrained and variant cases), licenses, NOTICE files, documentation. Check: the full CI-equivalent checks.

Video milestone:

1. **Tiny tracker and policy parity.** Single-step parity from an injected state at 1 and 2 buckets, the replay test,
   new-object-by-points parity; fixes logged. Each threshold and tie rule gains its reference location in the appendix,
   the policy docstrings and `tracker.py`. Check: identical decision trace, `allclose` 1e-5.
1. **Full-weight video.** The hosted clip and the video part of `tests/test_sam3_reference.py`. Check: identical trace
   and IDs, mask IoU at least 0.99, margin and coverage assertions pass, bfloat16 video criterion, peak memory recorded
   against 15 GB at 4 buckets.

Before declaring the work complete, run the focused tests, then the CI-equivalent checks from `AGENTS.md` (`ruff check`,
`ruff format --check`, `mdformat --check`, `ty check`, `pytest`), `uv run python tests/ci_shards.py validate`, and a
fresh `uv build --no-sources --out-dir <staging-directory>` checked by the distribution tests. The full-weight parity
runs are local, on the RTX 4080, because the scheduled job has no GPU runner. On GPU they set
`JAX_DEFAULT_MATMUL_PRECISION=highest`: XLA's default GPU matmul precision alone puts trunk features 9e-4 from exact
float32 arithmetic, against 3e-6 at the highest precision (measured). The CPU job needs no setting.

______________________________________________________________________

*Drafted with EGM*
