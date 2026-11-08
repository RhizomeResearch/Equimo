# equimo.vision.models.sam3

## Overview

SAM 3.1 in JAX and Equinox: a ViTDet image encoder with three feature pyramids, a DETR-based concept detector with a
presence score, a SAM 2-style interactive head, and the tracker's prompt encoder, memory encoder, memory attention and
multiplex mask decoder. Every endpoint is unbatched and batched with `vmap`. Text arrives as features from the separate
`sam31_text` model in `equimo.language`. Only the converted weights are under the SAM License; the code stays under
Equimo's license.

## Interface

- `Sam3(*, key, in_channels, img_size, patch_size, embed_dim, depth, num_heads, mlp_features, window_size, global_blocks, dim, head_num_heads, ffn_features, geometry_depth, fusion_depth, decoder_depth, num_queries, two_way_depth, memory_depth, memory_heads, num_slots)`,
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
- Tracker modules on `Sam3.tracker`: `prompt_encoder`, `memory_encoder(features, masks (2 * num_slots, H, W))`,
  `memory_attention(x, memory, memory_pos, *, num_spatial_keys, key_lengths)` and
  `mask_decoder(image, image_pos, sparse, dense, high_res, slot_valid) -> MultiplexOutput`. `[tracker.py:Sam3Tracker]`,
  `[tracker.py:MemoryEncoder]`, `[tracker.py:MemoryAttention]`, `[interactive.py:MultiplexMaskDecoder]`

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
  padded points, the box corners when there is no box, and the appended padding point when there is one.
  `[interactive.py:PromptEncoder]`, `[detector.py:GeometryEncoder]`
- Attention softmax, RoPE rotation, layer norms, the DETR feed-forward, and mask logits run in float32; outputs return
  in the model dtype, except mask logits, which stay float32. `[common.py:PositionedAttention]`,
  `[common.py:layer_norm]`, `[detector.py:DecoderBlock]`
- The joint logit is `logit(sigmoid(class) * sigmoid(presence))`, computed so that it stays finite for large logits.
  `decode_concepts` does not apply presence again. `[detector.py:joint_logit]`, `[decoding.py:decode_concepts]`
- Mask selection counts point labels including the two box corners: with at most one, the best of the three multimask
  outputs by predicted IoU; with more, the single-mask output unless its stability score is below 0.98.
  `[interactive.py:_select_mask]`
- `single_frame_nms` measures overlap against the higher-ranked mask's area, and a suppressed mask still suppresses
  later ones; `greedy_iom_nms` uses intersection over minimum, and only kept masks suppress. Ranking uses a stable sort.
  `[decoding.py:single_frame_nms]`, `[decoding.py:greedy_iom_nms]`
- The multiplex decoder sets the mask logits of a slot whose object score is not positive to -1024.
  `[interactive.py:MultiplexMaskDecoder]`
- Memory attention expects the valid keys packed first: spatial keys of each frame, then object pointers, then unused
  keys. RoPE repeats per memory frame and leaves pointers unrotated. Without cuDNN, queries run in blocks of 1,024,
  because float32 logits against every key would not fit in memory. `[tracker.py:MemoryAttention]`,
  `[tracker.py:_memory_key_rotary]`, `[tracker.py:_chunked_attention]`
- `MemoryEncoder` resizes masks to 16 times the feature grid in float32 before downsampling.
  `[tracker.py:MemoryEncoder]`

## Side Effects

- Importing the package registers `"sam3"` in the model registry. `[model.py:Sam3]`
- `sam31_multiplex(pretrained=True)` downloads through `equimo.serialization.load_weights`. `[model.py:sam31_multiplex]`

## Verification

- Run: `devenv shell -- uv run pytest tests/test_sam3.py -q`
- cuDNN attention needs a CUDA machine; see "Environment and Commands" in `AGENTS.md` for running the GPU-only tests.
