# equimo.vision

## Overview

The vision modality of Equimo: registered image models, vision-specific layers, query-segmentation training utilities,
and a few top-level helpers for image loading, segmentation post-processing, and bounded PMT checkpoints. `__init__.py`
holds only a docstring; import subpackages explicitly (`equimo.vision.models`, `equimo.vision.layers`,
`equimo.vision.query_training`) or the top-level modules below, including the sampling operations in
`equimo.vision.ops`.

## Interface

### Subpackages

- [`layers`](./layers/EGM.md): attention, convolution, patch, downsampling, positional-embedding,
  squeeze-and-excitation, and wavelet layers with name-keyed registries and the cross-registry resolver `get_layer`.
- [`models`](./models/EGM.md): registered architectures (`vit`, `vision_parcae`, `convnext`, `deq`, `eomt`, `pmt`,
  `fastervit`, `iformer`, `lowformer`, `mlla`, `vssd`, `attnet`, `mobilenetv3`, `partialformer`, `reduceformer`,
  `shvit`, `sam3`) and named preset factories, auto-exported through each module's `__all__`.
- [`query_training`](./query_training/EGM.md): JAX-native query matching, point sampling, and Mask2Former-style
  segmentation losses for EoMT and PMT outputs.

### Image loading (`io.py`)

- `load_image(path, mean=None, std=None, size=None, center_crop=False) -> [3, H, W]`: reads an RGB image with Pillow,
  scales to `[0, 1]` float32, optionally center-crops to a square, resizes bilinearly to `size x size`, normalizes with
  `mean` / `std` when both are given, and returns channel-first. `[io.py:load_image]`

### Sampling operations (`ops.py`)

- `roi_align(features (C, H, W), boxes (N, 4), output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False) -> (N, C, out_h, out_w)`:
  `torchvision.ops.roi_align`'s CUDA kernel for one image, with `(x1, y1, x2, y2)` boxes multiplied by `spatial_scale`;
  samples outside the map read the border pixel. `[ops.py:roi_align]`
- `resize_bilinear(x (..., H, W), size) -> (..., out_h, out_w)`: float32 resize that matches
  `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)` on CUDA, including its float32 sample
  positions. `[ops.py:resize_bilinear]`
- `resize_bilinear_antialias(x (..., H, W), size) -> (..., out_h, out_w)`: float32 resize that matches
  `torch.nn.functional.interpolate(mode="bilinear", align_corners=False, antialias=True)`: each axis uses a triangle
  filter widened by its downscale factor, and upscaling is plain bilinear. `[ops.py:resize_bilinear_antialias]`
- `grid_sample(features (C, H, W), points (N, 2)) -> (N, C)`: `torch.nn.functional.grid_sample` with bilinear
  interpolation, `align_corners=False` and zero padding, for `(x, y)` points in `[-1, 1]`. `[ops.py:grid_sample]`

### Segmentation post-processing (`segmentation.py`)

- `QueryMaskPrediction` (protocol with `class_logits`, `mask_logits`) and `QuerySegmentationOutput` (protocol with
  `final`, `auxiliary`): the prediction view shared by EoMT, PMT, and `query_training`.
  `[segmentation.py:QueryMaskPrediction]`, `[segmentation.py:QuerySegmentationOutput]`
- `PanopticPrediction(class_ids, segment_ids)`: class and segment maps; `num_classes` and `-1` mean void.
  `[segmentation.py:PanopticPrediction]`
- `resize_mask_logits(mask_logits, size)`: half-pixel bilinear resize of `(Q, H, W)` logits before the sigmoid.
  `[segmentation.py:resize_mask_logits]`
- `semantic_scores(mask_logits, class_logits, *, output_size=None) -> (K, H, W)` and `semantic_class_map(...)`: softmax
  class probabilities without the no-object column, combined with sigmoid masks. `[segmentation.py:semantic_scores]`,
  `[segmentation.py:semantic_class_map]`
- `semantic_crop_origins(scaled_size, crop_size)` and
  `merge_semantic_crops(crop_scores, origins, scaled_size, output_size)`: evenly spaced sliding-window crops and their
  averaged merge. `[segmentation.py:semantic_crop_origins]`, `[segmentation.py:merge_semantic_crops]`
- `restore_panoptic_mask_logits(mask_logits, *, padded_size, resized_size, output_size)`: upsample, remove right and
  bottom padding, resize to the original size. `[segmentation.py:restore_panoptic_mask_logits]`
- `panoptic_predictions(mask_logits, class_logits, *, stuff_classes, mask_threshold=0.8, overlap_threshold=0.8) -> PanopticPrediction`:
  assigns query masks to thing instances and merged stuff regions. `[segmentation.py:panoptic_predictions]`
- Mask operations on boolean `(N, H, W)` masks, all fixed-shape:
  - `iom_matrix(a, b)` (intersection over the smaller area, 0 when either mask is empty) and `iou_matrix(a, b)` (0 when
    both are empty). `[segmentation.py:iom_matrix]`, `[segmentation.py:iou_matrix]`
  - `non_greedy_iom_nms(masks, threshold)`, for masks sorted by decreasing score, where a suppressed mask still
    suppresses later ones, and `greedy_iom_nms(masks, scores, candidates, threshold)`, where only kept masks suppress
    and ranking uses a stable sort. `[segmentation.py:non_greedy_iom_nms]`, `[segmentation.py:greedy_iom_nms]`
  - `boxes_from_masks(masks)` (xyxy extents, zeros when empty) and `remove_overlaps(masks)` (each pixel to the first
    mask covering it, for masks sorted by decreasing priority). `[segmentation.py:boxes_from_masks]`,
    `[segmentation.py:remove_overlaps]`
- Internal helpers used by the query models: `_query_attention_mask` (per-query stochastic attention mask in
  query/prefix/patch order) and `_anneal_mask_probabilities` (polynomial per-block mask retention).
  `[segmentation.py:_query_attention_mask]`, `[segmentation.py:_anneal_mask_probabilities]`

### PMT checkpoints (`pmt_checkpoint.py`)

- `save_pmt_checkpoint(path, model, *, state=None, mask_state=None) -> Path` and
  `load_pmt_checkpoint(path, template, *, state=None, limits=None) -> (PMT, state, PMTMaskState)`: full encoder and
  decoder snapshots. The loaded model's `backbone_digest` is set to the digest of its loaded encoder.
  `[pmt_checkpoint.py:save_pmt_checkpoint]`, `[pmt_checkpoint.py:load_pmt_checkpoint]`
- `save_pmt_decoder(...)` and `load_pmt_decoder(...)`: decoder-only artifacts bound to exact encoder arrays.
  `[pmt_checkpoint.py:save_pmt_decoder]`, `[pmt_checkpoint.py:load_pmt_decoder]`

### Encoder identity (`_encoder_identity.py`)

- `encoder_array_digest(encoder, *, legacy_layout=False) -> str`: SHA-256 over every array leaf's path, shape, dtype,
  and bytes, plus the static fields of modules that compute values from their configuration (RoPE and Swin position
  embeddings), so encoders that differ only in RoPE settings get different digests; used for PMT `backbone_digest` and
  checkpoint binding. `legacy_layout=True` instead hashes the values `equimo._legacy_layouts` re-inserts, as encoders
  before Equimo 2.8 stored them. `[_encoder_identity.py:encoder_array_digest]`,
  `[../_legacy_layouts.py:computed_value_settings]`

## Invariants

- `load_image` raises `ValueError` when only one of `mean` / `std` is given, and requires Pillow, raising an
  `ImportError` naming the `extras` extra when it is missing. `[io.py:load_image]`
- `roi_align` with `sampling_ratio <= 0` takes `ceil(roi_size / output_size)` samples per bin and axis, evaluated
  against a static maximum of `ceil(H / out_h)` by `ceil(W / out_w)` with masking, so it matches torchvision for boxes
  no larger than the map and samples larger boxes with that maximum. `roi_align` and `grid_sample` compute coordinates
  and weights in float32 and return the dtype of `features`; gradients reach `features` and, through the interpolation
  weights, the boxes or points. `[ops.py:roi_align]`, `[ops.py:_bin_weights]`, `[ops.py:grid_sample]`
- `semantic_scores`, `semantic_class_map` and `panoptic_predictions` expect nonempty `(Q, H, W)` mask logits and
  `(Q, K + 1)` class logits whose last column is no-object, and raise `ValueError` otherwise. `resize_mask_logits` only
  requires a `(Q, H, W)` shape and a positive size, so it accepts an empty `Q`. Class softmax and mask sigmoid are
  computed in float32. `[segmentation.py:_validate_logits]`, `[segmentation.py:semantic_scores]`,
  `[segmentation.py:semantic_class_map]`, `[segmentation.py:panoptic_predictions]`,
  `[segmentation.py:resize_mask_logits]`
- `panoptic_predictions` requires unique stuff classes in `[0, K)` and thresholds in `[0, 1]`.
  `[segmentation.py:panoptic_predictions]`
- `merge_semantic_crops` checks on the host that crops stay inside the scaled image and cover every pixel (an uncovered
  pixel would spread NaN through the final resize); origins must therefore be static.
  `[segmentation.py:merge_semantic_crops]`
- `_anneal_mask_probabilities` requires equal nonempty start/end lists, each end after its start, and positive power;
  probabilities become exactly zero at the end step rather than when the float32 quotient reaches one.
  `_query_attention_mask` keeps a mask only when `uniform < probability`, so probability zero never keeps a mask.
  `[segmentation.py:_anneal_mask_probabilities]`, `[segmentation.py:_query_attention_mask]`
- PMT checkpoints: BatchNorm PMT models require a complete `eqx.nn.State`; stateless models reject a nonempty state; the
  mask state length must equal the decoder depth. Loading compares the recorded architecture configuration (schema,
  `kind`, PMT config, group count, masked attention, `backbone_id`, `input_view`, `class_ontology`, feature endpoint,
  prefix tokens, patch size, positional configuration) with the template and raises on mismatch.
  `[pmt_checkpoint.py:_check_state]`, `[pmt_checkpoint.py:_mask_state]`, `[pmt_checkpoint.py:_configuration]`,
  `[pmt_checkpoint.py:_check_configuration]`
- Full checkpoints verify the loaded encoder digest against the recorded `base_digest`; decoder-only artifacts require a
  bound `backbone_id` (not `"unbound"`) and the template's current encoder digest must match the recorded one. A
  recorded digest that matches the encoder's pre-2.8 layout is accepted too. `[pmt_checkpoint.py:load_pmt_checkpoint]`,
  `[pmt_checkpoint.py:save_pmt_decoder]`, `[pmt_checkpoint.py:load_pmt_decoder]`,
  `[pmt_checkpoint.py:_matching_base_digest]`
- `encoder_array_digest` pulls arrays to the host and must run outside traced functions.
  `[_encoder_identity.py:encoder_array_digest]`
- Global conventions for the subtree: modules are unbatched, forward endpoints take `key: PRNGKeyArray | None = None`
  and run keyless under `inference=True` (except masked-attention query models, and `VisionParcae` unless
  `state_init="zero"`), and string layer names resolve through the registries in [`layers`](./layers/EGM.md).
  `[models/parcae.py:VisionParcae._features_with_aux]`

## Side Effects

- `load_image` reads the image file from disk. `[io.py:load_image]`
- `save_pmt_checkpoint` and `save_pmt_decoder` write checkpoints through `equimo.serialization.save_model`, which
  appends `.tar.lz4` by default, so the returned path can differ from `path`. The loaders read `metadata.json` and
  weights through `inspect_checkpoint`, `_resolve_weights_dir`, and `load_weights` with optional `CheckpointLimits`; an
  archive is extracted into a sibling `<archive>.extracted/` directory, so loading can write to disk.
  `[pmt_checkpoint.py:save_pmt_checkpoint]`, `[pmt_checkpoint.py:_recorded_configuration]`,
  `[../serialization.py:save_model]`, `[../serialization.py:_decompress_archive]`
- `encoder_array_digest` transfers every array to the host with `jax.device_get`.
  `[_encoder_identity.py:encoder_array_digest]`
- Importing `equimo.vision.models` or `equimo.vision.layers` mutates the global model and layer registries (see the
  child contracts).

## Verification

- Run:
  `devenv shell -- uv run pytest tests/test_io.py tests/test_vision_ops.py tests/test_eomt.py tests/test_pmt.py tests/test_query_training.py tests/test_query_training_models.py tests/test_models.py tests/test_attention.py tests/test_convolution.py -q`
