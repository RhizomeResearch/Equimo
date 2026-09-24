# EoMT segmentation

`equimo.vision.models.EoMT` adds learned mask queries to the final blocks of a dynamic-size `VisionTransformer`. The
model returns class logits, mask logits, and auxiliary predictions in encoder execution order. The last prediction is
`output.final`; each auxiliary prediction is made immediately before a selected query-processing block. Inputs are
unbatched, channel-first images already normalised for the chosen backbone. Use `jax.vmap` for batches.

```python
import jax.random as jr

from equimo.vision.models import EoMT, dinov3_vits16_pretrain_lvd1689m
from equimo.vision.segmentation import semantic_scores, panoptic_predictions

backbone = dinov3_vits16_pretrain_lvd1689m(key=jr.PRNGKey(0))
model = EoMT(
    backbone,
    num_classes=19,
    num_queries=100,
    num_blocks=4,
    key=jr.PRNGKey(1),
)
prediction = model(normalised_image, key=jr.PRNGKey(2)).final

semantic = semantic_scores(
    prediction.mask_logits,
    prediction.class_logits,
    output_size=original_image_size,
)
panoptic = panoptic_predictions(
    prediction.mask_logits,
    prediction.class_logits,
    stuff_classes=(0, 1, 2),
)
```

The example uses randomly initialised weights. Load a compatible backbone and trained EoMT checkpoint for meaningful
predictions. `num_classes` excludes the additional no-object class in the query classifier. Class indices and the
`stuff_classes` sequence must follow the checkpoint's ontology. The image size must be divisible by the patch size.

`joint_features(image, key=..., inference=..., mask_state=...)` returns final normalized tokens in query/prefix/patch
order. `token_trace(...)` returns the joint tokens at every prediction point. The existing `features(...)` method
returns the same final joint tokens. For code that accepts either EoMT or PMT predictions, annotate the result with
`equimo.vision.segmentation.QuerySegmentationOutput`.

`semantic_scores` softmaxes query class logits, excludes the no-object class, sigmoids each mask, and sums their
products. Resize mask **logits** before calling it when output geometry differs. `semantic_class_map` returns the
per-pixel argmax. For sliding-window evaluation, call `semantic_scores` on each crop and use `merge_semantic_crops` to
average overlapping scores before the final resize. The supplied crops must cover the entire scaled image.

For panoptic evaluation, use `restore_panoptic_mask_logits` when the model input was resized and padded; pass the
restored logits to `panoptic_predictions`. The adapter keeps queries whose best class is not no-object and exceeds the
class threshold, assigns each pixel to the strongest score-weighted mask, rejects segments below the overlap threshold,
and merges masks of the same stuff class. `PanopticPrediction.class_ids` uses `num_classes` for void pixels and
`segment_ids` uses `-1` for void pixels. Each non-void thing receives its own segment ID. Class and overlap thresholds
default to `0.8`.

## Training and inference state

Use `eomt_full_finetune(model)` from `equimo.finetune.vision` to prepare the full-encoder plan. This includes the query
embeddings, encoder, class and mask heads, and upscaler. Task losses and target matching are supplied by the training
loop.

`anneal_mask_state(step, start_steps, end_steps)` computes a polynomial mask-retention probability for each
query-processing block at a given optimizer step. Pass its `EoMTMaskState` and an explicit PRNG key to the model during
training. Store the step, per-block probabilities, and schedule boundaries with the checkpoint so a resumed run uses the
same mask policy.

```python
from equimo.vision.models.eomt import anneal_mask_state, mask_free_eomt

state = anneal_mask_state(step, start_steps, end_steps)
output = model(normalised_image, key=step_key, inference=False, mask_state=state)

# Only after every schedule has ended:
terminal = anneal_mask_state(final_step, start_steps, end_steps)
inference_model = mask_free_eomt(model, terminal)
prediction = inference_model(normalised_image).final
```

The mask-free model emits only the final prediction. Disabling masks on an intermediate checkpoint changes its
evaluation behavior and requires separate validation. Save full EoMT weights with `equimo.serialization.save_model` and
restore into the same architecture with `load_weights`.

The numerical reference in `tests/data/eomt_tiny_reference.npz` was generated from the
[EoMT authors' implementation](https://github.com/tue-mps/eomt) at revision `7bd19ddd621c5c6adedcd260458a34783cd4a45f`.
Its recorded provenance and generation command are in `tests/data/reference_provenance.json`.
