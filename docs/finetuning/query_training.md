# Query-segmentation training

`equimo.vision.query_training` supplies JAX-native matching, point sampling, and set losses for outputs implementing
`QuerySegmentationOutput`, including EoMT and PMT. It has no optimizer or training-loop dependency. The Hungarian solver
runs on the JAX device, supports `jit` and `vmap`, and does not invoke a host solver or callback. Model inference does
not import the training utilities.

## Targets and supervision

Pass one `QueryTargets` per image:

| Field           | Shape     | Meaning                                                     |
| --------------- | --------- | ----------------------------------------------------------- |
| `class_ids`     | `(T,)`    | Integer task class indices in `[0,C)`                       |
| `masks`         | `(T,H,W)` | Target membership in `[0,1]` on supervised pixels           |
| `target_valid`  | `(T,)`    | Boolean; true marks a present target slot                   |
| `spatial_valid` | `(H,W)`   | Boolean; true marks a supervised, unpadded pixel            |
| `example_valid` | scalar    | Boolean, defaults to true; false excludes the whole example |

Prediction class logits have shape `(Q,C+1)`; index `C` is no-object. Mask logits have shape `(Q,Hm,Wm)` and can differ
in resolution from targets. The static target capacity must satisfy `0 <= T <= Q`, even when some slots are invalid. Use
`jax.vmap` to batch these interfaces. Validity here means inclusion; it is distinct from feature-exclusion masks.

Target validity is authoritative. A valid target with an empty mask remains a target. The data adapter must mark absent
classes or objects as invalid slots. Ignored target values and invalid slot class IDs can be arbitrary; they are removed
before arithmetic or indexing. Active class IDs and mask values must be valid, and supervised predictions must be
finite.

An image with supervised pixels but no valid target slots supplies **negative supervision**: its queries receive
no-object classification loss. An image with no supervised pixels, or `example_valid=False`, contributes zero to every
loss and normalizer. In partially annotated images, the adapter is responsible for declaring the target set under the
task's annotation-completeness policy. These utilities cannot infer whether an unlisted object is absent or unannotated.

## Prepare once, evaluate differentiably

```python
from equimo.vision.query_training import (
    QueryLossConfig,
    prepare_query_loss,
    query_segmentation_loss,
)

config = QueryLossConfig(matching_num_points=1024, loss_num_points=1024)
output = model(image, key=forward_key, inference=False)
context = prepare_query_loss(output, targets, key=sampling_key, config=config)
result = query_segmentation_loss(output, targets, context)
loss = result.total
```

Place these operations together inside `eqx.filter_value_and_grad` and `eqx.filter_jit`. Matching and point selection
are stopped from differentiation; the loss differentiates the original predictions. `context.final` and
`context.auxiliary` retain assignments, matching points, and per-target mask-loss points. Pass the same context to
replay the objective without resampling. Context is bound by the caller to the same targets and stochastic forward
realization; it does not hash model arrays or detect reuse after a parameter update.

`QueryLossConfig` has separate `(class, mask BCE, Dice)` `cost_weights` and `loss_weights`, both defaulting to
`(2,5,5)`, plus `no_object_weight=0.1`. All are configurable. Matching uses uniform sampling. Mask losses use
`sampling_policy="uncertainty"` by default, with `oversample_ratio=3`, `importance_sample_ratio=0.75`, and 12,544 points
for each of matching and loss. Set `sampling_policy="uniform"` to use only uniform mask-loss points. Point counts are
positive static integers; small examples and tests should set appropriately small counts.

Each auxiliary output is independently matched by default. `auxiliary_matching="reuse_final"` reuses final matches; each
prediction still selects its own loss points. `auxiliary_weights=None` gives every auxiliary output unit weight, `()`
disables auxiliary losses, and a tuple supplies one nonnegative weight per output in model execution order. The total is
the weighted sum of final and auxiliary losses, without division by decoder depth. Keys are folded by prediction index,
with final index zero, and then split between matching and loss sampling.

Malformed shapes or static configuration raise errors. Invalid active values produce `result.is_valid=False` and
`result.total=NaN`; the matcher also exposes `assignment.is_valid`. This explicit device-side failure signal avoids
silently repairing invalid predictions and allows the training loop to reject the update. Always check finite loss and
input validity before committing parameters or model state.

Assignment loops are explicitly bounded. Internal float32 arithmetic overflow also returns an invalid assignment instead
of continuing an infeasible augmenting-path search.

## Costs, reductions, and point geometry

`query_matching_costs` exposes each `(Q,T)` cost matrix. Classification cost is negative class probability, with the
no-object entry included in the softmax. Mask BCE is a mean over sampled valid points. Dice cost is
`1 - (2 * sum(p*y) + 1) / (sum(p) + sum(y) + 1)`, where `p = sigmoid(mask_logits)`. Invalid target columns are zero and
excluded by `match_queries`.

`match_queries` assigns every valid target to a distinct query, returning fixed-capacity indices in target order.
Invalid slots have query index `-1`. The solver processes targets in index order, chooses the lowest column index when
search distances tie, and keeps an existing predecessor on equal costs. This is deterministic tie handling, not a
promise of lexicographically minimal assignments. Its worst-case work is `O(T²Q)`; cost storage is `O(TQ)`. Cost
construction uses matrix products rather than a `Q*T*P` tensor.

Every prediction's result exposes `LossTerm.numerator`, `.denominator`, and `.value`:

`query_prediction_loss` evaluates these unweighted terms for a single prediction and `PredictionLossContext`, without
auxiliary composition. `query_segmentation_loss` applies that operation to the full output.

| Term           | Numerator                                                         | Denominator                                                                                    |
| -------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Classification | Sum of weighted query negative log-likelihoods                    | Sum of query weights; matched queries have weight 1, unmatched queries have `no_object_weight` |
| Mask BCE       | Sum of each matched mask's mean BCE over its valid sampled points | Number of matched masks                                                                        |
| Dice           | Sum of matched masks' smoothed Dice losses                        | Number of matched masks                                                                        |

Empty denominators stay zero in the result and evaluate to zero loss. `supervised_examples` and `matched_masks` are also
exposed. Use `reduce_query_losses(vmapped_result)` to sum numerators and denominators before division; averaging
per-image scalar losses gives different weighting. `axis_name="devices"` additionally reduces across a named device
axis. `axis=None` means no local reduction, useful inside a named map. Each auxiliary output retains separate
statistics. These reductions do not aggregate parameter gradients or implement optimizer accumulation.

`sample_query_points` returns `PointSamples` with normalized `(x,y)` coordinates and boolean validity. Pixel centers are
`((x+0.5)/W,(y+0.5)/H)`. Uniform sampling chooses valid pixel cells with replacement and jitters within the chosen cell.
One valid pixel is sufficient; an empty domain returns all-invalid samples. With `mask_logits`, uncertainty sampling
selects the lowest absolute interpolated logits from an oversized uniform pool, then adds fresh uniform points. Tied
uncertainty preserves candidate order. Uniform outputs have shape `(P,2)`; uncertainty outputs have shape `(Q,P,2)`.

`sample_mask_points` accepts shared `(P,2)` coordinates and returns sampled values and validity; `vmap` it for distinct
per-mask points. It uses half-pixel bilinear interpolation. Prediction logits use border extension. For targets, pass
`spatial_valid`: invalid and out-of-bounds neighbors contribute neither values nor weights, and remaining weights are
renormalized. Points without support return zero and false. Ignored target zeros never become background negatives.
Validity-normalized targets and border-extended predictions are deliberate differences from reference zero-padded
`grid_sample` behavior at boundaries. No upstream random-stream or boundary parity is claimed. Sensitive calculations
and losses use float32, including with bfloat16 model outputs.

## PMT normalization and state

PMT and `PlainMaskDecoder` accept `example_valid=True` on forward/decode. For BatchNorm, false excludes the image from
all lateral moments and observation counts. It does not change the frozen encoder or mask model tokens. Pass finite
padded image inputs. Decide normalization participation independently from pixel annotation validity; for supervised
training that excludes wholly unlabelled images, pass `targets.supervised`.

```python
config = PMTConfig.small(
    norm_layer="batchnorm",
    norm_kwargs={"axis_name": ("pmt_batch", "devices")},
)
```

Use the usual `vmap(..., axis_name="pmt_batch", out_axes=(0,None))` within a `pmap(..., axis_name="devices")`. BatchNorm
reduces valid sums and observation counts across both axes, then centered squared deviations. Training normalization
uses population variance; running variance uses the unbiased estimate. Prefix tokens remain observations. A device with
no valid images still participates in collectives. A globally empty batch returns zero normalization outputs and leaves
running state unchanged; a nonempty batch with only one observation is rejected. The state counter counts nonempty
normalization updates, not observations.

The default axis remains `"pmt_batch"`. Parameter/state array paths and default checkpoint configuration are preserved.
Explicit collective axes are recorded in `norm_kwargs` and checked during checkpoint loading; inference does not need
those axes to be bound. The qualified distributed route uses `pmap`. Checked `shard_map` with nested batch collectives
is not qualified on the tested JAX version.

Pass image batches and mutable normalization state as explicit arguments to compiled training steps, including `pmap`.
With the minimum supported JAX 0.8.1, capturing both a frozen encoder and the images as compile-time constants produced
different encoder values in an eager/JIT comparison. The dynamic-image route passes the minimum-version training checks;
this change does not repair that older compiler behavior. Reusing a prepared context also avoids selecting different
points when roundoff changes an uncertainty tie between eager and compiled forward execution.

Return normalization state as auxiliary output of the differentiated forward and commit it once when accepting an
update. Preparation and loss evaluation do not call the model again. If a caller intentionally recomputes a forward,
replay its keys, attention-mask state, sampling context, and incoming normalization state. Equinox state is single-use
in eager execution: clone a saved state using a PyTree flatten/unflatten (or identity `jax.tree.map`) before a replay.
Do not feed the provisional updated state into that recomputation.

## Ownership and evidence

Data adapters own ontology, target-set conversion, annotation completeness, paired geometry, and validity construction.
Training applications own coefficient/budget selection, metrics, normalization participation, all-ignore update
skipping, and evaluation policy. Optimizer integrations own gradient aggregation, accumulation, update rejection, state
rollback, and counters. Complete run checkpoints must additionally bind optimizer state, data position, configuration,
and random streams; model checkpoint helpers do not replace that transaction.

The implementation is grounded in these pinned official sources:

- [Optax Hungarian assignment](https://github.com/google-deepmind/optax/blob/225a7079f4630bf75bee94ec78de9aa69c60fab3/optax/assignment/_hungarian_algorithm.py),
  adapted for padded targets and fixed-capacity results; see `NOTICE`.
- [Mask2Former matcher](https://github.com/facebookresearch/Mask2Former/blob/161be514814ea63440fae520d2ec72707b62ba54/mask2former/modeling/matcher.py)
  and
  [criterion](https://github.com/facebookresearch/Mask2Former/blob/161be514814ea63440fae520d2ec72707b62ba54/mask2former/modeling/criterion.py),
  for class/BCE/Dice mathematics.
- [PointRend sampling](https://github.com/facebookresearch/detectron2/blob/9f8d35e653674932fbe7a579feee36ed9dc8e8c5/projects/PointRend/point_rend/point_features.py),
  for interpolation coordinates and uncertainty selection.
- [PMT criterion defaults](https://github.com/tue-mps/pmt/blob/0e803722aa5737a242b383dec1b90c2c66b86baa/image/training/mask_classification_semantic.py),
  for configurable starting weights and sampling budgets.

`tests/data/query_training_reference.json` contains values and gradients generated by upstream functions, with source
digests and environment versions. `models/qualify_query_training.py` reproduces it from digest-verified source files
without importing Equimo. Offline tests also compare tiny assignments with exhaustive enumeration, larger cases with
SciPy, and normalization outputs/gradients with independent NumPy calculations over concatenated valid observations.
Distributed tests use two logical CPU devices. Accelerator performance, multi-host execution, and selected pretrained
training runs need separate qualification.

See [the synthetic training example](../../examples/query_segmentation_training.py) for an executable single-step
update.
