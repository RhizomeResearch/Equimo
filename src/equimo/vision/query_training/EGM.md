# equimo.vision.query_training

## Overview

Optimizer-independent, JAX-native training utilities for query-based mask segmentation models (Mask2Former-style matcher
and criterion, PointRend point selection). The package prepares nondifferentiable query-to-target assignments and
sampling points for one stochastic forward, then evaluates differentiable classification, mask BCE, and Dice losses as
additive numerator/denominator statistics that can be reduced across batches and devices. Model outputs are consumed
through the `QueryMaskPrediction` / `QuerySegmentationOutput` protocols from
[`equimo/vision/segmentation.py`](../segmentation.py); the classification softmax and NLL come from `equimo.core.ops`.

## Interface

Public names are re-exported by `[__init__.py:__all__]`.

- `QueryAssignment(query_indices, valid, is_valid)`: fixed-capacity matches in target order; `query_indices[t]` is the
  query matched to target slot `t`, `-1` when unmatched. `[_assignment.py:QueryAssignment]`
- `match_queries(costs, target_valid) -> QueryAssignment`: padded rectangular Hungarian assignment on `(Q, T)` costs
  with `T <= Q`, adapted from Optax, run entirely on device. `[_assignment.py:match_queries]`
- `PointSamples(coordinates, valid)`: normalized `(..., points, 2)` x/y coordinates and boolean validity.
  `[_sampling.py:PointSamples]`
- `sample_mask_points(masks, points, *, spatial_valid=None) -> (values, valid)`: bilinear sampling of `(..., H, W)`
  masks at shared `(P, 2)` points; border extension for logits, renormalized valid-neighbor interpolation for targets.
  `[_sampling.py:sample_mask_points]`
- `sample_query_points(spatial_valid, *, key, num_points, mask_logits=None, oversample_ratio=3.0, importance_sample_ratio=0.75) -> PointSamples`:
  uniform sampling of valid pixel cells, `(P, 2)`; with `(Q, H, W)` logits, uncertainty-prioritized `(Q, P, 2)`
  sampling. `[_sampling.py:sample_query_points]`
- `QueryTargets(class_ids, masks, target_valid, spatial_valid, *, example_valid=True)`: one prepared target set with
  `(T,)` classes and validity, `(T, H, W)` masks in `[0, 1]`, `(H, W)` spatial validity; `supervised` property.
  `[_loss.py:QueryTargets]`
- `QueryLossConfig`: frozen dataclass of matcher/loss policy (`cost_weights`, `loss_weights` as (class, mask BCE, Dice),
  `no_object_weight`, `matching_num_points`, `loss_num_points`, `sampling_policy`, `oversample_ratio`,
  `importance_sample_ratio`, `auxiliary_weights`, `auxiliary_matching`). `[_loss.py:QueryLossConfig]`
- `MatchingCosts(classification, mask, dice, total, is_valid)`: inspectable `(Q, T)` costs. `[_loss.py:MatchingCosts]`
- `PredictionLossContext(assignment, matching_points, mask_points)` and `QueryLossContext(final, auxiliary, config)`:
  replayable assignment and sampling context. `[_loss.py:PredictionLossContext]`, `[_loss.py:QueryLossContext]`
- `LossTerm(numerator, denominator)` with `value` property; `PredictionLoss` (classification, mask, dice terms,
  `supervised_examples`, `matched_masks`, `is_valid`); `QueryLossResult(final, auxiliary, config)` with `is_valid` and
  `total` properties. `[_loss.py:LossTerm]`, `[_loss.py:PredictionLoss]`, `[_loss.py:QueryLossResult]`
- `query_matching_costs(prediction, targets, points, *, config=QueryLossConfig()) -> MatchingCosts`: class (negative
  probability), sampled BCE, and Dice costs on shared points. `[_loss.py:query_matching_costs]`
- `prepare_query_loss(output, targets, *, key, config=QueryLossConfig()) -> QueryLossContext`: samples matching points,
  computes costs, matches queries, and samples loss points for final and auxiliary predictions. Matching points are
  always uniform; `sampling_policy` selects only the loss points (per-query uncertainty points, or uniform points shared
  by all queries). With `auxiliary_matching="reuse_final"`, auxiliaries reuse the final assignment and matching points
  and sample only their own loss points. `[_loss.py:prepare_query_loss]`
- `query_prediction_loss(prediction, targets, context, *, config=QueryLossConfig()) -> PredictionLoss`: terms for one
  prediction before `loss_weights`; the classification term still applies `no_object_weight` per query.
  `[_loss.py:query_prediction_loss]`
- `query_segmentation_loss(output, targets, context) -> QueryLossResult`: evaluates final and auxiliary losses with a
  retained context. `[_loss.py:query_segmentation_loss]`
- `reduce_query_losses(result, *, axis=0, axis_name=None) -> QueryLossResult`: sums statistics over a local vmapped axis
  (booleans AND-reduced with `jnp.all`) and optionally across named devices (`psum`; booleans AND-reduced via `pmin`).
  `axis=None` skips the local reduction rather than reducing every dimension. `[_loss.py:reduce_query_losses]`

## Invariants

- `match_queries` requires 2-D costs, `target_valid.shape == (T,)`, `Q > 0`, `T <= Q`, and boolean `target_valid`
  (raises `ValueError` / `TypeError`). Costs are cast to float32 and wrapped in `lax.stop_gradient`; nonfinite active
  costs or overflow return `is_valid=False` with all indices `-1`. Every augmenting-path loop is bounded by the row
  index. `[_assignment.py:match_queries]`
- Unmatched queries and targets scatter into a spare slot instead of out of range, because deterministic GPU scatters
  mishandle out-of-range writes (openxla/xla#49380); `match_queries` adds zero for unvisited columns for the same
  reason. `[_assignment.py:match_queries]`, `[_loss.py:query_prediction_loss]`
- `sample_mask_points` requires positive spatial dims, shared `(P, 2)` coordinates, boolean validity matching the
  coordinates, and boolean `(H, W)` `spatial_valid`. Coordinates are stopped from differentiation, values are computed
  in float32, pixel centers are at `((x + .5) / W, (y + .5) / H)`, and invalid or unsupported points return `0` and
  `False`. Zero-weight neighbors cannot inject NaNs. `[_sampling.py:sample_mask_points]`
- `sample_query_points` requires a nonempty boolean `(H, W)` domain, a positive non-bool `int` `num_points`, finite
  `oversample_ratio >= 1`, and `importance_sample_ratio` in `[0, 1]`; uncertainty mode requires nonempty `(Q, H, W)`
  logits and stops their gradient. An empty domain returns all-invalid points. `[_sampling.py:sample_query_points]`
- `QueryTargets.__init__` raises unless masks are `(T, H, W)` with positive `H, W`, class IDs and slot validity are
  `(T,)`, `spatial_valid` is `(H, W)`, `example_valid` is scalar, class IDs are integers, and all validity arrays are
  boolean. `supervised` is `example_valid & any(spatial_valid)`. `[_loss.py:QueryTargets]`
- `QueryLossConfig.__post_init__` raises on non-3-element, nonfinite, negative, or all-zero weights; nonfinite or
  negative `no_object_weight`; non-integer or non-positive point counts; `sampling_policy` outside
  `{"uniform", "uncertainty"}`; nonfinite or `< 1` `oversample_ratio`; `importance_sample_ratio` outside `[0, 1]`;
  `auxiliary_matching` outside `{"independent", "reuse_final"}`; or nonfinite or negative auxiliary weights.
  `[_loss.py:QueryLossConfig]`
- Predictions must provide nonempty `(Q, C+1)` class logits and `(Q, H, W)` mask logits with the last class as
  no-object, and target capacity must not exceed the query count. Logits are cast to float32 for losses.
  `[_loss.py:_inputs]`
- `auxiliary_weights=None` weights every auxiliary output by one, `()` disables auxiliary losses, otherwise one weight
  per auxiliary output is required. `[_loss.py:_auxiliary_weights]`
- Malformed active values (nonfinite logits, invalid labels or mask values, infeasible or duplicated assignment) set
  `is_valid=False` and make `QueryLossResult.total` NaN instead of raising; callers must reject such updates. Shape and
  configuration errors raise before execution. `[_loss.py:QueryLossResult]`, `[_loss.py:query_prediction_loss]`
- `prepare_query_loss` folds `key` by prediction index (final `0`, auxiliary `i + 1`) and splits it into matching and
  loss keys, so disabling auxiliaries does not change the final prediction's stream. A context must be reused only with
  the same targets and forward draw. `[_loss.py:prepare_query_loss]`, `[_loss.py:QueryLossContext]`
- `LossTerm.value` divides by 1 when the denominator is not positive, so it returns the numerator, which is zero for the
  terms `query_prediction_loss` builds; aggregate numerators and denominators (for example with `reduce_query_losses`)
  before reading `total`. `[_loss.py:LossTerm]`, `[_loss.py:query_prediction_loss]`, `[_loss.py:reduce_query_losses]`
- `QueryLossContext.config` and `QueryLossResult.config` are static Equinox fields. `[_loss.py:QueryLossContext]`,
  `[_loss.py:QueryLossResult]`

## Side Effects

- Consumes the explicitly passed PRNG keys for uniform and uncertainty point sampling; no hidden randomness.
  `[_sampling.py:sample_query_points]`, `[_loss.py:prepare_query_loss]`
- `reduce_query_losses` performs cross-device collectives (`jax.lax.psum`, `jax.lax.pmin`) when `axis_name` is given.
  `[_loss.py:reduce_query_losses]`
- No host callbacks, file, or network access: assignment runs on device without a host solver.
  `[_assignment.py:match_queries]`

## Verification

- Run: `devenv shell -- uv run pytest tests/test_query_training.py tests/test_query_training_models.py -q`
