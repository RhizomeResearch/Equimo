"""Query-mask costs and losses with explicit, reusable supervision context.

The mathematical reference is Mask2Former's matcher and criterion, revision
161be514814ea63440fae520d2ec72707b62ba54. Point selection follows PointRend.
Validity handling, structured reductions, and border interpolation are explicit
extensions; these utilities do not claim identical random streams to PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from equimo.vision.segmentation import QueryMaskPrediction, QuerySegmentationOutput
from ._assignment import QueryAssignment, match_queries
from ._sampling import PointSamples, sample_mask_points, sample_query_points


class QueryTargets(eqx.Module):
    """One prepared target set, independent of dataset label encodings.

    ``class_ids`` and boolean ``target_valid`` have shape (T,); ``masks`` has
    shape (T,H,W) and valid values in [0,1]. ``spatial_valid`` is boolean (H,W).
    A false scalar ``example_valid`` excludes the entire example. A real image
    with valid pixels and zero valid target slots supplies negative supervision;
    one without valid pixels supplies no classification or mask supervision.
    Target validity is authoritative: empty masks are not silently discarded.
    """

    class_ids: jax.Array
    masks: jax.Array
    target_valid: jax.Array
    spatial_valid: jax.Array
    example_valid: jax.Array

    def __init__(
        self, class_ids, masks, target_valid, spatial_valid, *, example_valid=True
    ):
        self.class_ids = jnp.asarray(class_ids)
        self.masks = jnp.asarray(masks)
        self.target_valid = jnp.asarray(target_valid)
        self.spatial_valid = jnp.asarray(spatial_valid)
        self.example_valid = jnp.asarray(example_valid)
        if self.masks.ndim != 3 or min(self.masks.shape[-2:]) <= 0:
            raise ValueError("Target masks must have shape (T,H,W), with positive H,W.")
        if (
            self.class_ids.shape != self.masks.shape[:1]
            or self.target_valid.shape != self.class_ids.shape
        ):
            raise ValueError("Target classes and slot validity must have shape (T,).")
        if (
            self.spatial_valid.shape != self.masks.shape[-2:]
            or self.example_valid.shape != ()
        ):
            raise ValueError(
                "Expected (H,W) spatial validity and scalar example validity."
            )
        if not jnp.issubdtype(self.class_ids.dtype, jnp.integer):
            raise TypeError("Target class IDs must be integers.")
        if any(
            value.dtype != jnp.bool_
            for value in (self.target_valid, self.spatial_valid, self.example_valid)
        ):
            raise TypeError("All target validity arrays must be boolean.")

    @property
    def supervised(self) -> jax.Array:
        """Whether this example contributes query classification supervision."""
        return self.example_valid & jnp.any(self.spatial_valid)


@dataclass(frozen=True)
class QueryLossConfig:
    """Static matcher/loss policy; coefficients are (class, mask BCE, Dice).

    ``auxiliary_weights=None`` gives every auxiliary output unit weight; an
    empty tuple disables auxiliary losses. Otherwise supply one nonnegative
    weight per auxiliary output. ``auxiliary_matching`` is ``independent`` or
    ``reuse_final``. Mask points are selected separately for every prediction.
    """

    cost_weights: tuple[float, float, float] = (2.0, 5.0, 5.0)
    loss_weights: tuple[float, float, float] = (2.0, 5.0, 5.0)
    no_object_weight: float = 0.1
    matching_num_points: int = 12544
    loss_num_points: int = 12544
    sampling_policy: str = "uncertainty"
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    auxiliary_weights: tuple[float, ...] | None = None
    auxiliary_matching: str = "independent"

    def __post_init__(self):
        for weights in (self.cost_weights, self.loss_weights):
            if (
                len(weights) != 3
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not any(weights)
            ):
                raise ValueError(
                    "Cost/loss weights require three finite nonnegative values, not all zero."
                )
        if not math.isfinite(self.no_object_weight) or self.no_object_weight < 0:
            raise ValueError("no_object_weight must be finite and nonnegative.")
        for count in (self.matching_num_points, self.loss_num_points):
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("Point counts must be positive integers.")
        if self.sampling_policy not in {"uniform", "uncertainty"}:
            raise ValueError("Unknown sampling_policy.")
        if (
            not math.isfinite(self.oversample_ratio)
            or self.oversample_ratio < 1
            or not 0 <= self.importance_sample_ratio <= 1
        ):
            raise ValueError("Invalid point oversampling or importance fraction.")
        if self.auxiliary_matching not in {"independent", "reuse_final"}:
            raise ValueError("Unknown auxiliary_matching policy.")
        if self.auxiliary_weights is not None and any(
            not math.isfinite(w) or w < 0 for w in self.auxiliary_weights
        ):
            raise ValueError("Auxiliary weights must be finite and nonnegative.")


class MatchingCosts(eqx.Module):
    """Inspectable (Q,T) costs; invalid target columns are zero."""

    classification: jax.Array
    mask: jax.Array
    dice: jax.Array
    total: jax.Array
    is_valid: jax.Array


class PredictionLossContext(eqx.Module):
    """One prediction's assignment and shared/per-target sampling coordinates."""

    assignment: QueryAssignment
    matching_points: PointSamples
    mask_points: PointSamples


class QueryLossContext(eqx.Module):
    """Replayable context; reuse only with the same targets and forward draw."""

    final: PredictionLossContext
    auxiliary: tuple[PredictionLossContext, ...]
    config: QueryLossConfig = eqx.field(static=True)


class LossTerm(eqx.Module):
    """Additive numerator and denominator; empty reductions evaluate to zero."""

    numerator: jax.Array
    denominator: jax.Array

    @property
    def value(self) -> jax.Array:
        return self.numerator / jnp.where(self.denominator > 0, self.denominator, 1)


class PredictionLoss(eqx.Module):
    """Unweighted terms, supervision counts, and active-input validity."""

    classification: LossTerm
    mask: LossTerm
    dice: LossTerm
    supervised_examples: jax.Array
    matched_masks: jax.Array
    is_valid: jax.Array


class QueryLossResult(eqx.Module):
    """Final/auxiliary terms; aggregate numerators before evaluating ``total``.

    Malformed active values (nonfinite logits, invalid labels/mask values, or
    infeasible assignment) set ``is_valid=False`` and ``total=NaN``. This is an
    explicit compiled failure signal without a host callback. Callers must
    reject that update. Shape/configuration errors raise before execution.
    """

    final: PredictionLoss
    auxiliary: tuple[PredictionLoss, ...]
    config: QueryLossConfig = eqx.field(static=True)

    @property
    def is_valid(self) -> jax.Array:
        valid = self.final.is_valid
        for result in self.auxiliary:
            valid = valid & result.is_valid
        return valid

    @property
    def total(self) -> jax.Array:
        def weighted(result):
            terms = (result.classification, result.mask, result.dice)
            return sum(
                w * term.value
                for w, term in zip(self.config.loss_weights, terms, strict=True)
            )

        total = weighted(self.final)
        weights = _auxiliary_weights(self.config, len(self.auxiliary))
        for weight, result in zip(weights, self.auxiliary, strict=True):
            total += weight * weighted(result)
        return jnp.where(self.is_valid, total, jnp.nan)


def _auxiliary_weights(config: QueryLossConfig, count: int) -> tuple[float, ...]:
    if config.auxiliary_weights is None:
        return (1.0,) * count
    if config.auxiliary_weights == ():
        return ()
    if len(config.auxiliary_weights) != count:
        raise ValueError("Expected one weight per auxiliary prediction.")
    return config.auxiliary_weights


def _inputs(prediction: QueryMaskPrediction, targets: QueryTargets):
    classes, masks = prediction.class_logits, prediction.mask_logits
    if (
        classes.ndim != 2
        or masks.ndim != 3
        or classes.shape[0] != masks.shape[0]
        or min(masks.shape) <= 0
        or classes.shape[1] < 2
    ):
        raise ValueError("Expected nonempty (Q,C+1) class and (Q,H,W) mask logits.")
    if targets.masks.shape[0] > classes.shape[0]:
        raise ValueError("Target capacity must not exceed query count.")
    supervised = targets.supervised
    active = targets.target_valid & supervised
    pixels = targets.spatial_valid & supervised
    known = active[:, None, None] & pixels
    labels_ok = (targets.class_ids >= 0) & (targets.class_ids < classes.shape[1] - 1)
    masks_ok = jnp.isfinite(targets.masks) & (targets.masks >= 0) & (targets.masks <= 1)
    is_valid = jnp.all(~active | labels_ok) & jnp.all(~known | masks_ok)
    is_valid &= ~supervised | (
        jnp.all(jnp.isfinite(classes)) & jnp.all(jnp.isfinite(masks))
    )
    labels = jnp.where(active & labels_ok, targets.class_ids, 0)
    truth = jnp.where(known & masks_ok, targets.masks, 0).astype(jnp.float32)
    classes = jnp.where(supervised, classes, 0).astype(jnp.float32)
    masks = jnp.where(supervised, masks, 0).astype(jnp.float32)
    return classes, masks, labels, truth, active, pixels, is_valid


def query_matching_costs(
    prediction: QueryMaskPrediction,
    targets: QueryTargets,
    points: PointSamples,
    *,
    config: QueryLossConfig = QueryLossConfig(),
) -> MatchingCosts:
    """Compute class, sampled BCE, and Dice costs on shared points.

    Class cost is negative probability, not negative log probability. BCE is
    averaged over valid sampled points. Dice has additive smoothing of one in
    numerator and denominator. Query softmax includes no-object. Ignored target
    entries never enter interpolation, costs, or normalizers.
    """
    classes, masks, labels, truth, active, pixels, is_valid = _inputs(
        prediction, targets
    )
    pred, pred_valid = sample_mask_points(masks, points)
    target, point_valid = sample_mask_points(truth, points, spatial_valid=pixels)
    point_valid &= pred_valid
    count = jnp.sum(point_valid)
    pred = jnp.where(point_valid, pred, 0)
    target = jnp.where(point_valid, target, 0)
    positive = jnp.where(point_valid, jax.nn.softplus(-pred), 0)
    negative = jnp.where(point_valid, jax.nn.softplus(pred), 0)
    binary = (
        positive @ target.T + negative @ (jnp.where(point_valid, 1 - target, 0)).T
    ) / jnp.maximum(count, 1)
    probabilities = jnp.where(point_valid, jax.nn.sigmoid(pred), 0)
    dice = 1 - (2 * (probabilities @ target.T) + 1) / (
        probabilities.sum(-1)[:, None] + target.sum(-1)[None, :] + 1
    )
    classification = -jax.nn.softmax(classes, axis=-1)[:, labels]
    components = tuple(
        jnp.where(active[None, :], value, 0) for value in (classification, binary, dice)
    )
    is_valid &= ~jnp.any(active) | (count > 0)
    total = sum(
        weight * value
        for weight, value in zip(config.cost_weights, components, strict=True)
    )
    is_valid &= jnp.all(jnp.where(active[None, :], jnp.isfinite(total), True))
    total = jnp.where(active[None, :] & ~is_valid, jnp.nan, total)
    return MatchingCosts(*components, total, is_valid)


def prepare_query_loss(
    output: QuerySegmentationOutput,
    targets: QueryTargets,
    *,
    key: jax.Array,
    config: QueryLossConfig = QueryLossConfig(),
) -> QueryLossContext:
    """Prepare nondifferentiable matches and points for one stochastic forward.

    May run inside JIT and differentiation: costs/decisions are stopped, while
    subsequent loss evaluation differentiates the original model outputs.
    Keys are folded by prediction index (final=0, auxiliary=i+1), then split
    for matching and loss sampling. Disabling an auxiliary does not alter the
    final prediction's random stream.
    """
    weights = _auxiliary_weights(config, len(output.auxiliary))
    pixels = targets.spatial_valid & targets.supervised

    def prepare(prediction, index, reuse=None):
        matching_key, loss_key = jr.split(jr.fold_in(key, index))
        if reuse is None:
            points = sample_query_points(
                pixels, key=matching_key, num_points=config.matching_num_points
            )
            costs = query_matching_costs(prediction, targets, points, config=config)
            assignment = match_queries(
                costs.total, targets.target_valid & targets.supervised
            )
            assignment = eqx.tree_at(
                lambda a: a.is_valid, assignment, assignment.is_valid & costs.is_valid
            )
        else:
            assignment = reuse.assignment
            points = reuse.matching_points
        indices = jnp.maximum(assignment.query_indices, 0)
        capacity = indices.shape[0]
        if config.sampling_policy == "uncertainty" and capacity > 0:
            # Unsupervised examples have no valid pixels, so their logits are unused.
            mask_points = sample_query_points(
                pixels,
                key=loss_key,
                num_points=config.loss_num_points,
                mask_logits=prediction.mask_logits[indices],
                oversample_ratio=config.oversample_ratio,
                importance_sample_ratio=config.importance_sample_ratio,
            )
        else:
            shared = sample_query_points(
                pixels, key=loss_key, num_points=config.loss_num_points
            )
            mask_points = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (capacity, *x.shape)), shared
            )
        mask_points = PointSamples(
            mask_points.coordinates, mask_points.valid & assignment.valid[:, None]
        )
        return PredictionLossContext(assignment, points, mask_points)

    final = prepare(output.final, 0)
    reuse = final if config.auxiliary_matching == "reuse_final" else None
    # An empty weight tuple disables auxiliary losses.
    auxiliary = tuple(
        prepare(prediction, index + 1, reuse)
        for index, prediction in enumerate(output.auxiliary if weights else ())
    )
    return QueryLossContext(final, auxiliary, config)


def query_prediction_loss(
    prediction: QueryMaskPrediction,
    targets: QueryTargets,
    context: PredictionLossContext,
    *,
    config: QueryLossConfig = QueryLossConfig(),
) -> PredictionLoss:
    """Evaluate classification and matched BCE/Dice for a single prediction.

    The context supplies target-ordered matches and per-target loss points.
    Returns separate additive numerators and denominators, without coefficients
    or auxiliary-output composition. See :func:`query_segmentation_loss` for
    the reduction definitions and :class:`QueryLossResult` for failure signals.
    """
    classes, masks, labels, truth, active, pixels, is_valid = _inputs(
        prediction, targets
    )
    assignment = context.assignment
    capacity = targets.class_ids.shape[0]
    if not assignment.query_indices.shape == assignment.valid.shape == (capacity,):
        raise ValueError("Assignment shapes disagree with target capacity.")
    if context.mask_points.coordinates.shape != (
        capacity,
        config.loss_num_points,
        2,
    ) or context.mask_points.valid.shape != (capacity, config.loss_num_points):
        raise ValueError(
            "Mask point shapes disagree with target capacity or configuration."
        )
    indices = jnp.clip(assignment.query_indices, 0, classes.shape[0] - 1)
    matched = active & assignment.valid
    is_valid &= assignment.is_valid & jnp.all(assignment.valid == active)
    is_valid &= jnp.all(
        ~matched
        | (
            (assignment.query_indices >= 0)
            & (assignment.query_indices < classes.shape[0])
        )
    )
    # This also rejects duplicated active query indices in externally supplied contexts.
    counts = (
        jnp.zeros((classes.shape[0],), dtype=jnp.int32)
        .at[indices]
        .add(matched.astype(jnp.int32))
    )
    is_valid &= jnp.all(counts <= 1)
    no_object = classes.shape[1] - 1
    assigned_labels = jnp.full((classes.shape[0],), no_object, dtype=jnp.int32)
    destinations = jnp.where(matched, indices, classes.shape[0])
    assigned_labels = assigned_labels.at[destinations].set(labels, mode="drop")
    weights = (
        jnp.where(assigned_labels == no_object, config.no_object_weight, 1.0)
        * targets.supervised
    )
    nll = -jnp.take_along_axis(
        jax.nn.log_softmax(classes), assigned_labels[:, None], axis=-1
    )[:, 0]
    classification = LossTerm(
        jnp.sum(jnp.where(weights > 0, nll, 0) * weights), jnp.sum(weights)
    )

    def pair_loss(prediction_mask, target_mask, points, valid):
        prediction_values, prediction_valid = sample_mask_points(
            prediction_mask, points
        )
        target_values, target_valid = sample_mask_points(
            target_mask, points, spatial_valid=pixels
        )
        keep = prediction_valid & target_valid & valid
        count = jnp.sum(keep)
        prediction_values = jnp.where(keep, prediction_values, 0)
        target_values = jnp.where(keep, target_values, 0)
        bce = target_values * jax.nn.softplus(-prediction_values) + (
            1 - target_values
        ) * jax.nn.softplus(prediction_values)
        bce = jnp.sum(jnp.where(keep, bce, 0)) / jnp.maximum(count, 1)
        probability = jnp.where(keep, jax.nn.sigmoid(prediction_values), 0)
        dice = 1 - (2 * jnp.sum(probability * target_values) + 1) / (
            jnp.sum(probability) + jnp.sum(target_values) + 1
        )
        return bce, dice, count

    if capacity:
        bce, dice, point_counts = jax.vmap(pair_loss)(
            masks[indices], truth, context.mask_points, matched
        )
        is_valid &= jnp.all(~matched | (point_counts > 0))
        bce_sum, dice_sum = (
            jnp.sum(jnp.where(matched, bce, 0)),
            jnp.sum(jnp.where(matched, dice, 0)),
        )
    else:
        bce_sum, dice_sum = jnp.asarray(0.0), jnp.asarray(0.0)
    count = jnp.sum(matched).astype(jnp.float32)
    return PredictionLoss(
        classification,
        LossTerm(bce_sum, count),
        LossTerm(dice_sum, count),
        targets.supervised.astype(jnp.int32),
        count,
        is_valid,
    )


def query_segmentation_loss(
    output: QuerySegmentationOutput,
    targets: QueryTargets,
    context: QueryLossContext,
) -> QueryLossResult:
    """Evaluate differentiable losses using retained assignments and points.

    Classification divides weighted NLL by the sum of query weights. Mask BCE
    first averages valid points per mask, then averages matched masks. Dice
    averages matched masks. Empty denominators yield zero, remaining exposed as
    zero for correct aggregation and caller-controlled update skipping.
    """
    weights = _auxiliary_weights(context.config, len(output.auxiliary))
    if len(context.auxiliary) != len(weights):
        raise ValueError("Context and output auxiliary lengths disagree.")
    final = query_prediction_loss(
        output.final, targets, context.final, config=context.config
    )
    auxiliary = tuple(
        query_prediction_loss(prediction, targets, prepared, config=context.config)
        for prediction, prepared in zip(
            output.auxiliary if weights else (), context.auxiliary, strict=True
        )
    )
    return QueryLossResult(final, auxiliary, context.config)


def reduce_query_losses(
    result: QueryLossResult,
    *,
    axis: int | tuple[int, ...] | None = 0,
    axis_name: str | tuple[str, ...] | None = None,
) -> QueryLossResult:
    """Sum sufficient statistics locally and optionally across named devices.

    Call on a vmapped result. ``axis=None`` performs no local reduction (useful
    inside a named map); it does not mean reduce all dimensions. Boolean input
    validity is AND-reduced. Coefficients are applied only by ``result.total``.
    Different auxiliary outputs retain separate denominators.
    """

    def reduce(value):
        boolean = value.dtype == jnp.bool_
        if axis is not None:
            value = jnp.all(value, axis=axis) if boolean else jnp.sum(value, axis=axis)
        if axis_name is not None:
            value = (
                jax.lax.pmin(value.astype(jnp.int32), axis_name).astype(jnp.bool_)
                if boolean
                else jax.lax.psum(value, axis_name)
            )
        return value

    return jax.tree_util.tree_map(reduce, result)
