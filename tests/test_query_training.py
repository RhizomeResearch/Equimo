"""Independent mathematics and transformation checks for query training."""

from itertools import permutations
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from equimo.vision.models.eomt import EoMTOutput, MaskPrediction
from equimo.vision.query_training import (
    PointSamples,
    PredictionLossContext,
    QueryLossConfig,
    QueryLossContext,
    QueryTargets,
    match_queries,
    prepare_query_loss,
    query_matching_costs,
    query_segmentation_loss,
    reduce_query_losses,
    sample_mask_points,
    sample_query_points,
)


def _points():
    return PointSamples(
        jnp.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
        jnp.ones(4, bool),
    )


def _case():
    prediction = MaskPrediction(
        jnp.log(jnp.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])),
        jnp.array(
            [
                [[2.0, -1.0], [-2.0, 1.0]],
                [[-1.0, 2.0], [1.0, -2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )
    targets = QueryTargets(
        jnp.array([0, 1]),
        jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]),
        jnp.ones(2, bool),
        jnp.ones((2, 2), bool),
    )
    return EoMTOutput(prediction, ()), targets


def _fixed_context(output, targets, config=None):
    config = (
        QueryLossConfig(matching_num_points=4, loss_num_points=4)
        if config is None
        else config
    )
    points = _points()
    costs = query_matching_costs(output.final, targets, points, config=config)
    assignment = match_queries(costs.total, targets.target_valid & targets.supervised)
    count = targets.class_ids.shape[0]
    mask_points = PointSamples(
        jnp.broadcast_to(points.coordinates, (count, 4, 2)),
        jnp.broadcast_to(points.valid, (count, 4)),
    )
    prepared = PredictionLossContext(assignment, points, mask_points)
    return QueryLossContext(prepared, (), config)


@pytest.mark.parametrize("targets,queries", [(0, 3), (1, 3), (3, 3), (3, 5)])
def test_assignment_exhaustive_and_ties(targets, queries):
    rng = np.random.default_rng(12)
    matrices = rng.integers(-4, 5, (8, queries, targets)).astype(np.float32)
    validity = rng.random((8, targets)) > 0.3
    solve = jax.jit(jax.vmap(match_queries))
    result = solve(jnp.array(matrices), jnp.array(validity))
    for cost, active, indices in zip(
        matrices, validity, np.asarray(result.query_indices), strict=True
    ):
        selected = np.flatnonzero(active)
        optimum = min(
            sum(cost[q, t] for q, t in zip(perm, selected, strict=True))
            for perm in permutations(range(queries), len(selected))
        )
        assert (
            sum(cost[q, t] for q, t in zip(indices[active], selected, strict=True))
            == optimum
        )
        assert len(set(indices[active])) == len(selected)
        np.testing.assert_array_equal(indices[~active], -1)
    tied = jax.jit(match_queries)(
        jnp.zeros((queries, targets)), jnp.ones(targets, bool)
    )
    np.testing.assert_array_equal(tied.query_indices, np.arange(targets))
    assert bool(jnp.all(result.is_valid))


def test_assignment_scipy_and_invalid_values():
    costs = np.random.default_rng(7).normal(size=(100, 30)).astype(np.float32)
    valid = np.arange(30) % 3 != 0
    result = jax.jit(match_queries)(jnp.array(costs), jnp.array(valid))
    rows, cols = linear_sum_assignment(costs[:, valid])
    np.testing.assert_allclose(
        costs[np.asarray(result.query_indices)[valid], np.flatnonzero(valid)].sum(),
        costs[:, valid][rows, cols].sum(),
        rtol=1e-6,
    )
    padded = np.where(valid, costs, np.nan)
    np.testing.assert_array_equal(
        match_queries(jnp.array(padded), jnp.array(valid)).query_indices,
        result.query_indices,
    )
    padded[0, 1] = np.nan
    rejected = jax.jit(match_queries)(jnp.array(padded), jnp.array(valid))
    assert not bool(rejected.is_valid)
    assert not bool(jnp.any(rejected.valid))
    with pytest.raises(ValueError, match="capacity"):
        match_queries(jnp.zeros((1, 2)), jnp.ones(2, bool))
    overflow = jnp.array([[3e38, -3e38], [-3e38, 3e38]])
    rejected = jax.jit(match_queries)(overflow, jnp.ones(2, bool))
    assert not bool(rejected.is_valid)
    np.testing.assert_array_equal(rejected.query_indices, -1)


def test_costs_and_loss_hand_calculation():
    output, targets = _case()
    config = QueryLossConfig(matching_num_points=4, loss_num_points=4)
    costs = query_matching_costs(output.final, targets, _points(), config=config)
    logits = np.asarray(output.final.mask_logits).reshape(3, 4)
    truth = np.asarray(targets.masks).reshape(2, 4)
    probabilities = 1 / (1 + np.exp(-logits))
    expected_bce = np.empty((3, 2))
    expected_dice = np.empty((3, 2))
    for query in range(3):
        for target in range(2):
            expected_bce[query, target] = np.mean(
                np.logaddexp(0, logits[query]) - truth[target] * logits[query]
            )
            expected_dice[query, target] = 1 - (
                2 * np.dot(probabilities[query], truth[target]) + 1
            ) / (probabilities[query].sum() + truth[target].sum() + 1)
    expected_class = -np.array([[0.7, 0.2], [0.1, 0.6], [0.2, 0.2]])
    np.testing.assert_allclose(costs.classification, expected_class, atol=1e-7)
    np.testing.assert_allclose(costs.mask, expected_bce, atol=1e-7)
    np.testing.assert_allclose(costs.dice, expected_dice, atol=1e-7)
    np.testing.assert_allclose(
        costs.total,
        2 * expected_class + 5 * expected_bce + 5 * expected_dice,
        atol=1e-6,
    )
    context = _fixed_context(output, targets)
    np.testing.assert_array_equal(context.final.assignment.query_indices, [0, 1])
    result = query_segmentation_loss(output, targets, context)
    expected_ce = -np.log(0.7) - np.log(0.6) - 0.1 * np.log(0.6)
    np.testing.assert_allclose(
        result.final.classification.numerator, expected_ce, atol=1e-7
    )
    np.testing.assert_allclose(result.final.classification.denominator, 2.1)
    np.testing.assert_allclose(
        result.final.mask.numerator, expected_bce[0, 0] + expected_bce[1, 1], atol=1e-7
    )
    assert float(result.final.mask.denominator) == 2
    np.testing.assert_allclose(
        result.total,
        2 * expected_ce / 2.1
        + 5 * (expected_bce[0, 0] + expected_bce[1, 1]) / 2
        + 5 * (expected_dice[0, 0] + expected_dice[1, 1]) / 2,
        atol=1e-6,
    )


def test_validity_normalized_interpolation_and_gradient():
    mask = jnp.array([[1.0, jnp.nan], [0.0, jnp.nan]])
    valid = jnp.array([[True, False], [True, False]])
    points = PointSamples(
        jnp.array([[0.5, 0.5], [0.0, 0.0], [0.75, 0.5], [0.25, 0.5]]), jnp.ones(4, bool)
    )
    values, kept = sample_mask_points(mask, points, spatial_valid=valid)
    np.testing.assert_allclose(values, [0.5, 1.0, 0.0, 0.5])
    np.testing.assert_array_equal(kept, [True, True, False, True])
    gradient = jax.jit(
        jax.grad(lambda x: sample_mask_points(x, points, spatial_valid=valid)[0].sum())
    )(mask)
    np.testing.assert_allclose(gradient, [[2.0, 0.0], [1.0, 0.0]])
    changed = jnp.where(valid, mask, 1e30)
    np.testing.assert_array_equal(
        sample_mask_points(changed, points, spatial_valid=valid)[0], values
    )
    logits, _ = sample_mask_points(jnp.array([[1.0, 3.0], [5.0, 7.0]]), points)
    np.testing.assert_allclose(logits, [4.0, 1.0, 5.0, 3.0])


@pytest.mark.parametrize("uncertainty", [False, True])
def test_sampling_sparse_empty_and_replay(uncertainty):
    valid = jnp.zeros((3, 4), bool).at[1, 2].set(True)
    kwargs = dict(key=jr.key(9), num_points=20)
    if uncertainty:
        kwargs["mask_logits"] = jnp.zeros((2, 2, 2))
    points = jax.jit(lambda v: sample_query_points(v, **kwargs))(valid)
    replay = sample_query_points(valid, **kwargs)
    np.testing.assert_allclose(points.coordinates, replay.coordinates, atol=1e-7)
    assert bool(jnp.all(points.valid))
    assert bool(
        jnp.all(
            (points.coordinates[..., 0] >= 0.5) & (points.coordinates[..., 0] < 0.75)
        )
    )
    assert bool(
        jnp.all(
            (points.coordinates[..., 1] >= 1 / 3) & (points.coordinates[..., 1] < 2 / 3)
        )
    )
    assert not bool(jnp.any(sample_query_points(jnp.zeros_like(valid), **kwargs).valid))


@pytest.mark.parametrize("kind", ["negative", "ignored", "padded"])
def test_empty_targets_and_no_object_are_distinct(kind):
    output, _ = _case()
    targets = QueryTargets(
        jnp.zeros(0, jnp.int32),
        jnp.zeros((0, 2, 2)),
        jnp.zeros(0, bool),
        jnp.full((2, 2), kind != "ignored"),
        example_valid=kind != "padded",
    )
    config = QueryLossConfig(matching_num_points=4, loss_num_points=4)

    def loss(logits):
        candidate = eqx.tree_at(lambda o: o.final.class_logits, output, logits)
        context = prepare_query_loss(candidate, targets, key=jr.key(5), config=config)
        result = query_segmentation_loss(candidate, targets, context)
        return result.total, result

    (value, result), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(
        output.final.class_logits
    )
    assert bool(result.is_valid)
    assert float(result.final.mask.denominator) == 0
    if kind == "negative":
        np.testing.assert_allclose(
            value, -2 * np.log([0.1, 0.3, 0.6]).mean(), rtol=1e-6
        )
        assert bool(jnp.any(gradient != 0))
    else:
        assert float(value) == 0
        assert int(result.final.supervised_examples) == 0
        assert float(result.final.classification.denominator) == 0
        np.testing.assert_array_equal(gradient, 0)


def test_ignored_values_cannot_change_assignment_loss_or_gradient():
    output, targets = _case()
    targets = eqx.tree_at(
        lambda t: (t.spatial_valid, t.target_valid),
        targets,
        (jnp.array([[True, False], [True, False]]), jnp.array([True, False])),
    )
    config = QueryLossConfig(matching_num_points=7, loss_num_points=5)

    def evaluate(targets):
        def loss(masks):
            candidate = eqx.tree_at(lambda o: o.final.mask_logits, output, masks)
            context = prepare_query_loss(
                candidate, targets, key=jr.key(15), config=config
            )
            return query_segmentation_loss(
                candidate, targets, context
            ).total, context.final.assignment.query_indices

        return jax.jit(jax.value_and_grad(loss, has_aux=True))(output.final.mask_logits)

    altered = eqx.tree_at(
        lambda t: (t.masks, t.class_ids),
        targets,
        (
            jnp.where(
                targets.target_valid[:, None, None] & targets.spatial_valid,
                targets.masks,
                jnp.nan,
            ),
            jnp.array([0, -999]),
        ),
    )
    original = evaluate(targets)
    changed = evaluate(altered)
    for left, right in zip(
        jax.tree.leaves(original), jax.tree.leaves(changed), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert bool(jnp.all(jnp.isfinite(original[1])))


def _with_reversed_auxiliary():
    output, targets = _case()
    auxiliary = MaskPrediction(
        output.final.class_logits[::-1], output.final.mask_logits[::-1]
    )
    return EoMTOutput(output.final, (auxiliary,)), targets


def test_auxiliary_key_folding_and_matching_policies():
    output, targets = _with_reversed_auxiliary()
    config = QueryLossConfig(
        matching_num_points=4, loss_num_points=4, auxiliary_weights=(0.5,)
    )
    context = prepare_query_loss(output, targets, key=jr.key(4), config=config)
    final_only = QueryLossConfig(
        matching_num_points=4, loss_num_points=4, auxiliary_weights=()
    )
    final_context = prepare_query_loss(
        output, targets, key=jr.key(4), config=final_only
    )
    np.testing.assert_array_equal(
        context.final.mask_points.coordinates,
        final_context.final.mask_points.coordinates,
    )
    assert len(final_context.auxiliary) == 0
    reused = prepare_query_loss(
        output,
        targets,
        key=jr.key(4),
        config=QueryLossConfig(
            matching_num_points=4, loss_num_points=4, auxiliary_matching="reuse_final"
        ),
    )
    np.testing.assert_array_equal(
        reused.final.assignment.query_indices,
        reused.auxiliary[0].assignment.query_indices,
    )
    assert not np.array_equal(
        context.final.assignment.query_indices,
        context.auxiliary[0].assignment.query_indices,
    )


def test_bfloat16_predictions_give_float32_loss_and_finite_gradients():
    output, targets = _with_reversed_auxiliary()
    config = QueryLossConfig(
        matching_num_points=4, loss_num_points=4, auxiliary_weights=(0.5,)
    )
    context = prepare_query_loss(output, targets, key=jr.key(4), config=config)
    low = jax.tree.map(lambda x: x.astype(jnp.bfloat16), output)
    value, gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(
            lambda o: query_segmentation_loss(o, targets, context).total
        )
    )(low)
    assert value.dtype == jnp.float32
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gradient))


def test_invalid_active_inputs_signal_failure_without_callback():
    output, targets = _case()
    config = QueryLossConfig(matching_num_points=4, loss_num_points=4)
    targets = eqx.tree_at(lambda t: t.class_ids, targets, jnp.array([9, 1]))

    def evaluate(classes):
        candidate = eqx.tree_at(lambda o: o.final.class_logits, output, classes)
        context = prepare_query_loss(candidate, targets, key=jr.key(0), config=config)
        return query_segmentation_loss(candidate, targets, context)

    result = jax.jit(evaluate)(output.final.class_logits)
    assert not bool(result.is_valid)
    assert bool(jnp.isnan(result.total))
    jaxpr = str(jax.make_jaxpr(evaluate)(output.final.class_logits))
    assert "callback" not in jaxpr


def test_pinned_official_loss_gradient_and_assignment_reference():
    reference = json.loads(
        (Path(__file__).parent / "data/query_training_reference.json").read_text()
    )
    prediction = MaskPrediction(
        jnp.array(reference["class_logits"]), jnp.array(reference["mask_logits"])
    )
    targets = QueryTargets(
        jnp.array([0, 1]),
        jnp.array(reference["target_masks"]),
        jnp.ones(2, bool),
        jnp.ones((4, 6), bool),
    )
    points = PointSamples(jnp.array(reference["points"]), jnp.ones(7, bool))
    config = QueryLossConfig(matching_num_points=7, loss_num_points=7)
    costs = query_matching_costs(prediction, targets, points, config=config)
    for actual, name in (
        (costs.classification, "class_cost"),
        (costs.mask, "mask_cost"),
        (costs.dice, "dice_cost"),
    ):
        np.testing.assert_allclose(actual, reference[name], atol=2e-7, rtol=2e-6)
    assignment = match_queries(costs.total, targets.target_valid)
    np.testing.assert_array_equal(assignment.query_indices, reference["query_indices"])
    context = QueryLossContext(
        PredictionLossContext(
            assignment,
            points,
            PointSamples(
                jnp.broadcast_to(points.coordinates, (2, 7, 2)), jnp.ones((2, 7), bool)
            ),
        ),
        (),
        config,
    )

    def loss(prediction):
        result = query_segmentation_loss(EoMTOutput(prediction, ()), targets, context)
        return result.total, result

    (value, result), gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(loss, has_aux=True)
    )(prediction)
    for actual, name in (
        (result.final.classification.value, "class_loss"),
        (result.final.mask.value, "mask_loss"),
        (result.final.dice.value, "dice_loss"),
        (value, "total"),
        (gradient.class_logits, "class_gradient"),
        (gradient.mask_logits, "mask_gradient"),
    ):
        np.testing.assert_allclose(actual, reference[name], atol=2e-7, rtol=3e-6)
    assignment = match_queries(
        jnp.array(reference["assignment_costs"]), jnp.ones(3, bool)
    )
    np.testing.assert_array_equal(
        assignment.query_indices, reference["optax_query_indices"]
    )


def test_uncertainty_sampling_concentrates_near_zero_logits():
    valid = jnp.ones((4, 4), bool)
    logits = jnp.broadcast_to(jnp.linspace(-6.0, 6.0, 4), (1, 4, 4))
    uniform = sample_query_points(valid, key=jr.key(36), num_points=100)
    important = sample_query_points(
        valid,
        key=jr.key(36),
        num_points=100,
        mask_logits=logits,
        oversample_ratio=4,
        importance_sample_ratio=1.0,
    )
    chosen = PointSamples(important.coordinates[0], important.valid[0])
    ordinary_values, _ = sample_mask_points(logits[0], uniform)
    selected_values, _ = sample_mask_points(logits[0], chosen)
    assert float(jnp.mean(jnp.abs(selected_values))) < 0.4 * float(
        jnp.mean(jnp.abs(ordinary_values))
    )


def test_aggregation_uses_distinct_weighted_denominators():
    output, targets = _case()
    config = QueryLossConfig(matching_num_points=4, loss_num_points=4)

    def evaluate(valid):
        target = eqx.tree_at(lambda t: t.target_valid, targets, valid)
        return query_segmentation_loss(
            output, target, _fixed_context(output, target, config)
        )

    batch = jax.vmap(evaluate)(jnp.array([[True, True], [True, False], [False, False]]))
    reduced = reduce_query_losses(batch)
    ce = (
        np.asarray(batch.final.classification.numerator).sum()
        / np.asarray(batch.final.classification.denominator).sum()
    )
    bce = np.asarray(batch.final.mask.numerator).sum() / 3
    dice = np.asarray(batch.final.dice.numerator).sum() / 3
    np.testing.assert_allclose(reduced.total, 2 * ce + 5 * bce + 5 * dice, rtol=1e-6)
    assert not np.isclose(float(reduced.total), float(jnp.mean(batch.total)))
    named = jax.vmap(
        lambda result: reduce_query_losses(result, axis=None, axis_name="batch"),
        axis_name="batch",
    )(batch)
    np.testing.assert_allclose(named.total, reduced.total, rtol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cost_weights": (0.0, 0.0, 0.0)},
        {"loss_weights": (1.0, -1.0, 1.0)},
        {"no_object_weight": float("nan")},
        {"matching_num_points": 0},
        {"loss_num_points": True},
        {"oversample_ratio": 0.5},
        {"importance_sample_ratio": float("nan")},
        {"sampling_policy": "invalid"},
        {"auxiliary_matching": "invalid"},
        {"auxiliary_weights": (-1.0,)},
    ],
)
def test_invalid_static_configuration(kwargs):
    with pytest.raises(ValueError):
        QueryLossConfig(**kwargs)


def test_no_object_zero_weight_and_wholly_ignored_nonfinite_padding():
    output, targets = _case()
    config = QueryLossConfig(
        matching_num_points=4, loss_num_points=4, no_object_weight=0.0
    )
    empty = eqx.tree_at(lambda t: t.target_valid, targets, jnp.zeros(2, bool))
    context = prepare_query_loss(output, empty, key=jr.key(3), config=config)
    result = query_segmentation_loss(output, empty, context)
    assert float(result.total) == 0
    assert float(result.final.classification.denominator) == 0
    ignored = eqx.tree_at(lambda t: t.example_valid, targets, jnp.array(False))
    bad = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), output)

    def evaluate(output):
        context = prepare_query_loss(output, ignored, key=jr.key(1), config=config)
        return query_segmentation_loss(output, ignored, context).total

    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(evaluate))(bad)
    assert float(value) == 0
    for leaf in jax.tree.leaves(gradient):
        np.testing.assert_array_equal(leaf, 0)
