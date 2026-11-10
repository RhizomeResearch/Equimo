"""Numerical contracts for probability operations and classification losses."""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equimo.core.ops import (
    inverse_sigmoid,
    log_softmax,
    softmax,
    softmax_cross_entropy_with_integer_labels,
    softmax_cross_entropy_with_logit_targets,
)


def _probabilities(logits):
    values = [Decimal.from_float(float(x)) for x in logits]
    maximum = max(values)
    weights = [(x - maximum).exp() for x in values]
    total = sum(weights)
    return [x / total for x in weights]


@pytest.mark.parametrize(
    "dtype,margin,rtol",
    [
        (jnp.bfloat16, 30, 0.02),
        (jnp.float16, 10, 0.002),
        (jnp.float32, 30, 2e-6),
        (jnp.float64, 50, 1e-13),
    ],
)
def test_confident_predictions_preserve_loss_and_gradients(dtype, margin, rtol):
    with jax.enable_x64(True), localcontext() as context:
        context.prec = 80
        x = jnp.array([margin, 0, 0], dtype=dtype)
        working = jnp.promote_types(dtype, jnp.float32)
        np.testing.assert_array_equal(
            softmax(x), jax.nn.softmax(x.astype(working)).astype(dtype)
        )
        probabilities = _probabilities(np.asarray(x, dtype=np.float64))
        expected_loss = float(-probabilities[0].ln())
        expected_grad = np.array(
            [float(p - (i == 0)) for i, p in enumerate(probabilities)]
        )
        functions = (
            (lambda z: -log_softmax(z)[0], expected_loss, expected_grad),
            (
                lambda z: softmax_cross_entropy_with_integer_labels(z, jnp.array(0)),
                expected_loss,
                expected_grad,
            ),
            (
                lambda z: softmax(z)[0],
                float(probabilities[0]),
                -float(probabilities[0]) * expected_grad,
            ),
        )
        for fn, expected_value, expected_gradient in functions:
            for evaluate in (jax.value_and_grad(fn), jax.jit(jax.value_and_grad(fn))):
                value, gradient = evaluate(x)
                assert value.dtype == gradient.dtype == dtype
                np.testing.assert_allclose(
                    np.asarray(value, float), expected_value, rtol=rtol, atol=0
                )
                np.testing.assert_allclose(
                    np.asarray(gradient, float), expected_gradient, rtol=rtol, atol=0
                )
                assert abs(np.asarray(gradient, float).sum()) <= rtol * abs(
                    expected_gradient[0]
                )


@pytest.mark.parametrize(
    "fn, reference, axis",
    [
        *((softmax, jax.nn.softmax, axis) for axis in (-1, 0, (0, 2), None)),
        *((log_softmax, jax.nn.log_softmax, axis) for axis in (-1, 1)),
    ],
)
def test_softmax_transformations_and_masks(fn, reference, axis):
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 7
    mask = jnp.arange(24).reshape(x.shape) % 3 != 0
    call = lambda z: fn(z, axis=axis, where=mask)
    expected = lambda z: reference(z, axis=axis, where=mask)
    np.testing.assert_allclose(jax.jit(call)(x), expected(x), rtol=2e-6, atol=1e-6)
    forward = jax.jacfwd(call)(x)
    reverse = jax.jacrev(call)(x)
    np.testing.assert_allclose(forward, jax.jacfwd(expected)(x), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(reverse, forward, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        jax.vmap(call)(jnp.stack([x, x + 1]))[0], call(x), rtol=1e-6
    )


def test_singleton_tied_and_masked_distributions():
    for x in (jnp.array([3.0]), jnp.array([3.0, -jnp.inf, -jnp.inf]), jnp.zeros(3)):
        for fn in (softmax, log_softmax):
            np.testing.assert_allclose(
                fn(x), getattr(jax.nn, fn.__name__)(x), atol=1e-7
            )
            gradient = jax.jit(jax.grad(lambda z: fn(z)[0]))(x)
            assert jnp.all(jnp.isfinite(gradient))
            np.testing.assert_allclose(gradient.sum(), 0, atol=1e-7)
        loss, gradient = jax.jit(
            jax.value_and_grad(softmax_cross_entropy_with_integer_labels)
        )(x, jnp.array(0))
        assert jnp.isfinite(loss) & jnp.all(jnp.isfinite(gradient))
    mask = jnp.zeros(3, dtype=bool)
    for fn in (softmax, log_softmax):
        gradient = jax.grad(lambda z: fn(z, where=mask).sum())(jnp.ones(3))
        np.testing.assert_array_equal(gradient, 0)
    np.testing.assert_array_equal(softmax(jnp.ones(3), where=mask), 0)
    np.testing.assert_array_equal(log_softmax(jnp.ones(3), where=mask), -jnp.inf)


@pytest.mark.parametrize(
    "dtype,rtol", [(jnp.bfloat16, 0.02), (jnp.float32, 2e-5), (jnp.float64, 2e-5)]
)
@pytest.mark.parametrize("teacher_values", [(2.0, 1.0, 0.0), (30.0, 0.0, 0.0)])
def test_distillation_gradients_from_equal_nearby_and_distant_logits(
    dtype, rtol, teacher_values
):
    with jax.enable_x64(True), localcontext() as context:
        context.prec = 80
        teacher = jnp.array(teacher_values, dtype=dtype)
        nearby = teacher.at[0].set(jnp.nextafter(teacher[0], jnp.asarray(0, dtype)))
        students = jnp.stack(
            [
                teacher,
                teacher + 8,
                nearby,
                teacher + jnp.array([0.0, 0.5, 0.0], dtype=dtype),
                teacher + jnp.array([0.0, 1.0, 0.0], dtype=dtype),
                jnp.array([-1000.0, 1000.0, 0.0], dtype=dtype),
            ]
        )
        evaluate = jax.vmap(
            jax.value_and_grad(softmax_cross_entropy_with_logit_targets),
            in_axes=(0, None),
        )
        p = _probabilities(np.asarray(teacher, dtype=np.float64))
        expected_gradients = []
        expected_losses = []
        for student in np.asarray(students, dtype=np.float64):
            q = _probabilities(student)
            expected_gradients.append(
                [float(qi - pi) for qi, pi in zip(q, p, strict=True)]
            )
            expected_losses.append(
                float(-sum(pi * qi.ln() for qi, pi in zip(q, p, strict=True)))
            )
        for call in (evaluate, jax.jit(evaluate)):
            loss, gradient = call(students, teacher)
            assert loss.dtype == gradient.dtype == dtype
            np.testing.assert_array_equal(gradient[:2], 0)
            np.testing.assert_allclose(
                np.asarray(loss, float), expected_losses, rtol=rtol, atol=0
            )
            np.testing.assert_allclose(
                np.asarray(gradient, float), expected_gradients, rtol=rtol, atol=0
            )
        np.testing.assert_array_equal(
            jax.grad(softmax_cross_entropy_with_logit_targets, argnums=1)(
                nearby, teacher
            ),
            0,
        )


def test_integer_labels_outside_classes_give_nan_with_maskable_gradients():
    logits = jnp.array([[2.0, 1.0, 0.0]] * 3)
    labels = jnp.array([-1, 3, 1])
    losses = softmax_cross_entropy_with_integer_labels(logits, labels)
    np.testing.assert_array_equal(jnp.isnan(losses), [True, True, False])
    gradient = jax.grad(
        lambda z: jnp.where(
            labels == 1, softmax_cross_entropy_with_integer_labels(z, labels), 0
        ).sum()
    )(logits)
    np.testing.assert_array_equal(gradient[:2], 0)
    assert jnp.all(jnp.isfinite(gradient))


def test_loss_batching_and_curvature():
    x = jnp.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-100.0, 100.0, 0.0]])
    labels = jnp.array([0, 1, 0])
    actual = jax.jit(softmax_cross_entropy_with_integer_labels)(x, labels)
    expected = -jnp.take_along_axis(jax.nn.log_softmax(x), labels[:, None], -1)[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    np.testing.assert_allclose(
        actual,
        jax.vmap(softmax_cross_entropy_with_integer_labels)(x, labels),
        rtol=1e-6,
    )
    teacher = x[0]
    direction = jnp.array([1.0, -2.0, 0.5])
    p = jax.nn.softmax(teacher)
    expected_hvp = p * (direction - jnp.sum(p * direction))
    losses = (
        lambda z: softmax_cross_entropy_with_logit_targets(z, teacher),
        lambda z: softmax_cross_entropy_with_integer_labels(z, jnp.array(0)),
    )
    for loss in losses:
        hvp = jax.jit(lambda z, v: jax.jvp(jax.grad(loss), (z,), (v,))[1])(
            teacher, direction
        )
        np.testing.assert_allclose(hvp, expected_hvp, rtol=2e-6, atol=1e-7)


def test_inverse_sigmoid_clamps_the_logit_and_stays_finite():
    x = jnp.asarray([0.0, 1e-4, 0.25, 0.5, 1.0], jnp.bfloat16)
    logits = inverse_sigmoid(x)
    assert logits.dtype == jnp.float32
    limit = float(np.log(1e3))
    np.testing.assert_allclose(logits[jnp.asarray([0, 4])], [-limit, limit], rtol=1e-4)
    np.testing.assert_allclose(logits[1], logits[0], atol=1e-3)
    np.testing.assert_allclose(logits[3], 0.0, atol=1e-6)
    np.testing.assert_allclose(jax.nn.sigmoid(logits[2]), 0.25, rtol=1e-3)
