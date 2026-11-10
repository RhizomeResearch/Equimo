"""Probability operations with cancellation-resistant derivatives.

Based on Barron et al., "A Fix for Softmax and Cross-Entropy"
(https://arxiv.org/html/2610.31642v1). Softmax computes JAX's forward values in
at least float32; its custom JVP anchors logit differences at a dominant
probability.
"""

from functools import partial

import jax
import jax.numpy as jnp


def softmax(
    x: jax.Array,
    axis: int | tuple[int, ...] | None = -1,
    where: jax.Array | None = None,
) -> jax.Array:
    """Softmax with a cancellation-resistant derivative.

    ``axis`` and ``where`` follow :func:`jax.nn.softmax`. Masked outputs are zero,
    including fully masked rows. Computation uses at least float32 precision and
    returns the input dtype; float32 inputs export as the usual softmax graph.

    The correction applies whenever differentiated, including deterministic
    inference calls.
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x.dtype, 0.0))
    working = x.astype(jnp.result_type(x.dtype, jnp.float32))
    return _softmax(working, axis, where).astype(x.dtype)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(x, axis, where):
    return jax.nn.softmax(x, axis=axis, where=where)


@_softmax.defjvp
def _softmax_jvp(axis, primals, tangents):
    (x, where), (x_dot, _) = primals, tangents
    p = _softmax(x, axis, where)
    # Rounding cannot place two probabilities above 3/4, so a slice has at most
    # one anchor. Without one, the ordinary derivative is accurate; masked
    # entries have zero probability and need no special handling.
    anchor = p > 0.75
    t_anchor = jnp.sum(jnp.where(anchor, x_dot, 0), axis=axis, keepdims=True)
    # Explicitly exclude the anchor so transposition reconstructs its gradient
    # from the other entries, instead of subtracting two numbers near one.
    centered = jnp.where(anchor, 0, x_dot - t_anchor)
    return p, p * (centered - jnp.sum(p * centered, axis=axis, keepdims=True))


def _anchor_mask(x: jax.Array) -> jax.Array:
    return jnp.arange(x.shape[-1]) == jnp.argmax(x, axis=-1, keepdims=True)


def _log_softmax_parts(x: jax.Array, where: jax.Array | None = None):
    """Keep shifted logits and the small log normalizer separate."""
    if where is not None:
        x = jnp.where(where, x, -jnp.inf)
    anchor = _anchor_mask(x)
    maximum = jnp.sum(jnp.where(anchor, x, 0), axis=-1, keepdims=True)
    empty = False if where is None else ~jnp.any(where, axis=-1, keepdims=True)
    maximum = jnp.where(empty, 0, maximum)
    shifted = jnp.where(anchor, 0, x - maximum)
    others = jnp.where(anchor, 0, jnp.exp(shifted))
    normalizer = jnp.log1p(jnp.sum(others, axis=-1, keepdims=True))
    normalizer = jnp.where(jnp.isfinite(maximum), normalizer, jnp.nan)
    return shifted, normalizer


def _log_softmax_last(x: jax.Array, where: jax.Array | None = None) -> jax.Array:
    shifted, normalizer = _log_softmax_parts(x, where)
    result = shifted - normalizer
    return result if where is None else jnp.where(where, result, -jnp.inf)


def log_softmax(
    x: jax.Array, axis: int = -1, where: jax.Array | None = None
) -> jax.Array:
    """Log probabilities that retain small losses and dominant-logit gradients.

    ``axis`` is a single axis; ``where`` follows :func:`jax.nn.log_softmax`.
    Exactly one maximum is excluded from the exponential sum, then combined with
    it using log1p. Computation uses at least float32 precision and returns the
    input dtype.
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x.dtype, 0.0))
    working = x.astype(jnp.result_type(x.dtype, jnp.float32))
    valid = None if where is None else jnp.broadcast_to(where, x.shape)
    result = _log_softmax_last(
        jnp.moveaxis(working, axis, -1),
        None if valid is None else jnp.moveaxis(valid, axis, -1),
    )
    return jnp.moveaxis(result, -1, axis).astype(x.dtype)


def inverse_sigmoid(x: jax.Array) -> jax.Array:
    """Return ``logit(x)`` in float32, with ``x`` and ``1 - x`` at least 1e-3.

    The clamp is the one DETR-style box refinement uses. It caps the result at
    about 6.9 and keeps it finite at 0 and 1.
    """
    x = jnp.clip(x.astype(jnp.float32), 0.0, 1.0)
    return jnp.log(jnp.maximum(x, 1e-3) / jnp.maximum(1.0 - x, 1e-3))


def softmax_cross_entropy_with_integer_labels(
    logits: jax.Array, labels: jax.Array
) -> jax.Array:
    """Unreduced cross-entropy with integer labels on the last axis.

    ``logits`` has shape ``(..., classes)`` and ``labels`` has shape ``(...)``.
    The result has shape ``(...)`` and the logits' dtype. Labels outside
    ``[0, classes)`` give NaN; masking those losses leaves zero gradients.
    Computation uses at least float32 precision.
    """
    logits, labels = jnp.asarray(logits), jnp.asarray(labels)
    logits = logits.astype(jnp.result_type(logits.dtype, 0.0))
    if labels.shape != logits.shape[:-1] or not jnp.issubdtype(
        labels.dtype, jnp.integer
    ):
        raise ValueError(
            "labels must be integer class indices with shape logits.shape[:-1]."
        )
    # Indexing would wrap negative labels onto the wrong class. Evaluating
    # invalid labels as class 0 keeps their masked-out gradients finite.
    in_range = (labels >= 0) & (labels < logits.shape[-1])
    labels = jnp.where(in_range, labels, 0)
    working = logits.astype(jnp.result_type(logits.dtype, jnp.float32))
    target = jnp.take_along_axis(working, labels[..., None], axis=-1)
    is_target = jnp.arange(logits.shape[-1]) == labels[..., None]
    others = jnp.where(is_target, -jnp.inf, working - target)
    maximum = jax.lax.stop_gradient(jnp.max(others, axis=-1))
    empty = maximum == -jnp.inf
    maximum = jnp.where(empty, 0, maximum)
    total = jnp.sum(jnp.exp(others - maximum[..., None]), axis=-1)
    # A singleton class (or only one finite logit) has zero loss and gradient;
    # avoid differentiating log(0) even in an unselected branch.
    log_total = maximum + jnp.log(jnp.where(empty, 1, total))
    loss = jnp.where(empty, 0, jax.nn.softplus(log_total))
    return jnp.where(in_range, loss, jnp.nan).astype(logits.dtype)


def _probability_difference(logits: jax.Array, target_logits: jax.Array) -> jax.Array:
    shifted_q, normalizer_q = _log_softmax_parts(logits)
    shifted_p, normalizer_p = _log_softmax_parts(target_logits)
    q, p = jnp.exp(shifted_q - normalizer_q), jnp.exp(shifted_p - normalizer_p)
    delta = logits - target_logits
    # Remove a common offset before centering, making constant shifts exact.
    delta = delta - delta[..., :1]
    centered = delta - jnp.sum(p * delta, axis=-1, keepdims=True)
    nearby = jnp.max(jnp.abs(centered), axis=-1, keepdims=True) <= 0.5
    # Both branches must be safe under vmap/select. Near equality, avoid
    # subtracting separately rounded log probabilities; far away, avoid expm1
    # overflow by using their ordinary log ratio instead. The bound of 0.5
    # keeps expm1 in a small interval and its weighted sum safely above -1.
    small = jnp.where(nearby, centered, 0)
    log_ratio = small - jnp.log1p(jnp.sum(p * jnp.expm1(small), axis=-1, keepdims=True))
    # Subtract normalizers separately so a tiny normalization change is not
    # lost when both shifted logits are large and negative.
    ordinary_ratio = (shifted_q - shifted_p) - (normalizer_q - normalizer_p)
    log_ratio = jnp.where(nearby, log_ratio, ordinary_ratio)
    nonpositive = log_ratio <= 0
    difference = jnp.where(
        nonpositive,
        p * jnp.expm1(jnp.where(nonpositive, log_ratio, 0)),
        -q * jnp.expm1(jnp.where(nonpositive, 0, -log_ratio)),
    )
    return difference - q * jnp.sum(difference, axis=-1, keepdims=True)


@jax.custom_jvp
def _cross_entropy_with_logit_targets(logits, target_logits):
    p = jnp.exp(_log_softmax_last(target_logits))
    return -jnp.sum(p * _log_softmax_last(logits), axis=-1)


@_cross_entropy_with_logit_targets.defjvp
def _cross_entropy_with_logit_targets_jvp(primals, tangents):
    logits, target_logits = primals
    logits_dot, _ = tangents
    loss = _cross_entropy_with_logit_targets(logits, target_logits)
    gradient = _probability_difference(logits, target_logits)
    return loss, jnp.sum(gradient * logits_dot, axis=-1)


def softmax_cross_entropy_with_logit_targets(
    logits: jax.Array, target_logits: jax.Array
) -> jax.Array:
    """Unreduced cross-entropy against the softmax of target logits.

    Inputs are finite floating-point logits with matching shape ``(..., classes)``.
    Returns shape ``(...)`` in their common dtype, computing in at least float32.
    Target gradients are stopped. The gradient is exactly zero for equal logits
    and remains accurate near equality. Forward/reverse and higher-order
    differentiation are supported.

    For distillation at temperature ``T``, pass both logits divided by ``T`` and
    multiply the returned loss by ``T**2``. Probability targets cannot recover
    differences already lost through rounding; pass the teacher's logits directly.
    """
    logits, target_logits = jnp.asarray(logits), jnp.asarray(target_logits)
    if logits.shape != target_logits.shape:
        raise ValueError("Logits and target logits must have the same shape.")
    dtype = jnp.result_type(logits.dtype, target_logits.dtype, 0.0)
    working_dtype = jnp.result_type(dtype, jnp.float32)
    working = logits.astype(working_dtype)
    targets = jax.lax.stop_gradient(target_logits.astype(working_dtype))
    return _cross_entropy_with_logit_targets(working, targets).astype(dtype)
