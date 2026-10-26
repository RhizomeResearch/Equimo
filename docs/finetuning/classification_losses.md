# Classification losses and numerical stability

`equimo.core.ops` provides optimizer-independent probability operations based on
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/html/2610.31642v1). They support JIT, batching with `jax.vmap`,
and forward/reverse and higher-order differentiation. They compute in at least float32, preserving float64 when JAX's
64-bit mode is enabled, and return the input floating-point dtype.

## Integer labels

```python
from equimo.core.ops import softmax_cross_entropy_with_integer_labels

# logits: (..., classes); labels: (...) integer class indices.
loss = softmax_cross_entropy_with_integer_labels(logits, labels).mean()
```

The loss excludes the target from the exponential sum and combines the non-target log-sum-exp with softplus. This
retains both a small positive loss and a negative target gradient when ordinary cross-entropy rounds them to zero.
Query-segmentation training uses this operation internally.

The function returns one loss per label; reductions and weighting belong to the caller. Labels outside `[0, classes)`
give NaN instead of wrapping like indices. To ignore such entries, mask their losses, which leaves them zero gradients.

## Teacher-logit distillation

```python
from equimo.core.ops import softmax_cross_entropy_with_logit_targets

temperature = 4.0
loss = temperature**2 * softmax_cross_entropy_with_logit_targets(
    student_logits / temperature,
    teacher_logits / temperature,
).mean()
```

Pass finite logits and target logits with the same shape `(..., classes)`; the result has shape `(...)`. Target
gradients are stopped.

The gradient is exactly zero for identical logits. Near equality, the derivative computes probability differences from
logit differences using `expm1` and `log1p`, instead of subtracting separately rounded probabilities. Widely separated
logits use a log-ratio form that cannot overflow. The forward value is cross-entropy, so at equality it equals the
teacher's entropy rather than zero.

Already rounded teacher probabilities cannot recover the lost information, so pass the teacher's logits. Existing
feature distillation with MSE or cosine distance serves a different objective.

## Softmax and log-softmax

```python
from equimo.core.ops import log_softmax, softmax

probabilities = softmax(logits, axis=-1)
log_probabilities = log_softmax(logits, axis=-1)
```

`softmax` accepts the axes of `jax.nn.softmax`; `log_softmax` takes one axis. Both accept `where`: masked softmax
entries are zero, masked log-softmax entries are negative infinity, and an entirely masked slice has those outputs and
zero gradients.

`softmax` computes `jax.nn.softmax` in at least float32 and returns the input dtype; its derivative changes. When one
probability exceeds 3/4, the derivative works with logit differences from that entry, retaining its small gradient
instead of rounding `1 - p` to zero. The correction applies whenever differentiated, including calls with
`inference=True`. Equimo uses it for softmaxes outside token-to-token attention: attention pooling, attention-pooling
heads, adapter fusion, the tabular decoder, and the `"softmax"` activation. Attention between tokens computes
`jax.nn.softmax` in float32; there the correction costs 5-10% of a ViT training step. The
[performance notes](../performance.md#softmax-precision) give the measurements.

Log-softmax excludes one maximum from the exponential sum and uses `log1p` for normalization. For probability labels
such as label smoothing, use `-(labels * log_softmax(logits)).sum(axis=-1)`; this does not provide the teacher-logit
cancellation correction.

These operations remain limited by the input dtype's resolution and exponent range: they cannot recover logit
differences already lost during model computation, or gradients that underflow.

## Inference and ONNX export

Softmax's custom derivative adds no operations to a forward-only graph: for float32 inputs, jax2onnx exports the same
graph as for `jax.nn.softmax`, and lower-precision inputs add a cast on each side. The correction needs no
training/inference flag or checkpoint changes.

Exporting stable log-softmax or the losses needs separate numerical qualification: jax2onnx 0.15.1 lowers `log1p` to
`Log(Add(x, 1))` and `expm1` to `Sub(Exp(x), 1)`, which lose small-value accuracy.
