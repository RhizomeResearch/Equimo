# equimo.core.ops

## Overview

Pure-JAX numerical kernels.

- `scan.py` holds the state-space kernels: a selective scan, segment-sum, chunked SSD for Mamba2, and VSSD non-causal
  linear attention. These kernels take a single example without a batch axis, so callers batch them with `jax.vmap`.
- `probability.py` holds softmax, log-softmax, two cross-entropy losses and a clamped inverse sigmoid. The softmaxes and
  losses compute in at least float32 and use custom JVPs that stay accurate at high confidence.

## Interface

All of the following are re-exported by `equimo.core.ops` ([`__init__.py`](./__init__.py)).

- `[scan.py:selective_scan]`:
  `selective_scan(u (d, l), delta (d, l), A (d, n), B (n, l), C (n, l), D (d,), delta_bias=None, delta_softplus=False) -> (d, l)`
  in `u.dtype`. Runs a `lax.scan` over the sequence and adds the skip term `u * D`.
- `[scan.py:segsum]`: `segsum(x (..., T)) -> (..., T, T)`, a stable segment-sum matrix. Entries above the diagonal are
  `-inf`.
- `[scan.py:ssd]`:
  `ssd(x (l, h, p), A (l, h), B (l, h, n), C (l, h, n), chunk_size, initial_states=None) -> (Y (l, h, p), final_state (h, p, n))`.
  Chunked structured state-space duality.
- `[scan.py:non_causal_linear_attn]`:
  `non_causal_linear_attn(x (l, h, d), dt (l, h), A (h,), B (l, d_state), C (l, d_state), D (h,), n_groups=1) -> (l, h, d)`.
  VSSD non-causal attention duality, used by `Mamba2Mixer` ([../layers/EGM.md](../layers/EGM.md)).
- `[probability.py:inverse_sigmoid]`: `inverse_sigmoid(x)` in float32, with `x` and `1 - x` at least 1e-3, so the result
  stays within about +/-6.9 and is finite at 0 and 1.
- `[probability.py:softmax]`: `softmax(x, axis=-1, where=None)`. Same signature as `jax.nn.softmax`, computed in at
  least float32, returned in the input's float dtype.
- `[probability.py:log_softmax]`: `log_softmax(x, axis=-1, where=None)`. `axis` must be a single axis. Masked positions
  return `-inf`.
- `[probability.py:softmax_cross_entropy_with_integer_labels]`: `(logits (..., C), labels (...)) -> loss (...)`,
  unreduced, in the logits' dtype.
- `[probability.py:softmax_cross_entropy_with_logit_targets]`:
  `(logits (..., C), target_logits (..., C)) -> loss (...)`, unreduced. Gradients do not flow into the targets
  (`stop_gradient`). For distillation at temperature `T`, divide both inputs by `T` and multiply the loss by `T**2` (per
  the docstring).

## Invariants

- `[probability.py:_softmax_jvp]`: `softmax` has a `custom_jvp`.
  - At most one probability per slice can exceed 0.75. That entry is the anchor, and it is excluded from the centered
    tangent, so its gradient is rebuilt from the other entries rather than computed by subtracting two numbers near 1.
  - Slices without an anchor use the ordinary derivative.
  - Masked entries have probability 0, and fully masked rows return zeros.
- `[probability.py:softmax]`, `[probability.py:log_softmax]` and
  `[probability.py:softmax_cross_entropy_with_integer_labels]` follow the same dtype rule: they compute in
  `result_type(dtype, float32)` and return the input's float dtype. Integer inputs are promoted to float.
  `[probability.py:softmax_cross_entropy_with_logit_targets]` applies that rule to the common dtype of its two inputs,
  `result_type(logits.dtype, target_logits.dtype, 0.0)`, and returns that dtype.
- `[probability.py:_log_softmax_parts]`: exactly one maximum, the first argmax, is left out of the exponential sum and
  added back through `log1p`.
- `[probability.py:softmax_cross_entropy_with_integer_labels]`: input and output contract.
  - It raises `ValueError` unless `labels` is an integer array with shape `logits.shape[:-1]`.
  - Labels outside `[0, classes)` give `NaN` loss. They are evaluated as class 0, so the gradient stays finite once
    those losses are masked, and negative labels never wrap around to another class.
  - A slice with a single finite class gives zero loss and zero gradient.
- `[probability.py:softmax_cross_entropy_with_logit_targets]`: raises `ValueError` if the two inputs have different
  shapes.
- `[probability.py:_cross_entropy_with_logit_targets_jvp]`: this `custom_jvp` takes its tangent from
  `_probability_difference`. The gradient is exactly zero when the logits equal the targets and stays accurate near
  equality.
- `[scan.py:selective_scan]`, `[scan.py:ssd]`: `exp` arguments (`delta * A` and the cumulative sums of `A`) are promoted
  to float32, and outputs are cast back to the input dtype (`u.dtype` or `x.dtype`).
- `[scan.py:ssd]`: raises `ValueError` in these cases:
  - `chunk_size <= 0`;
  - the inputs do not have the documented ranks or matching leading shapes, or `B.shape != C.shape`;
  - the sequence length is not divisible by `chunk_size`;
  - `initial_states` is neither `(h, p, n)` nor `(1, h, p, n)`.
- `[scan.py:non_causal_linear_attn]`: raises `ValueError` when `n_groups != 1` and `h % n_groups != 0`. This function
  does no dtype casting.

## Side Effects

- None. Every function is pure JAX, with no global state, randomness, or I/O.

## Verification

- Run: `devenv shell -- uv run pytest tests/test_probability.py tests/test_scan.py tests/test_activation.py -q`
