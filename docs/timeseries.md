# Time-Series Models

> **Experimental:** The `equimo.timeseries` namespace and its public exports
> are not covered by Equimo's v2 compatibility contract. The raw tensor
> interface may change in a minor release while a checkpoint-faithful
> forecasting adapter is developed.

Equimo currently provides the T0 and T0-alpha patch-transformer backbones. The
`T0`, `t0()`, and `t0_alpha()` interfaces expose the direct raw forward pass;
`T0.predict()` additionally provides the upstream-style forecasting adapter.

## Raw Input Contract

The model accepts four unbatched arrays with the same `(variates, time)` shape:

| Input | Dtype | Contract |
| ----- | ----- | -------- |
| `values` | floating point | Values consumed exactly as supplied. Replace unavailable values with a finite placeholder such as zero and describe their meaning with `mask`. |
| `mask` | integer | Per-cell reason: `0` valid, `1` padding, `2` missing, `3` censored, or `4` withheld for prediction. |
| `group_ids` | integer | Nonnegative sample identifier. Target and covariate rows belonging to the same sample use the same ID; negative IDs are reserved for padding. |
| `variate_type` | integer | Per-cell role: `0` target, `1` historical covariate, or `2` known-future covariate. |

Multiple samples and their variates can be flattened together along the
`variates` axis and separated with `group_ids`. The role and group metadata for
a patch are taken from its first cell, so each patch should not cross a group
or role boundary.

Do not pass NaNs directly to the backbone. The upstream input adapter replaces
NaNs with zero and marks the corresponding cells as missing before the forward
pass.

## Scaling and Forecasting Boundary

The pretrained checkpoint was trained behind an upstream forecasting pipeline.
`T0.predict()` applies that pipeline. Callers using the raw backbone directly
must reproduce these steps around `model(...)`:

- target and historical rows use causal running location and scale statistics;
- known-future covariates use per-row global statistics;
- standardized values receive an `arcsinh` transform;
- decoded quantiles are inverse-transformed back to the original value domain;
- requested non-native quantiles are interpolated;
- target/withheld horizon cells are selected from the raw output; and
- horizons longer than the direct prediction window use autoregressive rollout.

`T0.predict()` performs those operations for raw context arrays and returns
finite forecasts in the original value domain. Direct `model(...)` calls still
consume values as-is and return quantiles in the supplied domain. The native
quantile levels are available as `model.quantile_levels`.

## Padding, Outputs, and Features

For `V` variates, input length `T`, patch size `P`, and `Q` native quantiles,
the forward pass returns:

```text
(V, ceil(T / P), P, Q)
```

Quantiles are ordered like `model.quantile_levels` and are monotonically
nondecreasing along the final axis. T0 left-pads all four inputs to a multiple
of `P`. The raw output therefore includes positions for that leading padding.
Flatten the patch axes and discard the same number of leading cells to align
the output with the original input:

```python
pad = (-time_length) % model.patch_size
aligned = raw_quantiles.reshape(
    num_variates,
    -1,
    len(model.quantile_levels),
)[:, pad : pad + time_length]
```

The model produces quantiles for every aligned cell. Select target cells marked
`WITHHELD` (`mask == 4`) to obtain the raw forecast horizon before applying the
required inverse scaling.

`model.features(...)` returns post-normalization tokens shaped
`(V * ceil(T / P), embed_dim)` in variate-major, patch-minor order. The token
sequence includes the leading padded patch when padding is required.

## Example Raw Forward Pass

This example assumes that `context_scaled` has already been transformed with
checkpoint-faithful statistics. It demonstrates the array layout and output
selection, not a complete forecasting pipeline.

```python
import jax.numpy as jnp
import jax.random as jr

import equimo.timeseries.models as tm

key = jr.PRNGKey(0)
model = tm.t0_alpha(pretrained=True, key=key)

context_scaled = jnp.zeros((1, 64), dtype=jnp.float32)
horizon = 32
values = jnp.concatenate(
    (context_scaled, jnp.zeros((1, horizon), dtype=jnp.float32)),
    axis=-1,
)
mask = jnp.concatenate(
    (
        jnp.zeros_like(context_scaled, dtype=jnp.int8),  # VALID
        jnp.full((1, horizon), 4, dtype=jnp.int8),  # WITHHELD
    ),
    axis=-1,
)
group_ids = jnp.zeros(values.shape, dtype=jnp.int32)
variate_type = jnp.zeros(values.shape, dtype=jnp.int32)  # TARGET

raw_quantiles = model(
    values,
    mask,
    group_ids,
    variate_type,
    key=key,
    inference=True,
)

pad = (-values.shape[-1]) % model.patch_size
aligned = raw_quantiles.reshape(
    values.shape[0],
    -1,
    len(model.quantile_levels),
)[:, pad : pad + values.shape[-1]]
forecast_scaled = aligned[mask == 4]
```

For additional forecasting and preprocessing details, consult the upstream
[`tfc-t0`](https://github.com/theforecastingcompany/tfc-t0) project and
[`t0-alpha`](https://huggingface.co/theforecastingcompany/t0-alpha) model card.

## License and Attribution

Equimo's T0 implementation is a modified JAX/Equinox adaptation of
Apache-2.0-licensed upstream work. The distributed
[`NOTICE`](../NOTICE) preserves the TFC, Datadog, and Chronos-2 attributions;
the corresponding license is distributed at
[`LICENSES/tfc-t0-APACHE-2.0.txt`](../LICENSES/tfc-t0-APACHE-2.0.txt).
The [source-provenance audit](./licensing/t0-source-provenance.md) documents why
the package retains that source-code attribution.

The converted T0-alpha pretrained weights are a separate distribution concern.
Their bundled snapshot, current upstream license link, and last-checked date are
listed in the
[pretrained-model license index](../LICENSES/pretrained/README.md#t0-alpha).
Check the latest upstream terms before using the weights because the bundled
copy can become outdated.
