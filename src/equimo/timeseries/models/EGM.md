# equimo.timeseries.models

## Overview

Time-series model families. The package `__init__` auto-imports every sibling module and re-exports its `__all__`. The
only family today is T0, a patch-transformer quantile forecaster adapted from tfc-t0 (see the repository `NOTICE`),
exposed both as a raw backbone over flattened variates and through a `predict` rollout adapter.

## Interface

- `T0(*, embed_dim=512, num_layers=24, num_heads=8, mlp_hidden_dim=2048, patch_size=32, group_every_n=3, dropout=0.1, quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9), patch_encoder_layer="patchencoder", block_layer="t0block", decoder_layer="residualmlp", key)`
  \[t0.py:T0\]: registered as model `"t0"` with `modality="timeseries"`.
  - `__call__(values, mask, group_ids, variate_type, *, key=None, inference=None)` on `(variates, time)` arrays returns
    `(variates, patches, patch_size, quantiles)` in the model's scaled space [t0.py:T0.\_\_call\_\_].
  - `features(...)` returns post-norm tokens `(variates * patches, embed_dim)` in variate-major, patch-minor order
    [t0.py:T0.features].
  - `predict(context, horizon, quantiles=(0.1, 0.5, 0.9), future_covariates=None) -> Forecast` accepts `context` of
    shape `[T]`, `[B, T]`, or `[B, V, T]` (NaN means missing) and optional `future_covariates` of shape
    `[B, F, T + horizon]` [t0.py:T0.predict].
- `Forecast(quantiles, quantile_levels)` frozen dataclass with a `median` property that indexes level 0.5 or
  interpolates it [t0.py:Forecast]. From `predict`, `quantiles` is float32 with shape `[B, V, horizon, Q]` for a 3D
  context and `[B, horizon, Q]` otherwise (a 1D context gives `B = 1`), and `quantile_levels` holds the query quantiles
  [t0.py:T0.predict, t0.py:\_prepare_input].
- Factories `t0(pretrained=False, inference_mode=True, **kwargs)` and `t0_alpha(...)` [t0.py:t0, t0.py:t0_alpha].
- Re-exported registry helpers `get_model_cls`, `register_model` [\_\_init\_\_.py].
- Internal catalog hook `_catalog_model_variants()` describing `timeseries/t0_alpha` [t0.py:\_catalog_model_variants].

## Invariants

- Constructor checks raise `ValueError` unless `embed_dim % num_heads == 0`, `embed_dim / num_heads` is even (XPos),
  `group_every_n` divides `num_layers` when positive, `patch_size >= 1`, and `quantile_levels` is non-empty with every
  value in `(0, 1)`; levels are stored sorted [t0.py:T0.\_\_init\_\_].
- Every `group_every_n`-th block (1-based) uses group attention across variates; the rest use time attention
  [t0.py:T0.\_\_init\_\_].
- Per-cell mask codes are `0` valid, `1` padding, `2` missing, `4` withheld; variate types are `0` target and `2` future
  covariate (`1` is treated as causal history in scaling) [t0.py:\_VALID, t0.py:\_scale_input].
- Inputs whose length is not a multiple of `patch_size` are left-padded with mask `1`, group id `-1`, and type `-1`
  [t0.py:T0.\_patch]. Time attention is restricted to the same group, is causal except that future-covariate query
  patches attend in both directions, and excludes fully padded key patches; group attention is restricted to the same
  group [t0.py:T0.\_masks].
- Decoded quantiles are monotone non-decreasing along the quantile axis (first value plus cumulative softplus)
  [t0.py:T0.\_\_call\_\_].
- `predict` requires `horizon >= 1`, query quantiles strictly ascending, unique, and in `(0, 1)`, and a `patch_size`
  that divides `DEFAULT_MAX_HORIZON` (1024), all checked with `ValueError`; a model with another `patch_size` builds but
  cannot `predict` [t0.py:T0.predict]. It scales inputs with causal or global `arcsinh` normalization, rescales outputs
  with `sinh`, decodes at most `DEFAULT_MAX_HORIZON` (1024) steps per window, rolls out longer horizons with
  probability-weighted quantile paths, and replaces NaN and infinities in the result with 0 \[t0.py:T0.predict,
  t0.py:\_scale_input, t0.py:\_rescale_predictions, t0.py:\_weighted_quantile\].
- `predict` always runs with `inference=True` and a fixed internal key `jr.PRNGKey(42)`, and uses host-side Python
  control flow for the rollout loop [t0.py:T0.predict, t0.py:\_predict_step].
- Pretrained builds accept no config overrides (`allow_pretrained_overrides=False`); both `t0` and `t0_alpha` load the
  `t0_alpha` checkpoint [t0.py:\_build_t0, t0.py:\_T0_PRETRAINED_IDENTIFIERS].

## Side Effects

- Importing registers `T0` in the global model registry via `@register_model("t0", modality="timeseries")` [t0.py:T0].
- `pretrained=True` loads weights through `equimo.core.factory.build_model_variant`, which downloads the checkpoint
  [t0.py:\_build_t0, src/equimo/core/factory.py:build_model_variant].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_t0.py -q`
