# equimo.timeseries

## Overview

Time-series modality. The package root holds only a docstring [\_\_init\_\_.py]; public symbols live in the two
subpackages, which together implement the T0 patch-transformer quantile forecaster adapted from tfc-t0.

## Interface

- [layers](./layers/EGM.md): `AxisAttention`, `T0Block`, `PatchEncoder`, `ResidualMlp`, and the time-series
  `register_layer` / `get_layer` registry with core-layer fallback.
- [models](./models/EGM.md): `T0` (model `"t0"`), `Forecast`, and the `t0` / `t0_alpha` factories.

## Invariants

- The backbone works on flattened `(variates, time)` arrays with per-cell `mask`, `group_ids`, and `variate_type`
  metadata; `T0.predict` is the adapter from user-shaped context arrays to that layout \[models/t0.py:T0.\_\_call\_\_,
  models/t0.py:T0.predict\].
- `T0` resolves its patch encoder, blocks, and decoder by name through the time-series `get_layer`, which prefers
  time-series registrations over core layers with the same name \[models/t0.py:T0.\_\_init\_\_,
  layers/registry.py:get_layer\].

## Side Effects

- Importing the subpackages mutates the time-series layer registry and the global model registry (see child contracts).
  Pretrained loading downloads the `t0_alpha` checkpoint [models/t0.py:\_build_t0].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_t0.py -q`
