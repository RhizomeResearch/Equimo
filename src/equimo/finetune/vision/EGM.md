# equimo.finetune.vision

## Overview

Vision-specific fine-tuning utilities: a dense (per-patch) linear probe built on `extract_features`, dense feature
adapter and distillation presets, and model-specific fine-tuning plans for the EoMT and PMT segmentation models
(`src/equimo/vision/models/eomt.py`, `src/equimo/vision/models/pmt.py`). Plans are produced by the parent package's
`prepare_finetune`.

## Interface

- `DenseVisionConfig`: frozen dataclass with `activation="identity"`, `dropout=0.0`, `bias=True`, and a
  `FeatureDistillationConfig.dense` distillation default [dense.py:DenseVisionConfig]
- `DenseProbe(backbone, head, *, feature_spec)`: Equinox module returning class-first patch logits
  `(classes, grid_h, grid_w)` for one example [dense.py:DenseProbe]
- `make_dense_probe(backbone, *, in_features, out_features, key, feature_spec, head=None) -> DenseProbe`: replaces the
  backbone head with an identity when one resolves and attaches a pointwise `LinearHead` [dense.py:make_dense_probe]
- `dense_feature_adapter(in_features, out_features, *, key, config=None) -> DenseFeatureAdapter`
  [dense.py:dense_feature_adapter]
- `dense_distillation_config(*, layers=("25%", "50%", "75%", "100%"), normalize_features=True)`: dense
  `FeatureDistillationConfig` preset [dense.py:dense_distillation_config]
- `eomt_full_finetune(model: EoMT, *, decay=0.8) -> FineTunePlan`: full fine-tuning with layer-wise learning-rate decay
  [eomt.py:eomt_full_finetune]
- `pmt_head_finetune(model: PMT, *, state=None) -> FineTunePlan`: trains lateral, decoder, query, and prediction
  parameters with the ViT frozen [pmt.py:pmt_head_finetune]
- Package exports: the above plus the `dense` submodule [\_\_init\_\_.py:\_\_all\_\_]

## Invariants

- `DenseProbe` and `make_dense_probe` reject a `FeatureSpec` that is pooled, has `return_metadata=False`, keeps separate
  feature levels, or is not either `BNC` with `token_selection="patches"` or `BCHW` with `token_selection="all"`
  [dense.py:\_validate_dense_feature_spec]
- `DenseProbe.__call__` is unbatched: `BNC` features must be `(grid_h * grid_w, in_features)` and `BCHW` features
  `(in_features, grid_h, grid_w)`; batching is done with `jax.vmap` [dense.py:DenseProbe]
- `DenseProbe.__call__` raises `ValueError` unless `extract_features` returns a `FeatureResult` with exactly one array
  level, explicit `grid_size` metadata, and a `feature_width` equal to `head.linear.in_features` [dense.py:DenseProbe]
- `DenseProbe.__call__` defaults to `inference=True` and forwards `key` and `inference` to `extract_features`
  [dense.py:DenseProbe]
- `DenseProbe.feature_spec` is a static Equinox field [dense.py:DenseProbe]
- `make_dense_probe` wraps the backbone with the parent package's private `_headless_backbone` and builds the head with
  `_linear_probe_head`. `_headless_backbone` returns the backbone unchanged when `replace_head` raises `ValueError` (no
  unique head resolves), so the original head stays in place [dense.py:make_dense_probe]
  [../feature_extraction.py:\_headless_backbone]
- `eomt_full_finetune` raises `TypeError` for non-`EoMT` models; it trains all parameters in `mode="full"` except
  buffers and the inexact leaves under `backbone.mask_token` and `backbone.local_cls_norm`, which EoMT does not use, and
  labels groups with `LLRDConfig(decay=decay)` [eomt.py:eomt_full_finetune]
- `pmt_head_finetune` raises `TypeError` for non-`PMT` models and `ValueError` when `model.is_stateful()` (BatchNorm)
  and no `eqx.nn.State` is passed; it uses `TrainableSpec(mode="head")`, `LLRDConfig(decay=1.0)`, and forwards the state
  as `model_state` [pmt.py:pmt_head_finetune]

## Side Effects

- None: functions return new modules or plans; `make_dense_probe` and `dense_feature_adapter` consume only the PRNG key
  passed in [dense.py:make_dense_probe] [dense.py:dense_feature_adapter]

## Verification

- Run:
  `devenv shell -- uv run pytest tests/finetune/test_dense_probe.py tests/finetune/test_plan_identities.py tests/test_eomt.py tests/test_pmt.py tests/test_pmt_compositions.py tests/test_query_training_models.py -q`
- `DenseVisionConfig`, `dense_feature_adapter`, and `dense_distillation_config` have no behavioral tests; only
  `tests/finetune/test_scaffolding.py` checks that `dense_feature_adapter` is exported
