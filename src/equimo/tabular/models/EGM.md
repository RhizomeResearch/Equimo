# equimo.tabular.models

## Overview

Tabular model families. The package `__init__` auto-imports every sibling module and re-exports its `__all__`. The only
family today is TabPFN (v3 architecture): an in-context predictor that takes a whole dataset (training rows with labels
followed by test rows) and returns predictions for the test rows.

## Interface

- `TabPFN(*, num_classes=160, num_buckets=5000, task_type="classification", dim=128, depths=(3, 3, 24), num_heads=(8, 8, 8), ..., key)`
  \[tabpfn.py:TabPFN\]: registered as model `"tabpfn"` with `modality="tabular"`. Submodules are resolved by name from
  the tabular layer registries (`context_block`, `preprocessor_layer`, `label_embedding_layer`, `decoder_layer`).
  - `features(x, y, n_train, key=None, inference=None)` returns context tokens `(rows, dim * num_cls_tokens)`
    [tabpfn.py:TabPFN.features].
  - `intermediate_features(x, y, n_train, ..., indices=None, n_last_blocks=None)` returns selected context-block outputs
    [tabpfn.py:TabPFN.intermediate_features].
  - `forward_features(...)` returns `x_norm_train`, `x_norm_test`, `x_prenorm` [tabpfn.py:TabPFN.forward_features].
  - `__call__(x, y, n_train, ...)` returns decoder output for the test rows `x[n_train:]`
    [tabpfn.py:TabPFN.\_\_call\_\_].
- Custom submodule classes must accept the arguments `TabPFN` passes; `act_layer` arrives as a resolved callable and
  `norm_layer` as a resolved class \[tabpfn.py:TabPFN.\_\_init\_\_, tabpfn.py:TabPFN.features,
  tabpfn.py:TabPFN.\_\_call\_\_\]:
  - `context_block` is built by `BlockChunk` with keywords `dim` (`dim * num_cls_tokens`), `num_heads`, `mlp_ratio`,
    `num_kv_heads_test`, `scaling_mlp_hidden_dim`, `act_layer`, `norm_layer`, `eps`, `drop_path`, and `key`, and is
    called as `(x, n_train=n_train, key=..., inference=...)`.
  - `preprocessor_layer` is built with keywords `feature_group_size` and `use_nan_indicators` and called as
    `(x, n_train)`.
  - `label_embedding_layer` is built as `(num_outputs, width, key=...)` for widths `dim` and `dim * num_cls_tokens` and
    called on `y[:n_train]`.
  - `decoder_layer` is built as `(num_outputs, dim * num_cls_tokens, ...)` with keywords `head_dim`, `num_heads`,
    `use_softmax_scaling`, `scaling_mlp_hidden_dim`, `mlp_ratio`, `act_layer`, and `key`, and is called as
    `(train_tokens, test_tokens, y[:n_train], n_train)`.
- Factories accepting `pretrained`, `inference_mode`, `key`, and config overrides: `tabpfn`, `tabpfn_regressor`,
  `tabpfn_v3_classifier_default`, `tabpfn_v3_classifier_binary`, `tabpfn_v3_classifier_multiclass`,
  `tabpfn_v3_classifier_ood`, `tabpfn_v3_regressor_default`, `tabpfn_v3_regressor_mediumdata`,
  `tabpfn_v3_regressor_ood`, `tabpfn_v3_regressor_timeseries` [tabpfn.py:\_\_all\_\_, tabpfn.py:\_TABPFN_REGISTRY].
- Re-exported registry helpers `get_model_cls`, `register_model` from `equimo.registry` [\_\_init\_\_.py].
- Internal catalog hook `_catalog_model_variants()` describing `tabular/tabpfn_v3_classifier_default`
  [tabpfn.py:\_catalog_model_variants].

## Invariants

- Inputs are one unbatched dataset: `x` is `(rows, columns)`, `y` holds labels or targets for at least the first
  `n_train` rows, and the first `n_train` rows are the in-context training set; use `jax.vmap` over datasets
  [tabpfn.py:TabPFN, tabpfn.py:TabPFN.features].
- `n_train` must be a concrete integer, not a traced value, because it slices arrays and drives Python control flow
  [src/equimo/tabular/layers/preprocessing.py:Preprocessor.\_\_call\_\_], although the catalog lists it as an `int32`
  scalar input [tabpfn.py:\_catalog_model_variants].
- `features` and `intermediate_features` ignore extra `**kwargs`, and `forward_features` and `__call__` pass theirs to
  `features` \[tabpfn.py:TabPFN.features, tabpfn.py:TabPFN.intermediate_features, tabpfn.py:TabPFN.forward_features,
  tabpfn.py:TabPFN.\_\_call\_\_\].
- `task_type` must be `"classification"` or `"regression"`; `depths` must have exactly three entries (feature encoder,
  column aggregator, context blocks); `dim` must be divisible by the first two head counts and `dim * num_cls_tokens` by
  the third (`ValueError` otherwise) [tabpfn.py:TabPFN.\_\_init\_\_]. `num_heads` is an int or a list or tuple of
  exactly three entries; another length raises `AssertionError` [src/equimo/utils.py:to_list].
- Output width is `num_classes` for classification and `num_buckets` for regression [tabpfn.py:TabPFN.\_\_init\_\_].
- The embedding input width is `feature_group_size * 2` with NaN indicators, `feature_group_size` without
  [tabpfn.py:TabPFN.\_\_init\_\_].
- Label embeddings are added only to training rows, both before the feature encoder (per column) and before the context
  blocks [tabpfn.py:TabPFN.features].
- Regressor variants switch to `linearlabelembedding`, `regressiondecoder`, and no decoder softmax scaling
  [tabpfn.py:\_TABPFN_REGRESSOR_CFG].
- `tabpfn` and `tabpfn_regressor` load the `tabpfn_v3_classifier_default` and `tabpfn_v3_regressor_default` checkpoints
  when pretrained [tabpfn.py:\_TABPFN_PRETRAINED_IDENTIFIERS].

## Side Effects

- Importing registers `TabPFN` in the global model registry via `@register_model("tabpfn", modality="tabular")`
  [tabpfn.py:TabPFN].
- `pretrained=True` loads weights through `equimo.core.factory.build_model_variant`, which downloads the checkpoint
  [tabpfn.py:\_build_tabpfn, src/equimo/core/factory.py:build_model_variant]. The catalog notes that pretrained TabPFN-3
  weights use the `tabpfn-3-license-v1.0` license [tabpfn.py:\_catalog_model_variants].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_tabpfn.py -q`
