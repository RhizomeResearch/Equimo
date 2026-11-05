# equimo.tabular

## Overview

Tabular modality. The package root holds only a docstring [\_\_init\_\_.py]; all public symbols live in the two
subpackages. Layers implement TabPFN-style in-context learning over `(rows, columns)` datasets, and models assemble them
into the TabPFN v3 classifier and regressor variants.

## Interface

- [layers](./layers/EGM.md): `Preprocessor`, tabular `Attention` / `CrossAttention` / `InContextAttention` /
  `SoftmaxScaling`, attention blocks, `FeatureDistributionEncoder`, `ColumnAggregator`, label embeddings,
  `AttentionDecoder`, `RegressionDecoder`, and the per-family `register_*` / `get_*` registries.
- [models](./models/EGM.md): `TabPFN` (model `"tabpfn"`) and the `tabpfn*` variant factories.

## Invariants

- The leading `n_train` rows are the training context everywhere: preprocessing statistics, label embeddings, and
  in-context keys and values come only from them \[layers/preprocessing.py:Preprocessor.\_\_call\_\_,
  layers/attention.py:InContextAttention.\_\_call\_\_, models/tabpfn.py:TabPFN.features\].
- Modules are unbatched; batch datasets with `jax.vmap` [models/tabpfn.py:TabPFN].
- `TabPFN` resolves its pluggable submodules by name through the tabular family registries, so custom layers must be
  registered with the matching `register_*` decorator [models/tabpfn.py:TabPFN.\_\_init\_\_].

## Side Effects

- Importing the subpackages mutates module-level layer registries and the global model registry (see child contracts).
  Pretrained loading downloads checkpoints [models/tabpfn.py:\_build_tabpfn].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_tabular_layers.py tests/test_tabpfn.py -q`
