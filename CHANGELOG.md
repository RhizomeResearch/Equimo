# Changelog

All notable changes to Equimo are documented here. Equimo follows Semantic Versioning from version 2.0.0 onward.

## [2.3.0] - 2026-09-24

### Added

- Exact spatial and intermediate ViT feature contracts, including patch grids, prefix-token order, normalization stage,
  positional metadata, and executable `FeatureSpec` endpoint options.
- A pointwise spatial linear probe with explicit feature geometry, ten-class pooled-probe qualification, and tested
  head-only and final-block fine-tuning plans.
- Native LingBot-Vision Small support with strict checkpoint conversion, pinned source and weight identities, and
  intermediate-feature reference checks.
- Native EoMT query segmentation with encoder-block query insertion, mask annealing, auxiliary predictions, semantic
  score reduction, and a full-encoder fine-tuning plan.
- A reusable plain mask decoder and frozen-encoder PMT compositions for DINOv3 Small and LingBot-Vision Small, with
  explicit lateral normalization state, encoder-bound feature caches, and full or decoder-only checkpoints.

### Changed

- Checkpoint readers accept caller-selected resource limits and validate array declarations before allocation.
- Pretrained loading verifies pinned archive identities. Selected probes have offline numerical and ONNX qualification
  records; EoMT and PMT have pinned numerical references.

## [2.2.0] - 2026-09-05

### Added

- Expanded ConvNeXt model factories, including ConvNeXt V2 with Global Response Normalization, RMS-normalized
  variants, and overlapping stems.
- Pretrained weights for 66 ConvNeXt checkpoints: 40 V1 and 26 V2 variants, including FCMAE backbones and fine-tuned
  classifiers. Public factories support `pretrained=True` and named upstream checkpoint variants.
- Trusted timm reference fixtures for ConvNeXt Atto, ConvNeXt Zepto RMS with an overlapping stem, and ConvNeXt V2
  Atto. Feature and classifier-logit parity runs through the same reference suite as DINOv3, with elementwise
  `atol=1e-4` and `rtol=1e-5`; direct Torch conversion tests also cover V2 blocks with overlapping stems.
- Bundled upstream license snapshots, a pretrained-model license index, and conversion attribution notices in
  source and wheel distributions. ConvNeXt V2 weights retain their upstream CC BY-NC 4.0 terms. T0 source provenance
  is documented separately from its checkpoint license.
- A reproducible CPU performance benchmark and [performance audit](docs/performance.md).

### Changed

- Pinned the pretrained Hugging Face repository to `6916a4a3d460cfa804419454baf3d153885f3b6a` and registered trusted
  SHA-256 digests for all new ConvNeXt archives.
- ConvNeXt conversion now uses pinned upstream revisions, strict parameter mapping, native-resolution feature and
  logit comparisons for two deterministic inputs, and an exact archive-reload check. The checked-in conversion
  report records source revisions, archive digests, and measured errors for all 66 checkpoints.
- Reduced tabular attention work by projecting keys and values only for training rows, and replaced PartialFormer's
  second permutation sort with an integer scatter.
- Reduced fine-tuning setup and merge/unmerge overhead through ancestor-based selector matching, batched PEFT wrapper
  replacement, and triangular solves in RegMean's Cholesky solver.
- Consolidated shared feature-stage execution, tabular registry registration, adapter forwarding, and LoRA lifecycle
  helpers while retaining their existing behavior.

### Fixed

- Global Response Normalization now has finite gradients for zero-valued channels while preserving its forward values.
- Pretrained ConvNeXt models use exact GELU, including the activated overlapping stem, to match upstream checkpoint
  numerics. Existing randomly initialized model defaults retain their approximate GELU behavior.

## [2.1.0] - 2026-08-10

### Added

- Layout-aware `RotaryFactors` and canonical rotary generation/application helpers under `equimo.core.layers`.
- An experimental `equimo.timeseries` namespace with the raw T0/T0-alpha patch-transformer backbone, converted
  pretrained weights, and a documented tensor, scaling, padding, and output contract.
- Apache-2.0 licensing and upstream attribution for the T0-derived modules, alongside Equimo's MIT-licensed original
  code.

### Removed

- **Breaking:** raw `(sin, cos)` rotary plumbing (`get_sincos()` and `rope_sincos=`), the general `qk_transform`
  attention callback, and rotary helper re-exports from attention modules. Rotary producers now return `RotaryFactors`
  from `get_factors()`, and attention accepts `rotary=`.
- **Breaking:** unused fine-tuning API surface identified by a code audit (nothing removed was exercised by
  documentation, examples, or tests beyond its own definition): the method-profile layer (`profiles.py`,
  `MethodProfile`, `FineTunePlan.profile`), the `*Recipe` shells duplicating the real `*Config` classes (`LoRARecipe`,
  `DoRARecipe`, `AdapterRecipe`, `VPTShallowRecipe`, `VPTDeepRecipe`, `LinearProbeRecipe`, `LinearProbeConfig`,
  `FineTuneRecipe`, `HeadSpec`, `SAMMetadata`), the `equimo.finetune.integrations` package, the
  `equimo.finetune.tabular` package and other one-line aliases (`partition_for_training`, `locked_tower`,
  `linear_probe`, `attention_pool_probe`, `adapter_transformer_strong`, `merge_and_save`), the unused
  `MergeMethod`/`MergePlan`/`KnOTSMerging` and `PEFTModuleMixin` protocols, and `CompactLeafMap`,
  `TokenClassificationHead`, `TokenIndexPool`, `is_layer_norm`. Method fidelity notes moved to
  `docs/finetuning/method_defaults.md`.
- Constructor parameters that were accepted but silently ignored: `C3(n)`, `FasterNetBlock(kernel_size, padding)`,
  `DoubleConvBlock(norm_max_group)`, `LinearAngularAttention(qk_norm)`, `RFAttentionBlock(residual_mbconv)`,
  `ConvAttention(norm_kwargs)`, `Vssd(d_state, d_conv)`, and `interpolate_antialias` on `FasterViT`/`PartialFormer`.
- `equimo.catalog`'s runtime descriptor-validation framework and `create_model` remain; the authoring contract is now
  enforced by tests.

### Changed

- **Breaking:** `save_delta` and `load_delta` accept a single, documented call order:
  `save_delta(model, path, *, base_model=None, spec=None, ...)` and `load_delta(base_model, path_or_bundle)`. The
  on-disk bundle format is unchanged.
- `replace_head` always validates the replacement head; the `validate_shape` and `preserve_old_head_metadata` flags are
  removed.
- Internal consolidation with no behavioral or checkpoint impact: shared registry factories, model-variant factory,
  drop-path and layer-scale helpers, PEFT micro-helpers and wrapper walkers, entry-based delta codec drivers,
  feature-forward mixins, and shared ViT/Parcae embedding builders. Saved-checkpoint compatibility is guarded by a new
  structure-signature test (`tests/test_checkpoint_signature_stability.py`).
- Core attention accepts leading batch axes and optional Q/K transforms. Nonzero masks identify allowed pairs, and fully
  masked query rows now contribute zero attention probability mass.

### Fixed

- Corrected ViT-5 and direct vision RoPE feature pairing, T0 XPos low-precision dtypes, rectangular MLLA RoPE grids, and
  TabPFN custom `rope_base` isolation from Equinox's theta-blind global factor cache.
- Deterministic built-in model and layer paths no longer stage dead JAX PRNG operations when `inference=True`, enabling
  direct export through converters that do not lower randomness primitives while preserving training key schedules.
  VisionParcae's explicitly stochastic state initializers remain key-dependent; `state_init="zero"` provides its
  PRNG-free export path.

## [2.0.0] - 2026-07-15

### Added

- Public modality namespaces for vision, language, audio, and tabular models, with shared layers under `equimo.core`.
- The `equimo.finetune` subsystem, including trainability plans, PEFT methods, feature extraction, calibration, merging,
  and portable delta bundles.
- Versioned model-checkpoint metadata with model-structure and SHA-256 validation. Newly written model, delta, and
  calibration archives are staged atomically and read with explicit resource limits.
- An experimental, read-only model catalog in `equimo.catalog`.

### Changed

- Python support is 3.12 through 3.14. The declared JAX and Equinox minima now match the effective requirements of the
  runtime dependency graph.
- Model loading uses a constructed model plus `equimo.serialization.load_weights`. The built-in Hugging Face repository
  points to an immutable revision, and every uploaded archive is verified against its pinned Git LFS SHA-256 digest.
- Fine-tuning paths now escape dots, backslashes, and integer-looking string keys, keeping mapping keys distinct from
  sequence indices.
- Language pooling and positional/mask operations preserve low-precision model dtypes while accumulating sensitive
  reductions in float32.

### Fixed

- Corrected the state-space duality scan recurrence, chunk transitions, and output dtype.
- Corrected the DINO projection head by removing the activation after its final linear layer.
- Corrected convolutional patch-embedding output shapes for odd image sizes.
- Made `DropPath` and `DropPathAdd` accept scalar JAX probabilities and handle a probability of one deterministically.
- Preserved unknown token ID zero instead of treating every zero token as padding.
- Forwarded `key` and `inference` independently through feature extractors and heads.
- Preserved unmatched destination parameters during non-strict PyTorch conversion and handled CPU, conjugate/negative,
  bfloat16, and destination dtype conversion safely.
- Made AdaLoRA, LoRA-FA, FourierFT, and ordinary trainable leaves round-trip through fine-tuning bundles; merge,
  unmerge, and strip now cover the complete supported LoRA family.
- Rejected non-finite calibration statistics.

### Security and reliability

- Archive extraction no longer replaces a caller-owned sibling directory.
- Model, delta, and calibration readers reject duplicate, unexpected, non-regular, oversized, truncated, or
  checksum-mismatched members.
- Archive extraction locks work on both POSIX and Windows.

### Removed

- The v1 top-level `equimo.models`, `equimo.layers`, `equimo.io`, `equimo.implicit`, and `equimo.experimental` layouts.
- The v1 metadata-driven `load_model` constructor. Equimo v1 archives are not part of the v2 compatibility contract.

Existing model archives uploaded for the v2 alpha releases remain loadable and do not need to be regenerated. See the
[v2 migration guide](docs/migration-v2.md) and [stability policy](docs/stability.md).
