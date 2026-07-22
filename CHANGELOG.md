# Changelog

All notable changes to Equimo are documented here. Equimo follows Semantic
Versioning from version 2.0.0 onward.

## [2.0.0] - 2026-07-15

### Added

- Public modality namespaces for vision, language, audio, and tabular models,
  with shared layers under `equimo.core`.
- The `equimo.finetune` subsystem, including trainability plans, PEFT methods,
  feature extraction, calibration, merging, and portable delta bundles.
- Versioned model-checkpoint metadata with model-structure and SHA-256
  validation. Newly written model, delta, and calibration archives are staged
  atomically and read with explicit resource limits.
- An experimental, read-only model catalog in `equimo.catalog`.

### Changed

- Python support is 3.12 through 3.14. The declared JAX and Equinox minima now
  match the effective requirements of the runtime dependency graph.
- Model loading uses a constructed model plus `equimo.serialization.load_weights`.
  The built-in Hugging Face repository points to an immutable revision, and
  every uploaded archive is verified against its pinned Git LFS SHA-256 digest.
- Fine-tuning paths now escape dots, backslashes, and integer-looking string
  keys, keeping mapping keys distinct from sequence indices.
- Language pooling and positional/mask operations preserve low-precision model
  dtypes while accumulating sensitive reductions in float32.

### Fixed

- Corrected the state-space duality scan recurrence, chunk transitions, and
  output dtype.
- Corrected the DINO projection head by removing the activation after its final
  linear layer.
- Corrected convolutional patch-embedding output shapes for odd image sizes.
- Made `DropPath` and `DropPathAdd` accept scalar JAX probabilities and handle
  a probability of one deterministically.
- Preserved unknown token ID zero instead of treating every zero token as
  padding.
- Forwarded `key` and `inference` independently through feature extractors and
  heads.
- Preserved unmatched destination parameters during non-strict PyTorch
  conversion and handled CPU, conjugate/negative, bfloat16, and destination
  dtype conversion safely.
- Made AdaLoRA, LoRA-FA, FourierFT, and ordinary trainable leaves round-trip
  through fine-tuning bundles; merge, unmerge, and strip now cover the complete
  supported LoRA family.
- Rejected non-finite calibration statistics.

### Security and reliability

- Archive extraction no longer replaces a caller-owned sibling directory.
- Model, delta, and calibration readers reject duplicate, unexpected,
  non-regular, oversized, truncated, or checksum-mismatched members.
- Archive extraction locks work on both POSIX and Windows.

### Removed

- The v1 top-level `equimo.models`, `equimo.layers`, `equimo.io`,
  `equimo.implicit`, and `equimo.experimental` layouts.
- The v1 metadata-driven `load_model` constructor. Equimo v1 archives are not
  part of the v2 compatibility contract.

Existing model archives uploaded for the v2 alpha releases remain loadable and
do not need to be regenerated. See the [v2 migration guide](docs/migration-v2.md)
and [stability policy](docs/stability.md).
