# equimo Blueprint

## System Overview

`equimo` is the installable package of a JAX/Equinox research library for multimodal models. The package root holds the
infrastructure that every modality shares: a modality-scoped model-class registry, an experimental model-variant
catalog, digest-pinned checkpoint save/download/load with resource limits, and general model and visualization helpers.
`__init__.py` defines only `__version__`, so callers import from submodules (`equimo.serialization`,
`equimo.vision.models`, ...).

The subpackages layer as follows:

- `core` is the shared base: layers with string-keyed registries, numerical kernels, PRNG helpers that allow key-free
  inference, the config-registry model factory, and Deep Equilibrium layers.
- `vision`, `language`, `audio`, `tabular` and `timeseries` are modalities. Each one builds models from `core` layers
  plus its own layers, and registers them in the root registry under its modality name.
- `finetune` operates on any of those models: trainable selection, PEFT wrappers, heads, merging, and delta
  serialization.
- `conversion` ports PyTorch weights into Equinox models for the maintainer scripts in the repository's `models/`
  directory.

Pretrained loading always flows the same way: a modality factory calls `core/factory.py:build_model_variant`, which
calls `serialization.py:load_weights`, which downloads into `~/.cache/equimo/` when needed.

## Local Interface & Invariants

### Interface

- `[__init__.py:__version__]`: the package version string. `save_model` records it as `equimo_version` in checkpoint
  metadata.
- `[registry.py:register_model]`: `register_model(name=None, force=False, modality=None)`, a class decorator that stores
  an `eqx.Module` subclass under a lowercased `(name, modality)` key. Every modality's model classes use it.
- `[registry.py:get_model_cls]`: `get_model_cls(cls, modality=None)` resolves a registry key to a class. Non-string
  values pass through unchanged.
- `[catalog.py:__all__]` (experimental): `MetadataStatus`, the frozen dataclasses `ModelInput`, `PretrainedWeights`,
  `ModelProvenance` and `ModelVariant` (each with `to_dict()`), `list_models(*, modality, family, pretrained)`,
  `model_info(key)` and `create_model(key, **kwargs)`. Variants come from the `_catalog_model_variants()` hooks of the
  modules listed in `[catalog.py:_CATALOG_PROVIDERS]` (ViT, AST, TabPFN, T0).
- `[serialization.py:save_model]`:
  `save_model(path, model, model_config, torch_hub_cfg=None, timm_cfg=None, compression=True) -> Path` writes a
  deterministic `.tar.lz4` archive, or a directory when `compression=False`.
- `[serialization.py:download]`:
  `download(identifier, repository, timeout=60, expected_sha256=None, *, limits=None) -> Path` fetches
  `{repository}/{model}/{identifier}.tar.lz4` into the cache.
- `[serialization.py:inspect_checkpoint]`: returns a `[serialization.py:CheckpointInfo]` (format, version, weights
  digest, model class and signature, Equimo version, verified/legacy flags) without loading weights.
- `[serialization.py:load_weights]`: loads a checkpoint (by identifier from `DEFAULT_REPOSITORY_URL`, or by local path)
  into a model template, with optional expected digests, expected config and `CheckpointLimits`. It returns
  `eqx.nn.inference_mode(model, inference_mode)` (default `inference_mode=True`), so the flag is set in both directions.
- `[serialization.py:DEFAULT_REPOSITORY_URL]`: the Hugging Face `poiretclement/equimo` repository, pinned to
  `[_pretrained.py:DEFAULT_REPOSITORY_REVISION]`.
- `[utils.py]`: `normalize`, `make_divisible`, `make_2tuple`, `to_list`, `nearest_power_of_2_divisor`,
  `make_drop_path_schedule`, `pool_sd` (jitted token pooling), `count_params`, `cost_analysis`, `PCAVisualizer` and
  `plot_image_and_feature_map`. Model and layer modules across the modalities import these.
- Private modules that other packages import:
  - `[_checkpoint_limits.py:CheckpointLimits]`, `resolve_limits`, `LimitedReader`, `array_template` and
    `scan_array_stream` are used by `finetune/serialization.py` and `vision/pmt_checkpoint.py`.
  - `[_io.py:atomic_file]`, `sha256_file`, `sha256_stream`, `validate_sha256`, `read_limited` and `copy_limited` are
    used by `finetune/serialization.py`.
  - `[_pretrained.py:PRETRAINED_ARCHIVE_SHA256]` maps about 130 pretrained identifiers to archive digests. It is used by
    `serialization.download` and `vision/models/convnext.py`.
  - `[_legacy_layouts.py:legacy_leaves_with_path]`, `legacy_layout` and `restore_layout` describe pre-2.8 layouts. They
    are used by `serialization.py`, `finetune/peft/_compat.py` (bundle hashes) and `vision/_encoder_identity.py` (PMT
    encoder digests).

### Invariants

- `[registry.py:register_model]` raises `TypeError` for a class that is not an `eqx.Module` subclass, and `ValueError`
  for an already registered `(name, modality)` pair unless `force=True`.
- `[registry.py:get_model_cls]` raises `ValueError` for an unknown key, for an unknown modality on that key, and for an
  ambiguous key when `modality=None` and several modalities share the name.
- `[registry.py:_SAFE_IDENTIFIER_RE]` and `[serialization.py:_validate_identifier]` allow only `[a-zA-Z0-9_-]` in
  identifiers, which prevents path traversal and URL injection.
- `[catalog.py:_load_catalog]` raises `ValueError` on duplicate keys. `[catalog.py:_resolve_model]` accepts an exact key
  or a unique bare variant name, and otherwise raises `ValueError` with suggestions.
- Checkpoint format (`[serialization.py:_CHECKPOINT_FORMAT]`, `[serialization.py:_CHECKPOINT_FORMAT_VERSION]`):
  - an archive holds exactly `metadata.json` then `weights.eqx`, with format `equimo.model.checkpoint` version 1;
  - unexpected, duplicate, non-file, oversized and truncated members are rejected
    (`[serialization.py:_extract_model_archive_stream]`);
  - `inspect_checkpoint` rejects schema-less v2-alpha (legacy) checkpoints unless `allow_legacy=True`, while
    `load_weights` always allows them and emits a `RuntimeWarning` when neither the weights nor the archive checksum
    could be verified (`[serialization.py:_inspect_checkpoint_directory]`, `[serialization.py:load_weights]`);
  - a supplied model must match the recorded class (`module.qualname`) and signature (`sha256:` over the array keypaths
    and shapes in PyTree traversal order; `sort_keys=True` orders only the keys inside each entry)
    (`[serialization.py:_model_class]`, `[serialization.py:_model_signature]`);
  - checkpoints written before Equimo 2.8 also stored values that modules now compute (classes naming them in
    `_legacy_stored_leaves`, such as RoPE frequencies). When the signature (versioned) or the weight stream
    (schema-less) matches that older layout, `load_weights` reads it through `[_legacy_layouts.py:legacy_layout]` and
    drops the values, raising `ValueError` if one differs from the computed value by more than four machine epsilons of
    its dtype (`[serialization.py:_layout_for_signature]`, `[serialization.py:_layout_for_stream]`,
    `[_legacy_layouts.py:restore_layout]`). Hashes of older bundles and PMT artifacts include those values, recomputed
    on the CPU backend, whose results the published checkpoints match bit for bit
    (`[_legacy_layouts.py:_stored_value]`).
- Download trust (`[serialization.py:download]`):
  - the expected archive digest is `expected_sha256` if given, otherwise the pinned `PRETRAINED_ARCHIVE_SHA256` entry
    for the default repository, otherwise the server's `X-Linked-Etag`/`ETag` SHA-256;
  - a mismatch raises `ValueError`;
  - size is capped at `min(64 GiB, limits.max_archive_bytes)`, checked against both `Content-Length` and the streamed
    byte count;
  - TLS verification is always on.
- `[_checkpoint_limits.py:CheckpointLimits]` fields must be positive `int`s. `[_checkpoint_limits.py:scan_array_stream]`
  accepts only NPY header versions 1-3 and rejects Fortran order, object, structured and subarray dtypes. It enforces
  rank, dimension, per-tensor and total byte limits, and requires every stored array to match the template's shape and
  dtype with no trailing data.
- `[_io.py:validate_sha256]` requires exactly 64 hex characters. `[_io.py:read_limited]` and `[_io.py:copy_limited]`
  raise `ValueError` past `max_bytes`.
- `[serialization.py:load_weights]` stages the weights in a spooled temporary file and re-hashes them, raising
  `ValueError` if the bytes changed during loading.
- `[serialization.py:_resolve_weights_dir]` raises `ValueError` when both or neither of `identifier` and `path` are
  given, and when `expected_sha256` is combined with a directory or a path whose suffixes are not `.tar.lz4`.
- `[utils.py:pool_sd]` raises `ValueError` for an unknown `pool_type`, and for `cls_patch_mean` without prefix tokens.
  `[utils.py:PCAVisualizer]` and `[utils.py:plot_image_and_feature_map]` raise `ImportError` without the `extras` extra.

### Side effects

- `[serialization.py:download]` makes an HTTPS GET and writes `~/.cache/equimo/{model}/{identifier}.tar.lz4` plus a JSON
  `.sha256` sidecar. `[serialization.py:_verify_cached_download]` rewrites the sidecar and warns if the cache is
  read-only.
- `[serialization.py:_decompress_archive]` extracts to a sibling `<archive>.extracted/` directory with a `.complete`
  sentinel, under a file lock on `<archive>.extracted.lock`.
- `[_io.py:atomic_file]`/`[_io.py:atomic_directory]` write through hidden temporaries in the target's parent directory
  and commit with `os.replace`. `[serialization.py:save_model]` uses `atomic_directory` when `compression=False`. With
  `compression=True` it first writes `weights.eqx` and `metadata.json` to a system `tempfile.TemporaryDirectory()`, and
  only the final archive goes through `atomic_file`. `[_io.py:file_lock]` creates lock files and uses `flock` on POSIX
  or `msvcrt.locking` on Windows.
- `[serialization.py]` logs through `loguru`.
- `[registry.py:register_model]` mutates the global `_MODEL_REGISTRY`. `[registry.py:_import_builtin_models]`, called by
  `get_model_cls`, imports the modality `models` packages, which register their classes.
- `[catalog.py:_load_catalog]` and `[catalog.py:create_model]` import provider and factory modules dynamically, and
  `create_model(..., pretrained=True)` can download weights.
- `[utils.py:PCAVisualizer]` samples with the global NumPy RNG. `[utils.py:plot_image_and_feature_map]` writes a figure
  to `save_path`. `[utils.py:cost_analysis]` JIT-compiles the forward pass with a fixed `PRNGKey(42)`.

## Subdirectories & System Boundaries

- core: [./core/EGM.md](./core/EGM.md) - shared layers with string-keyed registries (attention with XLA and opt-in cuDNN
  paths, rotary, FFNs, norms, drop-path, Mamba2), float32-safe softmax/cross-entropy and SSM kernels, PRNG mode helpers,
  the EMA update, `build_model_variant`, and Deep Equilibrium layers.
- vision: [./vision/EGM.md](./vision/EGM.md) - registered image architectures (ViT family with DINOv2/v3, SigLIP2, TIPS,
  EUPE, LingBot and ViT-5 presets, ConvNeXt, EoMT, PMT, and others), vision layers and registries, query-segmentation
  training losses, image loading, segmentation post-processing, and PMT checkpoints.
- language: [./language/EGM.md](./language/EGM.md) - `TextTransformerEncoder` and the SentencePiece tokenizer, which
  downloads its tokenizer files.
- audio: [./audio/EGM.md](./audio/EGM.md) - the Audio Spectrogram Transformer and its layers, plus host-side,
  checkpoint-pinned waveform-to-fbank preprocessing.
- tabular: [./tabular/EGM.md](./tabular/EGM.md) - TabPFN v3 in-context classifier and regressor, and their layers.
- timeseries: [./timeseries/EGM.md](./timeseries/EGM.md) - the T0 patch-transformer quantile forecaster with its
  `predict` rollout adapter.
- finetune: [./finetune/EGM.md](./finetune/EGM.md) - fine-tuning plans, `TargetSpec` selection, PEFT wrappers, heads and
  probes, recipes, regularization, calibration, model merging, and delta and bundle serialization.
- conversion: [./conversion/EGM.md](./conversion/EGM.md) - PyTorch-to-Equinox weight conversion helpers.

## Global Invariants

- Modules are unbatched: models and layers process one example, and callers batch with `jax.vmap`. See the per-package
  Invariants in [./core/layers/EGM.md](./core/layers/EGM.md) and [./vision/models/EGM.md](./vision/models/EGM.md).
- Forward and feature endpoints take `key=None` and split keys with `core/_prng.py:split_for_mode`, so they run without
  a key only under static `inference=True` ([./core/EGM.md](./core/EGM.md)). The exceptions are masked-attention
  EoMT/PMT and `VisionParcae` ([./vision/models/EGM.md](./vision/models/EGM.md)), and the vision convolution blocks that
  fall back to a fixed key through `default_key_for_mode` ([./vision/layers/EGM.md](./vision/layers/EGM.md)).
- Numerically sensitive operations (attention logits, softmax, norms, scan exponents, pooling accumulators) compute in
  at least float32 and cast back to the input dtype. Token attention uses `jax.nn.softmax`, while other softmaxes and
  the classification losses use `equimo.core.ops` ([./core/ops/EGM.md](./core/ops/EGM.md)).
- Layers and models are resolved by lowercased string names through per-family registries built from
  `core/layers/_registry.py`, and models through `[registry.py:register_model]`. Registering a duplicate name raises
  unless `force=True`.
- Pretrained weights load only through `[serialization.py:load_weights]`, with digest pinning and `CheckpointLimits`.
  `finetune` bundles use their own format (`manifest.json` plus `arrays.eqx`, also never pickled) and the same limit and
  I/O helpers ([./finetune/EGM.md](./finetune/EGM.md)).

## Global Side Effects

- Network: the Hugging Face repository `poiretclement/equimo` at a pinned revision, for model archives
  (`[serialization.py:download]`) and tokenizer files (`language/tokenizers.py`). `conversion` may also call `torch.hub`
  or `timm`.
- Disk: the cache under `~/.cache/equimo/` (archives, sidecars, extracted directories, lock files, tokenizers), plus
  checkpoints and bundles written by `save_model`, `vision/pmt_checkpoint.py` and `finetune/serialization.py`.
- Import-time global state: importing modality packages fills the layer registries and the root model registry.
- Optional dependencies are imported lazily and raise `ImportError` that names the extra: `extras` (Pillow,
  scikit-learn, matplotlib), `audio` (torch, torchaudio), `language` (tensorflow, tensorflow-text) and `torch` (torch,
  timm).
- Cross-device collectives: `vision/query_training` (`psum`, `pmin`) and PMT's `ReferenceBatchNorm` (`psum`) when an
  `axis_name` is given.

## System Verification

- Run: `devenv shell -- uv run pytest -q`
- Focused root-module tests:
  `devenv shell -- uv run pytest tests/test_io.py tests/test_checkpoint_expectations.py tests/test_checkpoint_signature_stability.py tests/test_model_catalog.py tests/test_pretrained_coverage.py -q`
