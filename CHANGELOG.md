# Changelog

All notable changes to Equimo are documented here. Equimo follows Semantic Versioning from version 2.0.0 onward.

## [Unreleased]

### Added

- `equimo.core.parameters` describes how optimizers should treat each leaf, for every training workflow:
  `trainable_mask` (an Equinox filter excluding buffers), `weight_decay_mask` (also usable as an Optax `mask`),
  `DEFAULT_NO_DECAY_TAGS`, `buffer_mask`, `module_tags`, and the field markers `buffer()` and `tagged(...)`. The path
  tagger moved to `equimo.core.tags`; `equimo.finetune` and `equimo.finetune.tags` re-export the same functions.

### Changed

- RoPE and Swin position embeddings compute the values that follow from their configuration (`VisionRoPE.freqs`,
  `DinoRoPE.periods`, `RoPE.rotations`, and the `PosEmbMLPSwinv1D`/`PosEmbMLPSwinv2D` coordinate and index tables) when
  used instead of storing them as PyTree leaves, so optimizers no longer see them and weight decay can no longer change
  them. They remain readable as attributes but cannot be replaced with `eqx.tree_at`; set them through constructor
  arguments. The structural signatures of models using them change. Checkpoints, fine-tuning bundles, and PMT artifacts
  saved by earlier versions still load; `load_weights` raises `ValueError` when a stored value differs from the computed
  one by more than rounding, for example after weight decay, and `load_delta` raises `FineTuneBundleError` for a bundle
  whose trained delta changed such a value. Earlier versions cannot read checkpoints, bundles, or PMT artifacts that 2.8
  saves for these models. The value hashes that identify a model (fine-tuning bundle base-checkpoint hashes, merge
  checks, PMT encoder digests) cover the static fields of these modules instead, such as RoPE `base` and
  `periods_dtype`, so a base model that differs only in those settings is rejected.
- Frozen PEFT arrays are declared buffers and excluded from every trainability mode: VeRA `vera_A`/`vera_B`, LoRA-FA
  `frozen_A`/`correction_matrix`, RandLoRA bases, PiSSA `base_weight_delta`, OFT `unmerged_weight`, and a fixed
  AdaptFormer `scale`. `mode="full"` previously trained the VeRA bases, the LoRA-FA correction matrix, and the OFT
  backup weight.
- The default no-weight-decay policy (`LLRDConfig.no_weight_decay_tags`, `DEFAULT_NO_DECAY_TAGS`) also excludes layer
  scales (`block.layer_scale`, including PMD lateral scales) and state-space dynamics (`ssm`: Mamba `A_log`, `D`,
  `dt_bias`; Parcae `A_log`, `dt_bias`, `B`, `C`). Leaves of registered normalization layers are tagged `norm` whatever
  their field name, including norms inside `eqx.nn.Sequential`, and leaves named `*_bias` are tagged `bias`. The path
  tagger also tags the Swin-v2 continuous position-bias MLP (`pos_emb_funct`, used by FasterViT HAT blocks)
  `embedding.position`. These tags change tag-based selection (`TargetSpec` tags, `train_norm`, `train_bias`, and the
  `norm`, `head_plus_norm`, `bias`, and surgical modes), parameter roles, plan labels, and plan fingerprints.
  `CANONICAL_TAGS` gains `ssm`, `buffer`, and `vector`.
- The path tagger tags arrays with at most one axis `vector`, and the default no-weight-decay policy excludes them, as
  timm, MAE, and OpenCLIP do. With the tags above, built-in backbones need no `vector` tag; PEFT and probe-head vectors
  no longer decay: DoRA magnitudes, AdaLoRA singular values, VeRA and RandLoRA scales, IA3 vectors, FourierFT
  coefficients, scale/shift parameters, adapter residual scales, and the `AttentionPool` and
  `AttentionPoolingClassifierHead` query vectors. Their plan labels change from `*_decay` to `*_no_decay`.
- Position-embedding tables are evaluated while tracing, so `jax.jit` programs and ONNX exports hold them as constants.
  MLLA attention rotates queries and keys with one RoPE call, so its programs hold one table per block. Exporting `RoPE`
  (MLLA) or the Swin tables to ONNX needs a jax2onnx release that includes enpasos/jax2onnx#291; jax2onnx 0.17.1 and
  0.18.0 fail on them.
- `VisionParcae.weight_decay_mask()` follows the default policy, so it also leaves biases, norms, layer scales, vectors,
  and position embeddings undecayed.
- `update_ema_model` averages only trainable leaves; integer arrays and buffers take the current model's values.
- `l2_sp_loss`, `adapter_norm_loss`, and `task_vector_norm_loss` skip buffers.
- `VisionTransformer` raises `TypeError` for unknown constructor arguments instead of ignoring them.
- `DropPath` keeps or drops the whole unbatched input with one draw, matching `DropPathAdd`; it previously dropped rows
  of the leading axis (tokens or channels) independently.
- `PromptedModel.__call__` raises `ValueError` when the base model has no `head`; use `features()` for headless models.
- `apply_adalora` raises `ValueError` for a nonzero `dropout`, which AdaLoRA wrappers cannot apply.
- `load_image` raises `ValueError` when only one of `mean` and `std` is given.
- Tokenizer files are cached under a directory derived from the full URL, so pinned revisions no longer share a cache
  entry; existing tokenizers are downloaded once more.

### Fixed

- `MllaBlock` (used by `Mlla` and `Vssd`) applies its output projection to the gated attention output and keeps the
  first convolutional position encoding in the residual stream, as in MLLA and VSSD. Outputs of these models change.
- EUPE ViTs use the DINOv3 patch RoPE configuration, including training-time coordinate rescaling, which their config
  previously set under an ignored key. Inference outputs are unchanged.
- EUPE ViTs compute their RoPE periods in bfloat16, as the EUPE checkpoints store them. Pretrained outputs are
  unchanged; models built with `pretrained=False` previously used float32 periods.
- LoRA-FA with `custom_vjp=False`, RandLoRA, and AdaptFormer with a fixed scale stop gradients to their frozen arrays.
- `GroupSpec.tags` holds the tags shared by every leaf of the group instead of the first leaf's tags, so tag-based
  optimizer rules, such as Rollfast's `GroupRule(tag=...)`, no longer depend on which leaf a group happened to start
  with.
- `apply_lora_rank_pattern` keeps the `metadata` of LoRA wrappers, such as PiSSA's initialization record.
- `apply_adalora` honors `scaling="alpha_over_sqrt_r"`.
- The `dynamictanh` norm (`DyT`) can be used as a block `norm_layer`.
- Adapters, AdaptFormer, scale/shift, and `AttentionPool` return the activation dtype instead of promoting bfloat16 to
  float32.
- `save_delta` without `base_model` accepts a `TrainableSpec(mode="peft")` that trains no head, norm, or bias.
- `SentencePieceTokenizer` checks for the `language` extra before downloading, writes downloads atomically, and times
  out after 60 seconds.
- PyTorch conversion renames only whole `scale` and `kernel` path components, keeps dictionary keys in parameter names,
  and rejects an unknown `source` even when a `torch_model` is given.

## [2.7.0] - 2026-10-26

### Added

- `equimo.core.ops` exports `softmax`, `log_softmax`, `softmax_cross_entropy_with_integer_labels`, and
  `softmax_cross_entropy_with_logit_targets`, following "A Fix for Softmax and Cross-Entropy". They compute in at least
  float32, return the input dtype, and keep the dominant logit's small gradient at high confidence. The logit-target
  loss, for distillation, also preserves near-equality gradients, stops target gradients, and supports higher-order
  derivatives.
- Experimental fused attention: `equimo.core.layers.with_attention_implementation(model, "cudnn")` runs core,
  prefix-tuned, and tabular attention through cuDNN's fused kernel for bfloat16 or float16 models on CUDA GPUs, and
  attention modules accept the same choice as an `implementation` argument. On an RTX 4080, it shortened ViT and EoMT
  training steps by 7-22% and reduced their temporary memory by 17-56%. It does not support attention dropout or
  jax2onnx export; the default `"xla"` implementation is unchanged.
- `semantic_crop_origins` places the fewest evenly spaced crops that cover a scaled image, for sliding-window evaluation
  with `merge_semantic_crops`. With the shorter side scaled to the crop size, it matches the EoMT reference's windows,
  except that its last crop always ends at the image edge: the reference's floating-point spacing stops a pixel short
  for some sizes, such as a 1,347-pixel side with 448-pixel crops.

### Changed

- Equimo requires JAX and jaxlib 0.11.1 or newer. With it, a compiled step that captures both a frozen encoder and its
  images as constants matches eager execution, which JAX 0.8.1 did not.
- Attention pooling, attention-pooling heads, adapter fusion, the tabular decoder, and the `"softmax"` activation
  compute softmax in at least float32 with the corrected derivative of `equimo.core.ops.softmax`, returning the model
  dtype. Tabular attention computes its softmax in at least float32. Attention logits are cast before scaling, avoiding
  a second bfloat16 rounding when the square root of the head dimension is not a power of two. Float32 forward results
  and checkpoint structure are unchanged.

### Fixed

- Query classification uses stable cross-entropy, so confident correct predictions retain nonzero losses and target
  gradients.
- `selective_scan` returns the input dtype. Its float32 scan state leaked into bfloat16 outputs, unlike `ssd`.
- `merge_semantic_crops` raises a `ValueError` when the crops leave part of the scaled image uncovered. It returned NaN
  scores there instead, and any final resize spread them to every output pixel, so the argmax labelled the whole image
  class 0.

## [2.6.0] - 2026-10-17

### Fixed

- `anneal_mask_state` and `anneal_pmt_mask_state` return exactly zero mask probability from the declared end step. On
  GPUs, the float32 progress quotient could round just below one (600 / 600 is 0.99999994 on an H100), which left a
  probability near 3e-7 at that step and made `mask_free_eomt` and `mask_free_pmt` refuse a model whose annealing had
  ended.
- Masked attention in EoMT and PMT keeps each query mask with exactly the annealed probability. A query kept its mask
  when its uniform draw was at most the probability, so at probability zero a draw of exactly 0.0 (about one in 8.4
  million per query) still applied it. Only draws exactly equal to the probability change outcome.

## [2.5.2] - 2026-10-16

### Fixed

- The EoMT and PMT mask-head `ScaleBlock` accumulates its upscaling matmul in the input precision when that exceeds
  float32. In 2.5.1, float64 models computed the upscale in float32 on CPU and failed on GPU, where cuBLAS has no
  float64-input, float32-output GEMM.

## [2.5.1] - 2026-10-16

### Changed

- The EoMT and PMT mask-head `ScaleBlock` computes its 2 × 2, stride-2 transposed convolution as a matmul followed by
  depth-to-space. This avoids an untuned cuDNN fallback for the bfloat16 weight gradient that made GPU training steps
  several times slower at batch sizes of 12 and above. Parameters and checkpoints are unchanged; outputs match the
  previous implementation within floating-point rounding.

### Fixed

- `match_queries` and the query-segmentation losses no longer rely on dropping out-of-range scatter updates. Under
  `vmap` with `--xla_gpu_deterministic_ops` or `--xla_gpu_exclude_nondeterministic_ops`, jaxlib 0.11.2 and earlier write
  such updates into the next batch row (openxla/xla#49380), which silently produced non-optimal query assignments and
  wrong classification targets.

## [2.5.0] - 2026-10-15

### Changed

- Model forward and feature endpoints default `key` to `None` instead of a fixed `jax.random.PRNGKey(42)`. Deterministic
  inference (`inference=True`) needs no key; a stochastic forward pass without a key raises a `ValueError` instead of
  silently reusing the same randomness on every call.

### Fixed

- `extract_features`, `DenseProbe`, and the other fine-tuning wrappers no longer fail with their default `key=None` on
  ViT-family models (including DINOv2, DINOv3, SigLIP2, EUPE, and LingBot-Vision), VisionParcae, MobileNetV3, AST, the
  text encoder, TabPFN, and T0, whose feature endpoints required a PRNG key even in inference mode.
- VisionParcae with `state_init="zero"` runs deterministic inference without a key. Key-dependent configurations,
  including the default `state_init="like-init"`, raise an actionable error when the key is missing.

## [2.4.0] - 2026-09-24

### Added

- JAX-native query-segmentation assignment, validity-aware point sampling, class/mask/Dice losses, auxiliary-output
  composition, and inspectable batch/device reductions, with pinned upstream numerical references.
- Padding-aware PMT BatchNorm and configurable collective axes for synchronized statistics, preserving default model
  checkpoint layouts.

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

### Fixed

- `MBConv` no longer fails when its pre-pointwise activation is `None`; a missing activation is applied as identity.

## [2.2.0] - 2026-09-05

### Added

- Expanded ConvNeXt model factories, including ConvNeXt V2 with Global Response Normalization, RMS-normalized variants,
  and overlapping stems.
- Pretrained weights for 66 ConvNeXt checkpoints: 40 V1 and 26 V2 variants, including FCMAE backbones and fine-tuned
  classifiers. Public factories support `pretrained=True` and named upstream checkpoint variants.
- Trusted timm reference fixtures for ConvNeXt Atto, ConvNeXt Zepto RMS with an overlapping stem, and ConvNeXt V2 Atto.
  Feature and classifier-logit parity runs through the same reference suite as DINOv3, with elementwise `atol=1e-4` and
  `rtol=1e-5`; direct Torch conversion tests also cover V2 blocks with overlapping stems.
- Bundled upstream license snapshots, a pretrained-model license index, and conversion attribution notices in source and
  wheel distributions. ConvNeXt V2 weights retain their upstream CC BY-NC 4.0 terms. T0 source provenance is documented
  separately from its checkpoint license.
- A reproducible CPU performance benchmark and [performance audit](docs/performance.md).

### Changed

- Pinned the pretrained Hugging Face repository to `6916a4a3d460cfa804419454baf3d153885f3b6a` and registered trusted
  SHA-256 digests for all new ConvNeXt archives.
- ConvNeXt conversion now uses pinned upstream revisions, strict parameter mapping, native-resolution feature and logit
  comparisons for two deterministic inputs, and an exact archive-reload check. The checked-in conversion report records
  source revisions, archive digests, and measured errors for all 66 checkpoints.
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
