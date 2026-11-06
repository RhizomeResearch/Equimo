# equimo.finetune

## Overview

Equinox-native fine-tuning scaffolding for Equimo models. The package turns a model into a `FineTunePlan`: a
trainable/frozen partition with optimizer-group labels, layer-wise learning-rate decay metadata, stable parameter
identities, and a fingerprinted report. Around that core it provides semantic parameter tagging and `TargetSpec`
selection, feature extraction and probes, task heads and pooling, model surgery, recipe presets, regularization losses,
calibration statistics, model merging, and a safe on-disk format for PEFT deltas and calibration artifacts. PEFT
wrappers live in `peft`; modality presets live in `audio`, `language`, and `vision`. The package does not ship an
optimizer or training loop: plans expose labels and masks for the caller's optimizer.

## Interface

### Planning, selection, and inspection

- Config and schema types: `TargetSpec`, `TrainableSpec` (modes `frozen`, `head`, `head_plus_norm`, `norm`, `bias`,
  `scale_shift`, `partial`, `surgical`, `full`, `peft`), `StatePolicy`, `WeightLayout`, `ProjectionSegment`,
  `ParamIdentity`, `FeatureSpec`, `CalibrationArtifact`, `AuxLossSpec`, `ModelLineage`, `LLRDConfig` (presets `uniform`,
  `vit_base`, `vit_large_or_huge`, `audio_transformer`), `ParamInfo`, `GroupSpec`, `ParameterReport`, `TrainableReport`,
  `FineTunePlan` (`combine()`), `FineTuneBundle`, `FineTuneBundleError` [config.py:TargetSpec] [config.py:TrainableSpec]
  [config.py:FeatureSpec] [config.py:LLRDConfig] [config.py:FineTunePlan] [config.py:FineTuneBundle]
- Typing aliases `Path` (from `equimo.core.tags`), `PyTree`, `FilterSpec`, `LeafPredicate` [../core/tags.py:Path]
  [\_typing.py:PyTree]
- Paths: `key_path_to_path`, `path_to_str`, `str_to_path`, `is_path_prefix`, `iter_param_leaves`, `iter_param_paths`,
  `extract_param_paths`, `make_path_tree`, `make_param_info_tree` [paths.py:path_to_str] [paths.py:str_to_path]
- Tags: `CANONICAL_TAGS`, `Tagger`, `canonical_tags_for_path`, `infer_depth` (re-exported from `equimo.core.tags`),
  `make_tag_tree`, `iter_param_infos`, `make_param_info`, `infer_role`, `merge_taggers`
  [../core/tags.py:canonical_tags_for_path] [tags.py:make_param_info]
- Selection: `resolve_target(model, target, *, allow_empty=False, tagger)`, `resolve_target_paths`, `is_linear`
  [selectors.py:resolve_target]
- Masks and labels: `make_trainable_filter`, `resolve_trainable_paths`, `make_param_labels`,
  `make_labeled_param_info_tree` [masks.py:make_trainable_filter] [masks.py:resolve_trainable_paths]
  [labels.py:make_param_labels] [labels.py:make_labeled_param_info_tree]
- Surgery: `prepare_finetune(model, *, trainable, labels=None, model_state=None, ...) -> FineTunePlan`, `replace_head`,
  `transfer_head`, `set_stochastic_depth_rate`, `disable_stochastic_depth`, `set_dropout_rate`, `disable_dropout`,
  `extract_subtree` [surgery.py:prepare_finetune] [surgery.py:replace_head]
- Inspection: `inspect_trainables`, `inspect_plan`, `make_trainable_report`,
  `validate_plan(plan, *, expected_fingerprint=None)` [inspection.py:make_trainable_report]
  [inspection.py:validate_plan]

### Features, heads, pooling, and recipes

- Feature extraction: `extract_features(model, *args, pool="auto", feature_spec=None, key=None, inference=True, ...)`,
  `FeatureLevelMetadata`, `FeatureResult`, `FeatureExtractor`, `LinearProbe`, `AttentionPoolingProbe`,
  `make_linear_probe`, `make_attention_pool_probe`, `make_attention_pool_input_from_forward_features`,
  `make_attention_pool_input_from_intermediates` [feature_extraction.py:extract_features]
  [feature_extraction.py:make_linear_probe] [feature_extraction.py:AttentionPoolingProbe]
- Heads: `IdentityHead`, `LinearHead`, `LayerNormReadoutHead`, `MultiLabelHead`, `MLPHead`,
  `AttentionPoolingClassifierHead`, `ProjectionHead`, `ContrastiveProjectionHead`, `CTCHead`, `DenseFeatureAdapter`
  [heads.py:LinearHead] [heads.py:AttentionPoolingClassifierHead]
- Pooling: `PoolName`, `CLSPool`, `MeanTokenPool`, `MeanPatchPool`, `CLSPatchMeanPool`, `MeanFramePool`,
  `AttentionPool`, `GeMPool`, `LastTokenPool`, `GlobalAveragePool`, `IdentityPool`, `pool_features`
  [pooling.py:pool_features] [pooling.py:AttentionPool]
- Recipes: `HeadPlusNormConfig`, `PartialUnfreezeConfig`, `StagePolicy`, `FineTuneStage`, `LPFTRecipe`, `lpft`,
  `prepare_lpft_stage1_model`, `prepare_lpft_stage2_model`, `head_plus_norm`, `full_ft_llrd`,
  `partial_ft_last_k_blocks`, `partial_unfreeze`, `adapter_transformer`, `adaptformer_transformer`, `lora_transformer`,
  `lora_transformer_all_linear`, `vpt_deep`, `task_adapter_bank` [recipes.py:LPFTRecipe] [recipes.py:full_ft_llrd]
  [recipes.py:task_adapter_bank]
- Continued self-supervised adaptation: `ContinuedSSLAdaptationConfig`, `SupervisedAfterSSLConfig`, `ContinuedSSLPlan`,
  `continued_ssl_trainable_spec`, `supervised_after_ssl_trainable_spec`, `continued_ssl_adaptation`,
  `supervised_after_ssl` [continued_ssl.py:continued_ssl_adaptation] [continued_ssl.py:supervised_after_ssl]

### Losses, calibration, merging, and serialization

- Regularization: `L2SPConfig`, `FeatureDistillationConfig` (`dense()` preset), `DELTAAttentionConfig`, `EWCConfig`,
  `MixoutConfig`, `l2_sp_loss`, `adapter_norm_loss`, `adalora_orthogonality_loss`,
  `adalora_orthogonality_aux_loss_spec`, `delta_attention_aux_loss_spec`, `task_vector_norm_loss`,
  `delta_attention_loss`, `ewc_loss`, `feature_distillation_loss`, `select_feature_taps`,
  `feature_distillation_loss_from_taps`, `mixout_leaf`, `mixout_tree` [regularization.py:l2_sp_loss]
  [regularization.py:mixout_tree]
- Calibration: `CalibrationCollectorState`, `initialize_calibration_collector`, `update_calibration_collector`,
  `combine_calibration_collectors`, `finalize_calibration_collector`, `validate_calibration_artifacts`,
  `input_covariance_from_artifact` [calibration.py:initialize_calibration_collector]
  [calibration.py:finalize_calibration_collector]
- Merging: `WiSEFTConfig`, `UniformSoupConfig`, `GreedySoupConfig`, `TaskVectorConfig`, `TaskVector`, `TIESConfig`,
  `DARETransform`, `BreadcrumbsConfig`, `FisherMergeConfig`, `RegMeanConfig`, `KnOTSConfig`, `interpolate_models`,
  `uniform_soup`, `greedy_soup`, `task_vector`, `apply_task_vector`, `ties_merge`, `dare_task_vector`,
  `breadcrumbs_task_vector`, `fisher_merge`, `regmean_merge`, `knots_task_vector`, `task_vector_bundle`
  [merging.py:task_vector] [merging.py:ties_merge] [merging.py:regmean_merge]
- Serialization: `save_delta(model, path, *, base_model=None, spec=None, method="lora", ...)`, `load_delta`,
  `save_finetune_bundle`, `load_finetune_bundle`, `save_calibration_artifacts`, `load_calibration_artifacts`
  [serialization.py:save_delta] [serialization.py:load_delta] [serialization.py:load_finetune_bundle]
- Package surface: `__init__.py` eagerly re-exports the symbols above and the PEFT exports, except
  `make_trainable_report`, `resolve_trainable_paths`, `make_labeled_param_info_tree`, `key_path_to_path`,
  `is_path_prefix`, `Tagger`, `infer_role`, `make_param_info`, `merge_taggers`, and `PoolName`, which are public only
  through their modules. `__all__` lists the `audio`, `language`, `recipes`, and `vision` submodules; `peft` is
  importable as `equimo.finetune.peft` but is not in `__all__` [\_\_init\_\_.py:\_\_all\_\_]

### Subpackages

- [peft](./peft/EGM.md): PEFT wrapper modules (LoRA family, DoRA, adapters, OFT/BOFT, IA3, VeRA, scale/shift, prefix and
  prompt tuning, BitFit) with apply/merge/strip and delta extract/load
- [audio](./audio/EGM.md): audio semantic tags and AST-style probe, head, adapter, and LoRA presets
- [language](./language/EGM.md): text semantic tags and LoRA, prefix, and projection-head presets
- [vision](./vision/EGM.md): dense patch probe, dense adapter and distillation presets, and EoMT/PMT fine-tuning plans

## Invariants

- Only inexact (floating) array leaves are parameters; trainable masks are `False` and labels `None` for every other
  leaf [masks.py:make_trainable_filter] [labels.py:make_labeled_param_info_tree]
- Trainable paths are resolved from the mode, then `train_head`/`train_norm`/`train_bias`, then wrapper bases with
  `train_base=True` in `peft` mode, minus `freeze` targets and leaves tagged `peft.metadata` or `buffer` (fields
  declared with `equimo.core.parameters.buffer`, such as VeRA bases or a fixed AdaptFormer `scale`), in every mode. An
  empty selection raises `ValueError` unless the mode is `frozen`; `TrainableSpec.expected_logical_ids` must match the
  selection exactly [masks.py:resolve_trainable_paths] [masks.py:\_base_trainable_paths]
- `ParamInfo` records built from a model (`iter_param_infos`, `make_tag_tree`, selectors, masks, labels) combine path
  tags with the tags modules give a leaf (`equimo.core.parameters.module_tags`): `norm` for leaves of registered
  normalization layers, `block.layer_scale` and `ssm` from field declarations, and `buffer`. `make_param_info` adds
  module tags only through its `module_tags` argument [tags.py:\_model_param_info] [tags.py:make_param_info]
- `LLRDConfig.no_weight_decay_tags` defaults to `equimo.core.parameters.DEFAULT_NO_DECAY_TAGS`, the same policy as
  `equimo.core.parameters.weight_decay_mask` for the leaves a plan trains. `make_param_labels` without `trainable_paths`
  treats every floating leaf except buffers as trainable and gives buffers the label `None` [config.py:LLRDConfig]
  [labels.py:make_labeled_param_info_tree]
- `TargetSpec` resolution is the union of `tags_all`, `tags_any`, `include`, and `predicate` (all inexact leaves when
  none is set), filtered by depth bounds and `exclude`. `predicate` is also called on whole `eqx.nn` `Linear`,
  `LayerNorm`, `Embedding`, and `Conv1d`/`Conv2d`/`Conv3d` modules, and a matching module selects every leaf under it.
  An empty result raises `ValueError` unless `allow_empty`; `expected_logical_ids` must match exactly
  [selectors.py:resolve_target] [selectors.py:\_resolve_predicate_paths] [selectors.py:\_check_expected_ids]
- `path_to_str` escapes `\` and `.` and prefixes integer-looking string parts with `\s`, so `str_to_path` round-trips
  paths unambiguously [paths.py:\_format_path_part] [paths.py:\_decode_path_part]
- Canonical tags (`attention.qkv`, `attention.proj`, `attention.q/k/v`, `mlp.fc1`, `mlp.fc2`, `mlp.hidden`, `head`,
  `block.N`, `bias` for `bias` and `*_bias` leaves, `weight`) are derived from path parts, and `vector` from the leaf:
  any array with at most one axis, so PEFT vectors (DoRA magnitudes, AdaLoRA singular values, VeRA and RandLoRA scales,
  IA3 vectors, FourierFT coefficients, scale/shift, adapter scales) get `*_no_decay` labels by default. Equimo ViT block
  depth uses execution order across chunks [../core/tags.py:canonical_tags_for_path] [tags.py:make_param_info]
  [\_blocks.py:vision_block_depths]
- Labels are `<base>_decay` or `<base>_no_decay`, and leaves sharing a label must agree on multiplier and weight decay.
  PEFT leaves use the method as base (`lora_A`, `vera`, ...); unless every multiplier below is 1.0, the base also names
  the location, as other leaves are labeled (`lora_A_block_03`, `lora_B_head`), so each label keeps one rate. LLRD
  multipliers: head leaves use `head_lr_multiplier`; block leaves use
  `top_block_lr_multiplier * decay ** (max_depth - depth)`; embedding leaves use `embedding_lr_multiplier`, or
  `top_block_lr_multiplier * decay ** (max_depth + 1)` when it is `None`; other leaves use 1.0. `max_depth` is the
  deepest trainable block with `rebase_selected_depth=True` and the deepest block otherwise [labels.py:\_label_base]
  [labels.py:\_uniform_learning_rates] [labels.py:\_lr_multiplier] [labels.py:\_merge_group_spec]
- `GroupSpec.tags` is the intersection of the group's leaf tags and `tags_all` their union; `role` is the first leaf's
  role and `roles` every leaf's role [labels.py:\_merge_group_spec]
- `validate_plan` recomputes the report from `plan.combine()` and raises `ValueError` on any mismatch in fingerprint,
  partitions, labels, identities, mask, or group specs. Fingerprints are `sha256:` hashes of sorted compact JSON
  [inspection.py:validate_plan] [inspection.py:\_hash_json]
- All config and spec dataclasses are frozen. `FeatureSpec.__post_init__` validates layout, token selection, pooling,
  mask field, layer aggregation, and endpoint options; `TrainableSpec` and `LLRDConfig` values are not validated at
  construction [config.py:FeatureSpec] [config.py:TrainableSpec] [config.py:LLRDConfig]
- Modules take unbatched inputs (use `jax.vmap`); `extract_features` accepts each layout at its declared rank or one
  less. Forward methods default to `inference=True`, and dropout with `inference` false or `None` requires a key
  [feature_extraction.py:\_compile_feature_spec] [heads.py:MLPHead] [heads.py:AttentionPoolingClassifierHead]
- Heads return raw logits. Attention pooling softmaxes go through `equimo.core.ops.softmax` with `where=mask` and logits
  of at least float32. `AttentionPoolingClassifierHead` casts the probabilities back to the value dtype, and
  `AttentionPool` (float32 query) returns the pooled features in the feature dtype
  [heads.py:AttentionPoolingClassifierHead] [pooling.py:AttentionPool]
- The two mask conventions are opposite: the endpoint argument named by `FeatureSpec.mask_field` is a padding mask (True
  marks padding), while the `mask` argument of `pool_features` modules and heads marks valid tokens (True keeps the
  token) [feature_extraction.py:\_masked_mean] [feature_extraction.py:\_attention_pool_explicit]
  [pooling.py:MeanTokenPool] [pooling.py:AttentionPool] [pooling.py:LastTokenPool]
- `extract_features` with `FeatureSpec(pooling="attention")` raises `ValueError` without a key and draws a fresh random
  query from the key on every call [feature_extraction.py:\_attention_pool_explicit]
- Surgery and recipes return new trees through `eqx.tree_at`; `prepare_finetune` splits with `eqx.partition` and
  `FineTunePlan.combine` rejoins with `eqx.combine` [surgery.py:prepare_finetune] [config.py:FineTunePlan]
- Merging touches only inexact leaves, skips head leaves unless `include_head`, checks architecture, checkpoint, and
  logical-ID hashes when present, and requires exactly one PRNG source for DARE [merging.py:\_mergeable]
  [merging.py:\_check_task_vector_base] [merging.py:dare_task_vector]
- Calibration collectors keep int32 counts and accumulate means and centered sums in float32 (float64 only with
  `jax_enable_x64`), combine with Chan's update only when every provenance field matches, and refuse to finalize with
  zero samples [calibration.py:initialize_calibration_collector] [calibration.py:combine_calibration_collectors]
  [calibration.py:finalize_calibration_collector]
- Bundle archives contain exactly `manifest.json` and `arrays.eqx`, use format `equimo.finetune.bundle` version 1, and
  never pickle; size limits, the arrays SHA-256, schema version, architecture hash, `sha256:` base-checkpoint hash, and
  lineage hashes are checked on load and raise `FineTuneBundleError`. Architecture and base-checkpoint hashes recorded
  before Equimo 2.8, which also covered values Equimo now computes such as RoPE frequencies, are accepted; a trained
  delta entry for such a value (a module's `_legacy_stored_leaves`) raises `FineTuneBundleError`
  [serialization.py:\_read_archive] [serialization.py:\_encode_value] [serialization.py:\_check_schema]
  [serialization.py:\_check_hash] [serialization.py:\_check_base_checkpoint]
  [serialization.py:\_check_bundle_lineage_consistency] [serialization.py:\_apply_trainable_delta]
  [peft/lora.py:matches_architecture_hash]
- `l2_sp_loss`, `adapter_norm_loss`, and `task_vector_norm_loss` cover trainable leaves only
  (`equimo.core.parameters.trainable_mask`), so buffers are never penalized [regularization.py:\_trainable_entries]
- Calibration archives use the same two members, size limits, and arrays SHA-256 check, with format
  `equimo.finetune.calibration` version 1 and the encoded artifacts under the manifest's `artifacts` key.
  `load_calibration_artifacts` raises `FineTuneBundleError` on another format or version and validates the loaded
  artifacts [serialization.py:save_calibration_artifacts] [serialization.py:load_calibration_artifacts]
- Delta method dispatch uses the codec registry (`lora`, `dora`, `adapter`, `prompt`, `prefix`, `scale_shift`, `ia3`,
  `vera`); the registered config types must equal `peft.PEFTConfig`. `save_delta` without `base_model` raises
  `ValueError` when any wrapper has `train_base=True`, when `spec` is a `TrainableSpec` whose mode trains base leaves
  (any mode but `peft` and `frozen`), or when `spec` has a truthy `train_head`, `train_norm`, or `train_bias`
  [serialization.py:\_CODECS] [serialization.py:\_validate_codec_config_families]
  [serialization.py:\_require_base_for_unrepresented_trainables]

## Side Effects

- Disk I/O only in `serialization.py`: saves create parent directories, stage `manifest.json` and `arrays.eqx` in a
  system `tempfile.TemporaryDirectory()`, and write an LZ4-compressed tar atomically through `equimo._io.atomic_file`;
  loads stream the archive and spool `arrays.eqx` to a temporary file above 64 MiB
  [serialization.py:save_finetune_bundle] [serialization.py:\_write_archive] [serialization.py:\_read_archive]
- `RuntimeWarning` when loading a legacy archive without an arrays checksum [serialization.py:\_read_archive]
- Importing `serialization` builds the codec registry and can raise `RuntimeError` if it disagrees with `PEFTConfig`;
  bundle enrichment reads installed package metadata for the Equimo version
  [serialization.py:\_validate_codec_config_families] [serialization.py:\_equimo_version]
- Importing the package eagerly imports every submodule, including `audio`, `language`, `recipes`, and `vision`
  [\_\_init\_\_.py:\_\_all\_\_]
- Randomness only from caller-supplied keys: head and adapter initialization, Mixout, DARE (or `config.seed`), and
  `pool_features(pool="attention")` and `extract_features` with `FeatureSpec(pooling="attention")`, which draw a fresh
  random query on every call [regularization.py:mixout_tree] [merging.py:dare_task_vector] [pooling.py:pool_features]
  [feature_extraction.py:\_attention_pool_explicit]
- `greedy_soup` calls the user's `score_fn` once per candidate (plus once per model with a config)
  [merging.py:greedy_soup]
- Host synchronization that is not jit-traceable: calibration finalize and validation, and full-value hashing of base
  models during bundle enrichment and merge checks [calibration.py:finalize_calibration_collector]
  [serialization.py:\_checkpoint_hash] [merging.py:\_checkpoint_hash]. These value hashes also cover the static fields
  of modules that compute values from their configuration (RoPE and Swin position embeddings), so base models that
  differ only in RoPE settings do not match [peft/\_compat.py:hash_inexact_pytree]
  [../\_legacy_layouts.py:computed_value_settings]. A base-checkpoint or architecture hash that does not match is
  computed again in the pre-2.8 layout, which evaluates the computed tables on the CPU backend
  [serialization.py:\_check_base_checkpoint] [../\_legacy_layouts.py:legacy_leaves_with_path]

## Verification

- Run: `devenv shell -- uv run pytest tests/finetune -q`
