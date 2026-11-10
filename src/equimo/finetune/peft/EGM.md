# equimo.finetune.peft

## Overview

Parameter-efficient fine-tuning (PEFT) methods implemented as Equinox wrapper modules. Each method has a frozen config
dataclass, an `apply_*` function that resolves a `TargetSpec` against canonical parameter tags and swaps target modules
for wrappers with `eqx.tree_at`, and, where the math allows, merge/unmerge functions that fold the learned delta back
into the base weights. LoRA-family and adapter methods also produce and consume portable `FineTuneBundle` deltas, which
the parent package's `serialization.py` writes to disk. Methods: LoRA (plus rsLoRA, PiSSA, static rank masks, AdaLoRA,
LoRA-FA, RandLoRA, EVA init, FourierFT), DoRA, bottleneck/parallel adapters, AdaptFormer, adapter banks and
AdapterFusion, OFT/BOFT orthogonal adapters, IA3, VeRA, scale/shift (SSF), prefix tuning, prompt tuning (soft prompts,
VPT, P-Tuning v2), and BitFit.

## Interface

- `PEFTConfig`: union alias of
  `LoRAConfig | DoRAConfig | AdapterConfig | OrthogonalAdapterConfig | FourierFTConfig | PromptConfig | PrefixConfig | ScaleShiftConfig | IA3Config | VeRAConfig`
  (excludes `BitFitConfig`, `AdaptFormerConfig`, `AdapterBankConfig`, `AdapterFusionConfig`, `EVAInitializerConfig`,
  `LoRAPlusLabelConfig`, and `CalibrationSpec`) [\_\_init\_\_.py:PEFTConfig]
- Package exports: public names of `adapters`, `bitfit`, `dora`, `ia3`, `lora`, `prefix`, `prompts`, `scale_shift`, and
  `vera`. `architecture_hash`, `lora_config_to_dict`, `iter_dora_modules`, `strip_prefixes`, and
  `iter_prefix_attentions` are only in their module `__all__`; `base`, `_common`, and `_compat` are not re-exported
  [\_\_init\_\_.py:\_\_all\_\_]
- Private helpers in `_common.py`, besides `inside_wrapper` and `dropout`: `apply_last_axis` applies a vector module
  over the last axis, `split_optional_key` passes an absent key through, `linear_module_path` strips a trailing `weight`
  or `bias`, `target_linear_paths` resolves a target to the modules owning matched leaves, and
  `target_mentions_qkv_segment` and `selected_qkv_segment_names` read Q, K and V segments from a target's tags
  [\_common.py:apply_last_axis] [\_common.py:split_optional_key] [\_common.py:linear_module_path]
  [\_common.py:target_linear_paths] [\_common.py:target_mentions_qkv_segment] [\_common.py:selected_qkv_segment_names]
- Private helpers in `_compat.py`, besides `hash_inexact_pytree`: `bundle_get_path` resolves a delta path with the
  stable bundle error, `is_head_path` recognizes classifier-head paths, `is_linear_like`, `linear_weight` and
  `linear_bias` validate linear-like modules, `linear_state` and `linear_from_state` encode and decode the stable
  Equinox linear-layer state, and `validate_logical_parameter_schema` canonicalizes a logical-ID to feature-width schema
  [\_compat.py:bundle_get_path] [\_compat.py:is_head_path] [\_compat.py:is_linear_like] [\_compat.py:linear_weight]
  [\_compat.py:linear_bias] [\_compat.py:linear_state] [\_compat.py:linear_from_state]
  [\_compat.py:validate_logical_parameter_schema]
- Wrapper traversal: `get_path`, `iter_wrappers(model, wrapper_types)` returning `(Path, module)` pairs, and
  `map_wrappers(model, wrapper_types, fn)` applying one `eqx.tree_at` [base.py:iter_wrappers] [base.py:map_wrappers]

### LoRA family (`lora.py`)

- Configs: `LoRAConfig` (rank 8, alpha 16, target tags `attention.qkv`/`attention.proj`), `RsLoRAConfig`, `PiSSAConfig`,
  `StaticRankMaskedLoRAConfig`, `AdaLoRAConfig`, `LoRAFAConfig`, `RandLoRAConfig`, `LoRAPlusLabelConfig`,
  `AdaLoRAMetadata`, `CalibrationSpec`, `EVAInitializerConfig`, `FourierFTConfig` [lora.py:LoRAConfig]
  [lora.py:RsLoRAConfig] [lora.py:EVAInitializerConfig] [lora.py:FourierFTConfig]
- Wrappers: `LoRALinear` (`lora_A (rank, in)`, `lora_B (out, rank)`, optional `rank_mask`, `base_weight_delta`,
  `projection_segments`), `LoRAMergedLinear` (fused QKV marker), `LoRAFALinear`, `RandLoRALinear`, `FourierFTLinear`,
  `AdaLoRAModule` (with `orthogonality_loss`) [lora.py:LoRALinear] [lora.py:LoRAFALinear] [lora.py:RandLoRALinear]
  [lora.py:FourierFTLinear] [lora.py:AdaLoRAModule]
- `apply_lora`, `apply_adalora`, `apply_lora_fa`, `apply_randlora`,
  `apply_eva_lora(model, config, *, activation_artifacts, key, ...)`, `apply_fourierft` [lora.py:apply_lora]
  [lora.py:apply_eva_lora] [lora.py:apply_fourierft]
- `merge_lora` / `unmerge_lora` (all LoRA-family wrappers) and per-family `merge_lora_fa`, `unmerge_lora_fa`,
  `merge_randlora`, `unmerge_randlora`, `merge_fourierft`, `unmerge_fourierft` [lora.py:merge_lora]
  [lora.py:unmerge_lora]
- `extract_lora_delta(model, ...) -> FineTuneBundle`, `load_lora_delta(base_model, bundle)`, `strip_lora(model)`
  [lora.py:extract_lora_delta] [lora.py:load_lora_delta] [lora.py:strip_lora]
- `iter_lora_modules`, `iter_adalora_modules`, `iter_lora_fa_modules`, `iter_fourierft_modules`,
  `iter_randlora_modules`, `lora_rank_groups`,
  `apply_lora_rank_pattern(model, rank_pattern, *, strict=True, final=False)` [lora.py:lora_rank_groups]
  [lora.py:apply_lora_rank_pattern]
- `architecture_hash(model) -> str` (SHA-256 over inexact-leaf paths, shapes, dtypes) and `lora_config_to_dict`
  [lora.py:architecture_hash] [lora.py:lora_config_to_dict]. `matches_architecture_hash(expected, model) -> bool` also
  accepts the hash of the model's pre-2.8 layout; it is module-level only, outside `lora.__all__`
  [lora.py:matches_architecture_hash]

### Other methods

- DoRA: `DoRAConfig`, `DoRALinear`, `DoRAMergedLinear`, `apply_dora`, `merge_dora`, `iter_dora_modules`
  [dora.py:DoRALinear] [dora.py:apply_dora] [dora.py:merge_dora]
- Adapters: `AdapterConfig`, `ParallelAdapterConfig`, `AdaptFormerConfig` (presets `safe_default` and `paper_chen2022`,
  the latter with `activation="relu"`, `scale_init=0.1` and `scale_trainable=False`), `AdapterBankConfig`,
  `AdapterFusionConfig`, `OrthogonalAdapterConfig`; modules `BottleneckAdapter`, `AdaptFormerAdapter`,
  `SerialAdapterBlock`, `OutputAdapterModule`, `ParallelAdapterBlock`, `AdaptFormerBlock`, `AdapterFusion`,
  `OrthogonalLinear` [adapters.py:AdapterConfig] [adapters.py:OrthogonalLinear] [adapters.py:AdapterFusion]. Only delta
  loading constructs `SerialAdapterBlock` [adapters.py:\_wrapper_from_entry]
- Adapter functions: `apply_adapters`, `add_adapter`, `set_active_adapter`, `configure_adapter_bank`,
  `apply_adapter_fusion`, `adapter_fusion_trainable_spec`, `apply_adaptformer`, `apply_orthogonal_adapters`,
  `merge_orthogonal_adapters`, `unmerge_orthogonal_adapters`, `extract_adapter_delta`, `load_adapter_delta`,
  `strip_adapters`, `iter_adapter_wrappers`, `iter_orthogonal_adapters` [adapters.py:apply_adapters]
  [adapters.py:add_adapter] [adapters.py:configure_adapter_bank] [adapters.py:extract_adapter_delta]
  [adapters.py:load_adapter_delta]. With one active adapter at a site, `apply_adapter_fusion` stores every adapter name
  at that site as active, dropping the earlier selection. Sites match on path parts only, so a `SerialAdapterBlock` at a
  block path never matches `after_mlp`, and no matching site raises `ValueError` [adapters.py:apply_adapter_fusion]
  [adapters.py:\_fusion_active_names]
- IA3: `IA3Config`, `IA3Linear`, `apply_ia3`, `merge_ia3`, `iter_ia3_modules` [ia3.py:IA3Linear] [ia3.py:apply_ia3]
- VeRA: `VeRAConfig`, `VeRALinear`, `apply_vera`, `merge_vera`, `strip_vera`, `iter_vera_modules` [vera.py:VeRALinear]
  [vera.py:apply_vera]
- Scale/shift: `ScaleShiftConfig` (presets `safe_default`, `paper_lian2022_vit`, `convnet`), `ScaleShift`,
  `ScaleShiftWrapper`, `apply_scale_shift`, `merge_scale_shift`, `iter_scale_shift_wrappers`
  [scale_shift.py:ScaleShiftConfig] [scale_shift.py:apply_scale_shift] [scale_shift.py:merge_scale_shift]
- Prefix tuning: `PrefixConfig`, `PrefixTunedModel`, `PrefixAttention`, `PrefixProjection`, `apply_prefixes`,
  `strip_prefixes`, `iter_prefix_attentions` [prefix.py:PrefixTunedModel] [prefix.py:apply_prefixes]
- Prompt tuning: `PromptConfig`, `SoftPromptConfig`, `VPTShallowConfig`, `VPTDeepConfig`, `PTuningV2Config`,
  `PromptedModel`, `apply_prompts` [prompts.py:PromptConfig] [prompts.py:PromptedModel] [prompts.py:apply_prompts]
- BitFit: `BitFitConfig`, `bitfit_trainable_spec(config=None) -> TrainableSpec` (no wrapper; trainability only)
  [bitfit.py:bitfit_trainable_spec]

## Invariants

- Hyperparameters on wrappers (rank, alpha, scaling mode, dropout, `train_base`, `mergeable`, `merged`, projection
  segments, adapter names, active adapter) are `eqx.field(static=True)`, so changing one changes the treedef. Array
  leaves are factors and masks plus `OrthogonalLinear.unmerged_weight` (a full weight copy once merged),
  `FourierFTLinear.frequency_indices` (int32), `BottleneckAdapter.residual_scale`, and `AdaptFormerAdapter.scale`
  [lora.py:LoRALinear] [dora.py:DoRALinear] [adapters.py:SerialAdapterBlock] [adapters.py:OrthogonalLinear]
  [lora.py:FourierFTLinear] [adapters.py:BottleneckAdapter] [adapters.py:AdaptFormerAdapter]
- Every `apply_*`, `merge_*`, `unmerge_*`, and `strip_*` returns a new PyTree; inputs are not mutated. Module swaps use
  `eqx.tree_at`; `apply_prompts` instead wraps the whole model in `PromptedModel`, and `apply_prefixes` swaps attentions
  and then wraps the result in `PrefixTunedModel` [lora.py:apply_lora] [base.py:map_wrappers] [prompts.py:apply_prompts]
  [prefix.py:apply_prefixes]
- Delta starts at zero: LoRA `lora_B`, DoRA `lora_B`, RandLoRA `basis_scales`, FourierFT coefficients, AdaLoRA
  `singular`, VeRA `vera_output_scale` (`trainable_output_scale_init=0.0`), bottleneck adapter up projections and
  biases, and OFT skew parameters are zero-initialized; AdapterFusion starts with zero query/key and identity value
  [lora.py:apply_lora] [lora.py:RandLoRALinear] [lora.py:AdaLoRAModule] [dora.py:\_init_lora] [vera.py:VeRAConfig]
  [adapters.py:\_init_linear] [adapters.py:\_cayley] [adapters.py:\_adapter_fusion_from_state]
- PiSSA is the exception: `lora_B` holds the principal singular vectors and is nonzero. With
  `residual_handling="freeze_residual"` the base weight compensates; with `"none"` nothing does, so the wrapped model's
  output differs from the base at initialization [lora.py:\_pissa_init] [lora.py:\_pissa_prepare]
- LoRA scaling is `alpha / rank` for `alpha_over_r` and `alpha / sqrt(rank)` for `alpha_over_sqrt_r`; any other mode
  raises `ValueError`. `LoRALinear`, `LoRAFALinear`, `RandLoRALinear`, and `AdaLoRAModule` reject ranks below 1.
  `apply_adalora` applies the configured scaling mode, raises `ValueError` for a nonzero `config.dropout`, since
  `AdaLoRAModule` has no dropout path, and does not read `fan_in_fan_out`. `apply_lora_fa` and `apply_randlora` do not
  read `config.dropout` [lora.py:\_scaling] [lora.py:LoRALinear] [lora.py:LoRAFALinear] [lora.py:RandLoRALinear]
  [lora.py:AdaLoRAModule] [lora.py:apply_adalora] [lora.py:apply_lora_fa] [lora.py:apply_randlora]
- RandLoRA with `seed` set builds every wrapper's bases from `jr.PRNGKey(seed)`, so same-shape modules share bases; with
  `seed=None` each module uses its split subkey [lora.py:RandLoRALinear]
- Unsupported options raise `ValueError`: `apply_lora` with an init other than `kaiming_A_zero_B` or `pissa`,
  `apply_lora_fa` and `apply_randlora` with `fan_in_fan_out=True`, and `apply_prompts` with
  `PTuningV2Config(reparameterizer="mlp")` [lora.py:apply_lora] [lora.py:apply_lora_fa] [lora.py:apply_randlora]
  [prompts.py:\_validate_prompt_config]
- Target resolution raises `ValueError` when nothing resolves unless `target.allow_empty`; non-linear targets raise
  `TypeError`, except in `apply_scale_shift`, which wraps any module with a `weight`, `bias`, or `scale` leaf. Paths are
  processed in sorted order with one split subkey each; `apply_ia3` and `apply_orthogonal_adapters` take no key, and
  `apply_scale_shift` uses one only with `init="normal"` [lora.py:apply_lora] [dora.py:apply_dora]
  [ia3.py:\_target_linear_specs] [ia3.py:apply_ia3] [adapters.py:apply_orthogonal_adapters]
  [scale_shift.py:\_target_module_paths] [scale_shift.py:\_module_keys]
- Re-applying a method does not nest wrappers: the LoRA family, DoRA, VeRA, IA3, orthogonal adapters, and scale/shift
  skip a target that names or lies inside their own wrapper type, `apply_adapters` skips block wrappers and existing
  `OutputAdapterModule` sites, and `apply_adaptformer` skips `AdaptFormerBlock`s. `add_adapter` deliberately stacks: it
  adds a named adapter next to those already at a site [\_common.py:inside_wrapper] [vera.py:apply_vera]
  [scale_shift.py:apply_scale_shift] [adapters.py:apply_adapters] [adapters.py:apply_adaptformer]
  [adapters.py:add_adapter] [adapters.py:\_add_output_adapter_at_path]
- In `apply_lora` and `apply_dora`, a path containing `qkv` selects `LoRAMergedLinear` / `DoRAMergedLinear`;
  `apply_eva_lora` always builds `LoRALinear`. QKV projection segments require the output dim to be divisible by 3 and
  axis 0. `LoRALinear` and `RandLoRALinear` zero delta rows outside the selected q/k/v segments, only with
  `target_kind="projection_segment"`. IA3 ignores `target_kind`: it builds segments whenever the target names q, k, or
  v, and leaves the other rows at scale 1. LoRA-FA, AdaLoRA, FourierFT, and EVA change every row [lora.py:apply_lora]
  [lora.py:apply_eva_lora] [lora.py:LoRALinear] [lora.py:RandLoRALinear] [lora.py:\_projection_segments_for_target]
  [lora.py:\_mask_projection_segments] [ia3.py:\_projection_segments_for_target] [ia3.py:IA3Linear] [dora.py:apply_dora]
- Merge contract: `merge` raises `ValueError` if the wrapper is not mergeable or already merged, adds the delta cast to
  the base weight dtype, and sets `merged=True`; `unmerge` of an unmerged wrapper returns it unchanged; a merged wrapper
  returns `base(x)` [lora.py:AdaLoRAModule] [lora.py:\_merge_additive_lora] [adapters.py:OrthogonalLinear]
- `merge_dora`, `merge_ia3`, `merge_vera`, and `merge_scale_shift` replace each wrapper with its folded base module:
  there is no `merged` flag and no unmerge. They raise `ValueError` for `mergeable=False`; `merge_scale_shift` supports
  only `eqx.nn.Linear` and `eqx.nn.LayerNorm` bases and requires a bias when the shift is nonzero [dora.py:merge_dora]
  [ia3.py:merge_ia3] [vera.py:merge_vera] [scale_shift.py:\_merge_scale_shift_linear]
- Dropout runs only when `dropout > 0` and `not inference`, and then requires a PRNG key (`ValueError` otherwise). Most
  wrappers default to `inference=True`; adapter block wrappers and `PrefixAttention` default to `inference=None`, which
  counts as training. `ScaleShiftWrapper` also defaults to `inference=None` but has no dropout, and passes `key` and
  `inference` to its base only when they are not `None` [lora.py:LoRALinear] [\_common.py:dropout]
  [adapters.py:BottleneckAdapter] [adapters.py:SerialAdapterBlock] [prefix.py:PrefixAttention]
  [scale_shift.py:ScaleShiftWrapper] [scale_shift.py:\_call_base]
- Precision: AdapterFusion and prefix-attention logits are computed in at least float32 and the probabilities cast back
  to the activation dtype; LoRA-FA correction matrices, AdaLoRA orthogonality loss, and OFT orthogonality error are
  computed in float32 [adapters.py:AdapterFusion] [prefix.py:PrefixAttention] [lora.py:LoRAFALinear]
  [lora.py:AdaLoRAModule] [adapters.py:OrthogonalLinear]
- When the base attention has `implementation == "cudnn"`, `PrefixAttention` calls `_fused_attention` instead of the
  float32-logit softmax path and rejects active attention dropout [prefix.py:PrefixAttention]
- Adapter linears, bottleneck `residual_scale`, AdaptFormer `scale`, and `ScaleShift` scale and shift are float32
  parameters; `BottleneckAdapter`, `AdaptFormerAdapter`, and `ScaleShift` cast their output back to the activation dtype
  [adapters.py:BottleneckAdapter] [adapters.py:AdaptFormerAdapter] [scale_shift.py:ScaleShift]
  [scale_shift.py:\_init_parameter]
- LoRA-FA with `custom_vjp=True` returns a zero gradient for A; `frozen_A` and `corrected_v3` define the B gradient
  [lora.py:\_lora_fa_update_corrected_bwd]
- Frozen arrays are declared with `equimo.core.parameters.buffer`: VeRA `vera_A`/`vera_B`, LoRA-FA
  `frozen_A`/`correction_matrix`, RandLoRA `random_A`/`random_B`, PiSSA `base_weight_delta`, OFT `unmerged_weight`, and
  AdaptFormer `scale` when `scale_trainable=False`. Trainability plans, `trainable_mask`, and `weight_decay_mask`
  exclude them. Those the forward pass reads get zero gradients: the VeRA and RandLoRA bases and a fixed AdaptFormer
  `scale` through `jax.lax.stop_gradient`, LoRA-FA `frozen_A` through it with `custom_vjp=False` and otherwise through a
  VJP that returns zero for it and `correction_matrix`. The forward pass never reads PiSSA `base_weight_delta` (saving,
  loading, and stripping do) or OFT `unmerged_weight` (only `unmerge` does) [vera.py:VeRALinear] [lora.py:LoRAFALinear]
  [lora.py:RandLoRALinear] [lora.py:LoRALinear] [adapters.py:OrthogonalLinear] [adapters.py:AdaptFormerAdapter]
- VeRA with `shared=True` reuses one basis pair across modules with the same rank, shape, dtype, and init
  [vera.py:\_shared_vera_bases]
- PiSSA with `freeze_residual` stores `-scaling * B0 @ A0` in `base_weight_delta`; `strip_lora` and the loader undo or
  reapply it [lora.py:\_pissa_prepare] [lora.py:strip_lora] [lora.py:load_lora_delta]
- Delta loading: `load_lora_delta` and `load_adapter_delta` raise `FineTuneBundleError` on a wrong `bundle.method`, on a
  non-empty `architecture_hash` that `matches_architecture_hash(expected, base_model)` rejects, and (LoRA) on base
  weight or bias shape mismatch. The stored hash is computed on the stripped model; hashes recorded before Equimo 2.8,
  which also covered values Equimo now computes such as RoPE frequencies, still match [lora.py:load_lora_delta]
  [lora.py:extract_lora_delta] [lora.py:matches_architecture_hash] [adapters.py:load_adapter_delta]
- Adapter banks: more than one active adapter requires `allow_multiple_active=True`; `missing_adapter_policy` is `error`
  or `ignore`; `add_adapter` rejects parallel placement, duplicate names, and banks that already have fusion
  [adapters.py:\_normalize_active_adapter] [adapters.py:configure_adapter_bank] [adapters.py:add_adapter]
- Prefix tuning supports only attentions with `eqx.nn.Linear` `qkv` and `proj`, rejects targets other than
  `{attention.k, attention.v}` and `direct_kv=True`, and prepends an all-ones mask of width `num_prefix_tokens`. Rotary
  embedding is applied only to base tokens [prefix.py:\_validate_prefix_config] [prefix.py:\_is_supported_attention]
  [prefix.py:\_extend_mask] [prefix.py:PrefixAttention]
- `PrefixConfig(depth="shallow")` wraps only the first supported attention [prefix.py:apply_prefixes].
  `PrefixAttention.state` is a float array leaf not declared as a buffer; `PrefixTunedModel` overwrites it from
  `prefixes` and `prefix_projections` on every `__call__` and `features()` [prefix.py:PrefixTunedModel]
  [prefix.py:\_sync_prefixes]
- Prompt tuning supports Equimo ViT-like models and simple token models only (`ValueError` otherwise). Prompts are cast
  to the token dtype at use. `init="normal"` draws float32 prompts; `init="from_embedding_if_available"` (the
  `SoftPromptConfig` default) copies `token_embed.weight` rows in the embedding dtype, with the same rows for every
  layer, and falls back to float32 normal prompts when there is no token embedding [prompts.py:PromptedModel]
  [prompts.py:\_init_prompts] [prompts.py:\_prompt_from_embedding]
- `PromptedModel.__call__` reads prompted features out through the base's `head` and raises `ValueError` when the base
  has none; use `features()` for headless bases [prompts.py:PromptedModel]
- `PEFTConfig` must equal the set of config types registered by the parent package's serialization codecs, which is
  checked at import time [\_\_init\_\_.py:PEFTConfig] [../serialization.py:\_validate_codec_config_families]
- Accepted but currently unused: `LoRAPlusLabelConfig.label_A`, `label_B`, and `label_base`, `VeRAConfig.seed_required`,
  `DoRAConfig.magnitude_axis`, `LoRAConfig.weight_layout`, `EVAInitializerConfig.svd` and `allocation`,
  `ParallelAdapterConfig.branch` and `fusion`, `AdaptFormerConfig.placement`, and `OrthogonalAdapterConfig.eps`.
  `PiSSAConfig.svd` and `niter` are only validated and recorded in metadata, and PiSSA always runs a full SVD. No
  `apply_*` reads `train_head` on `IA3Config`, `ScaleShiftConfig`, `PrefixConfig`, `PromptConfig`, or
  `AdaptFormerConfig`; `save_delta` reads it only when such a config is passed as `spec` [lora.py:LoRAPlusLabelConfig]
  [vera.py:VeRAConfig] [dora.py:DoRAConfig] [ia3.py:IA3Config] [scale_shift.py:ScaleShiftConfig]
  [prefix.py:PrefixConfig] [prompts.py:PromptConfig] [adapters.py:AdaptFormerConfig]
  [../serialization.py:\_trainable_spec] [../serialization.py:\_require_base_for_unrepresented_trainables]

## Side Effects

- Randomness only from explicit keys passed to `apply_*`. Fixed `jr.PRNGKey(0)` (or a configured seed) is used for
  RandLoRA seeds, FourierFT without a key, PiSSA fallback, and placeholder construction during delta loading, where
  stored factors overwrite the random values [lora.py:RandLoRALinear] [lora.py:apply_fourierft]
  [lora.py:load_lora_delta] [adapters.py:\_bottleneck_from_state]
- Host transfers that are not jit-traceable: EVA rank allocation (`float(s[i] ** 2)`), VeRA key data stored in static
  fields, `merge_scale_shift` checking `bool(jnp.any(shift != 0))`, and value hashing via `np.asarray` in
  `_compat.hash_inexact_pytree` [lora.py:\_eva_rank_allocation] [vera.py:\_key_data_tuple]
  [scale_shift.py:\_merge_scale_shift_linear] [\_compat.py:hash_inexact_pytree]
- PiSSA and EVA catch any `Exception` from SVD initialization and fall back without a warning. PiSSA falls back only
  when `fallback_init == "kaiming_A_zero_B"` (otherwise it re-raises) and records the fallback in metadata
  [lora.py:\_pissa_prepare] [lora.py:\_eva_lora_A]
- Call helpers retry when the base raises a `TypeError` mentioning an unexpected keyword argument. `adapters._call_base`
  and `prefix._call_with_optional_key` retry once without `inference`; `scale_shift._call_base` retries once without
  `inference` and `key`; `prompts._call_block` retries without `inference`, then drops all keyword arguments, so a block
  can run up to three times [adapters.py:\_call_base] [prefix.py:\_call_with_optional_key] [scale_shift.py:\_call_base]
  [prompts.py:\_call_block]
- No disk or network I/O; bundles are in-memory `FineTuneBundle` objects [lora.py:extract_lora_delta]
  [adapters.py:extract_adapter_delta]

## Verification

- Run:
  `devenv shell -- uv run pytest tests/finetune/test_lora.py tests/finetune/test_lora_lifecycle.py tests/finetune/test_lora_merged_linear.py tests/finetune/test_lora_plus_labels.py tests/finetune/test_rslora.py tests/finetune/test_pissa.py tests/finetune/test_rank_masked_lora.py tests/finetune/test_dora.py tests/finetune/test_adapters.py tests/finetune/test_ia3.py tests/finetune/test_vera.py tests/finetune/test_scale_shift.py tests/finetune/test_prefix.py tests/finetune/test_prompts.py tests/finetune/test_peft_methods.py tests/finetune/test_peft_invariants.py tests/finetune/test_peft_wrapper_mapping.py tests/finetune/test_peft_compatibility_primitives.py -q`
