# equimo.vision.models

## Overview

Vision model architectures built from [`../layers`](../layers/EGM.md) and `equimo.core.layers`: plain and hierarchical
Vision Transformers, convolutional and hybrid backbones, a deep-equilibrium model, a recurrent-depth ViT, and two
query-based segmentation heads (EoMT and PMT). Every architecture class registers itself with
`equimo.registry.register_model(name, modality="vision")`. Named presets (for example `vit_small_patch16_224`,
`dinov3_vitb16_pretrain_lvd1689m`, `convnext_tiny`) are thin factories over per-file config registries and delegate
construction and optional pretrained loading to `equimo.core.factory.build_model_variant`. The package `__init__.py`
auto-imports every module and merges each module's `__all__` into the package namespace.

## Interface

### Package wiring

- `__init__.py` re-exports `get_model_cls` and `register_model` from `equimo.registry`, then imports every submodule
  found by `pkgutil.iter_modules` (skipping `registry`) and merges each module's `__all__`. Adding a module with a
  correct `__all__` is all that is needed to export it. `[__init__.py:__all__]`
- Registered model names: `vit` `[vit.py:VisionTransformer]`, `vision_parcae` `[parcae.py:VisionParcae]`, `convnext`
  `[convnext.py:ConvNeXt]`, `deq` `[deq.py:DEQ]`, `eomt` `[eomt.py:EoMT]`, `pmt` `[pmt.py:PMT]`, `fastervit`
  `[fastervit.py:FasterViT]`, `iformer` `[iformer.py:IFormer]`, `lowformer` `[lowformer.py:LowFormer]`, `mlla`
  `[mlla.py:Mlla]`, `vssd` `[vssd.py:Vssd]`, `attnet` `[attnet.py:AttNet]`, `mobilenetv3` `[mobilenet.py:MobileNetv3]`,
  `partialformer` `[partialformer.py:PartialFormer]`, `reduceformer` `[reduceformer.py:ReduceFormer]`, `shvit`
  `[shvit.py:SHViT]`.

### Vision Transformer (`vit.py`)

- `VisionTransformer(img_size, in_channels, dim, patch_size, num_heads, depths, *, key, ...)`: class, register, and mask
  tokens; optional learned absolute position embedding and/or local RoPE; string layer names resolved through the layer
  registries. `[vit.py:VisionTransformer]`
- Endpoints, all on one unbatched `[C, H, W]` image: `__call__(x, key=None, inference=None) -> [num_classes]`;
  `features(x, key=None, mask=None, inference=None) -> [seqlen, dim]` (pre-norm tokens, order cls, registers, patches);
  `intermediate_features(x, ..., indices=None, n_last_blocks=None, apply_norm=False)`; `forward_features(x) -> dict`
  (`x_norm_cls_token`, `x_norm_reg_tokens`, `x_norm_patchtokens`, `x_prenorm`);
  `feature_metadata(x, *, endpoint, endpoint_options)`; `prepare_tokens(x, *, key, inference)`; `num_blocks`;
  `block_at(index)`. `[vit.py:VisionTransformer.features]`, `[vit.py:VisionTransformer.intermediate_features]`,
  `[vit.py:VisionTransformer.forward_features]`, `[vit.py:VisionTransformer.feature_metadata]`,
  `[vit.py:VisionTransformer.prepare_tokens]`, `[vit.py:VisionTransformer.block_at]`
- `_feature_position_configuration()` describes the position-embedding setup; PMT checkpoints and features record it as
  identity metadata. `[vit.py:VisionTransformer._feature_position_configuration]`
- Factories over `_VIT_REGISTRY` via
  `_build_vit(variant, pretrained=False, inference_mode=True, key=None, **overrides)`: `vit_*` (take `**kwargs`), and
  `dinov2_*`, `dinov3_*`, `lingbot_*`, `eupe_*`, `siglip2_*`, `tips_*`, `vit5_*` (take
  `pretrained: bool = False, **kwargs`). `[vit.py:_build_vit]`, `[vit.py:_VIT_REGISTRY]`
- Internal catalog hook `_catalog_model_variants()` describing `vision/dinov2_vits14_reg` from `_VIT_REGISTRY`, consumed
  by `equimo.catalog`. `[vit.py:_catalog_model_variants]`

### Recurrent-depth ViT (`parcae.py`)

- `VisionParcae(img_size, in_channels, dim, patch_size, num_heads, *, key, ...)`: prelude blocks, a weight-shared core
  looped `h[t+1] = R(adapter(h[t], e))` inside a statically bounded `lax.scan`, then readout `C` and coda. `__call__`,
  `features`, `forward_features`, `intermediate_features` accept `num_steps` / `num_steps_pair`; `sample_num_steps`,
  `readout`, `weight_decay_mask`, and spectral diagnostics are public. `weight_decay_mask()` is
  `equimo.core.parameters.weight_decay_mask(self)`: the injection `A_log`, `dt_bias`, `B` and the readout `C` (weight
  and bias) carry the `ssm` tag, so they are not decayed, nor are other leaves with a `DEFAULT_NO_DECAY_TAGS` tag
  (biases, norms, layer scales, vectors, position embeddings, and class, register, and mask tokens). The patch embedding
  still decays. `[parcae.py:VisionParcae]`
- `VisionParcae.intermediate_features` indexes the fixed triple (recurrent state, `C` projection, coda output), not
  blocks. `[parcae.py:VisionParcae.intermediate_features]`
- Injection adapters `VisionParcaeDiagonalInjection`, `VisionParcaeDiagonalExactZOHInjection`,
  `VisionParcaeLinearInjection`, `VisionParcaeAdditiveInjection`, selected by `injection_type` (`diagonal`,
  `diagonal_exact_zoh`, `linear`, `add`); `dynamics_from_alpha(...)` maps a decay `alpha` to `(A_init, B_init_scale)`;
  `BInitMode` is the `Literal` type of `B_init_mode`. `[parcae.py:_get_vision_parcae_injection]`,
  `[parcae.py:dynamics_from_alpha]`, `[parcae.py:BInitMode]`
- Factories `vision_parcae_{tiny,small,base}_patch16_224`. `[parcae.py:_build_vision_parcae]`

### Query segmentation (`eomt.py`, `pmd.py`, `pmt.py`)

- `EoMT(backbone: VisionTransformer, num_classes, num_queries=100, num_blocks=4, *, key, masked_attention=True)`:
  learned queries injected into the last `num_blocks` ViT blocks.
  `__call__(x, *, key=None, inference=True, mask_state=None) -> EoMTOutput(final, auxiliary)` of
  `MaskPrediction(class_logits (Q, K+1), mask_logits (Q, h, w))`; `joint_features` / `features`, `token_trace`.
  `[eomt.py:EoMT]`, `[eomt.py:MaskPrediction]`, `[eomt.py:EoMTOutput]`
- `EoMTMaskState`, `anneal_mask_state(step, start_steps, end_steps, *, power=0.9)`, `mask_free_eomt(model, state)`, and
  `ScaleBlock(channels, *, key)` (twofold upscaling in the mask head). `[eomt.py:EoMTMaskState]`,
  `[eomt.py:anneal_mask_state]`, `[eomt.py:mask_free_eomt]`, `[eomt.py:ScaleBlock]`
- `PMT(backbone: VisionTransformer, num_classes, *, key, config=None, backbone_id="unbound", input_view="native", class_ontology=None)`:
  frozen ViT encoder plus a trainable `PlainMaskDecoder`. `encode(image) -> PMTFeatures`,
  `decode(features, state=..., *, inference, key, mask_state, example_valid)`, `__call__`, `is_stateful()`,
  `encoder_features` / `features`; `mask_free_pmt(model, mask_state)`. `[pmt.py:PMT]`, `[pmt.py:mask_free_pmt]`
- `PMTConfig` (frozen dataclass with `small()`, `base()`, `large()` presets), `PMTFeatures`, `PMTOutput`,
  `PMTMaskState`, `anneal_pmt_mask_state`, `ReferenceBatchNorm` (stateful, cross-device BatchNorm), `PlainMaskDecoder`.
  `PMDAttention` and `Lateral` are decoder internals outside `pmd.__all__`; import them from `pmd`. `Lateral.scale` is
  tagged `block.layer_scale`, so the default no-decay policy and layer-scale selectors treat it as a layer scale.
  `[pmd.py:PMTConfig]`, `[pmd.py:ReferenceBatchNorm]`, `[pmd.py:PlainMaskDecoder]`, `[pmd.py:PMDAttention]`,
  `[pmd.py:Lateral]`

### Convolutional and hybrid backbones

- `ConvNeXt` (V1/V2, `_ols` stems, tagged in12k/in22k/fcmae and EUPE checkpoints) with `convnext_sizes`,
  `convnext_v2_sizes`, and factories `convnext_*`, `convnextv2_*`, `eupe_convnext_*`. `[convnext.py:ConvNeXt]`,
  `[convnext.py:_build_convnext]`
- `DEQ` with fixed-point (`fpi`) stages, `readout(z_star)`, `get_fpi_cells()`, and `deq_convnext_t`; `__call__` returns
  `(logits, auxs)`. `[deq.py:DEQ]`, `[deq.py:deq_convnext_t]`
- `FasterViT` (HAT carrier tokens), `IFormer` (`iformer_*`), `LowFormer` (`lowformer_backbone_b0..b3`), `Mlla`, `Vssd`
  (both built by `build_mlla_stack`). `[fastervit.py:FasterViT]`, `[iformer.py:IFormer]`, `[lowformer.py:LowFormer]`,
  `[mlla.py:build_mlla_stack]`, `[vssd.py:Vssd]`
- `AttNet` (`attnet_*`), `MobileNetv3` (`mobilenetv3_small`, `mobilenetv3_large`), `PartialFormer`, `ReduceFormer`
  (`reduceformer_backbone_b1..b3`), `SHViT`. `[attnet.py:AttNet]`, `[mobilenet.py:MobileNetv3]`,
  `[partialformer.py:PartialFormer]`, `[reduceformer.py:ReduceFormer]`, `[shvit.py:SHViT]`
- Most backbones expose `features`, `intermediate_features(x, key, inference, indices, n_last_blocks)`, and `__call__`.
  `ConvNeXt` and `IFormer` inherit them from `DenseStageFeatures`; `Mlla` and `Vssd` from `TokenStemFeatures`.
  `[_features.py:DenseStageFeatures]`, `[_features.py:TokenStemFeatures]`

### Shared private helpers

- `build_token_embeddings(...) -> TokenEmbeddings` and `build_local_rope(...)` build ViT-style patch embedding, prefix
  tokens, absolute position embedding, and composite RoPE. `[_embedding.py:build_token_embeddings]`,
  `[_embedding.py:build_local_rope]`
- `_run_stages(blocks, x, keys, *, inference, indices=())` runs stages and keeps selected outputs.
  `[_features.py:_run_stages]`

## Invariants

- All forward endpoints take one unbatched channel-first `[C, H, W]` image; batch with `jax.vmap`.
  `[vit.py:VisionTransformer.feature_metadata]` raises `ValueError` when `x.ndim != 3`; `[eomt.py:EoMT]` and
  `[pmt.py:PMT.encode]` enforce the same.
- `key` defaults to `None` and is split with `split_for_mode`, so inference (`inference=True`) runs without a key while
  stochastic forwards without a key raise. `[vit.py:VisionTransformer.features]`, `[_features.py:DenseStageFeatures]`,
  `[deq.py:DEQ.features]`
- Exceptions to keyless inference: EoMT and PMT require a key whenever `masked_attention` is on, even at inference
  (`mask_free_eomt` / `mask_free_pmt` turn it off once all mask probabilities are zero); `VisionParcae` requires a key
  unless `inference is True and state_init == "zero"`. `[eomt.py:EoMT]`, `[pmd.py:PlainMaskDecoder]`,
  `[parcae.py:VisionParcae]`
- Heads are `eqx.nn.Linear` only when `num_classes` is positive, otherwise `eqx.nn.Identity`.
  `[vit.py:VisionTransformer]`, `[convnext.py:ConvNeXt]`, `[deq.py:DEQ]`
- ViT token order is cls, registers, patches; query models prepend queries (query, prefix, patch order) and insert
  identity rotary rows for them. `[vit.py:VisionTransformer.forward_features]`, `[eomt.py:EoMT]`
- `VisionTransformer` masking requires `use_mask_token=True` (assert) and keeps tokens where the mask is true. RoPE
  requires a static integer `num_heads`. `block_at` raises `IndexError` outside `[0, num_blocks)`.
  `[vit.py:VisionTransformer]`, `[_embedding.py:build_local_rope]`, `[vit.py:VisionTransformer.block_at]`
- `VisionTransformer.__init__` takes no `**kwargs`, so an unknown config key raises `TypeError` instead of being
  ignored. EUPE ViTs share the DINOv3 patch RoPE config (`rescale_coords=2.0` during training) but compute periods in
  bfloat16, as the EUPE checkpoints store them. `[vit.py:VisionTransformer]`, `[vit.py:_EUPE_BASE_CFG]`
- `build_token_embeddings` results must stay direct model fields: wrapping them in a submodule changes `jtu.keystr`
  checkpoint paths. `[_embedding.py:build_token_embeddings]`
- `DenseStageFeatures` and `TokenStemFeatures` mixins must declare no fields and no `__init__`; required attributes are
  declared only under `TYPE_CHECKING`. `[_features.py:DenseStageFeatures]`
- `EoMT` requires a `VisionTransformer` backbone with `dynamic_img_size`, an identity head, plain `AttentionBlock`
  chunks without downsampling, square power-of-two patches of at least 8, and `1 <= num_blocks <= backbone.num_blocks`.
  `[eomt.py:EoMT]`
- `PMT` requires `config.dim == backbone.dim`, `taps[-1] == backbone.num_blocks - 1`, matching heads, local RoPE,
  `dynamic_img_size`, an identity head, a nonempty `backbone_id` and `input_view`, and a unique `class_ontology` of
  length `num_classes`. `encode` runs the backbone with `inference=True` and `stop_gradient`, so the encoder is frozen.
  `decode` raises when the feature identity (`backbone_id`, `backbone_digest`, `input_view`, positional configuration)
  differs from the model. `[pmt.py:PMT]`, `[pmt.py:PMT.encode]`, `[pmt.py:PMT.decode]`
- `PMTConfig` validates positive sizes, `dim % num_heads == 0`, `(dim // num_heads) % 4 == 0`, sorted unique nonnegative
  taps, `norm_layer` in `{groupnorm, batchnorm, layernorm, rmsnorm, none}`, and a per-norm `norm_kwargs` allowlist.
  `[pmd.py:PMTConfig]`
- `ReferenceBatchNorm` computes in float32, casts back to the input dtype, zeroes invalid examples, stores unbiased
  running variance, and raises when a training batch has fewer than two tokens across `axis_name`. With
  `norm_layer="batchnorm"`, `PlainMaskDecoder` requires an `eqx.nn.State` and returns `(output, state)` whenever a state
  is passed. `[pmd.py:ReferenceBatchNorm]`, `[pmd.py:PlainMaskDecoder]`
- Mask states hold float32 per-block probabilities whose length must equal the decoder or query-block count. EoMT and
  PMT treat `mask_state=None` as all-ones probabilities, which keeps every mask while masked attention is on.
  `[eomt.py:EoMTMaskState]`, `[pmd.py:PMTMaskState]`, `[eomt.py:EoMT]`, `[pmd.py:PlainMaskDecoder]`
- `VisionParcae` validates recurrence settings (`mean_recurrence >= 1`, `max_recurrence >= mean_recurrence`,
  `max_recurrence >= mean_backprop_depth`, and others); its configuration fields are static, so changing
  `max_recurrence` requires a new model. Python step counts above `max_recurrence` raise; traced step counts are
  clamped. Diagonal injections are contractive by construction. Unlike `VisionTransformer`, `VisionParcae.__init__`
  raises `TypeError` on unknown kwargs. `[parcae.py:VisionParcae]`, `[parcae.py:VisionParcae._resolve_num_steps]`,
  `[parcae.py:VisionParcaeDiagonalInjection]`
- `DEQ.readout` raises unless exactly one stage is `fpi`. `[deq.py:DEQ.readout]`
- `LowFormer` and `ReduceFormer` pop the stem entry from the caller's `widths`, `depths`, and `block_types` lists; their
  factories pass `deepcopy_cfg=True` to protect the registry configs. `[lowformer.py:LowFormer]`,
  `[reduceformer.py:ReduceFormer]`
- Only ConvNeXt variants listed in `PRETRAINED_ARCHIVE_SHA256` accept `pretrained=True`; `_build_convnext` passes them
  as `pretrained_variants`. With `pretrained=True`, `convnext_*` and `convnextv2_*` variants default
  `act_layer="exactgelu"` (a caller override wins), and `convnext_zepto_rms_ols` also gets `exactgelu` in its default
  `stem_kwargs`. `[convnext.py:_CONVNEXT_PRETRAINED_VARIANTS]`, `[convnext.py:_build_convnext]`

## Side Effects

- Importing the package registers every architecture in the global model registry (duplicate names raise) and mutates
  module-level config registries such as the ConvNeXt tagged-variant loop. `[__init__.py:__all__]`,
  `[convnext.py:_CONVNEXT_REGISTRY]`
- Factories with `pretrained=True` delegate to `build_model_variant`, which downloads and deserializes checkpoints
  through `equimo.serialization.load_weights` (network and file IO). `[vit.py:_build_vit]`,
  `[convnext.py:_build_convnext]`
- `ReferenceBatchNorm` updates running statistics in `eqx.nn.State` during training and performs `lax.psum` over its
  `axis_name`. `[pmd.py:ReferenceBatchNorm]`
- EoMT and PMT spend PRNG keys sampling stochastic query attention masks whenever masked attention is enabled.
  `[eomt.py:EoMT]`, `[pmd.py:PlainMaskDecoder]`
- `PMT.__init__` hashes every backbone array on the host (`encoder_array_digest`); `mask_free_eomt` and `mask_free_pmt`
  read a concrete boolean and cannot run under `jit`. `[pmt.py:PMT]`, `[eomt.py:mask_free_eomt]`
- `VisionParcae` consumes keys for state initialization (every `state_init` except `zero`) and, with
  `sample_recurrence=True` and a `poisson-*` `sampling_scheme`, Poisson recurrence depths during training; the default
  `sampling_scheme="fixed"` is deterministic. `[parcae.py:VisionParcae.sample_num_steps]`
- Training forwards consume keys for dropout, drop-path, and RoPE augmentation. `[vit.py:VisionTransformer.features]`

## Verification

- Run:
  `devenv shell -- uv run pytest tests/test_models.py tests/test_model_factory.py tests/test_model_catalog.py tests/test_parcae.py tests/test_eomt.py tests/test_pmt.py tests/test_pmt_compositions.py tests/test_prng_free_inference.py -q`
