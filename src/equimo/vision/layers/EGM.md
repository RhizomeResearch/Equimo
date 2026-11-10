# equimo.vision.layers

## Overview

Vision-specific Equinox layers: attention variants and attention blocks, convolution blocks, patch embeddings, stems and
downsamplers, positional embeddings (learned, Swin MLP, RoPE, depthwise-conv), squeeze-and-excitation, and Haar wavelet
downsampling. Each family owns a name-keyed registry so models can name layers by string. `get_layer` resolves a name
across the vision registries and the shared `equimo.core.layers` registries. The package `__init__.py` also re-exports
shared core layers (activation, dropout, FFN, generic, Mamba mixer, norm) so models can import everything from
`equimo.vision.layers`.

## Interface

### Registries and resolution

- `get_layer(name_or_cls)`: resolves a string through, in order, vision attention blocks, core attention blocks,
  convolution, mixer, positional embedding, downsampler, patch, vision attention, core attention, normalization,
  feed-forward, dropout, squeeze-and-excitation, and wavelet registries; classes pass through unchanged.
  `[registry.py:get_layer]`
- Per-family `register_*` decorators (`register_x(name=None, force=False)`, default name `cls.__name__.lower()`) and
  `get_*` resolvers (case-insensitive string lookup, classes pass through), built with
  `equimo.core.layers._registry.make_register` / `make_get`:
  - `register_attn` / `get_attn` / `_ATTN_REGISTRY` and `register_attn_block` / `get_attn_block` /
    `_ATTN_BLOCK_REGISTRY` `[attention.py:register_attn]`, `[attention.py:register_attn_block]`
  - `register_conv` / `get_conv` / `_CONV_REGISTRY` `[convolution.py:register_conv]`
  - `register_patch` / `get_patch` `[patch.py:register_patch]`; `register_downsampler` / `get_downsampler`
    `[downsample.py:register_downsampler]`; `register_posemb` / `get_posemb` `[posemb.py:register_posemb]`;
    `register_se` / `get_se` `[squeeze_excite.py:register_se]`; `register_wavelet` / `get_wavelet`
    `[wavelet.py:register_wavelet]`
- `__all__` re-exports the vision classes plus core `get_act`, `register_act`, `get_dropout`, `register_dropout`,
  `get_ffn`, `register_ffn`, `get_mixer`, `register_mixer`, `get_norm`, `register_norm`, `DropPath`, `DropPathAdd`,
  `Mlp`, `SwiGlu`, `SwiGluFused`, `DINOHead`, `WeightNormLinear`, `BlockChunk`, `Residual`, `WindowedSequence`,
  `Mamba2Mixer`, `DyT`, `LayerNorm2d`, `LayerScale`, `RMSNorm2d`, and `RMSNormGated`. It holds no `_*_REGISTRY` dict.
  Convolution classes, `get_conv`, `register_conv`, `update_ghostnet`, `finalize_ghostnet`, `MaskedKeyAttention`,
  `ConvNeXtOverlapStem`, and the Haar functions are not re-exported; import them from their module or resolve registered
  classes by name. `[__init__.py:__all__]`

### Attention (`attention.py`)

- `Attention`, `AttentionBlock`: imported from `equimo.core.layers.attention` and registered here as `attention` and
  `attentionblock`. `[attention.py:Attention]`, `[attention.py:AttentionBlock]`
- `MaskedKeyAttention`: core `Attention` whose `_project_qkv` zeroes the key slice of the fused qkv bias.
  `[attention.py:MaskedKeyAttention]`
- Token-sequence (`[seqlen, dim]`) attention: `WindowedAttention` (Swin-v2 MLP relative position bias), `MMSA` (mixed
  multi-head self-attention with head-mixing projections), `SQA` (single-query attention,
  `__call__(x, q, key, inference)`), `LinearAngularAttention` (Castling-ViT, optional sparse softmax term),
  `LinearAttention` (Mlla linear attention with 2D RoPE and LePE). `[attention.py:WindowedAttention]`,
  `[attention.py:MMSA]`, `[attention.py:SQA]`, `[attention.py:LinearAngularAttention]`, `[attention.py:LinearAttention]`
- Channel-first (`[C, H, W]`) attention: `SHSA` (SHViT single-head attention on the first `pdim` channels), `SHMA`
  (iFormer single-head multi-scale attention with optional windowing), `RFAttention` (ReduceFormer ReLU linear
  attention), `ConvAttention` (LowFormer attention on a strided grid with transposed-conv upsampling).
  `[attention.py:SHSA]`, `[attention.py:SHMA]`, `[attention.py:RFAttention]`, `[attention.py:ConvAttention]`
- Blocks: `HATBlock` (FasterViT window plus carrier tokens; `__call__(x, carrier_tokens, key, inference) -> (x, ct)`,
  where `ct` is the carrier-token branch output from before the joint attention, or `carrier_tokens` when
  `sr_ratio == 1`), `SHMABlock`, `MllaBlock`, `PartialFormerBlock` (`__call__(x, qa, inference, key) -> (x, qa)`),
  `RFAttentionBlock`, `ConvAttentionBlock`, `LowFormerBlock`. `[attention.py:HATBlock]`, `[attention.py:SHMABlock]`,
  `[attention.py:MllaBlock]`, `[attention.py:PartialFormerBlock]`, `[attention.py:RFAttentionBlock]`,
  `[attention.py:ConvAttentionBlock]`, `[attention.py:LowFormerBlock]`
- `Mamba2Mixer` (from `equimo.core.layers.mamba`) is also reachable as attention name `mamba2mixer`.
  `[attention.py:_ATTN_REGISTRY]`

### Convolution (`convolution.py`, all `[C, H, W]`)

- Basic units: `SingleConvBlock` (conv or transposed conv, norm, act, dropout), `DoubleConvBlock` (two convs with
  LayerScale and drop-path residual), `Stem` (ViT image-to-token stem returning `(h*w, c)`).
  `[convolution.py:SingleConvBlock]`, `[convolution.py:DoubleConvBlock]`, `[convolution.py:Stem]`
- YOLO CSP blocks: `ConvBottleneck`, `C2f`, `C3k`, `C3`, `C3k2`. `[convolution.py:C2f]`, `[convolution.py:C3k2]`
- MobileNet-family blocks: `MBConv` (fused or unfused inverted residual), `DSConv`, `UIB`. `[convolution.py:MBConv]`,
  `[convolution.py:DSConv]`, `[convolution.py:UIB]`
- Model-specific blocks: `IFormerStem`, `IFormerBlock`, `ConvNeXtBlock` (V1 by default, V2 with
  `use_grn=True, init_values=None`), `PartialConv2d`, `FasterNetBlock`, `GLUConv`, `ATConv`, `ATConvBlock`, `S2Mixer`,
  `ShiftNeck`, `ShiftFFN`, `FreeNetBlock`. `[convolution.py:ConvNeXtBlock]`, `[convolution.py:FasterNetBlock]`,
  `[convolution.py:ATConvBlock]`, `[convolution.py:FreeNetBlock]`
- GhostNetV3 reparameterizable blocks: `GenericGhostModule`, `GhostBottleneck`; `update_ghostnet(model)` writes fused
  inference kernels while keeping training branches, `finalize_ghostnet(model)` fuses, drops training branches, and sets
  `inference=True`. `[convolution.py:GenericGhostModule]`, `[convolution.py:GhostBottleneck]`,
  `[convolution.py:update_ghostnet]`, `[convolution.py:finalize_ghostnet]`
- `upsample_twofold(conv, x)`: a 2x2 stride-2 `ConvTranspose2d` applied as a per-pixel matmul, equal to the transposed
  convolution, in the input dtype. `conv` must have a bias, otherwise this raises `AssertionError`.
  `[convolution.py:upsample_twofold]`

### Patch embedding, stems, and downsamplers

- `PatchEmbedding(in_channels, embed_dim, patch_size, *, img_size, flatten, dynamic_img_size, dynamic_img_pad, ...)`:
  conv with kernel = stride = patch size; returns `(num_patches, embed_dim)` or `(embed_dim, gh, gw)`;
  `dynamic_feat_size(img_size)` gives the grid. `[patch.py:PatchEmbedding]`
- `ConvPatchEmbed`, `PatchMerging` (tokens `(seqlen, in_dim) -> (seqlen/4, out_dim)`), `SEPatchMerging`.
  `[patch.py:ConvPatchEmbed]`, `[patch.py:PatchMerging]`, `[patch.py:SEPatchMerging]`
- `ConvNormDownsampler` (`simple` 2x or `double` 4x), `PWSEDownsampler`, `ConvNeXtStem`, `ConvNeXtOverlapStem` (timm
  `_ols` stems), `ConvNeXtDownsampler`. `[downsample.py:ConvNormDownsampler]`, `[downsample.py:PWSEDownsampler]`,
  `[downsample.py:ConvNeXtStem]`, `[downsample.py:ConvNeXtOverlapStem]`, `[downsample.py:ConvNeXtDownsampler]`

### Positional embeddings (`posemb.py`)

- `LearnedPosEmbed`: learned table with three prefix-token modes and `resample(new_size, ...)` interpolation;
  `__call__(x, *, cls_token, reg_tokens, dynamic_img_size, interpolation)`. `[posemb.py:LearnedPosEmbed]`
- `PosEmbMLPSwinv1D`, `PosEmbMLPSwinv2D` (Swin-v2 continuous relative position bias). `[posemb.py:PosEmbMLPSwinv2D]`
- RoPE: `RoPE(shape, base)` (interleaved), `DinoRoPE` (`get_factors(H, W, key, inference) -> RotaryFactors`,
  `split_half` layout), `VisionRoPE(strategy="period" | "mode", ...)` with `layout`,
  `get_factors(H, W, num_prefix_tokens, key, inference)` and `__call__`,
  `CompositeVisionRoPE(patch_rope, reg_rope, num_prefix_tokens, num_registers, reg_grid)` for
  `[CLS | registers | patches]` layouts, whose `get_factors(*, H, W, key=None, inference=True)` gives identity rows for
  prefix tokens, and for registers when `reg_rope` is `None`. `[posemb.py:RoPE]`, `[posemb.py:DinoRoPE]`,
  `[posemb.py:VisionRoPE]`, `[posemb.py:CompositeVisionRoPE]`
- `PosCNN` (tokens, square grid), `PosCNN2D` (`[C, H, W]`). `[posemb.py:PosCNN]`, `[posemb.py:PosCNN2D]`
- `VisionRoPE.width_first_factors(grid)`: factors for a square grid that rotate width before height, evaluated while
  tracing. `[posemb.py:VisionRoPE.width_first_factors]`
- Functions, all in float32: `sine_coordinates(coords, num_features, *, temperature=10000.0)` (DETR sines on even
  features and cosines on odd ones), `sine_position_encoding(height, width, num_features, *, temperature=10000.0)`
  (`[2 * num_features, H, W]`, y channels first, evaluated while tracing), `fourier_encoding(matrix, coords)` and
  `dense_fourier_encoding(matrix, grid, dtype)` (random-frequency encoding, returned in `dtype` for the grid).
  `[posemb.py:sine_coordinates]`, `[posemb.py:sine_position_encoding]`, `[posemb.py:fourier_encoding]`,
  `[posemb.py:dense_fourier_encoding]`

### Squeeze-and-excitation and wavelets

- `SEModule`, `EffectiveSEModule`: channel gating on `[C, H, W]`. `[squeeze_excite.py:SEModule]`,
  `[squeeze_excite.py:EffectiveSEModule]`
- `haar_dwt_split(x, dtype)` and `haar_dwt_split_conv` return `(LL, HL, LH, HH)` at half resolution; `haar_dwt_split`
  zero-pads odd `H`/`W` on the bottom/right, so each band is `ceil(H/2) x ceil(W/2)`;
  `inverse_haar_dwt_split(LL, HL, LH, HH, *, orig_h, orig_w, dtype)` inverts it, cropping to `orig_h`/`orig_w`;
  `HWDConv(in_channels, out_channels, *, mode="h_discard" | "band_grouped" | "accurate", ...)` is a Haar wavelet
  downsampling conv. `[wavelet.py:haar_dwt_split]`, `[wavelet.py:inverse_haar_dwt_split]`, `[wavelet.py:HWDConv]`

## Invariants

- Modules are unbatched. Token modules take `[seqlen, dim]`; convolutional modules take channel-first `[C, H, W]`; batch
  with `jax.vmap`. `[convolution.py:SingleConvBlock]`, `[attention.py:WindowedAttention]`
- Registration lowercases names, raises `ValueError` on a duplicate name unless `force=True`, and raises `TypeError` for
  non-`eqx.Module` classes; `get_*` raises `ValueError` listing available names for unknown strings.
  `[attention.py:register_attn]`, `[convolution.py:register_conv]`, `[posemb.py:register_posemb]`
- `get_layer` uses `collision_policy="first"` in the registry order listed above, so a vision attention block shadows a
  same-named entry in a later registry. `[registry.py:get_layer]`
- Names registered only in vision registries are unknown to `WindowedSequence`, which always uses the core `get_layer`,
  and to `BlockChunk` unless the caller passes `layer_resolver=equimo.vision.layers.get_layer`.
  `[../../core/layers/generic.py:WindowedSequence]`, `[../../core/layers/generic.py:BlockChunk]`
- `mamba2mixer` is inserted with a direct dict assignment, bypassing `register_attn` collision checks.
  `[attention.py:_ATTN_REGISTRY]`
- Forward passes split keys with `split_for_mode(key, n, inference=...)`: with `inference=True` a `None` key is allowed;
  otherwise a missing key raises `ValueError`. `[attention.py:MMSA]`, `[downsample.py:PWSEDownsampler]`
- Many convolution blocks instead call `default_key_for_mode`, which substitutes `jr.PRNGKey(0)` when `key is None` and
  `inference` is not `True`, so stochastic layers run with a fixed key rather than raising. `SingleConvBlock`
  substitutes a key only when `dropout.p > 0`. `[convolution.py:SingleConvBlock]`, `[convolution.py:MBConv]`,
  `[convolution.py:FasterNetBlock]`
- Key and `inference` defaults are not uniform. `WindowedAttention`, `MMSA`, `HATBlock`, `SHMA`, `SHMABlock`, `SQA`,
  `LinearAngularAttention`, `RFAttention`, `RFAttentionBlock`, `ConvAttention`, `ConvAttentionBlock`, `LowFormerBlock`,
  `PWSEDownsampler`, and `PosCNN2D` take `key` with no default, so a keyless `inference=True` call raises `TypeError`.
  `SHSA`, `LinearAttention`, `MllaBlock` and `PartialFormerBlock` default `key=None`. `HWDConv` takes a required keyword
  `key`; `PosCNN2D` and `HWDConv` default `inference=False`. `VisionRoPE` and `CompositeVisionRoPE` default
  `key=None, inference=True`, while `DinoRoPE.get_factors` requires a keyword `key` and defaults `inference=False`.
  `[attention.py:WindowedAttention]`, `[attention.py:MMSA]`, `[downsample.py:PWSEDownsampler]`, `[posemb.py:PosCNN2D]`,
  `[wavelet.py:HWDConv]`, `[posemb.py:VisionRoPE]`, `[posemb.py:CompositeVisionRoPE]`, `[posemb.py:DinoRoPE]`
- Attention logits are cast to float32 before softmax or sigmoid and probabilities are cast back to the input dtype;
  `LinearAngularAttention` accumulates `k^T v` and norms in float32 with `1e-6` guards; `RFAttention` sums in float32
  with denominator `max(sum_q * sum_k, eps)`. `[attention.py:WindowedAttention]`, `[attention.py:SHSA]`,
  `[attention.py:MMSA]`, `[attention.py:LinearAngularAttention]`, `[attention.py:RFAttention]`
- Head divisibility is asserted (`head_dim * num_heads == dim`). `[attention.py:WindowedAttention]`,
  `[attention.py:MMSA]`, `[attention.py:SQA]`, `[attention.py:LinearAngularAttention]`
- Several token modules assume a square grid (`h = w = int(l**0.5)`) without an explicit check; `LinearAttention` raises
  `ValueError` if `n != h * w`, and `PosCNN` raises if the length is not a perfect square. `[attention.py:MllaBlock]`,
  `[attention.py:PartialFormerBlock]`, `[patch.py:PatchMerging]`, `[attention.py:LinearAttention]`, `[posemb.py:PosCNN]`
- `MllaBlock` follows the MLLA/VSSD block: the residual stream is `x + cpe1(x)`, the attention branch runs on `norm1` of
  that stream, and with `use_dwc=True` the output projection is `out_proj(attn_output * act_res)`.
  `[attention.py:MllaBlock]`
- `PatchEmbedding` raises on non-positive spatial size. When `img_size` is set, it asserts the input matches `img_size`
  unless `dynamic_img_size`, and with `dynamic_img_size` asserts divisibility by the patch size unless
  `dynamic_img_pad`. With `img_size=None` neither check runs and the strided conv drops leftover pixels unless
  `dynamic_img_pad`. With `dynamic_img_pad` it pads right/bottom by `(p - H % p) % p`. Its geometry fields are static.
  `[patch.py:PatchEmbedding]`
- `MBConv`, `DSConv`, and `UIB` enable the residual only when `residual=True` (default `False`), `stride == 1`, and
  `in_channels == out_channels`; tuple arguments must have length 3 (`MBConv`) or 2 (`DSConv`).
  `[convolution.py:MBConv]`, `[convolution.py:DSConv]`, `[convolution.py:UIB]`
- `DoubleConvBlock` adds its residual only at `stride == 1`. `[convolution.py:DoubleConvBlock]`
- `RFAttentionBlock(residual=False)` computes the context branch and discards it, returning the local branch alone.
  `[attention.py:RFAttentionBlock]`
- Where a block exposes `norm_max_group`, GroupNorm group counts use
  `nearest_power_of_2_divisor(channels, norm_max_group)`; `HWDConv` has no such argument and uses 32, and the `SHSA`
  pre-norm always uses one group. `[convolution.py:SingleConvBlock]`, `[patch.py:SEPatchMerging]`,
  `[wavelet.py:HWDConv]`, `[attention.py:SHSA]`
- Ghost blocks pick paths with a Python `if` when `inference` is a Python bool (required after `finalize_ghostnet`
  removes branches) and with `jax.lax.cond` otherwise; reparameterization fuses kernels only (no bias) and keeps the
  shared GroupNorms. `[convolution.py:GenericGhostModule]`, `[convolution.py:_update_ghostmodule]`
- `PartialConv2d` requires `1 <= channels // n_dim <= channels` and `SAME` or centered odd integer padding.
  `[convolution.py:PartialConv2d]`
- `S2Mixer` raises when the sampled branch channels exceed the channel count. `[convolution.py:S2Mixer]`
- `HWDConv` with `band_grouped` requires `out_channels % 4 == 0`; the `LL, HL, LH, HH` band order must stay contiguous
  for grouped norms and convs. `[wavelet.py:HWDConv]`
- RoPE: `RoPE` requires `feature_dim % (2 * len(position_axes)) == 0`; `DinoRoPE` and period `VisionRoPE` require
  `dim % (4 * num_heads) == 0` and exactly one of `base` or `(min_period, max_period)`; periods are computed in
  `periods_dtype` (float32 by default); coordinate augmentations run only when `inference` is false.
  `CompositeVisionRoPE` requires matching patch and register layouts, raises when `num_registers > 0` is not a perfect
  square and `reg_grid` is omitted, and its `__call__` raises when `H * W` differs from the patch-token count.
  `[posemb.py:RoPE]`, `[posemb.py:DinoRoPE]`, `[posemb.py:VisionRoPE]`, `[posemb.py:CompositeVisionRoPE]`
- `VisionRoPE.freqs`, `DinoRoPE.periods`, `RoPE.rotations`, and the `PosEmbMLPSwinv1D`/`PosEmbMLPSwinv2D` coordinate and
  index tables are properties computed from static fields, not PyTree leaves, so optimizers never see them and
  `eqx.tree_at` cannot replace them: build a module with other constructor arguments (`base`, `min_period`,
  `max_period`, `periods_dtype`, or the mode fields) instead. `VisionRoPE(custom_freqs=...)` stores the values as a
  static tuple, so they must be concrete at construction. Each class names the tables in `_legacy_stored_leaves`;
  checkpoints written before Equimo 2.8 stored them and load through `equimo._legacy_layouts`, which raises `ValueError`
  when a stored value differs from the computed one by more than rounding (for example after weight decay).
  `[posemb.py:VisionRoPE]`, `[posemb.py:PosEmbMLPSwinv2D]`
- `[posemb.py:_static_table]` evaluates those tables even while tracing, so `jax.jit` programs and ONNX exports hold
  them as constants; `tests/test_onnx_export.py::test_posemb_tables_export_as_constants` checks that each table exports
  as a constant. Exporting `RoPE` and the Swin tables needs a jax2onnx release that includes enpasos/jax2onnx#291.
  `LinearAttention` applies its `RoPE` to queries and keys in one call, so programs hold one table per block.
  `[attention.py:LinearAttention]`
- `LearnedPosEmbed` validates `cls_token` and `reg_tokens` shapes and resamples the grid part only, leaving prefix rows
  unchanged. `[posemb.py:LearnedPosEmbed]`

## Side Effects

- Importing each module mutates its module-global registry through the `@register_*` decorators; registering an existing
  name again without `force=True` raises. `[attention.py:_ATTN_REGISTRY]`, `[convolution.py:_CONV_REGISTRY]`,
  `[posemb.py:_POSEMB_REGISTRY]`, `[patch.py:_PATCH_REGISTRY]`, `[downsample.py:_DOWNSAMPLER_REGISTRY]`,
  `[squeeze_excite.py:_SE_REGISTRY]`, `[wavelet.py:_WAVELET_REGISTRY]`
- Training-mode forwards consume PRNG keys for dropout, drop-path, and RoPE coordinate augmentation.
  `[attention.py:MMSA]`, `[convolution.py:ATConvBlock]`, `[posemb.py:DinoRoPE]`
- `PosEmbMLPSwinv2D` builds gather indices with host-side Python loops at trace time when `ct_correct` is set and global
  tokens are present. `[posemb.py:PosEmbMLPSwinv2D]`
- `update_ghostnet` and `finalize_ghostnet` return new pytrees; nothing mutates in place. No module performs file or
  network IO or uses `eqx.nn.State`. `[convolution.py:update_ghostnet]`, `[convolution.py:finalize_ghostnet]`

## Verification

- Run:
  `devenv shell -- uv run pytest tests/test_attention.py tests/test_convolution.py tests/test_downsample.py tests/test_patch.py tests/test_posemb.py tests/test_squeeze_excite.py tests/test_wavelet.py tests/test_layer_registry_coverage.py tests/test_generic.py -q`
- The computed position-embedding tables are also covered by
  `devenv shell -- uv run pytest tests/test_onnx_export.py::test_posemb_tables_export_as_constants tests/test_io.py tests/test_checkpoint_signature_stability.py -q`
  (constant export, which needs jax2onnx with enpasos/jax2onnx#291, and loading pre-2.8 checkpoints).
