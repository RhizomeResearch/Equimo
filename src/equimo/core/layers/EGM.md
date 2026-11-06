# equimo.core.layers

## Overview

This directory holds the shared building blocks that the vision, language, audio, and tabular models assemble.

- **Attention:** multi-head attention, with an XLA path and an opt-in cuDNN fused path, and a pre-norm transformer
  block.
- **Rotary embeddings:** sin/cos factors applied to queries and keys.
- **Feed-forward layers:** `Mlp`, the SwiGLU variants, and a DINO projection head.
- **Normalization:** norms and layer scale.
- **Stochastic depth:** drop-path layers.
- **Mamba2 / VSSD mixer.**
- **Generic containers:** residual wrappers, windowed sequences, and block chunks.

Each layer family has a string-keyed registry, so models can name layers in their configs. `get_layer` resolves a name
across all the core registries.

## Interface

The public package surface is `equimo.core.layers.__all__` ([`__init__.py`](./__init__.py)). The entries below mark
names that are not re-exported there.

### Registries

- `[_registry.py:make_register]`: builds `register(name=None, force=False)`, a decorator that stores a class under
  `name.lower()`, or under `cls.__name__.lower()` when no name is given.
- `[_registry.py:make_get]`: builds `get(module)`, a case-insensitive name lookup. A class passed in is returned
  unchanged.
- `[_registry.py:_resolve_from_registries]`: an internal helper used across modalities. It looks a name up in several
  `[_registry.py:NamedLayerRegistry]` pairs, optionally with `collision_policy="first"`.
- The per-family registry pairs, listed with the names registered at import:
  - `[activation.py:register_act]` / `[activation.py:get_act]`: any callable. Built-ins: `relu`, `gelu`, `exactgelu`,
    `silu`, `elu`, `sigmoid`, `hard_sigmoid`, `hard_swish`, `softmax` (which is `equimo.core.ops.softmax`), `relu6`.
  - `[attention.py:register_attn]` / `[attention.py:get_attn]`: `attention`.
  - `[attention.py:register_attn_block]` / `[attention.py:get_attn_block]`: `attentionblock`.
  - `[ffn.py:register_ffn]` / `[ffn.py:get_ffn]`: `weightnormlinear`, `dinohead`, `mlp`, `swiglu`, `swiglufused`.
  - `[norm.py:register_norm]` / `[norm.py:get_norm]`: `rmsnormgated`, `layerscale`, `dynamictanh` (`DyT`), `rmsnorm2d`,
    `simplenorm`, `simplenorm2d`, `grn`, `layernorm2d`, plus the Equinox built-ins `layernorm`, `rmsnorm`, `groupnorm`.
  - `[dropout.py:register_dropout]` / `[dropout.py:get_dropout]`: `droppath`, `droppathadd`.
  - `[mamba.py:register_mixer]` / `[mamba.py:get_mixer]`: `mamba2mixer`.
- `[generic.py:get_layer]`: resolves a name across the core registries in this order: attention block, mixer, attention,
  normalization, feed-forward, dropout (`[generic.py:_core_layer_registries]`).

### Attention and rotary

- `[attention.py:Attention]`: multi-head self-attention.
  - Constructor:
    `Attention(dim, num_heads, *, key, qkv_bias, proj_bias, qk_norm, attn_drop, proj_drop, norm_layer, eps, implementation="xla")`.
  - Call: `__call__(x (..., seq, dim), key, inference=None, mask=None, rotary=None)`. `key` has no default.
  - Rotary factors are applied after the optional q/k norms.
- `[attention.py:AttentionBlock]`: pre-norm transformer block that computes
  `x + drop_path(ls1(postnorm(attn(prenorm(x)))))`, then `x + drop_path(ls2(mlp(norm(x))))`. `postnorm` is a norm only
  when `post_attention_norm=True`, and `Identity` otherwise.
  - Call: `__call__(x (seqlen, dim), key, inference=None, *, mask=None, attn_mask=None, ffn_mask=None, rotary=...)`.
    `key` has no default.
  - Its sublayers are resolved by name through `get_attn`, `get_ffn`, `get_norm` and `get_act`.
- `[attention.py:with_attention_implementation]`: returns the model with every module that has an `implementation` field
  switched to `"xla"` or `"cudnn"`. Parameters are unchanged.
- `[attention.py:AttentionImplementation]`: `Literal["xla", "cudnn"]`.
- `[attention.py:_fused_attention]`: internal cuDNN path, a wrapper around `jax.nn.dot_product_attention`.
- `[rotary.py:RotaryFactors]`: holds `sin` and `cos` of shape `(sequence, feature)`, an optional `scale`, and a static
  `layout` (`[rotary.py:RotaryLayout]`: `"split_half"` or `"interleaved"`).
- `[rotary.py:apply_rotary]`: applies unscaled factors to one tensor along `sequence_axis=-2`.
- `[rotary.py:apply_rotary_qk]`: rotates queries and keys together. When `scale` is set, it multiplies `q` and divides
  `k` by it.
- `[rotary.py:make_1d_rotary_factors]`: builds factors with frequencies `1 / theta^(2i/F)`.
- `[rotary.py:insert_rotary_identity]`: inserts `count` identity rows (`sin=0`, `cos=1`) at `index`, for example for
  prefix tokens.

### Feed-forward

- `[ffn.py:Mlp]`:
  `Mlp(in_dim, *, key, out_dim, hidden_dim, act_layer="gelu", norm_layer=None, dropout_rate, bias, eps)`. It computes
  `fc1 -> act -> optional norm -> dropout -> fc2 -> dropout`, with an optional `mask` of shape `(seqlen, 1)`.
- `[ffn.py:SwiGlu]` and `[ffn.py:SwiGluFused]`: SwiGLU FFNs. The hidden width is `int(hidden_dim * 2 / 3)`, rounded up
  to a multiple of `align_to` (default 8). The fused variant uses a single `w12` projection and matches the DINOv2 giant
  model.
- `[ffn.py:DINOHead]`: `fc1 -> fc2 -> fc3 -> L2 normalize -> WeightNormLinear`. Its `key` and `inference` arguments are
  unused.
- `[ffn.py:WeightNormLinear]`: weight-normalized linear layer `x @ (g * v / ||v||).T`.

### Normalization

- `[norm.py:LayerScale]`: per-channel `gamma` applied along `axis`. `gamma` is tagged `block.layer_scale`, which keeps
  it out of weight decay by default.
- `[norm.py:maybe_layer_scale]`: returns `Identity` when `init_values is None`. Not re-exported.
- `[norm.py:RMSNormGated]`: RMS norm with an optional gate `z` multiplied in before normalizing.
- `[norm.py:DyT]`: dynamic tanh, `tanh(alpha * x) * weight + bias`.
- Channel-first `(C, H, W)` norms that normalize over axis 0 at each pixel: `[norm.py:RMSNorm2d]`,
  `[norm.py:LayerNorm2d]`, `[norm.py:SimpleNorm2d]`.
- `[norm.py:SimpleNorm]`: uncentered norm that uses the Bessel-corrected variance.
- `[norm.py:GRN]`: global response normalization. It is the identity at initialization.
- `SimpleNorm`, `SimpleNorm2d` and `GRN` are registered but not re-exported.

### Drop-path, mixer, and containers

- `[dropout.py:DropPath]`: stochastic depth for one unbatched example: a single draw zeroes the whole input with
  probability `p`, otherwise it is rescaled by `1 / (1 - p)`.
- `[dropout.py:DropPathAdd]`: returns `x1 + x2`, or `x1` alone when the branch is dropped. `x2` is not rescaled.
- `[dropout.py:split_drop_path]`: maps a float or a 1- or 2-element list to two rates. Not re-exported.
- `[mamba.py:Mamba2Mixer]`: Mamba2 / VSSD mixer over `(seqlen, dim)`. It uses `equimo.core.ops.non_causal_linear_attn`
  ([../ops/EGM.md](../ops/EGM.md)). `dt_bias`, `A_log` and `D` are tagged `ssm`, which keeps them out of weight decay by
  default.
- `[generic.py:Residual]`: computes `drop_path(x, ls(module(x)))`. The wrapped module receives `key` and `inference`
  only when `pass_args=True`.
- `[generic.py:WindowedSequence]`: pads a `(C, H, W)` input to a multiple of `window_size`, runs the blocks on each
  window with `vmap`, then crops back.
- `[generic.py:BlockChunk]`: `posemb -> optional downsample -> blocks -> optional downsample_last`.
  - String layer types are resolved through `layer_resolver`, which defaults to `get_layer`.
  - `intermediate_features(...)` returns `(x, selected block outputs)`.
- `[generic.py:make_transformer_block_chunk]`: builds a transformer `BlockChunk`, or returns `None` when `depth <= 0`.
  Not re-exported.
- `[generic.py:count_chunk_blocks]`: total number of blocks in a sequence of chunks. Not re-exported.

## Invariants

### Registries

- `[_registry.py:make_register]`: names are stored lowercased. A duplicate name raises `ValueError` unless `force=True`,
  which overwrites the entry silently. Registering a class that does not subclass `eqx.Module` raises `TypeError`.
  `[activation.py:register_act]` accepts any callable, without a type check.
- `[_registry.py:make_get]`, `[activation.py:get_act]`: an unknown name raises `ValueError` listing the available names.
- `[_registry.py:_resolve_from_registries]`: the same applies across registries. A name found in several of them raises
  `ValueError` ("ambiguous") unless `collision_policy="first"`, and `[generic.py:get_layer]` passes no policy. A name
  with no match raises `ValueError`.
- `[norm.py:_NORM_REGISTRY]`, `[activation.py:_ACT_REGISTRY]`: the built-in entries are assigned straight into the dict,
  so the duplicate check does not run for them.

### Attention

- `[attention.py:Attention.__init__]`: `dim` must be divisible by `num_heads` (an `assert`).
  `[attention.py:_check_implementation]` raises `ValueError` unless the implementation is `"xla"` or `"cudnn"`.
- `[attention.py:Attention]`: `dim`, `num_heads`, `head_dim` and `implementation` are static fields. Changing
  `implementation` triggers a retrace. `eqx.tree_at` cannot change it, so `[attention.py:with_attention_implementation]`
  rebuilds modules with `object.__new__`.
- `[attention.py:Attention.__call__]`: the XLA path is the reference behavior for attention numerics.
  - Logits are cast to float32 before they are scaled by `1 / sqrt(head_dim)`.
  - The softmax is `jax.nn.softmax(..., where=allowed)`. It does not use `equimo.core.ops.softmax`.
  - Probabilities are cast back to `x.dtype`, so fully masked rows get zero probability mass.
  - `rotary.feature_dim` must equal `head_dim`, or `ValueError` is raised.
- `[attention.py:_broadcast_attention_mask]`: any nonzero mask entry counts as allowed. A mask with the same rank as `x`
  gets a head axis inserted at `-3`. A mask that cannot broadcast raises `ValueError`.
- `[attention.py:_fused_attention]` (cuDNN path) has these requirements:
  - Query and key/value sequence lengths are zero-padded to multiples of 64 (`_CUDNN_SEQUENCE_MULTIPLE`), because the
    backward kernel needs it. The output is sliced back to the query length.
  - Masks are expanded to every head, because a broadcast head axis fails under `vmap` (jax-ml/jax#34920). The padding
    is filled with `False`.
  - Fully masked rows are zeroed, because cuDNN returns the mean value for them.
  - Inputs must be bfloat16 or float16, or `ValueError` is raised.
  - `[attention.py:_check_cudnn_dropout]` rejects attention dropout that is active.
  - The cuDNN path cannot be exported with jax2onnx (per the `with_attention_implementation` docstring).
- `[attention.py:AttentionBlock.__init__]`: does not pass `implementation` to `attn_layer`, so a newly built block uses
  the XLA path. Use `with_attention_implementation` to switch it afterwards.
- `[attention.py:AttentionBlock.__call__]`: `attn_mask` and `ffn_mask` default to `mask`.

### PRNG keys and drop-path

- `[attention.py:Attention.__call__]`, `[attention.py:AttentionBlock.__call__]`, `[ffn.py:Mlp.__call__]`,
  `[ffn.py:SwiGlu.__call__]`, `[generic.py:BlockChunk.__call__]` and `[generic.py:WindowedSequence.__call__]` split keys
  with `equimo.core._prng.split_for_mode`.
  - `key=None` is valid only under static `inference=True`. Otherwise `ValueError` is raised.
- `[dropout.py:DropPath]`, `[dropout.py:DropPathAdd]`: both are declared `strict=True`. `inference=None` falls back to
  `self.inference`, and `p == 0` acts as inference. In training they raise `RuntimeError` without a key.
  `[dropout.py:_drop_probability]` requires `p` to be a concrete scalar in `[0, 1]`.

### Normalization

- `[norm.py:RMSNormGated]`, `[norm.py:DyT]`, `[norm.py:RMSNorm2d]`, `[norm.py:SimpleNorm]`, `[norm.py:SimpleNorm2d]`,
  `[norm.py:GRN]`, `[norm.py:LayerNorm2d]`: compute in float32 and return the input dtype. `eps` is a static field on
  all of them except `DyT`, which accepts and ignores `eps` so it can be built as `norm_layer(dim, eps=eps)`.
- `[norm.py:LayerScale.__call__]`: raises `ValueError` on a channel mismatch, and casts `gamma` to `x.dtype`.
- `[norm.py:GRN.__call__]`: guards the `sqrt` gradient at 0 with `jnp.where`.
- `[ffn.py:DINOHead.__call__]`: L2-normalizes in float32 with `+1e-12` in the denominator, then casts back.
  `[ffn.py:WeightNormLinear.__call__]` computes the row norm in float32 with no epsilon.

### Rotary

- `[rotary.py:RotaryFactors.__init__]`: raises `ValueError` unless `sin` and `cos` are 2-D with the same shape and
  dtype, the feature dim is even and nonzero, the layout is valid, and `scale` (when set) has the same shape.
- `[rotary.py:apply_rotary]`: rejects scaled factors, since scaled factors must be applied to Q and K together.
- `[rotary.py:_apply_rotary_unscaled]`: raises `ValueError` if `x` does not match the factors' feature dim or sequence
  length. The computation runs in the factors' dtype, and the output is cast back to the input dtype.
- `[rotary.py:make_1d_rotary_factors]`, `[rotary.py:insert_rotary_identity]`: validate their arguments and raise
  `ValueError` on invalid input.

### Mixer and containers

- `[mamba.py:Mamba2Mixer.__init__]`: `int(dim * expand)` must be a multiple of `head_dim`, or `ValueError` is raised.
- `[mamba.py:Mamba2Mixer.__call__]`: computes `A = -exp(A_log)` and `softplus(dt + dt_bias)` in float32, then casts both
  back to the input dtype. Its `key` and `inference` arguments are unused.
- `[generic.py:BlockChunk.__init__]`: requires `module` or `downsampler` (an `assert`). A `drop_path` sequence whose
  length is not `depth` raises `ValueError`. List-valued `module_kwargs` entries of length `depth` are spread across the
  blocks.
  - `downsample_last` and `downsampler_needs_key` are static fields.
- `[generic.py:WindowedSequence.__init__]`: a `drop_path` list whose length is not `depth` raises `ValueError`. Inner
  blocks are always built with `window_size=0`.

## Side Effects

- Importing the package fills the module-global registries listed under Interface:

  - `[activation.py:_ACT_REGISTRY]`
  - `[attention.py:_ATTN_REGISTRY]`
  - `[attention.py:_ATTN_BLOCK_REGISTRY]`
  - `[ffn.py:_FFN_REGISTRY]`
  - `[norm.py:_NORM_REGISTRY]`
  - `[dropout.py:_DROPOUT_REGISTRY]`
  - `[mamba.py:_MIXER_REGISTRY]`

  The `register_*` decorators mutate these registries at runtime. Other modalities depend on these names. Membership in
  `_NORM_REGISTRY` also gives the leaves an instance holds directly the `norm` tag in `equimo.core.parameters` masks and
  fine-tuning tags, unless their field declares tags; the registry is read when the tags are computed.

- `[generic.py:_core_layer_registries]`: imports the attention, dropout, ffn, mamba and norm modules lazily, the first
  time it is called.

- Randomness comes only from keys the caller passes in: dropout in `[attention.py:Attention]` and `[ffn.py:Mlp]`,
  `[dropout.py:DropPath]` / `[dropout.py:DropPathAdd]` masks, and parameter initialization in constructors such as
  `[mamba.py:Mamba2Mixer.__init__]`.

- There is no I/O, no printing and no host callback.

## Verification

- Run:
  `devenv shell -- uv run pytest tests/test_core_attention.py tests/test_attention.py tests/test_activation.py tests/test_dropout.py tests/test_ffn.py tests/test_generic.py tests/test_norm.py tests/test_mamba.py tests/test_rotary.py tests/test_layer_registry_coverage.py -q`
