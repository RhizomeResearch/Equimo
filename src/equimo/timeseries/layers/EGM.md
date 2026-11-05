# equimo.timeseries.layers

## Overview

Layers for the T0 patch-transformer forecaster, adapted from tfc-t0 (see the repository `NOTICE`). They wrap Equimo's
shared core layers (`Attention`, `RMSNormGated`, `SwiGluFused`, `Mlp`) with T0's axis routing, XPos rotary scaling, and
patch encoding, and register themselves in a time-series layer registry that falls back to the core registries.

## Interface

- `register_layer(name=None, force=False)` \[registry.py:register_layer\]: registers into the time-series
  `_LAYER_REGISTRY`.
- `get_layer(name_or_cls)` \[registry.py:get_layer\]: resolves the time-series registry first, then the shared core
  layer registries, with `collision_policy="first"`.
- `AxisAttention(dim, num_heads, dropout, attention_type, *, key)` called as `(x, mask, *, key, inference=None)`
  \[axis_attn.py:AxisAttention\]: shared `equimo.core.layers.Attention` with `qk_norm=True`, `RMSNormGated` QK norm,
  `eps=1e-8`.
- `T0Block(dim, num_heads, hidden_dim, dropout, attention_type, *, attention_layer="axisattention", key, drop_path=0.0)`
  called as `(x, *, time_attn_mask, group_attn_mask, key, inference=None)` [blocks.py:T0Block].
- `PatchEncoder(embed_dim, patch_size, *, projection_layer="residualmlp", key)` called as
  `(values, mask, variate_type, *, key, inference=None)` [patch_encoder.py:PatchEncoder].
- `ResidualMlp(in_features, hidden_features, out_features, *, key)` called as `(x, *, key, inference=None)`: ReLU `Mlp`
  plus a linear residual projection over arbitrary leading axes [residual_mlp.py:ResidualMlp].

## Invariants

- `attention_type="time"` attends along the patch axis with T0's interleaved RoPE (theta 10000) plus reciprocal XPos
  scaling built in float32; `attention_type="group"` swaps the first two axes, attends across variates without rotary
  factors, and swaps back [axis_attn.py:\_time_rotary_factors, axis_attn.py:AxisAttention.\_\_call\_\_].
- `T0Block` picks `time_attn_mask` for time blocks and `group_attn_mask` otherwise; it is pre-norm with
  `RMSNormGated(eps=1e-8)`, and the FFN is `SwiGluFused` with `hidden_dim * 3 // 2` [blocks.py:T0Block]. `drop_path` is
  accepted and ignored [blocks.py:T0Block.\_\_init\_\_].
- `PatchEncoder` concatenates values, a per-patch time ramp `arange(patch_size) / patch_size`, and validity `mask == 0`,
  so its projection input width is `patch_size * 3`; it adds a type embedding indexed by `max(variate_type[:, :, 0], 0)`
  over three types [patch_encoder.py:PatchEncoder].
- Block keys are split with `split_for_mode`, so a missing key is allowed only under `inference=True`
  [blocks.py:T0Block.\_\_call\_\_].

## Side Effects

- Importing the modules registers `axisattention`, `t0block`, `patchencoder`, and `residualmlp` in the module-level
  `_LAYER_REGISTRY` \[axis_attn.py:AxisAttention, blocks.py:T0Block, patch_encoder.py:PatchEncoder,
  residual_mlp.py:ResidualMlp\]. No I/O.

## Verification

- Run: `devenv shell -- uv run pytest tests/test_t0.py -q`
