# equimo.tabular.layers

## Overview

Building blocks for TabPFN-style in-context tabular prediction: feature preprocessing, bias-free and in-context
attention, attention blocks, column-wise induced-attention encoders, per-row column aggregation, label embeddings, and
decoders. Each layer family has its own name-keyed registry, and every family registration is mirrored into a shared
tabular layer registry.

## Interface

- Registries \[registry.py:register_layer, registry.py:get_layer\]: `register_layer` adds to `_LAYER_REGISTRY`;
  `get_layer` resolves tabular names first, then falls back to `equimo.core.layers.get_layer`.
- Family registries, each a `register_*(name=None, force=False)` / `get_*(name_or_cls)` pair: `register_attn` /
  `get_attn` [attention.py:register_attn, attention.py:get_attn]; `register_attn_block` / `get_attn_block`
  [blocks.py:register_attn_block, blocks.py:get_attn_block]; `register_decoder` / `get_decoder` and `register_embedding`
  / `get_embedding` \[decoder.py:register_decoder, decoder.py:get_decoder, decoder.py:register_embedding,
  decoder.py:get_embedding\]; `register_preprocessor` / `get_preprocessor` \[preprocessing.py:register_preprocessor,
  preprocessing.py:get_preprocessor\].
- Attention \[attention.py\]:
  - `SoftmaxScaling(num_heads, head_dim, hidden_dim=64, *, key, act_layer="exactgelu")`: input-dependent query scaling
    `q * base_mlp(log n) * (1 + tanh(query_mlp(q)))` [attention.py:SoftmaxScaling].
  - `Attention(dim, num_heads, *, key, head_dim=None, implementation="xla")`: bias-free self-attention with
    `cross(q_tokens, kv_tokens, rope=None)` and optional rotary embedding [attention.py:Attention].
  - `CrossAttention(...)(q_tokens, kv_tokens)` with optional `SoftmaxScaling` [attention.py:CrossAttention].
  - `InContextAttention(..., num_kv_heads_test=None)(x, n_train)` [attention.py:InContextAttention].
- Blocks [blocks.py], all pre-norm with `DropPathAdd` residuals and a bias-free shared `Mlp`:
  `CrossAttentionBlock(q_tokens, kv_tokens)`, `AttentionBlock(x, rope=None)` plus
  `forward_cross(q_tokens, kv_tokens, rope)`, `InContextAttentionBlock(x, n_train=...)`,
  `InducedAttentionBlock(x, n_train=...)` (requires `num_inducing_points`) \[blocks.py:CrossAttentionBlock,
  blocks.py:AttentionBlock, blocks.py:InContextAttentionBlock, blocks.py:InducedAttentionBlock\].
- Encoders \[blocks.py\]: `FeatureDistributionEncoder(dim, num_heads, depth, *, key, num_inducing_points=128, ...)` maps
  `(rows, columns, dim)` by running a `BlockChunk` of `InducedAttentionBlock` per column
  [blocks.py:FeatureDistributionEncoder];
  `ColumnAggregator(dim, num_heads, depth, *, key, num_cls_tokens=4, use_rope=True, rope_base=100_000.0, ...)` maps
  `(rows, columns, dim)` to `(rows, num_cls_tokens, dim)` [blocks.py:ColumnAggregator].
- Embeddings and decoders \[decoder.py\]: `LabelEmbedding(num_classes, dim, *, key)` for integer labels,
  `LinearLabelEmbedding(num_outputs, dim, *, key)` for scalar targets,
  `AttentionDecoder(num_classes, dim, *, key, head_dim=64, num_heads=6, use_softmax_scaling=True, ...)` returning
  test-row log-probabilities, and `RegressionDecoder(num_outputs, dim, *, key, mlp_ratio=2.0, ...)` returning test-row
  bucket logits \[decoder.py:LabelEmbedding, decoder.py:LinearLabelEmbedding, decoder.py:AttentionDecoder,
  decoder.py:RegressionDecoder\].
- `Preprocessor(feature_group_size=3, use_nan_indicators=True)(x, n_train)` maps `(rows, columns)` to
  `(rows, columns, feature_group_size)`, or `(rows, columns, 2 * feature_group_size)` with NaN indicators
  [preprocessing.py:Preprocessor].
- `Mlp`, `get_ffn`, `register_ffn` are re-exported from `equimo.core.layers.ffn` [\_\_init\_\_.py].
- Internal helper `_call_mlp(mlp, x, *, key=None, inference=None)` applies the shared `Mlp` over arbitrary leading axes
  [mlp.py:\_call_mlp].

## Invariants

- Family registration checks collisions in both the family registry and `_LAYER_REGISTRY` before writing either, so a
  rejected registration leaves both unchanged; `force=True` overrides [registry.py:\_register_module].
- Attention layers require `head_dim * num_heads == dim` (`ValueError` otherwise) and validate `implementation` with
  `_check_implementation` [attention.py:Attention.\_\_init\_\_]. `implementation="cudnn"` routes to
  `equimo.core.layers.attention._fused_attention` [attention.py:\_scaled_dot_product_attention].
- On the XLA path, attention scores are promoted to at least float32 before `jax.nn.softmax`, and probabilities are cast
  back to the value dtype; fewer KV heads than query heads are repeated [attention.py:\_scaled_dot_product_attention].
- `AttentionDecoder` uses `equimo.core.ops.softmax` on float32-promoted scores, averages over heads, and returns
  `log(clip(probs, 1e-5) + 3e-5)` [decoder.py:AttentionDecoder.\_\_call\_\_].
- Training rows are the leading `n_train` rows: `InContextAttention` builds keys and values only from `x[:n_train]`;
  with `num_kv_heads_test` set, test rows attend through only the first `num_kv_heads_test` KV heads
  [attention.py:InContextAttention.\_\_call\_\_]. `InducedAttentionBlock` summarizes only training rows into inducing
  vectors [blocks.py:InducedAttentionBlock.\_\_call\_\_].
- `Preprocessor` computes imputation means and standardization statistics from training rows only (sample std, `ddof=1`,
  std of 1 for a single training row or zero-variance columns), clips to `[-100, 100]`, and stacks `feature_group_size`
  column copies, copy `i` rolled with `jnp.roll(x, -(2**i), axis=1)` for `i` in `range(feature_group_size)` (shifts 1,
  2, 4, ..., so no unshifted copy is kept); NaN/+inf/-inf indicators use values `-2`, `2`, `4` and are rolled and
  appended the same way [preprocessing.py:Preprocessor.\_\_call\_\_].
- `SoftmaxScaling` always runs its MLPs with `inference=True` [attention.py:SoftmaxScaling.\_\_call\_\_].
- The attention blocks accept `key=None` by returning `None` subkeys; otherwise keys are split with `split_for_mode`
  [blocks.py:\_split_optional_key]. `_call_mlp` replaces a `None` key with `jr.PRNGKey(0)` unless `inference=True`
  [mlp.py:\_call_mlp, src/equimo/core/\_prng.py:default_key_for_mode]; the blocks build their MLPs with
  `dropout_rate=0.0`, so that key does not affect their output \[blocks.py:CrossAttentionBlock,
  blocks.py:AttentionBlock, blocks.py:InContextAttentionBlock\].
- `FeatureDistributionEncoder.__call__` and `ColumnAggregator.__call__` take a required keyword `key`; it may be `None`
  only under `inference=True` (`ValueError` otherwise), because `BlockChunk` and `ColumnAggregator._row` split it with
  `split_for_mode` \[blocks.py:FeatureDistributionEncoder.\_\_call\_\_, blocks.py:ColumnAggregator.\_row,
  src/equimo/core/layers/generic.py:BlockChunk.\_\_call\_\_\]. Both reuse the same key for every column or row under
  `jax.vmap` [blocks.py:FeatureDistributionEncoder.\_\_call\_\_, blocks.py:ColumnAggregator.\_\_call\_\_].
- `ColumnAggregator` requires `depth >= 1`; its last layer is a cross-attention readout from the learned CLS tokens
  [blocks.py:ColumnAggregator.\_\_init\_\_, blocks.py:ColumnAggregator.\_row].
- `LinearLabelEmbedding` ignores `num_outputs`; `RegressionDecoder` ignores train embeddings, targets, and `n_train`
  [decoder.py:LinearLabelEmbedding, decoder.py:RegressionDecoder.\_\_call\_\_].

## Side Effects

- Importing the modules registers every decorated class in its family registry and in `_LAYER_REGISTRY`; the encoders
  `FeatureDistributionEncoder` and `ColumnAggregator` go only into `_LAYER_REGISTRY` via `register_layer`
  [blocks.py:FeatureDistributionEncoder, blocks.py:ColumnAggregator]. No I/O.

## Verification

- Run: `devenv shell -- uv run pytest tests/test_tabular_layers.py -q`
