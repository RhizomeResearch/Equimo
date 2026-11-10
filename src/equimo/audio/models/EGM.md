# equimo.audio.models

## Overview

Audio model families. The package `__init__` auto-imports every sibling module and re-exports its `__all__`, so adding a
model module here makes it visible through `equimo.audio.models` and the modality-scoped model registry. The only family
today is the Audio Spectrogram Transformer (AST).

## Interface

- `AudioSpectrogramTransformer(input_fdim, input_tdim, dim, patch_size, fstride, tstride, num_heads, depths, *, key, ...)`
  \[ast.py:AudioSpectrogramTransformer\]: registered as model `"ast"` with `modality="audio"`. Consumes one
  `(time, frequency)` spectrogram.
  - `features(x, key=None, inference=None)` returns pre-norm tokens `(2 + num_patches, dim)`
    [ast.py:AudioSpectrogramTransformer.features].
  - `intermediate_features(x, key=None, inference=None, indices=None, n_last_blocks=None)` returns a tuple of selected
    block outputs [ast.py:AudioSpectrogramTransformer.intermediate_features].
  - `forward_features(...)` returns a dict with `x_norm_cls_token`, `x_norm_dist_token`, `x_norm_patchtokens`,
    `x_prenorm` [ast.py:AudioSpectrogramTransformer.forward_features].
  - `__call__(...)` returns classification logits `(num_classes,)` (or pooled features when the head is `Identity`)
    [ast.py:AudioSpectrogramTransformer.\_\_call\_\_].
- Variant factories, each `(**kwargs) -> AudioSpectrogramTransformer` accepting `pretrained`, `inference_mode`, `key`,
  and config overrides \[ast.py:ast_tiny_patch16_224, ast.py:ast_small_patch16_224, ast.py:ast_base_patch16_224,
  ast.py:ast_base_patch16_384, ast.py:ast_base_patch16_audioset_10_10_0_4593,
  ast.py:ast_base_patch16_speechcommands_v2_10_10_0_9812\]. Configs live in `_AST_REGISTRY` [ast.py:\_AST_REGISTRY].
- Re-exported registry helpers `get_model_cls`, `register_model` from `equimo.registry` [\_\_init\_\_.py].
- Internal catalog hook `_catalog_model_variants()` describing `audio/ast_base_patch16_audioset_10_10_0_4593`
  [ast.py:\_catalog_model_variants], consumed by `equimo.catalog`.

## Invariants

- Two prefix tokens (CLS then distillation) precede the patch tokens: `num_prefix_tokens = 2`, and `pos_embed` has shape
  `(num_patches + 2, dim)` [ast.py:AudioSpectrogramTransformer.\_\_init\_\_]. Prefix tokens are cast to the patch
  embedding's output dtype before concatenation, and positional embeddings are cast to that dtype and added afterwards
  [ast.py:AudioSpectrogramTransformer.\_prepare_tokens].
- `global_pool="token"` averages the normalized CLS and distillation tokens; other modes go through
  `equimo.utils.pool_sd`. `"avg"`, `"avgmax"` and `"max"` exclude prefix tokens, while `"cls_patch_mean"` concatenates
  the CLS token with the patch-token mean and doubles the head input width to `2 * dim`
  \[ast.py:AudioSpectrogramTransformer.\_\_call\_\_, ast.py:AudioSpectrogramTransformer.\_\_init\_\_,
  src/equimo/utils.py:pool_sd\].
- The constructor accepts and discards extra `**kwargs`, so a mistyped factory override is ignored
  [ast.py:AudioSpectrogramTransformer.\_\_init\_\_, ast.py:\_build_ast].
- `num_classes=None` or `<= 0` replaces the head with `eqx.nn.Identity`; `head_norm_layer=None` gives an identity head
  norm [ast.py:AudioSpectrogramTransformer.\_\_init\_\_].
- Keys are split with `split_for_mode`, so forward methods run without a key only when `inference=True`
  [ast.py:AudioSpectrogramTransformer.features].
- `pretrained=True` is accepted only for `ast_base_patch16_audioset_10_10_0_4593` and
  `ast_base_patch16_speechcommands_v2_10_10_0_9812` [ast.py:\_AST_PRETRAINED_VARIANTS, ast.py:\_build_ast]. These two
  variants use `exactgelu`, a LayerNorm head norm, and `eps=1e-12`; the SpeechCommands variant uses `input_tdim=128` and
  35 classes [ast.py:\_AST_REGISTRY].
- Model modules may export public names only through `__all__`; the auto-importer skips a module named `registry`
  [\_\_init\_\_.py].

## Side Effects

- Importing the package imports every model module and registers `AudioSpectrogramTransformer` in the global model
  registry via `@register_model("ast", modality="audio")` [ast.py:AudioSpectrogramTransformer].
- `pretrained=True` makes `equimo.core.factory.build_model_variant` load weights through
  `equimo.serialization.load_weights`, which downloads the checkpoint \[ast.py:\_build_ast,
  src/equimo/core/factory.py:build_model_variant\].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_audio_ast.py -q`
