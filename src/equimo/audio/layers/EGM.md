# equimo.audio.layers

## Overview

Audio-specific Equinox layers. Today this is the spectrogram-to-token patch embedding used by the Audio Spectrogram
Transformer, plus name-keyed registries so model configs can refer to audio layers by string.

## Interface

- `SpectrogramPatchEmbedding(embed_dim, patch_size, *, input_fdim, input_tdim, fstride, tstride, key)`
  \[patch.py:SpectrogramPatchEmbedding\]: maps one `(time, frequency)` log-mel spectrogram to `(num_patches, embed_dim)`
  tokens through a single-input-channel 2D `eqx.nn.Conv`. Exposes static `patch_size`, `stride`, `img_size`
  (`(input_fdim, input_tdim)`), `grid_size`, and `num_patches`.
- `register_patch(name=None, force=False)` / `get_patch(module)` \[patch.py:register_patch, patch.py:get_patch\]:
  register and resolve audio patch modules in `_PATCH_REGISTRY`; built from `equimo.core.layers._registry.make_register`
  / `make_get`.
- `register_layer(name=None, force=False)` / `get_layer(module)` \[\_\_init\_\_.py:register_layer,
  \_\_init\_\_.py:get_layer\]: generic audio layer registry `_AUDIO_LAYER_REGISTRY` (empty by default).

## Invariants

- Inputs are unbatched `(time, frequency)` arrays; the module rearranges them to `(1, frequency, time)` before the
  convolution so `fstride` and `tstride` act on the AST reference axes, then flattens tokens in frequency-major `(f t)`
  order [patch.py:SpectrogramPatchEmbedding.\_\_call\_\_]. Batch with `jax.vmap`.
- The input shape must equal `(input_tdim, input_fdim)` exactly: a 2D mismatch raises `AssertionError` (explicit raise,
  not an `assert`), and a non-2D input raises `ValueError` when its shape is unpacked
  [patch.py:SpectrogramPatchEmbedding.\_\_call\_\_].
- The constructor raises `ValueError` when the patch does not fit, i.e. when
  `(input_fdim - patch_f) // fstride + 1 <= 0` or the time equivalent [patch.py:SpectrogramPatchEmbedding.\_\_init\_\_].
- Registry names are lowercased class names by default; registering an existing name without `force=True` raises
  `ValueError`, and only `eqx.Module` subclasses are accepted (`TypeError` otherwise)
  [src/equimo/core/layers/\_registry.py:make_register]. Unknown string names raise `ValueError` listing the available
  entries [src/equimo/core/layers/\_registry.py:make_get].

## Side Effects

- Importing `patch.py` registers `spectrogrampatchembedding` in the module-level `_PATCH_REGISTRY`
  [patch.py:SpectrogramPatchEmbedding]. No I/O.

## Verification

- Run: `devenv shell -- uv run pytest tests/test_audio_patch.py -q`
