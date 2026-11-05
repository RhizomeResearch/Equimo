# equimo.audio

## Overview

Audio modality: AST models and layers that operate on log-mel spectrograms, plus a checkpoint-pinned CPU adapter that
turns raw waveforms or 16-bit PCM WAV files into the exact spectrogram inputs the pretrained AST checkpoints expect. The
package root exports only the preprocessing API; models and layers are imported from their subpackages.

## Interface

- `AudioPreprocessingSpec` \[io.py:AudioPreprocessingSpec\]: frozen, slotted dataclass recording the full upstream
  preprocessing contract (checkpoint and revision, sample rate, resampler, Kaldi fbank options, normalization, crop/pad
  policy).
- `get_ast_preprocessing_spec(variant) -> AudioPreprocessingSpec` \[io.py:get_ast_preprocessing_spec\]: returns the
  pinned spec for `ast_base_patch16_audioset_10_10_0_4593` or `ast_base_patch16_speechcommands_v2_10_10_0_9812`
  [io.py:\_AST_PREPROCESSING_SPECS]; any other name raises `ValueError`.
- `preprocess_ast_waveform(waveform, sampling_rate, *, spec) -> np.ndarray` \[io.py:preprocess_ast_waveform\]: mono
  `(samples,)` or channel-first `(channels, samples)` float waveform to a `(spec.max_frames, spec.num_mel_bins)` float32
  NumPy array.
- `load_ast_wav(path, *, spec) -> np.ndarray` \[io.py:load_ast_wav\]: decodes an uncompressed 16-bit PCM WAV and calls
  `preprocess_ast_waveform`.
- Subpackages:
  - [layers](./layers/EGM.md): `SpectrogramPatchEmbedding` and audio layer/patch registries.
  - [models](./models/EGM.md): `AudioSpectrogramTransformer` (model `"ast"`) and its variant factories.

## Invariants

- `AudioPreprocessingSpec.__post_init__` only accepts the AST-faithful configuration: `mean_to_mono` channels,
  `float32_unit_range` amplitudes, `sinc_interp_hann` resampling, Hanning window, zero dither, `fft_length` equal to the
  next power of two covering one frame, Kaldi mel scale with natural log, log-power fbanks without an energy bin,
  normalization `after_crop_pad`, `truncate_end` / `pad_end`, positive normalization std and scale, and a finite padding
  value; anything else raises `ValueError` [io.py:AudioPreprocessingSpec.\_\_post_init\_\_].
- Waveforms must be 1D or 2D, non-empty, finite, and within `[-1, 1]`; multi-channel input is averaged to mono in
  float32 [io.py:\_mono_float32]. `sampling_rate` must be an `int` other than `bool` (`TypeError` otherwise) and
  positive (`ValueError` otherwise) [io.py:preprocess_ast_waveform].
- At least one frame of samples (`round(sample_rate * frame_length_ms / 1000)`) must remain after resampling
  [io.py:preprocess_ast_waveform].
- Features are truncated or end-padded to `max_frames`, then normalized as
  `(x - normalization_mean) / (normalization_std * normalization_scale)` [io.py:preprocess_ast_waveform].
- WAV input must be uncompressed with 2-byte samples; samples are scaled by `1 / 32768` [io.py:\_decode_pcm16_wav].
- Preprocessing is a host-side NumPy/PyTorch boundary and is not JAX-jittable [io.py:preprocess_ast_waveform]; the
  models themselves consume spectrograms only.

## Side Effects

- `load_ast_wav` reads a local file from disk [io.py:\_decode_pcm16_wav].
- `preprocess_ast_waveform` lazily imports `torch` and `torchaudio` (`compliance.kaldi.fbank`, `functional.resample`)
  and raises `ImportError` pointing to the `audio` extra when missing [io.py:\_require_audio_dependencies].
- Importing `audio.layers` and `audio.models` mutates module-level registries (see child contracts).

## Verification

- Run:
  `devenv shell -- uv run pytest tests/audio/test_io.py tests/audio/test_preprocessing_parity.py tests/test_audio_ast.py tests/test_audio_patch.py -q`
