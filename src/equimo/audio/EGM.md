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

- `AudioPreprocessingSpec.__post_init__` raises `ValueError` unless all of these hold: non-empty `variant`, `checkpoint`
  and `checkpoint_revision`; positive `sample_rate`, `resampling_lowpass_filter_width`, `frame_length_ms`,
  `frame_shift_ms`, `max_frames`, `normalization_std` and `normalization_scale`; `resampling_rolloff` in `(0, 1]`;
  `mean_to_mono` channels, `float32_unit_range` amplitudes, `sinc_interp_hann` resampling, Hanning window, zero dither;
  `round_to_power_of_two` set and `fft_length` equal to the next power of two covering one frame; `low_frequency_hz` in
  `[0, Nyquist)` and `high_frequency_hz` at most Nyquist; `num_mel_bins > 3`; Kaldi mel scale with natural log;
  log-power fbanks without an energy bin (`use_log_fbank` and `use_power` set, `use_energy` unset); normalization
  `after_crop_pad`; `truncate_end` / `pad_end`; and a finite `padding_value`
  [io.py:AudioPreprocessingSpec.\_\_post_init\_\_]. The fields `feature_extractor`, `feature_extractor_version`,
  `energy_floor`, `preemphasis_coefficient`, `raw_energy`, `remove_dc_offset`, `snip_edges`, `subtract_mean` and
  `normalization_mean` are not validated.
- Waveforms must be 1D or 2D and non-empty. Multi-channel input is averaged to mono in float32, and the finiteness and
  `[-1, 1]` checks apply to that mono mix, so channels of `+2.0` and `-2.0` pass [io.py:\_mono_float32]. `sampling_rate`
  must be an `int` other than `bool` (`TypeError` otherwise) and positive (`ValueError` otherwise)
  [io.py:preprocess_ast_waveform].
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
