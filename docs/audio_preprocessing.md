# Checkpoint-faithful AST waveform preprocessing

Equimo exposes raw-waveform preprocessing only for the two AST checkpoints
whose upstream contracts and end-to-end outputs are pinned in this repository.
Install it with `pip install "equimo[audio]"`. The base package and precomputed
spectrogram path do not import Torch or TorchAudio.

## Authoritative contracts

The contracts come from `transformers.ASTFeatureExtractor` 5.13.1 and the
`preprocessor_config.json` files at the exact model revisions below.

| Equimo variant | Upstream checkpoint and revision | Frames | Mean | Standard deviation |
|---|---|---:|---:|---:|
| `ast_base_patch16_audioset_10_10_0_4593` | `MIT/ast-finetuned-audioset-10-10-0.4593@f826b80d28226b62986cc218e5cec390b1096902` | 1024 | -4.2677393 | 4.5689974 |
| `ast_base_patch16_speechcommands_v2_10_10_0_9812` | `MIT/ast-finetuned-speech-commands-v2@315b0b847a3ca207e68b718503ad72066612eacd` | 128 | -6.845978 | 5.5654526 |

Both use normalized float32 audio at 16 kHz, 25 ms frames, a 10 ms hop, a
512-point FFT, Hanning windows, 128 Kaldi mel bins from 20 Hz to Nyquist,
0.97 pre-emphasis, DC-offset removal, no dither, power spectra, and the natural
logarithm. Complete frames are retained from the start; excess frames are
discarded at the end. Short inputs are zero-padded at the end before applying
`(fbank - mean) / (2 * std)`. `AudioPreprocessingSpec` records these settings,
the resampler settings, and checkpoint provenance as an immutable value.

Upstream requires callers to provide 16 kHz samples. Equimo's explicitly
recorded resampling policy uses TorchAudio's deterministic
`sinc_interp_hann`, low-pass width 6, and rolloff 0.99. Channel-first arrays and
PCM WAV channels are averaged to mono. Array amplitudes outside `[-1, 1]`,
empty arrays, non-finite samples, and ambiguous dimensions are rejected.

## Dependency decision

Measured installed sizes below are from the locked Linux Python 3.12 reference
environment and are directional rather than wheel-size promises.

| Candidate | Parity and determinism | Packaging/JAX boundary | Decision |
|---|---|---|---|
| TorchAudio 2.11 Kaldi filter bank + resampler | Exact intermediate parity (`max_abs=0`) with dither disabled; same maintained backend selected by the pinned upstream extractor | BSD; native wheels; about 9.7 MiB plus its required Torch stack (about 1.2 GiB in the CUDA-enabled reference environment); Python 3.12–3.14 resolves in the project lock; CPU/NumPy boundary | Selected as the `audio` extra |
| Transformers extractor | Exact, but its public extractor is checkpoint-loading machinery, still uses Torch, and does not resample mismatched rates | About 54 MiB plus transitive dependencies; unnecessary runtime model-hub surface | Kept in the maintainer-only reference workflow |
| librosa/SoundFile/soxr composition | Maintained and convenient for general audio, but STFT, mel, and padding conventions do not directly implement the pinned Kaldi contract | Additional native codec/resampler libraries; would require an independent compatibility layer | Rejected for this narrow path |
| Local NumPy/JAX DSP implementation | Could remove Torch after a substantial reimplementation and cross-platform tolerance study | Small runtime footprint but a new DSP maintenance surface | Rejected by the spike's scope limit |

WAV decoding uses Python's standard library and deliberately supports only
local, uncompressed 16-bit PCM. General codecs, remote media, streaming,
training augmentation, and automatic checkpoint discovery remain out of scope.

## Parity evidence and resource check

`models/audio_preprocessing.py` generates a license-neutral one-second mixture
of 440 Hz and 997 Hz sine waves. It records upstream filter-bank arrays, AST
class/distillation features, and logits for both pinned revisions. The fixture
hashes and schemas live in `tests/data/reference_provenance.json`.

With TorchAudio 2.11.0 and Transformers 5.13.1, Equimo's intermediate arrays
were byte-identical to upstream. Converted-model mean absolute errors were at
most `4.1e-6`, under the existing `5e-4` checkpoint tolerance. On the reference
Linux CPU environment, one-second AudioSet preprocessing averaged 2.3 ms after
warm-up across ten runs. Peak process RSS was 534 MiB including Python, NumPy,
Torch, and TorchAudio initialization; this is a pathology check, not a
performance guarantee.

Regenerate into a temporary directory for review:

```bash
uv sync --locked --group dev --group reference --extra audio
uv run --group reference --extra audio python models/audio_preprocessing.py \
  --output-dir /tmp/equimo-audio-references
uv run python models/validate_references.py
```
