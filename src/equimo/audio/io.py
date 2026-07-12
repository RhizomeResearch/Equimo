"""Checkpoint-linked waveform preprocessing for pretrained AST models."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType

import numpy as np

__all__ = [
    "AudioPreprocessingSpec",
    "get_ast_preprocessing_spec",
    "load_ast_wav",
    "preprocess_ast_waveform",
]


@dataclass(frozen=True, slots=True)
class AudioPreprocessingSpec:
    """Immutable preprocessing contract for one upstream AST checkpoint."""

    variant: str
    checkpoint: str
    checkpoint_revision: str
    feature_extractor: str
    feature_extractor_version: str
    sample_rate: int
    channel_policy: str
    amplitude_convention: str
    resampling_method: str
    resampling_lowpass_filter_width: int
    resampling_rolloff: float
    frame_length_ms: float
    frame_shift_ms: float
    fft_length: int
    window_type: str
    dither: float
    energy_floor: float
    low_frequency_hz: float
    high_frequency_hz: float
    num_mel_bins: int
    preemphasis_coefficient: float
    raw_energy: bool
    remove_dc_offset: bool
    round_to_power_of_two: bool
    snip_edges: bool
    subtract_mean: bool
    use_energy: bool
    use_log_fbank: bool
    use_power: bool
    mel_scale: str
    log_transform: str
    normalization_mean: float
    normalization_std: float
    normalization_scale: float
    normalization_order: str
    max_frames: int
    crop_policy: str
    padding_policy: str
    padding_value: float

    def __post_init__(self) -> None:
        if not self.variant or not self.checkpoint or not self.checkpoint_revision:
            raise ValueError("Variant, checkpoint, and revision must be non-empty.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.channel_policy != "mean_to_mono":
            raise ValueError("Only the 'mean_to_mono' channel policy is supported.")
        if self.amplitude_convention != "float32_unit_range":
            raise ValueError(
                "Only normalized float32 waveform amplitudes are supported."
            )
        if self.resampling_method != "sinc_interp_hann":
            raise ValueError(
                "Only TorchAudio's 'sinc_interp_hann' resampler is supported."
            )
        if self.resampling_lowpass_filter_width <= 0:
            raise ValueError("resampling_lowpass_filter_width must be positive.")
        if not 0.0 < self.resampling_rolloff <= 1.0:
            raise ValueError("resampling_rolloff must be in (0, 1].")
        if self.frame_length_ms <= 0 or self.frame_shift_ms <= 0:
            raise ValueError("Frame length and shift must be positive.")
        frame_length = round(self.sample_rate * self.frame_length_ms / 1000)
        expected_fft_length = 1 << (frame_length - 1).bit_length()
        if not self.round_to_power_of_two or self.fft_length != expected_fft_length:
            raise ValueError(
                "fft_length must be the next power of two covering one frame."
            )
        if self.window_type != "hanning":
            raise ValueError(
                "Checkpoint-faithful AST preprocessing uses a Hanning window."
            )
        if self.dither != 0.0:
            raise ValueError("Dither must be disabled for deterministic preprocessing.")
        if not 0 <= self.low_frequency_hz < self.sample_rate / 2:
            raise ValueError("low_frequency_hz must be below Nyquist.")
        if self.high_frequency_hz > self.sample_rate / 2:
            raise ValueError("high_frequency_hz must not exceed Nyquist.")
        if self.num_mel_bins <= 3:
            raise ValueError("num_mel_bins must be greater than three.")
        if self.mel_scale != "kaldi" or self.log_transform != "natural_log":
            raise ValueError(
                "Only Kaldi mel bins with a natural logarithm are supported."
            )
        if not self.use_log_fbank or not self.use_power or self.use_energy:
            raise ValueError(
                "AST requires log-power filter banks without an energy bin."
            )
        if self.normalization_std <= 0 or self.normalization_scale <= 0:
            raise ValueError(
                "Normalization scale and standard deviation must be positive."
            )
        if self.normalization_order != "after_crop_pad":
            raise ValueError("AST normalization must occur after cropping or padding.")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        if self.crop_policy != "truncate_end" or self.padding_policy != "pad_end":
            raise ValueError(
                "Only upstream AST end-cropping and end-padding are supported."
            )
        if not np.isfinite(self.padding_value):
            raise ValueError("padding_value must be finite.")


def _spec(
    *,
    variant: str,
    checkpoint: str,
    revision: str,
    max_frames: int,
    mean: float,
    std: float,
) -> AudioPreprocessingSpec:
    return AudioPreprocessingSpec(
        variant=variant,
        checkpoint=checkpoint,
        checkpoint_revision=revision,
        feature_extractor="transformers.ASTFeatureExtractor",
        feature_extractor_version="5.13.1",
        sample_rate=16_000,
        channel_policy="mean_to_mono",
        amplitude_convention="float32_unit_range",
        resampling_method="sinc_interp_hann",
        resampling_lowpass_filter_width=6,
        resampling_rolloff=0.99,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        fft_length=512,
        window_type="hanning",
        dither=0.0,
        energy_floor=1.0,
        low_frequency_hz=20.0,
        high_frequency_hz=0.0,
        num_mel_bins=128,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        snip_edges=True,
        subtract_mean=False,
        use_energy=False,
        use_log_fbank=True,
        use_power=True,
        mel_scale="kaldi",
        log_transform="natural_log",
        normalization_mean=mean,
        normalization_std=std,
        normalization_scale=2.0,
        normalization_order="after_crop_pad",
        max_frames=max_frames,
        crop_policy="truncate_end",
        padding_policy="pad_end",
        padding_value=0.0,
    )


_AST_PREPROCESSING_SPECS = MappingProxyType(
    {
        "ast_base_patch16_audioset_10_10_0_4593": _spec(
            variant="ast_base_patch16_audioset_10_10_0_4593",
            checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593",
            revision="f826b80d28226b62986cc218e5cec390b1096902",
            max_frames=1024,
            mean=-4.2677393,
            std=4.5689974,
        ),
        "ast_base_patch16_speechcommands_v2_10_10_0_9812": _spec(
            variant="ast_base_patch16_speechcommands_v2_10_10_0_9812",
            checkpoint="MIT/ast-finetuned-speech-commands-v2",
            revision="315b0b847a3ca207e68b718503ad72066612eacd",
            max_frames=128,
            mean=-6.845978,
            std=5.5654526,
        ),
    }
)


def get_ast_preprocessing_spec(variant: str) -> AudioPreprocessingSpec:
    """Return the pinned waveform preprocessing contract for ``variant``."""
    try:
        return _AST_PREPROCESSING_SPECS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(_AST_PREPROCESSING_SPECS))
        raise ValueError(
            f"No raw-waveform preprocessing spec is available for {variant!r}. "
            f"Supported variants: {supported}."
        ) from exc


def _require_audio_dependencies():
    try:
        import torch  # ty: ignore[unresolved-import]
        from torchaudio.compliance.kaldi import fbank  # ty: ignore[unresolved-import]
        from torchaudio.functional import resample  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "PyTorch and TorchAudio are required for checkpoint-faithful AST waveform "
            "preprocessing. Install Equimo with the 'audio' extra (for example, "
            'pip install "equimo[audio]").'
        ) from exc
    return torch, fbank, resample


def _mono_float32(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform)
    if array.ndim not in (1, 2):
        raise ValueError("waveform must have shape (samples,) or (channels, samples).")
    if array.ndim == 2:
        if array.shape[0] == 0:
            raise ValueError("waveform must contain at least one channel.")
        array = np.mean(array.astype(np.float32), axis=0)
    else:
        array = array.astype(np.float32)
    if array.size == 0:
        raise ValueError("waveform must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError("waveform samples must all be finite.")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("waveform samples must use the normalized [-1, 1] range.")
    return np.ascontiguousarray(array)


def preprocess_ast_waveform(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    spec: AudioPreprocessingSpec,
) -> np.ndarray:
    """Convert one normalized waveform to checkpoint-faithful AST input values.

    Waveform arrays are mono ``(samples,)`` or channel-first
    ``(channels, samples)``. Multi-channel inputs are averaged to mono. This is
    a deterministic CPU boundary and is not JAX-jittable.
    """
    if not isinstance(sampling_rate, int) or isinstance(sampling_rate, bool):
        raise TypeError("sampling_rate must be an integer.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    array = _mono_float32(waveform)
    frame_samples = round(spec.sample_rate * spec.frame_length_ms / 1000)
    if sampling_rate == spec.sample_rate and array.size < frame_samples:
        raise ValueError(
            f"waveform must contain at least {frame_samples} samples after resampling."
        )
    torch, fbank, resample = _require_audio_dependencies()
    tensor = torch.from_numpy(array)
    if sampling_rate != spec.sample_rate:
        tensor = resample(
            tensor,
            sampling_rate,
            spec.sample_rate,
            lowpass_filter_width=spec.resampling_lowpass_filter_width,
            rolloff=spec.resampling_rolloff,
            resampling_method=spec.resampling_method,
        )
    if tensor.numel() < frame_samples:
        raise ValueError(
            f"waveform must contain at least {frame_samples} samples after resampling."
        )

    features = fbank(
        tensor.unsqueeze(0),
        dither=spec.dither,
        energy_floor=spec.energy_floor,
        frame_length=spec.frame_length_ms,
        frame_shift=spec.frame_shift_ms,
        high_freq=spec.high_frequency_hz,
        htk_compat=False,
        low_freq=spec.low_frequency_hz,
        min_duration=0.0,
        num_mel_bins=spec.num_mel_bins,
        preemphasis_coefficient=spec.preemphasis_coefficient,
        raw_energy=spec.raw_energy,
        remove_dc_offset=spec.remove_dc_offset,
        round_to_power_of_two=spec.round_to_power_of_two,
        sample_frequency=spec.sample_rate,
        snip_edges=spec.snip_edges,
        subtract_mean=spec.subtract_mean,
        use_energy=spec.use_energy,
        use_log_fbank=spec.use_log_fbank,
        use_power=spec.use_power,
        window_type=spec.window_type,
    )
    features = features[: spec.max_frames]
    if features.shape[0] < spec.max_frames:
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, spec.max_frames - features.shape[0]),
            value=spec.padding_value,
        )
    features = (features - spec.normalization_mean) / (
        spec.normalization_std * spec.normalization_scale
    )
    return features.numpy().astype(np.float32, copy=False)


def _decode_pcm16_wav(path: str | PathLike[str]) -> tuple[np.ndarray, int]:
    with wave.open(str(Path(path)), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("Only uncompressed 16-bit PCM WAV files are supported.")
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    waveform = samples.astype(np.float32).T / 32768.0
    return waveform, sample_rate


def load_ast_wav(
    path: str | PathLike[str], *, spec: AudioPreprocessingSpec
) -> np.ndarray:
    """Decode a local 16-bit PCM WAV file and preprocess it for AST."""
    waveform, sampling_rate = _decode_pcm16_wav(path)
    return preprocess_ast_waveform(waveform, sampling_rate, spec=spec)
