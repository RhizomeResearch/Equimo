import builtins
import importlib
import os
import wave
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from equimo.audio import (
    get_ast_preprocessing_spec,
    load_ast_wav,
    preprocess_ast_waveform,
)

AUDIOSPEC = get_ast_preprocessing_spec("ast_base_patch16_audioset_10_10_0_4593")
SPEECHSPEC = get_ast_preprocessing_spec(
    "ast_base_patch16_speechcommands_v2_10_10_0_9812"
)


def _require_audio_extra():
    if os.environ.get("EQUIMO_TEST_OPTIONAL_EXTRA") == "audio":
        return importlib.import_module("torchaudio")
    return pytest.importorskip("torchaudio")


def _tone(sample_rate=16_000, seconds=1.0):
    time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (
        0.35 * np.sin(2 * np.pi * 440 * time) + 0.15 * np.sin(2 * np.pi * 997 * time)
    ).astype(np.float32)


def test_checkpoint_specs_are_immutable_and_complete():
    assert AUDIOSPEC.sample_rate == 16_000
    assert AUDIOSPEC.max_frames == 1024
    assert AUDIOSPEC.num_mel_bins == 128
    assert SPEECHSPEC.max_frames == 128
    assert SPEECHSPEC.normalization_mean == -6.845978
    assert asdict(AUDIOSPEC)["checkpoint_revision"] == (
        "f826b80d28226b62986cc218e5cec390b1096902"
    )
    assert type(AUDIOSPEC)(**asdict(AUDIOSPEC)) == AUDIOSPEC
    with pytest.raises(FrozenInstanceError):
        AUDIOSPEC.max_frames = 10


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_rate", 0, "sample_rate"),
        ("channel_policy", "left", "channel policy"),
        ("fft_length", 400, "power of two"),
        ("normalization_std", 0.0, "Normalization"),
        ("crop_policy", "center", "end-cropping"),
    ],
)
def test_checkpoint_spec_rejects_incompatible_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(AUDIOSPEC, **{field: value})


def test_unknown_variant_names_supported_specs():
    with pytest.raises(ValueError, match="Supported variants"):
        get_ast_preprocessing_spec("ast_base_patch16_224")


def test_waveform_happy_path_padding_and_dtype():
    _require_audio_extra()
    features = preprocess_ast_waveform(_tone(), 16_000, spec=AUDIOSPEC)

    assert features.shape == (1024, 128)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    expected_padding = (0.0 - AUDIOSPEC.normalization_mean) / (
        AUDIOSPEC.normalization_std * AUDIOSPEC.normalization_scale
    )
    np.testing.assert_allclose(features[-1], expected_padding)


def test_multichannel_policy_averages_channels():
    _require_audio_extra()
    waveform = _tone()
    stereo = np.stack([waveform, waveform * 0.5])

    actual = preprocess_ast_waveform(stereo, 16_000, spec=SPEECHSPEC)
    expected = preprocess_ast_waveform(waveform * 0.75, 16_000, spec=SPEECHSPEC)

    np.testing.assert_array_equal(actual, expected)


def test_resampling_is_deterministic():
    _require_audio_extra()
    waveform = _tone(sample_rate=8_000)

    first = preprocess_ast_waveform(waveform, 8_000, spec=SPEECHSPEC)
    second = preprocess_ast_waveform(waveform, 8_000, spec=SPEECHSPEC)

    np.testing.assert_array_equal(first, second)


def test_long_waveform_is_truncated_at_frame_boundary():
    _require_audio_extra()
    waveform = _tone(seconds=2.0)
    samples_for_128_frames = 400 + 127 * 160

    long_features = preprocess_ast_waveform(waveform, 16_000, spec=SPEECHSPEC)
    boundary_features = preprocess_ast_waveform(
        waveform[:samples_for_128_frames], 16_000, spec=SPEECHSPEC
    )

    np.testing.assert_array_equal(long_features, boundary_features)


@pytest.mark.parametrize(
    ("waveform", "sampling_rate", "error", "message"),
    [
        (np.array([], dtype=np.float32), 16_000, ValueError, "one sample"),
        (np.zeros(399, dtype=np.float32), 16_000, ValueError, "at least 400"),
        (np.zeros((1, 1, 1), dtype=np.float32), 16_000, ValueError, "shape"),
        (np.array([np.nan], dtype=np.float32), 16_000, ValueError, "finite"),
        (np.array([1.1], dtype=np.float32), 16_000, ValueError, "normalized"),
        (np.zeros(400, dtype=np.float32), 0, ValueError, "positive"),
        (np.zeros(400, dtype=np.float32), 16_000.0, TypeError, "integer"),
    ],
)
def test_invalid_waveform_inputs(waveform, sampling_rate, error, message):
    with pytest.raises(error, match=message):
        preprocess_ast_waveform(waveform, sampling_rate, spec=SPEECHSPEC)


def test_missing_audio_extra_has_actionable_error(monkeypatch):
    original_import = builtins.__import__
    missing = ImportError("No module named 'torchaudio'")

    def import_without_torchaudio(name, *args, **kwargs):
        if name == "torchaudio" or name.startswith("torchaudio."):
            raise missing
        if name == "torch":
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torchaudio)
    with pytest.raises(ImportError) as exc_info:
        preprocess_ast_waveform(_tone(), 16_000, spec=SPEECHSPEC)

    assert exc_info.value.__cause__ is missing
    assert "'audio' extra" in str(exc_info.value)
    assert 'pip install "equimo[audio]"' in str(exc_info.value)


def test_load_ast_wav_decodes_pcm16_and_applies_channel_policy(tmp_path):
    _require_audio_extra()
    waveform = _tone()
    stereo = np.stack([waveform, waveform * 0.5])
    interleaved = np.round(stereo.T * 32767).astype("<i2")
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(interleaved.tobytes())

    actual = load_ast_wav(path, spec=SPEECHSPEC)
    decoded = interleaved.astype(np.float32).T / 32768.0
    expected = preprocess_ast_waveform(decoded, 16_000, spec=SPEECHSPEC)

    np.testing.assert_array_equal(actual, expected)
