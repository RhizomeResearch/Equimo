import importlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import equimo.audio.models as am
from equimo.audio import get_ast_preprocessing_spec, preprocess_ast_waveform

if os.environ.get("EQUIMO_TEST_OPTIONAL_EXTRA") == "audio":
    importlib.import_module("torchaudio")
else:
    pytest.importorskip("torchaudio")

DATA_DIR = Path(__file__).parents[1] / "data"
KEY = jax.random.PRNGKey(42)
CASES = (
    (
        "ast_base_patch16_audioset_10_10_0_4593",
        am.ast_base_patch16_audioset_10_10_0_4593,
    ),
    (
        "ast_base_patch16_speechcommands_v2_10_10_0_9812",
        am.ast_base_patch16_speechcommands_v2_10_10_0_9812,
    ),
)


def _waveform():
    time = np.arange(16_000, dtype=np.float32) / 16_000
    return (
        0.35 * np.sin(2 * np.pi * 440 * time) + 0.15 * np.sin(2 * np.pi * 997 * time)
    ).astype(np.float32)


@pytest.mark.parametrize(("variant", "factory"), CASES)
def test_raw_waveform_matches_pinned_upstream_features(variant, factory):
    del factory
    reference = np.load(DATA_DIR / f"{variant}_raw_audio_reference.npz")
    actual = preprocess_ast_waveform(
        _waveform(), 16_000, spec=get_ast_preprocessing_spec(variant)
    )

    np.testing.assert_array_equal(actual, reference["input_values"])


@pytest.mark.parametrize(("variant", "factory"), CASES)
def test_raw_waveform_matches_pinned_upstream_ast_outputs(variant, factory):
    reference = np.load(DATA_DIR / f"{variant}_raw_audio_reference.npz")
    input_values = jnp.asarray(reference["input_values"])
    model = factory(pretrained=True)

    features = model.forward_features(input_values, key=KEY, inference=True)
    logits = model(input_values, key=KEY, inference=True)

    np.testing.assert_allclose(
        features["x_norm_cls_token"], reference["cls_token"], atol=5e-4, rtol=0
    )
    np.testing.assert_allclose(
        features["x_norm_dist_token"], reference["dist_token"], atol=5e-4, rtol=0
    )
    np.testing.assert_allclose(logits, reference["logits"], atol=5e-4, rtol=0)
