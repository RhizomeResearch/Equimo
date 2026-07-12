"""Calibration artifact validation and serialization tests."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

import equimo.finetune as eqft


def _artifacts():
    state = eqft.initialize_calibration_collector(
        kind="activation_svd",
        logical_parameter_dims={"a": 2, "b": 2},
        base_checkpoint_hash="sha256:checkpoint",
        data_fingerprint="sha256:data",
        rank=1,
    )
    state = eqft.update_calibration_collector(
        state,
        {"a": jnp.eye(2), "b": jnp.asarray([[2.0, 0.0], [0.0, 1.0]])},
        sample_count={"a": 2, "b": 2},
    )
    return eqft.finalize_calibration_collector(state)


def test_calibration_artifact_codec_roundtrip(tmp_path):
    path = tmp_path / "calibration.eqft"
    artifacts = _artifacts()
    eqft.save_calibration_artifacts(path, artifacts)
    loaded = eqft.load_calibration_artifacts(path)

    assert tuple(loaded) == ("a", "b")
    for logical_id in artifacts:
        expected = artifacts[logical_id]
        actual = loaded[logical_id]
        assert actual.kind == expected.kind
        assert actual.base_checkpoint_hash == expected.base_checkpoint_hash
        assert actual.logical_parameter_ids == (logical_id,)
        assert actual.sample_count == expected.sample_count
        assert actual.data_fingerprint == expected.data_fingerprint
        assert actual.accumulation_dtype == expected.accumulation_dtype
        assert actual.distributed_reduction == "none"
        assert actual.statistics["normalization"] == "sample_mean"
        assert jnp.array_equal(
            actual.statistics["right_singular_vectors"],
            expected.statistics["right_singular_vectors"],
        )
        assert jnp.array_equal(
            actual.statistics["singular_values"],
            expected.statistics["singular_values"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda artifacts: {
                **artifacts,
                "b": replace(artifacts["b"], base_checkpoint_hash="other"),
            },
            "checkpoint",
        ),
        (
            lambda artifacts: {
                **artifacts,
                "b": replace(artifacts["b"], data_fingerprint="other"),
            },
            "fingerprint",
        ),
        (
            lambda artifacts: {
                **artifacts,
                "b": replace(artifacts["b"], logical_parameter_ids=("a",)),
            },
            "duplicate logical parameter ID|does not identify mapping key",
        ),
        (
            lambda artifacts: {
                **artifacts,
                "b": replace(artifacts["b"], kind="input_covariance"),
            },
            "kind",
        ),
        (
            lambda artifacts: {
                **artifacts,
                "b": replace(
                    artifacts["b"],
                    statistics={
                        **artifacts["b"].statistics,
                        "singular_values": jnp.ones((2,)),
                    },
                ),
            },
            "incompatible SVD shapes",
        ),
    ),
)
def test_artifact_codec_rejects_mixed_or_duplicate_identity(
    tmp_path, mutation, message
):
    with pytest.raises(ValueError, match=message):
        eqft.save_calibration_artifacts(
            tmp_path / "invalid.eqft", mutation(_artifacts())
        )


def test_regmean_rejects_wrong_artifact_kind():
    artifact = next(iter(_artifacts().values()))
    with pytest.raises(ValueError, match="input_covariance"):
        eqft.regmean_merge([jnp.eye(2)], [artifact])
