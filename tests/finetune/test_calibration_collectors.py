"""Numerical and transformation tests for calibration collectors."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import equimo.finetune as eqft


def _initialize(kind="activation_covariance", **overrides):
    kwargs = {
        "kind": kind,
        "logical_parameter_dims": {"encoder.proj": 3},
        "base_checkpoint_hash": "sha256:checkpoint",
        "data_fingerprint": "sha256:preprocessing-and-data",
        "centered": False,
    }
    kwargs.update(overrides)
    return eqft.initialize_calibration_collector(**kwargs)


def test_streaming_covariance_matches_direct_masked_calculation():
    values = jnp.asarray(
        [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [4.0, 1.0, 2.0], [9.0, 9.0, 9.0]],
        dtype=jnp.float16,
    )
    mask = jnp.asarray([True, True, True, False])
    state = _initialize(centered=True)
    state = eqft.update_calibration_collector(
        state, {"encoder.proj": values}, sample_mask=mask
    )
    artifact = eqft.finalize_calibration_collector(state)["encoder.proj"]

    valid = values[:3].astype(jnp.float32)
    centered = valid - jnp.mean(valid, axis=0)
    expected = centered.T @ centered / valid.shape[0]
    assert artifact.sample_count == 3
    assert artifact.accumulation_dtype == "float32"
    assert artifact.statistics["normalization"] == "sample_mean"
    assert artifact.statistics["centered"] is True
    assert jnp.allclose(artifact.statistics["covariance"], expected, atol=1e-6)


def test_streaming_results_are_chunk_and_combine_invariant():
    values = jnp.arange(30, dtype=jnp.float32).reshape(10, 3) / 7
    direct = eqft.update_calibration_collector(
        _initialize(centered=True),
        {"encoder.proj": values},
        sample_count=10,
    )
    left = eqft.update_calibration_collector(
        _initialize(centered=True),
        {"encoder.proj": values[:4]},
        sample_count=4,
    )
    right = eqft.update_calibration_collector(
        _initialize(centered=True),
        {"encoder.proj": values[4:]},
        sample_count=6,
    )
    combined = eqft.combine_calibration_collectors(left, right)
    reverse_combined = eqft.combine_calibration_collectors(right, left)

    expected = eqft.finalize_calibration_collector(direct)["encoder.proj"]
    actual = eqft.finalize_calibration_collector(combined)["encoder.proj"]
    reverse = eqft.finalize_calibration_collector(reverse_combined)["encoder.proj"]
    assert actual.sample_count == expected.sample_count == 10
    assert jnp.allclose(
        actual.statistics["covariance"],
        expected.statistics["covariance"],
        atol=1e-6,
    )
    assert jnp.allclose(
        reverse.statistics["covariance"],
        expected.statistics["covariance"],
        atol=1e-6,
    )


def test_state_updates_are_jittable_and_pytree_compatible():
    state = _initialize()
    leaves = jax.tree.leaves(state)
    assert len(leaves) == 3

    update = jax.jit(
        lambda current, values, mask: eqft.update_calibration_collector(
            current, {"encoder.proj": values}, sample_mask=mask
        )
    )
    updated = update(
        state,
        jnp.ones((2, 4, 3), dtype=jnp.bfloat16),
        jnp.asarray([[True, True, False, False], [True, False, False, False]]),
    )
    artifact = eqft.finalize_calibration_collector(updated)["encoder.proj"]
    assert artifact.sample_count == 3
    assert artifact.statistics["covariance"].dtype == jnp.float32


def test_empty_and_insufficient_svd_states_fail_clearly():
    with pytest.raises(ValueError, match="no samples"):
        eqft.finalize_calibration_collector(_initialize())

    state = _initialize("activation_svd", centered=True, rank=2)
    state = eqft.update_calibration_collector(
        state,
        {"encoder.proj": jnp.ones((2, 3))},
        sample_count=2,
    )
    with pytest.raises(ValueError, match="effective samples"):
        eqft.finalize_calibration_collector(state)


def test_rank_limited_svd_reconstructs_covariance():
    values = jnp.asarray(
        [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]
    )
    state = _initialize("activation_svd", centered=True, rank=2)
    state = eqft.update_calibration_collector(
        state, {"encoder.proj": values}, sample_count=4
    )
    artifact = eqft.finalize_calibration_collector(state)["encoder.proj"]
    vh = artifact.statistics["right_singular_vectors"]
    singular_values = artifact.statistics["singular_values"]
    factor = singular_values[:, None] * vh
    expected = values.T @ values / values.shape[0]
    assert vh.shape == (2, 3)
    assert jnp.allclose(factor.T @ factor, expected, atol=1e-6)


def test_centered_singleton_is_zero_and_uncentered_is_outer_product():
    value = jnp.asarray([[1.0, 2.0, 3.0]])
    centered = eqft.update_calibration_collector(
        _initialize(centered=True), {"encoder.proj": value}, sample_count=1
    )
    uncentered = eqft.update_calibration_collector(
        _initialize(centered=False), {"encoder.proj": value}, sample_count=1
    )
    centered_cov = eqft.finalize_calibration_collector(centered)["encoder.proj"]
    uncentered_cov = eqft.finalize_calibration_collector(uncentered)["encoder.proj"]
    assert jnp.array_equal(centered_cov.statistics["covariance"], jnp.zeros((3, 3)))
    assert jnp.array_equal(uncentered_cov.statistics["covariance"], value.T @ value)


def test_input_covariance_matches_regmean_orientation_and_sum_normalization():
    inputs_a = jnp.asarray([[2.0, 0.0], [0.0, 1.0]])
    inputs_b = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    artifacts = []
    for checkpoint, inputs in (("a", inputs_a), ("b", inputs_b)):
        state = eqft.initialize_calibration_collector(
            kind="input_covariance",
            logical_parameter_dims={"weight": 2},
            base_checkpoint_hash=checkpoint,
            data_fingerprint="dataset",
        )
        state = eqft.update_calibration_collector(
            state, {"weight": inputs}, sample_count=2
        )
        artifacts.append(eqft.finalize_calibration_collector(state)["weight"])

    merged = eqft.regmean_merge(
        [jnp.asarray([[1.0, 2.0]]), jnp.asarray([[3.0, 4.0]])],
        artifacts,
        ridge=0.0,
    )
    direct = eqft.regmean_merge(
        [jnp.asarray([[1.0, 2.0]]), jnp.asarray([[3.0, 4.0]])],
        [inputs_a.T @ inputs_a, inputs_b.T @ inputs_b],
        ridge=0.0,
    )
    assert artifacts[0].statistics["normalization"] == "sum"
    assert jnp.allclose(merged, direct)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda state: replace(state, data_fingerprint="other"), "fingerprint"),
        (lambda state: replace(state, base_checkpoint_hash="other"), "checkpoint"),
        (lambda state: replace(state, kind="input_covariance"), "kind"),
        (lambda state: replace(state, feature_dims=(4,)), "schema"),
    ),
)
def test_combine_rejects_incompatible_states(mutation, message):
    state = _initialize()
    with pytest.raises(ValueError, match=message):
        eqft.combine_calibration_collectors(state, mutation(state))


def test_schema_and_update_identity_validation():
    common = {
        "kind": "activation_covariance",
        "base_checkpoint_hash": "checkpoint",
        "data_fingerprint": "dataset",
    }
    with pytest.raises(ValueError, match="duplicate logical parameter ID"):
        eqft.initialize_calibration_collector(
            logical_parameter_dims=(("proj", 2), ("proj", 2)), **common
        )
    with pytest.raises(ValueError, match="non-empty"):
        eqft.initialize_calibration_collector(logical_parameter_dims={}, **common)

    state = _initialize()
    with pytest.raises(ValueError, match="missing=.*encoder.proj"):
        eqft.update_calibration_collector(
            state, {"other": jnp.ones((2, 3))}, sample_count=2
        )
    with pytest.raises(ValueError, match="feature width"):
        eqft.update_calibration_collector(
            state, {"encoder.proj": jnp.ones((2, 4))}, sample_count=2
        )
    with pytest.raises(ValueError, match="match the flattened row count"):
        eqft.update_calibration_collector(
            state, {"encoder.proj": jnp.ones((2, 3))}, sample_count=1
        )
    with pytest.raises(ValueError, match="exactly one"):
        eqft.update_calibration_collector(
            state,
            {"encoder.proj": jnp.ones((2, 3))},
            sample_count=2,
            sample_mask=jnp.ones((2,), dtype=bool),
        )


def test_eva_consumes_collected_covariance_and_svd(tiny_vision_transformer):
    ids = ("blocks.0.attn.proj", "blocks.1.attn.proj")
    values = {
        ids[0]: jnp.diag(jnp.asarray([4.0, 3.0, 2.0, 1.0])),
        ids[1]: jnp.diag(jnp.asarray([4.0, 3.0, 2.0, 1.0])),
    }
    for kind, rank in (("activation_covariance", None), ("activation_svd", 4)):
        state = eqft.initialize_calibration_collector(
            kind=kind,
            logical_parameter_dims={logical_id: 4 for logical_id in ids},
            base_checkpoint_hash="checkpoint",
            data_fingerprint="dataset",
            rank=rank,
        )
        state = eqft.update_calibration_collector(
            state, values, sample_count={logical_id: 4 for logical_id in ids}
        )
        artifacts = eqft.finalize_calibration_collector(state)
        model = eqft.apply_eva_lora(
            tiny_vision_transformer,
            eqft.EVAInitializerConfig(rank_budget=2),
            activation_artifacts=artifacts,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            key=jax.random.PRNGKey(0),
        )
        rows = np.asarray(model.blocks[0].attn.proj.lora_A)
        assert rows.shape == (1, 4)
        assert np.argmax(np.abs(rows[0])) == 0
