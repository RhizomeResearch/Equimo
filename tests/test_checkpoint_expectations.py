"""Caller-supplied checkpoint identity and configuration contracts."""

import hashlib
import io
import json
import warnings

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.serialization import inspect_checkpoint, load_weights, save_model


class ScaledLinear(eqx.Module):
    linear: eqx.nn.Linear
    scale: float = eqx.field(static=True)

    def __init__(self, *, scale=1.0, seed=0):
        self.linear = eqx.nn.Linear(3, 2, key=jr.PRNGKey(seed))
        self.scale = scale

    def __call__(self, x):
        return self.scale * self.linear(x)


CONFIG = {"scale": 1.0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_checkpoint(path, *, legacy=False, model=None, config=None):
    save_model(
        path,
        ScaledLinear() if model is None else model,
        CONFIG if config is None else config,
        compression=False,
    )
    if legacy:
        metadata = json.loads((path / "metadata.json").read_text())
        for name in (
            "format",
            "format_version",
            "weights_sha256",
            "model_class",
            "model_signature",
        ):
            metadata.pop(name)
        (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def read_checkpoint(path, *, operation, **kwargs):
    if operation == "inspect":
        return inspect_checkpoint(path, allow_legacy=True, **kwargs)
    return load_weights(ScaledLinear(), path=path, **kwargs)


def test_local_archive_digest_is_checked_before_extraction(tmp_path, monkeypatch):
    archive = save_model(tmp_path / "model", ScaledLinear(), CONFIG)

    def unexpected_extraction(*args, **kwargs):
        pytest.fail("Unverified archive reached extraction")

    monkeypatch.setattr(
        "equimo.serialization._decompress_archive", unexpected_extraction
    )
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        load_weights(ScaledLinear(), path=archive, expected_sha256="0" * 64)


def test_local_archive_accepts_correct_digest_without_network(tmp_path, monkeypatch):
    archive = save_model(tmp_path / "model", ScaledLinear(), CONFIG)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Local loading attempted a download")

    monkeypatch.setattr("equimo.serialization.download", unexpected_download)
    loaded = load_weights(
        ScaledLinear(seed=1), path=archive, expected_sha256=digest(archive)
    )
    np.testing.assert_array_equal(loaded(jnp.ones(3)), ScaledLinear()(jnp.ones(3)))


def test_directory_rejects_archive_digest(tmp_path):
    path = make_checkpoint(tmp_path / "model")
    with pytest.raises(ValueError, match="expected_sha256.*archive"):
        load_weights(ScaledLinear(), path=path, expected_sha256="0" * 64)


@pytest.mark.parametrize("operation", ("inspect", "load"))
@pytest.mark.parametrize("legacy", (False, True))
def test_trusted_expectations_accept_modern_and_legacy(tmp_path, operation, legacy):
    path = make_checkpoint(tmp_path / "model", legacy=legacy)
    with warnings.catch_warnings(record=True) as caught:
        result = read_checkpoint(
            path,
            operation=operation,
            expected_weights_sha256=digest(path / "weights.eqx"),
            expected_model_config=CONFIG,
        )
    assert not caught
    if operation == "inspect":
        assert result.legacy is legacy
        assert result.verified is not legacy
    else:
        np.testing.assert_array_equal(result(jnp.ones(3)), ScaledLinear()(jnp.ones(3)))


@pytest.mark.parametrize("operation", ("inspect", "load"))
@pytest.mark.parametrize("legacy", (False, True))
def test_same_shaped_substitution_fails_trusted_digest(tmp_path, operation, legacy):
    path = make_checkpoint(tmp_path / "familiar_name", legacy=legacy)
    expected = digest(path / "weights.eqx")
    make_checkpoint(path, legacy=legacy, model=ScaledLinear(seed=1))
    with pytest.raises(ValueError, match="weights checksum mismatch"):
        read_checkpoint(path, operation=operation, expected_weights_sha256=expected)


@pytest.mark.parametrize("operation", ("inspect", "load"))
@pytest.mark.parametrize("legacy", (False, True))
def test_same_shaped_architecture_requires_matching_configuration(
    tmp_path, operation, legacy
):
    path = make_checkpoint(
        tmp_path / "model",
        legacy=legacy,
        model=ScaledLinear(scale=2.0),
        config={"scale": 2.0},
    )
    with pytest.raises(ValueError, match="model_config mismatch"):
        read_checkpoint(path, operation=operation, expected_model_config=CONFIG)


@pytest.mark.parametrize("expected", ({}, {"scale": 1.0, "extra": 2}, {"scale": True}))
def test_configuration_comparison_is_exact(tmp_path, expected):
    path = make_checkpoint(tmp_path / "model")
    with pytest.raises(ValueError, match="model_config mismatch"):
        inspect_checkpoint(path, expected_model_config=expected)


def test_configuration_uses_serialized_json_semantics(tmp_path):
    path = make_checkpoint(tmp_path / "model", config={"shape": (2, 3), "scale": 1.0})
    inspect_checkpoint(path, expected_model_config={"scale": 1.0, "shape": [2, 3]})


@pytest.mark.parametrize("operation", ("inspect", "load"))
def test_extraction_cache_does_not_bypass_trusted_weights(tmp_path, operation):
    archive = save_model(tmp_path / "model", ScaledLinear(), CONFIG)
    info = inspect_checkpoint(archive)
    extracted = archive.with_name(archive.name + ".extracted")
    # Replace both weights and their self-reported checksum; retain the cache sentinel.
    replacement = make_checkpoint(tmp_path / "replacement", model=ScaledLinear(seed=2))
    for name in ("weights.eqx", "metadata.json"):
        (extracted / name).write_bytes((replacement / name).read_bytes())
    with pytest.raises(ValueError, match="weights checksum mismatch"):
        read_checkpoint(
            archive, operation=operation, expected_weights_sha256=info.weights_sha256
        )


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("nested", (False, True))
def test_duplicate_metadata_keys_are_rejected(tmp_path, legacy, nested):
    path = make_checkpoint(tmp_path / "model", legacy=legacy)
    metadata_path = path / "metadata.json"
    payload = json.dumps(json.loads(metadata_path.read_text()))
    if nested:
        payload = payload.replace('"scale": 1.0', '"scale": 2.0, "scale": 1.0')
    else:
        payload = '{"model_config": {}, ' + payload[1:]
    metadata_path.write_text(payload)
    with pytest.raises(ValueError, match="duplicate.*key"):
        inspect_checkpoint(path, allow_legacy=True)


@pytest.mark.parametrize("damage", ("missing", "extra", "shape", "dtype"))
def test_serialized_tensor_inventory_is_validated(tmp_path, damage):
    path = make_checkpoint(tmp_path / "model")
    weights_path = path / "weights.eqx"
    stream = io.BytesIO(weights_path.read_bytes())
    arrays = [np.load(stream, allow_pickle=False), np.load(stream, allow_pickle=False)]
    if damage == "missing":
        arrays.pop()
    elif damage == "extra":
        arrays.append(arrays[-1])
    elif damage == "shape":
        arrays[0] = arrays[0][:1]
    else:
        arrays[0] = arrays[0].astype(np.float16)
    with weights_path.open("wb") as handle:
        for array in arrays:
            np.save(handle, array, allow_pickle=False)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_sha256"] = digest(weights_path)
    metadata_path.write_text(json.dumps(metadata))
    error = ValueError if damage == "extra" else RuntimeError
    message = "trailing" if damage == "extra" else "leaf|shape|dtype"
    with pytest.raises(error, match=message):
        load_weights(ScaledLinear(), path=path)


def test_bfloat16_checkpoint_preserves_dtype(tmp_path):
    model = ScaledLinear()
    model = eqx.tree_at(
        lambda m: (m.linear.weight, m.linear.bias),
        model,
        (
            model.linear.weight.astype(jnp.bfloat16),
            model.linear.bias.astype(jnp.bfloat16),
        ),
    )
    path = make_checkpoint(tmp_path / "model", model=model)
    loaded = load_weights(
        model, path=path, expected_weights_sha256=digest(path / "weights.eqx")
    )
    assert loaded.linear.weight.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        loaded(jnp.ones(3, dtype=jnp.bfloat16)), model(jnp.ones(3, dtype=jnp.bfloat16))
    )
