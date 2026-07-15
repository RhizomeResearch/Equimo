"""Tests for serialization and vision IO."""

import importlib
import hashlib
import io
import json
import multiprocessing
import os
import queue
import tarfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lz4.frame
import numpy as np
import pytest

from equimo.registry import _MODEL_REGISTRY, get_model_cls, register_model
from equimo.serialization import (
    DEFAULT_REPOSITORY_REVISION,
    DEFAULT_REPOSITORY_URL,
    _decompress_archive,
    _validate_identifier,
    load_weights,
    save_model,
)
from equimo.vision.io import _center_crop_square

KEY = jr.PRNGKey(0)


def _require_optional_dependency(module, extra):
    if os.environ.get("EQUIMO_TEST_OPTIONAL_EXTRA") == extra:
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _decompress_archive_worker(archive_path, start_barrier, result_queue):
    try:
        start_barrier.wait(timeout=10)
        decompressed = _decompress_archive(Path(archive_path))
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    else:
        result_queue.put(("ok", str(decompressed)))


def _read_lz4_tar(path):
    with lz4.frame.open(path, "rb") as archive:
        with tarfile.open(fileobj=archive, mode="r") as tar:
            return {
                member.name: tar.extractfile(member).read()
                for member in tar.getmembers()
            }


def _write_lz4_tar(path, members):
    with lz4.frame.open(path, "wb") as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


# _validate_identifier


class TestValidateIdentifier:
    @pytest.mark.parametrize(
        "identifier",
        [
            "vit_base_patch16",
            "mlla-small",
            "MyModel123",
            "a",
            "A1_b2-c3",
        ],
    )
    def test_valid_identifiers(self, identifier):
        _validate_identifier(identifier)  # must not raise

    @pytest.mark.parametrize(
        "identifier",
        [
            "../etc/passwd",
            "model/../../secret",
            "model?query=1",
            "model name",
            "model.tar.lz4",
            "model\x00null",
            "",
        ],
    )
    def test_invalid_identifiers_raise(self, identifier):
        with pytest.raises(ValueError, match="Unsafe model identifier"):
            _validate_identifier(identifier)

    def test_path_traversal_blocked(self):
        with pytest.raises(ValueError):
            _validate_identifier("../../etc/passwd")

    def test_url_special_chars_blocked(self):
        with pytest.raises(ValueError):
            _validate_identifier("model?foo=bar&baz=qux")


# _center_crop_square


class TestCenterCropSquare:
    def test_square_input_unchanged(self):
        arr = jnp.ones((64, 64, 3))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64, 3)

    def test_wide_image_crops_width(self):
        arr = jnp.ones((64, 128, 3))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64, 3)

    def test_tall_image_crops_height(self):
        arr = jnp.ones((128, 64, 3))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64, 3)

    def test_crop_is_centered_wide(self):
        """For a 1×4 array [0,1,2,3], center crop to 1×2 should give [1,2]."""
        arr = jnp.arange(4).reshape(1, 4)
        result = _center_crop_square(arr)
        assert result.shape == (1, 1)

    def test_crop_is_centered_tall(self):
        """For a 4×1 array, center crop to 1×1 should yield the middle element."""
        arr = jnp.arange(4).reshape(4, 1)
        result = _center_crop_square(arr)
        assert result.shape == (1, 1)

    def test_no_copy_for_square(self):
        """Square arrays must be returned as-is (same object)."""
        arr = jnp.ones((32, 32, 3))
        assert _center_crop_square(arr) is arr

    def test_1d_raises(self):
        arr = jnp.ones((64,))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            _center_crop_square(arr)

    def test_2d_hw_array(self):
        arr = jnp.ones((64, 128))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64)


# register_model / get_model_cls


class TestRegisterModel:
    def test_register_default_name(self):
        @register_model()
        class CustomModelA(eqx.Module):
            pass

        assert "custommodela" in _MODEL_REGISTRY
        assert get_model_cls("custommodela") is CustomModelA

    def test_register_custom_name(self):
        @register_model("my_custom_net")
        class CustomModelB(eqx.Module):
            pass

        assert "my_custom_net" in _MODEL_REGISTRY
        assert get_model_cls("my_custom_net") is CustomModelB

    def test_register_name_is_case_insensitive(self):
        @register_model("CamelCaseModel")
        class CustomModelC(eqx.Module):
            pass

        assert get_model_cls("camelcasemodel") is CustomModelC
        assert get_model_cls("CAMELCASEMODEL") is CustomModelC

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_model()
            class NotAModule:
                pass

    def test_register_duplicate_raises(self):
        @register_model()
        class UniqueModel(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_model(name="UniqueModel")
            class AnotherModel(eqx.Module):
                pass

    def test_register_same_name_different_modalities(self):
        @register_model("shared_test_model", modality="vision")
        class SharedVisionModel(eqx.Module):
            pass

        @register_model("shared_test_model", modality="language")
        class SharedLanguageModel(eqx.Module):
            pass

        assert (
            get_model_cls("shared_test_model", modality="vision") is SharedVisionModel
        )
        assert (
            get_model_cls("shared_test_model", modality="language")
            is SharedLanguageModel
        )
        with pytest.raises(ValueError, match="Ambiguous model class"):
            get_model_cls("shared_test_model")


class TestGetModelCls:
    def test_string_resolution_builtin(self):
        from equimo.vision.models import VisionTransformer

        assert get_model_cls("vit") is VisionTransformer

    def test_string_case_insensitive(self):
        from equimo.vision.models import VisionTransformer

        assert get_model_cls("VIT") is VisionTransformer
        assert get_model_cls("Vit") is VisionTransformer

    def test_class_passthrough(self):
        assert get_model_cls(eqx.nn.Linear) is eqx.nn.Linear

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown model class"):
            get_model_cls("nonexistent_model_xyz")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="Available"):
            get_model_cls("nonexistent_model_xyz")

    def test_all_builtin_models_registered(self):
        builtins = [
            "vit",
            "mlla",
            "vssd",
            "shvit",
            "fastervit",
            "partialformer",
            "iformer",
            "mobilenetv3",
            "reduceformer",
        ]
        for name in builtins:
            cls = get_model_cls(name)
            assert issubclass(cls, eqx.Module), f"{name} not an eqx.Module subclass"


# save_model / load_weights round-trip


class _TinyModel(eqx.Module):
    """Minimal model for save/load round-trip tests."""

    linear: eqx.nn.Linear
    label: str = eqx.field(static=True)

    def __init__(
        self, in_features: int, out_features: int, *, key, label: str = "default"
    ):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.label = label

    def __call__(self, x):
        return jax.vmap(self.linear)(x)


class TestSaveLoadRoundTrip:
    def _make_model(self):
        return _TinyModel(8, 4, key=KEY)

    def _model_config(self):
        return {"in_features": 8, "out_features": 4}

    def test_save_creates_lz4_file(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        assert (tmp_path / "model.tar.lz4").exists()

    def test_save_with_explicit_suffix(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model.tar.lz4"
        save_model(path, model, self._model_config())
        assert path.exists()

    def test_save_no_compression(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        assert (path / "metadata.json").exists()
        assert (path / "weights.eqx").exists()

    def test_metadata_contains_versions(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        assert "jax_version" in meta
        assert "equinox_version" in meta
        assert "equimo_version" in meta

    def test_metadata_contains_v2_format_and_integrity(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)

        metadata = json.loads((path / "metadata.json").read_text())
        weights = path / "weights.eqx"
        assert metadata["format"] == "equimo.model.checkpoint"
        assert metadata["format_version"] == 1
        assert (
            metadata["weights_sha256"]
            == hashlib.sha256(weights.read_bytes()).hexdigest()
        )
        assert metadata["model_signature"].startswith("sha256:")

    def test_metadata_contains_model_config(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        cfg = self._model_config()
        save_model(path, model, cfg, compression=False)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        assert meta["model_config"] == cfg

    def test_mutable_default_torch_hub_cfg(self, tmp_path):
        """torch_hub_cfg=None must not share a mutable dict across calls."""
        model = self._make_model()
        path1 = tmp_path / "m1"
        path2 = tmp_path / "m2"
        save_model(path1, model, self._model_config(), compression=False)
        save_model(path2, model, self._model_config(), compression=False)
        with open(path1 / "metadata.json") as f:
            meta1 = json.load(f)
        with open(path2 / "metadata.json") as f:
            meta2 = json.load(f)
        assert meta1["torch_hub_cfg"] == {}
        assert meta2["torch_hub_cfg"] == {}

    def test_mutable_default_timm_cfg(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "m_timm"
        save_model(path, model, self._model_config(), compression=False)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        assert meta["timm"] == []

    def test_load_weights_roundtrip_compressed(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        loaded = load_weights(self._make_model(), path=tmp_path / "model.tar.lz4")
        x = jr.normal(KEY, (4, 8))
        assert jnp.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_weights_roundtrip_uncompressed(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        loaded = load_weights(self._make_model(), path=path)
        x = jr.normal(KEY, (4, 8))
        assert jnp.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_weights_accepts_existing_alpha_metadata(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        for key in (
            "format",
            "format_version",
            "weights_sha256",
            "model_class",
            "model_signature",
        ):
            metadata.pop(key)
        metadata_path.write_text(json.dumps(metadata))

        with pytest.warns(RuntimeWarning, match="v2-alpha checkpoint"):
            loaded = load_weights(self._make_model(), path=path)

        x = jr.normal(KEY, (4, 8))
        assert jnp.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_weights_rejects_corrupted_new_checkpoint(self, tmp_path):
        path = tmp_path / "model_dir"
        save_model(path, self._make_model(), self._model_config(), compression=False)
        weights_path = path / "weights.eqx"
        payload = bytearray(weights_path.read_bytes())
        payload[-1] ^= 1
        weights_path.write_bytes(payload)

        with pytest.raises(ValueError, match="checksum mismatch"):
            load_weights(self._make_model(), path=path)

    def test_load_weights_rejects_incompatible_model_signature(self, tmp_path):
        path = tmp_path / "model_dir"
        save_model(path, self._make_model(), self._model_config(), compression=False)

        with pytest.raises(ValueError, match="model signature mismatch"):
            load_weights(_TinyModel(7, 4, key=KEY), path=path)

    def test_failed_compressed_save_preserves_existing_archive(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "model.tar.lz4"
        path.write_bytes(b"existing archive")

        def fail_serialization(*args, **kwargs):
            raise RuntimeError("serialization failed")

        monkeypatch.setattr(
            "equimo.serialization.eqx.tree_serialise_leaves", fail_serialization
        )
        with pytest.raises(RuntimeError, match="serialization failed"):
            save_model(path, self._make_model(), self._model_config())

        assert path.read_bytes() == b"existing archive"

    def test_load_weights_inference_mode_default(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        loaded = load_weights(self._make_model(), path=path, inference_mode=True)
        # inference_mode=True means no training state — model must still be callable
        x = jr.normal(KEY, (4, 8))
        assert loaded(x).shape == (4, 4)

    def test_load_weights_requires_identifier_or_path(self):
        with pytest.raises(ValueError, match="Both.*None"):
            load_weights(self._make_model())

    def test_load_weights_exclusive_identifier_path(self, tmp_path):
        with pytest.raises(ValueError, match="Both.*defined"):
            load_weights(self._make_model(), identifier="some_id", path=tmp_path / "x")

    def test_load_weights_compressed_cached_decompression(self, tmp_path):
        """Loading a compressed model twice must reuse the cached decompression dir."""
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        archive = tmp_path / "model.tar.lz4"
        load_weights(self._make_model(), path=archive)
        decompressed = archive.with_name(f"{archive.name}.extracted")
        assert decompressed.exists()
        # Second load must not fail
        load_weights(self._make_model(), path=archive)

    @pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called.*:RuntimeWarning")
    @pytest.mark.filterwarnings(
        "ignore:This process .* is multi-threaded.*:DeprecationWarning"
    )
    def test_decompress_archive_concurrent_processes_extract_once(
        self, tmp_path, monkeypatch
    ):
        """Concurrent archive decompression must not race on the target directory."""
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("requires multiprocessing fork start method")

        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        archive = tmp_path / "model.tar.lz4"
        decompressed = archive.with_name(f"{archive.name}.extracted")

        ctx = multiprocessing.get_context("fork")
        start_barrier = ctx.Barrier(2)
        result_queue = ctx.Queue()
        extraction_queue = ctx.Queue()
        from equimo import serialization

        original_extract = serialization._extract_model_archive

        def slow_extract(*args, **kwargs):
            extraction_queue.put("extract")
            time.sleep(0.2)
            return original_extract(*args, **kwargs)

        monkeypatch.setattr("equimo.serialization._extract_model_archive", slow_extract)

        processes = [
            ctx.Process(
                target=_decompress_archive_worker,
                args=(str(archive), start_barrier, result_queue),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()

        results = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=20))
            except queue.Empty:
                results.append(("error", "worker timed out"))

        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        errors = [message for status, message in results if status == "error"]
        assert errors == []
        assert {path for status, path in results if status == "ok"} == {
            str(decompressed)
        }
        assert all(process.exitcode == 0 for process in processes)

        extraction_events = []
        while True:
            try:
                extraction_events.append(extraction_queue.get_nowait())
            except queue.Empty:
                break

        assert extraction_events == ["extract"]
        assert decompressed.exists()
        assert (decompressed / ".complete").exists()
        assert list(tmp_path.glob(".tmp_extract_*")) == []

    def test_decompression_does_not_replace_natural_sibling(self, tmp_path):
        archive = tmp_path / "model.tar.lz4"
        save_model(archive, self._make_model(), self._model_config())
        natural_sibling = tmp_path / "model"
        natural_sibling.mkdir()
        marker = natural_sibling / "owned-by-caller"
        marker.write_text("keep")

        extracted = _decompress_archive(archive)

        assert extracted == tmp_path / "model.tar.lz4.extracted"
        assert marker.read_text() == "keep"

    def test_existing_alpha_extraction_cache_remains_loadable(self, tmp_path):
        archive = tmp_path / "model.tar.lz4"
        save_model(archive, self._make_model(), self._model_config())
        legacy_cache = tmp_path / "model"
        legacy_cache.mkdir()
        for name, payload in _read_lz4_tar(archive).items():
            (legacy_cache / name).write_bytes(payload)
        (legacy_cache / ".complete").touch()

        extracted = _decompress_archive(archive)

        assert extracted == legacy_cache

    def test_decompression_rejects_unexpected_archive_members(self, tmp_path):
        archive = tmp_path / "model.tar.lz4"
        save_model(archive, self._make_model(), self._model_config())
        members = _read_lz4_tar(archive)
        members["unexpected.txt"] = b"unexpected"
        _write_lz4_tar(archive, members)

        with pytest.raises(ValueError, match="unexpected member"):
            _decompress_archive(archive)


# download (identifier validation; no network calls)


class TestDownload:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

    def test_default_repository_is_immutable(self):
        assert len(DEFAULT_REPOSITORY_REVISION) == 40
        assert f"/resolve/{DEFAULT_REPOSITORY_REVISION}/" in DEFAULT_REPOSITORY_URL
        assert "/resolve/main/" not in DEFAULT_REPOSITORY_URL

    def test_invalid_identifier_raises(self):
        from equimo.serialization import download

        with pytest.raises(ValueError, match="Unsafe model identifier"):
            download("../../malicious", repository="http://example.com")

    def test_identifier_with_slash_raises(self):
        from equimo.serialization import download

        with pytest.raises(ValueError):
            download("model/subdir", repository="http://example.com")

    def test_cached_file_returned_without_request(self, tmp_path):
        """If the archive already exists on disk, no HTTP request should be made."""
        from equimo.serialization import download

        identifier = "vit_test_cache"
        model_name = identifier.split("_")[0]
        cache_path = Path(
            f"~/.cache/equimo/{model_name}/{identifier}.tar.lz4"
        ).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.touch()

        try:
            with patch("equimo.serialization.requests.get") as mock_get:
                result = download(identifier, repository="http://example.com")
                mock_get.assert_not_called()
            assert result == cache_path
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_name(f"{cache_path.name}.sha256").unlink(missing_ok=True)

    def test_download_makes_get_request(self, tmp_path):
        """When archive is absent, a streaming GET must be issued."""
        from equimo.serialization import download

        identifier = "vit_test_dl_xyz"
        model_name = identifier.split("_")[0]
        cache_path = Path(
            f"~/.cache/equimo/{model_name}/{identifier}.tar.lz4"
        ).expanduser()
        cache_path.unlink(missing_ok=True)

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake data"]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()

        try:
            with patch(
                "equimo.serialization.requests.get", return_value=mock_response
            ) as mock_get:
                download(identifier, repository="http://example.com")
                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args
                assert call_kwargs.kwargs.get("stream") is True
                assert call_kwargs.kwargs.get("timeout") is not None
                assert call_kwargs.kwargs.get("verify") is True
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_name(f"{cache_path.name}.sha256").unlink(missing_ok=True)

    def test_cached_file_checksum_is_verified(self):
        from equimo.serialization import download

        identifier = "vit_test_checksum"
        cache_path = Path(f"~/.cache/equimo/vit/{identifier}.tar.lz4").expanduser()
        checksum_path = cache_path.with_name(f"{cache_path.name}.sha256")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"corrupted")
        checksum_path.write_text(hashlib.sha256(b"expected").hexdigest() + "\n")

        try:
            with patch("equimo.serialization.requests.get") as mock_get:
                with pytest.raises(ValueError, match="checksum mismatch"):
                    download(identifier, repository="http://example.com")
                mock_get.assert_not_called()
        finally:
            cache_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)

    def test_read_only_legacy_cache_remains_loadable(self, monkeypatch):
        from equimo import serialization

        identifier = "vit_test_read_only_cache"
        cache_path = Path(f"~/.cache/equimo/vit/{identifier}.tar.lz4").expanduser()
        checksum_path = cache_path.with_name(f"{cache_path.name}.sha256")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"existing")
        checksum_path.unlink(missing_ok=True)

        def fail_checksum_write(*args, **kwargs):
            raise PermissionError("read-only cache")

        monkeypatch.setattr(serialization, "_write_checksum", fail_checksum_write)
        try:
            with patch("equimo.serialization.requests.get") as mock_get:
                with pytest.warns(RuntimeWarning, match="read-only cache"):
                    result = serialization.download(
                        identifier, repository="http://example.com"
                    )
                mock_get.assert_not_called()
            assert result == cache_path
        finally:
            cache_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)

    def test_unchanged_verified_cache_does_not_rehash(self, monkeypatch):
        from equimo import serialization

        identifier = "vit_test_verified_cache"
        cache_path = Path(f"~/.cache/equimo/vit/{identifier}.tar.lz4").expanduser()
        checksum_path = cache_path.with_name(f"{cache_path.name}.sha256")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"verified")
        serialization._write_checksum(
            checksum_path, hashlib.sha256(b"verified").hexdigest(), cache_path
        )

        def fail_hash(*args, **kwargs):
            raise AssertionError("unchanged cache should not be rehashed")

        monkeypatch.setattr(serialization, "sha256_file", fail_hash)
        with patch("equimo.serialization.requests.get") as mock_get:
            result = serialization.download(identifier, repository="http://example.com")
            mock_get.assert_not_called()
        assert result == cache_path

    def test_download_rejects_expected_checksum_mismatch(self):
        from equimo.serialization import download

        identifier = "vit_test_expected_checksum"
        cache_path = Path(f"~/.cache/equimo/vit/{identifier}.tar.lz4").expanduser()
        checksum_path = cache_path.with_name(f"{cache_path.name}.sha256")
        cache_path.unlink(missing_ok=True)
        checksum_path.unlink(missing_ok=True)
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"downloaded"]
        mock_response.headers = {}
        mock_response.__enter__ = lambda response: response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()

        try:
            with patch("equimo.serialization.requests.get", return_value=mock_response):
                with pytest.raises(ValueError, match="checksum mismatch"):
                    download(
                        identifier,
                        repository="http://example.com",
                        expected_sha256="0" * 64,
                    )
            assert not cache_path.exists()
        finally:
            cache_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)

    def test_download_enforces_size_limit(self, monkeypatch):
        from equimo import serialization

        identifier = "vit_test_size_limit"
        cache_path = Path(f"~/.cache/equimo/vit/{identifier}.tar.lz4").expanduser()
        checksum_path = cache_path.with_name(f"{cache_path.name}.sha256")
        cache_path.unlink(missing_ok=True)
        checksum_path.unlink(missing_ok=True)
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"too", b"large"]
        mock_response.headers = {}
        mock_response.__enter__ = lambda response: response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        monkeypatch.setattr(serialization, "_MAX_DOWNLOAD_BYTES", 4)

        try:
            with patch("equimo.serialization.requests.get", return_value=mock_response):
                with pytest.raises(ValueError, match="size limit"):
                    serialization.download(identifier, repository="http://example.com")
            assert not cache_path.exists()
        finally:
            cache_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)

    def test_default_repository_uses_embedded_archive_digest(self):
        from equimo.serialization import download

        identifier = "dinov2_vits14_reg"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"not the pinned archive"]
        mock_response.headers = {}
        mock_response.__enter__ = lambda response: response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()

        with patch("equimo.serialization.requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="checksum mismatch"):
                download(identifier, repository=DEFAULT_REPOSITORY_URL)


# load_image


class TestLoadImage:
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        img = PILImage.new("RGB", (64, 48), color=(128, 64, 32))
        path = tmp_path / "test.png"
        img.save(str(path))
        return str(path)

    @pytest.fixture
    def grayscale_image_path(self, tmp_path):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        img = PILImage.new("L", (32, 32), color=128)
        path = tmp_path / "gray.png"
        img.save(str(path))
        return str(path)

    def test_output_shape_chw(self, sample_image_path):
        from equimo.vision.io import load_image

        out = load_image(sample_image_path)
        assert out.ndim == 3
        assert out.shape[0] == 3  # channels first

    def test_output_dtype_float32(self, sample_image_path):
        from equimo.vision.io import load_image

        out = load_image(sample_image_path)
        assert out.dtype == jnp.float32

    def test_output_range_0_1(self, sample_image_path):
        from equimo.vision.io import load_image

        out = load_image(sample_image_path)
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0

    def test_grayscale_converted_to_rgb(self, grayscale_image_path):
        from equimo.vision.io import load_image

        out = load_image(grayscale_image_path)
        assert out.shape[0] == 3

    def test_resize(self, sample_image_path):
        from equimo.vision.io import load_image

        out = load_image(sample_image_path, size=32)
        assert out.shape == (3, 32, 32)

    def test_normalization_applied(self, sample_image_path):
        from equimo.vision.io import load_image

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out_norm = load_image(sample_image_path, mean=mean, std=std)
        out_raw = load_image(sample_image_path)
        assert not jnp.allclose(out_norm, out_raw)

    def test_normalization_formula(self, sample_image_path):
        from equimo.vision.io import load_image

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        out_norm = load_image(sample_image_path, mean=mean, std=std)
        out_raw = load_image(sample_image_path)
        expected = (out_raw - 0.5) / 0.5
        assert jnp.allclose(out_norm, expected, atol=1e-5)

    def test_center_crop(self, sample_image_path):
        from equimo.vision.io import load_image

        out = load_image(sample_image_path, center_crop=True)
        _, h, w = out.shape
        assert h == w

    def test_center_crop_then_resize(self, sample_image_path):
        from equimo.vision.io import load_image

        out = load_image(sample_image_path, center_crop=True, size=32)
        assert out.shape == (3, 32, 32)


def test_optional_extras_smoke(tmp_path):
    pil_image = _require_optional_dependency("PIL.Image", "extras")
    matplotlib = _require_optional_dependency("matplotlib", "extras")
    _require_optional_dependency("sklearn", "extras")

    from equimo.utils import PCAVisualizer, plot_image_and_feature_map
    from equimo.vision.io import load_image

    matplotlib.use("Agg")
    image_path = tmp_path / "image.png"
    pil_image.new("RGB", (4, 3), color=(128, 64, 32)).save(image_path)

    image = load_image(str(image_path))
    assert image.shape == (3, 3, 4)
    assert np.allclose(np.asarray(image[:, 0, 0]), np.array([128, 64, 32]) / 255.0)

    features = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    projected = PCAVisualizer(features, n_samples=8, n_components=2)(features)
    assert projected.shape == (4, 2)
    assert np.isfinite(projected).all()

    plot_path = tmp_path / "feature-map.png"
    plot_image_and_feature_map(
        np.asarray(image).transpose(1, 2, 0), projected, plot_path
    )
    assert plot_path.stat().st_size > 0
