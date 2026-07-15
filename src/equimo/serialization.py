import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
import warnings

import equinox as eqx
import jax
import jax.tree_util as jtu
import lz4.frame
import requests
from loguru import logger

from equimo import __version__
from equimo._io import (
    atomic_directory,
    atomic_file,
    copy_limited,
    file_lock,
    read_limited,
    sha256_file,
    validate_sha256,
)
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256
from equimo.registry import _SAFE_IDENTIFIER_RE

DEFAULT_REPOSITORY_REVISION = "bdf43d88f504d6fc3fc7850eb053df0bd762989c"
DEFAULT_REPOSITORY_URL = (
    "https://huggingface.co/poiretclement/equimo/resolve/"
    f"{DEFAULT_REPOSITORY_REVISION}/models/default"
)

_CHECKPOINT_FORMAT = "equimo.model.checkpoint"
_CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_MEMBERS = frozenset(("metadata.json", "weights.eqx"))
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_WEIGHTS_BYTES = 64 * 1024 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024


def _validate_identifier(identifier: str) -> None:
    """Raise ValueError if *identifier* contains characters unsafe for use in
    file paths or URLs (e.g. ``..``, ``/``, ``?``).
    """
    if not _SAFE_IDENTIFIER_RE.match(identifier):
        raise ValueError(
            f"Unsafe model identifier: {identifier!r}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )


def _decompress_archive(path: Path) -> Path:
    """Decompress a ``.tar.lz4`` archive to a managed sibling directory.

    Uses a sentinel file (``.complete``) so interrupted extractions are
    automatically retried on the next call. The cache name retains the full
    archive name so a caller-owned ``foo/`` is never replaced for
    ``foo.tar.lz4``.

    Returns:
        Path to the decompressed directory.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint archive does not exist: {path!s}")

    decompressed_dir = path.with_name(f"{path.name}.extracted")
    sentinel = decompressed_dir / ".complete"
    legacy_dir = path.with_suffix("").with_suffix("")
    lock_path = path.with_name(f"{path.name}.extracted.lock")

    if _extraction_is_current(path, sentinel):
        return decompressed_dir
    if _legacy_extraction_is_current(path, legacy_dir):
        _log_legacy_extraction(legacy_dir, decompressed_dir)
        return legacy_dir

    with file_lock(lock_path):
        if _extraction_is_current(path, sentinel):
            return decompressed_dir
        if _legacy_extraction_is_current(path, legacy_dir):
            _log_legacy_extraction(legacy_dir, decompressed_dir)
            return legacy_dir

        with atomic_directory(decompressed_dir) as temporary:
            _extract_model_archive(path, temporary)
            sentinel_payload = {
                "archive_size": path.stat().st_size,
                "archive_mtime_ns": path.stat().st_mtime_ns,
            }
            (temporary / ".complete").write_text(
                json.dumps(sentinel_payload, sort_keys=True)
            )

    return decompressed_dir


def _log_legacy_extraction(legacy_dir: Path, decompressed_dir: Path) -> None:
    logger.info(
        f"Using the v2-alpha extraction cache at {legacy_dir!s}. "
        f"New extractions use {decompressed_dir!s}."
    )


def _extraction_is_current(path: Path, sentinel: Path) -> bool:
    try:
        payload = json.loads(sentinel.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    stat = path.stat()
    return payload == {
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
    }


def _legacy_extraction_is_current(path: Path, directory: Path) -> bool:
    sentinel = directory / ".complete"
    required_files = tuple(directory / name for name in _CHECKPOINT_MEMBERS)
    try:
        return (
            sentinel.is_file()
            and sentinel.stat().st_mtime_ns >= path.stat().st_mtime_ns
            and all(file.is_file() for file in required_files)
        )
    except OSError:
        return False


def _extract_model_archive(path: Path, destination: Path) -> None:
    found: set[str] = set()
    total_size = 0
    with lz4.frame.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                if member.name not in _CHECKPOINT_MEMBERS:
                    raise ValueError(
                        f"Checkpoint archive contains unexpected member {member.name!r}."
                    )
                if member.name in found:
                    raise ValueError(
                        f"Checkpoint archive contains duplicate member {member.name!r}."
                    )
                if not member.isfile():
                    raise ValueError(
                        f"Checkpoint archive member {member.name!r} must be a file."
                    )
                member_limit = (
                    _MAX_METADATA_BYTES
                    if member.name == "metadata.json"
                    else _MAX_WEIGHTS_BYTES
                )
                if member.size > member_limit:
                    raise ValueError(
                        f"Checkpoint archive member {member.name!r} exceeds the "
                        f"{member_limit}-byte size limit."
                    )
                total_size += member.size
                if total_size > _MAX_METADATA_BYTES + _MAX_WEIGHTS_BYTES:
                    raise ValueError("Checkpoint archive exceeds its size limit.")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(
                        f"Checkpoint archive member {member.name!r} could not be read."
                    )
                with (destination / member.name).open("wb") as output:
                    copied = copy_limited(
                        source,
                        output,
                        member_limit,
                        label=f"Checkpoint archive member {member.name!r}",
                    )
                if copied != member.size:
                    raise ValueError(
                        f"Checkpoint archive member {member.name!r} is truncated."
                    )
                found.add(member.name)
    missing = _CHECKPOINT_MEMBERS - found
    if missing:
        raise ValueError(
            "Checkpoint archive is missing required members: "
            + ", ".join(sorted(missing))
            + "."
        )


def save_model(
    path: Path,
    model: eqx.Module,
    model_config: dict,
    torch_hub_cfg: list[str] | dict | None = None,
    timm_cfg: list | None = None,
    compression: bool = True,
) -> None:
    """Save an Equinox model with its configuration and metadata to disk.

    Args:
        path: Target path. When *compression* is ``True`` and *path* does not
            end with ``.tar.lz4``, the suffix is appended automatically.
        model: The Equinox model to save. Saved dtype is preserved — bf16 models
            are serialised in bf16.
        model_config: Hyperparameter dictionary used to reconstruct the model.
        torch_hub_cfg: Optional torch-hub configuration (list or dict).
            Defaults to ``{}`` when ``None``.
        timm_cfg: Optional timm configuration list.
            Defaults to ``[]`` when ``None``.
        compression: If ``True`` (default), create a LZ4-compressed tar archive.
            If ``False``, write a plain directory.
    """
    # Guard against mutable-default aliasing from callers.
    torch_hub_cfg = torch_hub_cfg if torch_hub_cfg is not None else {}
    timm_cfg = timm_cfg if timm_cfg is not None else []

    logger.info(f"Saving model to {path}...")

    metadata = {
        "format": _CHECKPOINT_FORMAT,
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "model_config": model_config,
        "torch_hub_cfg": torch_hub_cfg,
        "timm": timm_cfg,
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
        "equimo_version": __version__,
        "model_class": _model_class(model),
        "model_signature": _model_signature(model),
    }

    if compression:
        logger.info("Compressing...")
        if not path.name.endswith(".tar.lz4"):
            path = path.with_name(path.name + ".tar.lz4")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eqx.tree_serialise_leaves(tmp_path / "weights.eqx", model)
            metadata["weights_sha256"] = sha256_file(tmp_path / "weights.eqx")
            with (tmp_path / "metadata.json").open("w") as handle:
                json.dump(metadata, handle, sort_keys=True)

            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_file(path) as temporary_archive:
                with lz4.frame.open(temporary_archive, "wb") as output:
                    with tarfile.open(fileobj=output, mode="w") as archive:
                        archive.add(tmp_path / "metadata.json", arcname="metadata.json")
                        archive.add(tmp_path / "weights.eqx", arcname="weights.eqx")
    else:
        with atomic_directory(path) as temporary:
            eqx.tree_serialise_leaves(temporary / "weights.eqx", model)
            metadata["weights_sha256"] = sha256_file(temporary / "weights.eqx")
            with (temporary / "metadata.json").open("w") as handle:
                json.dump(metadata, handle, sort_keys=True)

    logger.debug(f"Metadata: {metadata}")

    logger.info("Model successfully saved.")


def download(
    identifier: str,
    repository: str,
    timeout: int = 60,
    expected_sha256: str | None = None,
) -> Path:
    """Download a model archive from a remote repository.

    Args:
        identifier: Unique model identifier. Must contain only alphanumeric
            characters, hyphens, and underscores (validated to prevent path
            traversal).
        repository: Base URL of the repository.
        timeout: HTTP request timeout in seconds. Defaults to 60.
        expected_sha256: Optional trusted SHA-256 digest for the complete archive.

    Returns:
        Local path to the downloaded (and cached) archive.

    Raises:
        ValueError: If *identifier* contains unsafe characters.
        requests.HTTPError: If the server returns a 4xx or 5xx response.
    """
    _validate_identifier(identifier)
    if expected_sha256 is None and repository.rstrip("/") == DEFAULT_REPOSITORY_URL:
        expected_sha256 = PRETRAINED_ARCHIVE_SHA256.get(identifier)
    if expected_sha256 is not None:
        expected_sha256 = validate_sha256(expected_sha256, label="expected_sha256")
    logger.info(f"Downloading {identifier}...")

    model = identifier.split("_")[0]
    url = f"{repository}/{model}/{identifier}.tar.lz4"
    path = Path(f"~/.cache/equimo/{model}/{identifier}.tar.lz4").expanduser()
    checksum_path = path.with_name(f"{path.name}.sha256")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        _verify_cached_download(path, checksum_path, expected_sha256)
        logger.info("Archive already downloaded, using cached file.")
        return path

    with atomic_file(path) as temporary:
        with requests.get(url, stream=True, timeout=timeout, verify=True) as res:
            res.raise_for_status()
            header_checksum = _response_checksum(res.headers)
            trusted_checksum = expected_sha256 or header_checksum
            content_length = _response_content_length(res.headers)
            if content_length is not None and content_length > _MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Download for {identifier!r} exceeds the "
                    f"{_MAX_DOWNLOAD_BYTES}-byte size limit."
                )
            digest = hashlib.sha256()
            total = 0
            with temporary.open("wb") as output:
                for chunk in res.iter_content(chunk_size=65_536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > _MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Download for {identifier!r} exceeds the "
                            f"{_MAX_DOWNLOAD_BYTES}-byte size limit."
                        )
                    digest.update(chunk)
                    output.write(chunk)
            actual_checksum = digest.hexdigest()
            if trusted_checksum is not None and actual_checksum != trusted_checksum:
                raise ValueError(
                    f"Downloaded archive checksum mismatch for {identifier!r}: "
                    f"expected {trusted_checksum}, got {actual_checksum}."
                )

    _write_checksum(checksum_path, actual_checksum, path)

    return path


def _verify_cached_download(
    path: Path,
    checksum_path: Path,
    expected_sha256: str | None,
) -> None:
    cached_checksum: str | None = None
    cached_stat: tuple[int, int] | None = None
    if checksum_path.exists():
        cached_checksum, cached_stat = _read_checksum(checksum_path)
        if expected_sha256 is not None and cached_checksum != expected_sha256:
            raise ValueError(
                f"Cached archive checksum mismatch for {path!s}: expected "
                f"{expected_sha256}, got {cached_checksum}."
            )
        stat = path.stat()
        if cached_stat == (stat.st_size, stat.st_mtime_ns):
            return

    actual_checksum = sha256_file(path)
    trusted_checksum = expected_sha256 or cached_checksum
    if trusted_checksum is not None and actual_checksum != trusted_checksum:
        raise ValueError(
            f"Cached archive checksum mismatch for {path!s}: expected "
            f"{trusted_checksum}, got {actual_checksum}."
        )
    try:
        _write_checksum(checksum_path, actual_checksum, path)
    except OSError as error:
        warnings.warn(
            f"Could not persist a checksum for cached archive {path!s}: "
            f"{error}. The archive remains loadable from this read-only cache.",
            RuntimeWarning,
            stacklevel=2,
        )


def _read_checksum(path: Path) -> tuple[str, tuple[int, int] | None]:
    payload = path.read_text().strip()
    try:
        record = json.loads(payload)
    except json.JSONDecodeError:
        return validate_sha256(payload, label=f"Checksum file {path!s}"), None
    if not isinstance(record, dict):
        raise ValueError(f"Checksum file {path!s} must contain a JSON object.")
    checksum = record.get("sha256")
    size = record.get("size")
    mtime_ns = record.get("mtime_ns")
    if (
        not isinstance(checksum, str)
        or not isinstance(size, int)
        or not isinstance(mtime_ns, int)
        or size < 0
        or mtime_ns < 0
    ):
        raise ValueError(
            f"Checksum file {path!s} contains invalid verification metadata."
        )
    return (
        validate_sha256(checksum, label=f"Checksum file {path!s}"),
        (size, mtime_ns),
    )


def _write_checksum(path: Path, checksum: str, archive_path: Path) -> None:
    stat = archive_path.stat()
    payload = {
        "sha256": checksum,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    with atomic_file(path) as temporary:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _response_checksum(headers) -> str | None:
    for name in ("X-Linked-Etag", "ETag"):
        value = headers.get(name)
        if not isinstance(value, str):
            continue
        value = value.strip().strip('"')
        if value.startswith("sha256:"):
            value = value.removeprefix("sha256:")
        try:
            return validate_sha256(value, label=name)
        except ValueError:
            continue
    return None


def _response_content_length(headers) -> int | None:
    value = headers.get("Content-Length")
    if not isinstance(value, str):
        return None
    try:
        content_length = int(value)
    except ValueError:
        return None
    return content_length if content_length >= 0 else None


def _resolve_weights_dir(
    identifier: str | None,
    path: Path | None,
    repository: str,
    expected_sha256: str | None,
) -> Path:
    """Return the local directory containing ``weights.eqx``.

    Handles downloading (when *identifier* is given) and decompression of
    ``.tar.lz4`` archives transparently.

    Raises:
        ValueError: If both or neither of *identifier*/*path* are provided.
    """
    if identifier is None and path is None:
        raise ValueError(
            "Both `identifier` and `path` are None. Please provide one of them."
        )
    if identifier is not None and path is not None:
        raise ValueError(
            "Both `identifier` and `path` are defined. Please provide only one of them."
        )

    if identifier is not None:
        path = download(identifier, repository, expected_sha256=expected_sha256)

    assert path is not None
    if path.suffixes == [".tar", ".lz4"]:
        logger.info("Decompressing...")
        path = _decompress_archive(path)

    return path


def load_weights(
    model: eqx.Module,
    identifier: str | None = None,
    path: Path | None = None,
    repository: str = DEFAULT_REPOSITORY_URL,
    inference_mode: bool = True,
    expected_sha256: str | None = None,
) -> eqx.Module:
    """Deserialise saved weights into an already-constructed model.

    This is the preferred loading path when using factory functions that
    already know the model configuration. Versioned metadata, model structure,
    and weight integrity are validated before deserialization. Existing v2
    alpha archives without these fields remain loadable.

    Args:
        model: A freshly-constructed model whose leaf shapes match the
            serialised checkpoint.
        identifier: Remote model identifier for downloading.  Mutually
            exclusive with *path*.
        path: Local path (directory or ``.tar.lz4`` archive) that contains
            ``weights.eqx``.  Mutually exclusive with *identifier*.
        repository: Base URL for model download.
            Defaults to :data:`DEFAULT_REPOSITORY_URL`.
        inference_mode: Pass ``True`` (default) to disable dropout for
            evaluation; ``False`` to keep training behaviour.
        expected_sha256: Optional trusted SHA-256 digest for a remote archive.

    Returns:
        Model with deserialised weights.  Dtype is whatever was stored
        (bf16 checkpoints are loaded as bf16).
    """
    load_path = _resolve_weights_dir(identifier, path, repository, expected_sha256)
    logger.info("Loading weights...")

    alpha_archive_verified = identifier is not None and (
        expected_sha256 is not None
        or (
            repository.rstrip("/") == DEFAULT_REPOSITORY_URL
            and identifier in PRETRAINED_ARCHIVE_SHA256
        )
    )
    _validate_checkpoint(
        load_path, model, alpha_archive_verified=alpha_archive_verified
    )
    model = eqx.tree_deserialise_leaves(load_path / "weights.eqx", model)
    model = eqx.nn.inference_mode(model, inference_mode)

    logger.info("Weights loaded successfully.")
    return model


def _validate_checkpoint(
    path: Path,
    model: eqx.Module,
    *,
    alpha_archive_verified: bool = False,
) -> dict:
    metadata_path = path / "metadata.json"
    weights_path = path / "weights.eqx"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"Checkpoint {path!s} must contain metadata.json and weights.eqx."
        )
    if metadata_path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError(
            f"Checkpoint metadata exceeds the {_MAX_METADATA_BYTES}-byte size limit."
        )
    with metadata_path.open("rb") as handle:
        payload = read_limited(handle, _MAX_METADATA_BYTES, label="Checkpoint metadata")
    try:
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Checkpoint metadata is not valid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a JSON object.")

    if "format" not in metadata:
        message = (
            "Loading a compatible v2-alpha checkpoint without versioned "
            "internal metadata."
        )
        if alpha_archive_verified:
            logger.info(message + " The complete archive checksum was verified.")
        else:
            warnings.warn(
                message + " The local weights checksum cannot be verified.",
                RuntimeWarning,
                stacklevel=2,
            )
        return metadata
    if metadata.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format {metadata.get('format')!r}.")
    if metadata.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format_version="
            f"{metadata.get('format_version')!r}; expected "
            f"{_CHECKPOINT_FORMAT_VERSION}."
        )

    expected_checksum = metadata.get("weights_sha256")
    if not isinstance(expected_checksum, str):
        raise ValueError("Checkpoint metadata is missing weights_sha256.")
    expected_checksum = validate_sha256(
        expected_checksum, label="Checkpoint weights_sha256"
    )
    actual_checksum = sha256_file(weights_path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            "Checkpoint weights checksum mismatch: expected "
            f"{expected_checksum}, got {actual_checksum}."
        )

    expected_class = metadata.get("model_class")
    if expected_class != _model_class(model):
        raise ValueError(
            f"Checkpoint model class mismatch: expected {expected_class!r}, "
            f"got {_model_class(model)!r}."
        )
    expected_signature = metadata.get("model_signature")
    actual_signature = _model_signature(model)
    if expected_signature != actual_signature:
        raise ValueError(
            "Checkpoint model signature mismatch: expected "
            f"{expected_signature!r}, got {actual_signature!r}."
        )
    return metadata


def _model_class(model: eqx.Module) -> str:
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _model_signature(model: eqx.Module) -> str:
    leaves = [
        {
            "path": jtu.keystr(path),
            "shape": tuple(int(dimension) for dimension in leaf.shape),
        }
        for path, leaf in jtu.tree_leaves_with_path(model)
        if eqx.is_array(leaf)
    ]
    payload = json.dumps(leaves, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
