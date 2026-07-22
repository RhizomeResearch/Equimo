"""Small, shared helpers for durable and bounded checkpoint I/O."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Protocol


_COPY_CHUNK_SIZE = 1024 * 1024


class _BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class _BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> object: ...


class _SeekableBinaryReader(_BinaryReader, Protocol):
    def tell(self) -> int: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: _SeekableBinaryReader) -> str:
    """Hash a seekable stream without changing its current position."""

    position = handle.tell()
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(_COPY_CHUNK_SIZE), b""):
        digest.update(chunk)
    handle.seek(position)
    return digest.hexdigest()


def validate_sha256(value: str, *, label: str) -> str:
    """Validate and normalize a hexadecimal SHA-256 digest."""

    value = value.lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest.")
    return value


@contextmanager
def atomic_file(path: Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically replace *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_directory(path: Path) -> Iterator[Path]:
    """Stage a directory and replace *path* only after a successful write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    )
    committed = False
    try:
        yield temporary
        _replace_directory(temporary, path)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(temporary, ignore_errors=True)


def read_limited(handle: _BinaryReader, max_bytes: int, *, label: str) -> bytes:
    """Read at most *max_bytes* and reject larger streams."""

    payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte size limit.")
    return payload


def copy_limited(
    source: _BinaryReader,
    destination: _BinaryWriter,
    max_bytes: int,
    *,
    label: str,
) -> int:
    """Copy a stream while enforcing an upper byte bound."""

    total = 0
    while chunk := source.read(_COPY_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte size limit.")
        destination.write(chunk)
    return total


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold a portable exclusive process lock for *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI.
            import msvcrt

            if path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace_directory(source: Path, destination: Path) -> None:
    if not destination.exists() and not destination.is_symlink():
        os.replace(source, destination)
        return

    backup = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".backup",
        )
    )
    backup.rmdir()
    os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    else:
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            with suppress(FileNotFoundError):
                backup.unlink()
