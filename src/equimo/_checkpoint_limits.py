"""Shared, caller-selected limits for checkpoint readers."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import prod
import struct
from typing import BinaryIO, Iterable, Protocol

import jax
import jax.numpy as jnp
import numpy as np


_GIB = 1024**3
_MIB = 1024**2


@dataclass(frozen=True)
class CheckpointLimits:
    """Maximum resources a checkpoint reader may consume.

    Limits apply to the bytes on disk as well as array declarations in the
    serialized stream. Smaller values can be supplied at each reader call.
    """

    max_archive_bytes: int = 64 * _GIB + 17 * _MIB
    max_metadata_bytes: int = 16 * _MIB
    max_member_bytes: int = 64 * _GIB
    max_member_count: int = 2
    max_expanded_bytes: int = 64 * _GIB + 17 * _MIB
    max_tensor_header_bytes: int = 64 * 1024
    max_tensor_count: int = 1_000_000
    max_tensor_rank: int = 64
    max_tensor_dimension: int = 2**31 - 1
    max_tensor_bytes: int = 64 * _GIB
    max_total_array_bytes: int = 64 * _GIB

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer.")


DEFAULT_CHECKPOINT_LIMITS = CheckpointLimits()


def resolve_limits(
    limits: CheckpointLimits | None, **defaults: int
) -> CheckpointLimits:
    """Return caller limits, or the defaults with reader-specific overrides."""

    if limits is None:
        return replace(DEFAULT_CHECKPOINT_LIMITS, **defaults)
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits instance.")
    return limits


class SeekableReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...


class LimitedReader:
    """Count actual decoded bytes, including tar headers and padding."""

    def __init__(self, source: BinaryIO, max_bytes: int):
        self.source = source
        self.max_bytes = max_bytes
        self.total = 0

    def read(self, size: int = -1) -> bytes:
        # A bounded request also prevents an unbounded read from allocating
        # the entire decoded archive before the limit is checked.
        if size < 0:
            size = 1024 * 1024
        data = self.source.read(min(size, self.max_bytes - self.total + 1))
        self.total += len(data)
        if self.total > self.max_bytes:
            raise ValueError("Checkpoint archive exceeds its expanded byte limit.")
        return data

    def drain(self) -> None:
        """Read to the end so trailing bytes also count toward the limit."""
        while self.read(1024 * 1024):
            pass


def array_template(leaves: Iterable[object]) -> list[tuple[tuple[int, ...], jnp.dtype]]:
    """Describe leaves written by Equinox's default array codec."""

    result = []
    for leaf in leaves:
        if isinstance(leaf, jax.ShapeDtypeStruct):
            result.append((tuple(leaf.shape), jnp.dtype(leaf.dtype)))
            continue
        if isinstance(
            leaf, (jax.Array, np.ndarray, np.generic, bool, int, float, complex)
        ):
            array = leaf if isinstance(leaf, jax.Array) else np.asarray(leaf)
            result.append((tuple(array.shape), jnp.dtype(array.dtype)))
    return result


def scan_array_stream(
    stream: SeekableReader,
    limits: CheckpointLimits,
    *,
    expected: list[tuple[tuple[int, ...], jnp.dtype]] | None = None,
) -> int:
    """Validate a sequence of NPY arrays without loading their payloads."""

    stream.seek(0, 2)
    end = stream.tell()
    stream.seek(0)
    count = 0
    total = 0
    while stream.tell() < end:
        count += 1
        if count > limits.max_tensor_count:
            raise ValueError("Checkpoint exceeds the tensor count limit.")
        try:
            version = np.lib.format.read_magic(stream)
            header_size_bytes = 2 if version == (1, 0) else 4
            if version not in ((1, 0), (2, 0), (3, 0)):
                raise ValueError(f"Unsupported tensor header version {version!r}.")
            length_bytes = stream.read(header_size_bytes)
            if len(length_bytes) != header_size_bytes:
                raise ValueError("Truncated tensor header length.")
            header_length = struct.unpack(
                "<H" if header_size_bytes == 2 else "<I", length_bytes
            )[0]
            if header_length > limits.max_tensor_header_bytes:
                raise ValueError("Tensor header exceeds its byte limit.")
            stream.seek(-header_size_bytes, 1)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=limits.max_tensor_header_bytes
                )
            elif version in ((2, 0), (3, 0)):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    stream, max_header_size=limits.max_tensor_header_bytes
                )
        except (EOFError, ValueError, UnicodeError) as error:
            raise ValueError(f"Invalid checkpoint tensor header: {error}") from error
        if (
            fortran_order
            or dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype
            or dtype.kind not in "biufcV"
            or (dtype.kind == "V" and dtype.itemsize not in (1, 2))
        ):
            raise ValueError("Unsupported checkpoint tensor encoding.")
        if len(shape) > limits.max_tensor_rank or any(
            dimension < 0 or dimension > limits.max_tensor_dimension
            for dimension in shape
        ):
            raise ValueError("Checkpoint tensor shape exceeds its limit.")
        size = prod(shape) * dtype.itemsize
        total += size
        if size > limits.max_tensor_bytes or total > limits.max_total_array_bytes:
            raise ValueError("Checkpoint tensor allocation exceeds its limit.")
        if expected is not None:
            if count > len(expected):
                raise ValueError("Checkpoint contains trailing tensor data.")
            expected_shape, expected_dtype = expected[count - 1]
            wire_dtype = np.dtype(expected_dtype)
            # JAX writes low-precision ml_dtypes as raw void bytes in NPY.
            if expected_dtype.kind == "V":
                wire_dtype = np.dtype(f"V{expected_dtype.itemsize}")
            if shape != expected_shape or dtype != wire_dtype:
                raise ValueError(
                    "Checkpoint tensor shape or dtype does not match the model."
                )
        if end - stream.tell() < size:
            raise ValueError("Checkpoint tensor payload is truncated.")
        stream.seek(size, 1)
    if expected is not None and count != len(expected):
        raise ValueError("Checkpoint is missing tensor leaves.")
    stream.seek(0)
    return count
