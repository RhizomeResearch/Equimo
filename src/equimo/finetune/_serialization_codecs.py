"""Private descriptors for built-in fine-tuning bundle codecs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._typing import PyTree
from .config import FineTuneBundle


@dataclass(frozen=True)
class DeltaCodec:
    """Method-specific behavior used by the schema-1 bundle serializer."""

    method: str
    config_types: tuple[type[Any], ...]
    extract: Callable[[PyTree], FineTuneBundle]
    load: Callable[[PyTree, FineTuneBundle], PyTree]
    strip_to_base: Callable[[PyTree], PyTree]
    can_infer_exact_base_checkpoint: Callable[[PyTree], bool]
    merge_for_save: Callable[[PyTree], PyTree]
    is_mergeable: Callable[[FineTuneBundle], bool]
    is_merged: Callable[[FineTuneBundle], bool]


def build_codec_registry(codecs: tuple[DeltaCodec, ...]) -> Mapping[str, DeltaCodec]:
    """Return an immutable registry, rejecting duplicate method names."""

    registry: dict[str, DeltaCodec] = {}
    for codec in codecs:
        if codec.method in registry:
            raise ValueError(f"Duplicate delta codec method {codec.method!r}.")
        registry[codec.method] = codec
    return MappingProxyType(registry)


def supported_methods_text(methods: tuple[str, ...]) -> str:
    """Format method names for the stable public save error."""

    quoted = tuple(repr(method) for method in methods)
    return f"{', '.join(quoted[:-1])}, or {quoted[-1]}"
