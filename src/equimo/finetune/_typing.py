"""Shared typing helpers for fine-tuning infrastructure."""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

from equimo.core.tags import Path

PyTree: TypeAlias = Any
FilterSpec: TypeAlias = Any
LeafPredicate: TypeAlias = Callable[[Path, Any], bool]

__all__ = ("FilterSpec", "LeafPredicate", "Path", "PyTree")
