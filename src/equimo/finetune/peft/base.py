"""Shared PEFT helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .._typing import PyTree


def get_path(tree: PyTree, path: tuple[str | int, ...]):
    """Resolve ``path`` in an Equinox PyTree/module."""

    node = tree
    for part in path:
        if isinstance(node, Mapping) or isinstance(part, int):
            node = node[part]
        else:
            node = getattr(node, part)
    return node


__all__ = ("get_path",)
