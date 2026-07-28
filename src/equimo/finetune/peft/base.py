"""Shared PEFT helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from .._typing import Path as TreePath
from .._typing import PyTree
from ..paths import key_path_to_path


def get_path(tree: PyTree, path: tuple[str | int, ...]):
    """Resolve ``path`` in an Equinox PyTree/module."""

    node = tree
    for part in path:
        if isinstance(node, Mapping) or isinstance(part, int):
            node = node[part]
        else:
            node = getattr(node, part)
    return node


__all__ = ("get_path", "iter_wrappers", "map_wrappers")


def iter_wrappers(
    model: PyTree,
    wrapper_types: type | tuple[type, ...],
) -> tuple[tuple[TreePath, Any], ...]:
    """Return path/module pairs for wrapper instances found in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, wrapper_types),
        )
        if isinstance(leaf, wrapper_types)
    )


def map_wrappers(
    model: PyTree,
    wrapper_types: type | tuple[type, ...],
    fn: Callable[[Any], Any],
) -> PyTree:
    """Return ``model`` with each matched wrapper replaced by ``fn(wrapper)``."""

    updated = model
    for path, module in iter_wrappers(updated, wrapper_types):
        updated = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p), updated, fn(module)
        )
    return updated
