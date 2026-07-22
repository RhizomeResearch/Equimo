"""Path utilities for Equimo fine-tuning PyTrees."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import ParamInfo

LeafFilter = Callable[[Any], bool]


def key_path_to_path(key_path: tuple[Any, ...]) -> Path:
    """Convert a JAX key path into Equimo's stable path representation."""

    return tuple(_key_to_part(key) for key in key_path)


def path_to_str(path: Path) -> str:
    """Format a path as an unambiguous dot-separated string.

    Ordinary identifier-like paths keep their historical representation. Dots and
    backslashes inside string components are escaped, while integer-looking string
    components use a ``\\s`` prefix so they cannot be confused with sequence indices.
    """

    return ".".join(_format_path_part(part) for part in path)


def str_to_path(path: str) -> Path:
    """Parse a path produced by :func:`path_to_str`.

    Legacy unescaped paths remain supported.
    """

    if not path:
        return ()
    return tuple(_decode_path_part(part) for part in _split_path(path))


def is_path_prefix(prefix: Path, path: Path) -> bool:
    """Return whether ``prefix`` identifies ``path`` or one of its parents."""

    return len(prefix) <= len(path) and path[: len(prefix)] == prefix


def iter_param_leaves(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> tuple[tuple[Path, Any], ...]:
    """Return stable paths and values for parameter-like leaves."""

    filtered = eqx.filter(tree, predicate)
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(filtered)
        if predicate(leaf)
    )


def iter_param_paths(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> tuple[Path, ...]:
    """Return stable paths for parameter-like leaves."""

    return tuple(path for path, _ in iter_param_leaves(tree, predicate=predicate))


def extract_param_paths(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> tuple[str, ...]:
    """Return dot-formatted paths for parameter-like leaves."""

    return tuple(
        path_to_str(path) for path in iter_param_paths(tree, predicate=predicate)
    )


def make_path_tree(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> PyTree:
    """Replace parameter-like leaves with their stable paths."""

    filtered = eqx.filter(tree, predicate)
    return jtu.tree_map_with_path(
        lambda key_path, _: key_path_to_path(key_path),
        filtered,
    )


def make_param_info_tree(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> PyTree:
    """Replace parameter-like leaves with base ``ParamInfo`` records."""

    filtered = eqx.filter(tree, predicate)
    return jtu.tree_map_with_path(_param_info_from_leaf, filtered)


def _param_info_from_leaf(key_path: tuple[Any, ...], leaf: Any) -> ParamInfo:
    return ParamInfo(
        path=key_path_to_path(key_path),
        logical_id=path_to_str(key_path_to_path(key_path)),
        is_array=eqx.is_array(leaf),
        is_inexact_array=eqx.is_inexact_array(leaf),
    )


def _key_to_part(key: Any) -> str | int:
    if isinstance(key, jtu.GetAttrKey):
        return key.name
    if isinstance(key, jtu.SequenceKey):
        return key.idx
    if isinstance(key, jtu.DictKey):
        return _normalise_part(key.key)

    flattened_index_key = getattr(jtu, "FlattenedIndexKey", None)
    if flattened_index_key is not None and isinstance(key, flattened_index_key):
        return _normalise_part(key.key)

    return _normalise_part(key)


def _normalise_part(part: Any) -> str | int:
    return part if isinstance(part, int) else str(part)


def _parse_path_part(part: str) -> str | int:
    try:
        return int(part)
    except ValueError:
        return part


def _format_path_part(part: str | int) -> str:
    if isinstance(part, int):
        return str(part)
    escaped = part.replace("\\", "\\\\").replace(".", "\\.")
    if not part or isinstance(_parse_path_part(part), int):
        return f"\\s{escaped}"
    return escaped


def _split_path(path: str) -> tuple[str, ...]:
    parts: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(path):
        char = path[index]
        if char == "\\":
            if index + 1 >= len(path):
                raise ValueError(
                    "Fine-tuning path cannot end with an escape character."
                )
            current.extend((char, path[index + 1]))
            index += 2
            continue
        if char == ".":
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current))
    return tuple(parts)


def _decode_path_part(part: str) -> str | int:
    force_string = part.startswith("\\s")
    if force_string:
        part = part[2:]

    decoded: list[str] = []
    index = 0
    while index < len(part):
        if part[index] == "\\":
            if index + 1 >= len(part):
                raise ValueError(
                    "Fine-tuning path cannot end with an escape character."
                )
            decoded.append(part[index + 1])
            index += 2
        else:
            decoded.append(part[index])
            index += 1
    value = "".join(decoded)
    return value if force_string else _parse_path_part(value)


__all__ = (
    "LeafFilter",
    "extract_param_paths",
    "is_path_prefix",
    "iter_param_leaves",
    "iter_param_paths",
    "key_path_to_path",
    "make_param_info_tree",
    "make_path_tree",
    "path_to_str",
    "str_to_path",
)
