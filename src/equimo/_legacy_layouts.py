"""Read artifacts that store values Equimo now computes.

Before Equimo 2.8, some modules stored arrays that follow from their static
configuration, such as RoPE frequencies and relative-coordinate tables, as
PyTree leaves. Those modules name the attributes in ``_legacy_stored_leaves``;
each attribute now computes the value, and older checkpoints stored it after
the module's other leaves. Weight streams are positional, so re-inserting the
values at those positions reproduces the older layout exactly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

PyTree = Any


class _LegacyLeaves(eqx.Module):
    module: Any
    leaves: tuple[Any, ...]
    names: tuple[str, ...] = eqx.field(static=True)


def _stored_names(node: Any) -> tuple[str, ...]:
    if not isinstance(node, eqx.Module):
        return ()
    return getattr(type(node), "_legacy_stored_leaves", ())


def _has_stored_leaves(node: Any) -> bool:
    return bool(_stored_names(node))


def _stored_value(node: Any, name: str) -> jax.Array:
    """Recompute a value older checkpoints stored, on the CPU when available.

    Accelerators can round transcendental functions differently, and the
    published checkpoints match the CPU values bit for bit, which hashes of
    older artifacts depend on.
    """
    try:
        cpu = jax.local_devices(backend="cpu")[0]
    except RuntimeError:  # JAX_PLATFORMS can exclude the CPU backend.
        return getattr(node, name)
    with jax.default_device(cpu):
        return getattr(node, name)


def _matches_stored(stored: np.ndarray, computed: np.ndarray) -> bool:
    """Whether *stored* equals *computed* up to a few units of rounding."""
    if stored.dtype != computed.dtype or stored.shape != computed.shape:
        return False
    if not jnp.issubdtype(computed.dtype, jnp.inexact):
        return np.array_equal(stored, computed)
    tolerance = 4 * float(jnp.finfo(computed.dtype).eps)
    return np.allclose(
        stored.astype(np.float64),
        computed.astype(np.float64),
        rtol=tolerance,
        atol=tolerance,
    )


def legacy_layout(model: PyTree) -> PyTree | None:
    """Return *model* with the values older checkpoints stored re-inserted.

    The result flattens to the leaf sequence of a pre-2.8 checkpoint and can be
    used as the deserialization template. Returns ``None`` when no module of
    *model* changed its layout.
    """

    modules = jtu.tree_leaves(model, is_leaf=_has_stored_leaves)
    if not any(_has_stored_leaves(node) for node in modules):
        return None

    def expand(node: Any) -> Any:
        if not _has_stored_leaves(node):
            return node
        names = _stored_names(node)
        return _LegacyLeaves(node, tuple(getattr(node, name) for name in names), names)

    return jtu.tree_map(expand, model, is_leaf=_has_stored_leaves)


def restore_layout(loaded: PyTree) -> PyTree:
    """Drop the stored values from a tree read through :func:`legacy_layout`.

    Raises:
        ValueError: If a stored value differs from the one the module computes
            by more than rounding. Weight decay applied while training changes such values, and the
            trained model depends on them, so they cannot be dropped silently.
    """

    def restore(path: tuple[Any, ...], node: Any) -> Any:
        if not isinstance(node, _LegacyLeaves):
            return node
        for name, stored in zip(node.names, node.leaves, strict=True):
            stored = np.asarray(stored)
            expected = np.asarray(_stored_value(node.module, name))
            if not _matches_stored(stored, expected):
                difference = np.max(
                    np.abs(stored.astype(np.float64) - expected.astype(np.float64))
                )
                raise ValueError(
                    f"Checkpoint value {jtu.keystr(path)}.{name} differs from the "
                    f"value Equimo computes (max abs difference {difference:.3g}). "
                    "It was probably changed by weight decay during training; "
                    "Equimo no longer stores it, so load this checkpoint with "
                    "equimo<2.8."
                )
        return node.module

    return jtu.tree_map_with_path(
        restore, loaded, is_leaf=lambda node: isinstance(node, _LegacyLeaves)
    )


def legacy_leaves_with_path(model: PyTree) -> list[tuple[tuple[Any, ...], Any]]:
    """Return the key paths and leaves a pre-2.8 version of *model* had."""

    entries: list[tuple[tuple[Any, ...], Any]] = []
    for path, node in jtu.tree_leaves_with_path(model, is_leaf=_has_stored_leaves):
        if not _has_stored_leaves(node):
            entries.append((path, node))
            continue
        entries.extend(
            ((*path, *subpath), leaf)
            for subpath, leaf in jtu.tree_leaves_with_path(node)
        )
        entries.extend(
            ((*path, jtu.GetAttrKey(name)), _stored_value(node, name))
            for name in _stored_names(node)
        )
    return entries


def computed_value_settings(model: PyTree) -> list[tuple[tuple[Any, ...], str, str]]:
    """Return ``(key path, field name, str(value))`` for the static fields of
    every module that computes values older artifacts stored.

    Hashes over arrays covered those values while modules stored them; they
    now follow from these fields, so hashes that identify a model include
    them. Strings keep the hashes independent of floating-point rounding.
    """

    settings: list[tuple[tuple[Any, ...], str, str]] = []
    for path, node in jtu.tree_leaves_with_path(model, is_leaf=_has_stored_leaves):
        if not _has_stored_leaves(node):
            continue
        settings.extend(
            (path, field.name, str(getattr(node, field.name)))
            for field in dataclasses.fields(node)
            if field.metadata.get("static", False)
        )
    return settings
