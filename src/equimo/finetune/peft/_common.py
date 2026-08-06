"""Shared private helpers for PEFT method implementations."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from equimo.core._prng import split_for_mode

from .._typing import Path, PyTree
from ..config import TargetSpec
from ..paths import path_to_str
from ..selectors import resolve_target
from ..tags import Tagger


def dropout(x: jax.Array, rate: float, key: jax.Array | None) -> jax.Array:
    if key is None:
        raise ValueError("A PRNG key is required when dropout is active.")
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def apply_last_axis(
    module: Callable[[jax.Array], jax.Array], x: jax.Array
) -> jax.Array:
    """Apply a vector-to-vector module over the last axis of ``x``."""
    if x.ndim == 1:
        return module(x)
    leading_shape = x.shape[:-1]
    x_flat = x.reshape((-1, x.shape[-1]))
    y_flat = jax.vmap(module)(x_flat)
    return y_flat.reshape((*leading_shape, y_flat.shape[-1]))


def activation(name: str) -> Callable[[jax.Array], jax.Array]:
    if name == "gelu":
        return jax.nn.gelu
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported activation {name!r}.")


def split_optional_key(
    key: jax.Array | None,
    count: int,
    *,
    inference: bool | None,
) -> tuple[jax.Array | None, ...]:
    if key is None:
        return (None,) * count
    return split_for_mode(key, count, inference=inference)


def linear_module_path(path: Path) -> Path:
    """Strip a trailing ``weight``/``bias`` element from a leaf path."""
    if path[-1:] in (("weight",), ("bias",)):
        return path[:-1]
    return path


def target_linear_paths(
    model: PyTree, target: TargetSpec, *, tagger: Tagger
) -> tuple[Path, ...]:
    """Resolve a target to the module paths owning matched weight/bias leaves."""
    paths = {
        info.path[:-1]
        for info in resolve_target(model, target, tagger=tagger)
        if info.path[-1:] in (("weight",), ("bias",))
    }
    return tuple(sorted(paths, key=path_to_str))


def target_mentions_qkv_segment(target: TargetSpec) -> bool:
    tags = set(target.tags_all) | set(target.tags_any)
    suffixes = (".q", ".k", ".v")
    return any(
        tag in {"attention.q", "attention.k", "attention.v"} or tag.endswith(suffixes)
        for tag in tags
    )


def selected_qkv_segment_names(target: TargetSpec) -> frozenset[str]:
    names: set[str] = set()
    for tag in (*target.tags_all, *target.tags_any):
        last = tag.rsplit(".", maxsplit=1)[-1]
        if last in {"q", "k", "v"}:
            names.add(last)
    return frozenset(names)
