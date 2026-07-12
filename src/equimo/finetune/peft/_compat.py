"""Compatibility-critical primitives shared by PEFT delta formats."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

from .._typing import Path, PyTree
from ..config import FineTuneBundleError
from ..paths import key_path_to_path, path_to_str
from .base import get_path


def bundle_get_path(model: PyTree, path: Path, *, method_name: str):
    """Resolve a delta path with the stable bundle compatibility error."""

    try:
        return get_path(model, path)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise FineTuneBundleError(
            f"{method_name} delta expects path {path_to_str(path)}, "
            "but the base model has no matching leaf."
        ) from error


def is_linear_like(module: Any) -> bool:
    """Return whether ``module`` exposes a callable two-dimensional weight."""

    weight = getattr(module, "weight", None)
    return (
        callable(module)
        and weight is not None
        and eqx.is_inexact_array(weight)
        and weight.ndim == 2
    )


def linear_weight(module: Any) -> jax.Array:
    """Return a validated linear-like weight array."""

    weight = getattr(module, "weight", None)
    if weight is None or not eqx.is_inexact_array(weight) or weight.ndim != 2:
        raise TypeError(f"{type(module).__name__} is not a linear-like module.")
    return weight


def linear_bias(module: Any) -> jax.Array | None:
    """Return a validated optional linear-like bias array."""

    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    if not eqx.is_inexact_array(bias):
        raise TypeError(f"{type(module).__name__}.bias is not an inexact array.")
    return bias


def linear_state(linear: eqx.nn.Linear) -> dict[str, Any]:
    """Encode an Equinox linear layer without changing its state schema."""

    return {
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "use_bias": linear.bias is not None,
        "weight": linear.weight,
        "bias": linear.bias,
    }


def linear_from_state(state: dict[str, Any]) -> eqx.nn.Linear:
    """Decode the stable Equinox linear-layer state schema."""

    linear = cast(
        eqx.nn.Linear,
        eqx.nn.Linear(
            int(state["in_features"]),
            int(state["out_features"]),
            use_bias=bool(state["use_bias"]),
            key=jr.PRNGKey(0),
        ),
    )
    linear = eqx.tree_at(lambda layer: layer.weight, linear, state["weight"])
    if state["bias"] is not None:
        linear = eqx.tree_at(lambda layer: layer.bias, linear, state["bias"])
    return linear


def hash_inexact_pytree(
    model: PyTree,
    *,
    include_head: bool,
    include_values: bool,
    logical_ids_only: bool,
    prefix: str,
) -> str:
    """Hash inexact leaves using explicit persisted-format policies."""

    digest = hashlib.sha256()
    filtered = eqx.filter(model, eqx.is_inexact_array)
    for key_path, leaf in jtu.tree_leaves_with_path(filtered):
        if not eqx.is_inexact_array(leaf):
            continue
        path = key_path_to_path(key_path)
        if not include_head and is_head_path(path):
            continue
        digest.update(path_to_str(path).encode())
        if logical_ids_only:
            digest.update(b"\0")
            continue
        digest.update(str(tuple(leaf.shape)).encode())
        digest.update(str(leaf.dtype).encode())
        if include_values:
            digest.update(np.asarray(leaf).tobytes())
    return f"{prefix}{digest.hexdigest()}"


def is_head_path(path: Path) -> bool:
    """Return whether a logical path belongs to a classifier head."""

    return any(str(part) in {"head", "classifier"} for part in path)


__all__ = (
    "bundle_get_path",
    "hash_inexact_pytree",
    "is_head_path",
    "is_linear_like",
    "linear_bias",
    "linear_from_state",
    "linear_state",
    "linear_weight",
)
