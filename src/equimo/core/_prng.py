"""Internal helpers for avoiding dead PRNG work in static inference."""

from typing import Any

import jax.random as jr


def split_for_mode(key: Any, num: int = 2, *, inference: bool | None) -> tuple:
    """Split ``key`` unless inference is statically known to be deterministic."""

    if inference is True:
        return (key,) * num
    return tuple(jr.split(key, num))


def fold_in_for_mode(key: Any, data: Any, *, inference: bool | None):
    """Fold ``data`` into ``key`` outside static deterministic inference."""

    if inference is True or key is None:
        return key
    return jr.fold_in(key, data)


def default_key_for_mode(
    key: Any,
    *,
    inference: bool | None,
    seed: int = 0,
):
    """Keep an absent key absent in static inference; otherwise use ``seed``."""

    if key is not None or inference is True:
        return key
    return jr.PRNGKey(seed)
