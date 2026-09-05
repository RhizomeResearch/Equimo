"""Traversal and replacement contracts shared by PEFT merge/unmerge helpers."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import pytest

from equimo.finetune.peft.base import map_wrappers


class Wrapper(eqx.Module):
    value: Any
    name: str = eqx.field(static=True)


def test_wrapper_mapping_stops_at_outer_matches_and_preserves_other_leaves():
    array = jnp.ones(2)
    inner = Wrapper(array, "inner")
    model = {"a": Wrapper(inner, "outer"), "b": [Wrapper(array, "sibling"), array]}
    calls = []

    def unwrap(module):
        calls.append(module.name)
        return module.value

    mapped = map_wrappers(model, Wrapper, unwrap)
    assert calls == ["outer", "sibling"]
    assert eqx.tree_equal(mapped["a"], inner)
    assert mapped["a"].value is array
    assert mapped["b"][0] is mapped["b"][1] is array
    assert model["a"].value is inner
    assert map_wrappers(inner, Wrapper, unwrap) is array


def test_wrapper_mapping_empty_selection_preserves_identity():
    model = {"a": jnp.ones(2), "b": None}
    assert (
        map_wrappers(model, Wrapper, lambda _: pytest.fail("unexpected match")) is model
    )


def test_wrapper_mapping_callback_failure_order():
    calls = []

    def replace(module):
        calls.append(module.name)
        if module.name == "b":
            raise ValueError("callback failed")
        return module.value

    with pytest.raises(ValueError, match="callback failed"):
        map_wrappers(tuple(Wrapper(None, name) for name in "abc"), Wrapper, replace)
    assert calls == ["a", "b"]
