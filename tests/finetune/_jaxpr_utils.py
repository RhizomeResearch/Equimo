"""Assertions for recursively inspecting JAXPRs in tests."""

from collections.abc import Mapping
from typing import Any, Callable

import jax


_PRNG_MARKERS = ("random", "prng", "threefry")


def primitive_names(value: Any) -> tuple[str, ...]:
    """Return primitive names from a JAXPR and every nested sub-JAXPR."""

    names: list[str] = []
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, (str, bytes, int, float, bool, type(None))):
            return

        item_id = id(item)
        if item_id in visited:
            return

        if hasattr(item, "eqns"):
            visited.add(item_id)
            for equation in item.eqns:
                names.append(equation.primitive.name)
                visit(equation.params)
            return

        if hasattr(item, "jaxpr"):
            visited.add(item_id)
            visit(item.jaxpr)
            return

        if isinstance(item, Mapping):
            visited.add(item_id)
            for nested in item.values():
                visit(nested)
            return

        if isinstance(item, (tuple, list)):
            visited.add(item_id)
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(names)


def prng_primitive_names(value: Any) -> tuple[str, ...]:
    """Return recursively staged PRNG primitive names."""

    return tuple(
        name
        for name in primitive_names(value)
        if any(marker in name.lower() for marker in _PRNG_MARKERS)
    )


def assert_prng_free_jaxpr(fn: Callable, *args: Any) -> None:
    """Assert that tracing ``fn`` stages no PRNG primitives at any depth."""

    jaxpr = jax.make_jaxpr(fn)(*args)
    names = prng_primitive_names(jaxpr)
    assert names == (), f"staged PRNG primitives: {names}"
