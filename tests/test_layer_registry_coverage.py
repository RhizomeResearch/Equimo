"""Completeness contracts for every public layer registry alias."""

import pytest

from cases.layer_cases import LAYER_REGISTRY_CASES


@pytest.mark.parametrize("case", LAYER_REGISTRY_CASES, ids=lambda case: case.scope)
def test_every_layer_alias_resolves_case_insensitively(case):
    assert case.registry, f"{case.scope} unexpectedly has no registered layers"

    for alias, implementation in case.registry.items():
        assert alias == alias.lower()
        assert case.resolve(alias.upper()) is implementation
        assert case.resolve(implementation) is implementation
