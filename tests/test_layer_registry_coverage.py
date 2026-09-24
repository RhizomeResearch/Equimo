"""Completeness contracts for every public layer registry alias."""

import equinox as eqx
import pytest

from cases.layer_cases import LAYER_REGISTRY_CASES

REGISTERING_CASES = tuple(
    case for case in LAYER_REGISTRY_CASES if case.register is not None
)


@pytest.mark.parametrize("case", LAYER_REGISTRY_CASES, ids=lambda case: case.scope)
def test_every_layer_alias_resolves_case_insensitively(case):
    assert case.registry, f"{case.scope} unexpectedly has no registered layers"

    for alias, implementation in case.registry.items():
        assert alias == alias.lower()
        assert case.resolve(alias.upper()) is implementation
        assert case.resolve(implementation) is implementation


@pytest.fixture(params=REGISTERING_CASES, ids=lambda case: case.scope)
def registering_case(request):
    """Yield a registry case and drop the test's registrations afterwards."""
    case = request.param
    snapshot = dict(case.registry)
    yield case
    case.registry.clear()
    case.registry.update(snapshot)


def test_register_uses_lowercased_class_name(registering_case):
    @registering_case.register()
    class CustomLayer(eqx.Module):
        pass

    assert registering_case.registry["customlayer"] is CustomLayer
    assert registering_case.resolve("customlayer") is CustomLayer


def test_register_uses_lowercased_custom_name(registering_case):
    @registering_case.register(name="MySuperLayer")
    class CustomLayer(eqx.Module):
        pass

    assert registering_case.registry["mysuperlayer"] is CustomLayer
    assert registering_case.resolve("mysuperlayer") is CustomLayer


def test_register_rejects_non_modules(registering_case):
    with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

        @registering_case.register()
        class NotAModule:
            pass


def test_register_rejects_duplicate_names(registering_case):
    @registering_case.register()
    class DuplicateLayer(eqx.Module):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registering_case.register(name="DuplicateLayer")
        class AnotherLayer(eqx.Module):
            pass
