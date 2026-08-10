import equinox as eqx

from equimo.core.layers._registry import _resolve_from_registries, make_register

_LAYER_REGISTRY: dict[str, type[eqx.Module]] = {}

register_layer = make_register(_LAYER_REGISTRY)


def get_layer(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a time-series layer, falling back to shared core layers."""
    if not isinstance(module, str):
        return module

    from equimo.core.layers.generic import _core_layer_registries

    return _resolve_from_registries(
        module,
        (("time-series", _LAYER_REGISTRY), *_core_layer_registries()),
        scope="time-series layer",
        collision_policy="first",
    )
