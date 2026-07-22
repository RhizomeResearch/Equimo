from collections.abc import Callable

import equinox as eqx

_LAYER_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_layer(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a time-series-specific layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )
        registry_name = name.lower() if name else cls.__name__.lower()
        if registry_name in _LAYER_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_LAYER_REGISTRY[registry_name]}."
            )
        _LAYER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_layer(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a time-series layer, falling back to shared core layers."""
    if not isinstance(module, str):
        return module
    module_lower = module.lower()
    if module_lower in _LAYER_REGISTRY:
        return _LAYER_REGISTRY[module_lower]

    from equimo.core.layers import get_layer as get_core_layer

    try:
        return get_core_layer(module)
    except ValueError as error:
        raise ValueError(
            f"Got an unknown time-series layer string: '{module}'. "
            f"Available modules: {list(_LAYER_REGISTRY)}"
        ) from error
