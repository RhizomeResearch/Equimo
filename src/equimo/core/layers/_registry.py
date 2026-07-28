from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Optional

import equinox as eqx

type LayerRegistry = Mapping[str, type[eqx.Module]]
type NamedLayerRegistry = tuple[str, LayerRegistry]


def make_register[M: eqx.Module](
    registry: dict[str, type[M]],
) -> Callable[..., Callable[[type[M]], type[M]]]:
    """Create a ``register(name=None, force=False)`` decorator bound to ``registry``.

    Why collision checking: prevents third-party extensions from silently
    overwriting registered classes, which can silently corrupt the
    computational graph.
    """

    def register(
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable[[type[M]], type[M]]:
        def decorator(cls: type[M]) -> type[M]:
            if not issubclass(cls, eqx.Module):
                raise TypeError(
                    f"Registered class must be a subclass of eqx.Module, "
                    f"got {type(cls)}"
                )

            registry_name = name.lower() if name else cls.__name__.lower()

            if registry_name in registry and not force:
                raise ValueError(
                    f"Cannot register '{registry_name}'. It is already registered "
                    f"to {registry[registry_name]}."
                )

            registry[registry_name] = cls
            return cls

        return decorator

    return register


def make_get[M: eqx.Module](
    registry: dict[str, type[M]],
    *,
    kind: str = "module",
    plural: str = "modules",
) -> Callable[[str | type[M]], type[M]]:
    """Create a name-to-class resolver bound to ``registry``.

    String keys are necessary because configs are stringified and stored as
    JSON files to allow (de)serialization.
    """

    def get(module: str | type[M]) -> type[M]:
        if not isinstance(module, str):
            return module

        module_lower = module.lower()
        if module_lower not in registry:
            raise ValueError(
                f"Got an unknown {kind} string: '{module}'. "
                f"Available {plural}: {list(registry.keys())}"
            )

        return registry[module_lower]

    return get


def _resolve_from_registries(
    name_or_cls: str | type[eqx.Module],
    registries: Sequence[NamedLayerRegistry],
    *,
    scope: str,
    collision_policy: Literal["first"] | None = None,
) -> type[eqx.Module]:
    """Resolve a layer from an explicit set of named registries."""
    if not isinstance(name_or_cls, str):
        return name_or_cls

    name = name_or_cls.lower()
    matches = [
        (registry_name, registry[name])
        for registry_name, registry in registries
        if name in registry
    ]

    if len(matches) > 1 and collision_policy is None:
        registry_names = [registry_name for registry_name, _ in matches]
        raise ValueError(
            f"Layer '{name_or_cls}' is ambiguous in the {scope} scope; "
            f"it is registered in {registry_names}."
        )
    if matches:
        return matches[0][1]

    available = sorted({key for _, registry in registries for key in registry})
    raise ValueError(
        f"Layer '{name_or_cls}' not found in the {scope} scope. Available: {available}"
    )
