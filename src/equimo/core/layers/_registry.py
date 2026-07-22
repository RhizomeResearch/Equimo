from collections.abc import Mapping, Sequence
from typing import Literal

import equinox as eqx

type LayerRegistry = Mapping[str, type[eqx.Module]]
type NamedLayerRegistry = tuple[str, LayerRegistry]


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
