import equinox as eqx

from equimo.core.layers._registry import make_get, make_register

from .patch import (
    SpectrogramPatchEmbedding,
    get_patch,
    register_patch,
)

_AUDIO_LAYER_REGISTRY: dict[str, type[eqx.Module]] = {}


register_layer = make_register(_AUDIO_LAYER_REGISTRY)

get_layer = make_get(_AUDIO_LAYER_REGISTRY, kind="audio layer", plural="layers")


__all__ = [
    "SpectrogramPatchEmbedding",
    "get_layer",
    "get_patch",
    "register_layer",
    "register_patch",
]
