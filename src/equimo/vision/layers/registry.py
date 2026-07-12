import equinox as eqx

from equimo.core.layers._registry import _resolve_from_registries


def get_layer(name_or_cls: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a class from the vision and shared core layer registries."""
    if not isinstance(name_or_cls, str):
        return name_or_cls

    from equimo.core.layers.attention import (
        _ATTN_BLOCK_REGISTRY as _CORE_ATTN_BLOCK_REGISTRY,
        _ATTN_REGISTRY as _CORE_ATTN_REGISTRY,
    )
    from equimo.core.layers.dropout import _DROPOUT_REGISTRY
    from equimo.core.layers.ffn import _FFN_REGISTRY
    from equimo.core.layers.mamba import _MIXER_REGISTRY
    from equimo.core.layers.norm import _NORM_REGISTRY
    from equimo.vision.layers.attention import (
        _ATTN_BLOCK_REGISTRY as _VISION_ATTN_BLOCK_REGISTRY,
        _ATTN_REGISTRY as _VISION_ATTN_REGISTRY,
    )
    from equimo.vision.layers.convolution import _CONV_REGISTRY
    from equimo.vision.layers.downsample import _DOWNSAMPLER_REGISTRY
    from equimo.vision.layers.patch import _PATCH_REGISTRY
    from equimo.vision.layers.posemb import _POSEMB_REGISTRY
    from equimo.vision.layers.squeeze_excite import _SE_REGISTRY
    from equimo.vision.layers.wavelet import _WAVELET_REGISTRY

    return _resolve_from_registries(
        name_or_cls,
        (
            ("vision attention block", _VISION_ATTN_BLOCK_REGISTRY),
            ("core attention block", _CORE_ATTN_BLOCK_REGISTRY),
            ("convolution", _CONV_REGISTRY),
            ("mixer", _MIXER_REGISTRY),
            ("positional embedding", _POSEMB_REGISTRY),
            ("downsampler", _DOWNSAMPLER_REGISTRY),
            ("patch", _PATCH_REGISTRY),
            ("vision attention", _VISION_ATTN_REGISTRY),
            ("core attention", _CORE_ATTN_REGISTRY),
            ("normalization", _NORM_REGISTRY),
            ("feed-forward", _FFN_REGISTRY),
            ("dropout", _DROPOUT_REGISTRY),
            ("squeeze-and-excitation", _SE_REGISTRY),
            ("wavelet", _WAVELET_REGISTRY),
        ),
        scope="vision layer",
        collision_policy="first",
    )
