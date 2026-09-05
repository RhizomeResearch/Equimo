# ty: ignore[invalid-assignment]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]
__all__ = [
    "ConvNeXt",
    "convnext_sizes",
    "convnext_v2_sizes",
    "convnext_atto",
    "convnext_a",
    "convnext_femto",
    "convnext_f",
    "convnext_pico",
    "convnext_p",
    "convnext_nano",
    "convnext_n",
    "convnext_zepto_rms",
    "convnext_z",
    "convnext_tiny",
    "convnext_t",
    "convnext_small",
    "convnext_s",
    "convnext_base",
    "convnext_b",
    "convnext_large",
    "convnext_l",
    "convnext_xlarge",
    "convnext_xl",
    "convnext_xxlarge",
    "convnext_xxl",
    "convnext_atto_ols",
    "convnext_a_ols",
    "convnext_femto_ols",
    "convnext_f_ols",
    "convnext_pico_ols",
    "convnext_p_ols",
    "convnext_nano_ols",
    "convnext_n_ols",
    "convnext_zepto_rms_ols",
    "convnext_z_ols",
    "convnextv2_atto",
    "convnextv2_a",
    "convnextv2_femto",
    "convnextv2_f",
    "convnextv2_pico",
    "convnextv2_p",
    "convnextv2_nano",
    "convnextv2_n",
    "convnextv2_tiny",
    "convnextv2_t",
    "convnextv2_base",
    "convnextv2_b",
    "convnextv2_large",
    "convnextv2_l",
    "convnextv2_huge",
    "convnextv2_h",
    "eupe_convnext_tiny",
    "eupe_convnext_small",
    "eupe_convnext_base",
]

from collections.abc import Callable

import equinox as eqx
import jax.random as jr
import numpy as np
from jaxtyping import PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.core.layers.generic import BlockChunk
from equimo.core.layers.norm import get_norm
from equimo.registry import register_model
from equimo.vision.layers import get_layer
from equimo.core.factory import build_model_variant
from equimo.vision.models._features import DenseStageFeatures

# ConvNeXt V1 sizes: the original paper's tiny..xlarge,
# and timm's atto/femto/pico/nano/xxlarge/zepto_rms variants.
convnext_sizes: dict[str, dict] = {
    "atto": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
    "femto": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384]},
    "pico": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},
    "nano": {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640]},
    "zepto_rms": {"depths": [2, 2, 4, 2], "dims": [32, 64, 128, 256]},
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
    "xlarge": {"depths": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]},
    "xxlarge": {"depths": [3, 4, 30, 3], "dims": [384, 768, 1536, 3072]},
}

# ConvNeXt V2 sizes, from Meta's official release.
convnext_v2_sizes: dict[str, dict] = {
    "atto": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
    "femto": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384]},
    "pico": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},
    "nano": {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640]},
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
    "huge": {"depths": [3, 3, 27, 3], "dims": [352, 704, 1408, 2816]},
}


@register_model("convnext", modality="vision")
class ConvNeXt(DenseStageFeatures, eqx.Module):
    """ConvNeXt: A ConvNet for the 2020s (Liu et al., 2022).

    Four-stage hierarchical CNN using depthwise separable convolutions with
    inverted bottleneck design, LayerNorm2d, and GELU activations.

    Each stage consists of an optional downsampler followed by repeated
    ConvNeXtBlocks. The stem (stage 0) defaults to a stride-4 convolution for
    initial patchification (``stem="convnextstem"``); the `_ols` size variants
    swap in two overlapping stride-2 convolutions instead
    (``stem="convnextoverlapstem"``). Subsequent stages use stride-2
    ConvNeXtDownsamplers. ``block_norm_layer`` selects the norm used inside
    each ConvNeXtBlock (LayerNorm2d by default, RMSNorm2d for zepto_rms).
    ``use_grn=True`` builds the ConvNeXt V2 size family from this same class:
    Global Response Normalization inside each block, with LayerScale disabled
    via ``layer_scale_init_value=None``.

    Input convention: (C, H, W). Output: (num_classes,) or (dim,) features.
    """

    blocks: tuple[BlockChunk, ...]
    dropout: eqx.nn.Dropout
    norm: eqx.Module
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int = 3,
        *,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        layer_scale_init_value: float = 1e-6,
        act_layer: str | Callable = "gelu",
        norm_layer: str | type[eqx.Module] = "layernorm",
        block_norm_layer: str | type[eqx.Module] = "layernorm2d",
        downsampler_norm_layer: str | type[eqx.Module] = "layernorm2d",
        use_grn: bool = False,
        stem: str | type[eqx.Module] = "convnextstem",
        stem_kwargs: dict | None = None,
        num_classes: int | None = 1000,
        eps: float = 1e-6,
        key: PRNGKeyArray,
        **kwargs,
    ):
        # default to ConvNeXt-tiny
        depths = [3, 3, 9, 3] if depths is None else depths
        dims = [96, 192, 384, 768] if dims is None else dims
        stem_kwargs = {} if stem_kwargs is None else stem_kwargs

        key_blk, key_head = jr.split(key, 2)

        depth = sum(depths)
        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)
        block_norm_layer = get_norm(block_norm_layer)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        # Stage 0: stem (4x downsample) + blocks
        # Stages 1-3: ConvNeXtDownsampler (2x downsample) + blocks
        downsamplers = [
            stem,
            "convnextdownsampler",
            "convnextdownsampler",
            "convnextdownsampler",
        ]
        downsampler_kwargs = [
            stem_kwargs,
            *([{"norm_layer": downsampler_norm_layer}] * 3),
        ]
        _bc_dim = [in_channels, *dims[:-1]]

        blocks = []
        block_keys = jr.split(key_blk, len(dims))
        for i, _k in enumerate(block_keys):
            blocks.append(
                BlockChunk(
                    depth=depths[i],
                    in_channels=_bc_dim[i],
                    out_channels=dims[i],
                    module="convnextblock",
                    module_kwargs={
                        "channels": dims[i],
                        "act_layer": act_layer,
                        "norm_layer": block_norm_layer,
                        "use_grn": use_grn,
                        # Not passed via BlockChunk's init_values=: it only
                        # forwards non-None values, silently dropping V2's None.
                        "init_values": layer_scale_init_value,
                    },
                    downsampler=downsamplers[i],
                    downsampler_kwargs=downsampler_kwargs[i],
                    downsample_last=False,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    layer_resolver=get_layer,
                    key=_k,
                )
            )
        self.blocks = tuple(blocks)

        self.dropout = eqx.nn.Dropout(p=dropout)
        self.norm = norm_layer(dims[-1], eps=eps)
        self.head = (
            eqx.nn.Linear(dims[-1], num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )


_CONVNEXT_BASE_CFG: dict = {
    "in_channels": 3,
    "layer_scale_init_value": 1e-6,
}

# EUPE ConvNeXt models are trained without a classification head (num_classes=0).
_EUPE_CONVNEXT_BASE_CFG: dict = {
    "in_channels": 3,
    "layer_scale_init_value": 1e-6,
    "num_classes": 0,
}

# Full-name registry entries
_CONVNEXT_FULL_NAME_REGISTRY: dict[str, tuple[dict, dict]] = {
    "convnext_atto": (_CONVNEXT_BASE_CFG, convnext_sizes["atto"]),
    "convnext_femto": (_CONVNEXT_BASE_CFG, convnext_sizes["femto"]),
    "convnext_pico": (_CONVNEXT_BASE_CFG, convnext_sizes["pico"]),
    "convnext_nano": (_CONVNEXT_BASE_CFG, convnext_sizes["nano"]),
    "convnext_zepto_rms": (
        _CONVNEXT_BASE_CFG,
        {
            **convnext_sizes["zepto_rms"],
            # timm's zepto_rms uses SimpleNorm2d everywhere
            # (stem, inter-stage downsamplers, per-block, and head)
            # rather than RMSNorm2d.
            "block_norm_layer": "simplenorm2d",
            "downsampler_norm_layer": "simplenorm2d",
            "norm_layer": "simplenorm",
            "stem_kwargs": {"norm_layer": "simplenorm2d"},
        },
    ),
    "convnext_tiny": (_CONVNEXT_BASE_CFG, convnext_sizes["tiny"]),
    "convnext_small": (_CONVNEXT_BASE_CFG, convnext_sizes["small"]),
    "convnext_base": (_CONVNEXT_BASE_CFG, convnext_sizes["base"]),
    "convnext_large": (_CONVNEXT_BASE_CFG, convnext_sizes["large"]),
    "convnext_xlarge": (_CONVNEXT_BASE_CFG, convnext_sizes["xlarge"]),
    "convnext_xxlarge": (_CONVNEXT_BASE_CFG, convnext_sizes["xxlarge"]),
}

# Short-form aliases
# resolved to the same tuple objects above, so they can't drift.
_CONVNEXT_ALIASES: dict[str, str] = {
    "convnext_a": "convnext_atto",
    "convnext_f": "convnext_femto",
    "convnext_p": "convnext_pico",
    "convnext_n": "convnext_nano",
    "convnext_z": "convnext_zepto_rms",
    "convnext_t": "convnext_tiny",
    "convnext_s": "convnext_small",
    "convnext_b": "convnext_base",
    "convnext_l": "convnext_large",
    "convnext_xl": "convnext_xlarge",
    "convnext_xxl": "convnext_xxlarge",
}

# '_ols' (overlapping stem) registry entries: same size configs as above, plus
# a ConvNeXtOverlapStem in place of the default ConvNeXtStem.
_CONVNEXT_OLS_REGISTRY: dict[str, tuple[dict, dict]] = {
    "convnext_atto_ols": (
        _CONVNEXT_BASE_CFG,
        {
            **convnext_sizes["atto"],
            "stem": "convnextoverlapstem",
            "stem_kwargs": {"mid_ratio": 0.5},
        },
    ),
    "convnext_femto_ols": (
        _CONVNEXT_BASE_CFG,
        {
            **convnext_sizes["femto"],
            "stem": "convnextoverlapstem",
            "stem_kwargs": {"mid_ratio": 0.5},
        },
    ),
    "convnext_pico_ols": (
        _CONVNEXT_BASE_CFG,
        {
            **convnext_sizes["pico"],
            "stem": "convnextoverlapstem",
            "stem_kwargs": {"mid_ratio": 0.5},
        },
    ),
    "convnext_nano_ols": (
        _CONVNEXT_BASE_CFG,
        {**convnext_sizes["nano"], "stem": "convnextoverlapstem"},
    ),
    "convnext_zepto_rms_ols": (
        _CONVNEXT_BASE_CFG,
        {
            **convnext_sizes["zepto_rms"],
            # timm uses SimpleNorm2d at every norm site for this size family.
            "block_norm_layer": "simplenorm2d",
            "downsampler_norm_layer": "simplenorm2d",
            "norm_layer": "simplenorm",
            "stem": "convnextoverlapstem",
            "stem_kwargs": {"act_layer": "gelu", "norm_layer": "simplenorm2d"},
        },
    ),
}

_CONVNEXT_OLS_ALIASES: dict[str, str] = {
    "convnext_a_ols": "convnext_atto_ols",
    "convnext_f_ols": "convnext_femto_ols",
    "convnext_p_ols": "convnext_pico_ols",
    "convnext_n_ols": "convnext_nano_ols",
    "convnext_z_ols": "convnext_zepto_rms_ols",
}

# ConvNeXt V2 ("A ConvNet for the 2020s v2", Woo et al., 2023) registry
# entries: the same ConvNeXt class, with Global Response Normalization
# (use_grn=True) and LayerScale disabled (layer_scale_init_value=None). V2's
# size range is atto..huge — it never had small/xlarge/xxlarge, unlike v1.
_CONVNEXTV2_BASE_CFG: dict = {
    "in_channels": 3,
    "layer_scale_init_value": None,
    "use_grn": True,
}

_CONVNEXTV2_FULL_NAME_REGISTRY: dict[str, tuple[dict, dict]] = {
    "convnextv2_atto": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["atto"]),
    "convnextv2_femto": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["femto"]),
    "convnextv2_pico": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["pico"]),
    "convnextv2_nano": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["nano"]),
    "convnextv2_tiny": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["tiny"]),
    "convnextv2_base": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["base"]),
    "convnextv2_large": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["large"]),
    "convnextv2_huge": (_CONVNEXTV2_BASE_CFG, convnext_v2_sizes["huge"]),
}

_CONVNEXTV2_ALIASES: dict[str, str] = {
    "convnextv2_a": "convnextv2_atto",
    "convnextv2_f": "convnextv2_femto",
    "convnextv2_p": "convnextv2_pico",
    "convnextv2_n": "convnextv2_nano",
    "convnextv2_t": "convnextv2_tiny",
    "convnextv2_b": "convnextv2_base",
    "convnextv2_l": "convnextv2_large",
    "convnextv2_h": "convnextv2_huge",
}

_CONVNEXT_REGISTRY: dict[str, tuple[dict, dict]] = {
    **_CONVNEXT_FULL_NAME_REGISTRY,
    **{
        alias: _CONVNEXT_FULL_NAME_REGISTRY[canonical]
        for alias, canonical in _CONVNEXT_ALIASES.items()
    },
    **_CONVNEXT_OLS_REGISTRY,
    **{
        alias: _CONVNEXT_OLS_REGISTRY[canonical]
        for alias, canonical in _CONVNEXT_OLS_ALIASES.items()
    },
    **_CONVNEXTV2_FULL_NAME_REGISTRY,
    **{
        alias: _CONVNEXTV2_FULL_NAME_REGISTRY[canonical]
        for alias, canonical in _CONVNEXTV2_ALIASES.items()
    },
    # EUPE pretrained variants
    "eupe_convnext_tiny": (
        _EUPE_CONVNEXT_BASE_CFG,
        {**convnext_sizes["tiny"], "act_layer": "exactgelu"},
    ),
    "eupe_convnext_small": (
        _EUPE_CONVNEXT_BASE_CFG,
        {**convnext_sizes["small"], "act_layer": "exactgelu"},
    ),
    "eupe_convnext_base": (
        _EUPE_CONVNEXT_BASE_CFG,
        {**convnext_sizes["base"], "act_layer": "exactgelu"},
    ),
}

# Variants with a real trusted archive in PRETRAINED_ARCHIVE_SHA256 today.
# Keep this in sync with what's actually published;
# pretrained=True on anything else raises a clear error
# instead of a 404 at download time.
_CONVNEXT_PRETRAINED_VARIANTS = {
    "eupe_convnext_tiny",
    "eupe_convnext_small",
    "eupe_convnext_base",
}


def _build_convnext(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> ConvNeXt:
    return build_model_variant(
        ConvNeXt,
        _CONVNEXT_REGISTRY,
        variant,
        pretrained=pretrained,
        inference_mode=inference_mode,
        key=key,
        pretrained_variants=frozenset(_CONVNEXT_PRETRAINED_VARIANTS),
        pretrained_label="ConvNeXt",
        **overrides,
    )


def convnext_atto(**kwargs) -> ConvNeXt:
    """ConvNeXt-Atto — dims [40,80,160,320], depths [2,2,6,2]."""
    return _build_convnext("convnext_atto", **kwargs)


def convnext_a(**kwargs) -> ConvNeXt:
    """Alias for convnext_atto."""
    return convnext_atto(**kwargs)


def convnext_femto(**kwargs) -> ConvNeXt:
    """ConvNeXt-Femto — dims [48,96,192,384], depths [2,2,6,2]."""
    return _build_convnext("convnext_femto", **kwargs)


def convnext_f(**kwargs) -> ConvNeXt:
    """Alias for convnext_femto."""
    return convnext_femto(**kwargs)


def convnext_pico(**kwargs) -> ConvNeXt:
    """ConvNeXt-Pico — dims [64,128,256,512], depths [2,2,6,2]."""
    return _build_convnext("convnext_pico", **kwargs)


def convnext_p(**kwargs) -> ConvNeXt:
    """Alias for convnext_pico."""
    return convnext_pico(**kwargs)


def convnext_nano(**kwargs) -> ConvNeXt:
    """ConvNeXt-Nano — dims [80,160,320,640], depths [2,2,8,2]."""
    return _build_convnext("convnext_nano", **kwargs)


def convnext_n(**kwargs) -> ConvNeXt:
    """Alias for convnext_nano."""
    return convnext_nano(**kwargs)


def convnext_zepto_rms(**kwargs) -> ConvNeXt:
    """ConvNeXt-Zepto (RMSNorm2d blocks) — dims [32,64,128,256], depths [2,2,4,2]."""
    return _build_convnext("convnext_zepto_rms", **kwargs)


def convnext_z(**kwargs) -> ConvNeXt:
    """Alias for convnext_zepto_rms."""
    return convnext_zepto_rms(**kwargs)


def convnext_tiny(**kwargs) -> ConvNeXt:
    """ConvNeXt-Tiny — dims [96,192,384,768], depths [3,3,9,3]."""
    return _build_convnext("convnext_tiny", **kwargs)


def convnext_t(**kwargs) -> ConvNeXt:
    """Alias for convnext_tiny."""
    return convnext_tiny(**kwargs)


def convnext_small(**kwargs) -> ConvNeXt:
    """ConvNeXt-Small — dims [96,192,384,768], depths [3,3,27,3]."""
    return _build_convnext("convnext_small", **kwargs)


def convnext_s(**kwargs) -> ConvNeXt:
    """Alias for convnext_small."""
    return convnext_small(**kwargs)


def convnext_base(**kwargs) -> ConvNeXt:
    """ConvNeXt-Base — dims [128,256,512,1024], depths [3,3,27,3]."""
    return _build_convnext("convnext_base", **kwargs)


def convnext_b(**kwargs) -> ConvNeXt:
    """Alias for convnext_base."""
    return convnext_base(**kwargs)


def convnext_large(**kwargs) -> ConvNeXt:
    """ConvNeXt-Large — dims [192,384,768,1536], depths [3,3,27,3]."""
    return _build_convnext("convnext_large", **kwargs)


def convnext_l(**kwargs) -> ConvNeXt:
    """Alias for convnext_large."""
    return convnext_large(**kwargs)


def convnext_xlarge(**kwargs) -> ConvNeXt:
    """ConvNeXt-XLarge — dims [256,512,1024,2048], depths [3,3,27,3]."""
    return _build_convnext("convnext_xlarge", **kwargs)


def convnext_xl(**kwargs) -> ConvNeXt:
    """Alias for convnext_xlarge."""
    return convnext_xlarge(**kwargs)


def convnext_xxlarge(**kwargs) -> ConvNeXt:
    """ConvNeXt-XXLarge — dims [384,768,1536,3072], depths [3,4,30,3]."""
    return _build_convnext("convnext_xxlarge", **kwargs)


def convnext_xxl(**kwargs) -> ConvNeXt:
    """Alias for convnext_xxlarge."""
    return convnext_xxlarge(**kwargs)


def convnext_atto_ols(**kwargs) -> ConvNeXt:
    """ConvNeXt-Atto with an overlapping-conv stem instead of the patchify stem."""
    return _build_convnext("convnext_atto_ols", **kwargs)


def convnext_a_ols(**kwargs) -> ConvNeXt:
    """Alias for convnext_atto_ols."""
    return convnext_atto_ols(**kwargs)


def convnext_femto_ols(**kwargs) -> ConvNeXt:
    """ConvNeXt-Femto with an overlapping-conv stem instead of the patchify stem."""
    return _build_convnext("convnext_femto_ols", **kwargs)


def convnext_f_ols(**kwargs) -> ConvNeXt:
    """Alias for convnext_femto_ols."""
    return convnext_femto_ols(**kwargs)


def convnext_pico_ols(**kwargs) -> ConvNeXt:
    """ConvNeXt-Pico with an overlapping-conv stem instead of the patchify stem."""
    return _build_convnext("convnext_pico_ols", **kwargs)


def convnext_p_ols(**kwargs) -> ConvNeXt:
    """Alias for convnext_pico_ols."""
    return convnext_pico_ols(**kwargs)


def convnext_nano_ols(**kwargs) -> ConvNeXt:
    """ConvNeXt-Nano with an overlapping-conv stem instead of the patchify stem."""
    return _build_convnext("convnext_nano_ols", **kwargs)


def convnext_n_ols(**kwargs) -> ConvNeXt:
    """Alias for convnext_nano_ols."""
    return convnext_nano_ols(**kwargs)


def convnext_zepto_rms_ols(**kwargs) -> ConvNeXt:
    """ConvNeXt-Zepto (RMSNorm2d blocks) with an overlapping-conv+GELU stem."""
    return _build_convnext("convnext_zepto_rms_ols", **kwargs)


def convnext_z_ols(**kwargs) -> ConvNeXt:
    """Alias for convnext_zepto_rms_ols."""
    return convnext_zepto_rms_ols(**kwargs)


def convnextv2_atto(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Atto (GRN, no LayerScale) — dims [40,80,160,320], depths [2,2,6,2]."""
    return _build_convnext("convnextv2_atto", **kwargs)


def convnextv2_a(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_atto."""
    return convnextv2_atto(**kwargs)


def convnextv2_femto(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Femto (GRN, no LayerScale) — dims [48,96,192,384], depths [2,2,6,2]."""
    return _build_convnext("convnextv2_femto", **kwargs)


def convnextv2_f(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_femto."""
    return convnextv2_femto(**kwargs)


def convnextv2_pico(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Pico (GRN, no LayerScale) — dims [64,128,256,512], depths [2,2,6,2]."""
    return _build_convnext("convnextv2_pico", **kwargs)


def convnextv2_p(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_pico."""
    return convnextv2_pico(**kwargs)


def convnextv2_nano(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Nano (GRN, no LayerScale) — dims [80,160,320,640], depths [2,2,8,2]."""
    return _build_convnext("convnextv2_nano", **kwargs)


def convnextv2_n(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_nano."""
    return convnextv2_nano(**kwargs)


def convnextv2_tiny(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Tiny (GRN, no LayerScale) — dims [96,192,384,768], depths [3,3,9,3]."""
    return _build_convnext("convnextv2_tiny", **kwargs)


def convnextv2_t(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_tiny."""
    return convnextv2_tiny(**kwargs)


def convnextv2_base(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Base (GRN, no LayerScale) — dims [128,256,512,1024], depths [3,3,27,3]."""
    return _build_convnext("convnextv2_base", **kwargs)


def convnextv2_b(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_base."""
    return convnextv2_base(**kwargs)


def convnextv2_large(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Large (GRN, no LayerScale) — dims [192,384,768,1536], depths [3,3,27,3]."""
    return _build_convnext("convnextv2_large", **kwargs)


def convnextv2_l(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_large."""
    return convnextv2_large(**kwargs)


def convnextv2_huge(**kwargs) -> ConvNeXt:
    """ConvNeXt V2-Huge (GRN, no LayerScale) — dims [352,704,1408,2816], depths [3,3,27,3]."""
    return _build_convnext("convnextv2_huge", **kwargs)


def convnextv2_h(**kwargs) -> ConvNeXt:
    """Alias for convnextv2_huge."""
    return convnextv2_huge(**kwargs)


def eupe_convnext_tiny(pretrained: bool = False, **kwargs) -> ConvNeXt:
    """EUPE ConvNeXt-Tiny (LVD-1689M pretrained backbone, no classification head)."""
    return _build_convnext("eupe_convnext_tiny", pretrained=pretrained, **kwargs)


def eupe_convnext_small(pretrained: bool = False, **kwargs) -> ConvNeXt:
    """EUPE ConvNeXt-Small (LVD-1689M pretrained backbone, no classification head)."""
    return _build_convnext("eupe_convnext_small", pretrained=pretrained, **kwargs)


def eupe_convnext_base(pretrained: bool = False, **kwargs) -> ConvNeXt:
    """EUPE ConvNeXt-Base (LVD-1689M pretrained backbone, no classification head)."""
    return _build_convnext("eupe_convnext_base", pretrained=pretrained, **kwargs)
