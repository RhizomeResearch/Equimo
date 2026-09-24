"""Complete registry inventory used by layer contract tests."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from equimo.audio.layers.patch import _PATCH_REGISTRY as AUDIO_PATCHES
from equimo.audio.layers.patch import get_patch as get_audio_patch
from equimo.core.layers.activation import _ACT_REGISTRY as CORE_ACTIVATIONS
from equimo.core.layers.activation import get_act
from equimo.core.layers.attention import (
    _ATTN_BLOCK_REGISTRY as CORE_ATTENTION_BLOCKS,
)
from equimo.core.layers.attention import _ATTN_REGISTRY as CORE_ATTENTION
from equimo.core.layers.attention import get_attn, get_attn_block
from equimo.core.layers.dropout import _DROPOUT_REGISTRY as CORE_DROPOUT
from equimo.core.layers.dropout import get_dropout, register_dropout
from equimo.core.layers.ffn import _FFN_REGISTRY as CORE_FFN
from equimo.core.layers.ffn import get_ffn, register_ffn
from equimo.core.layers.mamba import _MIXER_REGISTRY as CORE_MIXERS
from equimo.core.layers.mamba import get_mixer, register_mixer
from equimo.core.layers.norm import _NORM_REGISTRY as CORE_NORMS
from equimo.core.layers.norm import get_norm, register_norm
from equimo.tabular.layers.attention import _ATTN_REGISTRY as TABULAR_ATTENTION
from equimo.tabular.layers.attention import get_attn as get_tabular_attn
from equimo.tabular.layers.blocks import (
    _ATTN_BLOCK_REGISTRY as TABULAR_ATTENTION_BLOCKS,
)
from equimo.tabular.layers.blocks import get_attn_block as get_tabular_attn_block
from equimo.tabular.layers.decoder import _DECODER_REGISTRY as TABULAR_DECODERS
from equimo.tabular.layers.decoder import _EMBEDDING_REGISTRY as TABULAR_EMBEDDINGS
from equimo.tabular.layers.decoder import get_decoder, get_embedding
from equimo.tabular.layers.preprocessing import (
    _PREPROCESSOR_REGISTRY as TABULAR_PREPROCESSORS,
)
from equimo.tabular.layers.preprocessing import get_preprocessor
from equimo.timeseries.layers.registry import _LAYER_REGISTRY as TIMESERIES_LAYERS
from equimo.timeseries.layers.registry import get_layer as get_timeseries_layer
from equimo.vision.layers.attention import (
    _ATTN_BLOCK_REGISTRY as VISION_ATTENTION_BLOCKS,
)
from equimo.vision.layers.attention import _ATTN_REGISTRY as VISION_ATTENTION
from equimo.vision.layers.attention import get_attn as get_vision_attn
from equimo.vision.layers.attention import get_attn_block as get_vision_attn_block
from equimo.vision.layers.convolution import _CONV_REGISTRY as VISION_CONVOLUTIONS
from equimo.vision.layers.convolution import get_conv
from equimo.vision.layers.downsample import (
    _DOWNSAMPLER_REGISTRY as VISION_DOWNSAMPLERS,
)
from equimo.vision.layers.downsample import get_downsampler, register_downsampler
from equimo.vision.layers.patch import _PATCH_REGISTRY as VISION_PATCHES
from equimo.vision.layers.patch import get_patch as get_vision_patch
from equimo.vision.layers.patch import register_patch as register_vision_patch
from equimo.vision.layers.posemb import _POSEMB_REGISTRY as VISION_POSITIONAL
from equimo.vision.layers.posemb import get_posemb, register_posemb
from equimo.vision.layers.squeeze_excite import _SE_REGISTRY as VISION_SE
from equimo.vision.layers.squeeze_excite import get_se, register_se
from equimo.vision.layers.wavelet import _WAVELET_REGISTRY as VISION_WAVELETS
from equimo.vision.layers.wavelet import get_wavelet, register_wavelet


@dataclass(frozen=True)
class LayerRegistryCase:
    scope: str
    registry: Mapping[str, Any]
    resolve: Callable[[Any], Any]
    register: Callable[..., Callable[[type], type]] | None = None


LAYER_REGISTRY_CASES = (
    LayerRegistryCase("core-activation", CORE_ACTIVATIONS, get_act),
    LayerRegistryCase("core-attention", CORE_ATTENTION, get_attn),
    LayerRegistryCase("core-attention-block", CORE_ATTENTION_BLOCKS, get_attn_block),
    LayerRegistryCase("core-dropout", CORE_DROPOUT, get_dropout, register_dropout),
    LayerRegistryCase("core-ffn", CORE_FFN, get_ffn, register_ffn),
    LayerRegistryCase("core-mixer", CORE_MIXERS, get_mixer, register_mixer),
    LayerRegistryCase("core-norm", CORE_NORMS, get_norm, register_norm),
    LayerRegistryCase("vision-attention", VISION_ATTENTION, get_vision_attn),
    LayerRegistryCase(
        "vision-attention-block", VISION_ATTENTION_BLOCKS, get_vision_attn_block
    ),
    LayerRegistryCase("vision-convolution", VISION_CONVOLUTIONS, get_conv),
    LayerRegistryCase(
        "vision-downsampler",
        VISION_DOWNSAMPLERS,
        get_downsampler,
        register_downsampler,
    ),
    LayerRegistryCase(
        "vision-patch", VISION_PATCHES, get_vision_patch, register_vision_patch
    ),
    LayerRegistryCase(
        "vision-positional", VISION_POSITIONAL, get_posemb, register_posemb
    ),
    LayerRegistryCase("vision-squeeze-excite", VISION_SE, get_se, register_se),
    LayerRegistryCase("vision-wavelet", VISION_WAVELETS, get_wavelet, register_wavelet),
    LayerRegistryCase("audio-patch", AUDIO_PATCHES, get_audio_patch),
    LayerRegistryCase("tabular-attention", TABULAR_ATTENTION, get_tabular_attn),
    LayerRegistryCase(
        "tabular-attention-block",
        TABULAR_ATTENTION_BLOCKS,
        get_tabular_attn_block,
    ),
    LayerRegistryCase("tabular-decoder", TABULAR_DECODERS, get_decoder),
    LayerRegistryCase("tabular-embedding", TABULAR_EMBEDDINGS, get_embedding),
    LayerRegistryCase("tabular-preprocessor", TABULAR_PREPROCESSORS, get_preprocessor),
    LayerRegistryCase("time-series-layer", TIMESERIES_LAYERS, get_timeseries_layer),
)
