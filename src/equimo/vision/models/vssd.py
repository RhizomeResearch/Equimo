__all__ = ["Vssd"]

from typing import Callable, List, Tuple

import equinox as eqx
from jaxtyping import PRNGKeyArray

from equimo.registry import register_model
from equimo.utils import to_list
from equimo.vision.models._features import TokenStemFeatures
from equimo.vision.models.mlla import build_mlla_stack


@register_model("vssd", modality="vision")
class Vssd(TokenStemFeatures, eqx.Module):
    """Vision Mamba with Non-Causal State Space Duality (VSSD)[1].

    A hybrid vision architecture that combines Mamba state space models with attention
    mechanisms in a hierarchical structure. Features progressive spatial reduction
    and channel expansion through multiple stages.

    The model processes images through:
    1. Patch embedding using a stem layer
    2. Multiple stages of Mamba/Attention blocks with downsampling
    3. Global pooling and classification head

    Attributes:
        num_features: Number of features in final stage
        patch_embed: Initial patch embedding stem
        pos_drop: Positional dropout layer
        blocks: List of processing blocks with downsampling
        head: Classification head with normalization

    References:
        [1]: Shi, et al., 2024. https://arxiv.org/abs/2407.18559
    """

    num_features: int = eqx.field(static=True)

    patch_embed: eqx.Module
    pos_drop: eqx.Module
    blocks: Tuple[eqx.Module, ...]
    norm: eqx.Module
    head: eqx.Module

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        dim: int = 64,
        expand: int = 2,
        patch_size: int = 4,
        depths: List[int] = [2, 4, 12, 4],
        num_heads: List[int] = [2, 4, 8, 16],
        attentions_layers: Tuple[str | type[eqx.Module], ...]
        | str
        | type[eqx.Module] = (
            "mamba2mixer",
            "mamba2mixer",
            "mamba2mixer",
            "attention",
        ),
        act_layer: str | Callable = "gelu",
        ffn_layer: str | type[eqx.Module] = "mlp",
        norm_layer: str | type[eqx.Module] = "layernorm",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        mlp_ratio: float = 4.0,
        eps: float = 1e-5,
        num_classes: int | None = 1000,
        **kwargs,
    ):
        """Initialize VSSD model.

        Args:
            img_size: Input image size
            in_channels: Number of input channels
            key: PRNG key for initialization
            dim: Initial model dimension
            expand: Expansion factor for attention head dimension
            patch_size: Size of image patches
            depths: Number of blocks in each stage
            num_heads: Number of attention heads in each stage
            attentions_layers: Types of attention/mixing layers per stage (names or classes)
            act_layer: Activation function (name or callable)
            ffn_layer: FFN block type (name or class)
            norm_layer: Normalization layer type (name or class)
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            drop_path_uniform: Whether to use uniform drop path rate
            mlp_ratio: MLP expansion ratio
            eps: Epsilon for normalization layers
            num_classes: Number of classification classes
            **kwargs: Additional arguments
        """
        n_chunks = len(depths)
        num_heads = to_list(num_heads, n_chunks)
        head_dims = [int(dim * 2**i) * expand // num_heads[i] for i in range(n_chunks)]
        (
            self.num_features,
            self.patch_embed,
            self.pos_drop,
            self.blocks,
            self.norm,
            self.head,
        ) = build_mlla_stack(
            img_size=img_size,
            in_channels=in_channels,
            key=key,
            dim=dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            attentions_layers=attentions_layers,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            mlp_ratio=mlp_ratio,
            eps=eps,
            num_classes=num_classes,
            use_dwc=False,
            head_dims=head_dims,
        )
