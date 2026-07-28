# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]
# ty: ignore[unresolved-attribute]
__all__ = ["Mlla"]

from typing import Callable, List, Tuple

import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.vision.layers.convolution import Stem
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.generic import BlockChunk
from equimo.core.layers.norm import get_norm
from equimo.registry import register_model
from equimo.utils import make_drop_path_schedule, to_list
from equimo.vision.layers import get_layer
from equimo.vision.models._features import TokenStemFeatures


def build_mlla_stack(
    *,
    img_size: int,
    in_channels: int,
    key: PRNGKeyArray,
    dim: int,
    patch_size: int,
    depths: List[int],
    num_heads: List[int],
    attentions_layers: Tuple[str | type[eqx.Module], ...] | str | type[eqx.Module],
    act_layer: str | Callable,
    ffn_layer: str | type[eqx.Module],
    norm_layer: str | type[eqx.Module],
    drop_rate: float,
    drop_path_rate: float,
    drop_path_uniform: bool,
    mlp_ratio: float,
    eps: float,
    num_classes: int | None,
    use_dwc: bool,
    head_dims: List[int] | None = None,
) -> tuple[int, eqx.Module, eqx.Module, tuple, eqx.Module, eqx.Module]:
    """Build the shared MLLA/VSSD component stack.

    Returns ``(num_features, patch_embed, pos_drop, blocks, norm, head)``
    for the caller to assign to its own fields.
    """

    act_layer = get_act(act_layer)
    ffn_layer = get_ffn(ffn_layer)
    norm_layer = get_norm(norm_layer)

    key_stem, key_head, *block_subkeys = jr.split(key, 2 + len(depths))

    n_chunks = len(depths)
    num_features = int(dim * 2 ** (n_chunks - 1))

    patch_embed = Stem(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=dim,
        key=key_stem,
    )
    patches_resolution = patch_embed.patches_resolution

    pos_drop = eqx.nn.Dropout(drop_rate)

    dpr = make_drop_path_schedule(drop_path_rate, depths, uniform=drop_path_uniform)

    num_heads = to_list(num_heads, n_chunks)
    attentions_layers = tuple(to_list(attentions_layers, n_chunks))
    blocks = tuple(
        BlockChunk(
            depth=depth,
            module="mllablock",
            module_kwargs={
                "dim": int(dim * 2**i),
                "input_resolution": (
                    patches_resolution[0] // (2**i),
                    patches_resolution[1] // (2**i),
                ),
                "num_heads": num_heads[i],
                "act_layer": act_layer,
                "use_dwc": use_dwc,
                "attn_layer": attentions_layers[i],
                "mlp_ratio": mlp_ratio,
                "ffn_layer": ffn_layer,
                "eps": eps,
                **({"head_dim": head_dims[i]} if head_dims is not None else {}),
            },
            downsampler="patchmerging" if (i < n_chunks - 1) else None,
            downsampler_kwargs={"in_dim": int(dim * 2**i)} if i < n_chunks - 1 else {},
            downsample_last=True,
            drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
            layer_resolver=get_layer,
            key=block_subkeys[i],
        )
        for i, depth in enumerate(depths)
    )

    norm = norm_layer(num_features, eps=eps)
    head = (
        eqx.nn.Linear(num_features, num_classes, key=key_head)
        if num_classes is not None and num_classes > 0
        else eqx.nn.Identity()
    )
    return num_features, patch_embed, pos_drop, blocks, norm, head


@register_model("mlla", modality="vision")
class Mlla(TokenStemFeatures, eqx.Module):
    """Mamba-like Linear Attention (MLLA) Vision Model[1].

    A vision transformer architecture that combines linear attention mechanisms
    inspired by Mamba with hierarchical feature processing. The model processes
    images through patches, applies position-aware dropouts, and uses a series
    of attention blocks with progressive feature resolution reduction.

    Attributes:
        num_features: Number of features in the final layer
        patch_embed: Patch embedding layer (Stem)
        pos_drop: Positional dropout layer
        blocks: List of processing blocks
        head: Classification head

    References:
        [1]: Han, et al., 2024. https://arxiv.org/abs/2405.16605
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
        dim: int = 96,
        patch_size: int = 4,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        attentions_layers: Tuple[str | type[eqx.Module], ...]
        | str
        | type[eqx.Module] = "linearattention",
        act_layer: str | Callable = "silu",
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
        """Initialize the MLLA model.

        Args:
            img_size: Input image size
            in_channels: Number of input channels
            key: PRNG key for random operations
            dim: Initial embedding dimension
            patch_size: Size of image patches
            depths: Number of blocks at each stage
            num_heads: Number of attention heads at each stage
            attentions_layers: Type of attention layer(s) to use (name or class)
            act_layer: Activation function (name or callable)
            ffn_layer: FFN block type (name or class)
            norm_layer: Normalization layer type (name or class)
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate
            drop_path_uniform: Whether to use uniform drop path rates
            mlp_ratio: MLP expansion ratio
            eps: Epsilon for normalization layers
            num_classes: Number of output classes (None for feature extraction)
        """
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
            use_dwc=True,
        )
