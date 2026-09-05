# ty: ignore[invalid-assignment]
# ty: ignore[call-non-callable]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core._prng import split_for_mode
from equimo.core.layers.activation import get_act
from equimo.vision.layers.convolution import DoubleConvBlock, SingleConvBlock
from equimo.core.layers.generic import Residual
from equimo.core.layers.norm import get_norm
from equimo.vision.layers.patch import SEPatchMerging
from equimo.utils import make_divisible, nearest_power_of_2_divisor
from equimo.core.layers._registry import make_get, make_register

_DOWNSAMPLER_REGISTRY: dict[str, type[eqx.Module]] = {}


register_downsampler = make_register(_DOWNSAMPLER_REGISTRY)


get_downsampler = make_get(_DOWNSAMPLER_REGISTRY)


@register_downsampler()
class ConvNormDownsampler(eqx.Module):
    """A module that performs spatial downsampling using strided convolution.

    This module reduces spatial dimensions (height and width) by a factor of 2
    (``"simple"`` mode) or 4 (``"double"`` mode) while optionally increasing the
    channel dimension.

    Modes:
        ``"simple"``: A single 3×3 strided conv followed by optional GroupNorm.
            Spatial dimensions are halved (stride 2).
        ``"double"``: Two consecutive 3×3 strided convs each with stride 2,
            separated by optional GroupNorm and an activation. Spatial dimensions
            are quartered overall. The intermediate channel count is
            ``out_channels // 2``.

    Note:
        ``act_layer`` is only applied in ``"double"`` mode between the two
        convolution layers. It is ignored in ``"simple"`` mode.

    Attributes:
        downsampler: Sequential module that performs the downsampling.
    """

    downsampler: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int | None = None,
        act_layer: str | Callable | None = None,
        use_bias: bool = False,
        use_norm: bool = True,
        mode: Literal["double", "simple"] = "simple",
        key: PRNGKeyArray,
    ):
        """Initialize ConvNormDownsampler.

        Args:
            in_channels: Number of input spatial channels.
            out_channels: Number of output spatial channels (default: 2 × in_channels).
            act_layer: Activation function inserted between the two convolutions in
                ``"double"`` mode. Ignored in ``"simple"`` mode. Default: None.
            use_bias: Whether to use bias in the convolutional layers.
            use_norm: Whether to insert GroupNorm after each convolution.
            mode: Downsampling strategy — ``"simple"`` (2×) or ``"double"`` (4×).
            key: PRNG key for initialization.
        """
        if act_layer is not None:
            act_layer = get_act(act_layer)

        out_channels = out_channels if out_channels else 2 * in_channels

        match mode:
            case "simple":
                layers = [
                    eqx.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_bias=use_bias,
                        key=key,
                    ),
                    eqx.nn.GroupNorm(
                        nearest_power_of_2_divisor(out_channels, 32), out_channels
                    )
                    if use_norm
                    else eqx.nn.Identity(),
                ]
                if act_layer is not None:
                    layers.append(eqx.nn.Lambda(act_layer))
                self.downsampler = eqx.nn.Sequential(layers)

            case "double":
                _d = out_channels // 2
                layers = [
                    eqx.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=_d,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_bias=use_bias,
                        key=jr.fold_in(key, 0),
                    ),
                    eqx.nn.GroupNorm(nearest_power_of_2_divisor(_d, 32), _d)
                    if use_norm
                    else eqx.nn.Identity(),
                ]
                if act_layer is not None:
                    layers.append(eqx.nn.Lambda(act_layer))
                layers += [
                    eqx.nn.Conv2d(
                        in_channels=_d,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_bias=use_bias,
                        key=jr.fold_in(key, 1),
                    ),
                    eqx.nn.GroupNorm(
                        nearest_power_of_2_divisor(out_channels, 32), out_channels
                    )
                    if use_norm
                    else eqx.nn.Identity(),
                ]
                self.downsampler = eqx.nn.Sequential(layers)

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        *args,
        **kwargs,
    ) -> Float[Array, "out_channels new_height new_width"]:
        return self.downsampler(x)


@register_downsampler()
class PWSEDownsampler(eqx.Module):
    """Downsampling module for spatial feature reduction.

    Combines convolution blocks and SE patch merging to reduce spatial dimensions
    (by 2×) while increasing the channel count. Uses residual connections and
    alternates between depthwise and pointwise convolutions.

    The pipeline is:
    1. Depthwise 3×3 conv with residual (``conv1``)
    2. Pointwise 1×1 double conv (``conv2``)
    3. SE patch merging — halves spatial dims, changes channel count (``patch_merging``)
    4. Depthwise 3×3 conv with residual (``conv3``)
    5. Pointwise 1×1 double conv (``conv4``)

    Attributes:
        conv1: Depthwise convolution with residual connection.
        conv2: Pointwise double convolution block.
        conv3: Depthwise convolution with residual connection (post-merge).
        conv4: Pointwise double convolution block (post-merge).
        patch_merging: SE patch merging layer that halves spatial dimensions.
    """

    conv1: eqx.Module
    conv2: eqx.Module
    conv3: eqx.Module
    conv4: eqx.Module
    patch_merging: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        drop_path: float = 0.0,
        **kwargs,
    ):
        """Initialize PWSEDownsampler.

        Args:
            in_channels: Number of input spatial channels.
            out_channels: Number of output spatial channels.
            key: PRNG key for initialization.
            drop_path: Drop path rate applied inside conv blocks.
            **kwargs: Ignored; present for drop-in use via the registry.
        """
        key_conv1, key_conv2, key_conv3, key_conv4, key_pm = jr.split(key, 5)
        self.conv1 = Residual(
            SingleConvBlock(
                in_channels,
                in_channels,
                act_layer=None,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                key=key_conv1,
            ),
            drop_path=drop_path,
        )
        self.conv2 = DoubleConvBlock(
            channels=in_channels,
            hidden_channels=in_channels * 2,
            act_layer=None,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv2,
        )
        self.patch_merging = SEPatchMerging(
            in_channels=in_channels,
            out_channels=out_channels,
            key=key_pm,
        )
        self.conv3 = Residual(
            SingleConvBlock(
                out_channels,
                out_channels,
                act_layer=None,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                key=key_conv3,
            ),
            drop_path=drop_path,
        )
        self.conv4 = DoubleConvBlock(
            channels=out_channels,
            hidden_channels=out_channels * 2,
            act_layer=None,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv4,
        )

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        key: PRNGKeyArray,
        inference: bool | None = None,
    ) -> Float[Array, "out_channels new_height new_width"]:
        """Apply downsampling to input features.

        Args:
            x: Input spatial feature tensor of shape (C, H, W).
            key: PRNG key for stochastic operations (drop path).
            inference: If True, disables stochastic operations.

        Returns:
            Downsampled feature tensor with halved spatial dimensions and
            increased channel count.
        """
        key_conv1, key_conv2, key_conv3, key_conv4 = split_for_mode(
            key, 4, inference=inference
        )
        x = self.conv2(
            self.conv1(x, inference=inference, key=key_conv1),
            inference=inference,
            key=key_conv2,
        )
        x = self.patch_merging(x)
        x = self.conv4(
            self.conv3(x, inference=inference, key=key_conv3),
            inference=inference,
            key=key_conv4,
        )

        return x


@register_downsampler()
class ConvNeXtStem(eqx.Module):
    """ConvNeXt stem: 4x spatial downsampling via stride-4 convolution + norm.

    This is the initial patchification layer used in ConvNeXt, analogous to a
    ViT patch embedding with patch_size=4.

    Attributes:
        conv: Stride-4 convolution that patchifies the input.
        norm: Channel-wise norm applied after convolution
            (LayerNorm2d by default; SimpleNorm2d for the zepto_rms size family).
    """

    conv: eqx.nn.Conv2d
    norm: eqx.Module

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int,
        norm_layer: str | type[eqx.Module] = "layernorm2d",
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=4,
            key=key,
        )
        self.norm = get_norm(norm_layer)(out_channels, eps=1e-6)

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        *args,
        **kwargs,
    ) -> Float[Array, "out_channels new_height new_width"]:
        return self.norm(self.conv(x))


@register_downsampler()
class ConvNeXtOverlapStem(eqx.Module):
    """4x downsampling via two overlapping stride-2 convolutions ('_ols' stems).

    Used by timm's ConvNeXt "_ols" variants
    in place of ConvNeXtStem's single non-overlapping stride-4 conv.
    ``mid_ratio``/``act_layer`` cover all three timm stem flavors from one class:

    - ``mid_ratio=0.5``: 'overlap_tiered' (atto_ols/femto_ols/pico_ols)
    - ``mid_ratio=1.0``: 'overlap' (nano_ols)
    - ``mid_ratio=1.0`` with ``act_layer`` set: 'overlap_act' (zepto_rms_ols)

    Attributes:
        conv1: First stride-2 3x3 convolution,
            to the (optionally reduced) mid channel count.
        conv2: Second stride-2 3x3 convolution, to out_channels.
        act: Optional activation between the two convolutions.
        norm: Channel-wise norm applied after the second convolution
            (LayerNorm2d by default; SimpleNorm2d for the zepto_rms size family).
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    act: Callable | None = eqx.field(static=True)
    norm: eqx.Module

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int,
        mid_ratio: float = 1.0,
        act_layer: str | Callable | None = None,
        norm_layer: str | type[eqx.Module] = "layernorm2d",
        key: PRNGKeyArray,
        **kwargs,
    ):
        key1, key2 = jr.split(key, 2)
        mid_channels = make_divisible(out_channels * mid_ratio, 8)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            key=key1,
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            key=key2,
        )
        self.act = get_act(act_layer) if act_layer is not None else None
        self.norm = get_norm(norm_layer)(out_channels, eps=1e-6)

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        *args,
        **kwargs,
    ) -> Float[Array, "out_channels new_height new_width"]:
        x = self.conv1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.conv2(x)
        return self.norm(x)


@register_downsampler()
class ConvNeXtDownsampler(eqx.Module):
    """ConvNeXt inter-stage downsampler: norm + stride-2 convolution.

    Reduces spatial dimensions by 2x while changing the channel count.
    Normalization is applied before the convolution, following the ConvNeXt
    design where each downsampling layer normalizes the input first.

    Attributes:
        norm: Channel-wise norm applied before convolution
            (LayerNorm2d by default; SimpleNorm2d for the zepto_rms size family).
        conv: Stride-2 convolution that halves spatial dimensions.
    """

    norm: eqx.Module
    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int,
        norm_layer: str | type[eqx.Module] = "layernorm2d",
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.norm = get_norm(norm_layer)(in_channels, eps=1e-6)
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        *args,
        **kwargs,
    ) -> Float[Array, "out_channels new_height new_width"]:
        return self.conv(self.norm(x))
