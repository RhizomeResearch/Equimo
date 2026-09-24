__all__ = [
    "SpectrogramPatchEmbedding",
    "get_patch",
    "register_patch",
]

from typing import Tuple

import equinox as eqx
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.utils import make_2tuple
from equimo.core.layers._registry import make_get, make_register

_PATCH_REGISTRY: dict[str, type[eqx.Module]] = {}


register_patch = make_register(_PATCH_REGISTRY)


get_patch = make_get(_PATCH_REGISTRY, kind="audio patch module", plural="modules")


@register_patch()
class SpectrogramPatchEmbedding(eqx.Module):
    """Patch embedding for single-channel log-mel spectrograms.

    Inputs follow the AST convention ``(time, frequency)`` and are internally
    projected as ``(1, frequency, time)`` so ``fstride`` and ``tstride`` map to
    the same axes as the AST reference implementation.
    """

    patch_size: Tuple[int, int] = eqx.field(static=True)
    stride: Tuple[int, int] = eqx.field(static=True)
    img_size: Tuple[int, int] = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)

    proj: eqx.nn.Conv

    def __init__(
        self,
        embed_dim: int,
        patch_size: int | Tuple[int, int],
        *,
        input_fdim: int,
        input_tdim: int,
        fstride: int,
        tstride: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = make_2tuple(patch_size)
        self.stride = (fstride, tstride)
        self.img_size = (input_fdim, input_tdim)

        f_dim = (input_fdim - self.patch_size[0]) // fstride + 1
        t_dim = (input_tdim - self.patch_size[1]) // tstride + 1
        if f_dim <= 0 or t_dim <= 0:
            raise ValueError(
                "Patch size must fit within the configured spectrogram dimensions."
            )

        self.grid_size = (f_dim, t_dim)
        self.num_patches = f_dim * t_dim
        self.proj = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "time frequency"]
    ) -> Float[Array, "num_patches dim"]:
        T, F = x.shape
        if T != self.img_size[1]:
            raise AssertionError(
                f"Input time dimension ({T}) doesn't match model ({self.img_size[1]})"
            )
        if F != self.img_size[0]:
            raise AssertionError(
                f"Input frequency dimension ({F}) doesn't match model ({self.img_size[0]})"
            )

        x = rearrange(x, "t f -> 1 f t")
        x = self.proj(x)
        return rearrange(x, "c f t -> (f t) c")
