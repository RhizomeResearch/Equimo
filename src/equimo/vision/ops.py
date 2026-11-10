"""Bilinear sampling operations on channel-first feature maps.

Both operations follow the torchvision and PyTorch definitions, so that boxes
and points prepared for those functions give the same values here.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def _bilinear_weights(
    coords: Float[Array, "..."], size: int
) -> Float[Array, "... size"]:
    """Interpolation weights over one axis, as torchvision's CUDA ROI Align samples.

    A sample outside the map reads the border pixel. torchvision's CPU kernel
    zeroes samples more than one pixel outside instead; the reference runs on
    CUDA, so the CUDA behavior is the one reproduced here.
    """
    coords = jnp.maximum(coords, 0.0)
    low = jnp.floor(coords).astype(jnp.int32)
    at_border = low >= size - 1
    low = jnp.where(at_border, size - 1, low)
    high = jnp.where(at_border, size - 1, low + 1)
    coords = jnp.where(at_border, low.astype(coords.dtype), coords)
    fraction = (coords - low)[..., None]
    weights = (
        jax.nn.one_hot(low, size, dtype=coords.dtype) * (1 - fraction)
        + jax.nn.one_hot(high, size, dtype=coords.dtype) * fraction
    )
    return weights


def _bin_weights(
    start: Float[Array, " boxes"],
    roi_size: Float[Array, " boxes"],
    grid: Float[Array, " boxes"],
    *,
    bins: int,
    max_grid: int,
    size: int,
) -> Float[Array, "boxes bins size"]:
    """Summed sample weights of each bin along one axis of each box."""
    bin_size = (roi_size / bins)[:, None, None]
    bin_index = jnp.arange(bins, dtype=jnp.float32)[None, :, None]
    sample_index = jnp.arange(max_grid, dtype=jnp.float32)[None, None, :]
    # A zero grid has no valid sample; dividing by 1 keeps its coordinates finite.
    divisor = jnp.maximum(grid, 1.0)[:, None, None]
    coords = (
        start[:, None, None]
        + bin_index * bin_size
        + (sample_index + 0.5) * bin_size / divisor
    )
    valid = sample_index < grid[:, None, None]
    weights = _bilinear_weights(coords, size) * valid[..., None]
    return weights.sum(axis=2)


def roi_align(
    features: Float[Array, "channels height width"],
    boxes: Float[Array, "boxes 4"],
    output_size: int | tuple[int, int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Float[Array, "boxes channels out_height out_width"]:
    """Pool each box into a grid of averaged bilinear samples.

    Matches ``torchvision.ops.roi_align`` for one image. ``boxes`` are ``(x1, y1,
    x2, y2)`` and are multiplied by ``spatial_scale`` to reach feature pixels.
    Each output bin averages ``sampling_ratio`` by ``sampling_ratio`` samples.
    With ``sampling_ratio <= 0`` the count is ``ceil(roi_size / output_size)``
    per axis and depends on the box, so it is evaluated against a static maximum
    of ``ceil(height / out_height)`` by ``ceil(width / out_width)`` samples with
    masking. Results match torchvision for boxes no larger than the feature map;
    a larger box is sampled with that maximum. With ``aligned=False`` boxes are
    at least one pixel wide and tall; ``aligned=True`` shifts them by half a
    pixel instead.

    Sampling runs in float32; the output has the dtype of ``features``.
    """
    out_height, out_width = (
        (output_size, output_size) if isinstance(output_size, int) else output_size
    )
    height, width = features.shape[-2:]
    offset = 0.5 if aligned else 0.0
    scaled = boxes.astype(jnp.float32) * spatial_scale - offset
    x1, y1, x2, y2 = (scaled[:, i] for i in range(4))
    roi_width, roi_height = x2 - x1, y2 - y1
    if not aligned:
        roi_width = jnp.maximum(roi_width, 1.0)
        roi_height = jnp.maximum(roi_height, 1.0)

    if sampling_ratio > 0:
        max_grid_height = max_grid_width = sampling_ratio
        grid_height = jnp.full_like(roi_height, sampling_ratio)
        grid_width = jnp.full_like(roi_width, sampling_ratio)
    else:
        max_grid_height = max(math.ceil(height / out_height), 1)
        max_grid_width = max(math.ceil(width / out_width), 1)
        grid_height = jnp.minimum(jnp.ceil(roi_height / out_height), max_grid_height)
        grid_width = jnp.minimum(jnp.ceil(roi_width / out_width), max_grid_width)

    rows = _bin_weights(
        y1,
        roi_height,
        grid_height,
        bins=out_height,
        max_grid=max_grid_height,
        size=height,
    )
    columns = _bin_weights(
        x1,
        roi_width,
        grid_width,
        bins=out_width,
        max_grid=max_grid_width,
        size=width,
    )
    count = jnp.maximum(grid_height * grid_width, 1.0)
    pooled = jnp.einsum(
        "npy,cyx,nqx->ncpq", rows, features.astype(jnp.float32), columns
    )
    return (pooled / count[:, None, None, None]).astype(features.dtype)


def _resize_axis(x: Float[Array, "..."], size: int, axis: int) -> Float[Array, "..."]:
    """Bilinearly resize one axis of float32 ``x``, at torch's sample positions."""
    length = x.shape[axis]
    scale = jnp.asarray(length / size, jnp.float32)
    positions = (jnp.arange(size, dtype=jnp.float32) + 0.5) * scale - 0.5
    low = jnp.clip(jnp.floor(positions), 0, length - 1).astype(jnp.int32)
    high = jnp.clip(low + 1, 0, length - 1)
    fraction = jnp.clip(positions - low, 0.0, 1.0).reshape((-1,) + (1,) * (-axis - 1))
    return (1.0 - fraction) * jnp.take(x, low, axis=axis) + fraction * jnp.take(
        x, high, axis=axis
    )


def resize_bilinear(
    x: Float[Array, "... height width"], size: tuple[int, int]
) -> Float[Array, "... out_height out_width"]:
    """Resize the last two axes bilinearly, in float32.

    Matches ``torch.nn.functional.interpolate(mode="bilinear", align_corners=False)``
    on CUDA, including its float32 sample positions. ``jax.image.resize`` computes
    those positions differently, which moves pixels on a block edge by about 1e-5.
    """
    height, width = size
    x = x.astype(jnp.float32)
    return _resize_axis(_resize_axis(x, width, axis=-1), height, axis=-2)


def _antialias_weights(length: int, size: int) -> np.ndarray:
    """Triangle-filter weights of each output sample over the input axis.

    They only depend on the two static sizes, so they are computed once while
    tracing, the way torch computes its antialiased kernels.
    """
    scale = length / size
    support = max(scale, 1.0)
    weights = np.zeros((size, length), np.float32)
    for index in range(size):
        center = (index + 0.5) * scale
        first = max(int(center - support + 0.5), 0)
        last = min(int(center + support + 0.5), length)
        distance = (np.arange(first, last) + 0.5 - center) / support
        values = np.maximum(1.0 - np.abs(distance), 0.0)
        weights[index, first:last] = values / values.sum()
    return weights


def resize_bilinear_antialias(
    x: Float[Array, "... height width"], size: tuple[int, int]
) -> Float[Array, "... out_height out_width"]:
    """Resize the last two axes with an antialiased bilinear filter, in float32.

    Matches ``torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
    antialias=True)``: each axis uses a triangle filter widened by its downscale
    factor, and upscaling is plain bilinear.
    """
    height, width = size
    rows = jnp.asarray(_antialias_weights(x.shape[-2], height))
    columns = jnp.asarray(_antialias_weights(x.shape[-1], width))
    return rows @ x.astype(jnp.float32) @ columns.T


def grid_sample(
    features: Float[Array, "channels height width"],
    points: Float[Array, "points 2"],
) -> Float[Array, "points channels"]:
    """Bilinearly sample ``features`` at normalized ``(x, y)`` points.

    Matches ``torch.nn.functional.grid_sample`` with ``mode="bilinear"``,
    ``align_corners=False`` and ``padding_mode="zeros"``: -1 and 1 are the outer
    edges of the border pixels, and neighbors outside the map count as zero.
    Coordinates are computed in float32; the output has the dtype of
    ``features``.
    """
    height, width = features.shape[-2:]
    points = points.astype(jnp.float32)
    x = ((points[:, 0] + 1) * width - 1) / 2
    y = ((points[:, 1] + 1) * height - 1) / 2

    def sample(channel: jax.Array) -> jax.Array:
        return jax.scipy.ndimage.map_coordinates(
            channel, [y, x], order=1, mode="constant", cval=0
        )

    return jax.vmap(sample)(features).T
