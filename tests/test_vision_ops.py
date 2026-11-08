import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.vision.ops import grid_sample, roi_align


KEY = jr.PRNGKey(0)


def _numpy_bilinear(channel, y, x):
    """One ROI Align sample, following torchvision's per-sample rules."""
    height, width = channel.shape
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return np.float32(0)
    y, x = max(y, np.float32(0)), max(x, np.float32(0))
    y_low, x_low = int(y), int(x)
    if y_low >= height - 1:
        y_low = y_high = height - 1
        y = np.float32(y_low)
    else:
        y_high = y_low + 1
    if x_low >= width - 1:
        x_low = x_high = width - 1
        x = np.float32(x_low)
    else:
        x_high = x_low + 1
    ly, lx = y - y_low, x - x_low
    hy, hx = 1 - ly, 1 - lx
    return (
        hy * hx * channel[y_low, x_low]
        + hy * lx * channel[y_low, x_high]
        + ly * hx * channel[y_high, x_low]
        + ly * lx * channel[y_high, x_high]
    )


def _numpy_roi_align(
    features, boxes, output_size, spatial_scale, sampling_ratio, aligned
):
    """Loop-based float32 ROI Align with one bilinear sample at a time."""
    features = np.asarray(features, np.float32)
    out_height, out_width = output_size
    scale, offset = np.float32(spatial_scale), np.float32(0.5 if aligned else 0.0)
    output = np.zeros(
        (len(boxes), features.shape[0], out_height, out_width), np.float32
    )
    for n, box in enumerate(np.asarray(boxes, np.float32)):
        x1, y1, x2, y2 = box * scale - offset
        roi_width, roi_height = x2 - x1, y2 - y1
        if not aligned:
            roi_width = max(roi_width, np.float32(1))
            roi_height = max(roi_height, np.float32(1))
        bin_height = roi_height / np.float32(out_height)
        bin_width = roi_width / np.float32(out_width)
        grid_height = sampling_ratio if sampling_ratio > 0 else math.ceil(bin_height)
        grid_width = sampling_ratio if sampling_ratio > 0 else math.ceil(bin_width)
        count = max(grid_height * grid_width, 1)
        for p in range(out_height):
            for q in range(out_width):
                for c, channel in enumerate(features):
                    total = np.float32(0)
                    for iy in range(grid_height):
                        y = (
                            y1
                            + p * bin_height
                            + np.float32(iy + 0.5)
                            * bin_height
                            / np.float32(grid_height)
                        )
                        for ix in range(grid_width):
                            x = (
                                x1
                                + q * bin_width
                                + np.float32(ix + 0.5)
                                * bin_width
                                / np.float32(grid_width)
                            )
                            total += _numpy_bilinear(channel, y, x)
                    output[n, c, p, q] = total / count
    return output


def _numpy_grid_sample(features, points):
    """Bilinear sampling with zero padding outside the map, one point at a time."""
    features = np.asarray(features, np.float64)
    _, height, width = features.shape
    output = []
    for px, py in np.asarray(points, np.float64):
        x = ((px + 1) * width - 1) / 2
        y = ((py + 1) * height - 1) / 2
        x0, y0 = math.floor(x), math.floor(y)
        value = np.zeros(features.shape[0])
        for yi, wy in ((y0, y0 + 1 - y), (y0 + 1, y - y0)):
            for xi, wx in ((x0, x0 + 1 - x), (x0 + 1, x - x0)):
                if 0 <= yi < height and 0 <= xi < width:
                    value += wy * wx * features[:, yi, xi]
        output.append(value)
    return np.stack(output)


ROI_BOXES = np.array(
    [
        [1.3, 2.1, 9.7, 7.4],  # interior
        [0.0, 0.0, 12.0, 10.0],  # the whole map
        [-0.8, 7.5, 3.2, 10.6],  # past the top-left and bottom borders
        [10.2, -0.6, 12.9, 4.3],  # past the right and top borders
        [5.4, 4.6, 5.7, 4.8],  # sub-pixel
        [6.0, 3.0, 4.0, 5.0],  # reversed x extent
    ],
    np.float32,
)


@pytest.mark.parametrize(
    "output_size, spatial_scale, sampling_ratio, aligned",
    [
        ((3, 4), 1.0, -1, False),
        ((3, 4), 1.0, -1, True),
        ((2, 2), 1.0, 2, False),
        ((2, 3), 0.5, -1, False),
        ((5, 5), 1.0, 0, True),
    ],
)
def test_roi_align_matches_numpy_reference(
    output_size, spatial_scale, sampling_ratio, aligned
):
    features = jr.normal(KEY, (3, 10, 12))
    boxes = ROI_BOXES / spatial_scale

    output = roi_align(
        features,
        jnp.asarray(boxes),
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )
    expected = _numpy_roi_align(
        features, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )

    assert output.shape == (len(boxes), 3, *output_size)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_roi_align_clamps_samples_up_to_one_pixel_outside_the_map():
    features = jr.normal(KEY, (3, 10, 12))
    # Single samples at (y, x) = (10, -1), (-1, 12), and (10.01, 5) just beyond.
    boxes = jnp.array(
        [[-1.5, 9.5, -0.5, 10.5], [11.5, -1.5, 12.5, -0.5], [4.5, 9.51, 5.5, 10.51]]
    )

    output = roi_align(features, boxes, 1, sampling_ratio=1)[:, :, 0, 0]

    np.testing.assert_allclose(output[0], features[:, 9, 0], rtol=1e-6)
    np.testing.assert_allclose(output[1], features[:, 0, 11], rtol=1e-6)
    np.testing.assert_array_equal(output[2], 0)


def test_roi_align_averages_a_linear_map_to_its_bin_centers():
    # Bilinear interpolation reproduces a linear map, so each bin averages to the
    # map's value at the bin center.
    rows, columns = jnp.meshgrid(jnp.arange(10.0), jnp.arange(12.0), indexing="ij")
    features = jnp.stack([2 * rows + 3 * columns + 1, -rows])
    box = jnp.array([[1.5, 2.0, 8.5, 7.0]])

    output = roi_align(features, box, (5, 7))

    center_y = 2.0 + (jnp.arange(5) + 0.5)
    center_x = 1.5 + (jnp.arange(7) + 0.5)
    expected = jnp.stack(
        [
            2 * center_y[:, None] + 3 * center_x[None, :] + 1,
            -jnp.broadcast_to(center_y[:, None], (5, 7)),
        ]
    )
    np.testing.assert_allclose(output[0], expected, rtol=1e-5, atol=1e-5)


def test_roi_align_traces_and_differentiates_in_low_precision():
    features = jr.normal(KEY, (4, 16, 16))
    boxes = jnp.array([[2.2, 3.1, 14.0, 9.5], [0.0, 0.0, 16.0, 16.0]])

    output = jax.jit(lambda f, b: roi_align(f, b, 7))(
        features.astype(jnp.bfloat16), boxes
    )
    reference = roi_align(features, boxes, 7)
    feature_gradient, box_gradient = jax.grad(
        lambda f, b: jnp.sum(roi_align(f, b, 7) ** 2), argnums=(0, 1)
    )(features, boxes)

    assert output.dtype == jnp.bfloat16
    np.testing.assert_allclose(output.astype(jnp.float32), reference, atol=0.05)
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(box_gradient))


def test_grid_sample_matches_numpy_reference():
    features = jr.normal(KEY, (3, 6, 8))
    points = jnp.array(
        [
            [0.1, -0.3],
            [-1.0, -1.0],  # outer corner: a quarter of the corner pixel
            [1.0, 0.0],
            [0.97, 0.99],
            [-1.2, 0.4],  # half a pixel outside: partly zero padding
            [1.6, 0.0],  # beyond the padding: zero
            [0.0, 0.0],
        ]
    )

    output = grid_sample(features, points)

    assert output.shape == (7, 3)
    np.testing.assert_allclose(
        output, _numpy_grid_sample(features, points), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(output[1], 0.25 * features[:, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(output[5], 0.0)
    np.testing.assert_allclose(
        output[6], features[:, 2:4, 3:5].mean(axis=(1, 2)), rtol=1e-5, atol=1e-6
    )


def test_grid_sample_traces_batches_and_differentiates():
    features = jr.normal(KEY, (2, 4, 6, 8))
    points = jr.uniform(jr.PRNGKey(1), (2, 5, 2), minval=-1.0, maxval=1.0)

    output = jax.jit(jax.vmap(grid_sample))(features.astype(jnp.bfloat16), points)
    gradient = jax.grad(lambda f: jnp.sum(jax.vmap(grid_sample)(f, points)))(features)

    assert output.shape == (2, 5, 4)
    assert output.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        output.astype(jnp.float32), jax.vmap(grid_sample)(features, points), atol=0.05
    )
    assert jnp.all(jnp.isfinite(gradient))
