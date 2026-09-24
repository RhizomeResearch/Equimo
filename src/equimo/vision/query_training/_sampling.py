"""Explicit-key point sampling with supervision-aware interpolation."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class PointSamples(eqx.Module):
    """Normalized ``(..., points, 2)`` x/y coordinates and boolean validity."""

    coordinates: jax.Array
    valid: jax.Array


def sample_mask_points(
    masks: jax.Array,
    points: PointSamples,
    *,
    spatial_valid: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Bilinearly sample ``(..., H, W)`` masks at shared ``(P,2)`` points.

    Coordinates are normalized x/y with pixel centers at ((x+.5)/W,(y+.5)/H).
    Logits (no ``spatial_valid``) use border extension. Targets use only valid
    in-bounds neighbors, renormalizing their weights. Unsupported or invalid
    points return zero and false. Invalid target values are removed *before*
    interpolation; even NaNs in ignored pixels cannot enter the result.
    """
    if masks.ndim < 2 or min(masks.shape[-2:]) <= 0:
        raise ValueError("Masks need positive spatial dimensions.")
    if points.coordinates.ndim != 2 or points.coordinates.shape[-1] != 2:
        raise ValueError("Expected shared (points, 2) x/y coordinates.")
    if (
        points.valid.shape != points.coordinates.shape[:-1]
        or points.valid.dtype != jnp.bool_
    ):
        raise ValueError("Point validity must be boolean and match coordinates.")
    height, width = masks.shape[-2:]
    if spatial_valid is not None and (
        spatial_valid.shape != (height, width) or spatial_valid.dtype != jnp.bool_
    ):
        raise ValueError("Spatial validity must be boolean (height, width).")
    coords = jax.lax.stop_gradient(points.coordinates.astype(jnp.float32))
    valid = points.valid & jnp.all(
        jnp.isfinite(coords) & (coords >= 0) & (coords <= 1), axis=-1
    )
    coords = jnp.where(valid[:, None], coords, 0.5)
    x = coords[:, 0] * width - 0.5
    y = coords[:, 1] * height - 0.5
    x0, y0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
    dx, dy = x - x0, y - y0
    values = masks.astype(jnp.float32)
    if spatial_valid is not None:
        values = jnp.where(spatial_valid, values, 0)
    result = jnp.zeros((*masks.shape[:-2], coords.shape[0]), dtype=jnp.float32)
    mass = jnp.zeros((coords.shape[0],), dtype=jnp.float32)
    for offset_x, offset_y, weight in (
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ):
        ix, iy = x0 + offset_x, y0 + offset_y
        cx, cy = jnp.clip(ix, 0, width - 1), jnp.clip(iy, 0, height - 1)
        supported = valid
        if spatial_valid is not None:
            supported = (
                supported
                & (ix >= 0)
                & (ix < width)
                & (iy >= 0)
                & (iy < height)
                & spatial_valid[cy, cx]
            )
        weight = jnp.where(supported, weight, 0)
        # A zero-weight neighbor must not inject its NaN into a valid point.
        result += jnp.where(weight > 0, values[..., cy, cx], 0) * weight
        mass += weight
    valid = valid & (mass > 0)
    return jnp.where(valid, result / jnp.where(mass > 0, mass, 1), 0), valid


def sample_query_points(
    spatial_valid: jax.Array,
    *,
    key: jax.Array,
    num_points: int,
    mask_logits: jax.Array | None = None,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
) -> PointSamples:
    """Sample valid pixel cells uniformly, optionally prioritizing uncertainty.

    Without logits, returns shared ``(P,2)`` coordinates. With ``(Q,Hm,Wm)``
    logits, returns ``(Q,P,2)`` coordinates: sample an oversized uniform pool,
    select the largest ``-abs(interpolated_logit)`` values, then append fresh
    uniform points, as in PointRend. Tied uncertainty preserves candidate order.
    Selection is stopped from differentiation. Sampling is with replacement;
    a single valid pixel suffices. An empty domain returns all-invalid points.
    """
    if (
        spatial_valid.ndim != 2
        or min(spatial_valid.shape) <= 0
        or spatial_valid.dtype != jnp.bool_
    ):
        raise ValueError("Spatial validity must be a nonempty boolean (H,W) array.")
    if (
        not isinstance(num_points, int)
        or isinstance(num_points, bool)
        or num_points <= 0
    ):
        raise ValueError("num_points must be a positive integer.")
    if (
        not math.isfinite(oversample_ratio)
        or oversample_ratio < 1
        or not 0 <= importance_sample_ratio <= 1
    ):
        raise ValueError(
            "Require oversample_ratio >= 1 and importance_sample_ratio in [0,1]."
        )

    def uniform(key, count):
        index_key, jitter_key = jr.split(key)
        height, width = spatial_valid.shape
        flat = spatial_valid.reshape(-1)
        count_valid = jnp.sum(flat, dtype=jnp.int32)
        # Fixed-size compaction avoids a P*H*W categorical/Gumbel allocation.
        indices = jnp.nonzero(flat, size=flat.size, fill_value=0)[0]
        rank = jr.randint(index_key, (count,), 0, jnp.maximum(count_valid, 1))
        selected = indices[rank]
        cells = jnp.stack((selected % width, selected // width), axis=-1)
        coords = (
            cells + jr.uniform(jitter_key, (count, 2), dtype=jnp.float32)
        ) / jnp.asarray((width, height))
        return PointSamples(coords, jnp.full((count,), count_valid > 0))

    if mask_logits is None:
        return uniform(key, num_points)
    if mask_logits.ndim != 3 or min(mask_logits.shape) <= 0:
        raise ValueError("Uncertainty sampling expects nonempty (Q,H,W) logits.")
    candidates = int(num_points * oversample_ratio)
    important = int(num_points * importance_sample_ratio)

    def per_mask(logits, key):
        candidate_key, random_key = jr.split(key)
        pool = uniform(candidate_key, candidates)
        values, _ = sample_mask_points(logits, pool)
        order = jnp.argsort(jnp.abs(values), stable=True)[:important]
        random = uniform(random_key, num_points - important)
        return PointSamples(
            jnp.concatenate((pool.coordinates[order], random.coordinates)),
            jnp.concatenate((pool.valid[order], random.valid)),
        )

    return jax.vmap(per_mask)(
        jax.lax.stop_gradient(mask_logits), jr.split(key, mask_logits.shape[0])
    )
