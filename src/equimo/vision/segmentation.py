# ty: ignore[invalid-return-type]
"""Prediction adapters for query-based image segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax


class PanopticPrediction(eqx.Module):
    """Class and segment maps; ``num_classes`` and ``-1`` mean void."""

    class_ids: jax.Array
    segment_ids: jax.Array


def resize_mask_logits(mask_logits: jax.Array, size: tuple[int, int]) -> jax.Array:
    """Resize query logits before sigmoid, using half-pixel bilinear sampling."""
    if mask_logits.ndim != 3 or min(size) <= 0:
        raise ValueError("Expected (queries, height, width) logits and positive size.")
    return jax.image.resize(
        mask_logits, (mask_logits.shape[0], *size), "linear", antialias=False
    )


def semantic_scores(
    mask_logits: jax.Array,
    class_logits: jax.Array,
    *,
    output_size: tuple[int, int] | None = None,
) -> jax.Array:
    """Combine query masks and class scores, excluding no-object afterward."""
    _validate_logits(mask_logits, class_logits)
    if output_size is not None:
        mask_logits = resize_mask_logits(mask_logits, output_size)
    classes = jax.nn.softmax(class_logits.astype(jnp.float32), axis=-1)[..., :-1]
    masks = jax.nn.sigmoid(mask_logits.astype(jnp.float32))
    return jnp.einsum("qc,qhw->chw", classes, masks)


def semantic_class_map(
    mask_logits: jax.Array,
    class_logits: jax.Array,
    *,
    output_size: tuple[int, int] | None = None,
) -> jax.Array:
    """Return the highest-scoring semantic class at each pixel."""
    return jnp.argmax(
        semantic_scores(mask_logits, class_logits, output_size=output_size), axis=0
    )


def merge_semantic_crops(
    crop_scores: jax.Array,
    origins: Sequence[tuple[int, int]],
    scaled_size: tuple[int, int],
    output_size: tuple[int, int],
) -> jax.Array:
    """Average overlapping crop scores, then restore original image size."""
    if crop_scores.ndim != 4 or crop_scores.shape[0] != len(origins):
        raise ValueError("Crop scores must match the supplied origins.")
    _, classes, crop_h, crop_w = crop_scores.shape
    scaled_h, scaled_w = scaled_size
    if min(scaled_h, scaled_w, *output_size) <= 0:
        raise ValueError("Semantic image sizes must be positive.")
    sums = jnp.zeros((classes, scaled_h, scaled_w), crop_scores.dtype)
    counts = jnp.zeros((1, scaled_h, scaled_w), crop_scores.dtype)
    for crop, (top, left) in zip(crop_scores, origins, strict=True):
        if top < 0 or left < 0 or top + crop_h > scaled_h or left + crop_w > scaled_w:
            raise ValueError("A semantic crop exceeds the scaled image.")
        sums = sums.at[:, top : top + crop_h, left : left + crop_w].add(crop)
        counts = counts.at[:, top : top + crop_h, left : left + crop_w].add(1)
    if len(origins) == 0:
        raise ValueError("At least one semantic crop is required.")
    merged = sums / counts
    return jax.image.resize(merged, (classes, *output_size), "linear", antialias=False)


def restore_panoptic_mask_logits(
    mask_logits: jax.Array,
    *,
    padded_size: tuple[int, int],
    resized_size: tuple[int, int],
    output_size: tuple[int, int],
) -> jax.Array:
    """Upsample, remove right/bottom padding, then restore original size."""
    if (
        any(value <= 0 for value in (*padded_size, *resized_size, *output_size))
        or resized_size[0] > padded_size[0]
        or resized_size[1] > padded_size[1]
    ):
        raise ValueError("Invalid panoptic resize and padding geometry.")
    padded = resize_mask_logits(mask_logits, padded_size)
    crop = padded[:, : resized_size[0], : resized_size[1]]
    return resize_mask_logits(crop, output_size)


def panoptic_predictions(
    mask_logits: jax.Array,
    class_logits: jax.Array,
    *,
    stuff_classes: Sequence[int],
    mask_threshold: float = 0.8,
    overlap_threshold: float = 0.8,
) -> PanopticPrediction:
    """Assign per-query masks to thing instances and merged stuff regions."""
    _validate_logits(mask_logits, class_logits)
    num_classes = class_logits.shape[-1] - 1
    if len(set(stuff_classes)) != len(stuff_classes) or any(
        not 0 <= cls < num_classes for cls in stuff_classes
    ):
        raise ValueError("Stuff classes must be unique valid class indices.")
    if not 0 <= mask_threshold <= 1 or not 0 <= overlap_threshold <= 1:
        raise ValueError("Panoptic thresholds must lie in [0, 1].")

    probabilities = jax.nn.softmax(class_logits.astype(jnp.float32), axis=-1)
    scores = jnp.max(probabilities, axis=-1)
    classes = jnp.argmax(probabilities, axis=-1).astype(jnp.int32)
    kept = (classes != num_classes) & (scores > mask_threshold)
    masks = jax.nn.sigmoid(mask_logits.astype(jnp.float32))
    weighted = jnp.where(kept[:, None, None], scores[:, None, None] * masks, -jnp.inf)
    winners = jnp.argmax(weighted, axis=0)
    height, width = mask_logits.shape[-2:]
    class_map = jnp.full((height, width), num_classes, dtype=jnp.int32)
    segment_map = jnp.full((height, width), -1, dtype=jnp.int32)
    stuff_ids = jnp.full((num_classes,), -1, dtype=jnp.int32)
    is_stuff = jnp.zeros((num_classes,), dtype=jnp.bool_)
    if stuff_classes:
        is_stuff = is_stuff.at[jnp.asarray(stuff_classes)].set(True)

    def assign(query: int, carry):
        class_map, segment_map, stuff_ids, next_id = carry
        original = masks[query] >= 0.5
        selected = winners == query
        area_original = jnp.sum(original)
        area_selected = jnp.sum(selected)
        final_mask = original & selected
        area_final = jnp.sum(final_mask)
        valid = (
            kept[query]
            & (area_original > 0)
            & (area_selected > 0)
            & (area_final > 0)
            & (area_selected / jnp.maximum(area_original, 1) >= overlap_threshold)
        )
        class_id = jnp.minimum(classes[query], num_classes - 1)
        existing = stuff_ids[class_id]
        reuse = is_stuff[class_id] & (existing >= 0)
        assigned_id = jnp.where(reuse, existing, next_id)
        draw = final_mask & valid
        class_map = jnp.where(draw, classes[query], class_map)
        segment_map = jnp.where(draw, assigned_id, segment_map)
        new_stuff = valid & is_stuff[class_id] & ~reuse
        stuff_ids = stuff_ids.at[class_id].set(
            jnp.where(new_stuff, next_id, stuff_ids[class_id])
        )
        next_id += jnp.asarray(valid & ~reuse, dtype=jnp.int32)
        return class_map, segment_map, stuff_ids, next_id

    class_map, segment_map, _, _ = lax.fori_loop(
        0,
        mask_logits.shape[0],
        assign,
        (class_map, segment_map, stuff_ids, jnp.asarray(0, dtype=jnp.int32)),
    )
    return PanopticPrediction(class_map, segment_map)


def _validate_logits(mask_logits: jax.Array, class_logits: jax.Array) -> None:
    if (
        mask_logits.ndim != 3
        or class_logits.ndim != 2
        or mask_logits.shape[0] != class_logits.shape[0]
        or mask_logits.shape[0] == 0
        or class_logits.shape[-1] < 2
    ):
        raise ValueError("Expected matching nonempty (Q,H,W) and (Q,K+1) logits.")


__all__ = [
    "PanopticPrediction",
    "merge_semantic_crops",
    "panoptic_predictions",
    "resize_mask_logits",
    "restore_panoptic_mask_logits",
    "semantic_class_map",
    "semantic_scores",
]
