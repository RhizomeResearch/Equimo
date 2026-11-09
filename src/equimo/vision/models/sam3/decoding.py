"""Fixed-shape decoding of SAM 3.1 detector outputs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from equimo.vision.models.sam3.detector import DetectorOutput
from equimo.vision.ops import resize_bilinear

_SINGLE_FRAME_NMS_THRESHOLD = 0.1


class ConceptDetections(eqx.Module):
    """Detections of one image, gathered to a fixed number of entries.

    ``keep`` marks the entries that hold a detection; entries are ordered by
    decreasing score, and the rest are zeros.
    """

    keep: Bool[Array, " max_detections"]
    masks: Bool[Array, "max_detections height width"]
    boxes: Float[Array, "max_detections 4"]
    scores: Float[Array, " max_detections"]
    query_index: jax.Array
    presence: Float[Array, ""]


def _intersections(
    masks_a: Bool[Array, "a height width"], masks_b: Bool[Array, "b height width"]
) -> Float[Array, "a b"]:
    a = masks_a.reshape(masks_a.shape[0], -1).astype(jnp.float32)
    b = masks_b.reshape(masks_b.shape[0], -1).astype(jnp.float32)
    return a @ b.T


def _areas(masks: Bool[Array, "n height width"]) -> Float[Array, " n"]:
    return masks.reshape(masks.shape[0], -1).astype(jnp.float32).sum(axis=-1)


def iom_matrix(
    masks_a: Bool[Array, "a height width"], masks_b: Bool[Array, "b height width"]
) -> Float[Array, "a b"]:
    """Intersection over the smaller mask's area, 0 when either mask is empty."""
    smaller = jnp.minimum(_areas(masks_a)[:, None], _areas(masks_b)[None, :])
    return jnp.where(
        smaller > 0, _intersections(masks_a, masks_b) / jnp.maximum(smaller, 1.0), 0.0
    )


def iou_matrix(
    masks_a: Bool[Array, "a height width"], masks_b: Bool[Array, "b height width"]
) -> Float[Array, "a b"]:
    """Intersection over union, 0 when both masks are empty."""
    intersection = _intersections(masks_a, masks_b)
    union = _areas(masks_a)[:, None] + _areas(masks_b)[None, :] - intersection
    return jnp.where(union > 0, intersection / jnp.maximum(union, 1.0), 0.0)


def single_frame_nms(
    masks: Bool[Array, "n height width"],
    threshold: float = _SINGLE_FRAME_NMS_THRESHOLD,
) -> Bool[Array, " n"]:
    """Keep mask of masks already sorted by decreasing score.

    Overlap is the intersection over the smaller of the two masks, and a
    suppressed mask still suppresses later ones.
    """
    return ~jnp.any(jnp.triu(iom_matrix(masks, masks) > threshold, k=1), axis=0)


def greedy_iom_nms(
    masks: Bool[Array, "n height width"],
    scores: Float[Array, " n"],
    candidates: Bool[Array, " n"],
    threshold: float,
) -> Bool[Array, " n"]:
    """Keep mask of a greedy NMS by intersection over minimum.

    Unlike :func:`single_frame_nms`, only masks that are kept suppress later
    ones, and masks outside ``candidates`` take no part. Scores are ranked with
    a stable sort.
    """
    count = scores.shape[0]
    order = jnp.argsort(-scores.astype(jnp.float32), stable=True)
    overlaps = (iom_matrix(masks, masks) > threshold)[order][:, order]
    ranked_candidates = candidates[order]

    def visit(kept: jax.Array, rank: jax.Array) -> tuple[jax.Array, None]:
        clear = ~jnp.any(kept & overlaps[rank])
        return kept.at[rank].set(ranked_candidates[rank] & clear), None

    kept, _ = jax.lax.scan(visit, jnp.zeros(count, bool), jnp.arange(count))
    return jnp.zeros(count, bool).at[order].set(kept)


def boxes_from_masks(
    masks: Bool[Array, "n height width"],
) -> Float[Array, "n 4"]:
    """``(x_min, y_min, x_max, y_max)`` of each mask's extent, zeros when empty."""
    height, width = masks.shape[-2:]
    columns = jnp.any(masks, axis=-2)
    rows = jnp.any(masks, axis=-1)
    x_min = jnp.argmax(columns, axis=-1)
    x_max = width - 1 - jnp.argmax(columns[..., ::-1], axis=-1)
    y_min = jnp.argmax(rows, axis=-1)
    y_max = height - 1 - jnp.argmax(rows[..., ::-1], axis=-1)
    boxes = jnp.stack((x_min, y_min, x_max, y_max), axis=-1).astype(jnp.float32)
    return jnp.where(jnp.any(columns, axis=-1)[..., None], boxes, 0.0)


def remove_overlaps(
    masks: Bool[Array, "n height width"],
) -> Bool[Array, "n height width"]:
    """Give each pixel to the first mask covering it.

    Masks must be sorted by decreasing priority.
    """
    winner = jnp.argmax(masks, axis=0)
    indices = jnp.arange(masks.shape[0])[:, None, None]
    return masks & (winner == indices)


def decode_concepts(
    out: DetectorOutput,
    *,
    output_size: tuple[int, int],
    max_detections: int,
    threshold: float = 0.5,
) -> ConceptDetections:
    """Turn detector outputs into the detections of one image.

    Probabilities come from the joint logit, which already includes presence,
    so presence is not applied a second time. The kept entries are gathered at
    the detector's resolution before upsampling, so memory scales with
    ``max_detections``.
    """
    probabilities = jax.nn.sigmoid(out.logits.astype(jnp.float32))
    order = jnp.argsort(-probabilities, stable=True)
    ranked_probabilities = probabilities[order]
    keep = ranked_probabilities >= threshold
    keep = keep & single_frame_nms(out.mask_logits[order] > 0)

    gathered = jnp.argsort(~keep, stable=True)[:max_detections]
    kept = keep[gathered]
    query_index = order[gathered]

    logits = out.mask_logits[query_index].astype(jnp.float32)
    logits = resize_bilinear(logits, output_size)
    masks = (logits > 0) & kept[:, None, None]
    boxes = boxes_from_masks(masks)

    return ConceptDetections(
        keep=kept,
        masks=remove_overlaps(masks),
        boxes=boxes,
        scores=jnp.where(kept, probabilities[query_index], 0.0),
        query_index=query_index,
        presence=jax.nn.sigmoid(out.presence_logit.astype(jnp.float32)),
    )


__all__ = ["ConceptDetections", "decode_concepts"]
