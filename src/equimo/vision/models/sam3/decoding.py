"""Fixed-shape decoding of SAM 3.1 detector outputs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from equimo.vision.models.sam3.detector import DetectorOutput
from equimo.vision.ops import resize_bilinear
from equimo.vision.segmentation import (
    boxes_from_masks,
    non_greedy_iom_nms,
    remove_overlaps,
)

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


def decode_concepts(
    out: DetectorOutput,
    *,
    output_size: tuple[int, int],
    max_detections: int | None = None,
    threshold: float = 0.5,
) -> ConceptDetections:
    """Turn detector outputs into the detections of one image.

    Probabilities come from the joint logit, which already includes presence,
    so presence is not applied a second time. The kept entries are gathered at
    the detector's resolution before upsampling, so memory scales with
    ``max_detections``, which defaults to every query.
    """
    if max_detections is None:
        max_detections = out.logits.shape[0]
    probabilities = jax.nn.sigmoid(out.logits.astype(jnp.float32))
    order = jnp.argsort(-probabilities, stable=True)
    ranked_probabilities = probabilities[order]
    keep = ranked_probabilities >= threshold
    keep = keep & non_greedy_iom_nms(
        out.mask_logits[order] > 0, _SINGLE_FRAME_NMS_THRESHOLD
    )

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
