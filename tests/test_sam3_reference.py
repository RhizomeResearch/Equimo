"""SAM 3.1 full-weight checks on one real image, with decoded detections."""

import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cases.reference_cases import local_archive
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256
from equimo.language import sam31_text
from equimo.serialization import load_weights
from equimo.vision.models.sam3 import ConceptPrompt, decode_concepts, sam31_multiplex

DATA_DIR = Path(__file__).parent / "data"
FIXTURE = DATA_DIR / "sam31_image_reference.npz"
MAX_DETECTIONS = 64
IMAGE_MASK_IOU = 0.99
BFLOAT16_MASK_IOU = 0.97
BFLOAT16_SCORE_DIFFERENCE = 0.02
BFLOAT16_DETECTION_THRESHOLD = 0.5

pytestmark = pytest.mark.reference_parity


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {name: fixture[name] for name in fixture.files}


def _require_archive(identifier: str) -> Path:
    archive = local_archive(identifier)
    if archive.is_file():
        return archive
    message = f"Converted archive {archive} is not cached locally."
    # An archive without a pinned digest can only come from a local conversion.
    if (
        os.environ.get("EQUIMO_REQUIRE_REFERENCE_CACHE") == "1"
        and identifier in PRETRAINED_ARCHIVE_SHA256
    ):
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def models():
    vision_archive = _require_archive("sam31_multiplex")
    text_archive = _require_archive("sam31_text")
    return (
        load_weights(sam31_multiplex(), path=vision_archive),
        load_weights(sam31_text(), path=text_archive),
    )


def _normalize(reference):
    pixels = jnp.asarray(reference["inputs.image"], jnp.float32) / 255.0
    return jnp.moveaxis((pixels - 0.5) / 0.5, -1, 0)


@eqx.filter_jit
def _detect(vision, text, image, token_ids, padding_mask):
    features = vision.encode_image(image)
    text_features = text.features(token_ids, padding_mask)
    dim = features.detector[2].shape[0]
    prompt = ConceptPrompt.create(
        text_features=text_features, text_padding_mask=padding_mask, dim=dim
    )
    return vision.detect(features, prompt)


def _to_bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def _source_size(reference) -> tuple[int, int]:
    return tuple(reference["emitted.000.masks"].shape[1:])


def _decode(reference, models, threshold: float, dtype=jnp.float32):
    vision, text = models
    output = _detect(
        vision,
        text,
        _normalize(reference).astype(dtype),
        jnp.asarray(reference["inputs.text.token_ids"][0], jnp.int32),
        jnp.asarray(reference["inputs.text.padding_mask"][0]),
    )
    return decode_concepts(
        output,
        output_size=_source_size(reference),
        max_detections=MAX_DETECTIONS,
        threshold=threshold,
    )


def _iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first, second = first.reshape(len(first), -1), second.reshape(len(second), -1)
    intersection = (first[:, None] & second[None]).sum(-1)
    union = (first[:, None] | second[None]).sum(-1)
    return intersection / np.maximum(union, 1)


def _assert_one_to_one(iou: np.ndarray, threshold: float) -> None:
    assert iou.shape[0] == iou.shape[1]
    best = iou.argmax(axis=1)
    assert np.unique(best).size == best.size, f"not one-to-one: {best.tolist()}"
    worst = float(iou[np.arange(iou.shape[0]), best].min())
    assert worst >= threshold, f"lowest mask IoU {worst:.4f} is below {threshold}"


@pytest.fixture(scope="module")
def float32_detections(reference, models):
    return _decode(reference, models, threshold=BFLOAT16_DETECTION_THRESHOLD)


def test_image_decoding_keeps_the_reference_detections(reference, float32_detections):
    kept = np.asarray(float32_detections.keep)
    masks = np.asarray(float32_detections.masks)[kept]

    _assert_one_to_one(
        _iou_matrix(masks, reference["emitted.000.masks"]), IMAGE_MASK_IOU
    )


def test_bfloat16_detections_stay_close_to_float32(
    reference, models, float32_detections
):
    vision, text = models
    bfloat16_models = (_to_bfloat16(vision), _to_bfloat16(text))
    detections = _decode(
        reference,
        bfloat16_models,
        threshold=BFLOAT16_DETECTION_THRESHOLD,
        dtype=jnp.bfloat16,
    )

    kept = np.asarray(float32_detections.keep)
    kept_bfloat16 = np.asarray(detections.keep)
    masks = np.asarray(float32_detections.masks)[kept]
    masks_bfloat16 = np.asarray(detections.masks)[kept_bfloat16]
    iou = _iou_matrix(masks, masks_bfloat16)
    _assert_one_to_one(iou, BFLOAT16_MASK_IOU)

    matched = iou.argmax(axis=1)
    differences = np.abs(
        np.asarray(float32_detections.scores)[kept]
        - np.asarray(detections.scores, np.float32)[kept_bfloat16][matched]
    )
    assert differences.max() <= BFLOAT16_SCORE_DIFFERENCE, json.dumps(
        differences.tolist()
    )
