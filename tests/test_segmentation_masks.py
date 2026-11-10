"""Tests for the mask operations in equimo.vision.segmentation."""

import jax.numpy as jnp
import numpy as np

from equimo.vision.segmentation import (
    boxes_from_masks,
    greedy_iom_nms,
    iom_matrix,
    iou_matrix,
    non_greedy_iom_nms,
    remove_overlaps,
)


def _row_masks(*spans: tuple[int, int]) -> jnp.ndarray:
    masks = np.zeros((len(spans), 1, 8), dtype=bool)
    for index, (start, stop) in enumerate(spans):
        masks[index, 0, start:stop] = True
    return jnp.asarray(masks)


def test_non_greedy_nms_lets_suppressed_masks_suppress_later_ones():
    masks = _row_masks((0, 4), (3, 7), (6, 8))
    keep = non_greedy_iom_nms(masks, 0.1)
    assert keep.tolist() == [True, False, False]


def test_greedy_nms_only_lets_kept_masks_suppress():
    masks = _row_masks((0, 4), (3, 7), (6, 8))
    scores = jnp.asarray([0.9, 0.8, 0.7])
    everything = jnp.ones(3, bool)
    assert greedy_iom_nms(masks, scores, everything, 0.1).tolist() == [
        True,
        False,
        True,
    ]
    skip_first = jnp.asarray([False, True, True])
    assert greedy_iom_nms(masks, scores, skip_first, 0.1).tolist() == [
        False,
        True,
        False,
    ]


def test_greedy_nms_breaks_score_ties_by_index():
    masks = _row_masks((0, 4), (0, 4))
    keep = greedy_iom_nms(masks, jnp.asarray([0.5, 0.5]), jnp.ones(2, bool), 0.1)
    assert keep.tolist() == [True, False]


def test_overlap_matrices_divide_by_the_smaller_mask_and_the_union():
    a = _row_masks((0, 4))
    b = _row_masks((2, 4), (6, 8))
    np.testing.assert_allclose(iom_matrix(a, b), [[1.0, 0.0]])
    np.testing.assert_allclose(iou_matrix(a, b), [[0.5, 0.0]])


def test_overlap_matrices_are_zero_for_empty_masks():
    empty = _row_masks((0, 0))
    full = _row_masks((0, 8))
    assert iom_matrix(empty, full).item() == 0.0
    assert iou_matrix(empty, empty).item() == 0.0


def test_boxes_from_masks_gives_the_extent_and_zeros_when_empty():
    masks = np.zeros((2, 4, 6), dtype=bool)
    masks[0, 1:3, 2:5] = True
    np.testing.assert_array_equal(
        boxes_from_masks(jnp.asarray(masks)), [[2, 1, 4, 2], [0, 0, 0, 0]]
    )


def test_remove_overlaps_gives_each_pixel_to_the_first_mask():
    masks = _row_masks((0, 4), (2, 6))
    resolved = remove_overlaps(masks)
    np.testing.assert_array_equal(resolved, _row_masks((0, 4), (4, 6)))
