"""Model-aware transformer block depths for fine-tuning plans."""

from __future__ import annotations

import jax.tree_util as jtu

from ._typing import Path, PyTree
from .paths import is_path_prefix, key_path_to_path


def vision_block_depths(model: PyTree) -> dict[Path, int]:
    """Map concrete ViT block paths to their execution-order depths."""

    from equimo.vision.models.vit import VisionTransformer

    depths: dict[Path, int] = {}
    for key_path, node in jtu.tree_leaves_with_path(
        model, is_leaf=lambda value: isinstance(value, VisionTransformer)
    ):
        if not isinstance(node, VisionTransformer):
            continue
        prefix = (*key_path_to_path(key_path), "blocks")
        depth = 0
        for chunk_index, chunk in enumerate(node.blocks):
            for block_index, _ in enumerate(chunk.blocks or ()):
                depths[(*prefix, chunk_index, "blocks", block_index)] = depth
                depth += 1
    return depths


def model_block_depth(path: Path, depths: dict[Path, int]) -> int | None:
    """Find the concrete block containing a parameter path."""

    matches = (
        (len(prefix), depth)
        for prefix, depth in depths.items()
        if is_path_prefix(prefix, path)
    )
    return max(matches, default=(0, None))[1]
