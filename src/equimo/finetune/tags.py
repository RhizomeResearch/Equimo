"""Semantic tags for fine-tuning selector infrastructure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from equimo.core import parameters
from equimo.core.tags import (
    CANONICAL_TAGS,
    Tagger,
    canonical_tags_for_path,
    infer_depth,
)

from ._typing import Path, PyTree
from .config import ParamInfo
from .paths import iter_param_leaves, key_path_to_path, path_to_str
from ._blocks import model_block_depth, vision_block_depths


def make_tag_tree(
    tree: PyTree,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Replace parameter-like leaves with tagged ``ParamInfo`` records."""

    param_info = _model_param_info(tree, tagger)
    return jtu.tree_map_with_path(
        lambda key_path, leaf: param_info(key_path_to_path(key_path), leaf),
        eqx.filter(tree, eqx.is_inexact_array),
    )


def iter_param_infos(
    tree: PyTree,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[ParamInfo, ...]:
    """Return tagged ``ParamInfo`` records for inexact array leaves."""

    param_info = _model_param_info(tree, tagger)
    return tuple(param_info(path, leaf) for path, leaf in iter_param_leaves(tree))


def make_param_info(
    path: Path,
    leaf: Any,
    *,
    tagger: Tagger = canonical_tags_for_path,
    block_depths: dict[Path, int] | None = None,
    adalora_roles: dict[Path, str] | None = None,
    module_tags: dict[Path, frozenset[str]] | None = None,
) -> ParamInfo:
    """Build one tagged ``ParamInfo`` record.

    *module_tags* holds the tags that the modules of the model give each path
    (see :func:`equimo.core.parameters.module_tags`), such as ``buffer``, which
    every trainability mode excludes.
    """

    tags = set(tagger(path, leaf))
    if adalora_roles is not None and path in adalora_roles:
        tags.update(("peft", "adalora", f"adalora.{adalora_roles[path]}"))
    if module_tags is not None:
        tags.update(module_tags.get(path, ()))
    actual_depth = (
        model_block_depth(path, block_depths) if block_depths is not None else None
    )
    if actual_depth is not None:
        tags.difference_update(
            {tag for tag in tags if tag.startswith("block.") and tag[6:].isdecimal()}
        )
        tags.update(("block", f"block.{actual_depth}"))
    path_string = path_to_str(path)
    return ParamInfo(
        path=path,
        logical_id=path_string,
        tags=frozenset(tags),
        role=infer_role(tags),
        depth=actual_depth if actual_depth is not None else infer_depth(path),
        is_array=eqx.is_array(leaf),
        is_inexact_array=eqx.is_inexact_array(leaf),
    )


def _model_param_info(tree: PyTree, tagger: Tagger) -> Callable[[Path, Any], ParamInfo]:
    """Bind ``make_param_info`` to the block depths, AdaLoRA roles, and modules."""

    return partial(
        make_param_info,
        tagger=tagger,
        block_depths=vision_block_depths(tree),
        adalora_roles=_adalora_roles(tree),
        module_tags=_module_tags_by_path(tree),
    )


def _adalora_roles(tree: PyTree) -> dict[Path, str]:
    from .peft.lora import iter_adalora_modules

    return {
        (*path, leaf): leaf
        for path, _ in iter_adalora_modules(tree)
        for leaf in ("P", "singular", "Q")
    }


def _module_tags_by_path(tree: PyTree) -> dict[Path, frozenset[str]]:
    return {
        key_path_to_path(key_path): tags
        for key_path, tags in jtu.tree_leaves_with_path(parameters.module_tags(tree))
        if tags
    }


def infer_role(tags: Iterable[str]) -> str:
    """Return a coarse role name from semantic tags."""

    tag_set = frozenset(tags)
    for tag, role in (
        ("adalora.P", "adalora.P"),
        ("adalora.singular", "adalora.singular"),
        ("adalora.Q", "adalora.Q"),
        ("lora.factor_A", "lora.factor_A"),
        ("lora.factor_B", "lora.factor_B"),
        ("lora_fa.factor_B", "lora_fa.factor_B"),
        ("adalora.singular", "adalora.singular"),
        ("fourierft", "fourierft"),
        ("orthogonal", "orthogonal"),
        ("randlora.scale", "randlora.scale"),
        ("randlora", "randlora"),
        ("lora", "lora"),
        ("head", "head"),
        ("attention.qkv", "attention.qkv"),
        ("attention.proj", "attention.proj"),
        ("mlp.hidden", "mlp.hidden"),
        ("mlp.fc1", "mlp.fc1"),
        ("mlp.fc2", "mlp.fc2"),
        ("norm", "norm"),
        ("embedding.patch", "embedding.patch"),
        ("embedding.position", "embedding.position"),
        ("embedding.class_token", "embedding.class_token"),
        ("embedding.register_token", "embedding.register_token"),
        ("embedding.distillation_token", "embedding.distillation_token"),
        ("embedding.mask_token", "embedding.mask_token"),
        ("embedding.token", "embedding.token"),
        ("block", "block"),
    ):
        if tag in tag_set:
            return role
    return ""


def merge_taggers(*taggers: Tagger) -> Tagger:
    """Combine multiple taggers into one tagger."""

    def merged(path: Path, leaf: Any) -> frozenset[str]:
        tags: set[str] = set()
        for tagger in taggers:
            tags.update(tagger(path, leaf))
        return frozenset(tags)

    return merged


__all__ = (
    "CANONICAL_TAGS",
    "Tagger",
    "canonical_tags_for_path",
    "infer_depth",
    "infer_role",
    "iter_param_infos",
    "make_param_info",
    "make_tag_tree",
    "merge_taggers",
)
