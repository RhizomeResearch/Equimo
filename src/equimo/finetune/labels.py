"""Parameter labels and learning-rate metadata for fine-tuning plans."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from ._blocks import model_block_depth, vision_block_depths
from .config import GroupSpec, LLRDConfig, ParamInfo
from .paths import iter_param_leaves, key_path_to_path
from .tags import (
    Tagger,
    _model_param_info,
    canonical_tags_for_path,
    infer_block_depth,
)


def make_param_labels(
    model: PyTree,
    llrd_config: LLRDConfig | None = None,
    *,
    trainable_paths: frozenset[Path] | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Return a PyTree of optimizer-group labels for parameter leaves."""

    info_tree, _ = make_labeled_param_info_tree(
        model,
        llrd_config,
        trainable_paths=trainable_paths,
        tagger=tagger,
    )
    return jtu.tree_map(_label_from_info, info_tree)


def make_labeled_param_info_tree(
    model: PyTree,
    llrd_config: LLRDConfig | None = None,
    *,
    trainable_paths: frozenset[Path] | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[PyTree, dict[str, GroupSpec]]:
    """Build a ``ParamInfo`` tree with trainability, labels, and group specs."""

    config = LLRDConfig(decay=1.0) if llrd_config is None else llrd_config
    block_depths = vision_block_depths(model)
    param_info = _model_param_info(model, tagger)
    all_depths = _depths(model, config, block_depths)
    selected_depths = _depths(model, config, block_depths, trainable_paths)
    filtered = eqx.filter(model, eqx.is_inexact_array)
    group_specs: dict[str, GroupSpec] = {}

    def make_info(key_path: tuple[Any, ...], leaf: Any) -> ParamInfo | None:
        if not eqx.is_inexact_array(leaf):
            return None

        path = key_path_to_path(key_path)
        base = param_info(path, leaf)
        base = replace(base, depth=_depth_for_path(path, config, block_depths))
        trainable = trainable_paths is None or path in trainable_paths
        weight_decay = trainable and _uses_weight_decay(base, config)
        lr_multiplier = (
            _lr_multiplier(base, config, all_depths, selected_depths)
            if trainable
            else None
        )
        label = _label_for_info(base, config, weight_decay) if trainable else None

        info = replace(
            base,
            trainable=trainable,
            weight_decay=weight_decay,
            lr_multiplier=lr_multiplier,
            label=label,
        )
        if trainable and label is not None and lr_multiplier is not None:
            if label in group_specs:
                group_specs[label] = _merge_group_spec(group_specs[label], info)
            else:
                group_specs[label] = GroupSpec(
                    label=label,
                    role=info.role,
                    depth=info.depth,
                    lr_multiplier=lr_multiplier,
                    weight_decay=weight_decay,
                    tags=tuple(sorted(info.tags)),
                    tags_all=tuple(sorted(info.tags)),
                    roles=(info.role,) if info.role else (),
                )
        return info

    return jtu.tree_map_with_path(make_info, filtered), group_specs


def _merge_group_spec(group: GroupSpec, info: ParamInfo) -> GroupSpec:
    if (
        group.lr_multiplier != info.lr_multiplier
        or group.weight_decay != info.weight_decay
    ):
        raise ValueError(
            f"Optimizer label {group.label!r} has inconsistent learning rate "
            f"or weight decay at {info.logical_id!r}."
        )
    tags_all = tuple(sorted(set(group.tags_all).union(info.tags)))
    roles = set(group.roles)
    if info.role:
        roles.add(info.role)
    sorted_roles = tuple(sorted(roles))
    return replace(
        group,
        tags_all=tags_all,
        roles=sorted_roles,
        mixed_roles=len(sorted_roles) > 1,
    )


def _label_from_info(info: ParamInfo | None) -> str | None:
    return info.label if isinstance(info, ParamInfo) else None


def _uses_weight_decay(info: ParamInfo, config: LLRDConfig) -> bool:
    return info.tags.isdisjoint(config.no_weight_decay_tags)


def _label_for_info(
    info: ParamInfo,
    config: LLRDConfig,
    weight_decay: bool,
) -> str:
    suffix = "decay" if weight_decay else "no_decay"
    return f"{_label_base(info, config)}_{suffix}"


def _label_base(info: ParamInfo, config: LLRDConfig) -> str:
    if "lora.factor_A" in info.tags:
        return "lora_A"
    if "lora.factor_B" in info.tags:
        return "lora_B"
    if "lora_fa.factor_B" in info.tags:
        return "lora_fa_B"
    if "adalora.P" in info.tags:
        return "adalora_P"
    if "adalora.singular" in info.tags:
        return "adalora_singular"
    if "adalora.Q" in info.tags:
        return "adalora_Q"
    if "fourierft" in info.tags:
        return "fourierft"
    if "orthogonal" in info.tags:
        return "orthogonal"
    if "randlora.scale" in info.tags:
        return "randlora_scale"
    if "randlora" in info.tags:
        return "randlora"
    if "lora" in info.tags:
        return "lora"
    if "dora.magnitude" in info.tags:
        return "dora_magnitude"
    if "dora" in info.tags:
        return "dora"
    if "adapter" in info.tags:
        return "adapter"
    if "prompt" in info.tags:
        return "prompt"
    if "prefix" in info.tags:
        return "prefix"
    if "scale_shift" in info.tags:
        return "scale_shift"
    if "ia3" in info.tags:
        return "ia3"
    if "vera" in info.tags:
        return "vera"
    if "head" in info.tags:
        return config.head_label
    if info.depth is not None:
        return config.block_label_format.format(depth=info.depth)
    if any(tag.startswith("embedding.") for tag in info.tags):
        return config.embedding_label
    if "norm" in info.tags:
        return "norm"
    if "bias" in info.tags:
        return "bias"
    return "param"


def _lr_multiplier(
    info: ParamInfo,
    config: LLRDConfig,
    all_depths: tuple[int, ...],
    selected_depths: tuple[int, ...],
) -> float:
    if "head" in info.tags:
        return float(config.head_lr_multiplier)

    max_depth = None
    if config.rebase_selected_depth and selected_depths:
        max_depth = max(selected_depths)
    elif all_depths:
        max_depth = max(all_depths)

    if info.depth is not None and max_depth is not None:
        return float(
            config.top_block_lr_multiplier * config.decay ** (max_depth - info.depth)
        )

    if any(tag.startswith("embedding.") for tag in info.tags):
        if config.embedding_lr_multiplier is not None:
            return float(config.embedding_lr_multiplier)
        if max_depth is not None:
            return float(
                config.top_block_lr_multiplier * config.decay ** (max_depth + 1)
            )

    return 1.0


def _depths(
    model: PyTree,
    config: LLRDConfig,
    block_depths: dict[Path, int],
    paths: frozenset[Path] | None = None,
) -> tuple[int, ...]:
    """Sorted depths of parameter leaves, optionally restricted to *paths*."""

    depths: set[int] = set()
    for path, _ in iter_param_leaves(model):
        if paths is not None and path not in paths:
            continue
        depth = _depth_for_path(path, config, block_depths)
        if depth is not None:
            depths.add(depth)
    return tuple(sorted(depths))


def _depth_for_path(
    path: Path, config: LLRDConfig, block_depths: dict[Path, int]
) -> int | None:
    parts = tuple(str(part) for part in path)
    if config.depth_axis == "block":
        actual = model_block_depth(path, block_depths)
        if actual is not None:
            return actual
        return infer_block_depth(path)
    if config.depth_axis == "stage":
        return _indexed_depth(parts, {"stages", "stage"})
    if config.depth_axis == "module":
        return _first_numeric_part(parts)
    raise ValueError("LLRDConfig.depth_axis must be 'block', 'stage', or 'module'.")


def _indexed_depth(parts: tuple[str, ...], names: set[str]) -> int | None:
    for index, part in enumerate(parts[:-1]):
        if part not in names:
            continue
        depth = _maybe_int(parts[index + 1])
        if depth is not None:
            return depth
    return None


def _first_numeric_part(parts: tuple[str, ...]) -> int | None:
    for part in parts:
        value = _maybe_int(part)
        if value is not None:
            return value
    return None


def _maybe_int(value: str) -> int | None:
    return int(value) if value.isdecimal() else None


__all__ = (
    "make_labeled_param_info_tree",
    "make_param_labels",
)
