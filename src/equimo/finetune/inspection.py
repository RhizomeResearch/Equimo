"""Inspection helpers for fine-tuning plans."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import jax.tree_util as jtu

from ._typing import PyTree
from .config import (
    FineTunePlan,
    ParameterReport,
    ParamInfo,
    TrainableReport,
    TrainableSpec,
)
from .paths import iter_param_leaves, key_path_to_path, path_to_str


def inspect_trainables(
    model: PyTree | FineTunePlan,
    trainable: TrainableSpec | None = None,
) -> TrainableReport:
    """Return a trainability report for a model or existing plan."""

    if isinstance(model, FineTunePlan):
        return model.report

    from .surgery import prepare_finetune

    spec = TrainableSpec(mode="full") if trainable is None else trainable
    return prepare_finetune(model, trainable=spec).report


def inspect_plan(plan: FineTunePlan) -> TrainableReport:
    """Return the report stored on a prepared fine-tuning plan."""

    return plan.report


def make_trainable_report(model: PyTree, param_info: PyTree) -> TrainableReport:
    """Summarize trainable/frozen parameters from a ``ParamInfo`` tree."""

    leaves_by_path = dict(iter_param_leaves(model))
    infos = _param_info_leaves(param_info)
    if set(leaves_by_path) != {info.path for info in infos}:
        raise ValueError("Parameter metadata does not match model floating leaves.")

    total_params = 0
    trainable_params = 0
    adapter_params = 0
    head_params = 0
    trainable_by_label: defaultdict[str, int] = defaultdict(int)
    frozen_by_label: defaultdict[str, int] = defaultdict(int)
    target_paths: list[str] = []
    estimated_delta_size_bytes = 0
    trainable_infos: list[ParamInfo] = []
    parameters: list[ParameterReport] = []
    logical_ids: set[str] = set()

    for info in infos:
        if info.logical_id in logical_ids:
            raise ValueError(f"Duplicate logical parameter ID {info.logical_id!r}.")
        logical_ids.add(info.logical_id)
        leaf = leaves_by_path[info.path]
        count = int(leaf.size)
        total_params += count
        parameters.append(
            ParameterReport(
                logical_id=info.logical_id,
                physical_path=info.path,
                shape=tuple(int(size) for size in leaf.shape),
                dtype=str(leaf.dtype),
                role=info.role,
                tags=tuple(sorted(info.tags)),
                depth=info.depth,
                trainable=info.trainable,
                label=info.label,
                lr_multiplier=info.lr_multiplier,
                weight_decay=info.weight_decay,
            )
        )

        if info.trainable:
            trainable_infos.append(info)
            trainable_params += count
            trainable_by_label[info.label or "trainable"] += count
            target_paths.append(path_to_str(info.path))
            estimated_delta_size_bytes += count * int(leaf.dtype.itemsize)
            if "head" in info.tags:
                head_params += count
            if _is_adapter_leaf(info):
                adapter_params += count
        else:
            frozen_by_label[_frozen_label(info)] += count

    trainable_fraction = trainable_params / total_params if total_params else 0.0
    signature = _hash_json(
        [
            (row.logical_id, row.shape, row.dtype)
            for row in sorted(parameters, key=lambda row: row.logical_id)
        ]
    )
    fingerprint = _hash_json(
        {
            "schema_version": 1,
            "model_signature": signature,
            "parameters": [
                asdict(row)
                for row in sorted(parameters, key=lambda row: row.logical_id)
            ],
        }
    )

    return TrainableReport(
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_fraction=trainable_fraction,
        trainable_by_label=dict(trainable_by_label),
        frozen_by_label=dict(frozen_by_label),
        adapter_params=adapter_params,
        head_params=head_params,
        mergeable=_is_mergeable_plan(trainable_infos),
        estimated_delta_size_bytes=estimated_delta_size_bytes,
        target_paths=tuple(target_paths),
        parameters=tuple(parameters),
        model_signature=signature,
        plan_fingerprint=fingerprint,
    )


def _param_info_leaves(tree: Any) -> tuple[ParamInfo, ...]:
    return tuple(leaf for leaf in jtu.tree_leaves(tree) if isinstance(leaf, ParamInfo))


def validate_plan(
    plan: FineTunePlan, *, expected_fingerprint: str | None = None
) -> None:
    """Check a prepared partition and an optional saved plan fingerprint."""

    model = plan.combine()
    actual = make_trainable_report(model, plan.param_info)
    if actual != plan.report:
        raise ValueError("Fine-tuning plan report does not match parameter metadata.")
    if (
        expected_fingerprint is not None
        and actual.plan_fingerprint != expected_fingerprint
    ):
        raise ValueError("Fine-tuning plan fingerprint does not match the saved plan.")

    selected = {row.physical_path for row in actual.parameters if row.trainable}
    all_paths = {row.physical_path for row in actual.parameters}
    if {path for path, _ in iter_param_leaves(plan.trainable)} != selected:
        raise ValueError("Trainable partition does not match selected parameter IDs.")
    if {path for path, _ in iter_param_leaves(plan.frozen)} != all_paths - selected:
        raise ValueError("Frozen partition does not match selected parameter IDs.")
    labels = {
        key_path_to_path(key_path): leaf
        for key_path, leaf in jtu.tree_leaves_with_path(plan.labels)
        if isinstance(leaf, str)
    }
    if labels != {
        row.physical_path: row.label
        for row in actual.parameters
        if row.label is not None
    }:
        raise ValueError("Optimizer labels do not match selected parameter IDs.")
    identities = {
        key_path_to_path(key_path): identity
        for key_path, identity in jtu.tree_leaves_with_path(plan.identities)
        if hasattr(identity, "logical_id")
    }
    if set(identities) != all_paths or any(
        identities[row.physical_path].logical_id != row.logical_id
        or identities[row.physical_path].physical_path != row.physical_path
        or identities[row.physical_path].tags != frozenset(row.tags)
        or identities[row.physical_path].depth != row.depth
        for row in actual.parameters
    ):
        raise ValueError("Parameter identities do not match the plan report.")
    masks = {
        key_path_to_path(key_path): mask
        for key_path, mask in jtu.tree_leaves_with_path(plan.trainable_mask)
        if isinstance(mask, bool)
    }
    if any(masks.get(path) != (path in selected) for path in all_paths):
        raise ValueError("Trainability mask does not match selected parameter IDs.")
    groups = {
        row.label
        for row in actual.parameters
        if row.trainable and row.label is not None
    }
    if set(plan.group_specs) != groups:
        raise ValueError("Optimizer groups do not match selected parameter labels.")
    for row in actual.parameters:
        if not row.trainable:
            continue
        if row.label is None or row.lr_multiplier is None:
            raise ValueError(f"Missing optimizer group for {row.logical_id!r}.")
        group = plan.group_specs[row.label]
        if (
            group.lr_multiplier != row.lr_multiplier
            or group.weight_decay != row.weight_decay
        ):
            raise ValueError(f"Optimizer group disagrees for {row.logical_id!r}.")


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _frozen_label(info: ParamInfo) -> str:
    if info.role:
        return info.role
    if info.depth is not None:
        return "block"
    return "frozen"


def _is_adapter_leaf(info: ParamInfo) -> bool:
    tags = info.tags
    return any(
        tag in tags
        for tag in (
            "adapter",
            "lora",
            "prompt",
            "prefix",
            "ia3",
            "scale_shift",
            "adalora",
        )
    )


def _is_mergeable_plan(infos: list[ParamInfo]) -> bool:
    mergeable_tags = {"lora", "dora", "ia3", "vera", "adalora"}
    nonmergeable_tags = {"adapter", "prompt", "prefix"}
    has_mergeable_leaf = False
    for info in infos:
        if not info.tags.isdisjoint(nonmergeable_tags):
            return False
        if not info.tags.isdisjoint(mergeable_tags):
            has_mergeable_leaf = True
    return has_mergeable_leaf


__all__ = (
    "inspect_plan",
    "inspect_trainables",
    "make_trainable_report",
    "validate_plan",
)
