"""Structural trainability checks for probe wrappers and parameter reports."""

from __future__ import annotations

import json
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.finetune.inspection import make_trainable_report
from equimo.vision.models.vit import VisionTransformer


def _probe(kind: str, *, depths=(2, 2, 1)):
    backbone = VisionTransformer(
        img_size=4,
        in_channels=3,
        dim=8,
        patch_size=2,
        num_heads=[2] * len(depths),
        depths=list(depths),
        num_classes=0,
        reg_tokens=1,
        key=jr.PRNGKey(0),
    )
    if kind == "pooled":
        return eqft.make_linear_probe(
            backbone,
            in_features=8,
            out_features=3,
            key=jr.PRNGKey(1),
            feature_spec=eqft.FeatureSpec("features", "BNC", "cls", None),
        )
    return eqft.vision.make_dense_probe(
        backbone,
        in_features=8,
        out_features=3,
        key=jr.PRNGKey(1),
        feature_spec=eqft.FeatureSpec(
            "features", "BNC", "patches", None, return_metadata=True
        ),
    )


def _selected_ids(plan):
    return {row.logical_id for row in plan.report.parameters if row.trainable}


@pytest.mark.parametrize("kind", ("pooled", "dense"))
def test_last_two_blocks_cross_chunks_and_freeze_other_backbone_leaves(kind):
    model = _probe(kind)
    plan = eqft.partial_ft_last_k_blocks(model, k=2, train_norm=False)
    expected = {
        eqft.path_to_str(("backbone", "blocks", chunk, "blocks", block, *leaf))
        for chunk, block in ((1, 1), (2, 0))
        for leaf in (
            ("prenorm", "weight"),
            ("prenorm", "bias"),
            ("attn", "qkv", "weight"),
            ("attn", "qkv", "bias"),
            ("attn", "proj", "weight"),
            ("attn", "proj", "bias"),
            ("norm", "weight"),
            ("norm", "bias"),
            ("mlp", "fc1", "weight"),
            ("mlp", "fc1", "bias"),
            ("mlp", "fc2", "weight"),
            ("mlp", "fc2", "bias"),
        )
    }
    expected.update(("head.linear.weight", "head.linear.bias"))
    assert _selected_ids(plan) == expected
    assert plan.trainable.backbone.norm.weight is None
    assert plan.trainable.backbone.reg_tokens is None
    assert plan.trainable.backbone.cls_token is None
    assert plan.trainable.backbone.patch_embed.proj.weight is None
    assert plan.group_specs["block_04_decay"].lr_multiplier == pytest.approx(1)
    assert plan.group_specs["block_03_decay"].lr_multiplier == pytest.approx(0.75)
    assert (
        plan.param_info.backbone.blocks[1].blocks[1].prenorm.weight.weight_decay
        is False
    )


def test_invalid_block_count_and_empty_head_fail():
    model = _probe("pooled")
    with pytest.raises(ValueError, match="k"):
        eqft.partial_ft_last_k_blocks(model, k=6)
    with pytest.raises(ValueError, match="depth range"):
        eqft.prepare_finetune(
            model,
            trainable=eqft.TrainableSpec(mode="partial", depth_range=(4, 6)),
        )
    with pytest.raises(ValueError, match="no trainable"):
        eqft.prepare_finetune(
            {"weight": jnp.ones(2)}, trainable=eqft.TrainableSpec(mode="head")
        )


def test_last_two_block_indices_use_numeric_execution_order():
    model = _probe("pooled", depths=(2, 9, 1))
    plan = eqft.partial_ft_last_k_blocks(model, k=2, train_norm=False)
    selected_blocks = {
        row.physical_path[2:5]
        for row in plan.report.parameters
        if row.trainable and row.depth is not None
    }
    assert selected_blocks == {(1, "blocks", 8), (2, "blocks", 0)}
    assert {row.depth for row in plan.report.parameters if row.trainable} == {
        None,
        10,
        11,
    }
    assert plan.group_specs["block_10_decay"].lr_multiplier == pytest.approx(0.75)
    assert plan.group_specs["block_11_decay"].lr_multiplier == pytest.approx(1)


def test_inconsistent_optimizer_labels_fail():
    with pytest.raises(ValueError, match="inconsistent learning rate"):
        eqft.prepare_finetune(
            _probe("pooled"),
            trainable=eqft.TrainableSpec(mode="full"),
            labels=eqft.LLRDConfig(block_label_format="all_blocks"),
        )


def test_duplicate_logical_id_report_fails():
    model = {"one": jnp.ones(1), "two": jnp.ones(1)}
    metadata = {
        "one": eqft.ParamInfo(path=("one",), logical_id="duplicate"),
        "two": eqft.ParamInfo(path=("two",), logical_id="duplicate"),
    }
    with pytest.raises(ValueError, match="Duplicate logical"):
        make_trainable_report(model, metadata)


def test_adalora_leaf_roles_and_groups_are_complete(tiny_vision_transformer):
    model = eqft.apply_adalora(
        tiny_vision_transformer,
        eqft.AdaLoRAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(9),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft", method_name="adalora", train_head=False
        ),
    )
    assert _selected_ids(plan) == {
        "blocks.0.attn.proj.P",
        "blocks.0.attn.proj.singular",
        "blocks.0.attn.proj.Q",
    }
    assert set(plan.group_specs) == {
        "adalora_P_decay",
        "adalora_singular_decay",
        "adalora_Q_decay",
    }
    assert plan.trainable.blocks[0].attn.proj.base.weight is None
    eqft.validate_plan(plan)


def test_logical_ids_escape_paths_and_expected_guards():
    model = {"a.b": jnp.ones(2), "a": {"b": jnp.zeros(2)}}
    plan = eqft.prepare_finetune(model, trainable=eqft.TrainableSpec(mode="full"))
    assert _selected_ids(plan) == {"a.b", "a\\.b"}
    with pytest.raises(ValueError, match="unexpected"):
        eqft.prepare_finetune(
            model,
            trainable=eqft.TrainableSpec(mode="full", expected_logical_ids=("a.b",)),
        )
    with pytest.raises(ValueError, match="missing"):
        eqft.resolve_target(
            model,
            eqft.TargetSpec(include=("a",), expected_logical_ids=("a.b", "renamed")),
        )


def test_report_roundtrip_and_structure_fingerprint():
    model = _probe("pooled")
    spec = eqft.TrainableSpec(mode="head")
    plan = eqft.prepare_finetune(model, trainable=spec)
    payload = json.loads(json.dumps(plan.report.to_dict()))
    assert payload["schema_version"] == 1
    assert payload["model_signature"].startswith("sha256:")
    assert payload["plan_fingerprint"].startswith("sha256:")
    assert len(payload["parameters"]) == len(eqft.iter_param_paths(model))
    eqft.validate_plan(plan, expected_fingerprint=payload["plan_fingerprint"])
    changed_value = eqx.tree_at(
        lambda tree: tree.head.linear.weight,
        model,
        model.head.linear.weight + 1,
    )
    same_structure = eqft.prepare_finetune(changed_value, trainable=spec)
    assert same_structure.report.plan_fingerprint == plan.report.plan_fingerprint
    changed_shape = eqx.tree_at(
        lambda tree: tree.head.linear.weight,
        model,
        jnp.ones((4, 8), dtype=jnp.float32),
    )
    assert (
        eqft.prepare_finetune(changed_shape, trainable=spec).report.model_signature
        != plan.report.model_signature
    )
    changed_dtype = eqx.tree_at(
        lambda tree: tree.head.linear.weight,
        model,
        model.head.linear.weight.astype(jnp.bfloat16),
    )
    assert (
        eqft.prepare_finetune(changed_dtype, trainable=spec).report.model_signature
        != plan.report.model_signature
    )
    renamed = eqft.prepare_finetune(
        {"probe": model}, trainable=eqft.TrainableSpec(mode="full")
    )
    with pytest.raises(ValueError, match="fingerprint"):
        eqft.validate_plan(renamed, expected_fingerprint=payload["plan_fingerprint"])
    with pytest.raises(ValueError, match="Optimizer groups"):
        eqft.validate_plan(replace(plan, group_specs={}))


@pytest.mark.parametrize("kind", ("pooled", "dense"))
def test_momentum_decay_updates_only_selected_leaves(kind):
    model = _probe(kind)
    plan = eqft.partial_ft_last_k_blocks(model, k=2, train_norm=False)
    before = dict(eqft.iter_param_leaves(model))
    trainable = plan.trainable
    velocity = jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else None,
        trainable,
    )
    for _ in range(2):
        gradients = jax.tree.map(
            lambda leaf: jnp.ones_like(leaf) if eqx.is_inexact_array(leaf) else None,
            trainable,
        )
        velocity = jax.tree.map(
            lambda old, grad, leaf: (
                0.9 * old + grad + 0.05 * leaf if eqx.is_inexact_array(leaf) else None
            ),
            velocity,
            gradients,
            trainable,
        )
        trainable = jax.tree.map(
            lambda leaf, step: (
                leaf - 0.01 * step if eqx.is_inexact_array(leaf) else None
            ),
            trainable,
            velocity,
        )
    after = dict(eqft.iter_param_leaves(plan.combine(trainable)))
    for path, old in before.items():
        if eqft.path_to_str(path) in _selected_ids(plan):
            assert not jnp.array_equal(old, after[path])
        else:
            assert jnp.array_equal(old, after[path])
