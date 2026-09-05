"""Fine-tuning selector-resolution tests."""

from __future__ import annotations

import pytest
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_predicate_order_root_and_nested_prefixes():
    linear = eqx.nn.Linear(2, 2, key=jr.PRNGKey(0))
    model = {"a": [linear], "ab": linear, "loose": jnp.ones(2)}
    calls = []

    def predicate(path, node):
        calls.append(path)
        return (isinstance(node, eqx.nn.Linear) and path == ("a", 0)) or path == (
            "loose",
        )

    paths = eqft.resolve_target_paths(
        model,
        eqft.TargetSpec(predicate=predicate, exclude=("*.bias",)),
    )
    assert paths == ("a.0.weight", "loose")
    assert calls == [
        ("a", 0, "weight"),
        ("a", 0, "bias"),
        ("ab", "weight"),
        ("ab", "bias"),
        ("loose",),
        ("a", 0),
        ("ab",),
    ]
    assert eqft.resolve_target_paths(
        linear, eqft.TargetSpec(predicate=eqft.is_linear)
    ) == ("weight", "bias")
    assert (
        eqft.resolve_target_paths(
            model, eqft.TargetSpec(predicate=lambda p, n: False, allow_empty=True)
        )
        == ()
    )


def test_selector_tags_qkv(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags_any=("attention.qkv",)),
    )

    assert paths == (
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.bias",
        "blocks.1.attn.qkv.weight",
        "blocks.1.attn.qkv.bias",
    )


def test_selector_broad_attention_and_mlp_tags(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags_any=("attention", "mlp")),
    )

    assert "blocks.0.attn.qkv.weight" in paths
    assert "blocks.0.attn.proj.bias" in paths
    assert "blocks.0.mlp.fc1.weight" in paths
    assert "blocks.1.mlp.fc2.bias" in paths


def test_selector_tags_patch_embed(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags_any=("embedding.patch",)),
    )

    assert paths == ("patch_embed.proj.weight", "patch_embed.proj.bias")


def test_selector_glob_qkv(tiny_vision_transformer):
    semantic_paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags_any=("attention.qkv",)),
    )
    glob_paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(include=("*.blocks.*.attn.qkv",)),
    )

    assert glob_paths == semantic_paths


def test_selector_exclude_pos_embed(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(include=("*",), exclude=("*.pos_embed", "*.cls_token")),
    )

    assert "patch_embed.proj.weight" in paths
    assert "head.bias" in paths
    assert "pos_embed" not in paths
    assert "cls_token" not in paths


def test_selector_callable_linear(tiny_linear_mlp):
    paths = eqft.resolve_target_paths(
        tiny_linear_mlp,
        eqft.TargetSpec(predicate=eqft.is_linear),
    )

    assert paths == ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")


def test_selector_empty_raises(tiny_vision_transformer):
    with pytest.raises(ValueError, match="missing.tag"):
        eqft.resolve_target_paths(
            tiny_vision_transformer,
            eqft.TargetSpec(tags_any=("missing.tag",)),
        )


def test_depth_resolution_for_blocks(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags_any=("block",), min_depth=1, max_depth=1),
    )

    assert paths
    assert all(path.startswith("blocks.1.") for path in paths)
    assert not any(path.startswith("blocks.0.") for path in paths)
