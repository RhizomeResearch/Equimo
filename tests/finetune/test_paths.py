"""Fine-tuning path-construction tests."""

from __future__ import annotations

import equinox as eqx
import jax.random as jr

import equimo.finetune as eqft

from fixtures import extract_paths


def test_make_path_tree_uses_stable_dot_path_parts(tiny_vision_transformer):
    path_tree = eqft.make_path_tree(tiny_vision_transformer)

    assert path_tree.patch_embed.proj.weight == ("patch_embed", "proj", "weight")
    assert path_tree.blocks[1].attn.qkv.bias == (
        "blocks",
        1,
        "attn",
        "qkv",
        "bias",
    )


def test_iter_param_paths_matches_fixture_extraction(tiny_vision_transformer):
    assert eqft.extract_param_paths(tiny_vision_transformer) == extract_paths(
        tiny_vision_transformer
    )


def test_make_param_info_tree_records_array_metadata(tiny_vision_transformer):
    info_tree = eqft.make_param_info_tree(tiny_vision_transformer)
    info = info_tree.blocks[0].attn.qkv.weight

    assert info.path == ("blocks", 0, "attn", "qkv", "weight")
    assert info.is_array is True
    assert info.is_inexact_array is True


def test_path_string_round_trip():
    path = ("blocks", 1, "attn", "qkv", "weight")

    assert eqft.path_to_str(path) == "blocks.1.attn.qkv.weight"
    assert eqft.str_to_path("blocks.1.attn.qkv.weight") == path


def test_path_string_round_trip_preserves_ambiguous_string_parts():
    paths = (
        ("0", "weight"),
        ("a.b", "weight"),
        ("a\\b", "weight"),
        ("", "weight"),
        ("-1", "weight"),
    )

    for path in paths:
        assert eqft.str_to_path(eqft.path_to_str(path)) == path

    assert eqft.path_to_str(("0", "weight")) != eqft.path_to_str((0, "weight"))


def test_apply_lora_supports_mapping_backed_pytrees():
    model = {"proj": eqx.nn.Linear(3, 4, key=jr.PRNGKey(0))}

    adapted = eqft.apply_lora(
        model,
        eqft.LoRAConfig(target=eqft.TargetSpec(include=("proj",))),
        key=jr.PRNGKey(1),
    )

    assert isinstance(adapted["proj"], eqft.LoRALinear)
