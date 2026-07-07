"""Baseline fine-tuning recipe tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.core.layers.dropout import DropPathAdd
from equimo.finetune.vision import recipes as vision_recipes

from fixtures import assert_tree_allclose


class _StochasticBlock(eqx.Module):
    proj: eqx.nn.Linear
    drop_path: DropPathAdd
    dropout: eqx.nn.Dropout

    def __init__(self, *, key):
        self.proj = eqx.nn.Linear(4, 4, key=key)
        self.drop_path = DropPathAdd(0.25)
        self.dropout = eqx.nn.Dropout(0.35)


class _StochasticHead(eqx.Module):
    linear: eqx.nn.Linear
    drop_path: DropPathAdd
    dropout: eqx.nn.Dropout

    def __init__(self, *, key):
        self.linear = eqx.nn.Linear(4, 2, key=key)
        self.drop_path = DropPathAdd(0.45)
        self.dropout = eqx.nn.Dropout(0.55)


class _StochasticLPFTModel(eqx.Module):
    block: _StochasticBlock
    head: _StochasticHead

    def __init__(self, *, key):
        block_key, head_key = jr.split(key, 2)
        self.block = _StochasticBlock(key=block_key)
        self.head = _StochasticHead(key=head_key)


def _head_weight_as_position_tagger(path, leaf):
    tags = set(eqft.canonical_tags_for_path(path, leaf))
    if path == ("head", "weight"):
        tags.add("embedding.position")
    return frozenset(tags)


def test_lpft_stage1_plan_disables_non_head_stochastic_regularization():
    model = _StochasticLPFTModel(key=jr.PRNGKey(0))
    recipe = eqft.lpft()

    plan = recipe.stage1_plan(model)
    stage1_model = plan.combine()

    assert stage1_model.block.drop_path.p == pytest.approx(0.0)
    assert stage1_model.block.dropout.p == pytest.approx(0.0)
    assert stage1_model.head.drop_path.p == pytest.approx(model.head.drop_path.p)
    assert stage1_model.head.dropout.p == pytest.approx(model.head.dropout.p)
    assert plan.trainable.head.linear.weight is not None
    assert plan.trainable.block.proj.weight is None


def test_lpft_stage1_policy_can_leave_stochastic_regularization_unchanged():
    model = _StochasticLPFTModel(key=jr.PRNGKey(0))
    recipe = eqft.lpft(
        stage1_policy=eqft.StagePolicy(deterministic_frozen_backbone=False)
    )

    stage1_model = recipe.stage1_plan(model).combine()

    assert stage1_model.block.drop_path.p == pytest.approx(model.block.drop_path.p)
    assert stage1_model.block.dropout.p == pytest.approx(model.block.dropout.p)


def test_lpft_stage2_transfers_head_and_restores_base_regularization():
    model = _StochasticLPFTModel(key=jr.PRNGKey(0))
    recipe = eqft.lpft(stage2_labels=eqft.LLRDConfig(decay=0.5))
    stage1 = recipe.prepare_stage1(model)
    trained_head = eqx.tree_at(
        lambda head: head.linear.weight,
        stage1.model.head,
        stage1.model.head.linear.weight + 1.0,
    )
    trained_stage1_model = eqx.tree_at(
        lambda stage_model: stage_model.head,
        stage1.model,
        trained_head,
    )

    stage2 = recipe.prepare_stage2(model, trained_stage1_model)
    direct_stage2_plan = recipe.stage2_plan(stage2.model)

    assert stage1.name == "stage1"
    assert stage2.name == "stage2"
    assert stage2.model.block.drop_path.p == pytest.approx(model.block.drop_path.p)
    assert stage2.model.block.dropout.p == pytest.approx(model.block.dropout.p)
    assert jnp.allclose(
        stage2.model.head.linear.weight,
        trained_stage1_model.head.linear.weight,
    )
    assert not jnp.allclose(stage2.model.head.linear.weight, model.head.linear.weight)
    assert stage2.plan.group_specs == direct_stage2_plan.group_specs


def test_lpft_stage1_noops_for_model_without_stochastic_regularization(
    tiny_vision_transformer,
):
    stage1 = eqft.lpft().prepare_stage1(tiny_vision_transformer)

    assert_tree_allclose(stage1.model, tiny_vision_transformer)


def test_lpft_stage_transition_preserves_head(tiny_vision_transformer):
    key = jr.PRNGKey(0)
    trained_like = eqft.replace_head(
        tiny_vision_transformer,
        eqft.LinearHead(4, 4, key=key),
    )
    recipe = eqft.lpft()

    stage2 = recipe.stage2_plan(trained_like)

    assert_tree_allclose(stage2.combine().head, trained_like.head)


def test_lpft_stage2_rejects_unsupported_head_reset(tiny_vision_transformer):
    recipe = eqft.lpft(preserve_trained_head=False)

    with pytest.raises(ValueError, match="preserve_trained_head=True"):
        recipe.stage2_plan(tiny_vision_transformer)


def test_full_ft_llrd_recipe_freezes_patch_embed(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(tiny_vision_transformer, decay=0.75)

    assert plan.trainable.patch_embed.proj.weight is None
    assert "block_01_decay" in plan.group_specs


def test_full_ft_llrd_recipe_accepts_custom_tagger(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(
        tiny_vision_transformer,
        tagger=_head_weight_as_position_tagger,
    )

    assert "embedding.position" in plan.param_info.head.weight.tags
    assert plan.param_info.head.weight.label == "head_no_decay"
    assert plan.param_info.head.weight.weight_decay is False


def test_full_ft_llrd_recipe_accepts_label_policy(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(
        tiny_vision_transformer,
        decay=0.1,
        labels=eqft.LLRDConfig.uniform(),
    )

    assert plan.group_specs["block_00_decay"].lr_multiplier == pytest.approx(1.0)


def test_head_plus_norm_recipe_accepts_config(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(train_head=False, train_norm=True),
    )

    assert plan.trainable.head.weight is None
    assert plan.trainable.blocks[0].norm1.weight is not None


def test_head_plus_norm_config_controls_norm_scale_and_bias(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(
            train_head=False,
            train_norm=True,
            train_norm_scale=False,
            train_norm_bias=True,
        ),
    )

    assert plan.trainable.blocks[0].norm1.weight is None
    assert plan.trainable.blocks[0].norm1.bias is not None


def test_head_plus_norm_config_can_include_embeddings(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(
            train_head=False,
            train_norm=False,
            include_embeddings=True,
            include_positional_parameters=True,
        ),
    )

    assert plan.trainable.patch_embed.proj.weight is not None
    assert plan.trainable.pos_embed is not None
    assert plan.trainable.blocks[0].norm1.weight is None
    assert plan.trainable.head.weight is None


def test_head_plus_norm_recipe_uses_custom_tagger(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(
            train_head=False,
            train_norm=False,
            include_positional_parameters=True,
        ),
        tagger=_head_weight_as_position_tagger,
    )

    assert plan.trainable.head.weight is not None
    assert plan.param_info.head.weight.label == "head_no_decay"


def test_partial_unfreeze_config_controls_span_and_embeddings(tiny_vision_transformer):
    plan = eqft.partial_unfreeze(
        tiny_vision_transformer,
        eqft.PartialUnfreezeConfig(
            fraction=0.5,
            train_embeddings=True,
            train_positional_parameters=True,
        ),
    )

    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert plan.trainable.blocks[1].attn.qkv.weight is not None
    assert plan.trainable.patch_embed.proj.weight is not None
    assert plan.trainable.pos_embed is not None


def test_partial_unfreeze_recipe_uses_custom_tagger(tiny_vision_transformer):
    plan = eqft.partial_unfreeze(
        tiny_vision_transformer,
        eqft.PartialUnfreezeConfig(
            fraction=0.5,
            train_head=False,
            train_norm=False,
            train_positional_parameters=True,
        ),
        tagger=_head_weight_as_position_tagger,
    )

    assert plan.trainable.head.weight is not None
    assert plan.param_info.head.weight.label == "head_no_decay"


def test_partial_ft_last_k_blocks_accepts_custom_tagger(tiny_vision_transformer):
    plan = eqft.partial_ft_last_k_blocks(
        tiny_vision_transformer,
        k=1,
        tagger=_head_weight_as_position_tagger,
    )

    assert plan.trainable.head.weight is not None
    assert plan.param_info.head.weight.label == "head_no_decay"


def test_lpft_stage_plans_accept_custom_tagger(tiny_vision_transformer):
    recipe = eqft.lpft()
    direct_stage2 = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=recipe.stage2,
        labels=recipe.stage2_labels,
        tagger=_head_weight_as_position_tagger,
    )

    stage1 = recipe.stage1_plan(
        tiny_vision_transformer,
        tagger=_head_weight_as_position_tagger,
    )
    stage2 = recipe.stage2_plan(
        tiny_vision_transformer,
        tagger=_head_weight_as_position_tagger,
    )

    assert stage1.param_info.head.weight.label == "head_no_decay"
    assert stage2.param_info.head.weight.label == direct_stage2.param_info.head.weight.label


def test_vision_partial_recipe_uses_last_blocks(tiny_vision_transformer):
    plan = vision_recipes.partial_ft_vit_llrd(
        tiny_vision_transformer,
        last_k_blocks=1,
    )

    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert plan.trainable.blocks[1].attn.qkv.weight is not None


def test_recommended_recipe_aliases_work(tiny_vision_transformer):
    lora = eqft.recipes.lora_transformer(
        tiny_vision_transformer,
        key=jr.PRNGKey(0),
        rank=2,
    )
    lora_all = eqft.recipes.lora_transformer_all_linear(
        tiny_vision_transformer,
        key=jr.PRNGKey(1),
        rank=2,
    )
    prompted = eqft.recipes.vpt_deep(
        tiny_vision_transformer,
        key=jr.PRNGKey(2),
        num_tokens=2,
    )
    bank = eqft.recipes.task_adapter_bank(
        tiny_vision_transformer,
        key=jr.PRNGKey(3),
        names=("task_a", "task_b"),
        bottleneck=3,
    )

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert lora_all.head.lora_A.shape[0] == 2
    assert prompted.prompts[0].shape == (2, 4)
    assert bank.blocks[0].mlp.adapter_names == ("task_a", "task_b")
