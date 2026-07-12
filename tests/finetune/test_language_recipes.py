"""Language recipe tests."""

from __future__ import annotations

import inspect

import jax.random as jr

from equimo.finetune.language import recipes


def test_projection_head_recipe_uses_feature_vocabulary():
    parameters = inspect.signature(recipes.projection_head).parameters

    assert "in_features" in parameters
    assert "out_features" in parameters
    assert "out_dim" not in parameters


def test_language_recipes_work_on_tiny_text(tiny_text_encoder):
    lora = recipes.lora_encoder(tiny_text_encoder, key=jr.PRNGKey(0), rank=2)
    prefix = recipes.prefix_encoder(
        tiny_text_encoder, key=jr.PRNGKey(1), num_prefix_tokens=2
    )
    head = recipes.projection_head(
        in_features=4,
        out_features=3,
        key=jr.PRNGKey(2),
    )
    frozen = recipes.locked_tower(tiny_text_encoder)

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert prefix.prefixes[0].shape == (2, 4)
    assert head.head.layers[-1].out_features == 3
    assert frozen.report.trainable_params == 0
