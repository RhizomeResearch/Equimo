"""LoRA+ label tests."""

from __future__ import annotations

import jax.random as jr

import equimo.finetune as eqft


def _lora(model):
    return eqft.apply_lora(
        model,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )


_LORA_ONLY = eqft.TrainableSpec(mode="peft", method_name="lora", train_head=False)


def test_lora_plus_labels_expose_a_and_b(tiny_vision_transformer):
    plan = eqft.prepare_finetune(_lora(tiny_vision_transformer), trainable=_LORA_ONLY)

    assert "lora_A_decay" in plan.group_specs
    assert "lora_B_decay" in plan.group_specs


def test_lora_labels_name_the_block_under_layer_wise_decay(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        _lora(tiny_vision_transformer),
        trainable=_LORA_ONLY,
        labels=eqft.LLRDConfig(decay=0.5),
    )

    rates = {label: group.lr_multiplier for label, group in plan.group_specs.items()}
    assert rates == {
        "lora_A_block_00_decay": 0.5,
        "lora_A_block_01_decay": 1.0,
        "lora_B_block_00_decay": 0.5,
        "lora_B_block_01_decay": 1.0,
    }
