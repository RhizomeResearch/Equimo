"""Continued SSL adaptation planning tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.vision.models.vit import VisionTransformer


def test_continued_ssl_default_applies_lora_and_unfreezes_last_block(
    tiny_vision_transformer,
):
    ssl_plan = eqft.continued_ssl_adaptation(
        tiny_vision_transformer,
        key=jr.PRNGKey(0),
    )
    plan = ssl_plan.plan

    assert ssl_plan.stage == "continued_ssl_delta"
    assert ssl_plan.metadata["unfreeze_blocks"] == 1
    assert plan.trainable.blocks[0].attn.qkv.lora_A is not None
    assert plan.trainable.blocks[0].attn.qkv.base.weight is None
    assert plan.trainable.blocks[1].attn.qkv.base.weight is not None
    assert plan.trainable.blocks[0].norm1.weight is not None
    assert plan.trainable.head.weight is None


def test_continued_ssl_trainable_spec_can_skip_peft_and_blocks(tiny_vision_transformer):
    spec = eqft.continued_ssl_trainable_spec(
        tiny_vision_transformer,
        eqft.ContinuedSSLAdaptationConfig(
            peft=None,
            unfreeze_blocks=0,
            train_norm=False,
            train_head=True,
        ),
    )

    assert spec.mode == "head"


def test_supervised_after_ssl_uses_existing_delta_and_head(tiny_vision_transformer):
    adapted = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(rank=2, alpha=4),
        key=jr.PRNGKey(0),
    )
    supervised = eqft.supervised_after_ssl(adapted)

    assert supervised.stage == "supervised_after_ssl"
    assert supervised.metadata["reuse_ssl_delta"] is True
    assert supervised.plan.trainable.blocks[0].attn.qkv.lora_A is not None
    assert supervised.plan.trainable.blocks[0].attn.qkv.base.weight is None
    assert supervised.plan.trainable.head.weight is not None


def test_supervised_after_ssl_requires_key_when_not_reusing_delta(
    tiny_vision_transformer,
):
    with pytest.raises(ValueError, match="PRNG key"):
        eqft.supervised_after_ssl(
            tiny_vision_transformer,
            eqft.SupervisedAfterSSLConfig(reuse_ssl_delta=False),
        )


@pytest.mark.parametrize(("policy", "expected"), [("last", (1, 1)), ("first", (0, 0))])
def test_continued_ssl_counts_blocks_across_vit_chunks(policy, expected):
    model = VisionTransformer(
        img_size=32,
        in_channels=3,
        dim=32,
        patch_size=8,
        num_heads=2,
        depths=[2, 2],
        num_classes=2,
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.continued_ssl_trainable_spec(
            model,
            eqft.ContinuedSSLAdaptationConfig(
                peft=None,
                unfreeze_blocks=1,
                unfreeze_policy=policy,
                train_norm=False,
            ),
        ),
    )

    trainable_blocks = [
        (chunk_index, block_index)
        for chunk_index, chunk in enumerate(plan.trainable.blocks)
        for block_index, block in enumerate(chunk.blocks)
        if jax.tree.leaves(eqx.filter(block, eqx.is_array))
    ]
    assert trainable_blocks == [expected]
