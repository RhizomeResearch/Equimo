"""Integration coverage of stable fine-tuning strategies on production models."""

from dataclasses import replace
from typing import get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from cases.finetune_cases import (
    FINETUNE_MODE_CASES,
    PEFT_CASES,
    build_model_invocation,
)
from cases.model_cases import extract_features
from equimo.finetune.config import TrainableMode


def test_every_stable_trainable_mode_has_a_real_model_case():
    assert {case.mode for case in FINETUNE_MODE_CASES} == set(get_args(TrainableMode))
    assert {case.model_name for case in FINETUNE_MODE_CASES} == {
        "vit",
        "convnext",
        "ast",
        "text_transformer_encoder",
        "tabpfn",
        "t0",
    }


@pytest.mark.parametrize("case", FINETUNE_MODE_CASES, ids=lambda case: case.mode)
def test_trainable_mode_partitions_and_recombines_a_real_model(case):
    key = jr.PRNGKey(70)
    invocation = build_model_invocation(case.model_name, key)
    model = case.adapt(invocation.model, key)
    plan = eqft.prepare_finetune(model, trainable=case.spec)
    combined = plan.combine(plan.trainable)

    assert jax.tree.structure(combined) == jax.tree.structure(model)
    assert eqx.tree_equal(combined, model)
    assert plan.report.total_params > 0
    if case.mode == "frozen":
        assert plan.report.trainable_params == 0
    else:
        assert plan.report.trainable_params > 0


@pytest.mark.parametrize(
    "model_name",
    ("vit", "convnext", "ast", "text_transformer_encoder", "tabpfn", "t0"),
)
def test_real_model_full_finetune_combination_is_jittable(model_name):
    invocation = build_model_invocation(model_name, jr.PRNGKey(71))
    plan = eqft.prepare_finetune(
        invocation.model,
        trainable=eqft.TrainableSpec(mode="full"),
    )

    def inference(*args):
        combined = plan.combine(plan.trainable)
        updated = replace(invocation, model=combined)
        return extract_features(updated, *args, key=invocation.key)

    eager = inference(*invocation.args)
    compiled = jax.jit(inference)(*invocation.args)

    assert eager.shape == compiled.shape
    assert eager.dtype == compiled.dtype
    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)


def test_every_public_peft_config_family_has_a_real_model_case():
    assert {case.config_type for case in PEFT_CASES} == set(get_args(eqft.PEFTConfig))


@pytest.mark.parametrize(
    "case",
    PEFT_CASES,
    ids=lambda case: case.config_type.__name__,
)
def test_peft_config_applies_to_and_partitions_a_real_vit(case):
    key = jr.PRNGKey(72)
    invocation = build_model_invocation("vit", key)
    adapted = case.apply(invocation.model, case.config, key)
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name=case.config_type.__name__.removesuffix("Config").lower(),
            train_head=False,
        ),
    )

    assert jax.tree.structure(plan.combine(plan.trainable)) == jax.tree.structure(
        adapted
    )
    assert plan.report.trainable_params > 0
    assert plan.report.trainable_params < plan.report.total_params
