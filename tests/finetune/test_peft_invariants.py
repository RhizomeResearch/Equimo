"""Cross-method PEFT invariant smoke tests."""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(
            partial(
                eqft.apply_lora,
                config=eqft.LoRAConfig(
                    rank=2, target=eqft.TargetSpec(tags_any=("attention.proj",))
                ),
                key=jr.PRNGKey(0),
            ),
            id="lora",
        ),
        pytest.param(
            partial(
                eqft.apply_dora,
                config=eqft.DoRAConfig(
                    rank=2, target=eqft.TargetSpec(tags_any=("attention.proj",))
                ),
                key=jr.PRNGKey(0),
            ),
            id="dora",
        ),
        pytest.param(
            partial(
                eqft.apply_adapters,
                config=eqft.AdapterConfig(bottleneck=2),
                key=jr.PRNGKey(0),
            ),
            id="adapters",
        ),
        pytest.param(
            partial(
                eqft.apply_prompts,
                config=eqft.PromptConfig(num_tokens=2),
                key=jr.PRNGKey(0),
            ),
            id="prompts",
        ),
        pytest.param(
            partial(
                eqft.apply_prefixes,
                config=eqft.PrefixConfig(num_prefix_tokens=2),
                key=jr.PRNGKey(0),
            ),
            id="prefixes",
        ),
        pytest.param(
            partial(
                eqft.apply_scale_shift,
                config=eqft.ScaleShiftConfig(
                    target=eqft.TargetSpec(include=("*.norm",))
                ),
            ),
            id="scale_shift",
        ),
        pytest.param(
            partial(
                eqft.apply_ia3,
                config=eqft.IA3Config(
                    target=eqft.TargetSpec(tags_any=("attention.proj",))
                ),
            ),
            id="ia3",
        ),
        pytest.param(
            partial(
                eqft.apply_vera,
                config=eqft.VeRAConfig(
                    rank=3, target=eqft.TargetSpec(tags_any=("attention.proj",))
                ),
                key=jr.PRNGKey(0),
            ),
            id="vera",
        ),
    ],
)
def test_peft_wrappers_filter_jit_and_vmap(apply, tiny_vision_transformer):
    model = apply(tiny_vision_transformer)
    x = jnp.ones((2, 3), dtype=jnp.float32)

    y_jit = eqx.filter_jit(model)(x)
    y_batch = jax.vmap(model)(jnp.stack([x, x]))

    assert y_jit.shape == (2,)
    assert y_batch.shape == (2, 2)
