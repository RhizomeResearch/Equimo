"""Cross-method PEFT invariant smoke tests."""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

import equimo.finetune as eqft
from equimo.core.parameters import buffer_mask


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


_PROJ = eqft.TargetSpec(tags_any=("attention.proj",))


@pytest.mark.parametrize(
    ("apply", "buffer_names"),
    [
        pytest.param(
            partial(eqft.apply_vera, config=eqft.VeRAConfig(rank=3, target=_PROJ)),
            {"vera_A", "vera_B"},
            id="vera",
        ),
        pytest.param(
            partial(eqft.apply_lora_fa, config=eqft.LoRAFAConfig(rank=2, alpha=4.0)),
            {"frozen_A", "correction_matrix"},
            id="lora_fa",
        ),
        pytest.param(
            partial(
                eqft.apply_lora_fa,
                config=eqft.LoRAFAConfig(rank=2, alpha=4.0, custom_vjp=False),
            ),
            {"frozen_A", "correction_matrix"},
            id="lora_fa_autodiff",
        ),
        pytest.param(
            partial(
                eqft.apply_randlora,
                config=eqft.RandLoRAConfig(rank=1, basis_count=2, target=_PROJ),
            ),
            {"random_A", "random_B"},
            id="randlora",
        ),
        pytest.param(
            partial(eqft.apply_lora, config=eqft.PiSSAConfig(rank=2, target=_PROJ)),
            {"base_weight_delta"},
            id="pissa",
        ),
        pytest.param(
            partial(
                eqft.apply_adaptformer, config=eqft.AdaptFormerConfig.paper_chen2022()
            ),
            {"scale"},
            id="adaptformer_fixed_scale",
        ),
    ],
)
def test_peft_buffers_get_no_gradient_and_are_never_trained(
    apply, buffer_names, tiny_vision_transformer
):
    # Offset every array so that no zero initialization hides a gradient path.
    model = jax.tree.map(
        lambda leaf: leaf + 0.1 if eqx.is_inexact_array(leaf) else leaf,
        apply(tiny_vision_transformer, key=jr.PRNGKey(0)),
    )
    buffers = buffer_mask(model)
    assert {
        path[-1].name
        for path, is_buffer in jtu.tree_leaves_with_path(buffers)
        if is_buffer
    } == buffer_names

    grads = eqx.filter_grad(lambda m: jnp.sum(m(jnp.ones((2, 3)))))(model)
    plan = eqft.prepare_finetune(model, trainable=eqft.TrainableSpec(mode="full"))

    def leaks(is_buffer, grad, trainable):
        if not is_buffer:
            return False
        return trainable is not None or (grad is not None and bool(jnp.any(grad)))

    leaked = jtu.tree_map(leaks, buffers, grads, plan.trainable)
    assert not [
        jtu.keystr(path) for path, leak in jtu.tree_leaves_with_path(leaked) if leak
    ]
