"""LoRA delta serialization tests."""

from __future__ import annotations

import pytest
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft

from fixtures import TinyVisionTransformer


def _trained_like_lora(model):
    lora = eqft.apply_lora(
        model,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    module = lora.blocks[0].attn.proj
    module = eqx.tree_at(
        lambda m: m.lora_B,
        module,
        jnp.ones_like(module.lora_B) * 0.01,
    )
    return eqx.tree_at(lambda m: m.blocks[0].attn.proj, lora, module)


def test_lora_delta_roundtrip(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    lora = _trained_like_lora(model)
    x = jnp.ones((2, 3))
    path = tmp_path / "delta.eqft"

    eqft.save_delta(lora, path)
    loaded = eqft.load_delta(model, path)

    assert jnp.allclose(lora(x), loaded(x), atol=1e-6)


def test_lora_delta_incompatible_shape_raises(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    lora = _trained_like_lora(model)
    incompatible = TinyVisionTransformer(dim=5, hidden_dim=10, key=jr.PRNGKey(1))
    path = tmp_path / "delta.eqft"
    eqft.save_delta(lora, path)

    with pytest.raises(eqft.FineTuneBundleError, match="architecture hash mismatch"):
        eqft.load_delta(incompatible, path)


@pytest.mark.parametrize("method", ("adalora", "lora_fa", "fourierft"))
def test_lora_family_delta_roundtrip(method, tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    target = eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0)

    if method == "adalora":
        adapted = eqft.apply_adalora(
            model,
            eqft.AdaLoRAConfig(rank=2, target=target),
            key=jr.PRNGKey(1),
        )
        module = adapted.blocks[0].attn.proj
        module = eqx.tree_at(
            lambda item: item.singular,
            module,
            jnp.asarray([0.1, -0.2], dtype=module.singular.dtype),
        )
        adapted = eqx.tree_at(lambda item: item.blocks[0].attn.proj, adapted, module)
        expected_type = eqft.AdaLoRAModule
        trained_leaf = "singular"
    elif method == "lora_fa":
        adapted = eqft.apply_lora_fa(
            model,
            eqft.LoRAFAConfig(rank=2, target=target),
            key=jr.PRNGKey(1),
        )
        module = adapted.blocks[0].attn.proj
        module = eqx.tree_at(
            lambda item: item.lora_fa_B,
            module,
            jnp.full_like(module.lora_fa_B, 0.05),
        )
        adapted = eqx.tree_at(lambda item: item.blocks[0].attn.proj, adapted, module)
        expected_type = eqft.LoRAFALinear
        trained_leaf = "lora_fa_B"
    else:
        adapted = eqft.apply_fourierft(
            model,
            eqft.FourierFTConfig(
                num_coefficients=2,
                target=target,
                frequency_selection="low_frequency",
            ),
        )
        module = adapted.blocks[0].attn.proj
        module = eqx.tree_at(
            lambda item: item.coefficients_real,
            module,
            jnp.asarray([0.25, -0.125], dtype=module.coefficients_real.dtype),
        )
        adapted = eqx.tree_at(lambda item: item.blocks[0].attn.proj, adapted, module)
        expected_type = eqft.FourierFTLinear
        trained_leaf = "coefficients_real"

    path = tmp_path / f"{method}.eqft"
    bundle = eqft.save_delta(adapted, path, method="lora")
    loaded = eqft.load_delta(model, path)

    loaded_module = loaded.blocks[0].attn.proj
    assert bundle.adapter_config["entries"]
    assert isinstance(loaded_module, expected_type)
    assert jnp.array_equal(
        getattr(loaded_module, trained_leaf),
        getattr(adapted.blocks[0].attn.proj, trained_leaf),
    )
    assert jnp.allclose(
        loaded(jnp.ones((2, 3))),
        adapted(jnp.ones((2, 3))),
        atol=1e-6,
    )


def test_delta_roundtrip_preserves_trainable_head_and_base_weight(tmp_path):
    base = TinyVisionTransformer(key=jr.PRNGKey(0))
    adapted = eqft.apply_lora(
        base,
        eqft.LoRAConfig(
            rank=2,
            train_base=True,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(1),
    )
    adapted = eqx.tree_at(
        lambda model: model.head.weight,
        adapted,
        adapted.head.weight + 0.25,
    )
    adapted = eqx.tree_at(
        lambda model: model.blocks[0].attn.proj.base.weight,
        adapted,
        adapted.blocks[0].attn.proj.base.weight - 0.125,
    )
    spec = eqft.TrainableSpec(
        mode="peft",
        method_name="lora",
        train_head=True,
    )
    path = tmp_path / "trained-base-and-head.eqft"

    bundle = eqft.save_delta(path, adapted, base, spec, method="lora")
    loaded = eqft.load_delta(path, base)

    assert bundle.delta_tree
    assert jnp.array_equal(loaded.head.weight, adapted.head.weight)
    assert jnp.array_equal(
        loaded.blocks[0].attn.proj.base.weight,
        adapted.blocks[0].attn.proj.base.weight,
    )
