"""LoRA delta serialization tests."""

from __future__ import annotations

import pytest
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import equimo.finetune as eqft
from equimo.vision.models import VisionTransformer

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


@pytest.mark.parametrize(
    "spec, needs_base",
    (
        (eqft.TrainableSpec(mode="peft", train_head=False), False),
        (eqft.TrainableSpec(mode="peft"), True),
        (eqft.TrainableSpec(mode="peft", train_head=False, train_bias=True), True),
        (eqft.TrainableSpec(mode="full"), True),
    ),
    ids=("adapters-only", "head", "bias", "full"),
)
def test_save_delta_requires_base_model_only_for_unrepresentable_trainables(
    tmp_path, spec, needs_base
):
    lora = _trained_like_lora(TinyVisionTransformer(key=jr.PRNGKey(0)))
    path = tmp_path / "delta.eqft"

    if needs_base:
        with pytest.raises(ValueError, match="requires base_model"):
            eqft.save_delta(lora, path, spec=spec)
    else:
        eqft.save_delta(lora, path, spec=spec)
        assert path.exists()


def test_lora_delta_roundtrip(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    lora = _trained_like_lora(model)
    x = jnp.ones((2, 3))
    path = tmp_path / "delta.eqft"

    eqft.save_delta(lora, path)
    loaded = eqft.load_delta(model, path)

    assert jnp.allclose(lora(x), loaded(x), atol=1e-6)


class _StoredFreqs(eqx.Module):
    """``VisionRoPE`` as models before Equimo 2.8 held it."""

    freqs: jax.Array


def _rope_vision_transformer(*, base: float = 100.0):
    return VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=16,
        patch_size=8,
        num_heads=2,
        depths=[1],
        num_classes=2,
        use_local_pos_embed=True,
        local_pos_embed_config_patch={
            "strategy": "period",
            "base": base,
            "normalize_coords": "separate",
            "dtype": jnp.float32,
        },
        key=jr.PRNGKey(0),
    )


def test_lora_delta_saved_with_stored_rope_frequencies_loads(tmp_path):
    model = _rope_vision_transformer()
    old_base = eqx.tree_at(
        lambda base: base.local_pos_embed.patch_rope,
        model,
        _StoredFreqs(model.local_pos_embed.patch_rope.freqs),
    )
    old_lora = eqft.apply_lora(old_base, eqft.LoRAConfig(rank=2), key=jr.PRNGKey(1))
    path = tmp_path / "delta.eqft"
    bundle = eqft.save_delta(old_lora, path)
    assert bundle.architecture_hash and bundle.base_checkpoint_id

    loaded = eqft.load_delta(model, path)

    assert jnp.array_equal(
        loaded.blocks[0].blocks[0].attn.qkv.lora_A,
        old_lora.blocks[0].blocks[0].attn.qkv.lora_A,
    )


def test_lora_delta_incompatible_shape_raises(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    lora = _trained_like_lora(model)
    incompatible = TinyVisionTransformer(dim=5, hidden_dim=10, key=jr.PRNGKey(1))
    path = tmp_path / "delta.eqft"
    eqft.save_delta(lora, path)

    with pytest.raises(eqft.FineTuneBundleError, match="architecture hash mismatch"):
        eqft.load_delta(incompatible, path)


def test_lora_delta_rejects_a_base_with_other_rope_periods(tmp_path):
    model = _rope_vision_transformer()
    other = _rope_vision_transformer(base=1000.0)
    lora = eqft.apply_lora(model, eqft.LoRAConfig(rank=2), key=jr.PRNGKey(1))
    path = tmp_path / "delta.eqft"
    eqft.save_delta(lora, path)

    arrays = [
        jtu.tree_leaves(eqx.filter(base, eqx.is_array)) for base in (model, other)
    ]
    assert all(jnp.array_equal(a, b) for a, b in zip(*arrays, strict=True))
    with pytest.raises(eqft.FineTuneBundleError, match="base checkpoint mismatch"):
        eqft.load_delta(other, path)


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

    bundle = eqft.save_delta(adapted, path, base_model=base, spec=spec, method="lora")
    loaded = eqft.load_delta(base, path)

    assert bundle.delta_tree
    assert jnp.array_equal(loaded.head.weight, adapted.head.weight)
    assert jnp.array_equal(
        loaded.blocks[0].attn.proj.base.weight,
        adapted.blocks[0].attn.proj.base.weight,
    )
