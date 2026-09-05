"""Preservation contracts for additive LoRA lifecycles and mixed bundles."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


FAMILIES = (
    ("LoRA", eqft.apply_lora, eqft.LoRAConfig(rank=2), "lora_B"),
    ("LoRA-FA", eqft.apply_lora_fa, eqft.LoRAFAConfig(rank=2), "lora_fa_B"),
    ("RandLoRA", eqft.apply_randlora, eqft.RandLoRAConfig(), "basis_scales"),
    (
        "FourierFT",
        eqft.apply_fourierft,
        eqft.FourierFTConfig(num_coefficients=2),
        "coefficients_real",
    ),
    ("AdaLoRA", eqft.apply_adalora, eqft.AdaLoRAConfig(rank=2), "singular"),
)


def _adapt_linear(base, family):
    _, apply, config, leaf_name = family
    config = replace(config, target=eqft.TargetSpec(include=("linear.weight",)))
    wrapper = apply({"linear": base}, config, key=jr.PRNGKey(1))["linear"]
    return eqx.tree_at(
        lambda module: getattr(module, leaf_name),
        wrapper,
        jnp.full_like(getattr(wrapper, leaf_name), 0.125),
    )


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family[0])
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_additive_lifecycle_preserves_output_dtype_and_errors(family, dtype):
    base = eqx.nn.Linear(4, 4, key=jr.PRNGKey(0))
    base = jax.tree.map(lambda value: value.astype(dtype), base)
    wrapper = _adapt_linear(base, family)
    x = jnp.asarray([0.5, -1.0, 1.5, -0.25], dtype=dtype)
    expected = wrapper(x)
    merged = wrapper.merge()
    restored = merged.unmerge()
    tolerance = 0.02 if dtype == jnp.bfloat16 else 1e-6

    assert wrapper.unmerge() is wrapper
    assert type(merged) is type(restored) is type(wrapper)
    assert merged.base.weight.dtype == restored.base.weight.dtype == dtype
    assert jnp.allclose(merged(x), expected, atol=tolerance, rtol=tolerance)
    assert jnp.allclose(restored(x), expected, atol=tolerance, rtol=tolerance)
    with pytest.raises(ValueError) as error:
        merged.merge()
    assert str(error.value) == f"{family[0]} module is already merged."


def test_unmergeable_policy_is_checked_before_merged_state():
    wrapper = eqft.LoRALinear(
        eqx.nn.Linear(4, 4, key=jr.PRNGKey(0)),
        rank=2,
        alpha=4.0,
        scaling="alpha_over_r",
        dropout=0.0,
        train_base=False,
        mergeable=False,
        merged=True,
        key=jr.PRNGKey(1),
    )
    with pytest.raises(ValueError) as error:
        wrapper.merge()
    assert str(error.value) == "This LoRA module is not mergeable."


@pytest.mark.parametrize("merged", (False, True))
def test_heterogeneous_lora_bundle_restores_each_family(merged):
    base = {name: eqx.nn.Linear(4, 4, key=jr.PRNGKey(0)) for name, *_ in FAMILIES}
    adapted = {family[0]: _adapt_linear(base[family[0]], family) for family in FAMILIES}
    if merged:
        adapted = eqft.merge_lora(adapted)
    bundle = eqft.extract_lora_delta(adapted)
    loaded = eqft.load_lora_delta(base, bundle)
    x = jnp.ones((4,))

    assert len(bundle.adapter_config["entries"]) == len(FAMILIES)
    for name, wrapper in adapted.items():
        assert type(loaded[name]) is type(wrapper)
        assert loaded[name].merged is merged
        assert jnp.allclose(loaded[name](x), wrapper(x), atol=1e-6)
