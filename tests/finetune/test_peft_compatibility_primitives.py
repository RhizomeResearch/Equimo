"""Golden tests for compatibility-critical PEFT primitives."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune.merging as merging
import equimo.finetune.serialization as serialization
from equimo.finetune import FineTuneBundleError
from equimo.finetune.peft import adapters, lora

from fixtures import TinyVisionTransformer


def test_compatibility_hashes_are_stable():
    model = TinyVisionTransformer(depth=1, key=jr.PRNGKey(0))

    assert (
        lora.architecture_hash(model)
        == "51528491e73c10ab36f4e6b632bc09be0aae343c611afcea7fca27b0c5a5dc94"
    )
    assert (
        merging._architecture_hash(model, include_head=True)
        == "51528491e73c10ab36f4e6b632bc09be0aae343c611afcea7fca27b0c5a5dc94"
    )
    assert (
        merging._architecture_hash(model, include_head=False)
        == "690a4ed8229aea35685a9d35b3c4050320e492a143b6f4740ed404ff106e6814"
    )
    assert (
        merging._checkpoint_hash(model, include_head=True)
        == "b73f5bd0a2e13aaf5bb7db4a8f9bf2ab39512274dc2c75b38ef064e02be65171"
    )
    assert (
        merging._checkpoint_hash(model, include_head=False)
        == "6f1174022a49dc9006dcfc659fc17cc84f6113cdc7c351b6772a004138361923"
    )
    assert (
        serialization._checkpoint_hash(model)
        == "sha256:b73f5bd0a2e13aaf5bb7db4a8f9bf2ab39512274dc2c75b38ef064e02be65171"
    )


def test_logical_id_hashes_are_stable():
    model = TinyVisionTransformer(depth=1, key=jr.PRNGKey(0))

    assert (
        merging._logical_id_table_hash(model, include_head=True)
        == "c86107fa0687ee831f4492e497e2c1d186af2c7733b59188e88dd80ad9c52c1f"
    )
    assert (
        merging._logical_id_table_hash(model, include_head=False)
        == "bdfb47f3bf7e419bcf88cb229f040c12f10227c25f8da02a806cf11ddfda0e64"
    )


@pytest.mark.parametrize(
    "get_path",
    (lora._bundle_get_path, adapters._bundle_get_path, serialization._bundle_get_path),
)
def test_missing_bundle_path_error_is_stable(get_path):
    model = TinyVisionTransformer(depth=1, key=jr.PRNGKey(0))

    with pytest.raises(FineTuneBundleError) as exc_info:
        get_path(
            model,
            ("blocks", 99, "attn", "proj"),
            method_name="TestPEFT",
        )

    assert str(exc_info.value) == (
        "TestPEFT delta expects path blocks.99.attn.proj, "
        "but the base model has no matching leaf."
    )


@pytest.mark.parametrize(
    ("to_state", "from_state"),
    (
        (adapters._linear_state, adapters._linear_from_state),
        (serialization._linear_state, serialization._linear_from_state),
    ),
)
def test_linear_state_roundtrip_is_stable(to_state, from_state):
    linear = eqx.nn.Linear(3, 2, use_bias=True, key=jr.PRNGKey(0))

    state = to_state(linear)
    restored = from_state(state)

    assert state["in_features"] == 3
    assert state["out_features"] == 2
    assert state["use_bias"] is True
    assert jnp.array_equal(restored.weight, linear.weight)
    assert jnp.array_equal(restored.bias, linear.bias)


class _InvalidLinear:
    weight = jnp.ones((2,), dtype=jnp.float32)
    bias = "not-an-array"

    def __call__(self, x):
        return x


@pytest.mark.parametrize(
    "linear_weight", (lora._linear_weight, adapters._linear_weight)
)
def test_linear_weight_validation_error_is_stable(linear_weight):
    with pytest.raises(TypeError) as exc_info:
        linear_weight(_InvalidLinear())

    assert str(exc_info.value) == "_InvalidLinear is not a linear-like module."


@pytest.mark.parametrize("linear_bias", (lora._linear_bias, adapters._linear_bias))
def test_linear_bias_validation_error_is_stable(linear_bias):
    with pytest.raises(TypeError) as exc_info:
        linear_bias(_InvalidLinear())

    assert str(exc_info.value) == "_InvalidLinear.bias is not an inexact array."
