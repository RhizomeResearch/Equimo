"""FeatureSpec conformance coverage for every built-in model family."""

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import equimo.finetune as eqft
from equimo.registry import _MODEL_REGISTRY
from equimo.vision.models import VisionTransformer
from cases.model_cases import (
    MODEL_CASES,
    TRANSFORM_CASES,
    assert_gradients_and_low_precision,
    assert_jit_and_vmap,
    extract_features,
)


MODEL_REFERENCES = json.loads(
    (Path(__file__).parents[1] / "data" / "model_output_references.json").read_text(
        encoding="utf-8"
    )
)


def test_every_builtin_model_family_has_a_conformance_case():
    registered = {
        (modality, name)
        for name, entries in _MODEL_REGISTRY.items()
        for modality, model_cls in entries.items()
        if model_cls.__module__.startswith("equimo.")
    }
    covered = {(case.modality, case.registry_name) for case in MODEL_CASES}

    assert covered == registered
    assert all(
        (case.low_precision_limitation is None) == case.supports_low_precision
        for case in MODEL_CASES
    )
    assert all(
        (case.onnx_batch_limitation is None) == case.onnx_batched
        for case in MODEL_CASES
    )


@pytest.mark.parametrize(
    ("index", "case"),
    enumerate(MODEL_CASES),
    ids=[case.registry_name for case in MODEL_CASES],
)
def test_builtin_model_family_executes_declared_feature_spec(index, case):
    invocation = case.build(jr.PRNGKey(MODEL_REFERENCES["seed_base"] + index))
    result = extract_features(invocation, *invocation.args, key=invocation.key)
    reference = MODEL_REFERENCES["cases"][case.registry_name]

    assert result.shape == invocation.expected.shape
    assert result.dtype == invocation.expected.dtype
    assert list(result.shape) == reference["shape"]
    assert str(result.dtype) == reference["dtype"]
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, invocation.expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result),
        np.asarray(reference["values"], dtype=reference["dtype"]),
        rtol=MODEL_REFERENCES["rtol"],
        atol=MODEL_REFERENCES["atol"],
    )


@pytest.mark.parametrize(
    "case",
    TRANSFORM_CASES,
    ids=lambda case: case.registry_name,
)
def test_feature_contract_groups_support_jit_and_vmap(case):
    invocation = case.build(jr.PRNGKey(30))
    assert_jit_and_vmap(invocation)
    assert_gradients_and_low_precision(
        invocation,
        supports_low_precision=case.supports_low_precision,
    )


def test_classless_vit_supports_patch_features_and_rejects_cls_selection():
    key = jr.PRNGKey(20)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[1],
        class_token=False,
        reg_tokens=0,
        global_pos_embed_cls=False,
        num_classes=0,
        key=key,
    )
    sample = jr.normal(jr.PRNGKey(21), (3, 16, 16))
    native = model.forward_features(sample, key=key, inference=True)
    patch_spec = eqft.FeatureSpec(
        "forward_features",
        "BNC",
        "patches",
        "mean_patch",
    )
    cls_spec = eqft.FeatureSpec("forward_features", "BNC", "cls", None)

    patches = eqft.extract_features(
        model,
        sample,
        feature_spec=patch_spec,
        key=key,
        inference=True,
    )

    assert jnp.allclose(
        patches,
        jnp.mean(native["x_norm_patchtokens"], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="requires x_norm_cls_token"):
        eqft.extract_features(
            model,
            sample,
            feature_spec=cls_spec,
            key=key,
            inference=True,
        )
