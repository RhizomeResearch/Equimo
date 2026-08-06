"""ONNX export coverage for every registered Equimo model family."""

import jax.random as jr
import numpy as np
import pytest

import equimo.finetune as eqft
from equimo.registry import _MODEL_REGISTRY
from finetune.test_feature_spec_model_coverage import CASES


def test_every_builtin_model_family_has_an_onnx_export_case():
    registered = {
        (modality, name)
        for name, entries in _MODEL_REGISTRY.items()
        for modality, model_cls in entries.items()
        if model_cls.__module__.startswith("equimo.")
    }
    covered = {(case.modality, case.registry_name) for case in CASES}

    assert covered == registered


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.registry_name)
def test_builtin_model_family_exports_to_onnx_with_runtime_parity(case):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    to_onnx = pytest.importorskip("jax2onnx").to_onnx

    invocation = case.build(jr.PRNGKey(40))

    def inference(*args):
        return eqft.extract_features(
            invocation.model,
            *args,
            feature_spec=invocation.spec,
            key=invocation.key,
            inference=True,
            **invocation.kwargs,
        )

    input_names = [f"input_{index}" for index in range(len(invocation.args))]
    model_proto = to_onnx(
        inference,
        inputs=list(invocation.args),
        opset=18,
        input_names=input_names,
        output_names=["features"],
    )
    onnx.checker.check_model(model_proto)

    session = ort.InferenceSession(
        model_proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    (actual,) = session.run(
        ["features"],
        {
            name: np.asarray(value)
            for name, value in zip(input_names, invocation.args, strict=True)
        },
    )
    expected = np.asarray(inference(*invocation.args))

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
