"""ONNX export coverage for every registered Equimo model family."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from jax2onnx import to_onnx

from equimo.core.ops import softmax

from cases.model_cases import MODEL_CASES, extract_features

# MODEL_CASES covers every registered built-in family; that completeness is
# enforced by tests/finetune/test_feature_spec_model_coverage.py.


@pytest.mark.parametrize("variant", ["plain", "vmap", "where"])
def test_softmax_exports_the_jax_softmax_graph(variant):
    x = jnp.array([[30.0, 0.0, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    mask = jnp.array([[True, True, False], [False, True, True], [False] * 3])

    def call(fn):
        if variant == "vmap":
            return jax.vmap(lambda z: fn(z))
        if variant == "where":
            return lambda z: fn(z, where=mask)
        return lambda z: fn(z)

    # jax2onnx swaps in its own jax.nn.softmax during export, so the reference
    # must look it up at call time, as Equimo's softmax does.
    model, reference = (
        to_onnx(call(fn), inputs=[x], opset=18, input_names=["x"], output_names=["y"])
        for fn in (softmax, lambda z, **kwargs: jax.nn.softmax(z, **kwargs))
    )
    onnx.checker.check_model(model)
    # The custom derivative must not add anything to an inference graph.
    assert [node.op_type for node in model.graph.node] == [
        node.op_type for node in reference.graph.node
    ]
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(["y"], {"x": np.asarray(x)})
    np.testing.assert_allclose(actual, call(jax.nn.softmax)(x), rtol=1e-6, atol=0)


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.registry_name)
def test_builtin_model_family_exports_to_onnx_with_runtime_parity(case):
    invocation = case.build(jr.PRNGKey(40))
    export_args = (
        [jnp.expand_dims(value, 0) for value in invocation.args]
        if case.onnx_batched
        else list(invocation.args)
    )

    def inference(*args):
        if case.onnx_batched:
            return jax.vmap(
                lambda *sample: extract_features(
                    invocation,
                    *sample,
                    key=invocation.key,
                )
            )(*args)
        return extract_features(invocation, *args, key=invocation.key)

    input_names = [f"input_{index}" for index in range(len(invocation.args))]
    model_proto = to_onnx(
        inference,
        inputs=export_args,
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
            for name, value in zip(input_names, export_args, strict=True)
        },
    )
    expected = np.asarray(inference(*export_args))

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
