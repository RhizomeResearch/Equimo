"""Regression coverage for PRNG-free deterministic inference."""

from importlib.metadata import version

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import equimo.finetune as eqft
from equimo.core._prng import fold_in_for_mode, split_for_mode
from equimo.vision.models import (
    VisionParcae,
    dinov3_vits16_pretrain_lvd1689m,
    mobilenetv3_small,
)
from _jaxpr_utils import assert_prng_free_jaxpr, prng_primitive_names
from finetune.test_feature_spec_model_coverage import CASES


def _legacy_mobilenet_call(model, image, *, key, inference):
    """MobileNetV3's key schedule before static-inference elision."""

    key_features, key_dropout = jr.split(key, 2)
    key_stem, key_layers = jr.split(key_features, 2)
    x = model.conv1(image, key=key_stem, inference=inference)
    for index, layer in enumerate(model.layers):
        x = layer(
            x,
            key=jr.fold_in(key_layers, index),
            inference=inference,
        )
    x = jnp.mean(x, axis=(1, 2))
    x = model.dropout(x, key=key_dropout, inference=inference)
    return model.classifier(x)


def test_mode_helpers_preserve_training_schedule_and_elide_static_inference():
    key = jr.PRNGKey(17)

    expected = tuple(jr.split(key, 4))
    actual_false = split_for_mode(key, 4, inference=False)
    actual_none = split_for_mode(key, 4, inference=None)
    assert all(jnp.array_equal(x, y) for x, y in zip(actual_false, expected))
    assert all(jnp.array_equal(x, y) for x, y in zip(actual_none, expected))
    assert jnp.array_equal(
        fold_in_for_mode(key, 3, inference=False),
        jr.fold_in(key, 3),
    )

    assert_prng_free_jaxpr(
        lambda runtime_key: jnp.stack(split_for_mode(runtime_key, 4, inference=True)),
        key,
    )
    assert_prng_free_jaxpr(
        lambda runtime_key: fold_in_for_mode(runtime_key, 3, inference=True),
        key,
    )


def test_mobilenetv3_exact_export_contract_is_prng_free_and_value_preserving():
    model = mobilenetv3_small(
        num_classes=10,
        dropout=0.0,
        key=jr.PRNGKey(0),
    )
    key = jr.PRNGKey(1)
    batch = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)

    def inference(images):
        keys = jnp.broadcast_to(key, (images.shape[0], *key.shape))
        return jax.vmap(
            lambda image, image_key: model(image, key=image_key, inference=True)
        )(images, keys).astype(jnp.float32)

    def legacy(images):
        keys = jnp.broadcast_to(key, (images.shape[0], *key.shape))
        return jax.vmap(
            lambda image, image_key: _legacy_mobilenet_call(
                model,
                image,
                key=image_key,
                inference=True,
            )
        )(images, keys).astype(jnp.float32)

    logits = inference(batch)
    assert logits.shape == (1, 10)
    assert logits.dtype == jnp.float32
    assert jnp.array_equal(logits, legacy(batch))
    assert_prng_free_jaxpr(inference, batch)
    assert_prng_free_jaxpr(
        lambda image: model.features(image, key=key, inference=True),
        batch[0],
    )
    assert_prng_free_jaxpr(
        lambda image: model.intermediate_features(
            image, key=key, inference=True, n_last_blocks=1
        ),
        batch[0],
    )


def test_mobilenetv3_training_retains_legacy_key_schedule():
    model = mobilenetv3_small(
        num_classes=10,
        dropout=0.5,
        key=jr.PRNGKey(2),
    )
    key = jr.PRNGKey(3)
    image = jr.normal(jr.PRNGKey(4), (3, 64, 64))

    actual = model(image, key=key, inference=False)
    expected = _legacy_mobilenet_call(
        model,
        image,
        key=key,
        inference=False,
    )

    assert jnp.array_equal(actual, expected)
    assert prng_primitive_names(
        jax.make_jaxpr(
            lambda x, runtime_key: model(x, key=runtime_key, inference=False)
        )(image, key)
    )


def test_named_dinov3_vits16_static_inference_is_prng_free():
    model = dinov3_vits16_pretrain_lvd1689m(
        pretrained=False,
        key=jr.PRNGKey(9),
    )
    image = jnp.zeros((3, 64, 64), dtype=jnp.float32)

    def inference(value, runtime_key):
        return model.forward_features(value, key=runtime_key, inference=True)

    assert_prng_free_jaxpr(inference, image, jr.PRNGKey(10))
    output1 = inference(image, jr.PRNGKey(10))
    output2 = inference(image, jr.PRNGKey(11))
    assert output1["x_norm_cls_token"].shape == (384,)
    assert output1["x_norm_patchtokens"].shape == (16, 384)
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(output1),
            jax.tree.leaves(output2),
            strict=True,
        )
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.registry_name)
def test_every_deterministic_builtin_model_family_has_prng_free_inference(case):
    invocation = case.build(jr.PRNGKey(5))

    def inference(*args_and_key):
        *args, runtime_key = args_and_key
        return eqft.extract_features(
            invocation.model,
            *args,
            feature_spec=invocation.spec,
            key=runtime_key,
            inference=True,
            **invocation.kwargs,
        )

    assert_prng_free_jaxpr(
        inference,
        *invocation.args,
        invocation.key,
    )


@pytest.mark.parametrize("state_init", ("normal", "embed", "unit", "like-init"))
def test_vision_parcae_random_state_modes_remain_explicitly_stochastic(state_init):
    model = VisionParcae(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        n_layers_in_prelude=0,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=0,
        mean_recurrence=1,
        mean_backprop_depth=1,
        max_recurrence=1,
        state_init=state_init,
        num_classes=0,
        key=jr.PRNGKey(6),
    )
    embedded = jnp.ones((5, 8), dtype=jnp.float32)
    key1, key2 = jr.PRNGKey(7), jr.PRNGKey(8)

    initialize = lambda runtime_key: model._initialize_state(embedded, key=runtime_key)
    assert prng_primitive_names(jax.make_jaxpr(initialize)(key1))
    assert not jnp.array_equal(initialize(key1), initialize(key2))


def test_mobilenetv3_exports_to_onnx_without_randomness_plugins(tmp_path):
    pytest.importorskip("jax2onnx")
    pytest.importorskip("onnxruntime")
    if tuple(int(part) for part in version("jax2onnx").split(".")[:2]) < (0, 15):
        pytest.skip("Jax2Onnx 0.15.0 or newer is required")

    from jax2onnx import to_onnx
    import onnxruntime as ort

    model = mobilenetv3_small(
        num_classes=10,
        dropout=0.0,
        key=jr.PRNGKey(0),
    )
    key = jr.PRNGKey(1)
    dummy = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)
    sample = jr.normal(jr.PRNGKey(2), dummy.shape, dtype=jnp.float32)

    def inference(batch):
        keys = jnp.broadcast_to(key, (batch.shape[0], *key.shape))
        return jax.vmap(
            lambda image, image_key: model(image, key=image_key, inference=True)
        )(batch, keys).astype(jnp.float32)

    output_path = tmp_path / "mobilenetv3-small.onnx"
    to_onnx(
        inference,
        inputs=[dummy],
        opset=18,
        return_mode="file",
        output_path=str(output_path),
        input_names=["image"],
        output_names=["logits"],
    )

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    try:
        (ort_logits,) = session.run(["logits"], {"image": np.asarray(sample)})
    except Exception as error:
        message = str(error)
        if (
            version("jax2onnx") == "0.15.0"
            and "Input shape:{1,1,16,32,32}" in message
            and "requested shape:{1,16,16,16,32,32}" in message
        ):
            pytest.xfail("Jax2Onnx 0.15.0 mislowers vmapped equinox.nn.GroupNorm")
        raise
    np.testing.assert_allclose(
        ort_logits,
        np.asarray(inference(sample)),
        rtol=1e-4,
        atol=1e-5,
    )


def test_dinov3_vits16_exports_to_onnx_without_randomness_plugins(tmp_path):
    pytest.importorskip("jax2onnx")
    pytest.importorskip("onnxruntime")
    if tuple(int(part) for part in version("jax2onnx").split(".")[:2]) < (0, 15):
        pytest.skip("Jax2Onnx 0.15.0 or newer is required")

    from jax2onnx import to_onnx
    import onnxruntime as ort

    model = dinov3_vits16_pretrain_lvd1689m(
        pretrained=False,
        key=jr.PRNGKey(0),
    )
    key = jr.PRNGKey(1)
    dummy = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)

    def inference(batch):
        keys = jnp.broadcast_to(key, (batch.shape[0], *key.shape))
        return jax.vmap(
            lambda image, image_key: model.forward_features(
                image,
                key=image_key,
                inference=True,
            )["x_norm_cls_token"]
        )(batch, keys).astype(jnp.float32)

    output_path = tmp_path / "dinov3-vits16.onnx"
    try:
        to_onnx(
            inference,
            inputs=[dummy],
            opset=18,
            return_mode="file",
            output_path=str(output_path),
            input_names=["image"],
            output_names=["features"],
        )
    except ValueError as error:
        if version("jax2onnx") == "0.15.0" and str(error) == (
            "split sizes must be positive"
        ):
            pytest.xfail("Jax2Onnx 0.15.0 rejects DINOv3's zero-length prefix split")
        raise

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    (ort_features,) = session.run(["features"], {"image": np.asarray(dummy)})
    np.testing.assert_allclose(
        ort_features,
        np.asarray(inference(dummy)),
        rtol=1e-4,
        atol=1e-5,
    )
