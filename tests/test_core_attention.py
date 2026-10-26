import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.core.layers import Attention, with_attention_implementation
from equimo.core.layers.attention import AttentionBlock, _fused_attention
from equimo.serialization import _model_signature
from equimo.tabular.layers.attention import InContextAttention
from equimo.tabular.models import TabPFN
from equimo.vision.models import VisionTransformer


KEY = jr.PRNGKey(0)
GPU = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="cuDNN attention requires a CUDA GPU"
)


def _cast(tree, dtype):
    return jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf, tree
    )


def _attention(tree):
    def is_attention(node):
        return hasattr(node, "implementation")

    return [
        node
        for node in jax.tree.leaves(tree, is_leaf=is_attention)
        if is_attention(node)
    ]


def test_attention_block_split_masks_preserve_common_mask_behavior():
    block = AttentionBlock(dim=8, num_heads=2, key=KEY)
    x = jr.normal(KEY, (4, 8))
    mask = jnp.array([[1], [1], [0], [0]])

    common_output = block(x, mask=mask, key=KEY, inference=True)
    split_output = block(
        x,
        attn_mask=mask,
        ffn_mask=mask,
        key=KEY,
        inference=True,
    )

    assert jnp.allclose(common_output, split_output)


def test_attention_block_accepts_distinct_attention_and_ffn_masks():
    block = AttentionBlock(dim=8, num_heads=2, key=KEY)
    x = jr.normal(KEY, (4, 8))
    attn_mask = jnp.array([[[1, 1, 0, 0]]])
    ffn_mask = jnp.array([[1], [1], [0], [0]])

    output = block(
        x,
        attn_mask=attn_mask,
        ffn_mask=ffn_mask,
        key=KEY,
        inference=True,
    )

    assert output.shape == x.shape
    assert jnp.all(jnp.isfinite(output))


@pytest.mark.parametrize("kv_heads", (2, 1))
@pytest.mark.parametrize("masked", (False, True))
def test_fused_attention_padding_and_masks_match_unfused_attention(kv_heads, masked):
    # implementation="xla" checks the cuDNN padding and masking on any backend.
    keys = jr.split(KEY, 4)
    q = jr.normal(keys[0], (3, 2, 5, 8))
    k = jr.normal(keys[1], (3, kv_heads, 7, 8))
    v = jr.normal(keys[2], (3, kv_heads, 7, 8))
    w = jr.normal(keys[3], q.shape)
    allowed = jnp.arange(5)[:, None] >= jnp.arange(7) - 2
    allowed = jnp.broadcast_to(allowed.at[1].set(False), (3, 5, 7))

    def fused(q, k, v, allowed):
        return _fused_attention(
            q, k, v, allowed if masked else None, implementation="xla"
        )

    def reference(q, k, v, allowed):
        k, v = (jnp.repeat(x, 2 // kv_heads, axis=0) for x in (k, v))
        logits = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(8.0)
        weights = jax.nn.softmax(logits, where=allowed if masked else None)
        return jnp.einsum("hqk,hkd->hqd", weights, v)

    def evaluate(attention):
        batched = jax.vmap(attention)
        loss = lambda q, k, v: jnp.sum(batched(q, k, v, allowed) * w)
        return batched(q, k, v, allowed), jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    # GPUs default to TF32 matmuls, which would dominate the comparison.
    with jax.default_matmul_precision("highest"):
        output, gradients = evaluate(fused)
        expected_output, expected_gradients = evaluate(reference)

    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if masked:
        np.testing.assert_array_equal(output[:, :, 1], 0)


def test_attention_implementation_switches_modules_and_keeps_parameters():
    vit = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=16,
        patch_size=8,
        num_heads=2,
        depths=[2],
        num_classes=0,
        key=KEY,
    )
    tabpfn = TabPFN(
        num_classes=4,
        dim=16,
        depths=(1, 1, 1),
        num_heads=(2, 2, 2),
        num_inducing_points=4,
        num_cls_tokens=2,
        decoder_head_dim=8,
        decoder_num_heads=2,
        key=KEY,
    )
    assert any(isinstance(node, InContextAttention) for node in _attention(tabpfn))
    for model in (vit, tabpfn):
        switched = with_attention_implementation(model, "cudnn")

        assert {node.implementation for node in _attention(model)} == {"xla"}
        assert {node.implementation for node in _attention(switched)} == {"cudnn"}
        assert jax.tree.structure(switched) != jax.tree.structure(model)
        assert _model_signature(switched) == _model_signature(model)
        for before, after in zip(
            jax.tree.leaves(model), jax.tree.leaves(switched), strict=True
        ):
            assert before is after
        restored = with_attention_implementation(switched, "xla")
        assert jax.tree.structure(restored) == jax.tree.structure(model)

    with pytest.raises(ValueError, match="implementation must be one of"):
        with_attention_implementation(vit, "flash")
    with pytest.raises(ValueError, match="implementation must be one of"):
        Attention(8, 2, implementation="flash", key=KEY)


def test_cudnn_attention_rejects_unsupported_inputs():
    model = Attention(8, 2, attn_drop=0.1, implementation="cudnn", key=KEY)
    x = jr.normal(KEY, (4, 8))

    with pytest.raises(ValueError, match="bfloat16 or float16"):
        model(x, key=None, inference=True)
    with pytest.raises(ValueError, match="attention dropout"):
        _cast(model, jnp.bfloat16)(x.astype(jnp.bfloat16), key=KEY, inference=False)


@GPU
@pytest.mark.parametrize("masked", (False, True))
def test_cudnn_attention_matches_xla_attention(masked):
    model = _cast(Attention(128, 2, key=KEY), jnp.bfloat16)
    fused = with_attention_implementation(model, "cudnn")
    x = jr.normal(jr.PRNGKey(1), (3, 20, 128), jnp.bfloat16)
    mask = jnp.broadcast_to(jnp.tril(jnp.ones((20, 20), bool)), (3, 20, 20))
    mask = mask.at[:, 4].set(False)
    w = jr.normal(jr.PRNGKey(2), x.shape, jnp.float32)

    def evaluate(model):
        call = jax.vmap(
            lambda x, mask: model(
                x, key=None, inference=True, mask=mask if masked else None
            )
        )
        loss = lambda model, x: jnp.sum(call(x, mask).astype(jnp.float32) * w)
        output = jax.jit(call)(x, mask)
        gradients = eqx.filter_jit(eqx.filter_grad(loss))(model, x)
        return output, gradients

    expected_output, expected_gradients = evaluate(model)
    output, gradients = evaluate(fused)

    assert output.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        output.astype(jnp.float32),
        expected_output.astype(jnp.float32),
        rtol=0.05,
        atol=0.02,
    )
    for actual, expected in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(expected_gradients), strict=True
    ):
        actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
        assert jnp.linalg.norm(actual - expected) <= 0.03 * jnp.linalg.norm(expected)


@GPU
def test_cudnn_in_context_attention_matches_xla_attention():
    module = _cast(
        InContextAttention(128, 2, num_kv_heads_test=1, key=KEY), jnp.bfloat16
    )
    fused = with_attention_implementation(module, "cudnn")
    x = jr.normal(jr.PRNGKey(1), (30, 128), jnp.bfloat16)

    expected = jax.jit(lambda x: module(x, 20))(x).astype(jnp.float32)
    actual = jax.jit(lambda x: fused(x, 20))(x).astype(jnp.float32)

    assert jnp.linalg.norm(actual - expected) <= 0.02 * jnp.linalg.norm(expected)
