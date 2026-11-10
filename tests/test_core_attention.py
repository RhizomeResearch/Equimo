import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.core.layers import (
    Attention,
    CrossAttention,
    with_attention_implementation,
)
from equimo.core.layers.attention import AttentionBlock, _fused_attention
from equimo.core.layers.rotary import RotaryFactors
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
@pytest.mark.parametrize("biased", (False, True))
def test_fused_attention_padding_and_masks_match_unfused_attention(
    kv_heads, masked, biased
):
    # implementation="xla" checks the cuDNN padding and masking on any backend.
    keys = jr.split(KEY, 5)
    q = jr.normal(keys[0], (3, 2, 5, 8))
    k = jr.normal(keys[1], (3, kv_heads, 7, 8))
    v = jr.normal(keys[2], (3, kv_heads, 7, 8))
    w = jr.normal(keys[3], q.shape)
    bias = jr.normal(keys[4], (3, 2, 5, 7)) if biased else jnp.zeros((3, 2, 5, 7))
    allowed = jnp.arange(5)[:, None] >= jnp.arange(7) - 2
    allowed = jnp.broadcast_to(allowed.at[1].set(False), (3, 5, 7))

    def fused(q, k, v, bias, allowed):
        return _fused_attention(
            q,
            k,
            v,
            allowed if masked else None,
            bias=bias if biased else None,
            implementation="xla",
        )

    def reference(q, k, v, bias, allowed):
        k, v = (jnp.repeat(x, 2 // kv_heads, axis=0) for x in (k, v))
        logits = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(8.0) + bias
        weights = jax.nn.softmax(logits, where=allowed if masked else None)
        return jnp.einsum("hqk,hkd->hqd", weights, v)

    def evaluate(attention):
        batched = jax.vmap(attention)
        loss = lambda q, k, v, bias: jnp.sum(batched(q, k, v, bias, allowed) * w)
        # Without a bias the fused call ignores it, so its gradient is not compared.
        argnums = (0, 1, 2, 3) if biased else (0, 1, 2)
        return batched(q, k, v, bias, allowed), jax.grad(loss, argnums=argnums)(
            q, k, v, bias
        )

    # GPUs default to TF32 matmuls, which would dominate the comparison.
    with jax.default_matmul_precision("highest"):
        output, gradients = evaluate(fused)
        expected_output, expected_gradients = evaluate(reference)

    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if masked:
        np.testing.assert_array_equal(output[:, :, 1], 0)


def test_fused_attention_key_lengths_match_dense_key_mask():
    # Packed keys: only the first `length` keys of each entry are valid.
    keys = jr.split(KEY, 4)
    q = jr.normal(keys[0], (3, 2, 5, 8))
    k = jr.normal(keys[1], (3, 2, 70, 8))
    v = jr.normal(keys[2], (3, 2, 70, 8))
    w = jr.normal(keys[3], q.shape)
    bias = jr.normal(jr.PRNGKey(1), (2, 5, 70))
    lengths = jnp.array([70, 9, 0])
    allowed = jnp.arange(70) < lengths[:, None, None, None]

    def evaluate(**kwargs):
        def attention(q, k, v):
            return _fused_attention(q, k, v, bias=bias, implementation="xla", **kwargs)

        loss = lambda q, k, v: jnp.sum(attention(q, k, v) * w)
        return attention(q, k, v), jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    with jax.default_matmul_precision("highest"):
        output, gradients = evaluate(key_value_seq_lengths=lengths)
        expected_output, expected_gradients = evaluate(allowed=allowed)

    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(output[2], 0)
    np.testing.assert_array_equal(gradients[2][2], 0)


def test_fused_attention_key_lengths_follow_vmap_and_reject_masks():
    keys = jr.split(KEY, 3)
    q = jr.normal(keys[0], (2, 2, 4, 8))
    k = jr.normal(keys[1], (2, 2, 6, 8))
    v = jr.normal(keys[2], (2, 2, 6, 8))
    lengths = jnp.array([3, 0])

    batched = _fused_attention(
        q, k, v, key_value_seq_lengths=lengths, implementation="xla"
    )
    mapped = jax.vmap(
        lambda q, k, v, length: _fused_attention(
            q, k, v, key_value_seq_lengths=length, implementation="xla"
        )
    )(q, k, v, lengths)

    np.testing.assert_array_equal(mapped, batched)
    np.testing.assert_array_equal(batched[1], 0)
    with pytest.raises(ValueError, match="not both"):
        _fused_attention(
            q,
            k,
            v,
            jnp.ones((2, 2, 4, 6), bool),
            key_value_seq_lengths=lengths,
            implementation="xla",
        )


def _numpy_cross_attention(
    module,
    x,
    context,
    key_mask=None,
    bias=None,
    *,
    values=None,
    rotary=None,
    key_rotary=None,
):
    """Float64 multi-head attention written directly from the definition."""

    def linear(layer, tokens):
        weight = np.asarray(layer.weight, np.float64)
        return np.asarray(tokens, np.float64) @ weight.T + np.asarray(layer.bias)

    heads, head_dim = module.num_heads, module.head_dim

    def split(tokens):
        return tokens.reshape(tokens.shape[0], heads, head_dim).transpose(1, 0, 2)

    def rotate(tokens, factors):
        pairs = tokens.reshape(*tokens.shape[:-1], -1, 2)
        turned = np.stack((-pairs[..., 1], pairs[..., 0]), axis=-1)
        sin, cos = (np.asarray(table, np.float64)[None] for table in factors)
        return tokens * cos + turned.reshape(tokens.shape) * sin

    q = split(linear(module.q, x))
    k = split(linear(module.k, context))
    v = split(linear(module.v, context if values is None else values))
    if rotary is not None:
        q = rotate(q, (rotary.sin, rotary.cos))
    if key_rotary is not None:
        k = rotate(k, (key_rotary.sin, key_rotary.cos))
    logits = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    if bias is not None:
        logits = logits + np.asarray(bias, np.float64)
    allowed = np.ones(logits.shape, bool)
    if key_mask is not None:
        allowed = np.broadcast_to(np.asarray(key_mask, bool), logits.shape)
    logits = np.where(allowed, logits, -np.inf)
    peak = np.max(logits, axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    unnormalized = np.where(allowed, np.exp(logits - peak), 0.0)
    total = unnormalized.sum(axis=-1, keepdims=True)
    weights = unnormalized / np.where(total > 0, total, 1.0)
    out = (weights @ v).transpose(1, 0, 2).reshape(x.shape[0], -1)
    return linear(module.proj, out)


def test_cross_attention_matches_numpy_reference_with_masks_and_bias():
    module = CrossAttention(16, 2, kv_dim=12, internal_dim=8, key=KEY)
    keys = jr.split(jr.PRNGKey(1), 3)
    x = jr.normal(keys[0], (5, 16))
    context = jr.normal(keys[1], (7, 12))
    bias = jr.normal(keys[2], (2, 5, 7))
    key_mask = jnp.array([True, False, True, True, False, True, True])
    # The third query attends to no key.
    row_mask = jnp.broadcast_to(key_mask, (5, 7)).at[2].set(False)

    assert module.head_dim == 4
    assert module.q.weight.shape == (8, 16)
    assert module.k.weight.shape == (8, 12)
    assert module.proj.weight.shape == (16, 8)
    with jax.default_matmul_precision("highest"):
        for kwargs in ({}, {"key_mask": key_mask}, {"bias": bias}):
            np.testing.assert_allclose(
                module(x, context, **kwargs),
                _numpy_cross_attention(module, x, context, **kwargs),
                rtol=1e-5,
                atol=1e-5,
            )
        masked = module(x, context, key_mask=row_mask, bias=bias)
        np.testing.assert_allclose(
            masked,
            _numpy_cross_attention(module, x, context, row_mask, bias),
            rtol=1e-5,
            atol=1e-5,
        )
    # A fully masked row attends to nothing, leaving only the output bias.
    np.testing.assert_allclose(masked[2], module.proj.bias, rtol=1e-6)

    gradient = jax.grad(
        lambda context: jnp.sum(module(x, context, key_mask=row_mask, bias=bias))
    )(context)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_array_equal(gradient[jnp.array([1, 4])], 0)


def test_cross_attention_separates_values_and_rotates_queries_and_keys():
    module = CrossAttention(16, 2, key=KEY)
    keys = jr.split(jr.PRNGKey(2), 7)
    x = jr.normal(keys[0], (5, 16))
    context = jr.normal(keys[1], (7, 16))
    values = jr.normal(keys[2], (7, 16))

    def factors(sin_key, cos_key, length):
        return RotaryFactors(
            jr.normal(sin_key, (length, 8)),
            jr.normal(cos_key, (length, 8)),
            layout="interleaved",
        )

    rotary = factors(keys[3], keys[4], 5)
    key_rotary = factors(keys[5], keys[6], 7)
    with jax.default_matmul_precision("highest"):
        for kwargs in (
            {"values": values},
            {"rotary": rotary},
            {"key_rotary": key_rotary},
            {"values": values, "rotary": rotary, "key_rotary": key_rotary},
        ):
            np.testing.assert_allclose(
                module(x, context, **kwargs),
                _numpy_cross_attention(module, x, context, **kwargs),
                rtol=1e-4,
                atol=1e-4,
            )


def test_cross_attention_keeps_low_precision_and_batches_with_vmap():
    module = _cast(CrossAttention(16, 4, key=KEY), jnp.bfloat16)
    x = jr.normal(KEY, (3, 5, 16), jnp.bfloat16)
    context = jr.normal(jr.PRNGKey(1), (3, 9, 16), jnp.bfloat16)
    key_mask = jnp.arange(9) < jnp.array([9, 4, 1])[:, None]

    output = jax.jit(
        jax.vmap(lambda x, context, mask: module(x, context, key_mask=mask))
    )(x, context, key_mask)

    assert output.shape == (3, 5, 16)
    assert output.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(output.astype(jnp.float32)))


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

    cross = CrossAttention(8, 2, key=KEY)
    switched = with_attention_implementation(cross, "cudnn")
    assert (cross.implementation, switched.implementation) == ("xla", "cudnn")
    assert _model_signature(switched) == _model_signature(cross)
    with pytest.raises(ValueError, match="implementation must be one of"):
        CrossAttention(8, 2, implementation="flash", key=KEY)
    with pytest.raises(ValueError, match="divisible by num_heads"):
        CrossAttention(8, 3, key=KEY)


def test_cudnn_attention_rejects_unsupported_inputs():
    model = Attention(8, 2, attn_drop=0.1, implementation="cudnn", key=KEY)
    x = jr.normal(KEY, (4, 8))

    with pytest.raises(ValueError, match="bfloat16 or float16"):
        model(x, key=None, inference=True)
    with pytest.raises(ValueError, match="attention dropout"):
        _cast(model, jnp.bfloat16)(x.astype(jnp.bfloat16), key=KEY, inference=False)

    cross = CrossAttention(8, 2, implementation="cudnn", key=KEY)
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        cross(x, x)


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
def test_cudnn_packed_keys_match_xla_dense_mask_under_vmap():
    keys = jr.split(jr.PRNGKey(1), 4)
    q = jr.normal(keys[0], (3, 2, 20, 64), jnp.bfloat16)
    k = jr.normal(keys[1], (3, 2, 90, 64), jnp.bfloat16)
    v = jr.normal(keys[2], (3, 2, 90, 64), jnp.bfloat16)
    w = jr.normal(keys[3], q.shape, jnp.float32)
    lengths = jnp.array([90, 33, 0])

    def evaluate(attention):
        call = jax.vmap(attention)
        loss = lambda q, k, v: jnp.sum(call(q, k, v, lengths).astype(jnp.float32) * w)
        output = jax.jit(call)(q, k, v, lengths)
        return output, jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)

    def packed(q, k, v, length):
        return _fused_attention(q, k, v, key_value_seq_lengths=length)

    def dense(q, k, v, length):
        allowed = jnp.arange(90) < length
        return _fused_attention(q, k, v, allowed, implementation="xla")

    output, gradients = evaluate(packed)
    expected_output, expected_gradients = evaluate(dense)

    np.testing.assert_array_equal(output[2].astype(jnp.float32), 0)
    np.testing.assert_allclose(
        output.astype(jnp.float32),
        expected_output.astype(jnp.float32),
        rtol=0.05,
        atol=0.02,
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.linalg.norm(actual - expected) <= 0.03 * jnp.linalg.norm(expected)


@GPU
def test_cudnn_cross_attention_with_bias_matches_xla_under_vmap():
    # A batched bias that depends on trained inputs, as the DETR box bias does.
    model = _cast(CrossAttention(128, 2, kv_dim=64, key=KEY), jnp.bfloat16)
    fused = with_attention_implementation(model, "cudnn")
    keys = jr.split(jr.PRNGKey(1), 4)
    x = jr.normal(keys[0], (3, 20, 128), jnp.bfloat16)
    context = jr.normal(keys[1], (3, 90, 64), jnp.bfloat16)
    bias = jr.normal(keys[2], (3, 2, 20, 90), jnp.float32)
    key_mask = jnp.arange(90) < jnp.array([90, 41, 7])[:, None]
    w = jr.normal(keys[3], x.shape, jnp.float32)

    def evaluate(model):
        call = jax.vmap(
            lambda x, context, bias, mask: model(x, context, key_mask=mask, bias=bias)
        )
        loss = lambda model, x, bias: jnp.sum(
            call(x, context, bias, key_mask).astype(jnp.float32) * w
        )
        output = jax.jit(call)(x, context, bias, key_mask)
        gradients = eqx.filter_jit(eqx.filter_grad(loss))(model, x, bias)
        bias_gradient = jax.jit(jax.grad(loss, argnums=2))(model, x, bias)
        return output, gradients, bias_gradient

    expected_output, expected_gradients, expected_bias_gradient = evaluate(model)
    output, gradients, bias_gradient = evaluate(fused)

    assert output.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        output.astype(jnp.float32),
        expected_output.astype(jnp.float32),
        rtol=0.05,
        atol=0.02,
    )
    for actual, expected in zip(
        jax.tree.leaves((gradients, bias_gradient)),
        jax.tree.leaves((expected_gradients, expected_bias_gradient)),
        strict=True,
    ):
        actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
        assert jnp.all(jnp.isfinite(actual))
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
