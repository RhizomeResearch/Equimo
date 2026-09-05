import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.core.layers.ffn import Mlp as CoreMlp
from equimo.tabular import layers
from equimo.tabular.layers import attention, blocks, decoder, preprocessing, registry
from equimo.tabular.layers.mlp import _call_mlp
from _jaxpr_utils import assert_prng_free_jaxpr


def _incontext_reference(model, x, n_train):
    """Full-row projections and per-query KV selection, including test-head reuse."""
    heads, width = model.num_heads, model.head_dim
    q, k, v = (
        jax.vmap(projection)(x).reshape(x.shape[0], heads, width).transpose(1, 0, 2)
        for projection in (model.q_proj, model.k_proj, model.v_proj)
    )
    k, v = k[:, :n_train], v[:, :n_train]
    if model.softmax_scaling is not None:
        q = model.softmax_scaling(q, n_train)
    test_heads = model.num_kv_heads_test or heads
    head_indices = jnp.arange(heads) // (heads // test_heads)
    training_query = jnp.arange(x.shape[0])[None, :, None, None] < n_train
    keys = jnp.where(training_query, k[:, None], k[head_indices, None])
    values = jnp.where(training_query, v[:, None], v[head_indices, None])
    scores = jnp.einsum("hqd,hqkd->hqk", q, keys) / jnp.sqrt(width)
    out = jnp.einsum("hqk,hqkd->hqd", jax.nn.softmax(scores, axis=-1), values)
    return jax.vmap(model.proj)(out.transpose(1, 0, 2).reshape(x.shape[0], -1))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
@pytest.mark.parametrize(
    "n_train,kv_heads,scaled",
    ((1, None, False), (3, 1, False), (3, 2, True), (5, None, True), (5, 1, True)),
)
def test_incontext_attention_projection_parity(dtype, n_train, kv_heads, scaled):
    scaling = (
        attention.SoftmaxScaling(2, 4, hidden_dim=4, key=jr.PRNGKey(2))
        if scaled
        else None
    )
    model = attention.InContextAttention(
        8,
        2,
        key=jr.PRNGKey(0),
        num_kv_heads_test=kv_heads,
        softmax_scaling=scaling,
    )
    model = jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf, model
    )
    x = jr.normal(jr.PRNGKey(1), (5, 8), dtype=dtype)
    expected = _incontext_reference(model, x, n_train)
    tolerance = 0.02 if dtype == jnp.bfloat16 else 1e-5
    call = lambda m, value: m(value, n_train)
    for actual in (call(model, x), eqx.filter_jit(call)(model, x)):
        assert actual.shape == x.shape
        assert actual.dtype == dtype
        assert jnp.allclose(actual, expected, atol=tolerance, rtol=tolerance)
    batched = jax.vmap(lambda value: call(model, value))(jnp.stack((x, x)))
    assert jnp.allclose(batched, expected[None], atol=tolerance, rtol=tolerance)

    def loss(state, reference):
        m, value = state
        y = _incontext_reference(m, value, n_train) if reference else call(m, value)
        return jnp.square(y.astype(jnp.float32)).mean()

    actual_grad = eqx.filter_jit(eqx.filter_grad(loss))((model, x), False)
    expected_grad = eqx.filter_jit(eqx.filter_grad(loss))((model, x), True)
    for actual, expected in zip(
        jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True
    ):
        assert actual.dtype == expected.dtype
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_default_tabular_layer_registries():
    assert layers.get_attn("attention") is layers.Attention
    assert layers.get_attn("crossattention") is layers.CrossAttention
    assert layers.get_attn("incontextattention") is layers.InContextAttention
    assert layers.get_attn("softmaxscaling") is layers.SoftmaxScaling

    assert layers.get_attn_block("attentionblock") is layers.AttentionBlock
    assert layers.get_attn_block("crossattentionblock") is layers.CrossAttentionBlock
    assert (
        layers.get_attn_block("incontextattentionblock")
        is layers.InContextAttentionBlock
    )
    assert (
        layers.get_attn_block("inducedattentionblock") is layers.InducedAttentionBlock
    )

    assert layers.Mlp is CoreMlp
    assert layers.get_ffn("mlp") is CoreMlp
    assert layers.get_preprocessor("preprocessor") is layers.Preprocessor
    assert layers.get_embedding("labelembedding") is layers.LabelEmbedding
    assert layers.get_decoder("attentiondecoder") is layers.AttentionDecoder

    assert layers.get_layer("attention") is layers.Attention
    assert layers.get_layer("attentionblock") is layers.AttentionBlock
    assert layers.get_layer("featuredistributionencoder") is (
        layers.FeatureDistributionEncoder
    )
    assert layers.get_layer("mlp") is CoreMlp
    assert layers.get_layer(layers.Attention) is layers.Attention


def test_tabular_register_layer_duplicate_and_force():
    class CustomTabularLayer(eqx.Module):
        pass

    assert (
        layers.register_layer("custom_tabular_layer")(CustomTabularLayer)
        is CustomTabularLayer
    )
    assert layers.get_layer("custom_tabular_layer") is CustomTabularLayer

    with pytest.raises(ValueError):
        layers.register_layer("custom_tabular_layer")(CustomTabularLayer)

    assert (
        layers.register_layer("custom_tabular_layer", force=True)(CustomTabularLayer)
        is CustomTabularLayer
    )


def test_tabular_family_registration_adds_global_layer_lookup():
    class CustomTabularFfn(eqx.Module):
        pass

    assert (
        layers.register_ffn("custom_tabular_ffn")(CustomTabularFfn) is CustomTabularFfn
    )
    assert layers.get_ffn("custom_tabular_ffn") is CustomTabularFfn
    assert layers.get_layer("custom_tabular_ffn") is CustomTabularFfn


@pytest.mark.parametrize(
    ("module", "family", "registry_attr"),
    (
        (attention, "attn", "_ATTN_REGISTRY"),
        (blocks, "attn_block", "_ATTN_BLOCK_REGISTRY"),
        (decoder, "decoder", "_DECODER_REGISTRY"),
        (decoder, "embedding", "_EMBEDDING_REGISTRY"),
        (preprocessing, "preprocessor", "_PREPROCESSOR_REGISTRY"),
    ),
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_family_registration_collisions_are_atomic(
    monkeypatch, module, family, registry_attr
):
    monkeypatch.setattr(module, registry_attr, {})
    monkeypatch.setattr(registry, "_LAYER_REGISTRY", {})
    register = getattr(module, f"register_{family}")
    get = getattr(module, f"get_{family}")

    class Existing(eqx.Module):
        pass

    class Replacement(eqx.Module):
        pass

    layers.register_layer("Shared")(Existing)
    with pytest.raises(ValueError, match="already registered"):
        register("SHARED")(Replacement)
    assert getattr(module, registry_attr) == {}
    assert layers.get_layer("shared") is Existing

    assert register("SHARED", force=True)(Replacement) is Replacement
    assert get("Shared") is Replacement
    assert layers.get_layer("shared") is Replacement

    layers.register_layer("shared", force=True)(Existing)
    with pytest.raises(ValueError, match="already registered"):
        register("shared")(Existing)
    assert get("shared") is Replacement
    assert layers.get_layer("shared") is Existing


def test_tabular_unknown_layer_raises():
    with pytest.raises(ValueError):
        layers.get_attn("__missing_tabular_attn__")
    with pytest.raises(ValueError):
        layers.get_layer("__missing_tabular_layer__")


def test_old_tabular_layer_names_are_not_exported():
    assert not hasattr(layers, "TabularMlp")
    assert not hasattr(layers, "TabularPreprocessor")
    assert not hasattr(layers, "ClassAttentionDecoder")


def test_drop_path_zero_block_is_deterministic_without_key():
    block = layers.AttentionBlock(8, 2, drop_path=0.0, key=jr.PRNGKey(0))
    x = jr.normal(jr.PRNGKey(1), (4, 8))

    out_inference = block(x, inference=True)
    out_training = block(x, inference=False)

    assert_prng_free_jaxpr(
        lambda value, key: block(value, key=key, inference=True),
        x,
        jr.PRNGKey(2),
    )

    assert jnp.allclose(out_inference, out_training)


def test_preprocessor_singleton_forward_and_gradients_are_finite():
    preprocessor = layers.Preprocessor()
    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, jnp.nan, 5.0],
            [3.0, 2.0, jnp.inf, -jnp.inf],
        ]
    )

    out = preprocessor(x, n_train=1)
    gradients = jax.grad(lambda inputs: preprocessor(inputs, n_train=1).sum())(x)

    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(jnp.isfinite(gradients)))


def test_preprocessor_multirow_uses_sample_standard_deviation():
    preprocessor = layers.Preprocessor(
        feature_group_size=2,
        use_nan_indicators=False,
    )
    x = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 7.0],
            [5.0, 2.0, 11.0],
        ]
    )
    train = x[:2]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, ddof=1, keepdims=True)
    std = jnp.where(std == 0, jnp.ones_like(std), std)
    normalized = jnp.clip(
        (x - mean) / (std + jnp.finfo(std.dtype).eps),
        min=-100,
        max=100,
    )
    expected = jnp.stack(
        [jnp.roll(normalized, -(2**i), axis=1) for i in range(2)],
        axis=-1,
    )

    assert jnp.allclose(preprocessor(x, n_train=2), expected)


def test_tabular_mlp_helper_preserves_leading_dimensions():
    mlp = layers.Mlp(
        in_dim=4,
        hidden_dim=8,
        out_dim=6,
        act_layer="exactgelu",
        dropout_rate=0.0,
        norm_layer=None,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (2, 3, 4))

    out = _call_mlp(mlp, x, key=jr.PRNGKey(2), inference=True)
    expected = mlp(
        x.reshape(-1, x.shape[-1]),
        key=jr.PRNGKey(2),
        inference=True,
    ).reshape(2, 3, 6)

    assert jnp.allclose(out, expected)
    assert_prng_free_jaxpr(
        lambda value, key: _call_mlp(mlp, value, key=key, inference=True),
        x,
        jr.PRNGKey(2),
    )
