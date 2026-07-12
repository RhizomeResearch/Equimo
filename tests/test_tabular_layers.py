import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.core.layers.ffn import Mlp as CoreMlp
from equimo.tabular import layers
from equimo.tabular.layers.mlp import _call_mlp


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
        4,
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
