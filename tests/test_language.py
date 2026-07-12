import jax.numpy as jnp
import jax.random as jr

from equimo.language.models import TextTransformerEncoder
from equimo.registry import get_model_cls


KEY = jr.PRNGKey(0)


def test_text_transformer_encoder_forward_and_features_shape():
    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=128,
        key=KEY,
    )
    ids = jnp.array([1, 2, 3, 0, 0])
    padding_mask = jnp.array([0, 0, 0, 1, 1])

    features = model.features(ids, padding_mask, key=KEY, inference=True)
    pooled = model(ids, padding_mask, key=KEY, inference=True)

    assert features.shape == (5, 16)
    assert pooled.shape == (16,)
    assert jnp.all(jnp.isfinite(features))
    assert jnp.all(jnp.isfinite(pooled))


def test_text_transformer_encoder_ignores_padded_token_ids():
    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=128,
        key=KEY,
    )
    ids_a = jnp.array([1, 2, 3, 4, 5])
    ids_b = jnp.array([1, 2, 3, 17, 18])
    padding_mask = jnp.array([0, 0, 0, 1, 1])

    features_a = model.features(ids_a, padding_mask, key=KEY, inference=True)
    features_b = model.features(ids_b, padding_mask, key=KEY, inference=True)
    pooled_a = model(ids_a, padding_mask, key=KEY, inference=True)
    pooled_b = model(ids_b, padding_mask, key=KEY, inference=True)

    assert jnp.allclose(features_a[:3], features_b[:3], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(pooled_a, pooled_b, rtol=1e-6, atol=1e-6)

    intermediates_a = model.intermediate_features(
        ids_a,
        padding_mask,
        key=KEY,
        inference=True,
        n_last_blocks=2,
    )
    intermediates_b = model.intermediate_features(
        ids_b,
        padding_mask,
        key=KEY,
        inference=True,
        n_last_blocks=2,
    )

    assert len(intermediates_a) == len(intermediates_b) == 2
    for intermediate_a, intermediate_b in zip(
        intermediates_a, intermediates_b, strict=True
    ):
        assert jnp.allclose(
            intermediate_a[:3], intermediate_b[:3], rtol=1e-6, atol=1e-6
        )


def test_language_model_registered_by_modality():
    assert (
        get_model_cls("text_transformer_encoder", modality="language")
        is TextTransformerEncoder
    )
