"""Fine-tuning head tests."""

from __future__ import annotations

import inspect

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_projection_head_signatures_use_feature_vocabulary():
    for head_cls in (eqft.ProjectionHead, eqft.ContrastiveProjectionHead):
        parameters = inspect.signature(head_cls).parameters

        assert "in_features" in parameters
        assert "out_features" in parameters
        assert "out_dim" not in parameters


class KeyInferenceHead(eqx.Module):
    def __call__(self, x, *, key, inference: bool | None = True):
        inference_flag = 0.0 if inference else 1.0
        return jnp.asarray([jnp.sum(x), inference_flag, jr.uniform(key, ())])


def test_multilabel_head_raw_logits():
    key = jr.PRNGKey(0)
    head = eqft.MultiLabelHead(4, 3, key=key)
    x = jnp.ones((4,))

    logits = head(x)

    assert logits.shape == (3,)
    assert jnp.any((logits < 0.0) | (logits > 1.0))


def test_ctc_head_frame_logits():
    key = jr.PRNGKey(1)
    head = eqft.CTCHead(4, 7, key=key)
    x = jnp.ones((5, 4))

    logits = head(x)

    assert logits.shape == (5, 7)
    assert head.blank_id == 0


def test_contrastive_projection_head_normalizes():
    key = jr.PRNGKey(2)
    head = eqft.ContrastiveProjectionHead(
        in_features=4,
        out_features=3,
        key=key,
    )
    x = jnp.ones((4,))

    y = head(x)

    assert y.shape == (3,)
    assert jnp.linalg.norm(y) == jnp.array(1.0)


def test_dense_feature_adapter_projects_last_axis():
    key = jr.PRNGKey(3)
    adapter = eqft.DenseFeatureAdapter(4, 2, key=key, activation="relu")
    x = jnp.ones((3, 5, 4))

    y = adapter(x)

    assert y.shape == (3, 5, 2)


def test_layer_norm_readout_head_shapes():
    key = jr.PRNGKey(4)
    head = eqft.LayerNormReadoutHead(4, eqft.LinearHead(4, 3, key=key))

    y = head(jnp.arange(4, dtype=jnp.float32))
    y_batched = head(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4))

    assert y.shape == (3,)
    assert y_batched.shape == (2, 3, 3)


def test_layer_norm_readout_head_forwards_key_and_inference():
    head = eqft.LayerNormReadoutHead(4, KeyInferenceHead())

    y = head(
        jnp.arange(4, dtype=jnp.float32),
        key=jr.PRNGKey(5),
        inference=False,
    )

    assert y.shape == (3,)
    assert y[1] == jnp.array(1.0)


def test_attention_pooling_classifier_head_shapes_and_mask():
    key = jr.PRNGKey(6)
    head = eqft.AttentionPoolingClassifierHead(
        4,
        3,
        key=key,
        embed_dim=8,
        num_heads=2,
    )
    tokens = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)

    logits = head(tokens, mask=jnp.asarray([1, 1, 0, 1, 0]), inference=True)
    all_masked = head(tokens, mask=jnp.zeros((5,), dtype=bool), inference=True)

    assert logits.shape == (3,)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.all(jnp.isfinite(all_masked))


def test_attention_pooling_classifier_head_validates_inputs():
    key = jr.PRNGKey(7)
    with pytest.raises(ValueError, match="divisible"):
        eqft.AttentionPoolingClassifierHead(4, 3, key=key, embed_dim=10, num_heads=3)

    head = eqft.AttentionPoolingClassifierHead(
        4,
        3,
        key=key,
        embed_dim=8,
        num_heads=2,
    )
    with pytest.raises(ValueError, match="tokens shaped"):
        head(jnp.ones((4,)))
    with pytest.raises(ValueError, match="mask must have shape"):
        head(jnp.ones((5, 4)), mask=jnp.ones((4,)))


def test_attention_pooling_classifier_head_requires_dropout_key():
    head = eqft.AttentionPoolingClassifierHead(
        4,
        3,
        key=jr.PRNGKey(8),
        embed_dim=8,
        num_heads=2,
        dropout=0.1,
    )

    with pytest.raises(ValueError, match="dropout"):
        head(jnp.ones((5, 4)), inference=False)
