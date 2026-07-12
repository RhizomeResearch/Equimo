"""Feature extraction and head replacement tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.audio.models import AudioSpectrogramTransformer
from equimo.language.models import TextTransformerEncoder
from equimo.vision.models import VisionTransformer

from fixtures import TinyVisionTransformer, assert_tree_allclose


class NoKwargFeatureModel(eqx.Module):
    def features(self, x):
        return jnp.stack([x, x + 1.0])


class ConvFeatureModel(eqx.Module):
    stem: tuple[()] = eqx.field(static=True)
    stages: tuple[()] = eqx.field(static=True)

    def __init__(self):
        self.stem = ()
        self.stages = ()

    def features(self, x):
        return x


class IntermediateTokenModel(eqx.Module):
    cls_token: jnp.ndarray
    norm: eqx.nn.LayerNorm
    num_prefix_tokens: int = eqx.field(static=True)

    def __init__(self):
        self.cls_token = jnp.zeros((1, 4), dtype=jnp.float32)
        self.norm = eqx.nn.LayerNorm(4)
        self.num_prefix_tokens = 1

    def intermediate_features(
        self,
        x,
        *,
        key=None,
        inference: bool | None = True,
        n_last_blocks: int | None = None,
        **kwargs,
    ):
        del key, inference, kwargs
        outputs = (
            jnp.concatenate([self.cls_token, x], axis=0),
            jnp.concatenate([self.cls_token + 1.0, x + 1.0], axis=0),
        )
        return outputs if n_last_blocks is None else outputs[-n_last_blocks:]


def test_replace_head_preserves_backbone(tiny_vision_transformer):
    key = jr.PRNGKey(0)
    new_head = eqft.LinearHead(4, 5, key=key)

    replaced = eqft.replace_head(tiny_vision_transformer, new_head)

    assert_tree_allclose(replaced.head, new_head)
    assert replaced.head(jnp.ones((4,))).shape == (5,)
    assert_tree_allclose(replaced.patch_embed, tiny_vision_transformer.patch_embed)
    assert_tree_allclose(replaced.blocks, tiny_vision_transformer.blocks)


def test_replace_head_validates_input_features(tiny_vision_transformer):
    with pytest.raises(ValueError, match="input-feature mismatch"):
        eqft.replace_head(
            tiny_vision_transformer,
            eqft.LinearHead(5, 3, key=jr.PRNGKey(0)),
        )


def test_replace_head_can_preserve_old_head_metadata(tiny_vision_transformer):
    replaced = eqft.replace_head(
        tiny_vision_transformer,
        eqft.LinearHead(4, 3, key=jr.PRNGKey(0)),
        preserve_old_head_metadata=True,
    )

    assert replaced.head.old_head_metadata["class_name"] == "Linear"
    assert replaced.head.old_head_metadata["in_features"] == 4
    assert replaced.head.old_head_metadata["out_features"] == 2
    assert replaced.head(jnp.ones((4,))).shape == (3,)


def test_extract_features_pool_cls_and_mean_patch(tiny_vision_transformer):
    x = jnp.ones((2, 3))

    cls = eqft.extract_features(tiny_vision_transformer, x, pool="cls")
    mean_patch = eqft.extract_features(tiny_vision_transformer, x, pool="mean_patch")
    cls_patch_mean = eqft.extract_features(
        tiny_vision_transformer,
        x,
        pool="cls_patch_mean",
    )

    assert cls.shape == (4,)
    assert mean_patch.shape == (4,)
    assert cls_patch_mean.shape == (8,)


def test_extract_features_real_vit_uses_normalized_readouts():
    key = jr.PRNGKey(10)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[1],
        reg_tokens=1,
        num_classes=0,
        key=key,
    )
    x = jr.normal(jr.PRNGKey(11), (3, 16, 16))

    raw = model.features(x, key=key, inference=True)
    normalized = jax.vmap(model.norm)(raw)
    cls = eqft.extract_features(model, x, pool="cls", key=key, inference=True)
    mean_patch = eqft.extract_features(
        model, x, pool="mean_patch", key=key, inference=True
    )
    unpooled = eqft.extract_features(model, x, pool="none", key=key, inference=True)

    assert jnp.allclose(cls, normalized[0])
    assert jnp.allclose(mean_patch, jnp.mean(normalized[2:], axis=0))
    assert jnp.array_equal(unpooled, raw)


def test_extract_features_cls_patch_mean_excludes_register_tokens():
    model = TinyVisionTransformer(num_reg_tokens=2, key=jr.PRNGKey(0))
    x = jnp.ones((2, 3))

    pooled = eqft.extract_features(model, x, pool="cls_patch_mean")
    features = model.features(x)
    expected = jnp.concatenate([features[0], jnp.mean(features[3:], axis=0)], axis=0)

    assert pooled.shape == (8,)
    assert jnp.allclose(pooled, expected)


def test_extract_features_auto_pool_audio_uses_native_token_readout(
    tiny_ast_like_encoder,
):
    x = jnp.ones((2, 6))

    auto = eqft.extract_features(tiny_ast_like_encoder, x, pool="auto")
    features = tiny_ast_like_encoder.features(x)

    assert jnp.allclose(auto, 0.5 * (features[0] + features[1]))


@pytest.mark.parametrize("global_pool", ["avg", "avgmax", "max", "cls_patch_mean"])
def test_extract_features_real_ast_auto_excludes_prefix_tokens(global_pool):
    key = jr.PRNGKey(12)
    model = AudioSpectrogramTransformer(
        input_fdim=16,
        input_tdim=16,
        dim=8,
        patch_size=8,
        fstride=8,
        tstride=8,
        num_heads=2,
        depths=[1],
        global_pool=global_pool,
        num_classes=0,
        key=key,
    )
    x = jr.normal(jr.PRNGKey(13), (16, 16))
    native = model.forward_features(x, key=key, inference=True)
    patches = native["x_norm_patchtokens"]

    pooled = eqft.extract_features(model, x, pool="auto", key=key, inference=True)

    if global_pool == "avg":
        expected = jnp.mean(patches, axis=0)
    elif global_pool == "avgmax":
        expected = 0.5 * (jnp.mean(patches, axis=0) + jnp.max(patches, axis=0))
    elif global_pool == "max":
        expected = jnp.max(patches, axis=0)
    else:
        expected = jnp.concatenate(
            [native["x_norm_cls_token"], jnp.mean(patches, axis=0)], axis=0
        )

    assert jnp.allclose(pooled, expected)


def test_extract_features_real_ast_auto_uses_normalized_cls_dist_readout():
    key = jr.PRNGKey(14)
    model = AudioSpectrogramTransformer(
        input_fdim=16,
        input_tdim=16,
        dim=8,
        patch_size=8,
        fstride=8,
        tstride=8,
        num_heads=2,
        depths=[1],
        global_pool="token",
        num_classes=0,
        key=key,
    )
    x = jr.normal(jr.PRNGKey(15), (16, 16))
    native = model.forward_features(x, key=key, inference=True)
    raw = model.features(x, key=key, inference=True)

    pooled = eqft.extract_features(model, x, pool="auto", key=key, inference=True)
    unpooled = eqft.extract_features(model, x, pool=None, key=key, inference=True)

    expected = 0.5 * (native["x_norm_cls_token"] + native["x_norm_dist_token"])
    assert jnp.allclose(pooled, expected)
    assert jnp.array_equal(unpooled, raw)


def test_extract_features_auto_pool_text_uses_mean_token(tiny_text_encoder):
    token_ids = jnp.asarray([0, 1, 2])

    auto = eqft.extract_features(tiny_text_encoder, token_ids, pool="auto")
    mean_token = eqft.extract_features(tiny_text_encoder, token_ids, pool="mean_token")

    assert jnp.allclose(auto, mean_token)


@pytest.mark.parametrize(
    ("ids", "changed_ids", "padding_mask", "mask_as_keyword"),
    [
        (
            jnp.array([1, 2, 3, 4, 5]),
            jnp.array([1, 2, 3, 17, 18]),
            jnp.array([0, 0, 0, 1, 1]),
            False,
        ),
        (
            jnp.array([4, 5, 1, 2, 3]),
            jnp.array([17, 18, 1, 2, 3]),
            jnp.array([1, 1, 0, 0, 0]),
            True,
        ),
    ],
)
def test_extract_features_real_text_masks_padding(
    ids, changed_ids, padding_mask, mask_as_keyword
):
    key = jr.PRNGKey(16)
    model = TextTransformerEncoder(
        dim=8,
        mlp_ratio=2.0,
        depth=1,
        num_heads=2,
        vocab_size=32,
        key=key,
    )

    if mask_as_keyword:
        pooled = eqft.extract_features(
            model,
            ids,
            padding_mask=padding_mask,
            pool="auto",
            key=key,
            inference=True,
        )
        changed = eqft.extract_features(
            model,
            changed_ids,
            padding_mask=padding_mask,
            pool="auto",
            key=key,
            inference=True,
        )
    else:
        pooled = eqft.extract_features(
            model, ids, padding_mask, pool="auto", key=key, inference=True
        )
        changed = eqft.extract_features(
            model, changed_ids, padding_mask, pool="auto", key=key, inference=True
        )
    features = model.features(ids, padding_mask, key=key, inference=True)
    expected = jnp.mean(features[padding_mask == 0], axis=0)

    assert jnp.allclose(pooled, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(changed, pooled, rtol=1e-6, atol=1e-6)


def test_extract_features_auto_pool_conv_features_uses_global_average():
    x = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 2, 2)

    pooled = eqft.extract_features(ConvFeatureModel(), x, pool="auto")

    assert jnp.array_equal(pooled, jnp.mean(x, axis=(1, 2)))


def test_extract_features_drops_optional_key_for_plain_features():
    features = eqft.extract_features(
        NoKwargFeatureModel(),
        jnp.ones((4,)),
        pool="cls",
        key=jr.PRNGKey(0),
    )

    assert features.shape == (4,)


def test_linear_probe_trainable_only_head(tiny_vision_transformer):
    key = jr.PRNGKey(1)
    probe = eqft.make_linear_probe(
        tiny_vision_transformer,
        in_features=4,
        out_features=3,
        key=key,
        pool="cls",
    )
    plan = eqft.prepare_finetune(
        probe,
        trainable=eqft.TrainableSpec(mode="head"),
    )

    assert isinstance(probe.backbone.head, eqft.IdentityHead)
    assert plan.trainable.head.linear.weight is not None
    assert plan.trainable.backbone.patch_embed.proj.weight is None
    assert plan.trainable.backbone.blocks[0].attn.qkv.weight is None
    assert plan.report.trainable_params == 15


def test_linear_probe_supports_cls_patch_mean(tiny_vision_transformer):
    key = jr.PRNGKey(1)
    probe = eqft.make_linear_probe(
        tiny_vision_transformer,
        in_features=8,
        out_features=3,
        key=key,
        pool="cls_patch_mean",
    )

    y = probe(jnp.ones((2, 3)))

    assert y.shape == (3,)
    assert probe.head.linear.in_features == 8


def test_linear_probe_cls_patch_mean_layer_norm_readout_is_trainable(
    tiny_vision_transformer,
):
    key = jr.PRNGKey(2)
    head = eqft.LayerNormReadoutHead(8, eqft.LinearHead(8, 3, key=key))
    probe = eqft.make_linear_probe(
        tiny_vision_transformer,
        in_features=8,
        out_features=3,
        key=key,
        pool="cls_patch_mean",
        head=head,
    )
    plan = eqft.prepare_finetune(
        probe,
        trainable=eqft.TrainableSpec(mode="head"),
    )

    y = probe(jnp.ones((2, 3)))

    assert y.shape == (3,)
    assert plan.trainable.head.norm.weight is not None
    assert plan.trainable.head.norm.bias is not None
    assert plan.trainable.head.head.linear.weight is not None
    assert plan.trainable.backbone.patch_embed.proj.weight is None
    assert plan.report.trainable_params == 43


def test_feature_extractor_filter_jit(tiny_vision_transformer):
    extractor = eqft.FeatureExtractor(tiny_vision_transformer, pool="cls")
    x = jnp.ones((2, 3))

    features = eqx.filter_jit(extractor)(x)

    assert features.shape == (4,)


def test_attention_pool_input_from_forward_features():
    features = {
        "x_norm_cls_token": jnp.ones((4,), dtype=jnp.float32),
        "x_norm_patchtokens": jnp.ones((3, 4), dtype=jnp.float32) * 2.0,
    }

    tokens = eqft.make_attention_pool_input_from_forward_features(
        features,
        prepend_cls_token=True,
        l2_normalize_cls=True,
    )

    assert tokens.shape == (4, 4)
    assert jnp.allclose(jnp.linalg.norm(tokens[0]), 1.0)
    assert jnp.allclose(tokens[1:], 2.0)


def test_attention_pool_input_from_intermediates_concatenates_layers():
    patch0 = jnp.ones((2, 4), dtype=jnp.float32)
    patch1 = patch0 * 2.0
    cls0 = jnp.ones((4,), dtype=jnp.float32)
    cls1 = cls0 * 3.0

    tokens = eqft.make_attention_pool_input_from_intermediates(
        ((patch0, cls0), (patch1, cls1)),
        2,
        prepend_cls_token=True,
    )

    assert tokens.shape == (3, 8)
    assert jnp.allclose(tokens[0], jnp.concatenate([cls0, cls1]))
    assert jnp.allclose(tokens[1:], jnp.concatenate([patch0, patch1], axis=-1))


def test_attention_pool_probe_uses_intermediate_features_and_trains_head_only():
    key = jr.PRNGKey(6)
    probe = eqft.make_attention_pool_probe(
        IntermediateTokenModel(),
        in_features=8,
        out_features=3,
        key=key,
        n_last_blocks=2,
        embed_dim=8,
        num_heads=2,
        prepend_cls_token=False,
    )
    x = jnp.ones((2, 4), dtype=jnp.float32)

    y = probe(x, key=jr.PRNGKey(7), inference=True)
    plan = eqft.prepare_finetune(probe, trainable=eqft.TrainableSpec(mode="head"))

    assert y.shape == (3,)
    assert plan.trainable.head.input_proj.weight is not None
    assert plan.trainable.backbone.cls_token is None
