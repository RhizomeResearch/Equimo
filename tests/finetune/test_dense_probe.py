"""Reusable classification and spatial linear-probe tests."""

from __future__ import annotations

from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.serialization import inspect_checkpoint, load_weights, save_model

from fixtures import TinyVisionTransformer


class SpatialTinyVisionTransformer(TinyVisionTransformer):
    """Tiny transformer with an exact rectangular patch-grid contract."""

    def feature_metadata(self, x, *, endpoint, endpoint_options):
        del endpoint, endpoint_options
        grid_h, grid_w = 2, 3
        if x.shape[0] != grid_h * grid_w:
            raise ValueError("expected six row-major patch vectors")
        return {
            "input_size": (4, 6),
            "patch_size": (2, 2),
            "patch_padding": (0, 0),
            "grid_size": (grid_h, grid_w),
            "prefix_tokens": ("cls",),
            "tokens_include_prefix": True,
            "endpoint_normalization": "encoder_final_norm",
        }


class ChannelFirstFeatures(eqx.Module):
    def features(self, x):
        return x


def _token_spec() -> eqft.FeatureSpec:
    return eqft.FeatureSpec(
        "features",
        "BNC",
        "patches",
        None,
        return_metadata=True,
    )


def _map_spec() -> eqft.FeatureSpec:
    return eqft.FeatureSpec(
        "features",
        "BCHW",
        "all",
        None,
        return_metadata=True,
    )


def _backbone(key) -> SpatialTinyVisionTransformer:
    return SpatialTinyVisionTransformer(
        dim=4,
        hidden_dim=8,
        depth=4,
        patch_features=3,
        num_patches=6,
        num_classes=2,
        key=key,
    )


def _classification_probe(key):
    return eqft.make_linear_probe(
        _backbone(key),
        in_features=4,
        out_features=3,
        key=jr.fold_in(key, 1),
        feature_spec=eqft.FeatureSpec("features", "BNC", "cls", None),
    )


def _dense_probe(key):
    return eqft.vision.make_dense_probe(
        _backbone(key),
        in_features=4,
        out_features=3,
        key=jr.fold_in(key, 2),
        feature_spec=_token_spec(),
    )


def _input():
    return jnp.arange(18, dtype=jnp.float32).reshape(6, 3) / 10


def _array_paths(tree):
    return {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
        if eqx.is_array(leaf)
    }


def _gradient_step(model, x, trainable):
    plan = eqft.prepare_finetune(model, trainable=trainable)

    def loss(trainable_tree):
        candidate = eqx.combine(trainable_tree, plan.frozen)
        return jnp.mean(jnp.square(candidate(x)))

    gradients = eqx.filter_grad(loss)(plan.trainable)
    updates = jax.tree.map(
        lambda gradient: None if gradient is None else -0.01 * gradient,
        gradients,
    )
    return eqx.combine(eqx.apply_updates(plan.trainable, updates), plan.frozen)


def _model_config(kind, spec, *, out_features=3):
    return {
        "wrapper": kind,
        "backbone": {
            "type": "SpatialTinyVisionTransformer",
            "dim": 4,
            "hidden_dim": 8,
            "depth": 4,
            "patch_features": 3,
            "num_patches": 6,
            "num_classes": 2,
        },
        "feature_spec": asdict(spec),
        "head": {
            "in_features": 4,
            "out_features": out_features,
            "bias": True,
            "weight_init": "trunc_normal_0.02",
            "bias_init": 0.0,
        },
        "output_layout": "C" if kind == "linear_probe" else "CHW",
    }


def test_dense_probe_token_projection_matches_every_patch():
    head = eqft.LinearHead(2, 3, key=jr.PRNGKey(0))
    head = eqx.tree_at(
        lambda module: (module.linear.weight, module.linear.bias),
        head,
        (
            jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            jnp.asarray([0.5, -0.5, 1.0]),
        ),
    )

    class CoordinateFeatures(eqx.Module):
        num_prefix_tokens: int = eqx.field(static=True, default=2)

        def features(self, x):
            prefix = jnp.full((2, 2), -100, dtype=x.dtype)
            tokens = jnp.stack((x.reshape(-1), x.reshape(-1) + 10), axis=-1)
            return jnp.concatenate((prefix, tokens), axis=0)

        def feature_metadata(self, x, *, endpoint, endpoint_options):
            del endpoint, endpoint_options
            return {
                "grid_size": x.shape,
                "prefix_tokens": ("cls", "register_0"),
                "tokens_include_prefix": True,
            }

    probe = eqft.vision.make_dense_probe(
        CoordinateFeatures(),
        in_features=2,
        out_features=3,
        key=jr.PRNGKey(1),
        feature_spec=_token_spec(),
        head=head,
    )
    coordinates = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    logits = probe(coordinates)
    token_features = jnp.stack(
        (coordinates.reshape(-1), coordinates.reshape(-1) + 10), axis=-1
    )
    expected = token_features @ head.linear.weight.T + head.linear.bias
    expected = jnp.moveaxis(expected.reshape(2, 3, 3), -1, 0)

    assert logits.shape == (3, 2, 3)
    assert jnp.array_equal(logits, expected)


def test_dense_probe_projects_channel_first_features():
    head = eqft.LinearHead(2, 2, key=jr.PRNGKey(2))
    head = eqx.tree_at(
        lambda module: (module.linear.weight, module.linear.bias),
        head,
        (jnp.asarray([[2.0, 0.0], [0.0, -1.0]]), jnp.asarray([1.0, 3.0])),
    )
    probe = eqft.vision.make_dense_probe(
        ChannelFirstFeatures(),
        in_features=2,
        out_features=2,
        key=jr.PRNGKey(3),
        feature_spec=_map_spec(),
        head=head,
    )
    features = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)

    logits = probe(features)

    assert logits.shape == (2, 2, 3)
    assert jnp.array_equal(logits[0], 2 * features[0] + 1)
    assert jnp.array_equal(logits[1], -features[1] + 3)


@pytest.mark.parametrize(
    ("spec", "message"),
    (
        (eqft.FeatureSpec("features", "BNC", "patches", None), "metadata"),
        (
            eqft.FeatureSpec(
                "features", "BNC", "patches", "mean_patch", return_metadata=True
            ),
            "unpooled",
        ),
        (
            eqft.FeatureSpec("features", "BNC", "all", None, return_metadata=True),
            "BNC patches",
        ),
        (
            eqft.FeatureSpec(
                "features",
                "BNC",
                "patches",
                None,
                layer_aggregation={"method": "separate"},
                return_metadata=True,
            ),
            "separate",
        ),
    ),
)
def test_dense_probe_rejects_incompatible_feature_specs(spec, message):
    with pytest.raises(ValueError, match=message):
        eqft.vision.make_dense_probe(
            ChannelFirstFeatures(),
            in_features=2,
            out_features=3,
            key=jr.PRNGKey(4),
            feature_spec=spec,
        )


@pytest.mark.parametrize(
    ("head", "message"),
    (
        (eqft.LinearHead(5, 3, key=jr.PRNGKey(5)), "input-feature mismatch"),
        (eqft.LinearHead(4, 2, key=jr.PRNGKey(6)), "output-feature mismatch"),
    ),
)
def test_make_dense_probe_rejects_head_dimension_mismatch(head, message):
    with pytest.raises(ValueError, match=message):
        eqft.vision.make_dense_probe(
            _backbone(jr.PRNGKey(7)),
            in_features=4,
            out_features=3,
            key=jr.PRNGKey(8),
            feature_spec=_token_spec(),
            head=head,
        )


@pytest.mark.parametrize(
    ("feature_width", "include_grid", "message"),
    ((5, True, "feature-width mismatch"), (4, False, "patch-grid metadata")),
)
def test_dense_probe_rejects_runtime_feature_contract_mismatch(
    feature_width, include_grid, message
):
    class ContractFeatures(eqx.Module):
        num_prefix_tokens: int = eqx.field(static=True, default=0)

        def features(self, x):
            return jnp.ones((6, feature_width), dtype=x.dtype)

        def feature_metadata(self, x, *, endpoint, endpoint_options):
            del x, endpoint, endpoint_options
            metadata = {"tokens_include_prefix": False}
            return metadata | ({"grid_size": (2, 3)} if include_grid else {})

    probe = eqft.vision.make_dense_probe(
        ContractFeatures(),
        in_features=4,
        out_features=3,
        key=jr.PRNGKey(8),
        feature_spec=_token_spec(),
    )

    with pytest.raises(ValueError, match=message):
        probe(jnp.ones((1,), dtype=jnp.float32))


def test_dense_probe_supports_deterministic_jit_vmap_and_gradients():
    probe = _dense_probe(jr.PRNGKey(9))
    x = _input()

    eager = probe(x)
    compiled = eqx.filter_jit(probe)(x)
    batched = eqx.filter_jit(lambda model, batch: jax.vmap(model)(batch))(
        probe, jnp.stack((x, x + 1))
    )
    gradient = jax.grad(lambda value: jnp.sum(probe(value)))(x)

    assert eager.shape == (3, 2, 3)
    assert eager.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(eager))
    assert jnp.array_equal(compiled, eager)
    assert batched.shape == (2, 3, 2, 3)
    assert gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_dense_probe_preserves_homogeneous_bfloat16():
    probe = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        _dense_probe(jr.PRNGKey(10)),
    )

    logits = probe(_input().astype(jnp.bfloat16))

    assert logits.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(logits))


@pytest.mark.parametrize("factory", (_classification_probe, _dense_probe))
def test_probe_head_only_update_preserves_backbone_bitwise(factory):
    probe = factory(jr.PRNGKey(11))
    before = _array_paths(probe)

    updated = _gradient_step(probe, _input(), eqft.TrainableSpec(mode="head"))
    after = _array_paths(updated)

    assert not jnp.array_equal(
        before[".head.linear.weight"], after[".head.linear.weight"]
    )
    for path, value in before.items():
        if path.startswith(".backbone"):
            assert jnp.array_equal(value, after[path]), path


@pytest.mark.parametrize("factory", (_classification_probe, _dense_probe))
def test_probe_explicit_two_block_update_changes_only_selected_subtrees(factory):
    probe = factory(jr.PRNGKey(12))
    before = _array_paths(probe)
    spec = eqft.TrainableSpec(
        mode="surgical",
        target=eqft.TargetSpec(
            include=("head", "backbone.blocks.2", "backbone.blocks.3")
        ),
    )

    updated = _gradient_step(probe, _input(), spec)
    after = _array_paths(updated)
    changed = {
        path for path in before if not jnp.array_equal(before[path], after[path])
    }

    assert any(path.startswith(".head") for path in changed)
    assert any(path.startswith(".backbone.blocks[2]") for path in changed)
    assert any(path.startswith(".backbone.blocks[3]") for path in changed)
    assert all(
        path.startswith((".head", ".backbone.blocks[2]", ".backbone.blocks[3]"))
        for path in changed
    )


@pytest.mark.parametrize(
    ("kind", "factory"),
    (("linear_probe", _classification_probe), ("dense_probe", _dense_probe)),
)
def test_probe_checkpoint_roundtrip_before_and_after_update(tmp_path, kind, factory):
    original = factory(jr.PRNGKey(13))
    spec = original.feature_spec
    config = _model_config(kind, spec)
    trained = _gradient_step(original, _input(), eqft.TrainableSpec(mode="head"))

    for name, model in (("initialized", original), ("trained", trained)):
        path = tmp_path / name
        save_model(path, model, config, compression=False)
        inspected = inspect_checkpoint(
            path,
            model=model,
            expected_model_config=config,
        )
        template = factory(jr.PRNGKey(99))
        template_logits = template(_input())
        loaded = load_weights(
            template,
            path=path,
            expected_model_config=config,
        )

        assert inspected.verified is True
        assert inspected.model_class is not None
        assert not jnp.array_equal(template_logits, model(_input()))
        assert jnp.array_equal(loaded(_input()), model(_input()))
        assert loaded.head.linear.in_features == 4
        assert loaded.head.linear.out_features == 3

    changed_config = _model_config(kind, spec, out_features=4)
    with pytest.raises(ValueError, match="model_config mismatch"):
        inspect_checkpoint(
            tmp_path / "initialized",
            model=original,
            expected_model_config=changed_config,
        )
