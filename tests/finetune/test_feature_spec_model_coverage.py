"""FeatureSpec conformance coverage for every built-in model family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.audio.models import AudioSpectrogramTransformer
from equimo.language.models import TextTransformerEncoder
from equimo.registry import _MODEL_REGISTRY
from equimo.tabular.models import TabPFN
from equimo.vision.models import (
    AttNet,
    ConvNeXt,
    DEQ,
    FasterViT,
    IFormer,
    LowFormer,
    Mlla,
    MobileNetv3,
    PartialFormer,
    ReduceFormer,
    SHViT,
    VisionParcae,
    VisionTransformer,
    Vssd,
)


@dataclass(frozen=True)
class _Invocation:
    model: Any
    args: tuple[jax.Array, ...]
    kwargs: dict[str, Any]
    spec: eqft.FeatureSpec
    expected: jax.Array
    key: jax.Array


@dataclass(frozen=True)
class _FeatureCase:
    modality: str
    registry_name: str
    build: Callable[[jax.Array], _Invocation]


def _keys(key: jax.Array, count: int = 2) -> tuple[jax.Array, ...]:
    return tuple(jr.split(key, count))


def _token_invocation(
    model: Any,
    sample: jax.Array,
    *,
    key: jax.Array,
) -> _Invocation:
    tokens = model.features(sample, key=key, inference=True)
    return _Invocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec("features", "BNC", "all", "global_avg"),
        expected=jnp.mean(tokens, axis=0),
        key=key,
    )


def _spatial_invocation(
    model: Any,
    sample: jax.Array,
    *,
    key: jax.Array,
) -> _Invocation:
    feature_map = model.features(sample, key=key, inference=True)
    return _Invocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec("features", "BCHW", "all", "global_avg"),
        expected=jnp.mean(feature_map, axis=(1, 2)),
        key=key,
    )


def _build_vit(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[1],
        reg_tokens=1,
        num_classes=0,
        key=model_key,
    )
    sample = jr.normal(sample_key, (3, 16, 16))
    native = model.forward_features(sample, key=model_key, inference=True)
    return _Invocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec(
            "forward_features",
            "BNC",
            "patches",
            "mean_patch",
        ),
        expected=jnp.mean(native["x_norm_patchtokens"], axis=0),
        key=model_key,
    )


def _build_vision_parcae(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = VisionParcae(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=1,
        mean_recurrence=1,
        mean_backprop_depth=1,
        max_recurrence=1,
        num_classes=0,
        key=model_key,
    )
    sample = jr.normal(sample_key, (3, 16, 16))
    native = model.forward_features(sample, key=model_key, inference=True)
    return _Invocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec(
            "forward_features",
            "BNC",
            "patches",
            "mean_patch",
        ),
        expected=jnp.mean(native["x_norm_patchtokens"], axis=0),
        key=model_key,
    )


def _build_ast(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = AudioSpectrogramTransformer(
        input_fdim=16,
        input_tdim=16,
        dim=8,
        patch_size=8,
        fstride=8,
        tstride=8,
        num_heads=2,
        depths=[1],
        num_classes=0,
        key=model_key,
    )
    sample = jr.normal(sample_key, (16, 16))
    native = model.forward_features(sample, key=model_key, inference=True)
    return _Invocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec(
            "forward_features",
            "BNC",
            "frames",
            "mean_frame",
        ),
        expected=jnp.mean(native["x_norm_patchtokens"], axis=0),
        key=model_key,
    )


def _build_fastervit(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = FasterViT(
        img_size=32,
        in_channels=3,
        dim=16,
        in_dim=8,
        num_heads=1,
        hat=False,
        depths=[1, 1],
        window_size=2,
        ct_size=1,
        num_classes=0,
        key=model_key,
    )
    return _token_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_partialformer(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = PartialFormer(
        img_size=32,
        in_channels=3,
        dim=16,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=0.5,
        patch_size=4,
        num_classes=0,
        key=model_key,
    )
    return _token_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_mlla(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = Mlla(
        img_size=32,
        in_channels=3,
        dim=8,
        patch_size=4,
        depths=[1, 1],
        num_heads=[1, 2],
        num_classes=0,
        key=model_key,
    )
    return _token_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_vssd(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = Vssd(
        img_size=32,
        in_channels=3,
        dim=8,
        d_state=8,
        patch_size=4,
        depths=[1, 1],
        num_heads=[1, 2],
        attentions_layers=("mamba2mixer", "attention"),
        num_classes=0,
        key=model_key,
    )
    return _token_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_attnet(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = AttNet(
        in_channels=3,
        dims=[32],
        depths=[1],
        exp_rates=[2],
        kernel_sizes=[3],
        glu_dwconv=[False],
        glu_norm=[False],
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_convnext(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = ConvNeXt(
        in_channels=3,
        depths=[1],
        dims=[8],
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 16, 16)),
        key=model_key,
    )


def _build_iformer(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = IFormer(
        in_channels=3,
        modules=["iformerblock"],
        module_kwargs=[{"kernel_size": 3, "expand_ratio": 2}],
        downsamplers=["iformerstem"],
        downsampler_kwargs=[{}],
        dims=[16],
        depths=[1],
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_lowformer(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = LowFormer(
        in_channels=3,
        widths=[8],
        depths=[1],
        att_strides=[1],
        block_types=["attention"],
        attention_type="softmax",
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 16, 16)),
        key=model_key,
    )


def _build_mobilenetv3(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = MobileNetv3(
        in_channels=3,
        layers_config=[(8, 8, 3, 1, False, "relu")],
        last_channels=8,
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 16, 16)),
        key=model_key,
    )


def _build_reduceformer(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = ReduceFormer(
        in_channels=3,
        widths=[32],
        depths=[1],
        block_types=["attention"],
        head_dim=8,
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_shvit(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = SHViT(
        in_channels=3,
        dim=[8],
        pdim=[4],
        qk_dim=[4],
        depths=[1],
        block_type=["s"],
        num_classes=0,
        key=model_key,
    )
    return _spatial_invocation(
        model,
        jr.normal(sample_key, (3, 32, 32)),
        key=model_key,
    )


def _build_text_transformer(key: jax.Array) -> _Invocation:
    model_key, _ = _keys(key)
    model = TextTransformerEncoder(
        dim=8,
        mlp_ratio=2.0,
        depth=1,
        num_heads=2,
        vocab_size=32,
        key=model_key,
    )
    ids = jnp.asarray([1, 2, 3, 4])
    padding_mask = jnp.asarray([0, 0, 0, 1])
    return _Invocation(
        model=model,
        args=(ids, padding_mask),
        kwargs={},
        spec=eqft.FeatureSpec(
            "features",
            "BTC",
            "all",
            "mean_token",
            mask_field="padding_mask",
        ),
        expected=model(ids, padding_mask, key=model_key, inference=True),
        key=model_key,
    )


def _build_tabpfn(key: jax.Array) -> _Invocation:
    model_key, x_key, y_key = _keys(key, 3)
    model = TabPFN(
        num_classes=4,
        dim=8,
        depths=(1, 1, 1),
        num_heads=(1, 1, 1),
        num_inducing_points=2,
        feature_group_size=2,
        num_cls_tokens=1,
        num_kv_heads_test=1,
        decoder_head_dim=4,
        decoder_num_heads=1,
        mlp_ratio=2.0,
        key=model_key,
    )
    x = jr.normal(x_key, (5, 4))
    y = jr.randint(y_key, (5,), 0, 4)
    return _Invocation(
        model=model,
        args=(x, y),
        kwargs={"n_train": 3},
        spec=eqft.FeatureSpec("__call__", "BC", "all", None),
        expected=model(x, y, n_train=3, key=model_key, inference=True),
        key=model_key,
    )


def _build_deq(key: jax.Array) -> _Invocation:
    model_key, sample_key = _keys(key)
    model = DEQ(
        in_channels=3,
        depths=[1],
        dims=[8],
        block_types=["fpi"],
        modules=["convnextblock"],
        module_kwargs=[{}],
        downsamplers=["convnextstem"],
        downsampler_kwargs=[{}],
        fpi_injector="add",
        fpi_stabilizer="identity",
        fpi_strategy="entry",
        fpi_maxsteps=2,
        num_classes=0,
        key=model_key,
    )
    sample = jr.normal(sample_key, (3, 16, 16))
    layers = model.intermediate_features(
        sample,
        key=model_key,
        inference=True,
        n_last_blocks=1,
    )
    return _Invocation(
        model=model,
        args=(sample,),
        kwargs={"n_last_blocks": 1},
        spec=eqft.FeatureSpec(
            "intermediate_features",
            "BCHW",
            "all",
            "global_avg",
            layer_aggregation={"method": "last"},
        ),
        expected=jnp.mean(layers[-1], axis=(1, 2)),
        key=model_key,
    )


CASES = (
    _FeatureCase("audio", "ast", _build_ast),
    _FeatureCase("language", "text_transformer_encoder", _build_text_transformer),
    _FeatureCase("tabular", "tabpfn", _build_tabpfn),
    _FeatureCase("vision", "attnet", _build_attnet),
    _FeatureCase("vision", "convnext", _build_convnext),
    _FeatureCase("vision", "deq", _build_deq),
    _FeatureCase("vision", "fastervit", _build_fastervit),
    _FeatureCase("vision", "iformer", _build_iformer),
    _FeatureCase("vision", "lowformer", _build_lowformer),
    _FeatureCase("vision", "mlla", _build_mlla),
    _FeatureCase("vision", "mobilenetv3", _build_mobilenetv3),
    _FeatureCase("vision", "partialformer", _build_partialformer),
    _FeatureCase("vision", "reduceformer", _build_reduceformer),
    _FeatureCase("vision", "shvit", _build_shvit),
    _FeatureCase("vision", "vision_parcae", _build_vision_parcae),
    _FeatureCase("vision", "vit", _build_vit),
    _FeatureCase("vision", "vssd", _build_vssd),
)


def _extract(invocation: _Invocation, *args: jax.Array, key: jax.Array) -> jax.Array:
    return eqft.extract_features(
        invocation.model,
        *args,
        feature_spec=invocation.spec,
        key=key,
        inference=True,
        **invocation.kwargs,
    )


def _assert_jit_and_vmap(invocation: _Invocation) -> None:
    def extract_one(*call_args):
        return _extract(invocation, *call_args[:-1], key=call_args[-1])

    eager = extract_one(*invocation.args, invocation.key)
    compiled = jax.jit(extract_one)(*invocation.args, invocation.key)
    batched_args = tuple(jnp.stack((arg, arg)) for arg in invocation.args)
    batched = jax.jit(jax.vmap(extract_one))(
        *batched_args,
        jr.split(invocation.key, 2),
    )

    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)
    assert batched.shape == (2, *eager.shape)
    assert jnp.all(jnp.isfinite(batched))


def test_every_builtin_model_family_has_a_conformance_case():
    registered = {
        (modality, name)
        for name, entries in _MODEL_REGISTRY.items()
        for modality, model_cls in entries.items()
        if model_cls.__module__.startswith("equimo.")
    }
    covered = {(case.modality, case.registry_name) for case in CASES}

    assert len(CASES) == 17
    assert covered == registered


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.registry_name)
def test_builtin_model_family_executes_declared_feature_spec(case):
    invocation = case.build(jr.PRNGKey(0))
    result = _extract(invocation, *invocation.args, key=invocation.key)

    assert result.shape == invocation.expected.shape
    assert result.dtype == invocation.expected.dtype
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, invocation.expected, rtol=1e-6, atol=1e-6)


def test_classless_vit_supports_patch_features_and_rejects_cls_selection():
    key = jr.PRNGKey(20)
    model = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[1],
        class_token=False,
        reg_tokens=0,
        global_pos_embed_cls=False,
        num_classes=0,
        key=key,
    )
    sample = jr.normal(jr.PRNGKey(21), (3, 16, 16))
    native = model.forward_features(sample, key=key, inference=True)
    patch_spec = eqft.FeatureSpec(
        "forward_features",
        "BNC",
        "patches",
        "mean_patch",
    )
    cls_spec = eqft.FeatureSpec("forward_features", "BNC", "cls", None)

    patches = eqft.extract_features(
        model,
        sample,
        feature_spec=patch_spec,
        key=key,
        inference=True,
    )

    assert jnp.allclose(
        patches,
        jnp.mean(native["x_norm_patchtokens"], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="requires x_norm_cls_token"):
        eqft.extract_features(
            model,
            sample,
            feature_spec=cls_spec,
            key=key,
            inference=True,
        )


@pytest.mark.parametrize(
    "builder",
    (
        pytest.param(_build_mlla, id="raw-token"),
        pytest.param(_build_convnext, id="spatial-map"),
        pytest.param(_build_text_transformer, id="masked-sequence"),
        pytest.param(_build_tabpfn, id="structured-output"),
        pytest.param(_build_deq, id="layer-aggregation"),
    ),
)
def test_feature_contract_groups_support_jit_and_vmap(builder):
    _assert_jit_and_vmap(builder(jr.PRNGKey(30)))
