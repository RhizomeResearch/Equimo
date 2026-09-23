"""Canonical tiny-model cases shared by cross-cutting contract tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import equimo.finetune as eqft
from equimo.audio.models import AudioSpectrogramTransformer
from equimo.language.models import TextTransformerEncoder
from equimo.tabular.models import TabPFN
from equimo.timeseries.models import T0
from equimo.vision.models import (
    AttNet,
    ConvNeXt,
    DEQ,
    EoMT,
    PMT,
    PMTConfig,
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
class ModelInvocation:
    model: Any
    args: tuple[jax.Array, ...]
    kwargs: dict[str, Any]
    spec: eqft.FeatureSpec
    expected: jax.Array
    key: jax.Array


@dataclass(frozen=True)
class ModelCase:
    modality: str
    registry_name: str
    build: Callable[[jax.Array], ModelInvocation]
    supports_low_precision: bool = True
    low_precision_limitation: str | None = None
    onnx_batched: bool = True
    onnx_batch_limitation: str | None = None


def _keys(key: jax.Array, count: int = 2) -> tuple[jax.Array, ...]:
    return tuple(jr.split(key, count))


def _token_invocation(
    model: Any,
    sample: jax.Array,
    *,
    key: jax.Array,
) -> ModelInvocation:
    tokens = model.features(sample, key=key, inference=True)
    return ModelInvocation(
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
) -> ModelInvocation:
    feature_map = model.features(sample, key=key, inference=True)
    return ModelInvocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec("features", "BCHW", "all", "global_avg"),
        expected=jnp.mean(feature_map, axis=(1, 2)),
        key=key,
    )


def _build_vit(key: jax.Array) -> ModelInvocation:
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
    return ModelInvocation(
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


def _build_eomt(key: jax.Array) -> ModelInvocation:
    backbone_key, model_key, sample_key = jr.split(key, 3)
    backbone = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[2],
        reg_tokens=1,
        num_classes=0,
        dynamic_img_size=True,
        key=backbone_key,
    )
    model = EoMT(
        backbone, 3, num_queries=2, num_blocks=1, key=model_key, masked_attention=False
    )
    sample = jr.normal(sample_key, (3, 16, 16))
    tokens = model.features(sample, key=model_key, inference=True)
    return ModelInvocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec("features", "BNC", "all", "global_avg"),
        expected=jnp.mean(tokens, axis=0),
        key=model_key,
    )


def _build_pmt(key: jax.Array) -> ModelInvocation:
    backbone_key, model_key, sample_key = jr.split(key, 3)
    backbone = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[2],
        reg_tokens=1,
        num_classes=0,
        dynamic_img_size=True,
        use_local_pos_embed=True,
        global_pos_embed_reg=True,
        key=backbone_key,
    )
    model = PMT(
        backbone,
        3,
        key=model_key,
        config=PMTConfig.small(
            dim=8,
            num_heads=2,
            hidden_dim=8,
            taps=(0, 1),
            num_queries=2,
            num_blocks=2,
        ),
    )
    sample = jr.normal(sample_key, (3, 16, 16))
    tokens = model.features(sample)
    return ModelInvocation(
        model=model,
        args=(sample,),
        kwargs={},
        spec=eqft.FeatureSpec("features", "BNC", "all", "global_avg"),
        expected=jnp.mean(tokens, axis=0),
        key=model_key,
    )


def _build_vision_parcae(key: jax.Array) -> ModelInvocation:
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
        state_init="zero",
        num_classes=0,
        key=model_key,
    )
    sample = jr.normal(sample_key, (3, 16, 16))
    native = model.forward_features(sample, key=model_key, inference=True)
    return ModelInvocation(
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


def _build_ast(key: jax.Array) -> ModelInvocation:
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
    return ModelInvocation(
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


def _build_fastervit(key: jax.Array) -> ModelInvocation:
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


def _build_partialformer(key: jax.Array) -> ModelInvocation:
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


def _build_mlla(key: jax.Array) -> ModelInvocation:
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


def _build_vssd(key: jax.Array) -> ModelInvocation:
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


def _build_attnet(key: jax.Array) -> ModelInvocation:
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


def _build_convnext(key: jax.Array) -> ModelInvocation:
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


def _build_iformer(key: jax.Array) -> ModelInvocation:
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


def _build_lowformer(key: jax.Array) -> ModelInvocation:
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


def _build_mobilenetv3(key: jax.Array) -> ModelInvocation:
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


def _build_reduceformer(key: jax.Array) -> ModelInvocation:
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


def _build_shvit(key: jax.Array) -> ModelInvocation:
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


def _build_text_transformer(key: jax.Array) -> ModelInvocation:
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
    return ModelInvocation(
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


def _build_tabpfn(key: jax.Array) -> ModelInvocation:
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
    return ModelInvocation(
        model=model,
        args=(x, y),
        kwargs={"n_train": 3},
        spec=eqft.FeatureSpec("__call__", "BC", "all", None),
        expected=model(x, y, n_train=3, key=model_key, inference=True),
        key=model_key,
    )


def _build_t0(key: jax.Array) -> ModelInvocation:
    model_key, sample_key = _keys(key)
    model = T0(
        embed_dim=8,
        num_layers=2,
        num_heads=2,
        mlp_hidden_dim=16,
        patch_size=4,
        group_every_n=2,
        dropout=0.0,
        key=model_key,
    )
    values = jr.normal(sample_key, (2, 8))
    mask = jnp.zeros(values.shape, dtype=jnp.int8)
    group_ids = jnp.broadcast_to(
        jnp.arange(values.shape[0], dtype=jnp.int32)[:, None],
        values.shape,
    )
    variate_type = jnp.zeros(values.shape, dtype=jnp.int32)
    tokens = model.features(
        values,
        mask,
        group_ids,
        variate_type,
        key=model_key,
        inference=True,
    )
    return ModelInvocation(
        model=model,
        args=(values, mask, group_ids, variate_type),
        kwargs={},
        spec=eqft.FeatureSpec("features", "BNC", "all", "mean_token"),
        expected=jnp.mean(tokens, axis=0),
        key=model_key,
    )


def _build_deq(key: jax.Array) -> ModelInvocation:
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
    return ModelInvocation(
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


MODEL_CASES = (
    ModelCase("audio", "ast", _build_ast),
    ModelCase("language", "text_transformer_encoder", _build_text_transformer),
    ModelCase("tabular", "tabpfn", _build_tabpfn),
    ModelCase("timeseries", "t0", _build_t0),
    ModelCase("vision", "attnet", _build_attnet),
    ModelCase("vision", "convnext", _build_convnext),
    ModelCase(
        "vision",
        "deq",
        _build_deq,
        supports_low_precision=False,
        low_precision_limitation=(
            "banax 0.1.2 initializes solver error state as float32, which is "
            "incompatible with bfloat16 loop updates"
        ),
    ),
    ModelCase("vision", "fastervit", _build_fastervit),
    ModelCase("vision", "iformer", _build_iformer),
    ModelCase("vision", "lowformer", _build_lowformer),
    ModelCase("vision", "mlla", _build_mlla),
    ModelCase("vision", "mobilenetv3", _build_mobilenetv3),
    ModelCase(
        "vision",
        "partialformer",
        _build_partialformer,
        onnx_batched=False,
        onnx_batch_limitation=(
            "jax2onnx 0.15.1 vmapped lowering drifts beyond the established "
            "runtime tolerance; unbatched lowering retains parity"
        ),
    ),
    ModelCase("vision", "reduceformer", _build_reduceformer),
    ModelCase("vision", "shvit", _build_shvit),
    ModelCase("vision", "vision_parcae", _build_vision_parcae),
    ModelCase("vision", "vit", _build_vit),
    ModelCase("vision", "vssd", _build_vssd),
    ModelCase("vision", "eomt", _build_eomt),
    ModelCase("vision", "pmt", _build_pmt),
)


def extract_features(
    invocation: ModelInvocation,
    *args: jax.Array,
    key: jax.Array,
) -> jax.Array:
    return eqft.extract_features(
        invocation.model,
        *args,
        feature_spec=invocation.spec,
        key=key,
        inference=True,
        **invocation.kwargs,
    )


def assert_jit_and_vmap(invocation: ModelInvocation) -> None:
    def extract_one(*call_args):
        return extract_features(invocation, *call_args[:-1], key=call_args[-1])

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


def assert_gradients_and_low_precision(
    invocation: ModelInvocation,
    *,
    supports_low_precision: bool,
) -> None:
    def loss(model):
        updated = replace(invocation, model=model)
        output = extract_features(updated, *updated.args, key=updated.key)
        return jnp.sum(output.astype(jnp.float32))

    gradients = eqx.filter_grad(loss)(invocation.model)
    array_gradients = [
        leaf for leaf in jax.tree.leaves(gradients) if eqx.is_inexact_array(leaf)
    ]
    assert array_gradients
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in array_gradients)

    if not supports_low_precision:
        return

    def to_bfloat16(value):
        return value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value

    low_precision = replace(
        invocation,
        model=jax.tree.map(to_bfloat16, invocation.model),
        args=tuple(jax.tree.map(to_bfloat16, arg) for arg in invocation.args),
    )
    output = extract_features(
        low_precision,
        *low_precision.args,
        key=low_precision.key,
    )
    assert output.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(output))


TRANSFORM_CASES = tuple(
    case
    for case in MODEL_CASES
    if case.registry_name
    in {"mlla", "convnext", "text_transformer_encoder", "tabpfn", "t0", "deq"}
)
