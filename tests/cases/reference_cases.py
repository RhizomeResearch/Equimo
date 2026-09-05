"""Trusted upstream numerical parity case inventory."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import equimo.audio.models as audio_models
import equimo.tabular.models as tabular_models
import equimo.vision.models as vision_models
from equimo.timeseries.models import t0_alpha


@dataclass(frozen=True)
class ReferenceCase:
    family: str
    conversion_path: str
    identifier: str
    fixture: str
    evaluate: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]


def _ast(identifier, factory):
    def evaluate(reference):
        values = jnp.asarray(reference["input_values"])
        model = factory(pretrained=True)
        features = model.forward_features(values, key=jr.PRNGKey(0), inference=True)
        return {
            "cls_token": np.asarray(features["x_norm_cls_token"]),
            "dist_token": np.asarray(features["x_norm_dist_token"]),
            "logits": np.asarray(model(values, key=jr.PRNGKey(0), inference=True)),
        }

    return ReferenceCase(
        "ast",
        "huggingface-spectrogram",
        identifier,
        f"{identifier}_reference.npz",
        evaluate,
    )


def _convnext(identifier, factory):
    def evaluate(reference):
        image = jnp.asarray(reference["img"])
        model = factory(pretrained=True)
        features = model.features(image, key=jr.PRNGKey(42), inference=True)
        return {
            "features": np.asarray(model.norm(features.mean((1, 2)))),
            "logits": np.asarray(model(image, key=jr.PRNGKey(42), inference=True)),
        }

    return ReferenceCase(
        identifier.split("_")[0],
        "timm",
        identifier,
        f"{identifier}_reference.npz",
        evaluate,
    )


def _dinov2(reference):
    model = vision_models.dinov2_vits14_reg(pretrained=True)
    output = model.forward_features(
        jnp.asarray(reference["img"]),
        key=jr.PRNGKey(42),
        inference=True,
    )
    return {"cls_token": np.asarray(output["x_norm_cls_token"])}


def _dinov3(reference):
    model = vision_models.dinov3_vits16_pretrain_lvd1689m(pretrained=True)
    output = model.forward_features(
        jnp.asarray(reference["img"]),
        key=jr.PRNGKey(42),
        inference=True,
    )
    return {"cls_token": np.asarray(output["x_norm_cls_token"])}


def _eupe(reference):
    model = vision_models.eupe_vitt16(pretrained=True)
    output = model.features(
        jnp.asarray(reference["img"]),
        key=jr.PRNGKey(42),
        inference=True,
    )
    return {"features": np.asarray(output)[None]}


def _siglip2(reference):
    model = vision_models.siglip2_vitb16_256(pretrained=True)
    features = model.features(
        jnp.asarray(reference["img"]),
        key=jr.PRNGKey(42),
        inference=True,
    )
    return {"patch_tokens": np.asarray(jax.vmap(model.norm)(features))}


def _t0(reference):
    model = t0_alpha(pretrained=True, key=jr.PRNGKey(0))
    output = model(
        *(
            jnp.asarray(reference[name])
            for name in ("values", "mask", "group_ids", "variate_type")
        ),
        key=jr.PRNGKey(0),
        inference=True,
    )
    return {"output": np.asarray(output)}


def _tabpfn(reference):
    model = tabular_models.tabpfn_v3_classifier_default(pretrained=True)
    output = model(
        jnp.asarray(reference["x"]),
        jnp.asarray(reference["y"]),
        int(reference["n_train"]),
        key=jr.PRNGKey(42),
        inference=True,
    )
    return {"logits": np.asarray(output)}


REFERENCE_CASES = (
    _convnext("convnext_atto", vision_models.convnext_atto),
    _convnext("convnext_zepto_rms_ols", vision_models.convnext_zepto_rms_ols),
    _convnext("convnextv2_atto", vision_models.convnextv2_atto),
    _ast(
        "ast_base_patch16_audioset_10_10_0_4593",
        audio_models.ast_base_patch16_audioset_10_10_0_4593,
    ),
    _ast(
        "ast_base_patch16_speechcommands_v2_10_10_0_9812",
        audio_models.ast_base_patch16_speechcommands_v2_10_10_0_9812,
    ),
    ReferenceCase(
        "dinov2",
        "timm-vit",
        "dinov2_vits14_reg",
        "dinov2_vits14_reg_reference.npz",
        _dinov2,
    ),
    ReferenceCase(
        "dinov3",
        "huggingface-vit",
        "dinov3_vits16_pretrain_lvd1689m",
        "dinov3_vits16_reference.npz",
        _dinov3,
    ),
    ReferenceCase(
        "eupe",
        "vit",
        "eupe_vitt16",
        "eupe_vitt16_reference.npz",
        _eupe,
    ),
    ReferenceCase(
        "siglip2",
        "huggingface-vit",
        "siglip2_vitb16_256",
        "siglip2_vitb16_256_reference.npz",
        _siglip2,
    ),
    ReferenceCase("t0", "tfc-t0", "t0_alpha", "t0_alpha_reference.npz", _t0),
    ReferenceCase(
        "tabpfn",
        "classifier",
        "tabpfn_v3_classifier_default",
        "tabpfn_v3_classifier_default_reference.npz",
        _tabpfn,
    ),
)
