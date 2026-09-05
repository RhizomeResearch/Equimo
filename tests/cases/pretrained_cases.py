"""Pretrained archive-family and conversion-path inventory."""

from dataclasses import dataclass
from typing import Any

import equimo.audio.models as audio_models
import equimo.tabular.models as tabular_models
import equimo.vision.models as vision_models
from equimo.timeseries.models import t0_alpha


@dataclass(frozen=True)
class PretrainedPathCase:
    family: str
    conversion_path: str
    identifier: str
    factory: Any | None
    reference_fixture: str | None


PRETRAINED_PATH_CASES = (
    PretrainedPathCase(
        "ast",
        "huggingface-spectrogram",
        "ast_base_patch16_audioset_10_10_0_4593",
        audio_models.ast_base_patch16_audioset_10_10_0_4593,
        "ast_base_patch16_audioset_10_10_0_4593_reference.npz",
    ),
    PretrainedPathCase(
        "ast",
        "huggingface-spectrogram",
        "ast_base_patch16_speechcommands_v2_10_10_0_9812",
        audio_models.ast_base_patch16_speechcommands_v2_10_10_0_9812,
        "ast_base_patch16_speechcommands_v2_10_10_0_9812_reference.npz",
    ),
    PretrainedPathCase(
        "convnext",
        "timm",
        "convnext_atto",
        vision_models.convnext_atto,
        "convnext_atto_reference.npz",
    ),
    PretrainedPathCase(
        "convnext",
        "timm",
        "convnext_zepto_rms_ols",
        vision_models.convnext_zepto_rms_ols,
        "convnext_zepto_rms_ols_reference.npz",
    ),
    PretrainedPathCase(
        "convnextv2",
        "timm",
        "convnextv2_atto",
        vision_models.convnextv2_atto,
        "convnextv2_atto_reference.npz",
    ),
    PretrainedPathCase(
        "dinov2",
        "timm-vit",
        "dinov2_vits14_reg",
        vision_models.dinov2_vits14_reg,
        "dinov2_vits14_reg_reference.npz",
    ),
    PretrainedPathCase(
        "dinov3",
        "huggingface-vit",
        "dinov3_vits16_pretrain_lvd1689m",
        vision_models.dinov3_vits16_pretrain_lvd1689m,
        "dinov3_vits16_reference.npz",
    ),
    PretrainedPathCase(
        "eupe",
        "vit",
        "eupe_vitt16",
        vision_models.eupe_vitt16,
        "eupe_vitt16_reference.npz",
    ),
    PretrainedPathCase(
        "eupe",
        "convnext",
        "eupe_convnext_tiny",
        vision_models.eupe_convnext_tiny,
        None,
    ),
    PretrainedPathCase(
        "siglip2",
        "huggingface-vit",
        "siglip2_vitb16_256",
        vision_models.siglip2_vitb16_256,
        "siglip2_vitb16_256_reference.npz",
    ),
    PretrainedPathCase(
        "t0",
        "tfc-t0",
        "t0_alpha",
        t0_alpha,
        "t0_alpha_reference.npz",
    ),
    PretrainedPathCase(
        "tabpfn",
        "classifier",
        "tabpfn_v3_classifier_default",
        tabular_models.tabpfn_v3_classifier_default,
        "tabpfn_v3_classifier_default_reference.npz",
    ),
    PretrainedPathCase(
        "tabpfn",
        "regressor",
        "tabpfn_v3_regressor_default",
        tabular_models.tabpfn_v3_regressor_default,
        None,
    ),
    PretrainedPathCase(
        "tips",
        "vision",
        "tips_vits14_hr",
        vision_models.tips_vits14_hr,
        None,
    ),
    PretrainedPathCase(
        "tips",
        "text",
        "tips_vits14_hr_text",
        None,
        None,
    ),
)


PRETRAINED_FAMILY_PREFIXES = {
    "ast": "ast_",
    "convnext": "convnext_",
    "convnextv2": "convnextv2_",
    "dinov2": "dinov2_",
    "dinov3": "dinov3_",
    "eupe": "eupe_",
    "siglip2": "siglip2_",
    "t0": "t0_",
    "tabpfn": "tabpfn_",
    "tips": "tips_",
}
