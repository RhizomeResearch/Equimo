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
import equimo.finetune as eqft
from equimo.language import sam31_text
from equimo.timeseries.models import t0_alpha
from equimo.vision.models.sam3 import (
    ConceptPrompt,
    InteractivePrompt,
    sam31_multiplex,
)
from equimo.vision.models.sam3.common import flatten_map, sine_position_encoding
from equimo.vision.models.sam3.detector import joint_logit


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
    cls_token = eqft.extract_features(
        model,
        jnp.asarray(reference["img"]),
        feature_spec=eqft.FeatureSpec(
            endpoint="forward_features",
            output_layout="BNC",
            token_selection="cls",
            pooling=None,
        ),
        key=jr.PRNGKey(42),
        inference=True,
    )
    return {"cls_token": np.asarray(cls_token)}


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


def _sam31_interactive_outputs(model, features, prompt, name, num_masks):
    sparse, valid, dense = model.interactive.prompt_encoder(prompt)
    output, tokens = model.interactive.predict_with_tokens(features, prompt)
    if num_masks == 1:
        selected = output.selected
        masks, iou, tokens = (
            output.mask_logits[selected][None],
            output.iou[selected][None],
            tokens[selected][None],
        )
    else:
        masks, iou, tokens = output.mask_logits[1:], output.iou[1:], tokens[1:]
    prefix = f"outputs.interactive.{name}"
    return {
        f"{prefix}.prompt_encoder.0": sparse,
        f"{prefix}.prompt_encoder.valid": valid,
        f"{prefix}.prompt_encoder.1": dense[None],
        f"{prefix}.mask_decoder.0": masks[None],
        f"{prefix}.mask_decoder.1": iou[None],
        f"{prefix}.mask_decoder.2": tokens[None],
        f"{prefix}.mask_decoder.3": output.object_score_logit[None, None],
    }


def _sam31_components(vision, text, image, token_ids, padding_mask, interactive):
    """Evaluate every image-path component in the reference's array layout."""
    text_features = text.features(token_ids, padding_mask)
    trunk = vision.trunk(image)
    features = vision.neck(trunk)
    outputs = {
        "outputs.text.1": text_features[:, None],
        "outputs.text.2": jax.vmap(text.token_embedding)(token_ids)[:, None],
        "outputs.trunk.0": trunk[None],
    }
    for index, pyramid in enumerate(
        (features.detector, features.interactive, features.propagation)
    ):
        for level, value in enumerate(pyramid):
            size = value.shape[-1]
            outputs[f"outputs.neck.{2 * index}.{level}"] = value[None]
            outputs[f"outputs.neck.{2 * index + 1}.{level}"] = sine_position_encoding(
                size, size, value.shape[0] // 2
            )[None]

    high, middle, low = features.detector
    dim, grid, _ = low.shape
    position = flatten_map(sine_position_encoding(grid, grid, dim // 2))
    image_tokens = flatten_map(low)
    prompt = ConceptPrompt.create(
        text_features=text_features, text_padding_mask=padding_mask, dim=dim
    )
    detector = vision.detector
    geometry, geometry_valid = detector.geometry(image_tokens, position, grid, prompt)
    prompt_tokens = jnp.concatenate((text_features, geometry), axis=0)
    prompt_valid = jnp.concatenate((~padding_mask, geometry_valid))
    memory = detector.head.fusion(image_tokens, position, prompt_tokens, prompt_valid)
    decoder = detector.head.decoder
    tokens, boxes, presence = decoder(
        memory, position, prompt_tokens, prompt_valid, grid
    )
    scores = jax.vmap(detector.head.scoring, in_axes=(0, None, None))(
        tokens, prompt_tokens, prompt_valid
    )
    pyramid = (high, middle, low)
    segmentation = detector.head.segmentation
    reference_boxes = jax.nn.sigmoid(decoder.reference_boxes)[None]
    joint = joint_logit(scores[-1], presence[-1, None])
    outputs |= {
        "outputs.detector.geometry.0": geometry[-1:, None],
        "outputs.detector.head.fusion.memory": memory[:, None],
        "outputs.detector.head.decoder.0": tokens[:, :, None],
        "outputs.detector.head.decoder.1": jnp.concatenate(
            (reference_boxes, boxes), axis=0
        )[:, :, None],
        "outputs.detector.head.decoder.2": presence[:, None, None],
        "outputs.detector.head.scoring": scores[:, None, :, None],
        "outputs.detector.head.segmentation.pred_masks": segmentation(
            pyramid, memory, tokens[-1], prompt_tokens, prompt_valid
        )[None],
        "outputs.detector.joint.pred_logits": joint[None, :, None],
        "outputs.detector.joint.pred_boxes": boxes[-1][None],
        "outputs.detector.joint.presence_logit_dec": presence[-1, None, None],
    }
    for name, (interactive_prompt, num_masks) in interactive.items():
        outputs |= _sam31_interactive_outputs(
            vision, features, interactive_prompt, name, num_masks
        )
    return outputs


def _sam31_image(reference):
    pixels = jnp.asarray(reference["inputs.image"], jnp.float32) / 255.0
    image = jnp.moveaxis((pixels - 0.5) / 0.5, -1, 0)
    prompts = {
        name: InteractivePrompt.create(
            box=(
                jnp.asarray(reference[f"inputs.interactive.{name}.points"]).ravel()
                * 1008
                if name == "box"
                else None
            ),
            points=(
                jnp.asarray(reference[f"inputs.interactive.{name}.points"]) * 1008
                if name == "point"
                else None
            ),
            point_labels=(
                jnp.asarray(reference[f"inputs.interactive.{name}.labels"])
                if name == "point"
                else None
            ),
            mask_size=288,
        )
        for name in ("point", "box")
    }
    interactive = {
        name: (
            prompt,
            reference[f"outputs.interactive.{name}.mask_decoder.1.values"].shape[1],
        )
        for name, prompt in prompts.items()
    }
    vision = sam31_multiplex(pretrained=True)
    text = sam31_text(pretrained=True)
    outputs = _sam31_components(
        vision,
        text,
        image,
        jnp.asarray(reference["inputs.text.token_ids"][0], jnp.int32),
        jnp.asarray(reference["inputs.text.padding_mask"][0]),
        interactive,
    )
    parts: dict[str, np.ndarray] = {}
    for name, value in outputs.items():
        if name.endswith(".valid"):
            continue
        array = np.asarray(value)
        valid = outputs.get(f"{name[: -len('.0')]}.valid")
        if name.endswith(".0") and valid is not None:
            array = array[np.asarray(valid)][None]
        indices = reference.get(f"{name}.indices")
        if indices is not None:
            array = array.ravel()[indices]
        parts[f"{name}.values"] = array
    return parts


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
    ReferenceCase(
        "sam31",
        "vision",
        "sam31_multiplex",
        "sam31_image_reference.npz",
        _sam31_image,
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
