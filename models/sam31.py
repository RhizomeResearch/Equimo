"""Convert the SAM 3.1 multiplex checkpoint to Equimo archives.

Every key either converts to named Equimo parameters, is computed by Equimo and
dropped, or is unused and dropped. A key outside those lists is an error. The
original PyTorch format is read only by this maintainer script, which runs in
the ``reference`` dependency group. The tests apply the same rules to the
reference fixtures through :func:`apply_state`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)

import numpy as np  # noqa: E402


CHECKPOINT_NAME = "sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
ARCHIVES = {"multiplex": "sam31_multiplex", "text": "sam31_text"}

VISION = "detector.backbone.vision_backbone"
TEXT = "detector.backbone.language_backbone"
GEOMETRY = "detector.geometry_encoder"
FUSION = "detector.transformer.encoder.layers.{i}"
DECODER = "detector.transformer.decoder"
SCORING = "detector.dot_prod_scoring"
SEGMENTATION = "detector.segmentation_head"
TRACKER = "tracker.model"
INTERACTIVE_DECODER = f"{TRACKER}.interactive_sam_mask_decoder"
MULTIPLEX_DECODER = f"{TRACKER}.sam_mask_decoder"
PROMPT_ENCODER = f"{TRACKER}.interactive_sam_prompt_encoder"
MEMORY = f"{TRACKER}.maskmem_backbone"

UNUSED_KEYS = frozenset(
    {
        f"{TEXT}.encoder.text_projection",
        f"{SEGMENTATION}.pixel_decoder.conv_layers.2.weight",
        f"{SEGMENTATION}.pixel_decoder.conv_layers.2.bias",
        f"{SEGMENTATION}.pixel_decoder.norms.2.weight",
        f"{SEGMENTATION}.pixel_decoder.norms.2.bias",
        f"{SEGMENTATION}.semantic_seg_head.weight",
        f"{SEGMENTATION}.semantic_seg_head.bias",
    }
)
COMPUTED_KEY = re.compile(re.escape(VISION) + r"\.trunk\.blocks\.\d+\.attn\.freqs_cis")

Shape = tuple[int, ...]
LAYOUTS: dict[str, Callable[[Shape], Shape]] = {
    "copy": lambda shape: shape,
    "split_qkv": lambda shape: (shape[0] // 3, *shape[1:]),
    "drop_cls_row": lambda shape: (shape[1] - 1, shape[2]),
    "transposed_conv": lambda shape: (shape[1], shape[0], *shape[2:]),
    "conv_bias": lambda shape: (shape[0], 1, 1),
    "linear_as_conv": lambda shape: (*shape, 1, 1),
    "squeeze": lambda shape: tuple(size for size in shape if size != 1),
    "stack_of_four": lambda shape: (4, shape[1]),
}


class Rule(NamedTuple):
    source: str
    pattern: re.Pattern[str]
    destinations: tuple[str, ...]
    layout: str
    archive: str


class Conversion(NamedTuple):
    archive: str
    destinations: tuple[str, ...]
    layout: str


class Classification(NamedTuple):
    converted: dict[str, Conversion]
    computed: list[str]
    unused: list[str]


def _rule(
    source: str,
    destinations: str | tuple[str, ...],
    layout: str = "copy",
    archive: str = "multiplex",
) -> Rule:
    """Map one checkpoint key, where ``{i}`` and ``{j}`` stand for module indices."""
    if isinstance(destinations, str):
        destinations = (destinations,)
    pattern = re.compile(
        re.escape(source).replace(r"\{i\}", r"(\d+)").replace(r"\{j\}", r"(\d+)")
    )
    return Rule(source, pattern, destinations, layout, archive)


def _linear(source: str, destination: str, archive: str = "multiplex") -> list[Rule]:
    return [
        _rule(f"{source}.{leaf}", f"{destination}.{leaf}", archive=archive)
        for leaf in ("weight", "bias")
    ]


def _conv(source: str, destination: str) -> list[Rule]:
    return [
        _rule(f"{source}.weight", f"{destination}.weight"),
        _rule(f"{source}.bias", f"{destination}.bias", "conv_bias"),
    ]


def _linear_as_conv(source: str, destination: str) -> list[Rule]:
    return [
        _rule(f"{source}.weight", f"{destination}.weight", "linear_as_conv"),
        _rule(f"{source}.bias", f"{destination}.bias", "conv_bias"),
    ]


def _transposed_conv(source: str, destination: str) -> list[Rule]:
    return [
        _rule(f"{source}.weight", f"{destination}.weight", "transposed_conv"),
        _rule(f"{source}.bias", f"{destination}.bias", "conv_bias"),
    ]


def _packed_attention(source: str, destination: str) -> list[Rule]:
    return [
        _rule(
            f"{source}.in_proj_{leaf}",
            tuple(f"{destination}.{part}.{leaf}" for part in "qkv"),
            "split_qkv",
        )
        for leaf in ("weight", "bias")
    ] + _linear(f"{source}.out_proj", f"{destination}.proj")


def _separate_attention(source_prefix: str, destination: str) -> list[Rule]:
    rules = _linear(f"{source_prefix}out_proj", f"{destination}.proj")
    for part in "qkv":
        rules += _linear(f"{source_prefix}{part}_proj", f"{destination}.{part}")
    return rules


def _feed_forward(block: str, destination: str) -> list[Rule]:
    return [
        *_linear(f"{block}.linear1", f"{destination}.mlp.fc1"),
        *_linear(f"{block}.linear2", f"{destination}.mlp.fc2"),
    ]


def _norms(block: str, destination: str, names: Iterable[str]) -> list[Rule]:
    rules: list[Rule] = []
    for name in names:
        rules += _linear(f"{block}.{name}", f"{destination}.{name}")
    return rules


def _text_rules() -> list[Rule]:
    encoder = f"{TEXT}.encoder"
    block = encoder + ".transformer.resblocks.{i}"
    target = "transformer.blocks.{i}"
    return [
        _rule(f"{encoder}.positional_embedding", "position_embedding", archive="text"),
        _rule(
            f"{encoder}.token_embedding.weight",
            "token_embedding.weight",
            archive="text",
        ),
        _rule(
            f"{block}.attn.in_proj_weight", f"{target}.attn.qkv.weight", archive="text"
        ),
        _rule(f"{block}.attn.in_proj_bias", f"{target}.attn.qkv.bias", archive="text"),
        *_linear(f"{encoder}.ln_final", "ln_final", "text"),
        *_linear(f"{TEXT}.resizer", "projection", "text"),
        *_linear(f"{block}.attn.out_proj", f"{target}.attn.proj", "text"),
        *_linear(f"{block}.ln_1", f"{target}.prenorm", "text"),
        *_linear(f"{block}.ln_2", f"{target}.norm", "text"),
        *_linear(f"{block}.mlp.c_fc", f"{target}.mlp.fc1", "text"),
        *_linear(f"{block}.mlp.c_proj", f"{target}.mlp.fc2", "text"),
    ]


def _trunk_rules() -> list[Rule]:
    trunk = f"{VISION}.trunk"
    block = trunk + ".blocks.{i}"
    target = "trunk.blocks.{i}"
    rules = [
        _rule(f"{trunk}.pos_embed", "trunk.pos_embed", "drop_cls_row"),
        _rule(f"{trunk}.patch_embed.proj.weight", "trunk.patch_embed.weight"),
        *_linear(f"{trunk}.ln_pre", "trunk.ln_pre"),
    ]
    for leaf in ("weight", "bias"):
        rules.append(
            _rule(
                f"{block}.attn.qkv.{leaf}",
                tuple(f"{target}.attn.{part}.{leaf}" for part in "qkv"),
                "split_qkv",
            )
        )
    for name in ("attn.proj", "mlp.fc1", "mlp.fc2", "norm1", "norm2"):
        rules += _linear(f"{block}.{name}", f"{target}.{name}")
    return rules


def _neck_rules() -> list[Rule]:
    rules: list[Rule] = []
    for source, neck in (
        ("convs", "detector"),
        ("interactive_convs", "interactive"),
        ("propagation_convs", "propagation"),
    ):
        level = f"{VISION}.{source}" + ".{i}"
        target = f"neck.{neck}" + ".levels.{i}"
        rules += [
            *_conv(f"{level}.conv_1x1", f"{target}.proj"),
            *_conv(f"{level}.conv_3x3", f"{target}.conv"),
            *_transposed_conv(f"{level}.dconv_2x2", f"{target}.upsample.0"),
            *_transposed_conv(f"{level}.dconv_2x2_0", f"{target}.upsample.0"),
            *_transposed_conv(f"{level}.dconv_2x2_1", f"{target}.upsample.1"),
        ]
    return rules


def _detector_rules() -> list[Rule]:
    geometry = "detector.geometry"
    geometry_block = f"{GEOMETRY}" + ".encode.{i}"
    geometry_target = f"{geometry}.blocks" + ".{i}"
    fusion_target = "detector.head.fusion.blocks.{i}"
    decoder_block = f"{DECODER}" + ".layers.{i}"
    decoder_target = "detector.head.decoder.blocks.{i}"
    decoder = "detector.head.decoder"
    segmentation = "detector.head.segmentation"
    return [
        _rule(f"{GEOMETRY}.label_embed.weight", f"{geometry}.label_embed"),
        _rule(f"{GEOMETRY}.cls_embed.weight", f"{geometry}.cls_token", "squeeze"),
        *_linear(f"{GEOMETRY}.points_direct_project", f"{geometry}.point_direct"),
        *_linear(f"{GEOMETRY}.points_pool_project", f"{geometry}.point_pool"),
        *_linear(f"{GEOMETRY}.points_pos_enc_project", f"{geometry}.point_position"),
        *_linear(f"{GEOMETRY}.boxes_direct_project", f"{geometry}.box_direct"),
        *_conv(f"{GEOMETRY}.boxes_pool_project", f"{geometry}.box_pool"),
        *_linear(f"{GEOMETRY}.boxes_pos_enc_project", f"{geometry}.box_position"),
        *_linear(f"{GEOMETRY}.final_proj", f"{geometry}.final_proj"),
        *_linear(f"{GEOMETRY}.norm", f"{geometry}.final_norm"),
        *_linear(f"{GEOMETRY}.img_pre_norm", f"{geometry}.image_norm"),
        *_linear(f"{GEOMETRY}.encode_norm", f"{geometry}.norm"),
        *_packed_attention(f"{geometry_block}.self_attn", f"{geometry_target}.attn"),
        *_packed_attention(
            f"{geometry_block}.cross_attn_image", f"{geometry_target}.cross_attn"
        ),
        *_feed_forward(geometry_block, geometry_target),
        *_norms(geometry_block, geometry_target, ("norm1", "norm2", "norm3")),
        *_packed_attention(f"{FUSION}.self_attn", f"{fusion_target}.attn"),
        *_packed_attention(f"{FUSION}.cross_attn_image", f"{fusion_target}.cross_attn"),
        *_feed_forward(FUSION, fusion_target),
        *_norms(FUSION, fusion_target, ("norm1", "norm2", "norm3")),
        *_packed_attention(f"{decoder_block}.self_attn", f"{decoder_target}.attn"),
        *_packed_attention(
            f"{decoder_block}.ca_text", f"{decoder_target}.cross_attn.0"
        ),
        *_packed_attention(
            f"{decoder_block}.cross_attn", f"{decoder_target}.cross_attn.1"
        ),
        *_feed_forward(decoder_block, decoder_target),
        *_linear(f"{decoder_block}.norm2", f"{decoder_target}.norm1"),
        *_linear(f"{decoder_block}.catext_norm", f"{decoder_target}.norm2"),
        *_linear(f"{decoder_block}.norm1", f"{decoder_target}.norm3"),
        *_linear(f"{decoder_block}.norm3", f"{decoder_target}.norm4"),
        _rule(f"{DECODER}.query_embed.weight", f"{decoder}.queries"),
        _rule(f"{DECODER}.reference_points.weight", f"{decoder}.reference_boxes"),
        _rule(
            f"{DECODER}.presence_token.weight", f"{decoder}.presence_token", "squeeze"
        ),
        *_linear(f"{DECODER}.norm", f"{decoder}.norm"),
        *_linear(f"{DECODER}.presence_token_out_norm", f"{decoder}.presence_norm"),
        *_linear(
            f"{DECODER}.bbox_embed.layers" + ".{i}",
            f"{decoder}.box_head.layers" + ".{i}",
        ),
        *_linear(
            f"{DECODER}.ref_point_head.layers" + ".{i}",
            f"{decoder}.position_proj.layers" + ".{i}",
        ),
        *_linear(
            f"{DECODER}.boxRPB_embed_x.layers" + ".{i}",
            f"{decoder}.relative_bias.embed_x.layers" + ".{i}",
        ),
        *_linear(
            f"{DECODER}.boxRPB_embed_y.layers" + ".{i}",
            f"{decoder}.relative_bias.embed_y.layers" + ".{i}",
        ),
        *_linear(
            f"{DECODER}.presence_token_head.layers" + ".{i}",
            f"{decoder}.presence_head.layers" + ".{i}",
        ),
        *_linear(
            f"{SCORING}.prompt_mlp.layers" + ".{i}",
            "detector.head.scoring.prompt_mlp.layers.{i}",
        ),
        *_linear(f"{SCORING}.prompt_mlp.out_norm", "detector.head.scoring.prompt_norm"),
        *_linear(f"{SCORING}.hs_proj", "detector.head.scoring.query_proj"),
        *_linear(f"{SCORING}.prompt_proj", "detector.head.scoring.prompt_proj"),
        *_packed_attention(
            f"{SEGMENTATION}.cross_attend_prompt", f"{segmentation}.cross_attn"
        ),
        *_linear(f"{SEGMENTATION}.cross_attn_norm", f"{segmentation}.norm"),
        *_conv(f"{SEGMENTATION}.instance_seg_head", f"{segmentation}.instance_proj"),
        *_linear(
            f"{SEGMENTATION}.mask_predictor.mask_embed.layers" + ".{i}",
            f"{segmentation}.mask_embed.layers" + ".{i}",
        ),
        *_conv(
            f"{SEGMENTATION}.pixel_decoder.conv_layers" + ".{i}",
            f"{segmentation}.stages" + ".{i}.conv",
        ),
        *_linear(
            f"{SEGMENTATION}.pixel_decoder.norms" + ".{i}",
            f"{segmentation}.stages" + ".{i}.norm",
        ),
    ]


def _mask_decoder_rules(source: str, destination: str) -> list[Rule]:
    """Rules shared by the interactive and multiplex mask decoders."""
    block = f"{source}.transformer" + ".layers.{i}"
    target = f"{destination}.transformer" + ".blocks.{i}"
    return [
        *_conv(f"{source}.conv_s0", f"{destination}.upsampler.conv_s0"),
        *_conv(f"{source}.conv_s1", f"{destination}.upsampler.conv_s1"),
        *_transposed_conv(
            f"{source}.output_upscaling.0", f"{destination}.upsampler.deconvs.0"
        ),
        *_linear(f"{source}.output_upscaling.1", f"{destination}.upsampler.norm"),
        *_transposed_conv(
            f"{source}.output_upscaling.3", f"{destination}.upsampler.deconvs.1"
        ),
        *_linear(
            f"{source}.output_hypernetworks_mlps" + ".{i}.layers.{j}",
            f"{destination}.mask_embed" + ".{i}.layers.{j}",
        ),
        *_linear(
            f"{source}.iou_prediction_head.layers" + ".{i}",
            f"{destination}.iou_head.layers" + ".{i}",
        ),
        *_linear(
            f"{source}.pred_obj_score_head.layers" + ".{i}",
            f"{destination}.object_head.layers" + ".{i}",
        ),
        *_linear(
            f"{source}.transformer.norm_final_attn",
            f"{destination}.transformer.norm",
        ),
        *_separate_attention(
            f"{source}.transformer.final_attn_token_to_image.",
            f"{destination}.transformer.cross_attn",
        ),
        *_separate_attention(f"{block}.self_attn.", f"{target}.attn"),
        *_separate_attention(
            f"{block}.cross_attn_token_to_image.", f"{target}.cross_attn.0"
        ),
        *_separate_attention(
            f"{block}.cross_attn_image_to_token.", f"{target}.cross_attn.1"
        ),
        *_linear(f"{block}.mlp.lin1", f"{target}.mlp.fc1"),
        *_linear(f"{block}.mlp.lin2", f"{target}.mlp.fc2"),
        *_norms(block, target, ("norm1", "norm2", "norm3", "norm4")),
    ]


def _interactive_rules() -> list[Rule]:
    encoder = "interactive.prompt_encoder"
    decoder = "interactive.mask_decoder"
    downscaling = f"{PROMPT_ENCODER}.mask_downscaling"
    return [
        _rule(
            f"{TRACKER}.interactivity_no_mem_embed",
            "interactive.no_memory_embed",
            "squeeze",
        ),
        _rule(
            f"{PROMPT_ENCODER}.pe_layer.positional_encoding_gaussian_matrix",
            f"{encoder}.fourier_matrix",
        ),
        _rule(
            f"{PROMPT_ENCODER}.point_embeddings" + ".{i}.weight",
            f"{encoder}.point_embeddings",
            "stack_of_four",
        ),
        _rule(
            f"{PROMPT_ENCODER}.not_a_point_embed.weight",
            f"{encoder}.not_a_point",
            "squeeze",
        ),
        _rule(
            f"{PROMPT_ENCODER}.no_mask_embed.weight",
            f"{encoder}.no_mask_embed",
            "squeeze",
        ),
        *_conv(f"{downscaling}.0", f"{encoder}.mask_convs.0"),
        *_linear(f"{downscaling}.1", f"{encoder}.mask_norms.0"),
        *_conv(f"{downscaling}.3", f"{encoder}.mask_convs.1"),
        *_linear(f"{downscaling}.4", f"{encoder}.mask_norms.1"),
        *_conv(f"{downscaling}.6", f"{encoder}.mask_convs.2"),
        _rule(
            f"{INTERACTIVE_DECODER}.iou_token.weight", f"{decoder}.iou_token", "squeeze"
        ),
        _rule(f"{INTERACTIVE_DECODER}.mask_tokens.weight", f"{decoder}.mask_tokens"),
        _rule(
            f"{INTERACTIVE_DECODER}.obj_score_token.weight",
            f"{decoder}.object_token",
            "squeeze",
        ),
        *_mask_decoder_rules(INTERACTIVE_DECODER, decoder),
    ]


def _memory_downsampler_rules() -> list[Rule]:
    downsampler = f"{MEMORY}.mask_downsampler.encoder"
    target = "tracker.memory_encoder"
    rules = _conv(f"{downsampler}.12", f"{target}.mask_proj")
    for stage in range(4):
        rules += _conv(f"{downsampler}.{3 * stage}", f"{target}.mask_convs.{stage}")
        rules += _linear(
            f"{downsampler}.{3 * stage + 1}", f"{target}.mask_norms.{stage}"
        )
    return rules


def _tracker_rules() -> list[Rule]:
    fuser = f"{MEMORY}.fuser.layers" + ".{i}"
    fuser_target = "tracker.memory_encoder.fuser.{i}"
    attention = f"{TRACKER}.transformer.encoder"
    block = attention + ".layers.{i}"
    target = "tracker.memory_attention.blocks.{i}"
    decoder = "tracker.mask_decoder"
    return [
        _rule(f"{TRACKER}.output_valid_embed", f"{decoder}.valid_embed"),
        _rule(f"{TRACKER}.output_invalid_embed", f"{decoder}.invalid_embed"),
        _rule(f"{MULTIPLEX_DECODER}.iou_token.weight", f"{decoder}.iou_token"),
        _rule(f"{MULTIPLEX_DECODER}.obj_score_token.weight", f"{decoder}.object_token"),
        _rule(f"{MULTIPLEX_DECODER}.mask_tokens.weight", f"{decoder}.mask_tokens"),
        _rule(f"{TRACKER}.maskmem_tpos_enc", "tracker.memory_time_embed", "squeeze"),
        _rule(
            f"{TRACKER}.image_pe_layer.positional_encoding_gaussian_matrix",
            "tracker.image_position.fourier_matrix",
        ),
        *_linear(
            f"{TRACKER}.obj_ptr_proj.layers" + ".{i}",
            "tracker.pointer_proj.layers.{i}",
        ),
        *_linear(f"{TRACKER}.obj_ptr_tpos_proj", "tracker.pointer_time_proj"),
        *_linear(
            f"{TRACKER}.interactive_obj_ptr_proj.layers" + ".{i}",
            "tracker.interactive_pointer_proj.layers.{i}",
        ),
        *_linear(f"{TRACKER}.no_obj_ptr_linear", "tracker.no_object_pointer"),
        _rule(f"{TRACKER}.no_obj_embed_spatial", "tracker.no_object_memory"),
        *_conv(
            f"{TRACKER}.interactive_mask_downsample",
            "tracker.interactive_mask_downsample",
        ),
        *_memory_downsampler_rules(),
        *_conv(f"{MEMORY}.pix_feat_proj", "tracker.memory_encoder.feature_proj"),
        _rule(f"{fuser}.gamma", f"{fuser_target}.ls.gamma"),
        *_conv(f"{fuser}.dwconv", f"{fuser_target}.dwconv"),
        *_linear_as_conv(f"{fuser}.pwconv1", f"{fuser_target}.pwconv1"),
        *_linear_as_conv(f"{fuser}.pwconv2", f"{fuser_target}.pwconv2"),
        *_linear(f"{fuser}.norm", f"{fuser_target}.norm"),
        *_linear(f"{attention}.norm", "tracker.memory_attention.norm"),
        *_separate_attention(f"{block}.self_attn_", f"{target}.attn"),
        *_separate_attention(f"{block}.cross_attn_", f"{target}.cross_attn"),
        *_linear(f"{block}.image_cross_attn_q_proj", f"{target}.image_q"),
        *_linear(f"{block}.image_cross_attn_k_proj", f"{target}.image_k"),
        *_feed_forward(block, target),
        *_norms(block, target, ("norm1", "norm2", "norm3")),
        *_mask_decoder_rules(MULTIPLEX_DECODER, "tracker.mask_decoder"),
    ]


RULES = (
    *_text_rules(),
    *_trunk_rules(),
    *_neck_rules(),
    *_detector_rules(),
    *_interactive_rules(),
    *_tracker_rules(),
)


def _fill_indices(destination: str, indices: tuple[str, ...]) -> str:
    for placeholder, index in zip(("{i}", "{j}"), indices):
        destination = destination.replace(placeholder, index)
    return destination


def _conversion_for(key: str) -> Conversion | None:
    for rule in RULES:
        match = rule.pattern.fullmatch(key)
        if match is None:
            continue
        destinations = tuple(
            _fill_indices(destination, match.groups())
            for destination in rule.destinations
        )
        return Conversion(rule.archive, destinations, rule.layout)
    return None


def classify_keys(keys: Iterable[str]) -> Classification:
    """Sort checkpoint keys by outcome, failing on keys no list accounts for."""
    converted: dict[str, Conversion] = {}
    computed: list[str] = []
    unused: list[str] = []
    unknown: list[str] = []
    for key in keys:
        if key in UNUSED_KEYS:
            unused.append(key)
        elif COMPUTED_KEY.fullmatch(key):
            computed.append(key)
        elif (conversion := _conversion_for(key)) is not None:
            converted[key] = conversion
        else:
            unknown.append(key)
    if unknown:
        listing = "\n".join(unknown)
        raise ValueError(f"Unknown checkpoint keys:\n{listing}")
    return Classification(converted, computed, unused)


def check_destinations(
    classification: Classification,
    source_shapes: Mapping[str, Shape],
    template_shapes: Mapping[str, Shape],
    archive: str,
) -> list[str]:
    """Check converted shapes and return the template parameters with no source."""
    covered: set[str] = set()
    for key, conversion in classification.converted.items():
        if conversion.archive != archive:
            continue
        expected = LAYOUTS[conversion.layout](source_shapes[key])
        for destination in conversion.destinations:
            if destination not in template_shapes:
                raise ValueError(f"{key} targets missing parameter {destination}.")
            if template_shapes[destination] != expected:
                raise ValueError(
                    f"{key} converts to shape {expected}, but {destination} has "
                    f"shape {template_shapes[destination]}."
                )
            covered.add(destination)
    return sorted(set(template_shapes) - covered)


CONVERTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "copy": lambda array: array,
    "drop_cls_row": lambda array: array[0, 1:],
    # Equinox correlates the stored kernel with the dilated input.
    "transposed_conv": lambda array: np.swapaxes(array, 0, 1)[:, :, ::-1, ::-1].copy(),
    "conv_bias": lambda array: array.reshape(array.shape[0], 1, 1),
    "linear_as_conv": lambda array: array.reshape(*array.shape, 1, 1),
    "squeeze": np.squeeze,
}


def convert_state(
    state: Mapping[str, np.ndarray], archive: str
) -> dict[str, np.ndarray]:
    """Return each Equimo parameter of ``archive`` from a checkpoint's arrays."""
    converted: dict[str, np.ndarray] = {}
    stacked: dict[str, list[tuple[int, np.ndarray]]] = {}
    classification = classify_keys(state)
    for key, conversion in classification.converted.items():
        if conversion.archive != archive:
            continue
        array = np.asarray(state[key])
        if conversion.layout == "split_qkv":
            for destination, part in zip(
                conversion.destinations, np.split(array, 3, axis=0)
            ):
                converted[destination] = part
        elif conversion.layout == "stack_of_four":
            index = int(re.search(r"\.(\d+)\.weight$", key)[1])
            stacked.setdefault(conversion.destinations[0], []).append((index, array))
        else:
            value = CONVERTERS[conversion.layout](array)
            for destination in conversion.destinations:
                converted[destination] = value
    for destination, parts in stacked.items():
        converted[destination] = np.concatenate(
            [array for _, array in sorted(parts, key=lambda item: item[0])], axis=0
        )
    return converted


def apply_state(model: Any, state: Mapping[str, np.ndarray], archive: str) -> Any:
    """Return ``model`` with the parameters ``state`` provides for ``archive``.

    Parameters without a source keep their values; ``check_destinations`` and
    the conversion test decide which of those are acceptable.
    """
    import jax
    import jax.numpy as jnp

    from equimo.conversion.utils import stringify_name

    converted = convert_state(state, archive)

    def replace(path: Any, leaf: Any) -> Any:
        array = converted.pop(stringify_name(path), None)
        if array is None:
            return leaf
        if array.shape != leaf.shape:
            raise ValueError(
                f"{stringify_name(path)}: converted shape {array.shape}, "
                f"template shape {leaf.shape}."
            )
        return jnp.asarray(array, leaf.dtype)

    model = jax.tree_util.tree_map_with_path(
        lambda path, leaf: replace(path, leaf) if hasattr(leaf, "shape") else leaf,
        model,
    )
    if converted:
        raise ValueError(f"Parameters missing from the model: {sorted(converted)}")
    return model


def report_classification(
    classification: Classification, parameters_without_source: list[str]
) -> int:
    """Print the outcome and return the process exit status."""
    print(
        f"converted={len(classification.converted)} "
        f"computed={len(classification.computed)} "
        f"unused={len(classification.unused)}"
    )
    for name in parameters_without_source:
        print(f"no source: {name}")
    if parameters_without_source:
        print(
            "error: the model does not cover the checkpoint; see the no source lines.",
            file=sys.stderr,
        )
        return 1
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: Path) -> dict[str, Any]:
    """Return the checkpoint's tensors after checking the file's digest."""
    actual_digest = sha256_file(path)
    if actual_digest != CHECKPOINT_SHA256:
        raise ValueError(
            f"Source checkpoint digest mismatch: expected {CHECKPOINT_SHA256}, "
            f"got {actual_digest}."
        )
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Reading the checkpoint requires the reference dependency group."
        ) from error

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a state dictionary in the source checkpoint.")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("Source checkpoint contains a non-tensor entry.")
    return state


def template_shapes(variant: str) -> dict[str, Shape]:
    """Return the parameter shapes of one Equimo model without allocating it."""
    import equinox as eqx
    import jax

    from equimo.conversion.utils import stringify_name
    from equimo.language import sam31_text
    from equimo.vision.models.sam3 import sam31_multiplex

    factory = {"multiplex": sam31_multiplex, "text": sam31_text}[variant]
    leaves = jax.tree_util.tree_flatten_with_path(eqx.filter_eval_shape(factory))[0]
    return {
        stringify_name(path): tuple(leaf.shape)
        for path, leaf in leaves
        if isinstance(leaf, jax.ShapeDtypeStruct)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the SAM 3.1 checkpoint to an Equimo archive."
    )
    parser.add_argument("variant", choices=sorted(ARCHIVES))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--source-dir",
        type=Path,
        help=f"Directory containing {CHECKPOINT_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.cache/equimo/sam31").expanduser(),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved paths.",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = args.source_dir / CHECKPOINT_NAME
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def _variant_model_and_config(variant: str) -> tuple[Any, dict[str, Any]]:
    """Build the Equimo model of ``variant`` and return it with its configuration."""
    from equimo.language import sam31_text
    from equimo.language.models import _CAUSAL_TEXT_REGISTRY
    from equimo.vision.models.sam3 import sam31_multiplex
    from equimo.vision.models.sam3.model import _SAM3_REGISTRY

    factory, registry = {
        "multiplex": (sam31_multiplex, _SAM3_REGISTRY),
        "text": (sam31_text, _CAUSAL_TEXT_REGISTRY),
    }[variant]
    base, variant_config = registry[ARCHIVES[variant]]
    return factory(), base | variant_config


def write_archive(
    variant: str,
    state: Mapping[str, Any],
    classification: Classification,
    output_dir: Path,
) -> Path:
    """Write ``variant``'s archive and check that it reads back identically."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from equimo.serialization import load_weights, save_model

    model, config = _variant_model_and_config(variant)
    model = apply_state(
        model, {name: np.asarray(state[name]) for name in state}, variant
    )
    identifier = ARCHIVES[variant]
    record = {
        "source": "facebook/sam3.1",
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "converted": len(classification.converted),
        "computed": len(classification.computed),
        "unused": len(classification.unused),
    }
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        path = save_model(Path(temporary) / identifier, model, config, record)
        restored = load_weights(model, path=path)
        same = jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            eqx.filter(restored, eqx.is_array),
            eqx.filter(model, eqx.is_array),
        )
        if not all(jax.tree.leaves(same)):
            raise ValueError(f"{identifier}: the archive does not read back equal.")
        destination = output_dir / path.name
        path.replace(destination)
    return destination


def main() -> int:
    args = parse_args()
    print(f"checkpoint={args.checkpoint}")
    print(f"output={args.output_dir / ARCHIVES[args.variant]}")
    if args.dry_run:
        return 0

    state = load_source(args.checkpoint)
    source_shapes = {name: tuple(value.shape) for name, value in state.items()}
    classification = classify_keys(source_shapes)
    missing_sources = check_destinations(
        classification, source_shapes, template_shapes(args.variant), args.variant
    )
    status = report_classification(classification, missing_sources)
    if status != 0:
        return status
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"wrote={write_archive(args.variant, state, classification, args.output_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
