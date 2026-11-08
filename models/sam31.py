"""Classify the keys of the SAM 3.1 multiplex checkpoint against Equimo's models.

Every key either converts to named Equimo parameters, is computed by Equimo and
dropped, is unused and dropped, or has no destination yet. A key outside those
lists is an error. The original PyTorch format is read only by this maintainer
script, which runs in the ``reference`` dependency group.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, NamedTuple

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)


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

UNUSED_KEYS = frozenset({f"{TEXT}.encoder.text_projection"})
COMPUTED_KEY = re.compile(re.escape(VISION) + r"\.trunk\.blocks\.\d+\.attn\.freqs_cis")

# Checkpoint modules the drafted model does not have, or has with another structure.
# They stay listed until the model is corrected, so that no trained weight is dropped
# silently and no module is filled from a checkpoint entry that means something else.
UNMAPPED_MODULES = (
    f"{GEOMETRY}.boxes_direct_project",
    f"{GEOMETRY}.boxes_pos_enc_project",
    f"{GEOMETRY}.cls_embed",
    f"{GEOMETRY}.encode_norm",
    f"{GEOMETRY}.final_proj",
    f"{GEOMETRY}.img_pre_norm",
    f"{GEOMETRY}.norm",
    f"{GEOMETRY}.points_direct_project",
    f"{GEOMETRY}.points_pool_project",
    f"{GEOMETRY}.points_pos_enc_project",
    f"{SCORING}.prompt_mlp",
    f"{SEGMENTATION}.instance_seg_head",
    f"{SEGMENTATION}.semantic_seg_head",
    f"{DECODER}.boxRPB_embed_x",
    f"{DECODER}.boxRPB_embed_y",
    f"{DECODER}.norm",
    f"{DECODER}.presence_token_out_norm",
    f"{DECODER}.ref_point_head.layers.0",
    f"{TRACKER}.interactive_mask_downsample",
    f"{TRACKER}.interactive_obj_ptr_proj",
    f"{TRACKER}.no_obj_embed_spatial",
    f"{TRACKER}.no_obj_ptr_linear",
    f"{TRACKER}.transformer.encoder.layers.{{i}}.image_cross_attn_k_proj",
    f"{TRACKER}.transformer.encoder.layers.{{i}}.image_cross_attn_q_proj",
    f"{MEMORY}.mask_downsampler",
    f"{INTERACTIVE_DECODER}.output_hypernetworks_mlps",
    f"{INTERACTIVE_DECODER}.transformer.final_attn_token_to_image",
    f"{MULTIPLEX_DECODER}.output_hypernetworks_mlps",
    f"{MULTIPLEX_DECODER}.transformer.final_attn_token_to_image",
    f"{MULTIPLEX_DECODER}.iou_token",
    f"{MULTIPLEX_DECODER}.mask_tokens",
    f"{MULTIPLEX_DECODER}.obj_score_token",
)
UNMAPPED_PATTERNS = tuple(
    re.compile(re.escape(module).replace(r"\{i\}", r"\d+") + r"(?:\..+)?")
    for module in UNMAPPED_MODULES
)

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
    unmapped: list[str]


def _rule(
    source: str,
    destinations: str | tuple[str, ...],
    layout: str = "copy",
    archive: str = "multiplex",
) -> Rule:
    """Map one checkpoint key, where ``{i}`` stands for a module index."""
    if isinstance(destinations, str):
        destinations = (destinations,)
    pattern = re.compile(re.escape(source).replace(r"\{i\}", r"(\d+)"))
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
        *_linear(f"{block}.linear1", f"{destination}.mlp.layers.0"),
        *_linear(f"{block}.linear2", f"{destination}.mlp.layers.1"),
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
    geometry_block = f"{GEOMETRY}" + ".encode.{i}"
    geometry_target = "detector.geometry.blocks.{i}"
    fusion_target = "detector.head.fusion.blocks.{i}"
    decoder_block = f"{DECODER}" + ".layers.{i}"
    decoder_target = "detector.head.decoder.blocks.{i}"
    decoder = "detector.head.decoder"
    segmentation = "detector.head.segmentation"
    return [
        _rule(f"{GEOMETRY}.label_embed.weight", "detector.geometry.label_embed"),
        *_conv(f"{GEOMETRY}.boxes_pool_project", "detector.geometry.box_pool"),
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
        # Shapes cannot tell these norms apart; the assignment follows the block's
        # order of operations and is confirmed by output comparison.
        *_linear(f"{decoder_block}.norm1", f"{decoder_target}.norm1"),
        *_linear(f"{decoder_block}.catext_norm", f"{decoder_target}.norm2"),
        *_linear(f"{decoder_block}.norm2", f"{decoder_target}.norm3"),
        *_linear(f"{decoder_block}.norm3", f"{decoder_target}.norm4"),
        _rule(f"{DECODER}.query_embed.weight", f"{decoder}.queries"),
        _rule(f"{DECODER}.reference_points.weight", f"{decoder}.reference_boxes"),
        _rule(
            f"{DECODER}.presence_token.weight", f"{decoder}.presence_token", "squeeze"
        ),
        *_linear(
            f"{DECODER}.bbox_embed.layers" + ".{i}",
            f"{decoder}.box_head.layers" + ".{i}",
        ),
        *_linear(
            f"{DECODER}.ref_point_head.layers.1", f"{decoder}.position_proj.layers.1"
        ),
        *_linear(
            f"{DECODER}.presence_token_head.layers" + ".{i}",
            f"{decoder}.presence_head.layers" + ".{i}",
        ),
        *_linear(f"{SCORING}.hs_proj", "detector.head.scoring.query_proj"),
        *_linear(f"{SCORING}.prompt_proj", "detector.head.scoring.prompt_proj"),
        *_packed_attention(
            f"{SEGMENTATION}.cross_attend_prompt", f"{segmentation}.cross_attn"
        ),
        *_linear(f"{SEGMENTATION}.cross_attn_norm", f"{segmentation}.norm"),
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
        *_linear(f"{source}.output_upscaling.1", f"{destination}.upsampler.norms.0"),
        *_transposed_conv(
            f"{source}.output_upscaling.3", f"{destination}.upsampler.deconvs.1"
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
        *_separate_attention(f"{block}.self_attn.", f"{target}.attn"),
        *_separate_attention(
            f"{block}.cross_attn_token_to_image.", f"{target}.cross_attn.0"
        ),
        *_separate_attention(
            f"{block}.cross_attn_image_to_token.", f"{target}.cross_attn.1"
        ),
        *_linear(f"{block}.mlp.lin1", f"{target}.mlp.layers.0"),
        *_linear(f"{block}.mlp.lin2", f"{target}.mlp.layers.1"),
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


def _tracker_rules() -> list[Rule]:
    fuser = f"{MEMORY}.fuser.layers" + ".{i}"
    fuser_target = "tracker.memory_encoder.fuser.{i}"
    attention = f"{TRACKER}.transformer.encoder"
    block = attention + ".layers.{i}"
    target = "tracker.memory_attention.blocks.{i}"
    return [
        _rule(f"{TRACKER}.output_valid_embed", "tracker.mask_decoder.valid_embed"),
        _rule(f"{TRACKER}.output_invalid_embed", "tracker.mask_decoder.invalid_embed"),
        _rule(f"{TRACKER}.maskmem_tpos_enc", "tracker.memory_time_embed", "squeeze"),
        _rule(
            f"{TRACKER}.image_pe_layer.positional_encoding_gaussian_matrix",
            "tracker.prompt_encoder.fourier_matrix",
        ),
        *_linear(
            f"{TRACKER}.obj_ptr_proj.layers" + ".{i}",
            "tracker.pointer_proj.layers.{i}",
        ),
        *_linear(f"{TRACKER}.obj_ptr_tpos_proj", "tracker.pointer_time_proj"),
        *_conv(f"{MEMORY}.pix_feat_proj", "tracker.memory_encoder.feature_proj"),
        _rule(f"{fuser}.gamma", f"{fuser_target}.ls.gamma"),
        *_conv(f"{fuser}.dwconv", f"{fuser_target}.dwconv"),
        *_linear_as_conv(f"{fuser}.pwconv1", f"{fuser_target}.pwconv1"),
        *_linear_as_conv(f"{fuser}.pwconv2", f"{fuser_target}.pwconv2"),
        *_linear(f"{fuser}.norm", f"{fuser_target}.norm"),
        *_linear(f"{attention}.norm", "tracker.memory_attention.norm"),
        *_separate_attention(f"{block}.self_attn_", f"{target}.attn"),
        *_separate_attention(f"{block}.cross_attn_", f"{target}.cross_attn"),
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


def _conversion_for(key: str) -> Conversion | None:
    for rule in RULES:
        match = rule.pattern.fullmatch(key)
        if match is None:
            continue
        destinations = tuple(
            destination.replace("{i}", match[1]) if match.groups() else destination
            for destination in rule.destinations
        )
        return Conversion(rule.archive, destinations, rule.layout)
    return None


def classify_keys(keys: Iterable[str]) -> Classification:
    """Sort checkpoint keys by outcome, failing on keys no list accounts for."""
    converted: dict[str, Conversion] = {}
    computed: list[str] = []
    unused: list[str] = []
    unmapped: list[str] = []
    unknown: list[str] = []
    for key in keys:
        if key in UNUSED_KEYS:
            unused.append(key)
        elif COMPUTED_KEY.fullmatch(key):
            computed.append(key)
        elif any(pattern.fullmatch(key) for pattern in UNMAPPED_PATTERNS):
            unmapped.append(key)
        elif (conversion := _conversion_for(key)) is not None:
            converted[key] = conversion
        else:
            unknown.append(key)
    if unknown:
        listing = "\n".join(unknown)
        raise ValueError(f"Unknown checkpoint keys:\n{listing}")
    return Classification(converted, computed, unused, unmapped)


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


def report_classification(
    classification: Classification, parameters_without_source: list[str]
) -> int:
    """Print the outcome and return the process exit status."""
    print(
        f"converted={len(classification.converted)} "
        f"computed={len(classification.computed)} "
        f"unused={len(classification.unused)} "
        f"unmapped={len(classification.unmapped)}"
    )
    for key in classification.unmapped:
        print(f"unmapped: {key}")
    for name in parameters_without_source:
        print(f"no source: {name}")
    if classification.unmapped or parameters_without_source:
        print(
            "error: the model does not cover the checkpoint yet; "
            "see the unmapped and no source lines.",
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


def load_source_shapes(path: Path) -> dict[str, Shape]:
    """Return every tensor's shape after checking the file's digest."""
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
    return {name: tuple(value.shape) for name, value in state.items()}


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
        description="Classify the SAM 3.1 checkpoint keys against an Equimo model."
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


def main() -> int:
    args = parse_args()
    print(f"checkpoint={args.checkpoint}")
    print(f"output={args.output_dir / ARCHIVES[args.variant]}")
    if args.dry_run:
        return 0

    source_shapes = load_source_shapes(args.checkpoint)
    classification = classify_keys(source_shapes)
    missing_sources = check_destinations(
        classification, source_shapes, template_shapes(args.variant), args.variant
    )
    return report_classification(classification, missing_sources)


if __name__ == "__main__":
    raise SystemExit(main())
