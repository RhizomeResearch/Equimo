from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)


# Equimo pretrained identifier -> conversion spec.
#
#   base_variant: the Equimo factory function name (in equimo.vision.models)
#     for this architecture *size*, independent of `identifier`'s tag suffix
#     (e.g. "convnext_tiny_fb_in22k"'s base_variant is "convnext_tiny")
#   timm_tag: the upstream timm pretrained tag to convert from.
#   num_classes: measured directly from timm's config for this tag
#     (1000 for Imagenet-1k classifiers, 0 for headless FCMAE backbones,
#     11821 for ImageNet-12k heads, 21841 for ImageNet-22k heads).
#   v2: whether this is a ConvNeXt V2 (GRN, no LayerScale) architecture.
#   is_ols: whether this uses the overlapping-conv stem
#     ("_ols" variants, ConvNeXtOverlapStem)
#     instead of the default single-conv patchify stem (ConvNeXtStem).
VARIANTS: dict[str, dict] = {
    "convnext_atto": {
        "base_variant": "convnext_atto",
        "timm_tag": "convnext_atto.d2_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_femto": {
        "base_variant": "convnext_femto",
        "timm_tag": "convnext_femto.d1_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_pico": {
        "base_variant": "convnext_pico",
        "timm_tag": "convnext_pico.d1_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_nano": {
        "base_variant": "convnext_nano",
        "timm_tag": "convnext_nano.d1h_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_nano_in12k": {
        "base_variant": "convnext_nano",
        "timm_tag": "convnext_nano.in12k",
        "num_classes": 11821,
        "v2": False,
        "is_ols": False,
    },
    "convnext_nano_in12k_ft_in1k": {
        "base_variant": "convnext_nano",
        "timm_tag": "convnext_nano.in12k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_nano_r384_ad_in12k": {
        "base_variant": "convnext_nano",
        "timm_tag": "convnext_nano.r384_ad_in12k",
        "num_classes": 11821,
        "v2": False,
        "is_ols": False,
    },
    "convnext_nano_r384_in12k": {
        "base_variant": "convnext_nano",
        "timm_tag": "convnext_nano.r384_in12k",
        "num_classes": 11821,
        "v2": False,
        "is_ols": False,
    },
    "convnext_nano_r384_in12k_ft_in1k": {
        "base_variant": "convnext_nano",
        "timm_tag": "convnext_nano.r384_in12k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.fb_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny_fb_in22k": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.fb_in22k",
        "num_classes": 21841,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny_fb_in22k_ft_in1k": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.fb_in22k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny_fb_in22k_ft_in1k_384": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.fb_in22k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny_in12k": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.in12k",
        "num_classes": 11821,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny_in12k_ft_in1k": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.in12k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_tiny_in12k_ft_in1k_384": {
        "base_variant": "convnext_tiny",
        "timm_tag": "convnext_tiny.in12k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.fb_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small_fb_in22k": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.fb_in22k",
        "num_classes": 21841,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small_fb_in22k_ft_in1k": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.fb_in22k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small_fb_in22k_ft_in1k_384": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.fb_in22k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small_in12k": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.in12k",
        "num_classes": 11821,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small_in12k_ft_in1k": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.in12k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_small_in12k_ft_in1k_384": {
        "base_variant": "convnext_small",
        "timm_tag": "convnext_small.in12k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_base": {
        "base_variant": "convnext_base",
        "timm_tag": "convnext_base.fb_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_base_fb_in22k": {
        "base_variant": "convnext_base",
        "timm_tag": "convnext_base.fb_in22k",
        "num_classes": 21841,
        "v2": False,
        "is_ols": False,
    },
    "convnext_base_fb_in22k_ft_in1k": {
        "base_variant": "convnext_base",
        "timm_tag": "convnext_base.fb_in22k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_base_fb_in22k_ft_in1k_384": {
        "base_variant": "convnext_base",
        "timm_tag": "convnext_base.fb_in22k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_large": {
        "base_variant": "convnext_large",
        "timm_tag": "convnext_large.fb_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_large_fb_in22k": {
        "base_variant": "convnext_large",
        "timm_tag": "convnext_large.fb_in22k",
        "num_classes": 21841,
        "v2": False,
        "is_ols": False,
    },
    "convnext_large_fb_in22k_ft_in1k": {
        "base_variant": "convnext_large",
        "timm_tag": "convnext_large.fb_in22k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_large_fb_in22k_ft_in1k_384": {
        "base_variant": "convnext_large",
        "timm_tag": "convnext_large.fb_in22k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_xlarge_fb_in22k": {
        "base_variant": "convnext_xlarge",
        "timm_tag": "convnext_xlarge.fb_in22k",
        "num_classes": 21841,
        "v2": False,
        "is_ols": False,
    },
    "convnext_xlarge": {
        "base_variant": "convnext_xlarge",
        "timm_tag": "convnext_xlarge.fb_in22k_ft_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_xlarge_fb_in22k_ft_in1k_384": {
        "base_variant": "convnext_xlarge",
        "timm_tag": "convnext_xlarge.fb_in22k_ft_in1k_384",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_zepto_rms_ols": {
        "base_variant": "convnext_zepto_rms_ols",
        "timm_tag": "convnext_zepto_rms_ols.ra4_e3600_r224_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": True,
    },
    "convnext_zepto_rms": {
        "base_variant": "convnext_zepto_rms",
        "timm_tag": "convnext_zepto_rms.ra4_e3600_r224_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": False,
    },
    "convnext_atto_ols": {
        "base_variant": "convnext_atto_ols",
        "timm_tag": "convnext_atto_ols.a2_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": True,
    },
    "convnext_femto_ols": {
        "base_variant": "convnext_femto_ols",
        "timm_tag": "convnext_femto_ols.d1_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": True,
    },
    "convnext_pico_ols": {
        "base_variant": "convnext_pico_ols",
        "timm_tag": "convnext_pico_ols.d1_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": True,
    },
    "convnext_nano_ols": {
        "base_variant": "convnext_nano_ols",
        "timm_tag": "convnext_nano_ols.d1h_in1k",
        "num_classes": 1000,
        "v2": False,
        "is_ols": True,
    },
    "convnextv2_atto_fcmae": {
        "base_variant": "convnextv2_atto",
        "timm_tag": "convnextv2_atto.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_atto": {
        "base_variant": "convnextv2_atto",
        "timm_tag": "convnextv2_atto.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_femto_fcmae": {
        "base_variant": "convnextv2_femto",
        "timm_tag": "convnextv2_femto.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_femto": {
        "base_variant": "convnextv2_femto",
        "timm_tag": "convnextv2_femto.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_pico_fcmae": {
        "base_variant": "convnextv2_pico",
        "timm_tag": "convnextv2_pico.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_pico": {
        "base_variant": "convnextv2_pico",
        "timm_tag": "convnextv2_pico.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_nano_fcmae": {
        "base_variant": "convnextv2_nano",
        "timm_tag": "convnextv2_nano.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_nano": {
        "base_variant": "convnextv2_nano",
        "timm_tag": "convnextv2_nano.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_nano_fcmae_ft_in22k_in1k": {
        "base_variant": "convnextv2_nano",
        "timm_tag": "convnextv2_nano.fcmae_ft_in22k_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_nano_fcmae_ft_in22k_in1k_384": {
        "base_variant": "convnextv2_nano",
        "timm_tag": "convnextv2_nano.fcmae_ft_in22k_in1k_384",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_tiny_fcmae": {
        "base_variant": "convnextv2_tiny",
        "timm_tag": "convnextv2_tiny.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_tiny": {
        "base_variant": "convnextv2_tiny",
        "timm_tag": "convnextv2_tiny.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_tiny_fcmae_ft_in22k_in1k": {
        "base_variant": "convnextv2_tiny",
        "timm_tag": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_tiny_fcmae_ft_in22k_in1k_384": {
        "base_variant": "convnextv2_tiny",
        "timm_tag": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_base_fcmae": {
        "base_variant": "convnextv2_base",
        "timm_tag": "convnextv2_base.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_base": {
        "base_variant": "convnextv2_base",
        "timm_tag": "convnextv2_base.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_base_fcmae_ft_in22k_in1k": {
        "base_variant": "convnextv2_base",
        "timm_tag": "convnextv2_base.fcmae_ft_in22k_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_base_fcmae_ft_in22k_in1k_384": {
        "base_variant": "convnextv2_base",
        "timm_tag": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_large_fcmae": {
        "base_variant": "convnextv2_large",
        "timm_tag": "convnextv2_large.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_large": {
        "base_variant": "convnextv2_large",
        "timm_tag": "convnextv2_large.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_large_fcmae_ft_in22k_in1k": {
        "base_variant": "convnextv2_large",
        "timm_tag": "convnextv2_large.fcmae_ft_in22k_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_large_fcmae_ft_in22k_in1k_384": {
        "base_variant": "convnextv2_large",
        "timm_tag": "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_huge_fcmae": {
        "base_variant": "convnextv2_huge",
        "timm_tag": "convnextv2_huge.fcmae",
        "num_classes": 0,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_huge": {
        "base_variant": "convnextv2_huge",
        "timm_tag": "convnextv2_huge.fcmae_ft_in1k",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_huge_fcmae_ft_in22k_in1k_384": {
        "base_variant": "convnextv2_huge",
        "timm_tag": "convnextv2_huge.fcmae_ft_in22k_in1k_384",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
    "convnextv2_huge_fcmae_ft_in22k_in1k_512": {
        "base_variant": "convnextv2_huge",
        "timm_tag": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
        "num_classes": 1000,
        "v2": True,
        "is_ols": False,
    },
}

IMG_SIZE = 224
_DEFAULT_MIN_FREE_GB = 5.0
_MAX_MEAN_ERROR = 5e-4
_ABSOLUTE_TOLERANCE = 5e-4
_RELATIVE_TOLERANCE = 1e-4

# The one `_ols` variant whose stem has an activation
# between the two convolutions (`stem_kwargs={"act_layer": "gelu"}` in
# equimo.vision.models.convnext's `_CONVNEXT_OLS_REGISTRY`).
# timm builds every overlap stem as
# `nn.Sequential(*filter(None, [conv1, act_or_None, conv2, norm]))`:
# when there is no activation,
# the `None` is filtered out before the Sequential is built,
# so conv2/norm land on indices 1/2;
# but here the activation is a real (parameterless) module
# that still consumes an index,
# so conv2/norm land on stem.{2,3} instead.
_OLS_STEM_HAS_ACT = "convnext_zepto_rms_ols"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert timm ConvNeXt/ConvNeXt V2 checkpoints to Equimo."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        choices=sorted(VARIANTS) + ["all"],
        metavar="{variant|all}",
        help="One or more Equimo identifiers to convert, or 'all' for the full inventory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.cache/equimo/convnext").expanduser(),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=_DEFAULT_MIN_FREE_GB,
        help="Abort the batch (before starting the next conversion) if free "
        "disk space on --output-dir's filesystem drops below this many GB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved conversion work.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    if "all" in args.targets:
        args.targets = sorted(VARIANTS)
    else:
        seen = set()
        deduped = []
        for target in args.targets:
            if target not in seen:
                seen.add(target)
                deduped.append(target)
        args.targets = deduped

    return args


def conversion_config(entry: dict, depths: list[int], conv_mlp: bool):
    """Map Equimo ConvNeXt parameter paths onto timm's ConvNeXt naming.

    timm's own module layout (stem.{...}, stages.{i}.blocks.{j}.{conv_dw,
    norm,mlp.fc1,mlp.grn,mlp.fc2}) differs from the original reference
    repository's layout that models/eupe.py's conversion_config targets, so
    this is a separate mapping, not a reuse of EUPE's.
    """
    is_ols = entry["is_ols"]
    stem_has_act = is_ols and entry["base_variant"] == _OLS_STEM_HAS_ACT

    replace_cfg: dict[str, str] = {}
    if stem_has_act:
        # ConvNeXtOverlapStem (conv1, conv2, norm) vs timm's
        # Sequential(conv1, act, conv2, norm) - see _OLS_STEM_HAS_ACT above.
        replace_cfg["blocks.0.downsample.conv1"] = "stem.0"
        replace_cfg["blocks.0.downsample.conv2"] = "stem.2"
        replace_cfg["blocks.0.downsample.norm"] = "stem.3"
        stem_conv_bias_paths = ["stem.0.bias", "stem.2.bias"]
    elif is_ols:
        # ConvNeXtOverlapStem (conv1, conv2, norm)
        # -> timm's Sequential(conv1, conv2, norm) indices.
        replace_cfg["blocks.0.downsample.conv1"] = "stem.0"
        replace_cfg["blocks.0.downsample.conv2"] = "stem.1"
        replace_cfg["blocks.0.downsample.norm"] = "stem.2"
        stem_conv_bias_paths = ["stem.0.bias", "stem.1.bias"]
    else:
        # Stage-0 stem:
        # Equimo nests it under blocks[0].downsample (conv, norm);
        # timm keeps it as a top-level stem.0 (conv) / stem.1 (norm).
        replace_cfg["blocks.0.downsample.conv"] = "stem.0"
        replace_cfg["blocks.0.downsample.norm"] = "stem.1"
        stem_conv_bias_paths = ["stem.0.bias"]

    replace_cfg.update(
        {
            # Stages 1-3 downsampler:
            # Equimo's norm-then-conv attribute names
            # -> timm's Sequential(norm, conv) indices.
            "downsample.norm": "downsample.0",
            "downsample.conv": "downsample.1",
            # Protect the inner per-block "blocks" index
            # from the outer-field rename below
            # (Equimo's blocks[i].blocks[j] vs timm's stages.{i}.blocks.{j}
            # keep the inner "blocks" token as-is).
            ".blocks.": ".BLOCKS_INNER.",
            "blocks.": "stages.",
            ".BLOCKS_INNER.": ".blocks.",
            # Block internals.
            "dwconv": "conv_dw",
            "pwconv1": "mlp.fc1",
            "pwconv2": "mlp.fc2",
            # Protect per-block norm
            # from the top-level pooled-feature norm rename below
            # (both are literally named "norm" in Equimo).
            "head.": "head.fc.",
            ".norm.": ".NORM_INNER.",
            "norm.": "head.norm.",
            ".NORM_INNER.": ".norm.",
        }
    )
    if entry["v2"]:
        # V2 blocks: grn sits inside timm's mlp submodule, alongside fc1/fc2.
        replace_cfg["grn"] = "mlp.grn"
    else:
        # V1 blocks: LayerScale, absent (Identity, no params) in V2.
        replace_cfg["ls.gamma"] = "gamma"

    # Equimo implements the stem/downsampler/pointwise convs
    # as 1x1 Conv2d (bias shape (C,1,1));
    # timm's `conv_mlp=False` sizes (tiny and up)
    # implement the pointwise convs as Linear (weight (out,in), bias (out,))
    # and store conv biases flat too,
    # so both weight and bias need the "after,2" expansion.
    # timm's `conv_mlp=True` sizes already implement them as 1x1 Conv2d,
    # so only the (still-flat) bias needs it.
    expand_cfg: dict[str, list] = {path: ["after", 2] for path in stem_conv_bias_paths}
    for stage_index, depth in enumerate(depths):
        if stage_index > 0:
            expand_cfg[f"stages.{stage_index}.downsample.1.bias"] = ["after", 2]
        for block_index in range(depth):
            base_path = f"stages.{stage_index}.blocks.{block_index}"
            suffixes = ["conv_dw.bias", "mlp.fc1.bias", "mlp.fc2.bias"]
            if not conv_mlp:
                suffixes += ["mlp.fc1.weight", "mlp.fc2.weight"]
            for suffix in suffixes:
                expand_cfg[f"{base_path}.{suffix}"] = ["after", 2]

    return replace_cfg, expand_cfg


def archive_path(output_dir: Path, identifier: str) -> Path:
    return output_dir / f"{identifier}.tar.lz4"


def kept_output_bytes(output_dir: Path) -> int:
    if not output_dir.is_dir():
        return 0
    return sum(p.stat().st_size for p in output_dir.glob("*.tar.lz4"))


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def cleanup_timm_cache(timm_tag: str) -> int:
    """Delete the raw HuggingFace hub cache for one timm checkpoint.

    Returns the number of bytes freed,
    0 if nothing was found or the scan failed.
    This is best-effort housekeeping, not load-bearing for correctness.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return 0

    repo_id = f"timm/{timm_tag}"
    try:
        cache_info = scan_cache_dir()
    except Exception as exc:
        print(f"  (cache scan failed, skipping cleanup: {exc!r})")
        return 0

    freed = 0
    for repo in cache_info.repos:
        if repo.repo_type == "model" and repo.repo_id == repo_id:
            freed += repo.size_on_disk
            shutil.rmtree(repo.repo_path, ignore_errors=True)
    return freed


def validate_outputs(actual, expected, label: str) -> dict[str, float]:
    """Check both average and elementwise float32 conversion parity."""
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape mismatch {actual.shape} != {expected.shape}")
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError(f"{label}: non-finite outputs")
    difference = np.abs(actual - expected)
    metrics = {
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
    }
    if metrics["mean_absolute_error"] >= _MAX_MEAN_ERROR:
        raise ValueError(f"{label}: conversion error {metrics}")
    np.testing.assert_allclose(
        actual,
        expected,
        atol=_ABSOLUTE_TOLERANCE,
        rtol=_RELATIVE_TOLERANCE,
        err_msg=label,
    )
    return metrics


def convert_one(identifier: str, entry: dict, output_dir: Path, seed: int) -> float:
    """Convert one (variant, tag) pair end to end. Returns the pre-head
    feature error against the real torch model. Raises on any failure
    (missing/extra params, shape mismatch, error above threshold, ...).
    """
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import timm
        import torch
    except ImportError as exc:
        raise ImportError(
            "`torch`, `timm`, Equinox, JAX, and NumPy are required"
        ) from exc

    import equimo.vision.models as em
    from equimo.conversion.utils import convert_torch_to_equinox
    from huggingface_hub import HfApi

    from equimo.serialization import load_weights, save_model
    from equimo.vision.models.convnext import _CONVNEXT_REGISTRY

    base_variant = entry["base_variant"]
    timm_tag = entry["timm_tag"]
    num_classes = entry["num_classes"]

    key = jax.random.PRNGKey(seed)
    print(
        f"Converting {identifier} ({base_variant}) from timm:{timm_tag} "
        f"num_classes={num_classes}..."
    )

    # timm's published weights expect exact (erf-based) GELU;
    # Equimo's default "gelu" is JAX's tanh approximation.
    # Using the wrong one still runs,
    # but silently degrades accuracy
    # (verified: ~2000x higher error on convnext_tiny).
    model_kwargs = {"act_layer": "exactgelu", "num_classes": num_classes}
    if base_variant == _OLS_STEM_HAS_ACT:
        _, variant_cfg = _CONVNEXT_REGISTRY[base_variant]
        model_kwargs["stem_kwargs"] = variant_cfg["stem_kwargs"] | {
            "act_layer": "exactgelu"
        }
    model = getattr(em, base_variant)(**model_kwargs)

    # Derived from the constructed model rather than hardcoded per size, so
    # VARIANTS doesn't need a `depths` table for all 16 distinct sizes.
    depths = [len(chunk.blocks) for chunk in model.blocks]

    # Build the torch model ourselves
    # (instead of letting convert_torch_to_equinox do it via timm_cfg)
    # so we can inspect its real mlp.fc1 weight rank
    # before deciding the expand_cfg:
    # timm's `conv_mlp=True` sizes
    # (atto..zepto_rms and their _ols variants, both v1 and v2)
    # implement the block MLP as 1x1 Conv2d (weight already (out,in,1,1)),
    # while `conv_mlp=False` sizes (tiny and up) implement it as Linear
    # (weight (out,in), needing Equimo's "after,2" expansion).
    # Detected from the real checkpoint,
    # so it can't silently go stale if timm's per-size defaults change.
    upstream_repo = f"timm/{timm_tag}"
    upstream_revision = HfApi().model_info(upstream_repo).sha
    torch_model = timm.create_model(
        timm_tag,
        pretrained=True,
        pretrained_cfg_overlay={"hf_hub_id": f"{upstream_repo}@{upstream_revision}"},
    )
    fc1_weight = dict(torch_model.named_parameters())[
        "stages.0.blocks.0.mlp.fc1.weight"
    ]
    conv_mlp = fc1_weight.ndim == 4

    replace_cfg, expand_cfg = conversion_config(entry, depths, conv_mlp)
    model, torch_model = convert_torch_to_equinox(
        model,
        replace_cfg,
        expand_cfg,
        {},
        [],
        [],
        strict=True,
        source="custom",
        torch_model=torch_model,
        return_torch=True,
    )
    model = eqx.nn.inference_mode(model, True)
    torch_model.eval()

    input_size = tuple(
        torch_model.pretrained_cfg.get("input_size", (3, IMG_SIZE, IMG_SIZE))
    )
    validation = []
    for validation_seed in (seed, seed + 1):
        arr = (
            np.random.default_rng(validation_seed)
            .standard_normal(input_size)
            .astype(np.float32)
        )
        jax_arr = jnp.asarray(arr)
        torch_arr = torch.from_numpy(arr).unsqueeze(0)
        jax_features = model.features(jax_arr, inference=True, key=key)
        jax_pooled = model.norm(jax_features.mean((1, 2)))
        jax_output = model.head(jax_pooled)
        with torch.no_grad():
            torch_features = torch_model.forward_features(torch_arr)
            torch_pooled = torch_model.forward_head(torch_features, pre_logits=True)
            torch_output = torch_model.forward_head(torch_features)
        metrics = {
            "seed": validation_seed,
            "input_size": list(input_size),
            "features": validate_outputs(
                jax_pooled, torch_pooled.squeeze(0).numpy(), "features"
            ),
            "output": validate_outputs(
                jax_output, torch_output.squeeze(0).numpy(), "output"
            ),
        }
        validation.append(metrics)
        print("parity:", metrics, flush=True)

    base_cfg, variant_cfg = _CONVNEXT_REGISTRY[base_variant]
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        path = save_model(
            Path(temporary) / identifier,
            model,
            base_cfg | variant_cfg | model_kwargs,
            {
                "source": "timm",
                "timm_tag": timm_tag,
                "upstream_revision": upstream_revision,
                "timm_version": timm.__version__,
                "torch_version": torch.__version__,
                "act_layer": "exactgelu",
                "pretrained_cfg": torch_model.pretrained_cfg,
                "validation": validation,
                "tolerances": {
                    "mean_absolute_error": _MAX_MEAN_ERROR,
                    "atol": _ABSOLUTE_TOLERANCE,
                    "rtol": _RELATIVE_TOLERANCE,
                },
            },
            compression=True,
        )
        restored = load_weights(model, path=path)
        np.testing.assert_array_equal(
            restored(jax_arr, key=key), jax_output, err_msg="archive round trip"
        )
        path.replace(archive_path(output_dir, identifier))
    return max(item["features"]["mean_absolute_error"] for item in validation)


def run_batch(
    identifiers: list[str],
    output_dir: Path,
    seed: int,
    min_free_gb: float = _DEFAULT_MIN_FREE_GB,
) -> list[dict]:
    """Convert each identifier one at a time:
        download
        -> convert
        -> verify
        -> save
        -> delete that checkpoint's raw timm/HF cache,
        before moving on.

    Resumable/idempotent: any identifier whose output archive already exists
    is skipped rather than reconverted.
    Aborts cleanly (without starting the next conversion)
    if free disk space on `output_dir`'s filesystem drops below `min_free_gb`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    min_free_bytes = min_free_gb * (1024**3)

    for identifier in identifiers:
        entry = VARIANTS[identifier]
        out_path = archive_path(output_dir, identifier)

        if out_path.exists():
            print(f"[skip] {identifier}: already converted -> {out_path}")
            results.append(
                {"identifier": identifier, "status": "skipped", "detail": str(out_path)}
            )
            continue

        free_bytes = shutil.disk_usage(output_dir).free
        if free_bytes < min_free_bytes:
            print(
                f"ABORTING before {identifier!r}: free disk space "
                f"({human_size(free_bytes)}) is below the {min_free_gb:.0f}GB floor."
            )
            results.append(
                {
                    "identifier": identifier,
                    "status": "aborted-disk-space",
                    "detail": f"free={human_size(free_bytes)}",
                }
            )
            break

        try:
            error = convert_one(identifier, entry, output_dir, seed)
            results.append({"identifier": identifier, "status": "ok", "detail": error})
        except Exception as exc:  # noqa: BLE001 - report every failure, never drop one
            print(f"[FAIL] {identifier}: {exc!r}")
            results.append(
                {"identifier": identifier, "status": "failed", "detail": repr(exc)}
            )
        finally:
            freed = cleanup_timm_cache(entry["timm_tag"])
            if freed:
                print(
                    f"  cleaned up raw cache for {entry['timm_tag']}: "
                    f"freed {human_size(freed)}"
                )

        print(
            f"  cumulative kept output size: {human_size(kept_output_bytes(output_dir))}"
        )

    return results


def main() -> int:
    """Return nonzero if any conversion fails or is aborted."""
    args = parse_args()
    for identifier in args.targets:
        entry = VARIANTS[identifier]
        print(
            f"{identifier}: base_variant={entry['base_variant']} "
            f"timm_tag={entry['timm_tag']} num_classes={entry['num_classes']} "
            f"is_ols={entry['is_ols']} v2={entry['v2']} seed={args.seed} "
            f"output={archive_path(args.output_dir, identifier)}"
        )
    if args.dry_run:
        return 0

    results = run_batch(args.targets, args.output_dir, args.seed, args.min_free_gb)

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['identifier']:45s} {r['status']:20s} {r['detail']}")
    print(f"\nTotal kept output size: {human_size(kept_output_bytes(args.output_dir))}")
    return int(any(r["status"] not in ("ok", "skipped") for r in results))


if __name__ == "__main__":
    sys.exit(main())
