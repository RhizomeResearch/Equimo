import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)

import jax
import jax.numpy as jnp
import numpy as np

import equimo.vision.models as em
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.serialization import save_model

CHECKPOINT_FILENAMES = {
    "tips_vits14_hr": "tips_oss_s14_highres_distilled_vision.npz",
    "tips_vitb14_hr": "tips_oss_b14_highres_distilled_vision.npz",
    "tips_vitl14_hr": "tips_oss_l14_highres_distilled_vision.npz",
    "tips_vitso400m14_hr": "tips_oss_so400m14_highres_largetext_distilled_vision.npz",
    "tips_vitg14_lr": "tips_oss_g14_lowres_vision.npz",
    "tips_vitg14_hr": "tips_oss_g14_highres_vision.npz",
}

IMAGE_FACTORIES = {
    "tips_vits14_hr": "vit_small",
    "tips_vitb14_hr": "vit_base",
    "tips_vitl14_hr": "vit_large",
    "tips_vitso400m14_hr": "vit_so400m",
    "tips_vitg14_lr": "vit_giant2",
    "tips_vitg14_hr": "vit_giant2",
}


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


configs = {
    "tips_vits14_hr": {
        "img_size": 448,
        "dim": 384,
        "num_heads": [6],
        "depths": [12],
    },
    "tips_vitb14_hr": {
        "img_size": 448,
        "dim": 768,
        "num_heads": [12],
        "depths": [12],
    },
    "tips_vitl14_hr": {
        "img_size": 448,
        "dim": 1024,
        "num_heads": [16],
        "depths": [24],
    },
    "tips_vitso400m14_hr": {
        "img_size": 448,
        "dim": 1152,
        "num_heads": [16],
        "depths": [27],
        "mlp_ratio": 4304 / 1152,
    },
    "tips_vitg14_lr": {
        "img_size": 224,
        "dim": 1536,
        "num_heads": [24],
        "depths": [40],
        "ffn_layer": "swiglufused",
    },
    "tips_vitg14_hr": {
        "img_size": 448,
        "dim": 1536,
        "num_heads": [24],
        "depths": [40],
        "ffn_layer": "swiglufused",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TIPS vision checkpoints to Equimo."
    )
    parser.add_argument("variants", nargs="*", choices=sorted(configs))
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the upstream `tips` Python package.",
    )
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Directory containing checkpoints with their upstream filenames.",
    )
    checkpoints.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path when converting exactly one variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.cache/equimo/tips").expanduser(),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--upstream-revision",
        help="Commit of the supplied TIPS source checkout for provenance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved conversion work.",
    )
    args = parser.parse_args()

    variants = args.variants or sorted(configs)
    if args.checkpoint is not None and len(variants) != 1:
        parser.error("--checkpoint requires exactly one variant")

    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        parser.error(f"source directory does not exist: {source_dir}")

    if args.checkpoint is not None:
        checkpoint_paths = {variants[0]: args.checkpoint.expanduser().resolve()}
    else:
        checkpoint_root = args.checkpoint_root.expanduser().resolve()
        if not checkpoint_root.is_dir():
            parser.error(f"checkpoint root does not exist: {checkpoint_root}")
        checkpoint_paths = {
            variant: checkpoint_root / CHECKPOINT_FILENAMES[variant]
            for variant in variants
        }

    for checkpoint_path in checkpoint_paths.values():
        if not checkpoint_path.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint_path}")

    args.variants = variants
    args.source_dir = source_dir
    args.checkpoint_paths = checkpoint_paths
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    for variant in args.variants:
        print(
            f"{variant}: source={args.source_dir} "
            f"checkpoint={args.checkpoint_paths[variant]} "
            f"upstream_revision={args.upstream_revision} seed={args.seed} "
            f"output={args.output_dir / variant}"
        )
    if args.dry_run:
        return

    try:
        import torch

        sys.path.insert(0, str(args.source_dir))
        from tips.pytorch import image_encoder
    except ImportError as exc:
        raise ImportError(
            "`torch` and the upstream `tips` package are required"
        ) from exc

    key = jax.random.PRNGKey(args.seed)
    rng = np.random.default_rng(args.seed)
    base_config = {
        # "img_size": 448,
        "in_channels": 3,
        # "dim": 384,
        "patch_size": 14,
        # "num_heads": [6],
        # "depths": [12],
        "num_classes": 0,
        "use_mask_token": True,
        "reg_tokens": 1,
        "init_values": 1e-5,
        "eps": 1e-6,
        "dynamic_img_size": False,
        "act_layer": "exactgelu",
    }

    for name in args.variants:
        config = configs[name]
        print(f"Converting {name}...")

        cfg = base_config | config

        tips = em.VisionTransformer(
            **cfg,
            key=key,
        )

        weights_image = dict(np.load(args.checkpoint_paths[name], allow_pickle=False))
        for k in weights_image:
            weights_image[k] = torch.tensor(weights_image[k])

        with torch.no_grad():
            # Load the vision encoder.
            model_image = getattr(image_encoder, IMAGE_FACTORIES[name])(
                img_size=224 if "lr" in name else 448,
                patch_size=14,
                ffn_layer="swiglu" if "vitg" in name else "mlp",
                block_chunks=0,
                init_values=1.0,
                interpolate_antialias=True,
                interpolate_offset=0.0,
            )
            model_image.load_state_dict(weights_image)

        replace_cfg = {
            "reg_tokens": "register_tokens",
            "blocks.0.blocks": "blocks",
            ".prenorm.": ".norm1.",
            ".norm.": ".norm2.",
        }
        expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
        squeeze_cfg = {
            "pos_embed": 0,
            "cls_token": 0,
            "register_tokens": 0,
        }
        whitelist = []

        tips, torch_model = convert_torch_to_equinox(
            tips,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict=True,
            source="custom",
            torch_model=model_image,
            return_torch=True,
        )

        arr = rng.standard_normal((3, cfg["img_size"], cfg["img_size"])).astype(
            np.float32
        )
        jax_arr = jnp.array(arr)
        torch_arr = torch.tensor(arr).unsqueeze(0).float()

        assert (
            compare(
                tips.features(jax_arr, key),
                torch_model.forward_features(torch_arr)["x_prenorm"],
            )
            < 1e-5
        )

        save_model(
            args.output_dir / name,
            tips,
            cfg,
            torch_hub_cfg={
                "source_dir": str(args.source_dir),
                "checkpoint": str(args.checkpoint_paths[name]),
                "upstream_revision": args.upstream_revision,
            },
            compression=True,
        )


if __name__ == "__main__":
    main()
