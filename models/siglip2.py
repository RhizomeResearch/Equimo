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
from equimo.serialization import load_weights, save_model


VARIANT = "siglip2_vitgiantopt16_384"
TIMM_MODEL = "vit_giantopt_patch16_siglip_gap_384.v2_webli"


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert a SigLIP2 checkpoint.")
    parser.add_argument("variant", nargs="?", default=VARIANT, choices=[VARIANT])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.cache/equimo").expanduser(),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--upstream-revision",
        required=True,
        help="Revision of the timm checkpoint repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print resolved work without importing Torch/timm.",
    )
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        f"{args.variant}: model={TIMM_MODEL} "
        f"upstream_revision={args.upstream_revision} "
        f"checkpoint={args.checkpoint} seed={args.seed} "
        f"output={args.output_dir / args.variant}"
    )
    if args.dry_run:
        return

    try:
        import torch
        from timm import create_model
    except ImportError as exc:
        raise ImportError("`torch` and `timm` are required") from exc

    key = jax.random.PRNGKey(args.seed)
    siglip2_config = {
        "img_size": 384,
        "in_channels": 3,
        "dim": 1536,
        "patch_size": 16,
        "num_heads": [16],
        "depths": [40],
        "mlp_ratio": 4,
        "num_classes": 0,
        "use_mask_token": False,
        "reg_tokens": 0,
        "class_token": False,
        "no_embed_class": True,
        "init_values": None,
        "eps": 1e-6,
        "dynamic_img_size": False,
        # "act_layer": "exactgelu",
        "act_layer": "gelu",
    }

    siglip2 = em.VisionTransformer(
        **siglip2_config,
        key=key,
    )

    torch_model = create_model(
        TIMM_MODEL,
        pretrained=False,
        checkpoint_path=str(args.checkpoint),
    ).eval()

    print(f"Converting {args.variant}...")

    replace_cfg = {
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

    siglip2, torch_model = convert_torch_to_equinox(
        siglip2,
        replace_cfg,
        expand_cfg,
        squeeze_cfg,
        whitelist,
        strict=True,
        source="custom",
        torch_model=torch_model,
        return_torch=True,
    )

    arr = (
        np.random.default_rng(args.seed)
        .standard_normal((3, siglip2_config["img_size"], siglip2_config["img_size"]))
        .astype(np.float32)
    )
    jax_arr = jnp.array(arr)
    torch_arr = torch.tensor(arr).unsqueeze(0).float()

    assert (
        compare(
            jax.vmap(siglip2.norm)(siglip2.features(jax_arr, key)),
            torch_model.forward_features(torch_arr),
        )
        < 1e-5
    )

    save_model(
        args.output_dir / args.variant,
        siglip2,
        siglip2_config,
        torch_hub_cfg={
            "checkpoint": str(args.checkpoint),
            "upstream_revision": args.upstream_revision,
        },
        compression=True,
    )

    _ = load_weights(
        em.VisionTransformer(**siglip2_config, key=key),
        path=args.output_dir / f"{args.variant}.tar.lz4",
    )


if __name__ == "__main__":
    main()
