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

from equimo.language.models import TextTransformerEncoder
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.serialization import save_model

CHECKPOINT_FILENAMES = {
    "tips_vits14_hr_text": "tips_oss_s14_highres_distilled_text.npz",
    "tips_vitb14_hr_text": "tips_oss_b14_highres_distilled_text.npz",
    "tips_vitl14_hr_text": "tips_oss_l14_highres_distilled_text.npz",
    "tips_vitso400m14_hr_text": "tips_oss_so400m14_highres_largetext_distilled_text.npz",
    "tips_vitg14_hr_text": "tips_oss_g14_highres_text.npz",
    "tips_vitg14_lr_text": "tips_oss_g14_lowres_text.npz",
}


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


configs = {
    "tips_vits14_hr_text": {
        "dim": 384,
        "num_heads": 6,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.005497702397406101,
    },
    "tips_vitb14_hr_text": {
        "dim": 768,
        "num_heads": 12,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.00397537462413311,
    },
    "tips_vitl14_hr_text": {
        "dim": 1024,
        "num_heads": 16,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.004205586854368448,
    },
    "tips_vitso400m14_hr_text": {
        "dim": 1152,
        "num_heads": 16,
        "depth": 27,
        "mlp_ratio": 4304 / 1152,
        "temperature": 0.002699660835787654,
    },
    "tips_vitg14_hr_text": {
        "dim": 1536,
        "num_heads": 24,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.003517505945637822,
    },
    "tips_vitg14_lr_text": {
        "dim": 1536,
        "num_heads": 24,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.003806645981967449,
    },
}


def get_text_config(v):
    return {
        "hidden_size": {
            "vits14": 384,
            "vitb14": 768,
            "vitl14": 1024,
            "vitso400m14": 1152,
            "vitg14": 1536,
        }[v],
        "mlp_dim": {
            "vits14": 1536,
            "vitb14": 3072,
            "vitl14": 4096,
            "vitso400m14": 4304,
            "vitg14": 6144,
        }[v],
        "num_heads": {
            "vits14": 6,
            "vitb14": 12,
            "vitl14": 16,
            "vitso400m14": 16,
            "vitg14": 24,
        }[v],
        "num_layers": {
            "vits14": 12,
            "vitb14": 12,
            "vitl14": 12,
            "vitso400m14": 27,
            "vitg14": 12,
        }[v],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TIPS text checkpoints to Equimo."
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
        from tips import text_encoder
    except ImportError as exc:
        raise ImportError(
            "`torch` and the upstream `tips` package are required"
        ) from exc

    key = jax.random.PRNGKey(args.seed)
    base_config = {
        "vocab_size": 32000,
        "scale_sqrt_depth": True,
        "act_layer": "relu",
    }

    for name in args.variants:
        config = configs[name]
        print(f"Converting {name}...")

        cfg = base_config | config

        tips_text = TextTransformerEncoder(
            **cfg,
            key=key,
        )

        weights_text = dict(np.load(args.checkpoint_paths[name], allow_pickle=False))
        for k in weights_text:
            weights_text[k] = torch.tensor(weights_text[k])

        with torch.no_grad():
            model_text = text_encoder.TextEncoder(
                get_text_config(name.split("_")[1]),
                vocab_size=32000,
            )
            temperature = weights_text.pop("temperature")

            assert cfg["temperature"] == temperature, (
                f"There is a temp mismatch. Got {cfg['temperature']}, expected {temperature}."
            )

            model_text.load_state_dict(weights_text)

        replace_cfg = {
            "reg_tokens": "register_tokens",
            "blocks.": "resblocks.",
            ".prenorm.": ".ln_1.",
            ".norm.": ".ln_2.",
            ".qkv.weight": ".in_proj_weight",
            ".qkv.bias": ".in_proj_bias",
            ".attn.proj.": ".attn.out_proj.",
            ".mlp.fc1.": ".mlp.c_fc.",
            ".mlp.fc2.": ".mlp.c_proj.",
        }
        expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
        squeeze_cfg = {
            "pos_embed": 0,
            "cls_token": 0,
            "register_tokens": 0,
        }
        whitelist = []

        tips_text, torch_model = convert_torch_to_equinox(
            tips_text,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict=True,
            source="custom",
            torch_model=model_text,
            return_torch=True,
        )

        ids = np.random.randint(0, 100, size=(64))
        paddings = np.zeros_like(ids)
        jax_ids = jnp.array(ids)
        jax_paddings = jnp.array(paddings)

        torch_ids = torch.from_numpy(ids).unsqueeze(0)
        torch_paddings = torch.from_numpy(paddings).unsqueeze(0)

        assert (
            compare(
                tips_text(jax_ids, jax_paddings, key=key),
                torch_model(torch_ids, torch_paddings),
            )
            < 1e-5
        )

        save_model(
            args.output_dir / name,
            tips_text,
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
