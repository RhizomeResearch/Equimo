from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)


VARIANTS = {
    "eupe_vitt16": {"family": "vit"},
    "eupe_vits16": {"family": "vit"},
    "eupe_vitb16": {"family": "vit"},
    "eupe_convnext_tiny": {"family": "convnext", "depths": [3, 3, 9, 3]},
    "eupe_convnext_small": {"family": "convnext", "depths": [3, 3, 27, 3]},
    "eupe_convnext_base": {"family": "convnext", "depths": [3, 3, 27, 3]},
}
IMG_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an EUPE checkpoint to Equimo."
    )
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to the upstream EUPE repository used by torch.hub.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.cache/equimo/eupe").expanduser(),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Trace layer outputs to help locate numerical divergence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved conversion work.",
    )
    args = parser.parse_args()

    args.source_dir = args.source_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.source_dir.is_dir():
        parser.error(f"source directory does not exist: {args.source_dir}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def trace_model_divergence_vit(jax_model, pt_model, arr, key):
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    import torch
    from einops import rearrange

    x_pt = torch.tensor(arr).unsqueeze(0).float()
    x_jax = jnp.array(arr)
    print(f"\n{'Module':<25} | {'Max Error':<15} | {'Mean Abs Error':<15}")
    print("-" * 62)

    x_pt, hw_tuple = pt_model.prepare_tokens_with_masks(x_pt)
    rope_sincos_pt = (
        pt_model.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
        if getattr(pt_model, "rope_embed", None) is not None
        else None
    )

    key_pos, *_ = jr.split(key, len(jax_model.blocks) + 1)
    x_jax_tokens = jax_model.patch_embed(x_jax)
    if jax_model.local_pos_embed is not None:
        if jax_model.dynamic_img_size:
            _, height, width = x_jax_tokens.shape
        else:
            height = width = jax_model.embed_size
    else:
        height = width = None

    if jax_model.global_pos_embed is not None:
        x_jax_tokens = jax_model.global_pos_embed(
            x_jax_tokens,
            cls_token=jax_model.cls_token,
            reg_tokens=jax_model.reg_tokens,
            dynamic_img_size=jax_model.dynamic_img_size,
        )
    else:
        prefix = [
            token
            for token in (jax_model.cls_token, jax_model.reg_tokens)
            if token is not None
        ]
        if jax_model.dynamic_img_size:
            x_jax_tokens = rearrange(x_jax_tokens, "c h w -> (h w) c")
        if prefix:
            x_jax_tokens = jnp.concatenate([*prefix, x_jax_tokens], axis=0)

    rope_sincos_jax = None
    if jax_model.local_pos_embed is not None:
        rope_sincos_jax = jax_model.local_pos_embed.get_sincos(
            H=height, W=width, inference=True, key=key_pos
        )

    diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax_tokens))
    print(f"{'Token Prep':<25} | {diff.max():<15.6f} | {diff.mean():<15.6f}")
    if diff.mean() > 1e-4:
        print("\n[!] Divergence isolated at Token Preparation.")
        return

    jax_blocks = [block for chunk in jax_model.blocks for block in chunk.blocks]
    for index, (pt_block, jax_block) in enumerate(zip(pt_model.blocks, jax_blocks)):
        x_pt = (
            pt_block(x_pt, rope_sincos_pt)
            if rope_sincos_pt is not None
            else pt_block(x_pt)
        )
        key, subkey = jax.random.split(key)
        x_jax_tokens = jax_block(
            x_jax_tokens,
            rope_sincos=rope_sincos_jax,
            inference=True,
            key=subkey,
        )
        diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax_tokens))
        print(f"Block {index:<20} | {diff.max():<15.6f} | {diff.mean():<15.6f}")
        if diff.mean() > 1e-4:
            print(f"\n[!] Divergence isolated at Block {index}.")
            return

    print("\nNo divergence found during macro-tracing.")


def trace_model_divergence_convnext(jax_model, pt_model, arr, key):
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    import torch

    x_pt = torch.tensor(arr).unsqueeze(0).float()
    x_jax = jnp.array(arr)
    print(f"\n{'Module':<25} | {'Max Error':<15} | {'Mean Abs Error':<15}")
    print("-" * 62)
    for stage_index in range(4):
        pt_down = pt_model.downsample_layers[stage_index]
        jax_down = jax_model.blocks[stage_index].downsample
        x_pt = pt_down(x_pt)
        x_jax = jax_down(x_jax, inference=True, key=jr.PRNGKey(42))
        diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax))
        print(
            f"Stage {stage_index} Downsampler{' ':<6} | "
            f"{diff.max():<15.6f} | {diff.mean():<15.6f}"
        )
        if diff.mean() > 1e-4:
            print(f"\n[!] Divergence isolated at Stage {stage_index} Downsampler.")
            return

        pt_stage = pt_model.stages[stage_index]
        jax_blocks = jax_model.blocks[stage_index].blocks
        for block_index, (pt_block, jax_block) in enumerate(zip(pt_stage, jax_blocks)):
            x_pt = pt_block(x_pt)
            key, subkey = jax.random.split(key)
            x_jax = jax_block(x_jax, inference=True, key=subkey)
            diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax))
            print(
                f"Stage {stage_index} Block {block_index:<10} | "
                f"{diff.max():<15.6f} | {diff.mean():<15.6f}"
            )
            if diff.mean() > 1e-4:
                print(
                    f"\n[!] Divergence isolated at Stage {stage_index}, "
                    f"Block {block_index}."
                )
                return

    print("\nNo divergence found during macro-tracing.")


def conversion_config(variant):
    info = VARIANTS[variant]
    if info["family"] == "vit":
        return (
            {
                "reg_tokens": "storage_tokens",
                "blocks.0.blocks": "blocks",
                ".prenorm.": ".norm1.",
                ".norm.": ".norm2.",
            },
            {"patch_embed.proj.bias": ["after", 2]},
            {"pos_embed": 0, "cls_token": 0, "storage_tokens": 0},
            ["local_pos_embed.patch_rope.freqs"],
        )

    replace_cfg = {
        "blocks.0.downsample.conv": "downsample_layers.0.0",
        "blocks.0.downsample.norm": "downsample_layers.0.1",
        "blocks.1.downsample.norm": "downsample_layers.1.0",
        "blocks.1.downsample.conv": "downsample_layers.1.1",
        "blocks.2.downsample.norm": "downsample_layers.2.0",
        "blocks.2.downsample.conv": "downsample_layers.2.1",
        "blocks.3.downsample.norm": "downsample_layers.3.0",
        "blocks.3.downsample.conv": "downsample_layers.3.1",
        ".blocks.": ".",
        "blocks.": "stages.",
        "ls.gamma": "gamma",
    }
    expand_cfg = {
        "downsample_layers.0.0.bias": ["after", 2],
        "downsample_layers.1.1.bias": ["after", 2],
        "downsample_layers.2.1.bias": ["after", 2],
        "downsample_layers.3.1.bias": ["after", 2],
    }
    for stage_index, depth in enumerate(info["depths"]):
        for block_index in range(depth):
            base_path = f"stages.{stage_index}.{block_index}"
            expand_cfg[f"{base_path}.dwconv.bias"] = ["after", 2]
            expand_cfg[f"{base_path}.pwconv1.weight"] = ["after", 2]
            expand_cfg[f"{base_path}.pwconv1.bias"] = ["after", 2]
            expand_cfg[f"{base_path}.pwconv2.weight"] = ["after", 2]
            expand_cfg[f"{base_path}.pwconv2.bias"] = ["after", 2]
    return replace_cfg, expand_cfg, {}, []


def compare(jax_array, torch_tensor) -> float:
    import numpy as np

    return float(
        np.mean(np.abs(np.asarray(jax_array) - torch_tensor.squeeze().detach().numpy()))
    )


def convert(args):
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError("`torch`, Equinox, JAX, and NumPy are required") from exc

    import equimo.vision.models as em
    from equimo.conversion.utils import convert_torch_to_equinox
    from equimo.serialization import save_model

    info = VARIANTS[args.variant]
    key = jax.random.PRNGKey(42)
    print(f"Converting {args.variant}...")
    eupe = getattr(em, args.variant)()
    torch_hub_cfg = {
        "repo_or_dir": str(args.source_dir),
        "model": args.variant,
        "source": "local",
        "pretrained": True,
        "weights": str(args.checkpoint),
    }
    replace_cfg, expand_cfg, squeeze_cfg, jax_whitelist = conversion_config(
        args.variant
    )
    eupe, torch_model = convert_torch_to_equinox(
        eupe,
        replace_cfg,
        expand_cfg,
        squeeze_cfg,
        [],
        jax_whitelist,
        strict=True,
        torch_hub_cfg=torch_hub_cfg,
        return_torch=True,
    )
    eupe = eqx.nn.inference_mode(eupe, True)

    arr = np.random.default_rng(42).standard_normal((3, IMG_SIZE, IMG_SIZE))
    jax_arr = jnp.array(arr)
    torch_arr = torch.tensor(arr).unsqueeze(0).float()
    if args.diagnose:
        diagnostic = (
            trace_model_divergence_vit
            if info["family"] == "vit"
            else trace_model_divergence_convnext
        )
        diagnostic(eupe, torch_model, arr, key)

    torch_features = torch_model.forward_features(torch_arr)
    if isinstance(torch_features, dict):
        torch_features = torch_features["x_prenorm"]
    error = compare(
        eupe.features(jax_arr, inference=True, key=key),
        torch_features,
    )
    assert error < 5e-4, f"Conversion error: {error}"
    print("err:", error)

    save_model(
        args.output_dir / args.variant,
        eupe,
        {},
        torch_hub_cfg,
        compression=True,
    )


def main():
    args = parse_args()
    print(
        f"{args.variant}: source={args.source_dir} "
        f"checkpoint={args.checkpoint} "
        f"output={args.output_dir / args.variant}"
    )
    if not args.dry_run:
        convert(args)


if __name__ == "__main__":
    main()
