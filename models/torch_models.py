"""Generate the vision reference outputs consumed by the test suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "data"
DEFAULT_SEED = 42

REFERENCES = {
    "dinov2_vits14_reg": {
        "filename": "dinov2_vits14_reg_reference.npz",
        "model_id": "timm/vit_small_patch14_reg4_dinov2.lvd142m",
        "revision": "c04b5193082a8d5b0c4856c7937384a48136c5de",
        "input_shape": (3, 518, 518),
    },
    "dinov3_vits16": {
        "filename": "dinov3_vits16_reference.npz",
        "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "revision": "114c1379950215c8b35dfcd4e90a5c251dde0d32",
        "input_shape": (3, 256, 256),
    },
    "eupe_vitt16": {
        "filename": "eupe_vitt16_reference.npz",
        "model_id": "facebook/EUPE-ViT-T",
        "revision": "0a5999ada906be5c16e210f8fcdcf4dd39d40312",
        "source_revision": "7319b8be9be7f38e6c8dff822695cd62f8e4cada",
        "checkpoint_sha256": (
            "b29b906339c9ae21d35a15602ef9d2fce9145828da9ad9cd797fac11ece60487"
        ),
        "rng_discard_shape": (3, 256, 256),
        "input_shape": (3, 224, 224),
    },
    "siglip2_vitb16_256": {
        "filename": "siglip2_vitb16_256_reference.npz",
        "model_id": "google/siglip2-base-patch16-256",
        "revision": "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
        "input_shape": (3, 256, 256),
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identifiers", nargs="+", choices=sorted(REFERENCES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to write reference .npz files.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--revision",
        help="Override the pinned upstream revision (requires one identifier).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Pinned upstream EUPE checkout (required for eupe_vitt16).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="EUPE checkpoint (required for eupe_vitt16).",
    )
    parser.add_argument(
        "--source-revision",
        help="Commit of the supplied EUPE checkout for provenance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved generation work without importing model libraries.",
    )
    args = parser.parse_args(argv)

    if args.revision is not None and len(args.identifiers) != 1:
        parser.error("--revision requires exactly one identifier")

    args.output_dir = args.output_dir.expanduser().resolve()
    if "eupe_vitt16" in args.identifiers:
        if args.source_dir is None or args.checkpoint is None:
            parser.error("eupe_vitt16 requires --source-dir and --checkpoint")
        args.source_dir = args.source_dir.expanduser().resolve()
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.source_dir.is_dir():
            parser.error(f"source directory does not exist: {args.source_dir}")
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
        expected_revision = REFERENCES["eupe_vitt16"]["source_revision"]
        args.source_revision = args.source_revision or expected_revision
    return args


def _revision(args: argparse.Namespace, identifier: str) -> str:
    return args.revision or REFERENCES[identifier]["revision"]


def _print_work(args: argparse.Namespace) -> None:
    for identifier in args.identifiers:
        info = REFERENCES[identifier]
        details = (
            f"{identifier}: model={info['model_id']} "
            f"revision={_revision(args, identifier)} seed={args.seed} "
            f"input_shape={info['input_shape']} "
            f"output={args.output_dir / info['filename']}"
        )
        if identifier == "eupe_vitt16":
            details += (
                f" source={args.source_dir} source_revision={args.source_revision}"
                f" checkpoint={args.checkpoint}"
            )
        print(details)


def _generate_huggingface(identifier, info, revision, image, output):
    import numpy as np
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(info["model_id"], revision=revision).eval()
    if identifier == "siglip2_vitb16_256":
        model = model.vision_model
    with torch.no_grad():
        output_tokens = model(torch.from_numpy(image).unsqueeze(0)).last_hidden_state

    if identifier == "dinov3_vits16":
        arrays = {"cls_token": output_tokens[0, 0].cpu().numpy(), "img": image}
    else:
        arrays = {"patch_tokens": output_tokens[0].cpu().numpy(), "img": image}
    np.savez(output, **arrays)


def _generate_dinov2(info, revision, image, output):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from timm import create_model

    checkpoint = hf_hub_download(
        repo_id=info["model_id"], filename="model.safetensors", revision=revision
    )
    model = create_model(
        "vit_small_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        checkpoint_path=checkpoint,
    ).eval()
    with torch.no_grad():
        tokens = model.forward_features(torch.from_numpy(image).unsqueeze(0))
    np.savez(output, cls_token=tokens[0, 0].cpu().numpy(), img=image)


def _generate_eupe(args, info, image, output):
    import hashlib

    import numpy as np
    import torch

    checksum = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if digest != info["checkpoint_sha256"]:
        raise ValueError(
            f"EUPE checkpoint SHA-256 is {digest}, expected {info['checkpoint_sha256']}"
        )
    model = torch.hub.load(
        str(args.source_dir),
        "eupe_vitt16",
        source="local",
        pretrained=True,
        weights=str(args.checkpoint),
    ).eval()
    with torch.no_grad():
        features = model.forward_features(torch.from_numpy(image).unsqueeze(0))
    if isinstance(features, dict):
        features = features["x_prenorm"]
    np.savez(output, features=features.cpu().numpy(), img=image)


def generate(args: argparse.Namespace) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("The reference dependency group is required") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for identifier in args.identifiers:
        info = REFERENCES[identifier]
        rng = np.random.default_rng(args.seed)
        if discard_shape := info.get("rng_discard_shape"):
            rng.standard_normal(discard_shape)
        image = rng.standard_normal(info["input_shape"]).astype(np.float32)
        output = args.output_dir / info["filename"]
        revision = _revision(args, identifier)
        if identifier == "dinov2_vits14_reg":
            _generate_dinov2(info, revision, image, output)
        elif identifier == "eupe_vitt16":
            _generate_eupe(args, info, image, output)
        else:
            _generate_huggingface(identifier, info, revision, image, output)
        print(f"Saved {identifier} reference to {output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _print_work(args)
    if not args.dry_run:
        generate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
