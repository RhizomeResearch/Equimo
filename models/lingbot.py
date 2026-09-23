"""Convert pinned LingBot-Vision checkpoints to local Equimo archives.

The original PyTorch format is read only by this maintainer script. Native
inference uses the resulting Equimo archive and does not import PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from equimo._io import atomic_file
from equimo.serialization import save_model
from equimo.vision.models.vit import _VIT_REGISTRY
import equimo.vision.models as vision_models


SOURCE_REVISION = "151e46321bae4399f8568829f190c7bdec216b49"
CONVERSION_VERSION = 1
VARIANTS = {
    "small": (
        "lingbot_vits16",
        "127cbcec380de0bcd55bdc1b1fad3819850a6514",
        "dca36562cb6b0b34504df6edc18fa282c5ef06fb375c3e91d5487247a1096f9d",
    ),
    "base": (
        "lingbot_vitb16",
        "f606f8c6c4002234ea68038f4d7c7cf57da96dfa",
        "783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825",
    ),
    "large": (
        "lingbot_vitl16",
        "5e0370623d4fa5db945d00bc47a8545eed407d6b",
        "5b5eb67ebbf990b747658ecf90f1cf2b93f5b0e8dfdfbceb060ae1fc364deb8f",
    ),
    "giant": (
        "lingbot_vitg16",
        "f87d0865a0ae06e640a09be153cb0ba2bda5156d",
        "ba9d0b3058b12166c491079b09a67aae6c9cd4cc46299717524c58ba6a4fe8bf",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_name(destination: str) -> str:
    """Map one Equimo array path to its author checkpoint tensor."""
    name = destination.lstrip(".")
    names = {
        "local_pos_embed.patch_rope.freqs": "rope_embed.periods",
        "reg_tokens": "storage_tokens",
    }
    if name in names:
        return names[name]
    block = re.fullmatch(r"blocks\[0\]\.blocks\[(\d+)\]\.(.+)", name)
    if block is not None:
        suffix = block[2].replace("prenorm.", "norm1.").replace("norm.", "norm2.")
        return f"blocks.{block[1]}.{suffix}"
    return name


def _json_config(value):
    if isinstance(value, dict):
        return {key: _json_config(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(part) for part in value]
    if isinstance(value, type):
        return np.dtype(value).name
    return value


def _load_source(path: Path, expected_digest: str) -> dict:
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"Source checkpoint digest mismatch: expected {expected_digest}, "
            f"got {actual_digest}."
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Source checkpoint contains duplicate archive members.")
    try:
        import torch
    except ImportError as error:
        raise ImportError("Checkpoint conversion requires the torch extra.") from error

    wrapped = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(wrapped, dict) or set(wrapped) != {"model"}:
        raise ValueError("Expected one 'model' state dictionary in source checkpoint.")
    state = wrapped["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Source checkpoint has no model tensor inventory.")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("Source checkpoint contains a non-tensor model entry.")
    return state


def _check_bias_masks(
    state: dict, *, depth: int, width: int, qkv_bias: bool
) -> set[str]:
    accounted = set()
    for index in range(depth):
        name = f"blocks.{index}.attn.qkv.bias_mask"
        if not qkv_bias:
            if name in state:
                raise ValueError(f"Unexpected key-bias mask {name}.")
            continue
        if name not in state:
            raise ValueError(f"Missing key-bias mask {name}.")
        mask = np.asarray(state[name])
        expected = np.concatenate((np.ones(width), np.zeros(width), np.ones(width)))
        if mask.shape != expected.shape or not np.array_equal(mask, expected):
            raise ValueError(f"Invalid key-bias mask {name}.")
        accounted.add(name)
    return accounted


def convert(variant: str, source: Path, output_dir: Path) -> tuple[Path, Path]:
    identifier, checkpoint_revision, digest = VARIANTS[variant]
    archive = output_dir / f"{identifier}.tar.lz4"
    record_path = output_dir / f"{identifier}.conversion.json"
    if archive.exists() or record_path.exists():
        raise FileExistsError(f"Conversion output already exists for {identifier}.")

    state = _load_source(source, digest)
    base, specific = _VIT_REGISTRY[identifier]
    model_config = base | specific
    width = model_config["dim"]
    depth = sum(model_config["depths"])
    accounted = _check_bias_masks(
        state, depth=depth, width=width, qkv_bias=model_config.get("qkv_bias", True)
    )
    model = getattr(vision_models, identifier)(pretrained=False)
    mapping: dict[str, str] = {}

    def replace(path, leaf):
        if not eqx.is_array(leaf):
            return leaf
        destination = jax.tree_util.keystr(path)
        name = source_name(destination)
        if name in accounted:
            raise ValueError(f"Source tensor {name!r} maps more than once.")
        if name not in state:
            raise ValueError(f"Missing source tensor {name!r} for {destination}.")
        value = np.asarray(state[name])
        if destination in {".cls_token", ".reg_tokens"}:
            value = value.squeeze(0)
        elif destination == ".patch_embed.proj.bias":
            value = value[:, None, None]
        if value.shape != leaf.shape:
            raise ValueError(
                f"Tensor {name!r} has shape {value.shape}, expected {leaf.shape}."
            )
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Tensor {name!r} is not finite float32.")
        accounted.add(name)
        mapping[name] = destination
        return jnp.asarray(value)

    model = jax.tree_util.tree_map_with_path(replace, model)
    unexpected = set(state) - accounted
    if unexpected:
        raise ValueError(f"Unaccounted source tensors: {sorted(unexpected)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = save_model(
        archive,
        eqx.nn.inference_mode(model, True),
        {"variant": identifier, "architecture": _json_config(model_config)},
    )
    record = {
        "conversion_version": CONVERSION_VERSION,
        "variant": identifier,
        "source_repository": "robbyant/lingbot-vision",
        "source_code_revision": SOURCE_REVISION,
        "source_checkpoint_repository": f"robbyant/lingbot-vision-vit-{variant}",
        "source_checkpoint_revision": checkpoint_revision,
        "source_checkpoint_sha256": digest,
        "source_code_license": "Apache-2.0",
        "source_checkpoint_license": "Apache-2.0",
        "converted_archive_sha256": sha256_file(archive),
        "model_config": {
            "variant": identifier,
            "architecture": _json_config(model_config),
        },
        "source_tensor_inventory": {
            name: {"shape": list(state[name].shape), "dtype": str(state[name].dtype)}
            for name in sorted(accounted)
        },
        "tensor_map": mapping,
        "validated_buffers": sorted(accounted - set(mapping)),
        "jax_version": jax.__version__,
    }
    with atomic_file(record_path) as temporary:
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return archive, record_path


def _download(variant: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Automatic source download requires huggingface_hub; provide "
            "--checkpoint-dir with the four verified local .pt files instead."
        ) from error
    _, revision, _ = VARIANTS[variant]
    return Path(
        hf_hub_download(
            repo_id=f"robbyant/lingbot-vision-vit-{variant}",
            filename="model.pt",
            revision=revision,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", choices=tuple(VARIANTS))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("~/.cache/equimo/lingbot").expanduser()
    )
    args = parser.parse_args()
    variants = tuple(VARIANTS) if args.all else (args.variant,)
    for variant in variants:
        source = (
            args.checkpoint_dir / f"{variant}.pt"
            if args.checkpoint_dir is not None
            else _download(variant)
        )
        archive, record = convert(variant, source, args.output_dir)
        print(f"{variant}: {archive} ({record})")


if __name__ == "__main__":
    main()
