"""Compare converted LingBot-Vision features with the pinned author implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from equimo._io import atomic_file
from equimo.serialization import load_weights
from equimo.vision.layers.posemb import VisionRoPE
import equimo.vision.models as vision_models
from models.lingbot import SOURCE_REVISION, VARIANTS, sha256_file


TAPS = {
    "small": (2, 5, 8, 11),
    "base": (2, 5, 8, 11),
    "large": (5, 11, 17, 23),
    "giant": (9, 19, 29, 39),
}
ATOL = 1.5e-4
RTOL = 1e-4


def _author_model(variant: str, author_dir: Path, checkpoint: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(author_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Author source revision mismatch: {revision}")
    sys.path.insert(0, str(author_dir))
    try:
        import torch
        from lingbot_vision.build import build_backbone_from_cfg
        from lingbot_vision.loader import load_config
    except ImportError as error:
        raise ImportError(
            "Reference comparison requires torch and omegaconf."
        ) from error

    torch.set_num_threads(1)
    letter = {"small": "s", "base": "b", "large": "l", "giant": "g"}[variant]
    config = load_config(
        author_dir / "lingbot_vision" / "configs" / f"lbot_vision_vit{letter}.yaml"
    )
    model, _ = build_backbone_from_cfg(config)
    wrapped = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(wrapped, dict) or set(wrapped) != {"model"}:
        raise ValueError("Reference checkpoint has an unexpected wrapper.")
    model.load_state_dict(wrapped["model"], strict=True)
    return model.eval(), torch


def _compare(actual, expected) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"Feature shape/dtype mismatch: {actual.shape}/{actual.dtype} "
            f"versus {expected.shape}/{expected.dtype}."
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Feature output contains non-finite values.")
    error = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_absolute_error": float(error.max(initial=0)),
        "violations": int(np.count_nonzero(error > ATOL + RTOL * np.abs(expected))),
    }


def _tokens(reference):
    patch, cls, registers = reference
    return np.concatenate(
        (cls[:, None, :].numpy(), registers.numpy(), patch.numpy()), axis=1
    )[0]


def compare_variant(
    variant: str,
    *,
    author_dir: Path,
    source_dir: Path,
    archive_dir: Path,
    shapes: tuple[tuple[int, int], ...],
    precision: str,
    fixture_dir: Path | None = None,
) -> dict:
    identifier, source_revision, expected_digest = VARIANTS[variant]
    checkpoint = source_dir / f"{variant}.pt"
    if sha256_file(checkpoint) != expected_digest:
        raise ValueError(f"{variant} source checkpoint digest mismatch.")
    record_path = archive_dir / f"{identifier}.conversion.json"
    record = json.loads(record_path.read_text())
    if (
        record["variant"] != identifier
        or record["source_code_revision"] != SOURCE_REVISION
        or record["source_checkpoint_revision"] != source_revision
        or record["source_checkpoint_sha256"] != expected_digest
    ):
        raise ValueError(f"{variant} conversion record identity mismatch.")
    archive = archive_dir / f"{identifier}.tar.lz4"
    if sha256_file(archive) != record["converted_archive_sha256"]:
        raise ValueError(f"{variant} converted archive digest mismatch.")

    author, torch = _author_model(variant, author_dir, checkpoint)
    native = load_weights(
        getattr(vision_models, identifier)(),
        path=archive,
        expected_sha256=record["converted_archive_sha256"],
        expected_model_config=record["model_config"],
    )
    if precision == "float64":
        author = author.double()
        jax.config.update("jax_enable_x64", True)
        architecture = record["model_config"]["architecture"]
        rope_config = architecture["local_pos_embed_config_patch"]
        rope = VisionRoPE(
            "period",
            dim=native.dim,
            num_heads=architecture["num_heads"],
            base=rope_config["base"],
            normalize_coords=rope_config["normalize_coords"],
            rescale_coords=rope_config["rescale_coords"],
            dtype=jnp.float64,
            periods_dtype=jnp.float64,
        )
        rope = eqx.tree_at(
            lambda layer: layer.freqs,
            rope,
            native.local_pos_embed.patch_rope.freqs.astype(jnp.float64),
        )
        native = eqx.tree_at(
            lambda model: model.local_pos_embed.patch_rope, native, rope
        )
        native = jax.tree_util.tree_map(
            lambda leaf: (
                leaf.astype(jnp.float64)
                if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
                else leaf
            ),
            native,
        )
    taps = TAPS[variant]
    cases = []
    fixture_arrays = None
    for height, width in shapes:
        image = (
            np.random.default_rng(42)
            .standard_normal((3, height, width))
            .astype(np.float32)
        )
        if precision == "float64":
            image = image.astype(np.float64)
        with torch.no_grad():
            reference_input = torch.from_numpy(image)[None]
            final_ref = author.forward_features(reference_input)
            raw_ref = author.get_intermediate_layers(
                reference_input,
                n=taps,
                norm=False,
                return_class_token=True,
                return_extra_tokens=True,
            )
            norm_ref = author.get_intermediate_layers(
                reference_input,
                n=taps,
                norm=True,
                return_class_token=True,
                return_extra_tokens=True,
            )
        key = jax.random.PRNGKey(42)
        final_native = native.forward_features(
            jnp.asarray(image), key=key, inference=True
        )
        raw_native = native.intermediate_features(
            jnp.asarray(image), key=key, inference=True, indices=taps
        )
        norm_native = native.intermediate_features(
            jnp.asarray(image), key=key, inference=True, indices=taps, apply_norm=True
        )
        if fixture_dir is not None:
            if len(shapes) != 1:
                raise ValueError("Compact reference fixtures require one image.")
            fixture_arrays = {
                "input": image,
                "class": final_ref["x_norm_clstoken"][0].numpy(),
                "registers": final_ref["x_storage_tokens"][0].numpy(),
                "patches": final_ref["x_norm_patchtokens"][0].numpy(),
                "prenorm": final_ref["x_prenorm"][0].numpy(),
            }
            for tap, raw, norm in zip(taps, raw_ref, norm_ref, strict=True):
                fixture_arrays[f"tap_{tap}_raw"] = _tokens(raw)
                fixture_arrays[f"tap_{tap}_norm"] = _tokens(norm)
        metrics = {
            "class": _compare(
                final_native["x_norm_cls_token"],
                final_ref["x_norm_clstoken"][0].numpy(),
            ),
            "registers": _compare(
                final_native["x_norm_reg_tokens"],
                final_ref["x_storage_tokens"][0].numpy(),
            ),
            "patches": _compare(
                final_native["x_norm_patchtokens"],
                final_ref["x_norm_patchtokens"][0].numpy(),
            ),
            "prenorm": _compare(
                final_native["x_prenorm"], final_ref["x_prenorm"][0].numpy()
            ),
        }
        metrics.update(
            {
                f"tap_{tap}_raw": _compare(actual, _tokens(expected))
                for tap, actual, expected in zip(taps, raw_native, raw_ref, strict=True)
            }
        )
        metrics.update(
            {
                f"tap_{tap}_norm": _compare(actual, _tokens(expected))
                for tap, actual, expected in zip(
                    taps, norm_native, norm_ref, strict=True
                )
            }
        )
        metadata = native.feature_metadata(
            jnp.asarray(image),
            endpoint="intermediate_features",
            endpoint_options={"indices": taps, "apply_norm": True},
        )
        expected_grid = (height // 16, width // 16)
        if metadata["grid_size"] != expected_grid:
            raise ValueError(f"Incorrect patch geometry: {metadata['grid_size']}")
        cases.append({"image_size": [height, width], "metrics": metrics})

    report = {
        "variant": identifier,
        "author_code_revision": SOURCE_REVISION,
        "source_checkpoint_revision": source_revision,
        "source_checkpoint_sha256": expected_digest,
        "converted_archive_sha256": record["converted_archive_sha256"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "evaluation_precision": precision,
        "checkpoint_weight_precision": "float32",
        "cases": cases,
        "passed": all(
            metric["violations"] == 0
            for case in cases
            for metric in case["metrics"].values()
        ),
    }
    if fixture_dir is not None and report["passed"]:
        assert fixture_arrays is not None
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_path = fixture_dir / f"{identifier}_reference.npz"
        with atomic_file(fixture_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **fixture_arrays)
        report["fixture"] = fixture_path.name
        report["fixture_sha256"] = sha256_file(fixture_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--author-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    shapes = ((32, 48),) if args.quick else ((512, 512), (320, 512), (512, 320))
    report = compare_variant(
        args.variant,
        author_dir=args.author_dir,
        source_dir=args.source_dir,
        archive_dir=args.archive_dir,
        shapes=shapes,
        precision=args.precision,
        fixture_dir=args.fixture_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"{report['variant']}.{args.precision}."
        f"{'quick' if args.quick else 'full'}.qualification.json"
    )
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(path), "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
