"""Qualify local vision probes against admitted upstream tensors and ONNX.

Inputs are a digest-pinned local checkpoint and an upstream reference package.
The reference package contains checkpoint-record.json, qualification-report.json,
numerical-results.json, and the case NPZ files named in numerical-results.json.
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import equimo.finetune as eqft
from equimo.serialization import CheckpointLimits, load_weights
from equimo.vision.models import dinov3_vits16_pretrain_lvd1689m


DEFAULT_CASES = ("noise-256", "coordinates-thin-512x512", "coordinates-thin-320x512")
ATOL = 1.5e-4
RTOL = 1e-4
READER_LIMITS = CheckpointLimits(
    max_archive_bytes=1 << 30,
    max_metadata_bytes=1 << 20,
    max_member_bytes=1 << 30,
    max_expanded_bytes=1 << 30,
    max_tensor_count=10_000,
    max_tensor_rank=8,
    max_tensor_dimension=65_536,
    max_tensor_bytes=1 << 30,
    max_total_array_bytes=1 << 30,
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def probe_specs() -> tuple[eqft.FeatureSpec, eqft.FeatureSpec]:
    return (
        eqft.FeatureSpec("forward_features", "BNC", "cls", None),
        eqft.FeatureSpec(
            "forward_features", "BNC", "patches", None, return_metadata=True
        ),
    )


def build_probes(backbone):
    pooled_spec, spatial_spec = probe_specs()
    pooled = eqft.make_linear_probe(
        backbone,
        in_features=384,
        out_features=10,
        key=jr.PRNGKey(51),
        feature_spec=pooled_spec,
    )
    spatial = eqft.vision.make_dense_probe(
        backbone,
        in_features=384,
        out_features=3,
        key=jr.PRNGKey(52),
        feature_spec=spatial_spec,
    )
    return pooled, spatial


def checked_metric(actual, expected) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"Reference shape/dtype mismatch: {actual.shape}/{actual.dtype} "
            f"versus {expected.shape}/{expected.dtype}."
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Reference comparison contains non-finite values.")
    delta = np.abs(actual - expected)
    allowance = ATOL + RTOL * np.abs(expected)
    violations = int(np.count_nonzero(delta > allowance))
    if violations:
        raise ValueError(
            f"Reference comparison has {violations} values outside tolerance."
        )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_absolute_error": float(delta.max(initial=0)),
        "violations": violations,
    }


def reference_package(
    directory: Path, cases: tuple[str, ...]
) -> tuple[dict, dict, dict]:
    checkpoint = read_json(directory / "checkpoint-record.json")
    report = read_json(directory / "qualification-report.json")
    numerical = read_json(directory / "numerical-results.json")
    if report.get("status") != "passed_synthetic":
        raise ValueError("Reference package has no accepted synthetic profile.")
    for name in ("checkpoint-record.json", "numerical-results.json"):
        if digest(directory / name) != report["report_digests"][name]:
            raise ValueError(f"Reference record {name} has changed.")
    rows = {row["case"]: row for row in numerical["cases"]}
    for case in cases:
        if case not in rows:
            raise ValueError(f"Reference case {case!r} is absent.")
        if digest(directory / f"{case}.npz") != rows[case]["fixture_sha256"]:
            raise ValueError(f"Reference case {case!r} has changed.")
    return checkpoint, report, rows


def loaded_backbone(checkpoint_path: Path, record: dict):
    if digest(checkpoint_path) != record["archive_sha256"]:
        raise ValueError("Checkpoint archive does not match its admitted digest.")
    return load_weights(
        dinov3_vits16_pretrain_lvd1689m(pretrained=False),
        path=checkpoint_path,
        expected_sha256=record["archive_sha256"],
        expected_weights_sha256=record["weights_sha256"],
        expected_model_config=record["model_config"],
        limits=READER_LIMITS,
    )


def compare_case(pooled, spatial, reference_dir: Path, name: str) -> dict:
    with np.load(reference_dir / f"{name}.npz", allow_pickle=False) as fixture:
        image = jnp.asarray(fixture["input"])
        cls_reference = np.asarray(fixture["x_norm_cls_token"])
        patch_reference = np.asarray(fixture["x_norm_patchtokens"])
    height, width = image.shape[-2:]
    grid = (height // 16, width // 16)
    if patch_reference.shape != (grid[0] * grid[1], 384):
        raise ValueError(f"Reference patch geometry is invalid for {name}.")
    key = jr.PRNGKey(42)
    pooled_features = eqft.extract_features(
        pooled.backbone,
        image,
        feature_spec=pooled.feature_spec,
        key=key,
        inference=True,
    )
    spatial_features = eqft.extract_features(
        spatial.backbone,
        image,
        feature_spec=spatial.feature_spec,
        key=key,
        inference=True,
    )
    metrics = {
        "pooled_features": checked_metric(pooled_features, cls_reference),
        "spatial_features": checked_metric(spatial_features.features, patch_reference),
    }
    pooled_head = pooled.head.linear
    spatial_head = spatial.head.linear
    expected_pooled = np.asarray(pooled_head.weight) @ cls_reference + np.asarray(
        pooled_head.bias
    )
    expected_spatial = (np.asarray(spatial_head.weight) @ patch_reference.T).reshape(
        3, *grid
    ) + np.asarray(spatial_head.bias)[:, None, None]
    pooled_logits = pooled(image, key=key, inference=True)
    spatial_logits = spatial(image, key=key, inference=True)
    metrics["pooled_logits"] = checked_metric(pooled_logits, expected_pooled)
    metrics["spatial_logits"] = checked_metric(spatial_logits, expected_spatial)
    return {"case": name, "grid": list(grid), "metrics": metrics}


def export_probe(args) -> dict:
    import onnx
    import onnxruntime as ort
    from jax2onnx import to_onnx

    record = read_json(args.reference_dir / "checkpoint-record.json")
    backbone = loaded_backbone(args.checkpoint, record)
    pooled, spatial = build_probes(backbone)
    probe = pooled if args.head == "pooled" else spatial
    with np.load(
        args.reference_dir / f"{args.export_case}.npz", allow_pickle=False
    ) as fixture:
        image = jnp.asarray(fixture["input"])[None]

    def inference(batch):
        return jax.vmap(
            lambda sample: probe(sample, key=jr.PRNGKey(42), inference=True)
        )(batch)

    proto = to_onnx(
        inference,
        inputs=[image],
        opset=18,
        input_names=["image"],
        output_names=["logits"],
    )
    onnx.checker.check_model(proto)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(["logits"], {"image": np.asarray(image)})
    metric = checked_metric(actual, inference(image))
    return {
        "status": "passed",
        "head": args.head,
        "input_shape": list(image.shape),
        "opset": 18,
        "graph_sha256": hashlib.sha256(proto.SerializeToString()).hexdigest(),
        "runtime_metric": metric,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--portable-output", type=Path)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--export-case", default="noise-256")
    parser.add_argument("--export-timeout", type=int, default=120)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--head", choices=("pooled", "spatial"))
    args = parser.parse_args()
    if args.export_only:
        if args.head is None:
            parser.error("--export-only requires --head")
        try:
            print(json.dumps(export_probe(args), sort_keys=True))
            return 0
        except Exception as error:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "head": args.head,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(limit=12),
                    },
                    sort_keys=True,
                )
            )
            return 1
    if args.output is None:
        parser.error("--output is required")
    cases = tuple(args.cases)
    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "runner": Path(__file__),
        "serialization": root / "src/equimo/serialization.py",
        "checkpoint_limits": root / "src/equimo/_checkpoint_limits.py",
        "vision_transformer": root / "src/equimo/vision/models/vit.py",
        "feature_extraction": root / "src/equimo/finetune/feature_extraction.py",
        "dense_probe": root / "src/equimo/finetune/vision/dense.py",
        "heads": root / "src/equimo/finetune/heads.py",
    }
    result = {
        "status": "running",
        "checkpoint": str(args.checkpoint),
        "reference_dir": str(args.reference_dir),
        "cases": cases,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "reader_limits": asdict(READER_LIMITS),
        "environment": {
            package: importlib.metadata.version(package)
            for package in (
                "equimo",
                "jax",
                "jaxlib",
                "equinox",
                "numpy",
                "jax2onnx",
                "onnxruntime",
            )
        },
        "backend": jax.default_backend(),
        "source_sha256": {name: digest(path) for name, path in source_paths.items()},
    }
    try:
        record, reference_report, rows = reference_package(args.reference_dir, cases)
        result["source_revision"] = record["upstream_revision"]
        result["source_weights_sha256"] = record["upstream_weights_sha256"]
        result["checkpoint_sha256"] = digest(args.checkpoint)
        result["reference_report_sha256"] = digest(
            args.reference_dir / "qualification-report.json"
        )
        result["reference_case_sha256"] = {
            name: rows[name]["fixture_sha256"] for name in cases
        }
        result["reference_profile"] = {
            "precision": reference_report["precision"],
            "preprocessing": reference_report["preprocessing"],
        }
        backbone = loaded_backbone(args.checkpoint, record)
        pooled, spatial = build_probes(backbone)
        result["feature_specs"] = [asdict(spec) for spec in probe_specs()]
        result["parity"] = [
            compare_case(pooled, spatial, args.reference_dir, name) for name in cases
        ]
        result["reference_status"] = "passed"
    except Exception as error:
        result["reference_status"] = "failed"
        result["reference_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=12),
        }
    result["onnx"] = []
    for head in ("pooled", "spatial"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--checkpoint",
            str(args.checkpoint),
            "--reference-dir",
            str(args.reference_dir),
            "--export-case",
            args.export_case,
            "--export-only",
            "--head",
            head,
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=args.export_timeout
            )
            try:
                export_result = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                export_result = {
                    "status": "blocked",
                    "head": head,
                    "error": "Exporter exited without a result record.",
                }
            export_result["exit_code"] = completed.returncode
            export_result["stderr_tail"] = completed.stderr[-4000:]
        except subprocess.TimeoutExpired:
            export_result = {
                "status": "blocked",
                "head": head,
                "error": f"Exporter exceeded {args.export_timeout} seconds.",
            }
        result["onnx"].append(export_result)
    result["status"] = (
        "passed"
        if result["reference_status"] == "passed"
        and all(row["status"] == "passed" for row in result["onnx"])
        else "blocked"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.portable_output is not None:
        portable = {
            key: result[key]
            for key in (
                "status",
                "reference_status",
                "source_revision",
                "source_weights_sha256",
                "checkpoint_sha256",
                "reference_report_sha256",
                "reference_case_sha256",
                "environment",
                "backend",
                "source_sha256",
                "reference_profile",
                "feature_specs",
                "tolerance",
                "reader_limits",
                "parity",
            )
            if key in result
        }
        portable["onnx"] = [
            {
                key: row[key]
                for key in (
                    "status",
                    "head",
                    "input_shape",
                    "opset",
                    "graph_sha256",
                    "runtime_metric",
                    "error_type",
                    "error",
                )
                if key in row
            }
            for row in result["onnx"]
        ]
        args.portable_output.parent.mkdir(parents=True, exist_ok=True)
        args.portable_output.write_text(
            json.dumps(portable, indent=2, sort_keys=True) + "\n"
        )
    print(args.output)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
