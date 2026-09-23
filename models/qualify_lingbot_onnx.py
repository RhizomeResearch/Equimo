"""Attempt Small backbone feature and dense-probe ONNX exports."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import equimo.finetune as eqft
from equimo.serialization import load_weights
from equimo.vision.models import lingbot_vits16

ATOL = 1.5e-4
RTOL = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        import onnx
        import onnxruntime as ort
        from jax2onnx import to_onnx
    except ImportError as error:
        raise ImportError(
            "ONNX qualification requires the dev dependencies."
        ) from error

    jax.config.update("jax_default_matmul_precision", "highest")
    identifier = "lingbot_vits16"
    record = json.loads(
        (args.archive_dir / f"{identifier}.conversion.json").read_text()
    )
    backbone = load_weights(
        lingbot_vits16(),
        path=args.archive_dir / f"{identifier}.tar.lz4",
        expected_sha256=record["converted_archive_sha256"],
        expected_model_config=record["model_config"],
    )
    probe = eqft.vision.make_dense_probe(
        backbone,
        in_features=384,
        out_features=3,
        key=jr.PRNGKey(51),
        feature_spec=eqft.FeatureSpec(
            "forward_features",
            "BNC",
            "patches",
            None,
            return_metadata=True,
        ),
    )
    input_image = jnp.ones((3, 32, 48), dtype=jnp.float32)
    cases = {
        "patch_features": lambda image: backbone.forward_features(
            image, key=jr.PRNGKey(0), inference=True
        )["x_norm_patchtokens"],
        "dense_probe": lambda image: probe(image, key=jr.PRNGKey(0), inference=True),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, inference in cases.items():
        try:
            graph = to_onnx(
                inference,
                inputs=[input_image],
                opset=18,
                input_names=["image"],
                output_names=["output"],
            )
            onnx.checker.check_model(graph)
            graph_path = args.output_dir / f"{identifier}_{name}.onnx"
            onnx.save_model(graph, graph_path)
            with graph_path.open("rb") as stream:
                graph_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            session = ort.InferenceSession(
                graph.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            (actual,) = session.run(["output"], {"image": np.asarray(input_image)})
            expected = np.asarray(inference(input_image))
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(
                    f"ONNX output has {actual.shape}/{actual.dtype}; "
                    f"expected {expected.shape}/{expected.dtype}."
                )
            error = np.abs(actual - expected)
            violations = int(np.count_nonzero(error > ATOL + RTOL * np.abs(expected)))
            results[name] = {
                "status": "passed" if violations == 0 else "numerical_mismatch",
                "graph": graph_path.name,
                "graph_sha256": graph_digest,
                "shape": list(actual.shape),
                "max_absolute_error": float(error.max(initial=0)),
                "violations": violations,
            }
        except Exception as error:
            results[name] = {
                "status": "blocked",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            gc.collect()

    report = {
        "variant": identifier,
        "converted_archive_sha256": record["converted_archive_sha256"],
        "input_shape": [3, 32, 48],
        "opset": 18,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "jax_version": jax.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "results": results,
    }
    path = args.output_dir / f"{identifier}_onnx_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))
    if any(result["status"] != "passed" for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
