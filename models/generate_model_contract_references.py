"""Generate small deterministic Equimo model-family regression references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)
    sys.path.insert(0, str(root / "tests"))

import jax
import jax.random as jr
import numpy as np

from cases.model_cases import MODEL_CASES, extract_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "model_output_references.json"


def generate(source_revision: str) -> dict:
    cases = {}
    for index, case in enumerate(MODEL_CASES):
        invocation = case.build(jr.PRNGKey(100 + index))
        output = np.asarray(
            extract_features(invocation, *invocation.args, key=invocation.key)
        )
        cases[case.registry_name] = {
            "modality": case.modality,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "values": output.tolist(),
        }
    return {
        "format_version": 1,
        "source_revision": source_revision,
        "jax_version": jax.__version__,
        "seed_base": 100,
        "rtol": 1e-5,
        "atol": 1e-6,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        parser.error(f"refusing to overwrite {args.output}; pass --force")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(generate(args.source_revision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
