"""Validate committed numerical references and compare generated candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "tests" / "data"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "reference_provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported reference provenance format")
    return manifest


def validate_fixture(path: Path, provenance: dict, *, verify_hash: bool) -> None:
    for field in ("generator", "upstream", "seed", "input_shape", "tolerance"):
        if field not in provenance:
            raise ValueError(f"{path.name}: missing provenance field {field!r}")
    upstream = provenance["upstream"]
    for field in ("repository", "revision", "checkpoint"):
        if not upstream.get(field):
            raise ValueError(f"{path.name}: missing upstream field {field!r}")

    if verify_hash:
        actual = _sha256(path)
        if actual != provenance.get("sha256"):
            raise ValueError(
                f"{path.name}: SHA-256 is {actual}, expected {provenance.get('sha256')}"
            )

    with np.load(path, allow_pickle=False) as fixture:
        expected_arrays = provenance["arrays"]
        if set(fixture.files) != set(expected_arrays):
            raise ValueError(
                f"{path.name}: keys are {sorted(fixture.files)}, "
                f"expected {sorted(expected_arrays)}"
            )
        for key, expected in expected_arrays.items():
            array = fixture[key]
            if str(array.dtype) != expected["dtype"]:
                raise ValueError(
                    f"{path.name}:{key}: dtype is {array.dtype}, "
                    f"expected {expected['dtype']}"
                )
            if list(array.shape) != expected["shape"]:
                raise ValueError(
                    f"{path.name}:{key}: shape is {list(array.shape)}, "
                    f"expected {expected['shape']}"
                )


def validate(data_dir: Path, manifest_path: Path, candidate_dir: Path | None) -> None:
    manifest = load_manifest(manifest_path)
    fixtures = manifest["fixtures"]
    committed_names = {path.name for path in data_dir.glob("*.npz")}
    if committed_names != set(fixtures):
        raise ValueError(
            "Provenance coverage mismatch: "
            f"fixtures={sorted(committed_names)}, manifest={sorted(fixtures)}"
        )

    for filename, provenance in fixtures.items():
        committed = data_dir / filename
        validate_fixture(committed, provenance, verify_hash=True)
        if candidate_dir is None:
            continue

        candidate = candidate_dir / filename
        if not candidate.is_file():
            raise ValueError(f"Missing candidate fixture: {candidate}")
        validate_fixture(candidate, provenance, verify_hash=False)
        tolerance = float(provenance["tolerance"])
        with (
            np.load(committed, allow_pickle=False) as old,
            np.load(candidate, allow_pickle=False) as new,
        ):
            for key in old.files:
                if not np.issubdtype(old[key].dtype, np.number):
                    continue
                delta = np.abs(old[key].astype(np.float64) - new[key])
                maximum = float(delta.max(initial=0.0))
                mean = float(delta.mean()) if delta.size else 0.0
                print(
                    f"{filename}:{key}: max_abs={maximum:.6g} "
                    f"mean_abs={mean:.6g} tolerance={tolerance:.6g}"
                )
                if mean > tolerance:
                    raise ValueError(
                        f"{filename}:{key}: mean delta {mean:.6g} exceeds "
                        f"{tolerance:.6g}; human parity review required"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Generated directory to compare with committed fixtures.",
    )
    args = parser.parse_args()
    validate(args.data_dir, args.manifest, args.candidate_dir)
    print(f"Validated {len(load_manifest(args.manifest)['fixtures'])} references")


if __name__ == "__main__":
    main()
