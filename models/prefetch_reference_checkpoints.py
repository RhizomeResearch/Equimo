"""Prefetch and digest-verify checkpoints required by reference parity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)

from equimo.serialization import DEFAULT_REPOSITORY_URL, download


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "data" / "reference_provenance.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    identifiers = sorted(
        {
            provenance["identifier"]
            for provenance in manifest["fixtures"].values()
            if "identifier" in provenance
        }
    )
    for identifier in identifiers:
        path = download(identifier, DEFAULT_REPOSITORY_URL)
        print(f"Verified {identifier}: {path}")


if __name__ == "__main__":
    main()
