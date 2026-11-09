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

sys.path.insert(0, str(ROOT / "tests"))

from cases.sam31_video_fixture import IDENTIFIER, download_fixture  # noqa: E402


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    identifiers = sorted(
        {
            identifier
            for provenance in manifest["fixtures"].values()
            if "identifier" in provenance
            for identifier in (
                provenance["identifier"],
                *provenance.get("additional_identifiers", ()),
            )
        }
    )
    for identifier in identifiers:
        path = download(identifier, DEFAULT_REPOSITORY_URL)
        print(f"Verified {identifier}: {path}")
    print(f"Verified {IDENTIFIER}: {download_fixture()}")


if __name__ == "__main__":
    main()
