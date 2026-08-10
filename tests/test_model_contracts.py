"""Cross-family numerical regression contracts for tiny production models."""

import json
from pathlib import Path

from cases.model_cases import MODEL_CASES


REFERENCE_PATH = Path(__file__).parent / "data" / "model_output_references.json"


def _manifest():
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def test_model_output_manifest_covers_every_registered_family_case():
    manifest = _manifest()
    assert manifest["format_version"] == 1
    assert set(manifest["cases"]) == {case.registry_name for case in MODEL_CASES}
    assert {(entry["modality"], name) for name, entry in manifest["cases"].items()} == {
        (case.modality, case.registry_name) for case in MODEL_CASES
    }
