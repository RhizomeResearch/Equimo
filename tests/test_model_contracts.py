"""Cross-family numerical regression contracts for tiny production models."""

import json
from pathlib import Path

from cases.model_cases import MODEL_CASES


REFERENCE_PATH = Path(__file__).parent / "data" / "model_output_references.json"


def _manifest():
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def test_every_reference_output_carries_signal():
    # Outputs near zero would let the regression check pass whatever the model
    # computes, since it compares them within an absolute tolerance.
    manifest = _manifest()
    floor = 1e3 * manifest["atol"]

    def largest(values):
        if isinstance(values, list):
            return max(largest(value) for value in values)
        return abs(values)

    weak = {
        name: largest(entry["values"])
        for name, entry in manifest["cases"].items()
        if largest(entry["values"]) < floor
    }
    assert not weak, weak


def test_model_output_manifest_covers_every_registered_family_case():
    manifest = _manifest()
    assert manifest["format_version"] == 1
    assert set(manifest["cases"]) == {case.registry_name for case in MODEL_CASES}
    assert {(entry["modality"], name) for name, entry in manifest["cases"].items()} == {
        (case.modality, case.registry_name) for case in MODEL_CASES
    }
