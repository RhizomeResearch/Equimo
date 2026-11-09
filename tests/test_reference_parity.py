"""Manifest-driven parity against trusted upstream numerical references."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from cases.pretrained_cases import PRETRAINED_PATH_CASES
from cases.reference_cases import REFERENCE_CASES, local_archive
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256


DATA_DIR = Path(__file__).parent / "data"
PROVENANCE = json.loads(
    (DATA_DIR / "reference_provenance.json").read_text(encoding="utf-8")
)


def _require_checkpoint(identifier):
    archive = local_archive(identifier)
    if archive.is_file():
        return
    message = f"Converted checkpoint {identifier!r} is not cached locally."
    # An archive without a pinned digest can only come from a local conversion.
    if (
        os.environ.get("EQUIMO_REQUIRE_REFERENCE_CACHE") == "1"
        and identifier in PRETRAINED_ARCHIVE_SHA256
    ):
        pytest.fail(message)
    pytest.skip(message)


def test_reference_matrix_tracks_every_available_pretrained_path_fixture():
    expected = {
        case.reference_fixture
        for case in PRETRAINED_PATH_CASES
        if case.reference_fixture is not None
    }
    covered = {
        case.fixture
        for case in REFERENCE_CASES
        if case.identifier in PRETRAINED_ARCHIVE_SHA256
    }

    assert expected == covered
    assert {case.family for case in REFERENCE_CASES} == {
        "ast",
        "convnext",
        "convnextv2",
        "dinov2",
        "dinov3",
        "eupe",
        "sam31",
        "siglip2",
        "t0",
        "tabpfn",
    }
    assert {
        (case.family, case.conversion_path)
        for case in PRETRAINED_PATH_CASES
        if case.reference_fixture is None
    } == {
        ("eupe", "convnext"),
        ("lingbot", "author-checkpoint"),
        ("tabpfn", "regressor"),
        ("tips", "vision"),
        ("tips", "text"),
    }


@pytest.mark.reference_parity
@pytest.mark.parametrize(
    "case",
    REFERENCE_CASES,
    ids=lambda case: f"{case.family}-{case.identifier}",
)
def test_pretrained_model_matches_trusted_upstream_reference(case):
    _require_checkpoint(case.identifier)
    provenance = PROVENANCE["fixtures"][case.fixture]
    # The field appears once the archive's digest is pinned.
    if case.identifier in PRETRAINED_ARCHIVE_SHA256:
        assert provenance["identifier"] == case.identifier
    comparison = provenance.get(
        "comparison",
        {"metric": "mean_absolute_error", "atol": provenance["tolerance"]},
    )
    with np.load(DATA_DIR / case.fixture, allow_pickle=False) as fixture:
        reference = {name: fixture[name] for name in fixture.files}
    actual = case.evaluate(reference)

    assert set(actual) <= set(reference)
    for name, value in actual.items():
        expected = reference[name]
        assert value.shape == expected.shape
        assert value.dtype == expected.dtype
        if comparison["metric"] == "mean_absolute_error":
            error = float(np.mean(np.abs(value - expected)))
            assert error < comparison["atol"], (
                f"{case.identifier}:{name} mean absolute error {error:.3g} "
                f"exceeds {comparison['atol']:.3g}"
            )
        elif comparison["metric"] == "allclose":
            np.testing.assert_allclose(
                value,
                expected,
                rtol=comparison["rtol"],
                atol=comparison["atol"],
            )
        else:
            raise AssertionError(f"Unknown comparison metric {comparison['metric']!r}")
