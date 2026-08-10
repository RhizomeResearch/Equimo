"""Completeness contracts for pretrained families and conversion paths."""

import inspect

import pytest

from cases.pretrained_cases import (
    PRETRAINED_FAMILY_PREFIXES,
    PRETRAINED_PATH_CASES,
)
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256


def test_every_trusted_archive_belongs_to_exactly_one_pretrained_family():
    assignments = {
        identifier: [
            family
            for family, prefix in PRETRAINED_FAMILY_PREFIXES.items()
            if identifier.startswith(prefix)
        ]
        for identifier in PRETRAINED_ARCHIVE_SHA256
    }

    assert all(len(families) == 1 for families in assignments.values())
    assert {families[0] for families in assignments.values()} == set(
        PRETRAINED_FAMILY_PREFIXES
    )


def test_every_pretrained_family_and_converter_path_has_a_representative_case():
    assert {case.family for case in PRETRAINED_PATH_CASES} == set(
        PRETRAINED_FAMILY_PREFIXES
    )
    assert {(case.family, case.conversion_path) for case in PRETRAINED_PATH_CASES} == {
        ("ast", "huggingface-spectrogram"),
        ("dinov2", "timm-vit"),
        ("dinov3", "huggingface-vit"),
        ("eupe", "vit"),
        ("eupe", "convnext"),
        ("siglip2", "huggingface-vit"),
        ("t0", "tfc-t0"),
        ("tabpfn", "classifier"),
        ("tabpfn", "regressor"),
        ("tips", "vision"),
        ("tips", "text"),
    }


@pytest.mark.parametrize(
    "case",
    PRETRAINED_PATH_CASES,
    ids=lambda case: f"{case.family}-{case.conversion_path}",
)
def test_pretrained_path_uses_a_trusted_archive_and_public_factory_when_available(case):
    digest = PRETRAINED_ARCHIVE_SHA256[case.identifier]
    assert len(digest) == 64
    assert int(digest, 16) >= 0

    if case.factory is None:
        assert case.identifier == "tips_vits14_hr_text"
    else:
        assert callable(case.factory)
        parameters = inspect.signature(case.factory).parameters
        assert "pretrained" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
