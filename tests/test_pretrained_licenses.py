"""Offline contracts for pretrained-weight license documentation."""

import datetime as dt
import hashlib
from pathlib import Path
import posixpath
import re
import tomllib

from cases.pretrained_cases import PRETRAINED_FAMILY_PREFIXES
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256
from equimo.utils import PCAVisualizer, make_divisible, normalize


ROOT = Path(__file__).parents[1]
INDEX = ROOT / "LICENSES" / "pretrained" / "README.md"
APACHE_LICENSE = ROOT / "LICENSES" / "Apache-2.0.txt"
HUB_CARD = ROOT / "models" / "huggingface" / "README.md"
HUB_NOTICE = ROOT / "models" / "huggingface" / "NOTICE"
CHECKED_DATE = dt.date(2026, 8, 21)
CONVNEXT_CHECKED_DATE = dt.date(2026, 9, 5)
TIPS_SOURCE = (
    "https://github.com/google-deepmind/tips/blob/"
    "72820c1841f973c9543d9c95c5ff2262ec621955/scenic/utils/feature_viz.py"
)
TENSORFLOW_SOURCE = (
    "https://github.com/tensorflow/models/blob/"
    "b41a080d61cd4298fea9894804e9657777e2a451/"
    "official/vision/modeling/layers/nn_layers.py"
)
EXPECTED_SNAPSHOTS = {
    "ast": "ast-BSD-3-Clause.txt",
    "convnext": "convnext-Apache-2.0.txt",
    "convnextv2": "convnextv2-CC-BY-NC-4.0.txt",
    "dinov2": "dinov2-Apache-2.0.txt",
    "dinov3": "dinov3-License.md",
    "eupe": "eupe-FAIR-Noncommercial-Research-License.md",
    "siglip2": "siglip2-Apache-2.0.txt",
    "t0": "t0-alpha-Apache-2.0.txt",
    "tabpfn": "tabpfn-3-License-v1.0.txt",
    "tips": "tips-CC-BY-4.0.txt",
}
EXPECTED_SNAPSHOT_SHA256 = {
    "convnext": "71b111620fa32c17a80ccd6db2da759d31a591e497d8b5f382174f59d1b59d49",
    "convnextv2": "9ab7947f327be897e841ce58b90996c4b814c35127897b383f17616510115b9b",
    "ast": "c6c18fd2915ae9d95fe070b619da57837d867e14bd3c62ed97c1b406362dad41",
    "dinov2": "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
    "dinov3": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",
    "eupe": "363d326fed0d13fd2418039c6f4bb56a809b0e316aaae3d5222d395581060288",
    "siglip2": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "t0": "30607a248f71246288217d02dbe731764d526da9189872a22ca64101f6a8a4c6",
    "tabpfn": "dfba2c0718911db70807b9ecd9efdd404a443d962097c7f5f8cff29eb42ba120",
    "tips": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
}
TABLE_START = "<!-- pretrained-license-table:begin -->"
TABLE_END = "<!-- pretrained-license-table:end -->"
MARKDOWN_LINK = re.compile(r"^\[[^]]+]\(([^)]+)\)$")


def _license_rows(path=INDEX):
    index = path.read_text()
    table = index.split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]
    rows = {}
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|") or cells[0] in {"Family", "---"}:
            continue
        assert len(cells) == 8
        assert cells[0] not in rows
        rows[cells[0]] = cells
    return index, rows


def _link_target(value):
    match = MARKDOWN_LINK.fullmatch(value)
    assert match is not None
    return match.group(1)


def test_license_index_covers_every_pretrained_identifier_exactly_once():
    _, rows = _license_rows()

    assert set(rows) == set(PRETRAINED_FAMILY_PREFIXES) == set(EXPECTED_SNAPSHOTS)
    for family, prefix in PRETRAINED_FAMILY_PREFIXES.items():
        cells = rows[family]
        documented_prefix = cells[1].strip("`")
        documented_count = int(cells[2])
        matching_identifiers = {
            identifier
            for identifier in PRETRAINED_ARCHIVE_SHA256
            if identifier.startswith(documented_prefix)
        }

        assert documented_prefix == prefix
        assert documented_count == len(matching_identifiers)

    assignments = {
        identifier: [
            family
            for family, cells in rows.items()
            if identifier.startswith(cells[1].strip("`"))
        ]
        for identifier in PRETRAINED_ARCHIVE_SHA256
    }
    assert all(len(families) == 1 for families in assignments.values())


def test_license_index_links_dated_snapshots_and_current_upstream_terms():
    _, rows = _license_rows()

    for family, snapshot_name in EXPECTED_SNAPSHOTS.items():
        cells = rows[family]
        snapshot_target = _link_target(cells[4])
        upstream_target = _link_target(cells[6])

        assert snapshot_target == snapshot_name
        snapshot = INDEX.parent / snapshot_target
        assert snapshot.is_file()
        assert (
            hashlib.sha256(snapshot.read_bytes()).hexdigest()
            == (EXPECTED_SNAPSHOT_SHA256[family])
        )
        assert upstream_target.startswith("https://")
        assert dt.date.fromisoformat(cells[7]) == (
            CONVNEXT_CHECKED_DATE
            if family in {"convnext", "convnextv2"}
            else CHECKED_DATE
        )


def test_license_index_warns_that_snapshots_can_become_outdated():
    index, _ = _license_rows()
    normalized = " ".join(index.lower().split())

    assert "may be outdated" in normalized
    assert "latest version" in normalized
    assert "issue or pull request" in normalized


def test_pretrained_license_materials_are_declared_for_distribution():
    license_files = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "license-files"
    ]
    expected = {
        "LICENSES/Apache-2.0.txt",
        "LICENSES/pretrained/README.md",
        *(f"LICENSES/pretrained/{name}" for name in EXPECTED_SNAPSHOTS.values()),
    }

    assert expected <= set(license_files)


def test_third_party_utilities_expose_apache_provenance():
    expected = {
        normalize: ("Copyright 2025 Google LLC", TIPS_SOURCE),
        PCAVisualizer: ("Copyright 2025 Google LLC", TIPS_SOURCE),
        make_divisible: ("Copyright 2025 The TensorFlow Authors", TENSORFLOW_SOURCE),
    }

    for utility, (copyright_notice, source) in expected.items():
        documentation = utility.__doc__ or ""
        assert copyright_notice in documentation
        assert f"Source: {source}" in documentation
        assert "License: Apache-2.0" in documentation
        assert "Modified by Equimo contributors" in documentation


def test_notice_attributes_third_party_utility_sources():
    notice = (ROOT / "NOTICE").read_text()

    assert "src/equimo/utils.py" in notice
    assert "normalize and PCAVisualizer" in notice
    assert "Copyright 2025 Google LLC" in notice
    assert TIPS_SOURCE in notice
    assert "make_divisible" in notice
    assert "Copyright 2025 The TensorFlow Authors" in notice
    assert TENSORFLOW_SOURCE in notice
    assert "LICENSES/Apache-2.0.txt" in notice


def test_hub_model_card_covers_registry_and_tips_tokenizer():
    card, rows = _license_rows(HUB_CARD)
    normalized = " ".join(card.lower().split())

    assert card.startswith("---\nlicense: other\n---\n")
    assert set(rows) == set(PRETRAINED_FAMILY_PREFIXES)
    for family, prefix in PRETRAINED_FAMILY_PREFIXES.items():
        cells = rows[family]
        assert cells[1].strip("`") == prefix
        assert int(cells[2]) == sum(
            identifier.startswith(prefix) for identifier in PRETRAINED_ARCHIVE_SHA256
        )
        assert _link_target(cells[4]) == (
            f"LICENSES/pretrained/{EXPECTED_SNAPSHOTS[family]}"
        )
        assert dt.date.fromisoformat(cells[7]) == (
            CONVNEXT_CHECKED_DATE
            if family in {"convnext", "convnextv2"}
            else CHECKED_DATE
        )

    assert "`models/tokenizers/sentencepiece_tips.model`" in card
    assert "may be outdated" in normalized
    assert "latest version" in normalized
    assert "issue or pull request" in normalized


def test_hub_notice_carries_required_model_attributions():
    notice = HUB_NOTICE.read_text()

    assert "TABPFN-3 Model is licensed by Prior Labs GmbH" in notice
    assert "Copyright © Prior Labs GmbH 2026" in notice
    assert "Copyright 2025 DeepMind Technologies Limited" in notice
    assert "models/tokenizers/sentencepiece_tips.model" in notice
    assert "No endorsement" in notice
    assert APACHE_LICENSE.is_file()


def test_hub_markdown_relative_links_target_published_files():
    published_paths = {
        "README.md",
        "NOTICE",
        "convnext-conversion.json",
        "LICENSES/pretrained/README.md",
        *(f"LICENSES/pretrained/{name}" for name in EXPECTED_SNAPSHOTS.values()),
    }

    for markdown, remote_path in (
        (HUB_CARD, "README.md"),
        (INDEX, "LICENSES/pretrained/README.md"),
    ):
        remote_directory = posixpath.dirname(remote_path)
        for target in re.findall(r"\[[^]]+]\(([^)]+)\)", markdown.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            resolved = posixpath.normpath(
                posixpath.join(remote_directory, target.split("#", 1)[0])
            )
            assert resolved in published_paths, f"Missing Hub target: {resolved}"
