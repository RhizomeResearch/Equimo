"""Check that published pretrained families have bundled license materials."""

from pathlib import Path
import re
import tomllib

from cases.pretrained_cases import PRETRAINED_FAMILY_PREFIXES
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256


ROOT = Path(__file__).parents[1]
INDEX = ROOT / "LICENSES" / "pretrained" / "README.md"
HUB_CARD = ROOT / "models" / "huggingface" / "README.md"
TABLE_START = "<!-- pretrained-license-table:begin -->"
TABLE_END = "<!-- pretrained-license-table:end -->"
SNAPSHOT_LINK = re.compile(r"\[snapshot\]\(([^)]+)\)")


def _license_rows(path: Path) -> dict[str, list[str]]:
    table = path.read_text().split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]
    rows = {}
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("| ").split("|")]
        if cells[0] == "Family" or re.fullmatch(r"-{3,}", cells[0]):
            continue
        assert len(cells) == 8
        assert cells[0] not in rows
        rows[cells[0]] = cells
    return rows


def test_pretrained_families_have_packaged_license_snapshots():
    index_rows = _license_rows(INDEX)
    hub_rows = _license_rows(HUB_CARD)
    assert set(index_rows) == set(hub_rows) == set(PRETRAINED_FAMILY_PREFIXES)

    declared_files = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["license-files"]
    )
    for family, prefix in PRETRAINED_FAMILY_PREFIXES.items():
        count = sum(name.startswith(prefix) for name in PRETRAINED_ARCHIVE_SHA256)
        assert count > 0
        index_row = index_rows[family]
        hub_row = hub_rows[family]
        for row in (index_row, hub_row):
            assert row[1] == f"`{prefix}`"
            assert int(row[2]) == count

        snapshot = SNAPSHOT_LINK.fullmatch(index_row[4])
        assert snapshot is not None
        relative_path = f"LICENSES/pretrained/{snapshot.group(1)}"
        assert (ROOT / relative_path).is_file()
        assert relative_path in declared_files
        assert hub_row[4] == f"[snapshot]({relative_path})"
