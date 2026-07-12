from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import equimo.finetune as ft


ROOT = Path(__file__).parents[2]
REFERENCE_DIR = ROOT / "docs" / "finetuning" / "api"
GENERATOR = ROOT / "docs" / "finetuning" / "generate_api_reference.py"
PUBLIC_MARKER = re.compile(r"<!-- equimo\.finetune:([A-Za-z_][A-Za-z0-9_]*) -->")


def _reference_text() -> str:
    return "\n".join(
        path.read_text()
        for path in sorted(REFERENCE_DIR.glob("*.md"))
        if path.name != "index.md"
    )


def test_reference_inventory_matches_public_exports_exactly_once():
    documented = PUBLIC_MARKER.findall(_reference_text())

    assert Counter(documented) == Counter(ft.__all__)
    assert all(hasattr(ft, name) for name in ft.__all__)


def test_reference_renders_representative_live_declarations():
    text = _reference_text()

    assert "class equimo.finetune.TrainableSpec(mode:" in text
    assert "class equimo.finetune.LinearHead(in_features:" in text
    assert "equimo.finetune.prepare_finetune(model:" in text
    assert "type equimo.finetune.LeafPredicate = Callable[" in text
    assert "equimo.finetune.CANONICAL_TAGS = (" in text
    assert not re.search(r" at 0x[0-9a-fA-F]+", text)


def test_reference_is_current_and_links_are_valid():
    result = subprocess.run(
        [sys.executable, GENERATOR, "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
