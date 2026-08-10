import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "tests" / "data"


def test_reference_schema_and_provenance_coverage():
    result = subprocess.run(
        [sys.executable, ROOT / "models" / "validate_references.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (DATA_DIR / "reference_provenance.json").read_text(encoding="utf-8")
    )
    assert f"Validated {len(manifest['fixtures'])} references" in result.stdout
