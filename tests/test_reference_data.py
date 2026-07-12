import subprocess
import sys
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
    assert "Validated 7 references" in result.stdout
