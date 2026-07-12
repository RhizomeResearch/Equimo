"""Run and validate the process-isolated CI test shards."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARDS: dict[str, tuple[str, ...]] = {
    "finetune": ("tests/finetune",),
    "models": (
        "tests/test_audio_ast.py",
        "tests/test_models.py",
        "tests/test_tabpfn.py",
    ),
    "core": (
        "tests",
        "--ignore=tests/finetune",
        "--ignore=tests/test_audio_ast.py",
        "--ignore=tests/test_models.py",
        "--ignore=tests/test_tabpfn.py",
    ),
}


def _collect(*pytest_args: str) -> set[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *pytest_args,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"pytest collection failed for: {' '.join(pytest_args)}")
    return {
        line
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def validate() -> int:
    full = _collect("tests")
    collected = {name: _collect(*pytest_args) for name, pytest_args in SHARDS.items()}
    counts = Counter(node_id for node_ids in collected.values() for node_id in node_ids)

    missing = sorted(full - counts.keys())
    unexpected = sorted(counts.keys() - full)
    duplicates = sorted(node_id for node_id, count in counts.items() if count > 1)
    empty = sorted(name for name, node_ids in collected.items() if not node_ids)

    if missing or unexpected or duplicates or empty:
        for label, values in (
            ("missing", missing),
            ("unexpected", unexpected),
            ("duplicated", duplicates),
            ("empty shards", empty),
        ):
            if values:
                print(f"{label}:")
                print("\n".join(f"  {value}" for value in values))
        return 1

    summary = ", ".join(
        f"{name}={len(node_ids)}" for name, node_ids in collected.items()
    )
    print(f"Validated {len(full)} tests across CI shards: {summary}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} [validate|{'|'.join(SHARDS)}] [pytest args]")
        return 2

    command, *pytest_args = sys.argv[1:]
    if command == "validate":
        if pytest_args:
            print("validate does not accept pytest arguments")
            return 2
        return validate()
    if command not in SHARDS:
        print(f"unknown shard: {command}")
        return 2
    return pytest.main([*SHARDS[command], *pytest_args])


if __name__ == "__main__":
    raise SystemExit(main())
