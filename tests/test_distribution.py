import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
DIST_DIR = ROOT / "dist"


@pytest.fixture(scope="module")
def built_distributions():
    if not DIST_DIR.exists():
        pytest.skip("distribution artifacts have not been built")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    distribution_name = project["name"].lower()
    version = project["version"]
    wheel = DIST_DIR / f"{distribution_name}-{version}-py3-none-any.whl"
    sdist = DIST_DIR / f"{distribution_name}-{version}.tar.gz"

    artifact_names = {
        path.name
        for path in DIST_DIR.iterdir()
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    }
    assert artifact_names == {wheel.name, sdist.name}
    return project, wheel, sdist


def test_wheel_metadata_matches_project(built_distributions):
    project, wheel, _ = built_distributions

    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            path for path in archive.namelist() if path.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_path))

    assert metadata["Name"] == project["name"]
    assert metadata["Version"] == project["version"]
    assert metadata["Requires-Python"] == project["requires-python"]
    assert metadata["License"] == project["license"]["text"]
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert (
        metadata.get_payload(decode=True).decode()
        == (ROOT / project["readme"]).read_text()
    )


def test_built_distributions_contain_all_package_modules(built_distributions):
    project, wheel, sdist = built_distributions
    source_modules = {
        path.relative_to(ROOT / "src").as_posix()
        for path in (ROOT / "src" / "equimo").rglob("*.py")
    }

    with zipfile.ZipFile(wheel) as archive:
        wheel_modules = {path for path in archive.namelist() if path.endswith(".py")}

    sdist_root = f"{project['name'].lower()}-{project['version']}"
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_contents = set(archive.getnames())
        sdist_modules = {
            path.removeprefix(f"{sdist_root}/src/")
            for path in sdist_contents
            if path.startswith(f"{sdist_root}/src/equimo/") and path.endswith(".py")
        }

    assert wheel_modules == source_modules
    assert sdist_modules == source_modules
    assert {
        f"{sdist_root}/PKG-INFO",
        f"{sdist_root}/README.md",
        f"{sdist_root}/pyproject.toml",
    } <= sdist_contents


def test_wheel_imports_from_clean_environment(built_distributions, tmp_path):
    project, wheel, _ = built_distributions
    venv = tmp_path / "wheel-venv"
    python = venv / "bin" / "python"

    subprocess.run(
        ["uv", "venv", str(venv), "--python", sys.executable],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        cwd=tmp_path,
    )

    smoke_test = """
import importlib
import importlib.metadata
from pathlib import Path
import sys

import equimo

expected_version = sys.argv[1]
assert equimo.__version__ == importlib.metadata.version("Equimo") == expected_version

environment = Path(sys.prefix).resolve()
for name in ("equimo", "equimo.vision", "equimo.language", "equimo.audio", "equimo.tabular"):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(environment)
"""
    subprocess.run(
        [str(python), "-I", "-c", smoke_test, project["version"]],
        check=True,
        cwd=tmp_path,
    )
