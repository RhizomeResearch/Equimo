import importlib.metadata
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet


PYTHON_CLASSIFIER_PREFIX = "Programming Language :: Python :: "
SUPPORTED_PYTHON_VERSIONS = {"3.12", "3.13", "3.14"}


def _python_versions(classifiers):
    return {
        classifier.removeprefix(PYTHON_CLASSIFIER_PREFIX)
        for classifier in classifiers
        if classifier.startswith(PYTHON_CLASSIFIER_PREFIX)
    }


def test_python_support_metadata_matches_project_configuration():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )["project"]
    package_metadata = importlib.metadata.metadata("equimo")

    assert pyproject["requires-python"] == ">=3.12,<3.15"
    assert SpecifierSet(package_metadata["Requires-Python"]) == SpecifierSet(
        pyproject["requires-python"]
    )
    assert _python_versions(pyproject["classifiers"]) == SUPPORTED_PYTHON_VERSIONS
    assert (
        _python_versions(package_metadata.get_all("Classifier"))
        == SUPPORTED_PYTHON_VERSIONS
    )


def test_stable_release_metadata_and_documents_are_present():
    root = Path(__file__).parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    package_metadata = importlib.metadata.metadata("equimo")

    assert pyproject["version"] == "2.0.0"
    assert package_metadata["Version"] == "2.0.0"
    assert "Development Status :: 5 - Production/Stable" in pyproject["classifiers"]
    assert "Development Status :: 5 - Production/Stable" in package_metadata.get_all(
        "Classifier"
    )
    for relative_path in (
        "CHANGELOG.md",
        "docs/migration-v2.md",
        "docs/stability.md",
        "LICENSE.md",
    ):
        assert (root / relative_path).is_file()

    readme = (root / "README.md").read_text()
    assert "[v2 migration guide](docs/migration-v2.md)" in readme
    assert "[stability policy](docs/stability.md)" in readme
    assert "[LICENSE.md](LICENSE.md)" in readme
