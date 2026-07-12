import importlib.metadata
import tomllib
from pathlib import Path


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

    assert pyproject["requires-python"] == ">=3.12"
    assert package_metadata["Requires-Python"] == pyproject["requires-python"]
    assert _python_versions(pyproject["classifiers"]) == SUPPORTED_PYTHON_VERSIONS
    assert (
        _python_versions(package_metadata.get_all("Classifier"))
        == SUPPORTED_PYTHON_VERSIONS
    )
