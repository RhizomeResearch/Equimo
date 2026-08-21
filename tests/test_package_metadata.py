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

    assert pyproject["version"] == "2.1.0"
    assert package_metadata["Version"] == "2.1.0"
    assert package_metadata["License-Expression"] == "MIT AND Apache-2.0"
    assert "Development Status :: 5 - Production/Stable" in pyproject["classifiers"]
    assert "Development Status :: 5 - Production/Stable" in package_metadata.get_all(
        "Classifier"
    )
    for relative_path in (
        "CHANGELOG.md",
        "docs/licensing/t0-source-provenance.md",
        "docs/migration-v2.md",
        "docs/stability.md",
        "docs/timeseries.md",
        "LICENSE.md",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/pretrained/README.md",
        "LICENSES/pretrained/ast-BSD-3-Clause.txt",
        "LICENSES/pretrained/dinov2-Apache-2.0.txt",
        "LICENSES/pretrained/dinov3-License.md",
        "LICENSES/pretrained/eupe-FAIR-Noncommercial-Research-License.md",
        "LICENSES/pretrained/siglip2-Apache-2.0.txt",
        "LICENSES/pretrained/t0-alpha-Apache-2.0.txt",
        "LICENSES/pretrained/tabpfn-3-License-v1.0.txt",
        "LICENSES/pretrained/tips-CC-BY-4.0.txt",
        "LICENSES/tfc-t0-APACHE-2.0.txt",
        "models/huggingface/NOTICE",
        "models/huggingface/README.md",
        "NOTICE",
    ):
        assert (root / relative_path).is_file()

    readme = (root / "README.md").read_text()
    assert "[v2 migration guide](docs/migration-v2.md)" in readme
    assert "[stability policy](docs/stability.md)" in readme
    assert "[time-series guide](docs/timeseries.md)" in readme
    assert "[MIT License](LICENSE.md)" in readme
    assert "[NOTICE](NOTICE)" in readme
    assert "[pretrained-model license index](LICENSES/pretrained/README.md)" in readme
    assert "[T0 Apache-2.0 license](LICENSES/tfc-t0-APACHE-2.0.txt)" in readme
