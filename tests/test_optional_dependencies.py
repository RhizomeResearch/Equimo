import builtins
from types import SimpleNamespace

import pytest

from equimo.conversion.utils import convert_params_from_torch
from equimo.language.tokenizers import _require_tensorflow_text
from equimo.utils import PCAVisualizer, plot_image_and_feature_map
from equimo.vision.io import load_image


def _make_import_fail(monkeypatch, missing_package, substitutes=None):
    original_import = builtins.__import__
    missing_error = ImportError(f"No module named {missing_package!r}")
    substitutes = substitutes or {}

    def import_with_missing_package(name, *args, **kwargs):
        if name == missing_package or name.startswith(f"{missing_package}."):
            raise missing_error
        if name in substitutes:
            return substitutes[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_package)
    return missing_error


def _assert_recovery_message(exc_info, cause, extra):
    assert exc_info.value.__cause__ is cause
    message = str(exc_info.value)
    assert f"'{extra}' extra" in message
    assert f'pip install "equimo[{extra}]"' in message


def test_pca_visualizer_missing_sklearn_names_extras_extra(monkeypatch):
    cause = _make_import_fail(monkeypatch, "sklearn")

    with pytest.raises(ImportError) as exc_info:
        PCAVisualizer(object())

    _assert_recovery_message(exc_info, cause, "extras")


def test_plot_missing_matplotlib_names_extras_extra(monkeypatch):
    cause = _make_import_fail(monkeypatch, "matplotlib")

    with pytest.raises(ImportError) as exc_info:
        plot_image_and_feature_map(object(), object(), "unused.png")

    _assert_recovery_message(exc_info, cause, "extras")


def test_load_image_missing_pillow_names_extras_extra(monkeypatch):
    cause = _make_import_fail(monkeypatch, "PIL")

    with pytest.raises(ImportError) as exc_info:
        load_image("unused.png")

    _assert_recovery_message(exc_info, cause, "extras")


def test_conversion_missing_torch_names_torch_extra(monkeypatch):
    cause = _make_import_fail(
        monkeypatch, "torch", substitutes={"timm": SimpleNamespace()}
    )

    with pytest.raises(ImportError) as exc_info:
        convert_params_from_torch(None, {}, {}, {}, [], [])

    _assert_recovery_message(exc_info, cause, "torch")


@pytest.mark.parametrize("missing_package", ["tensorflow", "tensorflow_text"])
def test_tokenizer_missing_dependency_names_language_extra(
    monkeypatch, missing_package
):
    substitutes = (
        {"tensorflow": SimpleNamespace()}
        if missing_package == "tensorflow_text"
        else None
    )
    cause = _make_import_fail(monkeypatch, missing_package, substitutes=substitutes)

    with pytest.raises(ImportError) as exc_info:
        _require_tensorflow_text()

    _assert_recovery_message(exc_info, cause, "language")
