"""Import gate shared by tests of optional Equimo extras."""

import importlib
import os
from types import ModuleType

import pytest


def require_extra(module: str, extra: str) -> ModuleType:
    """Import *module*, failing in a CI job for *extra* and skipping elsewhere."""
    if os.environ.get("EQUIMO_TEST_OPTIONAL_EXTRA") == extra:
        return importlib.import_module(module)
    return pytest.importorskip(module)
