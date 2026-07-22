import importlib


def test_audio_scaffold_imports_without_optional_dependencies():
    importlib.import_module("equimo.audio")
    importlib.import_module("equimo.audio.models")
    importlib.import_module("equimo.audio.layers")
    importlib.import_module("equimo.audio.io")


def test_tabular_scaffold_imports_without_optional_dependencies():
    importlib.import_module("equimo.tabular")
    importlib.import_module("equimo.tabular.models")
    importlib.import_module("equimo.tabular.layers")


def test_vision_model_exports_are_public_and_deterministic():
    models = importlib.import_module("equimo.vision.models")

    assert {
        "get_model_cls",
        "register_model",
        "VisionTransformer",
        "vit_tiny_patch16_224",
        "AttNet",
    } <= set(models.__all__)
    assert {
        "Path",
        "pkgutil",
        "importlib",
        "_pkg_path",
        "_mod_info",
        "_mod",
        "_exports",
    }.isdisjoint(models.__all__)
    assert len(models.__all__) == len(set(models.__all__))


def test_vision_rope_classes_are_exported_from_layers():
    layers = importlib.import_module("equimo.vision.layers")

    assert layers.VisionRoPE is layers.get_posemb("visionrope")
    assert layers.CompositeVisionRoPE is layers.get_posemb("compositevisionrope")


def test_old_top_level_packages_are_not_importable():
    for module_name in (
        "equimo.models",
        "equimo.layers",
        "equimo.implicit",
        "equimo.experimental",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} should not be importable")
