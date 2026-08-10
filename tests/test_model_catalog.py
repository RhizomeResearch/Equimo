import json
from dataclasses import replace
import importlib
from pathlib import Path
import re
import sys

import pytest

from equimo.catalog import (
    ModelVariant,
    _load_catalog,
    _resolve_model,
    create_model,
    list_models,
    model_info,
)

KEY_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9][a-z0-9_-]*$")
FIELD_NAMES = ("inputs", "pretrained", "provenance", "notes")
STATUSES = frozenset(("complete", "experimental", "unavailable"))


def validate_descriptor(descriptor: ModelVariant) -> None:
    """Enforce the authoring contract for in-repo catalog descriptors."""

    assert KEY_RE.fullmatch(descriptor.key), descriptor.key
    assert descriptor.key == f"{descriptor.modality}/{descriptor.variant}"
    for name in ("modality", "family", "variant", "model_registry_key"):
        value = getattr(descriptor, name)
        assert value and value == value.lower(), (descriptor.key, name)
    assert descriptor.constructor.endswith(f".{descriptor.variant}")
    assert descriptor.inputs, f"{descriptor.key} has no input contract"
    input_names = set()
    for item in descriptor.inputs:
        assert item.name and item.dtype and item.description
        assert len(item.shape) == len(item.axes), (descriptor.key, item.name)
        assert item.name not in input_names, (descriptor.key, item.name)
        input_names.add(item.name)
    assert descriptor.pretrained.available == (
        descriptor.pretrained.identifier is not None
    )
    statuses = dict(descriptor.field_status)
    assert tuple(statuses) == FIELD_NAMES
    assert len(statuses) == len(descriptor.field_status)
    assert set(statuses.values()) <= STATUSES
    if statuses["provenance"] == "complete":
        assert descriptor.provenance.conversion is not None
        assert descriptor.provenance.reference is not None
    if statuses["notes"] == "complete":
        assert descriptor.notes


ROOT = Path(__file__).parents[1]
EXPECTED_KEYS = (
    "audio/ast_base_patch16_audioset_10_10_0_4593",
    "tabular/tabpfn_v3_classifier_default",
    "timeseries/t0_alpha",
    "vision/dinov2_vits14_reg",
)


def test_catalog_is_deterministic_and_filterable():
    assert tuple(item.key for item in list_models()) == EXPECTED_KEYS
    assert tuple(item.key for item in list_models(pretrained=True)) == EXPECTED_KEYS
    assert list_models(pretrained=False) == ()
    assert tuple(item.key for item in list_models(modality="VISION")) == (
        "vision/dinov2_vits14_reg",
    )
    assert tuple(item.key for item in list_models(family="ast")) == (
        "audio/ast_base_patch16_audioset_10_10_0_4593",
    )


def test_representative_descriptors_are_complete_and_serializable():
    expected_registry_keys = {
        "audio": "ast",
        "tabular": "tabpfn",
        "timeseries": "t0",
        "vision": "vit",
    }
    for descriptor in list_models():
        assert (
            descriptor.model_registry_key == expected_registry_keys[descriptor.modality]
        )
        assert dict(descriptor.field_status) == {
            "inputs": "complete",
            "pretrained": "complete",
            "provenance": "complete",
            "notes": "complete",
        }
        assert descriptor.pretrained.available
        assert descriptor.pretrained.identifier == descriptor.variant
        assert descriptor.inputs
        serialized = json.dumps(descriptor.to_dict())
        assert "callable" not in serialized
        assert "Array" not in serialized


def test_catalog_derives_input_shape_from_modality_registry(monkeypatch):
    vit = importlib.import_module("equimo.vision.models.vit")
    variant = "dinov2_vits14_reg"
    base_cfg, variant_cfg = vit._VIT_REGISTRY[variant]
    monkeypatch.setitem(
        vit._VIT_REGISTRY,
        variant,
        (base_cfg, variant_cfg | {"img_size": 280}),
    )

    assert model_info("vision/dinov2_vits14_reg").inputs[0].shape == (3, 280, 280)


def test_discovery_does_not_instantiate_models(monkeypatch):
    descriptors = list_models()

    def fail_if_called(**kwargs):
        raise AssertionError(f"model factory was called with {kwargs}")

    for descriptor in descriptors:
        module_name, _, factory_name = descriptor.constructor.rpartition(".")
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, factory_name, fail_if_called)

    assert tuple(item.key for item in list_models()) == EXPECTED_KEYS


@pytest.mark.parametrize("descriptor", list_models())
def test_create_model_delegates_to_existing_factory(monkeypatch, descriptor):
    module_name, _, factory_name = descriptor.constructor.rpartition(".")
    module = importlib.import_module(module_name)
    sentinel = object()
    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, factory_name, factory)

    assert create_model(descriptor.key, marker=descriptor.modality) is sentinel
    assert received == {"marker": descriptor.modality}


def test_bare_variant_resolution_and_actionable_unknown_error():
    assert model_info("DINOV2_VITS14_REG").key == "vision/dinov2_vits14_reg"
    with pytest.raises(ValueError, match="Did you mean: vision/dinov2_vits14_reg"):
        model_info("dinov2_vits14_regs")


def test_ambiguous_bare_variant_requires_full_key():
    descriptor = model_info("dinov2_vits14_reg")
    collision = replace(
        descriptor,
        key=f"audio/{descriptor.variant}",
        modality="audio",
        model_registry_key="ast",
    )

    with pytest.raises(ValueError, match="Ambiguous model variant"):
        _resolve_model(descriptor.variant, (collision, descriptor))


def test_all_descriptors_satisfy_authoring_contract():
    for descriptor in _load_catalog():
        validate_descriptor(descriptor)


def test_incomplete_descriptor_fails_authoring_contract():
    descriptor = model_info("dinov2_vits14_reg")
    with pytest.raises(AssertionError, match="has no input contract"):
        validate_descriptor(replace(descriptor, inputs=()))


def test_query_does_not_import_conversion_only_dependencies():
    optional_roots = {"t0", "tabpfn", "timm", "torch", "transformers"}
    before = set(sys.modules)

    list_models()

    newly_imported_roots = {
        name.partition(".")[0] for name in set(sys.modules) - before
    }
    assert optional_roots.isdisjoint(newly_imported_roots)
    assert "equimo.conversion" not in set(sys.modules) - before


def test_readme_catalog_block_matches_covered_pretrained_identifiers():
    readme = (ROOT / "README.md").read_text()
    match = re.search(
        r"<!-- model-catalog:begin -->(.*?)<!-- model-catalog:end -->",
        readme,
        flags=re.DOTALL,
    )
    assert match is not None
    documented = tuple(re.findall(r"^- `([^`]+)`$", match.group(1), re.MULTILINE))
    expected = tuple(
        descriptor.pretrained.identifier
        for descriptor in list_models(pretrained=True)
        if descriptor.pretrained.identifier is not None
    )
    assert documented == expected
