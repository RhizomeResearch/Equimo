import equinox as eqx
import pytest

from equimo.core.factory import build_model_variant


class _ToyModel(eqx.Module):
    width: int = eqx.field(static=True)

    def __init__(self, width: int, *, key):
        del key
        self.width = width


def test_rejects_unsupported_pretrained_variant_before_construction():
    constructed = False

    def model_cls(**kwargs):
        nonlocal constructed
        constructed = True
        return _ToyModel(**kwargs)

    with pytest.raises(ValueError, match="Supported Toy pretrained variants: other"):
        build_model_variant(
            model_cls,
            {"tiny": ({"width": 1}, {})},
            "tiny",
            pretrained=True,
            pretrained_variants=frozenset({"other"}),
            pretrained_label="Toy",
        )

    assert not constructed


def test_rejects_pretrained_overrides_before_construction():
    constructed = False

    def model_cls(**kwargs):
        nonlocal constructed
        constructed = True
        return _ToyModel(**kwargs)

    with pytest.raises(ValueError, match=r"overrides; got: depth, width\."):
        build_model_variant(
            model_cls,
            {"tiny": ({"width": 1}, {})},
            "tiny",
            pretrained=True,
            pretrained_variants=frozenset({"tiny"}),
            pretrained_label="Toy",
            allow_pretrained_overrides=False,
            depth=3,
            width=2,
        )

    assert not constructed


def test_maps_pretrained_identifier_and_allows_overrides_by_default(monkeypatch):
    loaded = {}

    def fake_load_weights(model, *, identifier, inference_mode):
        loaded.update(identifier=identifier, inference_mode=inference_mode)
        return model

    monkeypatch.setattr("equimo.serialization.load_weights", fake_load_weights)
    model = build_model_variant(
        _ToyModel,
        {"default": ({"width": 1}, {})},
        "default",
        pretrained=True,
        inference_mode=False,
        pretrained_identifiers={"default": "published"},
        width=2,
    )

    assert model.width == 2
    assert loaded == {"identifier": "published", "inference_mode": False}
