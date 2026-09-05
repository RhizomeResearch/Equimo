import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from equimo.core.factory import build_model_variant
from equimo.core.layers.activation import get_act


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


@pytest.mark.parametrize(
    ("factory_name", "identifier"),
    [
        ("convnext_a", "convnext_atto"),
        ("convnextv2_a", "convnextv2_atto"),
        ("convnext_z_ols", "convnext_zepto_rms_ols"),
    ],
)
def test_convnext_pretrained_factories_preserve_exact_gelu(
    monkeypatch, factory_name, identifier
):
    import equimo.vision.models as models

    loaded = {}

    def load_weights(model, **kwargs):
        loaded.update(kwargs)
        return model

    monkeypatch.setattr("equimo.serialization.load_weights", load_weights)
    factory = getattr(models, factory_name)
    dimensions = {"depths": [1, 1, 1, 1], "dims": [8, 16, 32, 64]}
    model = factory(pretrained=True, inference_mode=False, **dimensions)
    arr = jnp.array([-2.0, -1.0, 1.0, 2.0])
    expected = get_act("exactgelu")(arr)

    assert loaded == {"identifier": identifier, "inference_mode": False}
    np.testing.assert_array_equal(model.blocks[0].blocks[0].act(arr), expected)
    if identifier.endswith("_ols"):
        np.testing.assert_array_equal(model.blocks[0].downsample.act(arr), expected)

    # Random initialization retains the existing approximate GELU default.
    untrained = factory(pretrained=False, **dimensions)
    np.testing.assert_array_equal(
        untrained.blocks[0].blocks[0].act(arr), get_act("gelu")(arr)
    )


def test_convnext_unpublished_size_rejects_pretrained():
    from equimo.vision.models import convnext_xxlarge

    with pytest.raises(ValueError, match="No pretrained weights are available"):
        convnext_xxlarge(pretrained=True)


@pytest.mark.parametrize(
    ("identifier", "num_classes"),
    [
        ("convnextv2_atto_fcmae", 0),
        ("convnext_nano_in12k", 11821),
        ("convnext_tiny_fb_in22k", 21841),
        ("convnextv2_nano_fcmae_ft_in22k_in1k_384", 1000),
    ],
)
def test_convnext_tagged_factories_select_checkpoint_heads(
    monkeypatch, identifier, num_classes
):
    import equimo.vision.models as models

    def load_weights(model, **kwargs):
        assert kwargs["identifier"] == identifier
        return model

    monkeypatch.setattr("equimo.serialization.load_weights", load_weights)
    model = getattr(models, identifier)(
        pretrained=True, depths=[1, 1, 1, 1], dims=[8, 16, 32, 64]
    )
    if num_classes == 0:
        assert isinstance(model.head, eqx.nn.Identity)
    else:
        assert model.head.weight.shape == (num_classes, 64)
    arr = jnp.array([-2.0, 2.0])
    np.testing.assert_array_equal(
        model.blocks[0].blocks[0].act(arr), get_act("exactgelu")(arr)
    )
