"""Model surgery preserves nested PyTree paths and excluded subtrees."""

import equinox as eqx
import pytest

import equimo.finetune as eqft
from equimo.core.layers.dropout import DropPath, DropPathAdd


@pytest.mark.parametrize(
    ("layer_type", "setter"),
    (
        (eqx.nn.Dropout, eqft.set_dropout_rate),
        (DropPath, eqft.set_stochastic_depth_rate),
        (DropPathAdd, eqft.set_stochastic_depth_rate),
    ),
)
def test_probability_surgery_preserves_excluded_nested_subtree(layer_type, setter):
    model = {
        "encoder": (eqx.nn.Sequential([layer_type(0.2)]),),
        "head": {"dropout": layer_type(0.5)},
    }

    updated = setter(model, 0.4, exclude=("head",))

    assert updated["encoder"][0].layers[0].p == 0.4
    assert updated["head"]["dropout"].p == 0.5
    assert model["encoder"][0].layers[0].p == 0.2
    assert isinstance(updated["encoder"], tuple)
    assert isinstance(updated["encoder"][0].layers[0], layer_type)
