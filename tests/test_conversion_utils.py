"""Contracts for converting model parameters and inference state."""

import sys
import types

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from equimo.conversion.utils import (
    convert_torch_to_equinox,
    stringify_name,
    torch_parameter_name,
)


def test_stringify_name_keeps_every_path_entry():
    path = (
        GetAttrKey("blocks"),
        SequenceKey(2),
        DictKey("q_proj"),
        GetAttrKey("weight"),
    )

    assert stringify_name(path) == "blocks.2.q_proj.weight"


@pytest.mark.parametrize(
    "parts, expected",
    (
        (("norm", "scale"), "norm.weight"),
        (("conv", "kernel"), "conv.weight"),
        (("scale", "linear", "bias"), "weight.linear.bias"),
        (("proj", "scale_x"), "proj.scale_x"),
        (("conv_kernel",), "conv_kernel"),
        (("layer", "rescale"), "layer.rescale"),
    ),
)
def test_torch_parameter_name_renames_only_whole_scale_and_kernel_components(
    parts, expected
):
    path = tuple(GetAttrKey(part) for part in parts)

    assert torch_parameter_name(path, replace_cfg={}) == expected


def test_torch_parameter_name_applies_replacements_in_order():
    path = (GetAttrKey("layers"), SequenceKey(0), GetAttrKey("weight"))

    name = torch_parameter_name(path, replace_cfg={"layers.": "", "0.": "fc."})

    assert name == "fc.weight"


def test_unknown_conversion_source_is_rejected_even_with_a_torch_model(monkeypatch):
    # The source is checked after the optional imports; stub them so the check runs
    # without the torch extra.
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "timm", types.ModuleType("timm"))
    jax_model = eqx.nn.Linear(3, 2, key=jr.PRNGKey(0))

    with pytest.raises(ValueError, match="Unknown conversion source"):
        convert_torch_to_equinox(jax_model, source="onnx", torch_model=object())


@pytest.mark.parametrize("return_torch", (False, True))
def test_conversion_preserves_weights_and_torch_eval_policy(return_torch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    torch_model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.5))
    with torch.no_grad():
        torch_model[0].weight.fill_(0.25)
        torch_model[0].bias.fill_(0.125)
    jax_model = eqx.nn.Sequential(
        [eqx.nn.Linear(3, 2, key=jr.PRNGKey(0)), eqx.nn.Dropout(0.5)]
    )

    result = convert_torch_to_equinox(
        jax_model,
        replace_cfg={"layers.": ""},
        source="custom",
        torch_model=torch_model,
        return_torch=return_torch,
    )

    if return_torch:
        converted, returned_torch = result
        assert returned_torch is torch_model
    else:
        converted = result
    assert torch_model.training is not return_torch
    assert converted.layers[1].inference is True
    assert jax_model.layers[1].inference is False
    assert jnp.array_equal(converted.layers[0].weight, jnp.full((2, 3), 0.25))
    assert jnp.array_equal(converted(jnp.ones((3,))), jnp.full((2,), 0.875))
