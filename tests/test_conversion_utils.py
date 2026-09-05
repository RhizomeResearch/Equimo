"""Contracts for converting model parameters and inference state."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.conversion.utils import convert_torch_to_equinox


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
