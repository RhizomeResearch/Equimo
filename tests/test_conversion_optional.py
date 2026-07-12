import importlib
import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.conversion.utils import convert_params_from_torch


def _require_torch_extra():
    modules = []
    for module in ("torch", "timm"):
        if os.environ.get("EQUIMO_TEST_OPTIONAL_EXTRA") == "torch":
            modules.append(importlib.import_module(module))
        else:
            modules.append(pytest.importorskip(module))
    return modules


def test_convert_tiny_torch_linear():
    torch, _ = _require_torch_extra()
    torch_linear = torch.nn.Linear(3, 2)
    with torch.no_grad():
        torch_linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]))
        torch_linear.bias.copy_(torch.tensor([0.5, -0.5]))

    equinox_linear = eqx.nn.Linear(3, 2, key=jr.PRNGKey(0))
    converted = convert_params_from_torch(
        equinox_linear,
        replace_cfg={},
        expand_cfg={},
        squeeze_cfg={},
        torch_whitelist=[],
        jax_whitelist=[],
        source="custom",
        torch_model=torch_linear,
    )

    assert np.array_equal(
        np.asarray(converted.weight), np.asarray(torch_linear.weight.detach())
    )
    assert np.array_equal(
        np.asarray(converted.bias), np.asarray(torch_linear.bias.detach())
    )
    assert jnp.allclose(converted(jnp.array([1.0, 1.0, 1.0])), jnp.array([6.5, -6.5]))
