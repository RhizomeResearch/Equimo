import importlib
import os

import equinox as eqx
import jax
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


def test_convert_non_strict_keeps_missing_destination_parameters():
    torch, _ = _require_torch_extra()
    torch_linear = torch.nn.Linear(3, 2, bias=False)
    equinox_linear = eqx.nn.Linear(3, 2, key=jr.PRNGKey(0))

    converted = convert_params_from_torch(
        equinox_linear,
        replace_cfg={},
        expand_cfg={},
        squeeze_cfg={},
        torch_whitelist=[],
        jax_whitelist=[],
        strict=False,
        source="custom",
        torch_model=torch_linear,
    )

    assert jnp.array_equal(converted.bias, equinox_linear.bias)


def test_convert_preserves_bfloat16_destination_dtype():
    torch, _ = _require_torch_extra()
    torch_linear = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
    equinox_linear = eqx.nn.Linear(3, 2, key=jr.PRNGKey(0))
    equinox_linear = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        equinox_linear,
    )

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

    assert converted.weight.dtype == jnp.bfloat16
    assert converted.bias.dtype == jnp.bfloat16
