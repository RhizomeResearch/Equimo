import importlib
import importlib.util
import os
from pathlib import Path

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


def test_convnext_zepto_overlap_conversion_matches_timm(tmp_path, monkeypatch):
    torch, timm = _require_torch_extra()
    import equimo.vision.models as em

    script = Path(__file__).parents[1] / "models" / "convnext.py"
    spec = importlib.util.spec_from_file_location("convnext_conversion", script)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    monkeypatch.setattr(converter, "IMG_SIZE", 32)

    identifier = "convnext_zepto_rms_ols"
    entry = converter.VARIANTS[identifier] | {"num_classes": 7}
    dimensions = {"depths": [1, 1, 1, 1], "dims": [8, 16, 32, 64]}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = timm.create_model(
            identifier, pretrained=False, num_classes=7, **dimensions
        ).eval()
        # Exercise the region where exact and approximate GELU diverge.
        with torch.no_grad():
            reference.stem[0].weight.mul_(10)

    def create_model(name, *, pretrained):
        assert name == entry["timm_tag"]
        assert pretrained is True
        return reference

    factory = getattr(em, identifier)
    monkeypatch.setattr(timm, "create_model", create_model)
    monkeypatch.setattr(em, identifier, lambda **kw: factory(**dimensions, **kw))

    error = converter.convert_one(identifier, entry, tmp_path, seed=42)

    assert error < 1e-6
    assert (tmp_path / f"{identifier}.tar.lz4").is_file()
