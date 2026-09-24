import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.conversion.utils import convert_params_from_torch
from _optional import require_extra


def _require_torch_extra():
    return [require_extra(module, "torch") for module in ("torch", "timm")]


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
    from huggingface_hub import HfApi
    import equimo.vision.models as em

    script = Path(__file__).parents[1] / "models" / "convnext.py"
    spec = importlib.util.spec_from_file_location("convnext_conversion", script)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    monkeypatch.setattr(
        HfApi, "model_info", lambda *a, **kw: SimpleNamespace(sha="a" * 40)
    )

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
    reference.pretrained_cfg["input_size"] = (3, 32, 32)

    def create_model(name, *, pretrained, pretrained_cfg_overlay):
        assert name == entry["timm_tag"]
        assert pretrained is True
        assert pretrained_cfg_overlay["hf_hub_id"].endswith("@" + "a" * 40)
        return reference

    factory = getattr(em, identifier)
    monkeypatch.setattr(timm, "create_model", create_model)
    monkeypatch.setattr(em, identifier, lambda **kw: factory(**dimensions, **kw))

    error = converter.convert_one(identifier, entry, tmp_path, seed=42)

    assert error < 1e-6
    assert (tmp_path / f"{identifier}.tar.lz4").is_file()


@pytest.mark.parametrize("conv_mlp", [False, True], ids=["linear-mlp", "conv-mlp"])
def test_convnext_v2_overlap_conversion_matches_timm(conv_mlp):
    torch, timm = _require_torch_extra()
    from equimo.conversion.utils import convert_torch_to_equinox
    from equimo.vision.models import ConvNeXt

    script = Path(__file__).parents[1] / "models" / "convnext.py"
    spec = importlib.util.spec_from_file_location("convnext_conversion", script)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    dimensions = {"depths": [1, 1, 1, 1], "dims": [8, 16, 32, 64]}

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = timm.create_model(
            "convnextv2_atto",
            pretrained=False,
            num_classes=7,
            stem_type="overlap",
            conv_mlp=conv_mlp,
            **dimensions,
        ).eval()
        with torch.no_grad():
            for stage in reference.stages:
                # GRN starts as identity; nonzero parameters exercise its math.
                stage.blocks[0].mlp.grn.weight.uniform_(-0.3, 0.3)
                stage.blocks[0].mlp.grn.bias.uniform_(-0.1, 0.1)
                stage.blocks[0].mlp.fc1.weight.mul_(10)

    model = ConvNeXt(
        **dimensions,
        num_classes=7,
        use_grn=True,
        layer_scale_init_value=None,
        stem="convnextoverlapstem",
        act_layer="exactgelu",
        key=jr.PRNGKey(42),
    )
    entry = converter.VARIANTS["convnextv2_atto"] | {"is_ols": True}
    replace_cfg, expand_cfg = converter.conversion_config(
        entry,
        dimensions["depths"],
        conv_mlp,
    )
    model = convert_torch_to_equinox(
        model,
        replace_cfg=replace_cfg,
        expand_cfg=expand_cfg,
        strict=True,
        source="custom",
        torch_model=reference,
    )
    # Non-square spatial maps exercise GRN reduction axes at every stage.
    arr = np.random.default_rng(42).standard_normal((3, 64, 96)).astype(np.float32)
    with torch.no_grad():
        value = reference.stem(torch.from_numpy(arr).unsqueeze(0))
        expected_stages = []
        for stage in reference.stages:
            value = stage(value)
            expected_stages.append(value.squeeze(0).numpy())
        expected_output = reference.forward_head(value).squeeze(0).numpy()

    actual_stages = model.intermediate_features(jnp.asarray(arr), inference=True)
    actual_output = eqx.filter_jit(model)(jnp.asarray(arr), inference=True)
    for actual, expected in zip(
        (*actual_stages, actual_output),
        (*expected_stages, expected_output),
        strict=True,
    ):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == np.float32
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
