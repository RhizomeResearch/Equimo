import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.core.layers import Attention, BlockChunk, Mlp, SwiGluFused
from equimo.registry import get_model_cls
from equimo.time_series.models import T0, load_t0_weights, t0, t0_alpha
from equimo.time_series.models.t0 import _T0_REGISTRY


KEY = jr.PRNGKey(0)


def _tiny(**overrides):
    cfg = dict(
        embed_dim=16,
        num_layers=3,
        num_heads=2,
        mlp_hidden_dim=32,
        patch_size=4,
        group_every_n=3,
        dropout=0.0,
    )
    cfg.update(overrides)
    return T0(**cfg, key=KEY)


def _inputs(time=7):
    values = jr.normal(jr.PRNGKey(1), (2, time))
    mask = jnp.zeros((2, time), dtype=jnp.int8)
    groups = jnp.zeros((2, time), dtype=jnp.int32)
    types = jnp.zeros((2, time), dtype=jnp.int32)
    return values, mask, groups, types


def test_forward_shape_finite_and_monotonic():
    output = _tiny()(*_inputs(), key=KEY, inference=True)
    assert output.shape == (2, 2, 4, 5)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.all(jnp.diff(output, axis=-1) >= 0))


def test_reuses_equimo_blocks_and_t0_pattern():
    model = _tiny()
    assert isinstance(model.blocks[0], BlockChunk)
    assert isinstance(model.patch_encoder.projection.mlp, Mlp)
    assert isinstance(model.blocks[0].blocks[0].attn.attention, Attention)
    assert isinstance(model.blocks[0].blocks[0].ffn, SwiGluFused)
    assert [block.attention_type for block in model.blocks[0].blocks] == [
        "time",
        "time",
        "group",
    ]


def test_factory_and_registry():
    kwargs = dict(
        embed_dim=16,
        num_layers=3,
        num_heads=2,
        mlp_hidden_dim=32,
        patch_size=4,
        key=KEY,
    )
    assert isinstance(t0(**kwargs), T0)
    assert isinstance(t0_alpha(**kwargs), T0)
    assert get_model_cls("t0", modality="time_series") is T0
    assert _T0_REGISTRY["t0_alpha"] == {
        "embed_dim": 512,
        "num_layers": 24,
        "num_heads": 8,
        "mlp_hidden_dim": 2048,
        "patch_size": 32,
        "group_every_n": 3,
        "dropout": 0.1,
        "quantile_levels": (0.1, 0.25, 0.5, 0.75, 0.9),
    }


def test_lfs_pointer_is_rejected(tmp_path):
    pointer = tmp_path / "model.safetensors"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_t0_weights(_tiny(), pointer)
