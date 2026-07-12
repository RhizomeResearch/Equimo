"""Tests for equimo.vision.layers.attention."""

import io
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx

from equimo.core.layers import Attention as CoreAttention
from equimo.core.layers import AttentionBlock as CoreAttentionBlock
from equimo.core.layers import DropPath, Mamba2Mixer, Mlp
from equimo.core.layers import get_layer as get_core_layer
from equimo.core.layers.attention import (
    rope_apply as core_rope_apply,
    rope_apply_qk_last_hw as core_rope_apply_qk_last_hw,
    rope_rotate_half as core_rope_rotate_half,
)
from equimo.core.layers.norm import LayerScale
from equimo.vision.layers import get_layer as get_vision_layer
from equimo.vision.layers.attention import (
    Attention,
    WindowedAttention,
    AttentionBlock,
    HATBlock,
    SHSA,
    SHMA,
    SHMABlock,
    LinearAttention,
    MllaBlock,
    MMSA,
    SQA,
    PartialFormerBlock,
    LinearAngularAttention,
    RFAttention,
    RFAttentionBlock,
    ConvAttention,
    ConvAttentionBlock,
    LowFormerBlock,
    get_attn,
    get_attn_block,
    rope_apply,
    rope_apply_qk_last_hw,
    rope_rotate_half,
)

KEY = jr.PRNGKey(0)
DIM = 32
NUM_HEADS = 4
SEQLEN = 16
H, W = 4, 4


class TestAttentionLayers:
    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (Attention, {"dim": DIM, "num_heads": NUM_HEADS, "qk_norm": False}),
            (Attention, {"dim": DIM, "num_heads": NUM_HEADS, "qk_norm": True}),
            (
                WindowedAttention,
                {"dim": DIM, "num_heads": NUM_HEADS, "resolution": 4, "seq_len": 16},
            ),
            (SHSA, {"dim": DIM, "qk_dim": 8, "pdim": 8}),
            (SHMA, {"dim": DIM, "num_heads": 1}),
            (
                LinearAttention,
                {"input_resolution": (4, 4), "dim": DIM, "num_heads": NUM_HEADS},
            ),
            (MMSA, {"dim": DIM, "num_heads": NUM_HEADS}),
            (SQA, {"dim": DIM, "num_heads": NUM_HEADS}),
            (LinearAngularAttention, {"dim": DIM, "num_heads": NUM_HEADS}),
            (RFAttention, {"in_channels": DIM, "out_channels": DIM}),
            (ConvAttention, {"in_channels": DIM}),
        ],
    )
    def test_attention_forward(self, cls, kwargs):
        key = KEY
        model = cls(**kwargs, key=key)

        if cls in (SHSA, SHMA, RFAttention, ConvAttention):
            x = jr.normal(key, (DIM, H, W))
        else:
            x = jr.normal(key, (SEQLEN, DIM))

        if cls is (SQA):
            q = jr.normal(key, (1, DIM))
            out = model(x, q, key=key, inference=True)
        else:
            out = model(x, key=key, inference=True)
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (AttentionBlock, {"dim": DIM, "num_heads": NUM_HEADS}),
            (
                HATBlock,
                {"dim": DIM, "num_heads": NUM_HEADS, "window_size": 4, "sr_ratio": 2},
            ),
            (SHMABlock, {"dim": DIM}),
            (
                MllaBlock,
                {"dim": DIM, "input_resolution": (4, 4), "num_heads": NUM_HEADS},
            ),
            (
                PartialFormerBlock,
                {
                    "dim": DIM,
                    "num_heads": NUM_HEADS,
                    "foreground_ratio": 0.5,
                    "patch_size": 2,
                },
            ),
            (RFAttentionBlock, {"in_channels": DIM}),
            (ConvAttentionBlock, {"dim": DIM}),
            (LowFormerBlock, {"dim": DIM}),
        ],
    )
    def test_block_forward(self, cls, kwargs):
        key = KEY
        model = cls(**kwargs, key=key)

        if cls in (SHMABlock, RFAttentionBlock, LowFormerBlock, ConvAttentionBlock):
            x = jr.normal(key, (DIM, H, W))
        else:
            x = jr.normal(key, (SEQLEN, DIM))

        if cls == HATBlock:
            sr_ratio = kwargs["sr_ratio"]
            ct_size = kwargs.get("ct_size", 1)
            ct_total = ct_size**2 * sr_ratio**2
            ct = jr.normal(key, (ct_total, DIM))
            out, ct_out = model(x, ct, key=key, inference=True)
            assert out.shape == x.shape
            assert ct_out.shape == ct.shape
        elif cls == PartialFormerBlock:
            qa = jr.normal(key, (1, DIM))
            out, qa_out = model(x, qa, key=key, inference=True)
            assert out.shape == x.shape
            assert qa_out.shape == qa.shape
        else:
            out = model(x, key=key, inference=True)
            assert out.shape == x.shape

        assert jnp.all(jnp.isfinite(out))

    def test_registry(self):
        assert get_attn("attention") is Attention
        assert get_attn_block("attentionblock") is AttentionBlock

    def test_standard_attention_reuses_core_implementation(self):
        assert Attention is CoreAttention
        assert AttentionBlock is CoreAttentionBlock
        assert get_attn("attention") is CoreAttention
        assert get_attn_block("attentionblock") is CoreAttentionBlock
        assert rope_rotate_half is core_rope_rotate_half
        assert rope_apply is core_rope_apply
        assert rope_apply_qk_last_hw is core_rope_apply_qk_last_hw

    @pytest.mark.parametrize("init_values", [None, 1e-5])
    def test_standard_attention_block_checkpoint_compatibility(self, init_values):
        source = AttentionBlock(
            DIM,
            NUM_HEADS,
            qk_norm=True,
            init_values=init_values,
            key=KEY,
        )
        template = CoreAttentionBlock(
            DIM,
            NUM_HEADS,
            qk_norm=True,
            init_values=init_values,
            key=jr.PRNGKey(1),
        )
        source_layout = [
            (jax.tree_util.keystr(path), getattr(leaf, "shape", None))
            for path, leaf in jax.tree_util.tree_flatten_with_path(source)[0]
        ]
        template_layout = [
            (jax.tree_util.keystr(path), getattr(leaf, "shape", None))
            for path, leaf in jax.tree_util.tree_flatten_with_path(template)[0]
        ]
        assert source_layout == template_layout

        checkpoint = io.BytesIO()
        eqx.tree_serialise_leaves(checkpoint, source)
        checkpoint.seek(0)
        restored = eqx.tree_deserialise_leaves(checkpoint, template)
        x = jr.normal(KEY, (SEQLEN, DIM))
        source_output = source(x, key=KEY, inference=True)
        restored_output = restored(x, key=KEY, inference=True)
        assert jnp.array_equal(source_output, restored_output)

    def test_zero_init_values_retains_zero_layer_scale(self):
        block = AttentionBlock(DIM, NUM_HEADS, init_values=0.0, key=KEY)

        assert isinstance(block.ls1, LayerScale)
        assert isinstance(block.ls2, LayerScale)
        assert jnp.array_equal(block.ls1.gamma, jnp.zeros((DIM,)))
        assert jnp.array_equal(block.ls2.gamma, jnp.zeros((DIM,)))

    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_standard_attention_mask_rope_and_dropout(self, qk_norm):
        model = Attention(
            DIM,
            NUM_HEADS,
            qk_norm=qk_norm,
            attn_drop=0.2,
            proj_drop=0.2,
            key=KEY,
        )
        x = jr.normal(KEY, (SEQLEN, DIM))
        angles = jr.normal(KEY, (H * W, DIM // NUM_HEADS))
        mask = jnp.tril(jnp.ones((1, SEQLEN, SEQLEN), dtype=bool))

        inference_output = model(
            x,
            mask=mask,
            rope_sincos=(jnp.sin(angles), jnp.cos(angles)),
            key=KEY,
            inference=True,
        )
        training_output = model(
            x,
            mask=mask,
            rope_sincos=(jnp.sin(angles), jnp.cos(angles)),
            key=KEY,
            inference=False,
        )

        assert inference_output.shape == training_output.shape == x.shape
        assert jnp.all(jnp.isfinite(inference_output))
        assert jnp.all(jnp.isfinite(training_output))
        assert not jnp.array_equal(inference_output, training_output)

    def test_reexported_rope_helper_preserves_shape_validation(self):
        q = jnp.zeros((NUM_HEADS, SEQLEN, DIM // NUM_HEADS))
        bad_sin = jnp.zeros((H * W, 1))
        cos = jnp.ones((H * W, DIM // NUM_HEADS))

        with pytest.raises(ValueError, match="head_dim"):
            rope_apply_qk_last_hw(q, q, bad_sin, cos)

    def test_layer_registry_is_scoped_by_modality(self):
        assert get_core_layer("attention") is CoreAttention
        assert get_vision_layer("attention") is Attention

    @pytest.mark.parametrize(
        "name, cls",
        [
            ("mlp", Mlp),
            ("layernorm", eqx.nn.LayerNorm),
            ("droppath", DropPath),
            ("mamba2mixer", Mamba2Mixer),
        ],
    )
    def test_vision_layer_registry_includes_shared_core_families(self, name, cls):
        assert get_vision_layer(name) is cls

    def test_unknown_layer_lists_only_names_in_scope(self):
        with pytest.raises(ValueError) as core_error:
            get_core_layer("missing")
        with pytest.raises(ValueError) as vision_error:
            get_vision_layer("missing")

        assert "convnextblock" not in str(core_error.value)
        assert "convnextblock" in str(vision_error.value)

    def test_vision_registry_import_failure_is_surfaced(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "equimo.vision.layers.convolution", None)

        with pytest.raises(ModuleNotFoundError, match="convolution"):
            get_vision_layer("attention")

    def test_low_precision(self):
        model = Attention(DIM, NUM_HEADS, key=KEY)
        model = jax.tree_util.tree_map(
            lambda leaf: (
                leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf
            ),
            model,
        )
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.bfloat16)
        out = model(x, key=KEY, inference=True)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))
