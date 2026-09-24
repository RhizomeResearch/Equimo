"""Tests for equimo.vision.layers.attention."""

import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx

from equimo.core.layers import Attention as CoreAttention
from equimo.core.layers import AttentionBlock as CoreAttentionBlock
from equimo.core.layers import DropPath, Mamba2Mixer, Mlp, RotaryFactors
from equimo.core.layers import get_layer as get_core_layer
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
)
from _jaxpr_utils import assert_prng_free_jaxpr

KEY = jr.PRNGKey(0)
DIM = 32
NUM_HEADS = 4
SEQLEN = 16
H, W = 4, 4


class _PartialScale(eqx.Module):
    scale: float

    def __call__(self, x, *args, **kwargs):
        return x * self.scale


@pytest.mark.parametrize("patch_size", (1, 2))
@pytest.mark.parametrize("ratio", (0.0, 0.5, 1.0))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_partialformer_restores_ranked_patches(patch_size, ratio, dtype):
    model = PartialFormerBlock(4, 1, ratio, patch_size, key=KEY)
    model = eqx.tree_at(
        lambda m: (m.posemb, m.mmsa, m.sqa, m.mlp),
        model,
        (
            _PartialScale(1.0),
            _PartialScale(2.0),
            _PartialScale(3.0),
            _PartialScale(0.0),
        ),
    )
    model = jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_array(leaf) else leaf, model
    )
    grid = 4 // patch_size
    scores = np.tile([3.0, 1.0, 3.0, 2.0], grid * grid // 4)
    # Stable ties in the primary ranking must retain their original patch order.
    foreground = np.argsort(-scores, kind="stable")[: int(len(scores) * ratio)]
    scales = np.full(len(scores), 3.0)
    scales[foreground] = 2.0

    def to_tokens(patches):
        image = np.repeat(
            np.repeat(patches.reshape(grid, grid), patch_size, 0), patch_size, 1
        )
        return jnp.broadcast_to(jnp.asarray(image, dtype=dtype).reshape(16, 1), (16, 4))

    x, multipliers = to_tokens(scores), to_tokens(scales)
    qa = jnp.ones((1, 4), dtype=dtype)
    call = lambda x: model(x, qa, inference=True)
    for out, qa_out in (call(x), eqx.filter_jit(call)(x)):
        assert out.dtype == qa_out.dtype == dtype
        assert jnp.array_equal(out, x + x * multipliers)
        assert jnp.array_equal(qa_out, jnp.zeros_like(qa))
    grad = jax.grad(lambda x: call(x)[0].astype(jnp.float32).sum())(x)
    assert jnp.array_equal(grad, 1 + multipliers)


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
            assert_prng_free_jaxpr(
                lambda value: model(value, q, key=key, inference=True), x
            )
            out = model(x, q, key=key, inference=True)
        else:
            assert_prng_free_jaxpr(
                lambda value: model(value, key=key, inference=True), x
            )
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
            assert_prng_free_jaxpr(
                lambda value: model(value, ct, key=key, inference=True), x
            )
            out, ct_out = model(x, ct, key=key, inference=True)
            assert out.shape == x.shape
            assert ct_out.shape == ct.shape
        elif cls == PartialFormerBlock:
            qa = jr.normal(key, (1, DIM))
            assert_prng_free_jaxpr(
                lambda value: model(value, qa, key=key, inference=True), x
            )
            out, qa_out = model(x, qa, key=key, inference=True)
            assert out.shape == x.shape
            assert qa_out.shape == qa.shape
        else:
            assert_prng_free_jaxpr(
                lambda value: model(value, key=key, inference=True), x
            )
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
            rotary=RotaryFactors(
                sin=jnp.sin(angles),
                cos=jnp.cos(angles),
                layout="split_half",
            ),
            key=KEY,
            inference=True,
        )
        training_output = model(
            x,
            mask=mask,
            rotary=RotaryFactors(
                sin=jnp.sin(angles),
                cos=jnp.cos(angles),
                layout="split_half",
            ),
            key=KEY,
            inference=False,
        )

        assert inference_output.shape == training_output.shape == x.shape
        assert jnp.all(jnp.isfinite(inference_output))
        assert jnp.all(jnp.isfinite(training_output))
        assert not jnp.array_equal(inference_output, training_output)

    def test_standard_attention_batched_matches_vmap(self):
        model = Attention(DIM, NUM_HEADS, qk_norm=True, key=KEY)
        x = jr.normal(jr.PRNGKey(1), (3, SEQLEN, DIM))
        mask = jnp.broadcast_to(
            jnp.tril(jnp.ones((SEQLEN, SEQLEN), dtype=bool)),
            (x.shape[0], SEQLEN, SEQLEN),
        )

        batched = model(x, mask=mask, key=KEY, inference=True)
        vmapped = jax.vmap(
            lambda sample, sample_mask: model(
                sample,
                mask=sample_mask,
                key=KEY,
                inference=True,
            )
        )(x, mask)

        assert batched.shape == x.shape
        assert jnp.allclose(batched, vmapped, rtol=1e-5, atol=1e-6)

    def test_standard_attention_fully_masked_rows_have_zero_contribution(self):
        model = Attention(DIM, NUM_HEADS, proj_bias=False, key=KEY)
        x = jr.normal(jr.PRNGKey(1), (2, SEQLEN, DIM))
        mask = jnp.ones((2, SEQLEN, SEQLEN), dtype=bool)
        mask = mask.at[0, 3].set(False)

        batch_mask_output = model(x, mask=mask, key=KEY, inference=True)
        head_mask_output = model(
            x,
            mask=mask[..., None, :, :],
            key=KEY,
            inference=True,
        )

        assert jnp.all(jnp.isfinite(batch_mask_output))
        assert jnp.array_equal(batch_mask_output[0, 3], jnp.zeros((DIM,)))
        assert jnp.array_equal(batch_mask_output, head_mask_output)

    def test_standard_attention_validates_rotary_sequence_length(self):
        model = Attention(DIM, NUM_HEADS, key=KEY)
        x = jr.normal(jr.PRNGKey(1), (2, SEQLEN, DIM))
        rotary = RotaryFactors(
            sin=jnp.zeros((SEQLEN - 1, DIM // NUM_HEADS)),
            cos=jnp.ones((SEQLEN - 1, DIM // NUM_HEADS)),
            layout="split_half",
        )

        with pytest.raises(ValueError, match="sequence length"):
            model(x, rotary=rotary, key=KEY, inference=True)

    def test_linear_attention_supports_configured_rectangular_grid(self):
        model = LinearAttention((2, 3), DIM, NUM_HEADS, key=KEY)
        x = jr.normal(KEY, (6, DIM))

        output = model(x, key=KEY, inference=True)

        assert output.shape == x.shape
        with pytest.raises(ValueError, match="configured grid"):
            model(jnp.ones((5, DIM)), key=KEY, inference=True)

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
