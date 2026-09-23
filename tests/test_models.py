import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import equimo.vision.models as em
from equimo.serialization import load_weights
from equimo.timeseries.models import t0_alpha
from equimo.core.layers.activation import get_act
from equimo.core.layers.rotary import apply_rotary
from equimo.vision.models.mlla import Mlla
from equimo.vision.models.partialformer import PartialFormer
from equimo.vision.layers.posemb import VisionRoPE
from equimo.vision.models.vit import dinov3_vits16_pretrain_lvd1689m
from equimo.utils import make_drop_path_schedule

# Helpers

KEY = jr.PRNGKey(0)
NUM_CLASSES = 10
IMG_64 = jr.normal(KEY, (3, 64, 64))


def test_make_drop_path_schedule_uniform():
    schedule = make_drop_path_schedule(0.1, [2, 3, 4], uniform=True)
    assert schedule == [0.1] * 9
    assert len(schedule) == 9


def test_make_drop_path_schedule_linear():
    schedule = make_drop_path_schedule(0.1, [2, 2], uniform=False)
    assert len(schedule) == 4
    assert schedule[0] == pytest.approx(0.0)
    assert schedule[-1] == pytest.approx(0.1)
    # Monotonically increasing
    assert all(a <= b for a, b in zip(schedule, schedule[1:]))


def test_make_drop_path_schedule_zero_rate():
    schedule = make_drop_path_schedule(0.0, [2, 3])
    assert all(v == pytest.approx(0.0) for v in schedule)


def test_get_act_hard_swish():
    act = get_act("hard_swish")
    x = jnp.array([-2.0, 0.0, 2.0])
    out = act(x)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


# VisionTransformer


def test_vit_classification():
    key = jr.PRNGKey(0)
    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=64,
        patch_size=8,
        num_heads=[2],
        depths=[2],
        num_classes=NUM_CLASSES,
        key=key,
    )
    y = model(IMG_64, key=key, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_vit_cls_patch_mean_global_pool():
    key = jr.PRNGKey(0)
    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=64,
        patch_size=8,
        num_heads=[2],
        depths=[2],
        num_classes=NUM_CLASSES,
        global_pool="cls_patch_mean",
        key=key,
    )
    y = model(IMG_64, key=key, inference=True)

    assert model.head.in_features == 128
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_vit_rope():
    # RoPE requires dynamic_img_size=True so patch_embed returns (c, h, w)
    # allowing H and W to be read from spatial dims.
    key = jr.PRNGKey(1)
    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=64,
        patch_size=8,
        num_heads=2,
        depths=[2],
        num_classes=NUM_CLASSES,
        use_global_pos_embed=False,
        use_local_pos_embed=True,
        dynamic_img_size=True,
        class_token=True,
        reg_tokens=0,
        key=key,
    )
    y_train = model(IMG_64, key=key, inference=False)
    y_infer = model(IMG_64, key=key, inference=True)
    assert y_train.shape == (NUM_CLASSES,)
    assert y_infer.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y_infer))


def test_vit_private_block_runner_preserves_untransformed_features():
    key = jr.PRNGKey(2)
    model = em.VisionTransformer(
        img_size=32,
        in_channels=3,
        dim=16,
        patch_size=8,
        num_heads=2,
        depths=[1, 1],
        num_classes=0,
        use_global_pos_embed=False,
        use_local_pos_embed=True,
        dynamic_img_size=True,
        key=key,
    )
    x = jr.normal(key, (3, 32, 48))
    key_pos, *block_keys = jr.split(key, len(model.blocks) + 1)
    prepared = model._prepare_tokens(
        x,
        key=key_pos,
        mask=None,
        inference=False,
    )
    expected, height, width, rotary = prepared

    for block, block_key in zip(model.blocks, block_keys, strict=True):
        key_pos, key_rope = jr.split(key_pos, 2)
        rotary = model.local_pos_embed.get_factors(
            H=height,
            W=width,
            inference=False,
            key=key_rope,
        )
        expected = block(
            expected,
            rotary=rotary,
            inference=False,
            key=block_key,
        )

    actual = model.features(x, key=key, inference=False)

    assert jnp.array_equal(actual, expected)


def test_vit_intermediate_features_last_block_matches_features():
    key = jr.PRNGKey(2)
    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=8,
        num_heads=[2],
        depths=[2],
        num_classes=0,
        key=key,
    )

    intermediates = model.intermediate_features(
        IMG_64,
        key=key,
        inference=True,
        n_last_blocks=2,
    )
    features = model.features(IMG_64, key=key, inference=True)

    assert len(intermediates) == 2
    assert intermediates[-1].shape == features.shape
    assert jnp.allclose(intermediates[-1], features)


def test_vit_forward_features_without_class_token_has_no_cls_token():
    key = jr.PRNGKey(3)
    model = em.VisionTransformer(
        img_size=32,
        in_channels=3,
        dim=16,
        patch_size=8,
        num_heads=[2],
        depths=[1],
        num_classes=0,
        class_token=False,
        reg_tokens=0,
        global_pos_embed_cls=False,
        key=key,
    )
    x = jr.normal(key, (3, 32, 32))

    fwd = model.forward_features(x, key=key, inference=True)

    assert fwd["x_norm_cls_token"] is None
    assert fwd["x_norm_reg_tokens"].shape == (0, 16)
    assert fwd["x_norm_patchtokens"].shape == (16, 16)


# VisionParcae


def _small_vision_parcae(**kwargs):
    cfg = {
        "img_size": 64,
        "in_channels": 3,
        "dim": 32,
        "patch_size": 8,
        "num_heads": 2,
        "num_classes": NUM_CLASSES,
        "n_layers_in_prelude": 1,
        "n_layers_in_recurrent_block": 1,
        "n_layers_in_coda": 1,
        "mean_recurrence": 2,
        "mean_backprop_depth": 1,
        "max_recurrence": 2,
        "key": KEY,
    }
    cfg.update(kwargs)
    return em.VisionParcae(**cfg)


def test_vision_parcae_forward():
    model = _small_vision_parcae()
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_vision_parcae_features_and_aux():
    model = _small_vision_parcae()
    feats = model.features(IMG_64, key=KEY, inference=True)
    fwd = model.forward_features(IMG_64, key=KEY, inference=True)

    assert feats.shape == (65, 32)
    assert fwd["x_norm_cls_token"].shape == (32,)
    assert fwd["x_norm_patchtokens"].shape == (64, 32)
    assert fwd["x_recurrent_state"].shape == (65, 32)
    assert int(fwd["num_steps"]) == 2
    assert int(fwd["num_steps_no_grad"]) == 0
    assert int(fwd["num_steps_with_grad"]) == 2
    assert jnp.all(jnp.isfinite(feats))
    assert jnp.all(jnp.isfinite(fwd["recurrent_residual"]))


@pytest.mark.parametrize("injection_type", ["diagonal", "linear", "add"])
def test_vision_parcae_injection_variants(injection_type):
    model = _small_vision_parcae(injection_type=injection_type)
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(model.recurrent_contraction_factor()))


def test_vision_parcae_tiny_factory_forward():
    model = em.vision_parcae_tiny_patch16_224(
        img_size=64,
        dim=32,
        num_heads=2,
        recurrent_dim=32,
        recurrent_num_heads=2,
        patch_size=8,
        num_classes=NUM_CLASSES,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=1,
        mean_recurrence=2,
        mean_backprop_depth=1,
        max_recurrence=2,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


# ReduceFormer


def test_fused_reduceformer():
    key = jr.PRNGKey(42)
    x = jr.normal(key, (3, 64, 64))
    model = em.reduceformer_backbone_b1(
        in_channels=3, num_classes=10, fuse_mbconv=True, key=key
    )
    y_hat = model(x, key=key)
    assert len(y_hat) == 10


# Mlla


def test_mlla_drop_path_schedule():
    """Verify per-block drop path is applied (not per-stage)."""
    depths = [1, 1, 2, 1]
    model = Mlla(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=depths,
        num_heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
        drop_path_rate=0.1,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)


# PartialFormer


def test_partialformer_tuple_blocks():
    """self.blocks must be a tuple, not a list."""
    model = PartialFormer(
        img_size=64,
        in_channels=3,
        dim=32,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=0.5,
        patch_size=4,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    assert isinstance(model.blocks, tuple)


def test_partialformer_foreground_ratios_tuple():
    """foreground_ratios as a 2-tuple (range) must not raise."""
    model = PartialFormer(
        img_size=64,
        in_channels=3,
        dim=32,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=(0.3, 0.7),
        patch_size=4,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)


# ConvNeXt


def test_convnext_intermediate_features_returns_native_stage_outputs():
    model = em.ConvNeXt(
        in_channels=3,
        depths=[1, 1],
        dims=[32, 64],
        num_classes=0,
        key=KEY,
    )
    x = jr.normal(KEY, (3, 64, 64))

    intermediates = model.intermediate_features(
        x,
        key=KEY,
        inference=True,
        n_last_blocks=1,
    )

    assert len(intermediates) == 1
    assert intermediates[0].ndim == 3
    assert jnp.all(jnp.isfinite(intermediates[0]))


def test_convnext_drop_path():
    model = em.ConvNeXt(
        in_channels=3,
        depths=[2, 2],
        dims=[32, 64],
        drop_path_rate=0.1,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    x = jr.normal(KEY, (3, 64, 64))
    y = model(x, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


# Reference contracts


def test_dinov3_local_rope_config_matches_official():
    model = dinov3_vits16_pretrain_lvd1689m(pretrained=False)
    rope = model.local_pos_embed.patch_rope

    D_head = model.dim // 6
    expected = 100.0 ** (
        2.0 * jnp.arange(D_head // 4, dtype=jnp.float32) / float(D_head // 2)
    )

    assert rope.freqs.dtype == jnp.float32
    assert rope.normalize_coords == "separate"
    assert rope.rescale_coords == 2.0
    np.testing.assert_allclose(
        np.asarray(rope.freqs), np.asarray(expected), rtol=0, atol=0
    )


def test_lingbot_vits16_rectangular_features():
    model = em.lingbot_vits16()
    image = jr.normal(KEY, (3, 32, 48))
    features = model.forward_features(image, key=KEY, inference=True)

    assert features["x_norm_cls_token"].shape == (384,)
    assert features["x_norm_reg_tokens"].shape == (4, 384)
    assert features["x_norm_patchtokens"].shape == (6, 384)
    assert jnp.isfinite(features["x_norm_patchtokens"]).all()

    metadata = model.feature_metadata(
        image,
        endpoint="intermediate_features",
        endpoint_options={"indices": (2, 5, 8, 11), "apply_norm": True},
    )
    assert metadata["grid_size"] == (2, 3)
    assert metadata["prefix_tokens"] == (
        "cls",
        "register_0",
        "register_1",
        "register_2",
        "register_3",
    )


@pytest.mark.live_reference_parity
def test_lingbot_vits16_matches_pinned_reference():
    directory = Path(
        os.environ.get("EQUIMO_LINGBOT_ARCHIVE_DIR", "~/.cache/equimo/lingbot")
    ).expanduser()
    archive = directory / "lingbot_vits16.tar.lz4"
    if not archive.is_file():
        pytest.skip("converted LingBot-Vision Small checkpoint is not cached locally")

    record = json.loads((directory / "lingbot_vits16.conversion.json").read_text())
    data_dir = Path(__file__).parent / "data"
    reference_path = data_dir / "lingbot_vits16_reference.npz"
    provenance = json.loads((data_dir / "reference_provenance.json").read_text())[
        "fixtures"
    ][reference_path.name]
    assert record["source_code_revision"] == provenance["author_code_revision"]
    assert record["source_checkpoint_revision"] == provenance["upstream"]["revision"]
    assert (
        record["source_checkpoint_sha256"]
        == provenance["upstream"]["checkpoint_sha256"]
    )
    assert record["converted_archive_sha256"] == provenance["converted_archive_sha256"]
    jax.config.update("jax_default_matmul_precision", "highest")
    with np.load(reference_path, allow_pickle=False) as reference:
        model = load_weights(
            em.lingbot_vits16(),
            path=archive,
            expected_sha256=provenance["converted_archive_sha256"],
            expected_model_config=record["model_config"],
        )
        image = jnp.asarray(reference["input"])
        key = jr.PRNGKey(42)
        output = model.forward_features(image, key=key, inference=True)
        values = {
            "class": output["x_norm_cls_token"],
            "registers": output["x_norm_reg_tokens"],
            "patches": output["x_norm_patchtokens"],
            "prenorm": output["x_prenorm"],
        }
        taps = (2, 5, 8, 11)
        raw = model.intermediate_features(image, key=key, inference=True, indices=taps)
        norm = model.intermediate_features(
            image, key=key, inference=True, indices=taps, apply_norm=True
        )
        values.update(
            {f"tap_{tap}_raw": value for tap, value in zip(taps, raw, strict=True)}
        )
        values.update(
            {f"tap_{tap}_norm": value for tap, value in zip(taps, norm, strict=True)}
        )
        for name, value in values.items():
            actual = np.asarray(value)
            expected = reference[name]
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(
                actual,
                expected,
                atol=provenance["comparison"]["atol"],
                rtol=provenance["comparison"]["rtol"],
            )


@pytest.mark.live_reference_parity
def test_predict_matches_tfc_t0_alpha():
    checkpoint = os.environ.get("T0_ALPHA_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set T0_ALPHA_CHECKPOINT to the upstream T0 snapshot")
    checkpoint_path = Path(checkpoint).expanduser()
    snapshot = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip(f"upstream T0 checkpoint is missing from {snapshot}")
    if not Path("~/.cache/equimo/t0/t0_alpha.tar.lz4").expanduser().is_file():
        pytest.skip("converted T0-alpha weights are not cached locally")

    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("t0")
    rng = np.random.default_rng(7)
    context = rng.standard_normal((1, 64)).astype(np.float32)
    context[0, 5] = np.nan
    horizon = 17
    quantiles = (0.1, 0.25, 0.9)

    reference = upstream.T0Forecaster.from_pretrained(str(snapshot)).eval()
    with torch.inference_mode():
        expected = reference.predict(context, horizon=horizon, quantiles=quantiles)
    actual = t0_alpha(pretrained=True, key=KEY).predict(
        context, horizon=horizon, quantiles=quantiles
    )

    np.testing.assert_allclose(
        np.asarray(actual.quantiles),
        expected.quantiles.numpy(),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual.median),
        expected.median.numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_vit5_rope_matches_official_interleaved_reference():
    reference = np.load(Path(__file__).parent / "data" / "vit5_rope_reference.npz")
    rope = VisionRoPE(
        strategy="mode",
        dim=int(reference["dim"]),
        pt_seq_len=int(reference["pt_seq_len"]),
        theta=float(reference["theta"]),
    )

    factors = rope.get_factors(
        H=int(reference["height"]),
        W=int(reference["width"]),
    )
    output = apply_rotary(
        jnp.asarray(reference["x"]),
        factors,
        sequence_axis=-3,
    )

    assert factors.layout == "interleaved"
    np.testing.assert_allclose(np.asarray(factors.sin), reference["sin"], atol=1e-6)
    np.testing.assert_allclose(np.asarray(factors.cos), reference["cos"], atol=1e-6)
    np.testing.assert_allclose(np.asarray(output), reference["output"], atol=1e-6)


# DEQ
#
# DEQ-specific concerns that these tests exercise:
#   * The forward pass composes a fixed-point solve inside a regular forward,
#     so both inference and training modes must behave sanely.
#   * The solver must actually converge — ``aux["error"]`` must land below the
#     configured tolerance, and ``||f(z*, x) - z*||`` must be small.
#   * Gradients must flow through the NeumannPhantom adjoint (the main reason
#     to use a DEQ at all).
#   * Switching injector / stabilizer / strategy via kwargs must work.


def test_deq_forward():
    """Default preset (prenorm_add + projected + entry) — full forward pass."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y, auxs = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    assert isinstance(auxs, list)
    # Default config has one FPI stage.
    assert len(auxs) == 1


def test_deq_forward_training_mode():
    """Training forward pass with DropPath enabled.

    The solver reuses a single RNG key across all Picard iterations to freeze
    stochastic ops; without this the fixed point does not exist. This test
    just verifies the training-mode forward runs and produces finite outputs.
    """
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        drop_path_rate=0.1,
        key=KEY,
    )
    y, auxs = model(IMG_64, key=KEY, inference=False)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    assert len(auxs) == 1


def test_deq_features():
    """features() returns a 3D feature map + the aux list."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=0,
        key=KEY,
    )
    feats, auxs = model.features(IMG_64, key=KEY, inference=True)
    assert feats.ndim == 3
    assert jnp.all(jnp.isfinite(feats))
    assert len(auxs) == 1


def test_deq_aux_structure():
    """Aux dict must expose everything downstream regularizers need."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    _, auxs = model(IMG_64, key=KEY, inference=True)
    aux = auxs[0]
    for k in ("z_star", "trajectory", "depth", "error", "key", "x_context", "z0"):
        assert k in aux, f"aux dict missing required key: {k}"
    assert jnp.all(jnp.isfinite(aux["z_star"]))


def test_deq_solver_converges():
    """Solver must report convergence below tolerance and not burn max_steps."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        fpi_tol=1e-3,
        fpi_maxsteps=50,
        key=KEY,
    )
    _, auxs = model(IMG_64, key=KEY, inference=True)
    aux = auxs[0]
    # Solver tolerance is 1e-3; give an order of magnitude slack.
    assert float(aux["error"]) < 1e-2, f"solver error: {float(aux['error']):.2e}"
    # If depth ≈ max_steps, the solver ran out of budget without converging.
    assert int(aux["depth"]) < 50


def test_deq_fixed_point_consistency():
    """At convergence, ``f(z*, x) ≈ z*`` must hold within solver tolerance."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=0,
        key=KEY,
    )
    _, auxs = model.features(IMG_64, key=KEY, inference=True)
    aux = auxs[0]
    z_star = aux["z_star"]
    x_context = aux["x_context"]
    key_solve = aux["key"]

    # Locate the DEQ block (one FPI stage at index 2 in the default config).
    deq_stage_idx = next(
        i for i, blk in enumerate(model.blocks) if blk.deq_block is not None
    )
    cell = model.blocks[deq_stage_idx].deq_block.cell

    z_next = cell(z_star, x_context, inference=True, key=key_solve)
    rel = float(jnp.linalg.norm(z_next - z_star) / (jnp.linalg.norm(z_star) + 1e-8))
    assert rel < 1e-2, f"|f(z*) - z*| / |z*| = {rel:.2e}"


def test_deq_gradients_finite_and_nonzero():
    """Backward pass through DEQ must produce finite, non-zero gradients.

    This is the canonical DEQ smoke test: if the implicit-differentiation
    adjoint is broken, gradients will be NaN, inf, or uniformly zero.
    """
    import equinox as eqx

    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        key=KEY,
    )

    def loss_fn(m, x):
        y, _ = m(x, key=KEY, inference=True)
        return jnp.mean(y**2)

    grads = eqx.filter_grad(loss_fn)(model, IMG_64)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves), "NaN/Inf in gradients"
    assert any(jnp.any(g != 0) for g in leaves), "All-zero gradients"


def test_deq_determinism_same_key():
    """Same input and same key must produce the same output.

    Non-determinism here would mean the fixed point does not exist.
    """
    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    y1, _ = model(IMG_64, key=KEY, inference=True)
    y2, _ = model(IMG_64, key=KEY, inference=True)
    assert jnp.allclose(y1, y2)


@pytest.mark.parametrize(
    "injector,stabilizer,strategy",
    [
        # Preset default: pre-norm injection + GroupNorm projection.
        ("prenorm_add", "projected", "entry"),
        # Pre-norm + damped projection.
        ("prenorm_add", "damped_projected", "entry"),
        # Projection stabilizer alone bounds the iterate even with trivial injection.
        ("add", "projected", "entry"),
        # Projection at every block.
        ("add", "projected", "per_block"),
        # Gated injector (init_gate=0.5) is a damped Picard at init;
        # projection on top gives a belt-and-suspenders setup.
        ("gated", "projected", "entry"),
        # Projection + damping with a projected forcing term.
        ("proj_add", "damped_projected", "entry"),
    ],
)
def test_deq_injector_stabilizer_strategy_combos(injector, stabilizer, strategy):
    """Each well-posed (injector, stabilizer, strategy) combo must converge.

    Combos that lack **any** bounding mechanism — e.g. ``add + identity`` or
    ``proj_add + damped`` — are deliberately absent: ConvNeXt blocks with
    ``LayerScale(1e-6)`` are near-identity, so without at least one of
    {pre-norm on z, projection on output, convex-mix gate with init_gate < 1}
    the Picard iteration ``z_{k+1} ≈ z_k + x`` diverges linearly and burns
    through ``max_steps``. That's expected behavior, not a bug — the minimal
    combo is the library's simple default for general use, and
    architecture-specific presets (like ``deq_convnext_t``) layer bounding on
    top. For a finiteness-only smoke matrix over all 32 combos, see
    ``test_implicit.py::test_deqblock_all_combinations_run``.
    """
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        fpi_injector=injector,
        fpi_stabilizer=stabilizer,
        fpi_strategy=strategy,
        key=KEY,
    )
    y, auxs = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    # Every well-posed combo must converge within the solver's budget.
    assert int(auxs[0]["depth"]) < 50


def test_deq_blocks_is_tuple():
    """Structural invariant: BlockChunk.blocks must be a pytree-friendly tuple."""
    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    assert isinstance(model.blocks, tuple)


def test_deq_jit_compatible():
    """The model must trace cleanly under filter_jit."""
    import equinox as eqx

    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)

    @eqx.filter_jit
    def forward(m, x, k):
        y, _ = m(x, key=k, inference=True)
        return y

    y = forward(model, IMG_64, KEY)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_deq_get_fpi_cells():
    """get_fpi_cells() must return a tuple of DEQCell instances."""
    from equimo.core.implicit import DEQCell

    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    cells = model.get_fpi_cells()

    assert isinstance(cells, tuple)
    assert len(cells) == 1
    assert isinstance(cells[0], DEQCell)
