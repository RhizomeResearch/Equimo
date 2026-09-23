"""Pretrained Small encoder and PMD compositions with local reference archives."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from equimo.finetune.vision import pmt_head_finetune
from equimo.serialization import load_weights
from equimo.vision.models import (
    PMT,
    dinov3_vits16_pretrain_lvd1689m,
    lingbot_vits16,
)


DATA = Path(__file__).parent / "data"


@pytest.mark.live_reference_parity
@pytest.mark.parametrize("provider", ["dinov3", "lingbot"])
def test_pretrained_small_composition_reference_and_decoder_fit(provider: str):
    if provider == "dinov3":
        archive = (
            Path.home() / ".cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4"
        )
        if not archive.is_file():
            pytest.skip("DINOv3 Small native checkpoint is not cached locally")
        backbone = dinov3_vits16_pretrain_lvd1689m(pretrained=True)
        fixture_path = DATA / "pmt_dinov3_taps_reference.npz"
    else:
        archive = Path.home() / ".cache/equimo/lingbot/lingbot_vits16.tar.lz4"
        record_path = archive.with_suffix("").with_suffix(".conversion.json")
        if not archive.is_file() or not record_path.is_file():
            pytest.skip("LingBot-Vision Small native checkpoint is not cached locally")
        record = json.loads(record_path.read_text())
        backbone = load_weights(
            lingbot_vits16(),
            path=archive,
            expected_sha256=record["converted_archive_sha256"],
            expected_model_config=record["model_config"],
        )
        fixture_path = DATA / "lingbot_vits16_reference.npz"

    model = PMT(backbone, 3, key=jr.PRNGKey(12), backbone_id=provider)
    with np.load(fixture_path) as reference:
        image = jnp.asarray(reference["image" if provider == "dinov3" else "input"])
        features = model.encode(image)
        assert features.taps == (2, 5, 8, 11)
        assert features.prefix_tokens == (
            "cls",
            "register_0",
            "register_1",
            "register_2",
            "register_3",
        )
        assert features.grid_size == (2, 3)
        for level, tap in zip(features.levels, features.taps, strict=True):
            expected = f"tap_{tap}" + ("" if provider == "dinov3" else "_norm")
            tolerance = 1e-4 if provider == "dinov3" else 1.5e-4
            np.testing.assert_allclose(
                level, reference[expected], atol=tolerance, rtol=tolerance
            )
        output = model.decode(features, key=jr.PRNGKey(13))
        assert output.final.mask_logits.shape == (
            100,
            4 * features.grid_size[0],
            4 * features.grid_size[1],
        )
        assert bool(jnp.all(jnp.isfinite(output.final.mask_logits)))

    plan = pmt_head_finetune(model)
    assert plan.report.total_params > plan.report.trainable_params > 0
    assert all(path.startswith("head.") for path in plan.report.target_paths)
    fit_features = model.encode(jr.normal(jr.PRNGKey(14), (3, 32, 48)))

    def loss(trainable):
        candidate = eqx.combine(trainable, plan.frozen)
        prediction = candidate.decode(fit_features, key=jr.PRNGKey(15)).final
        return jnp.mean(prediction.class_logits**2) + jnp.mean(
            prediction.mask_logits**2
        )

    value, gradient = eqx.filter_value_and_grad(loss)(plan.trainable)
    assert bool(jnp.isfinite(value))
    assert gradient.head.queries is not None
    updated = eqx.combine(
        eqx.apply_updates(
            plan.trainable,
            jtu.tree_map(
                lambda leaf: -1e-4 * leaf if eqx.is_array(leaf) else None,
                gradient,
            ),
        ),
        plan.frozen,
    )
    assert not np.array_equal(updated.head.queries, model.head.queries)
    for old, new in zip(
        jtu.tree_leaves(model.backbone), jtu.tree_leaves(updated.backbone)
    ):
        if eqx.is_array(old):
            np.testing.assert_array_equal(old, new)
    after = updated.encode(jr.normal(jr.PRNGKey(14), (3, 32, 48)))
    for old, new in zip(fit_features.levels, after.levels, strict=True):
        np.testing.assert_array_equal(old, new)
