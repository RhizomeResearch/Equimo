# ty: ignore[invalid-return-type]
"""Bounded native checkpoints for plain mask transformers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from equimo._checkpoint_limits import CheckpointLimits
from equimo.vision._encoder_identity import encoder_array_digest
from equimo.serialization import (
    DEFAULT_REPOSITORY_URL,
    _reader_limits,
    _resolve_weights_dir,
    inspect_checkpoint,
    load_weights,
    save_model,
)
from equimo.vision.models.pmd import PMTMaskState, PlainMaskDecoder, ReferenceBatchNorm
from equimo.vision.models.pmt import PMT


class _FullCheckpoint(eqx.Module):
    model: PMT
    state: eqx.nn.State | None
    mask_state: PMTMaskState


class _DecoderCheckpoint(eqx.Module):
    head: PlainMaskDecoder
    state: eqx.nn.State | None
    mask_state: PMTMaskState


def _base_digest(model: PMT) -> str:
    return encoder_array_digest(model.backbone)


def _check_state(model: PMT, state: eqx.nn.State | None) -> eqx.nn.State | None:
    if model.is_stateful():
        if not isinstance(state, eqx.nn.State):
            raise ValueError("BatchNorm PMT checkpoints require complete model state.")
        for lateral in model.head.lateral:
            norm = lateral.norm
            if not isinstance(norm, ReferenceBatchNorm):
                raise ValueError("PMT BatchNorm configuration disagrees with model.")
            state.get(norm.mean_index)
            state.get(norm.variance_index)
            state.get(norm.count_index)
        return state
    if state is not None and jtu.tree_leaves(state):
        raise ValueError("Stateless PMT checkpoints cannot contain model state.")
    return None


def _mask_state(model: PMT, mask_state: PMTMaskState | None) -> PMTMaskState:
    if mask_state is None:
        return PMTMaskState(jnp.ones((model.head.config.num_blocks,)))
    if mask_state.probabilities.shape != (model.head.config.num_blocks,):
        raise ValueError("PMT mask state does not match decoder depth.")
    return mask_state


def _configuration(model: PMT, *, kind: str, base_digest: str) -> dict:
    config = model.head.config
    groups = (
        model.head.lateral[0].norm.groups  # ty: ignore[unresolved-attribute]
        if config.norm_layer == "groupnorm"
        else None
    )
    return {
        "schema": 1,
        "kind": kind,
        "architecture": asdict(config),
        "group_count": groups,
        "masked_attention": model.head.masked_attention,
        "backbone_id": model.backbone_id,
        "base_digest": base_digest,
        "input_view": model.input_view,
        "class_ontology": model.class_ontology,
        "feature_endpoint": {
            "name": "intermediate_features",
            "apply_norm": True,
            "layer_indices": config.taps,
            "token_layout": "BNC",
        },
        "prefix_tokens": model.backbone.num_prefix_tokens,
        "patch_size": model.head.patch_size,
        "positional_configuration": model.backbone._feature_position_configuration(),
    }


def _recorded_configuration(
    path: Path, template: eqx.Module, *, limits: CheckpointLimits | None
) -> dict:
    info = inspect_checkpoint(path, model=template, allow_legacy=False, limits=limits)
    directory = _resolve_weights_dir(
        None, info.path, DEFAULT_REPOSITORY_URL, None, _reader_limits(limits)
    )
    with (directory / "metadata.json").open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    config = metadata.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("PMT checkpoint is missing its architecture configuration.")
    return config


def _check_configuration(recorded: dict, expected: dict, *, full: bool) -> None:
    if full:
        recorded = {
            key: value for key, value in recorded.items() if key != "base_digest"
        }
        expected = {
            key: value for key, value in expected.items() if key != "base_digest"
        }
    if recorded != json.loads(json.dumps(expected)):
        raise ValueError("PMT checkpoint configuration or base identity mismatch.")


def save_pmt_checkpoint(
    path: Path,
    model: PMT,
    *,
    state: eqx.nn.State | None = None,
    mask_state: PMTMaskState | None = None,
) -> Path:
    """Save all encoder/decoder parameters and optional BN/mask state."""
    snapshot = _FullCheckpoint(
        model, _check_state(model, state), _mask_state(model, mask_state)
    )
    return save_model(
        path,
        snapshot,
        _configuration(model, kind="full", base_digest=_base_digest(model)),
    )


def load_pmt_checkpoint(
    path: Path,
    template: PMT,
    *,
    state: eqx.nn.State | None = None,
    limits: CheckpointLimits | None = None,
) -> tuple[PMT, eqx.nn.State | None, PMTMaskState]:
    """Restore a full native checkpoint into the same architecture template."""
    snapshot = _FullCheckpoint(
        template, _check_state(template, state), _mask_state(template, None)
    )
    recorded = _recorded_configuration(path, snapshot, limits=limits)
    _check_configuration(
        recorded, _configuration(template, kind="full", base_digest=""), full=True
    )
    loaded = cast(
        _FullCheckpoint,
        load_weights(snapshot, path=path, inference_mode=False, limits=limits),
    )
    if _base_digest(loaded.model) != recorded["base_digest"]:
        raise ValueError("Full PMT checkpoint encoder digest mismatch.")
    model = eqx.tree_at(
        lambda candidate: candidate.backbone_digest,
        loaded.model,
        recorded["base_digest"],
    )
    return model, loaded.state, loaded.mask_state


def save_pmt_decoder(
    path: Path,
    model: PMT,
    *,
    state: eqx.nn.State | None = None,
    mask_state: PMTMaskState | None = None,
) -> Path:
    """Save decoder parameters and state bound to exact encoder arrays."""
    if model.backbone_id == "unbound":
        raise ValueError("Decoder-only artifacts require a bound backbone_id.")
    snapshot = _DecoderCheckpoint(
        model.head, _check_state(model, state), _mask_state(model, mask_state)
    )
    return save_model(
        path,
        snapshot,
        _configuration(model, kind="decoder", base_digest=_base_digest(model)),
    )


def load_pmt_decoder(
    path: Path,
    template: PMT,
    *,
    state: eqx.nn.State | None = None,
    limits: CheckpointLimits | None = None,
) -> tuple[PMT, eqx.nn.State | None, PMTMaskState]:
    """Restore a decoder only when encoder and ontology identities match."""
    if template.backbone_id == "unbound":
        raise ValueError("Decoder-only loading requires a bound backbone_id.")
    snapshot = _DecoderCheckpoint(
        template.head, _check_state(template, state), _mask_state(template, None)
    )
    recorded = _recorded_configuration(path, snapshot, limits=limits)
    expected = _configuration(
        template, kind="decoder", base_digest=_base_digest(template)
    )
    _check_configuration(recorded, expected, full=False)
    loaded = cast(
        _DecoderCheckpoint,
        load_weights(
            snapshot,
            path=path,
            inference_mode=False,
            expected_model_config=expected,
            limits=limits,
        ),
    )
    model = eqx.tree_at(lambda candidate: candidate.head, template, loaded.head)
    return model, loaded.state, loaded.mask_state


__all__ = [
    "save_pmt_checkpoint",
    "load_pmt_checkpoint",
    "save_pmt_decoder",
    "load_pmt_decoder",
]
