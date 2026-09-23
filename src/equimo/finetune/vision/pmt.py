"""Decoder-only fine-tuning for plain mask transformers."""

from __future__ import annotations

import equinox as eqx

from equimo.finetune.config import FineTunePlan, LLRDConfig, TrainableSpec
from equimo.finetune.surgery import prepare_finetune
from equimo.vision.models.pmt import PMT


def pmt_head_finetune(model: PMT, *, state: eqx.nn.State | None = None) -> FineTunePlan:
    """Select lateral, decoder, query, and prediction parameters; freeze ViT.

    The default GroupNorm route has no model state. For reference BatchNorm,
    pass the state returned by ``eqx.nn.make_with_state(PMT)`` so it accompanies
    the parameter plan and checkpoint. Mask-annealing position is separate.
    """
    if not isinstance(model, PMT):
        raise TypeError("Expected a PMT model.")
    if model.is_stateful() and not isinstance(state, eqx.nn.State):
        raise ValueError("BatchNorm PMT fine-tuning requires explicit model state.")
    return prepare_finetune(
        model,
        trainable=TrainableSpec(mode="head"),
        labels=LLRDConfig(decay=1.0),
        model_state=state,
    )


__all__ = ["pmt_head_finetune"]
