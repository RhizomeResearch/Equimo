"""Full-encoder fine-tuning for encoder-only mask transformers."""

from __future__ import annotations

import equinox as eqx

from equimo.finetune.config import FineTunePlan, LLRDConfig, TargetSpec, TrainableSpec
from equimo.finetune.surgery import prepare_finetune
from equimo.vision.models.eomt import EoMT


def eomt_full_finetune(model: EoMT, *, decay: float = 0.8) -> FineTunePlan:
    """Train the complete encoder and segmentation system, excluding unused leaves."""
    if not isinstance(model, EoMT):
        raise TypeError("Expected an EoMT model.")

    def unused(path, leaf):
        if not eqx.is_inexact_array(leaf):
            return False
        return path[:2] in {("backbone", "mask_token"), ("backbone", "local_cls_norm")}

    return prepare_finetune(
        model,
        trainable=TrainableSpec(
            mode="full",
            freeze=TargetSpec(predicate=unused, allow_empty=True),
        ),
        labels=LLRDConfig(decay=decay),
    )


__all__ = ["eomt_full_finetune"]
