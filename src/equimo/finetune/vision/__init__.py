"""Vision fine-tuning probes and dense-feature utilities."""

from . import dense
from .eomt import eomt_full_finetune
from .pmt import pmt_head_finetune
from .dense import (
    DenseProbe,
    DenseVisionConfig,
    dense_distillation_config,
    dense_feature_adapter,
    make_dense_probe,
)

__all__ = (
    "DenseProbe",
    "DenseVisionConfig",
    "dense",
    "dense_distillation_config",
    "dense_feature_adapter",
    "make_dense_probe",
    "eomt_full_finetune",
    "pmt_head_finetune",
)
