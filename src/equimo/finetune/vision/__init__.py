"""Vision fine-tuning probes and dense-feature utilities."""

from . import dense
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
)
