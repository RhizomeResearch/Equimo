"""Vision fine-tuning dense-feature utilities."""

from . import dense
from .dense import DenseVisionConfig, dense_distillation_config, dense_feature_adapter

__all__ = (
    "DenseVisionConfig",
    "dense",
    "dense_distillation_config",
    "dense_feature_adapter",
)
