"""Optimizer-independent, JAX-native query-segmentation training utilities."""

from ._assignment import QueryAssignment, match_queries
from ._sampling import PointSamples, sample_mask_points, sample_query_points
from ._loss import (
    LossTerm,
    MatchingCosts,
    PredictionLoss,
    PredictionLossContext,
    QueryLossConfig,
    QueryLossContext,
    QueryLossResult,
    QueryTargets,
    prepare_query_loss,
    query_matching_costs,
    query_prediction_loss,
    query_segmentation_loss,
    reduce_query_losses,
)

__all__ = [
    "LossTerm",
    "MatchingCosts",
    "PointSamples",
    "PredictionLoss",
    "PredictionLossContext",
    "QueryAssignment",
    "QueryLossConfig",
    "QueryLossContext",
    "QueryLossResult",
    "QueryTargets",
    "match_queries",
    "prepare_query_loss",
    "query_matching_costs",
    "query_prediction_loss",
    "query_segmentation_loss",
    "reduce_query_losses",
    "sample_mask_points",
    "sample_query_points",
]
