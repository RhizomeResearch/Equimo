"""Dense vision fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import PyTree
from ..config import FeatureSpec
from ..feature_extraction import (
    FeatureResult,
    _headless_backbone,
    _keeps_separate_levels,
    _linear_probe_head,
    extract_features,
)
from ..heads import ActivationName, DenseFeatureAdapter, LinearHead
from ..regularization import FeatureDistillationConfig


@dataclass(frozen=True)
class DenseVisionConfig:
    """Configuration for dense vision transfer helpers."""

    activation: ActivationName = "identity"
    dropout: float = 0.0
    bias: bool = True
    distillation: FeatureDistillationConfig = field(
        default_factory=FeatureDistillationConfig.dense
    )


class DenseProbe(eqx.Module):
    """Backbone plus a pointwise linear head returning class-first patch logits.

    One call consumes one example and returns ``(classes, grid_h, grid_w)``.
    Apply :func:`jax.vmap` to the probe for batched ``(batch, classes, grid_h,
    grid_w)`` output.
    """

    backbone: PyTree
    head: LinearHead
    feature_spec: FeatureSpec = eqx.field(static=True)

    def __init__(
        self,
        backbone: PyTree,
        head: LinearHead,
        *,
        feature_spec: FeatureSpec,
    ):
        _validate_dense_feature_spec(feature_spec)
        self.backbone = backbone
        self.head = head
        self.feature_spec = feature_spec

    def __call__(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ) -> jax.Array:
        result = extract_features(
            self.backbone,
            *args,
            feature_spec=self.feature_spec,
            key=key,
            inference=inference,
            **kwargs,
        )
        if not isinstance(result, FeatureResult):
            raise ValueError("DenseProbe requires feature metadata from FeatureSpec.")
        if len(result.levels) != 1 or isinstance(result.features, tuple):
            raise ValueError("DenseProbe requires exactly one feature level.")

        features = result.features
        if not eqx.is_array(features):
            raise ValueError("DenseProbe requires an array feature level.")
        metadata = result.levels[0]
        grid_size = metadata.grid_size
        if grid_size is None:
            raise ValueError("DenseProbe requires explicit patch-grid metadata.")
        grid_h, grid_w = grid_size
        expected_in = int(self.head.linear.in_features)
        if metadata.feature_width != expected_in:
            raise ValueError(
                "DenseProbe feature-width mismatch: metadata declares "
                f"{metadata.feature_width}, head expects {expected_in}."
            )

        if self.feature_spec.output_layout == "BNC":
            if features.ndim != 2:
                raise ValueError(
                    "DenseProbe BNC features must be unbatched and shaped (N, C); "
                    "use jax.vmap for batching."
                )
            if features.shape != (grid_h * grid_w, expected_in):
                raise ValueError(
                    "DenseProbe token features disagree with the declared grid "
                    f"or width: got {features.shape}, expected "
                    f"({grid_h * grid_w}, {expected_in})."
                )
            logits = self.head(features).reshape(grid_h, grid_w, -1)
            return jnp.moveaxis(logits, -1, 0)

        if features.ndim != 3:
            raise ValueError(
                "DenseProbe BCHW features must be unbatched and shaped (C, H, W); "
                "use jax.vmap for batching."
            )
        if features.shape != (expected_in, grid_h, grid_w):
            raise ValueError(
                "DenseProbe spatial features disagree with the declared grid or "
                f"width: got {features.shape}, expected "
                f"({expected_in}, {grid_h}, {grid_w})."
            )
        logits = self.head(jnp.moveaxis(features, 0, -1))
        return jnp.moveaxis(logits, -1, 0)


def make_dense_probe(
    backbone: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    feature_spec: FeatureSpec,
    head: LinearHead | None = None,
) -> DenseProbe:
    """Build a pointwise linear probe with an identity backbone head."""

    _validate_dense_feature_spec(feature_spec)
    probe_head = _linear_probe_head(
        head,
        in_features=in_features,
        out_features=out_features,
        key=key,
        context="make_dense_probe",
    )
    return DenseProbe(
        _headless_backbone(backbone), probe_head, feature_spec=feature_spec
    )


def _validate_dense_feature_spec(feature_spec: FeatureSpec) -> None:
    pooling = "none" if feature_spec.pooling is None else feature_spec.pooling
    if pooling != "none":
        raise ValueError("DenseProbe requires an unpooled FeatureSpec.")
    if not feature_spec.return_metadata:
        raise ValueError("DenseProbe requires FeatureSpec.return_metadata=True.")
    if _keeps_separate_levels(feature_spec):
        raise ValueError("DenseProbe does not accept separate feature levels.")
    valid_tokens = (
        feature_spec.output_layout == "BNC"
        and feature_spec.token_selection == "patches"
    )
    valid_map = (
        feature_spec.output_layout == "BCHW" and feature_spec.token_selection == "all"
    )
    if not (valid_tokens or valid_map):
        raise ValueError(
            "DenseProbe FeatureSpec must select BNC patches or all BCHW features."
        )


def dense_feature_adapter(
    in_features: int,
    out_features: int,
    *,
    key: jax.Array,
    config: DenseVisionConfig | None = None,
) -> DenseFeatureAdapter:
    """Create a dense feature adapter for spatial or token features."""

    config = DenseVisionConfig() if config is None else config
    return DenseFeatureAdapter(
        in_features,
        out_features,
        key=key,
        activation=config.activation,
        dropout=config.dropout,
        bias=config.bias,
    )


def dense_distillation_config(
    *,
    layers: tuple[str, ...] = ("25%", "50%", "75%", "100%"),
    normalize_features: bool = True,
) -> FeatureDistillationConfig:
    """Return the dense-task feature distillation preset."""

    return FeatureDistillationConfig.dense(
        layers=layers,
        normalize_features=normalize_features,
    )


__all__ = (
    "DenseProbe",
    "DenseVisionConfig",
    "dense_distillation_config",
    "dense_feature_adapter",
    "make_dense_probe",
)
