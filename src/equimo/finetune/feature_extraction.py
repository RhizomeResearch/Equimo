"""Feature-extraction wrappers for fine-tuning workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ._typing import PyTree
from .config import FeatureSpec
from .heads import AttentionPoolingClassifierHead, IdentityHead, LinearHead
from .pooling import (
    CLSPatchMeanPool,
    GlobalAveragePool,
    MeanPatchPool,
    MeanTokenPool,
    PoolName,
    _NativeReadoutPool,
    pool_features,
)
from .surgery import replace_head


@dataclass(frozen=True)
class _FeaturePolicy:
    """Validated, layout-resolved operations compiled from ``FeatureSpec``."""

    spec: FeatureSpec
    declared_rank: int
    feature_axis: int
    sequence_axes: tuple[int, ...]

    def axes_for(self, ndim: int) -> tuple[int, tuple[int, ...]]:
        if ndim == self.declared_rank:
            return self.feature_axis, self.sequence_axes
        if ndim == self.declared_rank - 1:
            return self.feature_axis - 1, tuple(axis - 1 for axis in self.sequence_axes)
        expected = (self.declared_rank - 1, self.declared_rank)
        raise ValueError(
            f"FeatureSpec output_layout={self.spec.output_layout!r} expects an "
            f"unbatched/batched rank in {expected}, got rank {ndim}."
        )


class FeatureExtractor(eqx.Module):
    """Wrap a backbone and return pooled features."""

    model: PyTree
    pool: PoolName | eqx.Module | None = eqx.field(static=True)
    feature_spec: FeatureSpec | None = eqx.field(static=True)

    def __init__(
        self,
        model: PyTree,
        *,
        pool: PoolName | eqx.Module | None = "auto",
        feature_spec: FeatureSpec | None = None,
    ):
        self.model = model
        self.pool = pool
        self.feature_spec = feature_spec

    def __call__(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ):
        return extract_features(
            self.model,
            *args,
            pool=self.pool,
            feature_spec=self.feature_spec,
            key=key,
            inference=inference,
            **kwargs,
        )


class LinearProbe(eqx.Module):
    """Backbone feature extractor plus a trainable task head."""

    backbone: PyTree
    head: eqx.Module
    pool: PoolName | eqx.Module | None = eqx.field(static=True)
    feature_spec: FeatureSpec | None = eqx.field(static=True)

    def __init__(
        self,
        backbone: PyTree,
        head: eqx.Module,
        *,
        pool: PoolName | eqx.Module | None = "auto",
        feature_spec: FeatureSpec | None = None,
    ):
        self.backbone = backbone
        self.head = head
        self.pool = pool
        self.feature_spec = feature_spec

    def __call__(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ):
        features = extract_features(
            self.backbone,
            *args,
            pool=self.pool,
            feature_spec=self.feature_spec,
            key=key,
            inference=inference,
            **kwargs,
        )
        return _call_head(self.head, features, key=key, inference=inference)


class AttentionPoolingProbe(eqx.Module):
    """Backbone plus a FINO-style attention-pooling classifier head."""

    backbone: PyTree
    head: AttentionPoolingClassifierHead

    n_last_blocks: int | None = eqx.field(static=True)
    prepend_cls_token: bool = eqx.field(static=True)
    l2_normalize_cls: bool = eqx.field(static=True)

    def __init__(
        self,
        backbone: PyTree,
        head: AttentionPoolingClassifierHead,
        *,
        n_last_blocks: int | None = None,
        prepend_cls_token: bool = False,
        l2_normalize_cls: bool = False,
    ):
        self.backbone = backbone
        self.head = head
        self.n_last_blocks = n_last_blocks
        self.prepend_cls_token = prepend_cls_token
        self.l2_normalize_cls = l2_normalize_cls

    def __call__(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        mask: jax.Array | None = None,
        **kwargs,
    ) -> jax.Array:
        key_backbone, key_head = _split_optional_key(key)
        if self.n_last_blocks is None:
            features = _call_forward_features(
                self.backbone,
                *args,
                key=key_backbone,
                inference=inference,
                **kwargs,
            )
            tokens = make_attention_pool_input_from_forward_features(
                features,
                prepend_cls_token=self.prepend_cls_token,
                l2_normalize_cls=self.l2_normalize_cls,
            )
        else:
            if not hasattr(self.backbone, "intermediate_features"):
                raise ValueError(
                    "AttentionPoolingProbe with n_last_blocks requires a backbone "
                    "with intermediate_features(...)."
                )
            intermediates = _call_with_optional_key(
                self.backbone.intermediate_features,
                *args,
                key=key_backbone,
                inference=inference,
                n_last_blocks=self.n_last_blocks,
                **kwargs,
            )
            tokens = make_attention_pool_input_from_intermediates(
                intermediates,
                self.n_last_blocks,
                norm=getattr(self.backbone, "norm", None),
                num_prefix_tokens=_model_prefix_count(self.backbone),
                has_cls_token=_has_cls_token(self.backbone),
                prepend_cls_token=self.prepend_cls_token,
                l2_normalize_cls=self.l2_normalize_cls,
            )

        return self.head(tokens, mask=mask, key=key_head, inference=inference)


def extract_features(
    model: PyTree,
    *args,
    pool: PoolName | eqx.Module | None = "auto",
    feature_spec: FeatureSpec | None = None,
    observed_preprocessing_fingerprint: str | None = None,
    key: jax.Array | None = None,
    inference: bool | None = True,
    **kwargs,
) -> Any:
    """Extract features through an explicit spec or the compatibility fallback.

    A supplied ``feature_spec`` controls endpoint traversal and tensor
    post-processing. Without one, the native/heuristic behavior is unchanged.
    """

    if feature_spec is not None:
        _validate_explicit_pool(pool, feature_spec)
        policy = _compile_feature_spec(feature_spec)
        return _extract_with_policy(
            model,
            policy,
            args,
            kwargs,
            key=key,
            inference=inference,
            observed_preprocessing_fingerprint=observed_preprocessing_fingerprint,
        )

    unpooled = pool is None or (isinstance(pool, str) and pool == "none")
    if not unpooled and hasattr(model, "forward_features"):
        features = _call_with_optional_key(
            model.forward_features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    elif hasattr(model, "features"):
        features = _call_with_optional_key(
            model.features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    elif hasattr(model, "forward_features"):
        features = _call_with_optional_key(
            model.forward_features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    else:
        features = _call_with_optional_key(
            model,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )

    pool = _resolve_pool(model, features, _prompt_aware_pool(model, pool))
    pool_kwargs = _pool_kwargs(model, args, kwargs)
    return pool_features(features, pool, key=key, **pool_kwargs)


def _compile_feature_spec(spec: FeatureSpec) -> _FeaturePolicy:
    layouts = {
        "BNC": (3, 2, (1,)),
        "BCHW": (4, 1, (2, 3)),
        "BTC": (3, 2, (1,)),
        "BCT": (3, 1, (2,)),
        "BC": (2, 1, ()),
    }
    declared_rank, feature_axis, sequence_axes = layouts[spec.output_layout]
    return _FeaturePolicy(spec, declared_rank, feature_axis, sequence_axes)


def _validate_explicit_pool(
    pool: PoolName | eqx.Module | None,
    spec: FeatureSpec,
) -> None:
    if pool == "auto":
        return
    if isinstance(pool, eqx.Module):
        raise ValueError(
            "An explicit FeatureSpec cannot be combined with a pool module; "
            "declare FeatureSpec.pooling instead."
        )
    supplied = "none" if pool is None else pool
    declared = "none" if spec.pooling is None else spec.pooling
    if supplied != declared:
        raise ValueError(
            f"Explicit pool={pool!r} contradicts FeatureSpec.pooling={spec.pooling!r}."
        )


def _extract_with_policy(
    model: PyTree,
    policy: _FeaturePolicy,
    args: tuple,
    kwargs: dict,
    *,
    key: jax.Array | None,
    inference: bool | None,
    observed_preprocessing_fingerprint: str | None,
) -> jax.Array:
    _check_preprocessing_fingerprint(
        model,
        policy.spec.preprocessing_fingerprint,
        observed_preprocessing_fingerprint,
    )
    endpoint = _resolve_feature_endpoint(model, policy.spec.endpoint)
    mask = _resolve_spec_mask(endpoint, args, kwargs, policy.spec.mask_field)
    features = _call_with_optional_key(
        endpoint,
        *args,
        key=key,
        inference=inference,
        **kwargs,
    )
    features = _aggregate_feature_layers(features, policy)
    return _apply_feature_policy(model, features, mask, policy, key=key)


def _check_preprocessing_fingerprint(
    model: PyTree,
    expected: str | None,
    observed: str | None,
) -> None:
    if expected is None:
        return
    if observed is None:
        observed = getattr(model, "preprocessing_fingerprint", None)
    if observed is None:
        raise ValueError(
            "FeatureSpec requires preprocessing_fingerprint "
            f"{expected!r}, but no observed fingerprint was supplied."
        )
    if observed != expected:
        raise ValueError(
            "FeatureSpec preprocessing fingerprint mismatch: "
            f"expected {expected!r}, got {observed!r}."
        )


def _resolve_feature_endpoint(model: PyTree, endpoint: str):
    if endpoint == "__call__":
        return model
    current = model
    for component in endpoint.split("."):
        if component.isdigit():
            try:
                current = current[int(component)]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"FeatureSpec endpoint {endpoint!r} cannot resolve index "
                    f"{component!r}."
                ) from error
        else:
            if not hasattr(current, component):
                raise ValueError(
                    f"FeatureSpec endpoint {endpoint!r} is missing component "
                    f"{component!r} on {type(current).__name__}."
                )
            current = getattr(current, component)
    if not callable(current):
        raise ValueError(f"FeatureSpec endpoint {endpoint!r} is not callable.")
    return current


def _resolve_spec_mask(endpoint, args: tuple, kwargs: dict, field: str | None):
    if field is None:
        return None
    try:
        bound = inspect.signature(endpoint).bind_partial(*args, **kwargs)
    except TypeError as error:
        raise ValueError(
            f"FeatureSpec mask_field={field!r} could not be bound to the endpoint."
        ) from error
    if field not in bound.arguments:
        raise ValueError(
            f"FeatureSpec mask_field={field!r} was not supplied to the endpoint."
        )
    mask = bound.arguments[field]
    if not eqx.is_array(mask):
        raise ValueError(f"FeatureSpec mask_field={field!r} must be an array.")
    return mask


def _aggregate_feature_layers(features: Any, policy: _FeaturePolicy) -> Any:
    aggregation = policy.spec.layer_aggregation
    if aggregation is None:
        if isinstance(features, (tuple, list)):
            raise ValueError(
                "Feature endpoint returned multiple layers but FeatureSpec has no "
                "layer_aggregation."
            )
        return features
    if not isinstance(features, (tuple, list)) or not features:
        raise ValueError(
            "FeatureSpec.layer_aggregation requires a non-empty tuple/list endpoint."
        )
    if not all(eqx.is_array(layer) for layer in features):
        raise ValueError("FeatureSpec layer aggregation only supports array layers.")

    layers = tuple(features)
    first = layers[0]
    feature_axis, _ = policy.axes_for(first.ndim)
    if any(layer.ndim != first.ndim for layer in layers[1:]):
        raise ValueError("FeatureSpec layer aggregation requires equal layer ranks.")
    method = aggregation["method"]
    if method == "last":
        return layers[-1]
    if method == "mean":
        if any(layer.shape != first.shape for layer in layers[1:]):
            raise ValueError(
                "FeatureSpec layer_aggregation='mean' requires equal shapes."
            )
        return jnp.mean(jnp.stack(layers, axis=0), axis=0)
    reference = first.shape[:feature_axis] + first.shape[feature_axis + 1 :]
    for layer in layers[1:]:
        shape = layer.shape[:feature_axis] + layer.shape[feature_axis + 1 :]
        if shape != reference:
            raise ValueError(
                "FeatureSpec layer_aggregation='concat' requires non-feature "
                "dimensions to match."
            )
    return jnp.concatenate(layers, axis=feature_axis)


def _apply_feature_policy(
    model: PyTree,
    features: Any,
    mask: jax.Array | None,
    policy: _FeaturePolicy,
    *,
    key: jax.Array | None,
) -> jax.Array:
    pooling = "none" if policy.spec.pooling is None else policy.spec.pooling
    if pooling == "native":
        carrier = (
            _feature_dict_carrier(features) if isinstance(features, dict) else features
        )
        if not eqx.is_array(carrier):
            raise ValueError(
                "FeatureSpec native endpoint did not return array features."
            )
        feature_axis, sequence_axes = policy.axes_for(carrier.ndim)
        result = _native_readout(
            model,
            features,
            feature_axis,
            sequence_axes,
            exclude_prompt_tokens=policy.spec.exclude_prompt_tokens,
        )
        return _normalize_features(result, result.ndim - 1, policy.spec.normalize)

    selected_already = False
    carrier = features
    if isinstance(features, dict):
        carrier = _feature_dict_carrier(features)
        features, selected_already = _select_feature_dict(
            features, policy.spec.token_selection
        )
    if not eqx.is_array(features) or not eqx.is_array(carrier):
        raise ValueError(
            "FeatureSpec endpoint must resolve to an array or feature dict."
        )

    feature_axis, sequence_axes = policy.axes_for(carrier.ndim)
    if selected_already and features.ndim == carrier.ndim - 1:
        feature_axis = features.ndim - 1
        sequence_axes = ()
    elif features.ndim != carrier.ndim:
        raise ValueError("Feature dictionary values do not match the declared layout.")

    if mask is not None:
        _validate_mask_shape(mask, carrier, policy.axes_for(carrier.ndim)[0])
    if not selected_already:
        features, feature_axis, sequence_axes = _select_declared_tokens(
            model,
            features,
            mask,
            policy.spec.token_selection,
            feature_axis,
            sequence_axes,
            exclude_prompt_tokens=policy.spec.exclude_prompt_tokens,
        )
    result, feature_axis = _pool_declared_features(
        model,
        features,
        mask,
        pooling,
        feature_axis,
        sequence_axes,
        tokens_are_patches=policy.spec.token_selection == "patches",
        exclude_prompt_tokens=policy.spec.exclude_prompt_tokens,
        key=key,
    )
    return _normalize_features(result, feature_axis, policy.spec.normalize)


def _feature_dict_carrier(features: dict[str, Any]) -> jax.Array:
    for name in (
        "x_norm_patchtokens",
        "last_hidden_state",
        "x_prenorm",
        "x_norm_test",
        "x_norm_train",
    ):
        value = features.get(name)
        if eqx.is_array(value):
            return cast(jax.Array, value)
    raise ValueError(
        "FeatureSpec endpoint returned a dict without a known feature array."
    )


def _select_feature_dict(
    features: dict[str, Any],
    selection: str,
) -> tuple[jax.Array, bool]:
    if selection == "cls":
        value = features.get("x_norm_cls_token")
        if not eqx.is_array(value):
            raise ValueError(
                "FeatureSpec token_selection='cls' requires x_norm_cls_token."
            )
        return cast(jax.Array, value), True
    if selection in {"patches", "frames"}:
        value = features.get("x_norm_patchtokens")
        if not eqx.is_array(value):
            raise ValueError(
                f"FeatureSpec token_selection={selection!r} requires "
                "x_norm_patchtokens."
            )
        return cast(jax.Array, value), True
    if selection == "last_valid":
        for name in ("last_hidden_state", "x_prenorm"):
            value = features.get(name)
            if eqx.is_array(value):
                return cast(jax.Array, value), False
        raise ValueError(
            "FeatureSpec token_selection='last_valid' requires token features."
        )
    for name in ("last_hidden_state", "x_prenorm", "x_norm_test"):
        value = features.get(name)
        if eqx.is_array(value):
            return cast(jax.Array, value), False
    return _feature_dict_carrier(features), False


def _validate_mask_shape(
    mask: jax.Array, features: jax.Array, feature_axis: int
) -> None:
    expected = tuple(
        size for axis, size in enumerate(features.shape) if axis != feature_axis
    )
    if mask.shape != expected:
        raise ValueError(
            "FeatureSpec mask shape must match every non-feature axis; "
            f"expected {expected}, got {mask.shape}."
        )


def _select_declared_tokens(
    model: PyTree,
    features: jax.Array,
    mask: jax.Array | None,
    selection: str,
    feature_axis: int,
    sequence_axes: tuple[int, ...],
    *,
    exclude_prompt_tokens: bool,
) -> tuple[jax.Array, int, tuple[int, ...]]:
    if selection in {"all", "frames"}:
        return features, feature_axis, sequence_axes
    if len(sequence_axes) != 1:
        raise ValueError(
            f"FeatureSpec token_selection={selection!r} requires one sequence axis."
        )
    sequence_axis = sequence_axes[0]
    if selection == "cls":
        index = _explicit_cls_index(model)
        result = jnp.take(features, index, axis=sequence_axis)
        return result, _axis_after_take(feature_axis, sequence_axis), ()
    if selection == "patches":
        result = _explicit_patch_tokens(
            model,
            features,
            sequence_axis,
            exclude_prompt_tokens=exclude_prompt_tokens,
        )
        return result, feature_axis, sequence_axes
    if selection == "last_valid":
        if mask is None:
            raise ValueError(
                "FeatureSpec token_selection='last_valid' requires mask_field."
            )
        result = _last_valid_token(features, mask, sequence_axis, feature_axis)
        return result, result.ndim - 1, ()
    raise AssertionError(f"Uncompiled FeatureSpec token selection {selection!r}.")


def _pool_declared_features(
    model: PyTree,
    features: jax.Array,
    mask: jax.Array | None,
    pooling: str,
    feature_axis: int,
    sequence_axes: tuple[int, ...],
    *,
    tokens_are_patches: bool,
    exclude_prompt_tokens: bool,
    key: jax.Array | None,
) -> tuple[jax.Array, int]:
    if pooling == "none":
        return features, feature_axis
    if not sequence_axes:
        raise ValueError(f"FeatureSpec pooling={pooling!r} requires sequence axes.")

    if pooling == "cls":
        if len(sequence_axes) != 1:
            raise ValueError("FeatureSpec pooling='cls' requires one sequence axis.")
        sequence_axis = sequence_axes[0]
        result = jnp.take(features, _explicit_cls_index(model), axis=sequence_axis)
        return result, _axis_after_take(feature_axis, sequence_axis)
    if pooling == "cls_patch_mean":
        if len(sequence_axes) != 1:
            raise ValueError(
                "FeatureSpec pooling='cls_patch_mean' requires one sequence axis."
            )
        sequence_axis = sequence_axes[0]
        cls = jnp.take(features, _explicit_cls_index(model), axis=sequence_axis)
        patches = (
            features
            if tokens_are_patches
            else _explicit_patch_tokens(
                model,
                features,
                sequence_axis,
                exclude_prompt_tokens=exclude_prompt_tokens,
            )
        )
        mean = jnp.mean(patches, axis=sequence_axis)
        result_feature_axis = _axis_after_take(feature_axis, sequence_axis)
        return jnp.concatenate(
            [cls, mean], axis=result_feature_axis
        ), result_feature_axis
    if pooling == "mean_patch":
        if len(sequence_axes) != 1:
            raise ValueError(
                "FeatureSpec pooling='mean_patch' requires one sequence axis."
            )
        sequence_axis = sequence_axes[0]
        patches = (
            features
            if tokens_are_patches
            else _explicit_patch_tokens(
                model,
                features,
                sequence_axis,
                exclude_prompt_tokens=exclude_prompt_tokens,
            )
        )
        return jnp.mean(patches, axis=sequence_axis), _axis_after_reduction(
            feature_axis, sequence_axes
        )
    if pooling in {"mean_token", "mean_frame"}:
        if mask is not None:
            if len(sequence_axes) != 1:
                raise ValueError("Masked mean pooling requires one sequence axis.")
            result = _masked_mean(features, mask, sequence_axes[0], feature_axis)
        else:
            result = jnp.mean(features, axis=sequence_axes)
        return result, _axis_after_reduction(feature_axis, sequence_axes)
    if pooling == "global_avg":
        return jnp.mean(features, axis=sequence_axes), _axis_after_reduction(
            feature_axis, sequence_axes
        )
    if pooling == "gem":
        clipped = jnp.clip(features, min=1e-6)
        result = jnp.mean(clipped**3.0, axis=sequence_axes) ** (1.0 / 3.0)
        return result, _axis_after_reduction(feature_axis, sequence_axes)
    if pooling == "last_token":
        if len(sequence_axes) != 1:
            raise ValueError(
                "FeatureSpec pooling='last_token' requires one sequence axis."
            )
        if mask is None:
            result = jnp.take(features, -1, axis=sequence_axes[0])
        else:
            result = _last_valid_token(features, mask, sequence_axes[0], feature_axis)
        return result, result.ndim - 1
    if pooling == "attention":
        if key is None:
            raise ValueError("FeatureSpec pooling='attention' requires a PRNG key.")
        if len(sequence_axes) != 1:
            raise ValueError(
                "FeatureSpec attention pooling requires one sequence axis."
            )
        result = _attention_pool_explicit(
            features,
            mask,
            sequence_axes[0],
            feature_axis,
            key,
        )
        return result, result.ndim - 1
    raise AssertionError(f"Uncompiled FeatureSpec pooling {pooling!r}.")


def _native_readout(
    model: PyTree,
    features: jax.Array | dict[str, Any],
    feature_axis: int,
    sequence_axes: tuple[int, ...],
    *,
    exclude_prompt_tokens: bool,
) -> jax.Array:
    pool_type = getattr(model, "global_pool", None)
    if pool_type is None:
        raise ValueError(
            "FeatureSpec pooling='native' requires model.global_pool metadata."
        )
    pool_type = _normalize_native_pool_type(pool_type)
    if isinstance(features, dict):
        feature_dict = cast(dict[str, Any], features)
        cls_value = feature_dict.get("x_norm_cls_token")
        dist_value = feature_dict.get("x_norm_dist_token")
        patches_value = feature_dict.get("x_norm_patchtokens")
        if pool_type == "token":
            if not eqx.is_array(cls_value):
                raise ValueError("Native token pooling requires x_norm_cls_token.")
            cls = cast(jax.Array, cls_value)
            if eqx.is_array(dist_value):
                return 0.5 * (cls + cast(jax.Array, dist_value))
            return cls
        if not eqx.is_array(patches_value):
            raise ValueError("Native aggregate pooling requires x_norm_patchtokens.")
        patches = cast(jax.Array, patches_value)
        if pool_type == "cls_patch_mean":
            if not eqx.is_array(cls_value):
                raise ValueError("Native CLS-patch pooling requires x_norm_cls_token.")
            cls = cast(jax.Array, cls_value)
            return jnp.concatenate([cls, jnp.mean(patches, axis=-2)], axis=-1)
        if pool_type == "avg":
            return jnp.mean(patches, axis=-2)
        if pool_type == "avgmax":
            return 0.5 * (jnp.mean(patches, axis=-2) + jnp.max(patches, axis=-2))
        if pool_type == "max":
            return jnp.max(patches, axis=-2)
        raise ValueError(f"Unsupported native pool type {pool_type!r}.")

    if len(sequence_axes) != 1:
        raise ValueError("Native transformer pooling requires one sequence axis.")
    sequence_axis = sequence_axes[0]
    cls = jnp.take(features, _explicit_cls_index(model), axis=sequence_axis)
    if pool_type == "token":
        if getattr(model, "dist_token", None) is not None:
            dist = jnp.take(
                features, _explicit_cls_index(model) + 1, axis=sequence_axis
            )
            return 0.5 * (cls + dist)
        return cls
    patches = _explicit_patch_tokens(
        model,
        features,
        sequence_axis,
        exclude_prompt_tokens=exclude_prompt_tokens,
    )
    if pool_type == "cls_patch_mean":
        result_axis = _axis_after_take(feature_axis, sequence_axis)
        return jnp.concatenate(
            [cls, jnp.mean(patches, axis=sequence_axis)], axis=result_axis
        )
    if pool_type == "avg":
        return jnp.mean(patches, axis=sequence_axis)
    if pool_type == "avgmax":
        return 0.5 * (
            jnp.mean(patches, axis=sequence_axis) + jnp.max(patches, axis=sequence_axis)
        )
    if pool_type == "max":
        return jnp.max(patches, axis=sequence_axis)
    raise ValueError(f"Unsupported native pool type {pool_type!r}.")


def _masked_mean(
    features: jax.Array,
    padding_mask: jax.Array,
    sequence_axis: int,
    feature_axis: int,
) -> jax.Array:
    valid = jnp.logical_not(padding_mask.astype(bool))
    weights = jnp.expand_dims(valid.astype(features.dtype), axis=feature_axis)
    total = jnp.sum(weights, axis=sequence_axis)
    summed = jnp.sum(features * weights, axis=sequence_axis)
    return jnp.where(total > 0, summed / jnp.maximum(total, 1), 0)


def _last_valid_token(
    features: jax.Array,
    padding_mask: jax.Array,
    sequence_axis: int,
    feature_axis: int,
) -> jax.Array:
    tokens = jnp.moveaxis(features, (sequence_axis, feature_axis), (-2, -1))
    valid = jnp.logical_not(padding_mask.astype(bool))
    positions = jnp.arange(valid.shape[-1], dtype=jnp.int32)
    index = jnp.max(jnp.where(valid, positions, 0), axis=-1)
    selected = jnp.take_along_axis(tokens, index[..., None, None], axis=-2)[..., 0, :]
    return jnp.where(jnp.any(valid, axis=-1)[..., None], selected, 0)


def _attention_pool_explicit(
    features: jax.Array,
    padding_mask: jax.Array | None,
    sequence_axis: int,
    feature_axis: int,
    key: jax.Array,
) -> jax.Array:
    tokens = jnp.moveaxis(features, (sequence_axis, feature_axis), (-2, -1))
    query = jr.normal(key, (tokens.shape[-1],), dtype=tokens.dtype) / jnp.sqrt(
        tokens.shape[-1]
    )
    scores = tokens @ query
    if padding_mask is not None:
        scores = jnp.where(padding_mask.astype(bool), -jnp.inf, scores)
    weights = jnp.nan_to_num(jax.nn.softmax(scores, axis=-1))
    return jnp.sum(tokens * weights[..., :, None], axis=-2)


def _normalize_features(
    features: jax.Array,
    feature_axis: int,
    normalization: str,
) -> jax.Array:
    if normalization == "none":
        return features
    dtype = features.dtype
    working = features.astype(jnp.float32)
    if normalization == "l2":
        scale = jnp.linalg.norm(working, axis=feature_axis, keepdims=True)
        normalized = working / jnp.maximum(scale, 1e-12)
    else:
        mean = jnp.mean(working, axis=feature_axis, keepdims=True)
        variance = jnp.mean((working - mean) ** 2, axis=feature_axis, keepdims=True)
        normalized = (working - mean) / jnp.sqrt(variance + 1e-6)
    return normalized.astype(dtype)


def _explicit_patch_tokens(
    model: PyTree,
    features: jax.Array,
    sequence_axis: int,
    *,
    exclude_prompt_tokens: bool,
) -> jax.Array:
    base_prefix = _model_prefix_count(model)
    prompt_count = int(getattr(model, "num_prompt_tokens", 0))
    if prompt_count == 0:
        return jax.lax.slice_in_dim(features, base_prefix, None, axis=sequence_axis)

    prepend_to = getattr(getattr(model, "config", None), "prepend_to", "after_cls")
    if prepend_to in {"before_all", "input"}:
        prompt_start = 0
        patch_start = prompt_count + base_prefix
    else:
        prompt_start = 1
        patch_start = prompt_count + base_prefix
    patches = jax.lax.slice_in_dim(features, patch_start, None, axis=sequence_axis)
    if exclude_prompt_tokens:
        return patches
    prompts = jax.lax.slice_in_dim(
        features,
        prompt_start,
        prompt_start + prompt_count,
        axis=sequence_axis,
    )
    return jnp.concatenate([prompts, patches], axis=sequence_axis)


def _explicit_cls_index(model: PyTree) -> int:
    prompt_count = int(getattr(model, "num_prompt_tokens", 0))
    prepend_to = getattr(getattr(model, "config", None), "prepend_to", "after_cls")
    return prompt_count if prepend_to in {"before_all", "input"} else 0


def _axis_after_take(feature_axis: int, removed_axis: int) -> int:
    return feature_axis - 1 if removed_axis < feature_axis else feature_axis


def _axis_after_reduction(feature_axis: int, reduced_axes: tuple[int, ...]) -> int:
    return feature_axis - sum(axis < feature_axis for axis in reduced_axes)


def make_attention_pool_input_from_forward_features(
    features: dict[str, jax.Array | None],
    *,
    prepend_cls_token: bool = False,
    l2_normalize_cls: bool = False,
    eps: float = 1e-12,
) -> jax.Array:
    """Build attention-pool tokens from a normalized feature dictionary."""

    if not isinstance(features, dict):
        raise ValueError(
            "make_attention_pool_input_from_forward_features expects a feature dict."
        )
    patch_tokens = features.get("x_norm_patchtokens")
    if patch_tokens is None:
        raise ValueError("feature dict must contain x_norm_patchtokens.")
    if not prepend_cls_token:
        return patch_tokens.astype(jnp.float32)

    cls_token = features.get("x_norm_cls_token")
    if cls_token is None:
        raise ValueError("prepend_cls_token=True requires x_norm_cls_token.")
    cls_token = _maybe_l2_normalize(cls_token, l2_normalize_cls, eps)
    return jnp.concatenate([cls_token[None, :], patch_tokens], axis=0).astype(
        jnp.float32
    )


def make_attention_pool_input_from_intermediates(
    intermediates,
    n_last_blocks: int,
    *,
    norm: eqx.Module | None = None,
    num_prefix_tokens: int = 0,
    has_cls_token: bool = False,
    prepend_cls_token: bool = False,
    l2_normalize_cls: bool = False,
    eps: float = 1e-12,
) -> jax.Array:
    """Concatenate final intermediate token features along the feature axis."""

    intermediates = tuple(intermediates)
    if n_last_blocks < 1:
        raise ValueError("n_last_blocks must be >= 1.")
    if n_last_blocks > len(intermediates):
        raise ValueError(
            f"n_last_blocks={n_last_blocks} exceeds available "
            f"intermediates={len(intermediates)}."
        )

    selected = intermediates[-n_last_blocks:]
    pairs = tuple(
        _as_patch_cls_tokens(
            item,
            norm=norm,
            num_prefix_tokens=num_prefix_tokens,
            has_cls_token=has_cls_token,
        )
        for item in selected
    )
    patch_tokens = jnp.concatenate([patch for patch, _ in pairs], axis=-1)

    if not prepend_cls_token:
        return patch_tokens.astype(jnp.float32)

    cls_tokens = [cls for _, cls in pairs]
    if any(cls is None for cls in cls_tokens):
        raise ValueError("prepend_cls_token=True requires class tokens.")
    present_cls_tokens = [cls for cls in cls_tokens if cls is not None]
    cls_token = jnp.concatenate(present_cls_tokens, axis=-1)
    cls_token = _maybe_l2_normalize(cls_token, l2_normalize_cls, eps)
    return jnp.concatenate([cls_token[None, :], patch_tokens], axis=0).astype(
        jnp.float32
    )


def make_linear_probe(
    backbone: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    pool: PoolName | eqx.Module | None = "auto",
    feature_spec: FeatureSpec | None = None,
    head: eqx.Module | None = None,
) -> LinearProbe:
    """Build a linear-probe wrapper with an identity backbone head."""

    probe_head = (
        LinearHead(in_features, out_features, key=key) if head is None else head
    )
    try:
        backbone = replace_head(backbone, IdentityHead())
    except ValueError:
        pass
    return cast(
        LinearProbe,
        LinearProbe(
            backbone,
            probe_head,
            pool=pool,
            feature_spec=feature_spec,
        ),
    )


def make_attention_pool_probe(
    backbone: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    n_last_blocks: int | None = None,
    embed_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.0,
    bias: bool = True,
    prepend_cls_token: bool = False,
    l2_normalize_cls: bool = False,
) -> AttentionPoolingProbe:
    """Build an attention-pooling probe with an identity backbone head."""

    head = cast(
        AttentionPoolingClassifierHead,
        AttentionPoolingClassifierHead(
            in_features,
            out_features,
            key=key,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        ),
    )
    try:
        backbone = replace_head(backbone, IdentityHead())
    except ValueError:
        pass
    return cast(
        AttentionPoolingProbe,
        AttentionPoolingProbe(
            backbone,
            head,
            n_last_blocks=n_last_blocks,
            prepend_cls_token=prepend_cls_token,
            l2_normalize_cls=l2_normalize_cls,
        ),
    )


def _call_with_optional_key(fn, *args, key, inference, **kwargs):
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return fn(*args, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        try:
            return fn(*args, **call_kwargs)
        except TypeError as second_error:
            if "unexpected keyword argument" not in str(second_error):
                raise
            call_kwargs.pop("key", None)
            return fn(*args, **call_kwargs)


def _call_forward_features(model: PyTree, *args, key, inference, **kwargs):
    if hasattr(model, "forward_features"):
        return _call_with_optional_key(
            model.forward_features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    if hasattr(model, "features"):
        return _call_with_optional_key(
            model.features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    return _call_with_optional_key(
        model,
        *args,
        key=key,
        inference=inference,
        **kwargs,
    )


def _split_optional_key(
    key: jax.Array | None,
) -> tuple[jax.Array | None, jax.Array | None]:
    if key is None:
        return None, None
    key_backbone, key_head = jr.split(key, 2)
    return key_backbone, key_head


def _as_patch_cls_tokens(
    item,
    *,
    norm: eqx.Module | None,
    num_prefix_tokens: int,
    has_cls_token: bool,
) -> tuple[jax.Array, jax.Array | None]:
    if isinstance(item, dict):
        patch_tokens = item.get("x_norm_patchtokens")
        if patch_tokens is None:
            raise ValueError(
                "intermediate feature dict must contain x_norm_patchtokens."
            )
        return patch_tokens, item.get("x_norm_cls_token")

    if (
        isinstance(item, tuple)
        and len(item) == 2
        and eqx.is_array(item[0])
        and eqx.is_array(item[1])
        and item[0].ndim == 2
        and item[1].ndim == 1
    ):
        return item

    if not eqx.is_array(item) or item.ndim != 2:
        raise ValueError(
            "Attention-pool intermediate inputs must be token arrays shaped "
            "[tokens, features] or (patch_tokens, cls_token) pairs."
        )

    tokens = _apply_norm_to_tokens(norm, item)
    cls_token = tokens[0] if has_cls_token else None
    patch_tokens = tokens[num_prefix_tokens:]
    return patch_tokens, cls_token


def _apply_norm_to_tokens(norm: eqx.Module | None, tokens: jax.Array) -> jax.Array:
    if norm is None:
        return tokens
    return jax.vmap(norm)(tokens)


def _maybe_l2_normalize(x: jax.Array, enabled: bool, eps: float) -> jax.Array:
    if not enabled:
        return x
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def _has_cls_token(model: PyTree) -> bool:
    return getattr(model, "cls_token", None) is not None


def _prompt_aware_pool(model: PyTree, pool: PoolName | eqx.Module | None):
    if pool == "cls_patch_mean" and (
        _is_vit_like(model) or _has_prefix_metadata(model)
    ):
        return _cls_patch_mean_pool_for_model(model)
    if (
        pool == "mean_patch"
        and getattr(model, "exclude_prompt_tokens_from_pool", False)
        and getattr(model, "num_prompt_tokens", 0)
    ):
        return MeanPatchPool(
            num_prefix_tokens=int(getattr(model, "num_base_prefix_tokens", 1)),
            num_prompt_tokens=int(model.num_prompt_tokens),
        )
    return pool


def _resolve_pool(
    model: PyTree,
    features: Any,
    pool: PoolName | eqx.Module | None,
) -> PoolName | eqx.Module | None:
    if pool != "auto":
        return pool

    model_name = model.__class__.__name__.lower()
    if _is_audio_model(model, model_name):
        global_pool = getattr(model, "global_pool", None)
        has_dist_token = getattr(model, "dist_token", None) is not None
        if global_pool is None:
            global_pool = "token" if has_dist_token else "avg"
        return _NativeReadoutPool(
            _normalize_native_pool_type(global_pool),
            num_prefix_tokens=_model_prefix_count(model),
            average_distillation_tokens=has_dist_token,
        )
    if _is_vit_like(model):
        global_pool = _normalize_native_pool_type(
            getattr(model, "global_pool", "token")
        )
        if global_pool in {"token", "cls_patch_mean", "avg", "avgmax", "max"}:
            return _NativeReadoutPool(
                global_pool,
                num_prefix_tokens=_model_prefix_count(model),
            )
    if isinstance(features, dict):
        return "auto"
    if not eqx.is_array(features):
        return pool

    if _is_text_model(model):
        if hasattr(model, "pooler") or hasattr(model, "cls_token"):
            return "cls"
        return MeanTokenPool()
    if _is_convnet_model(model):
        return GlobalAveragePool()
    if _is_mae_like(model, model_name):
        return _mean_patch_pool_for_model(model)
    if features.ndim <= 1:
        return "none"
    return "cls"


def _normalize_native_pool_type(pool_type: str) -> str:
    if pool_type in {"mean", "mean_patch"}:
        return "avg"
    return pool_type


def _pool_kwargs(model: PyTree, args: tuple, kwargs: dict) -> dict:
    pool_kwargs = dict(kwargs)
    if not _is_text_model(model):
        return pool_kwargs

    padding_mask = pool_kwargs.pop("padding_mask", None)
    if padding_mask is None and len(args) > 1:
        padding_mask = args[1]
    if padding_mask is not None:
        pool_kwargs["mask"] = padding_mask == 0
    return pool_kwargs


def _is_vit_like(model: PyTree) -> bool:
    return hasattr(model, "patch_embed") and hasattr(model, "blocks")


def _is_mae_like(model: PyTree, model_name: str) -> bool:
    return "mae" in model_name or getattr(model, "pool_policy", None) == "mean_patch"


def _is_audio_model(model: PyTree, model_name: str) -> bool:
    return (
        "ast" in model_name
        or "audio" in model_name
        or hasattr(model, "dist_token")
        or getattr(model, "modality", None) == "audio"
    )


def _is_text_model(model: PyTree) -> bool:
    return (
        hasattr(model, "token_embed")
        or hasattr(model, "token_embedding")
        or getattr(model, "modality", None) == "text"
    )


def _is_convnet_model(model: PyTree) -> bool:
    return (
        hasattr(model, "stem")
        and hasattr(model, "stages")
        and not hasattr(model, "patch_embed")
    )


def _mean_patch_pool_for_model(model: PyTree) -> MeanPatchPool:
    return cast(
        MeanPatchPool,
        MeanPatchPool(
            num_prefix_tokens=_model_prefix_count(model),
            num_prompt_tokens=int(getattr(model, "num_prompt_tokens", 0)),
        ),
    )


def _cls_patch_mean_pool_for_model(model: PyTree) -> CLSPatchMeanPool:
    return cast(
        CLSPatchMeanPool,
        CLSPatchMeanPool(
            num_prefix_tokens=_model_prefix_count(model),
            num_prompt_tokens=int(getattr(model, "num_prompt_tokens", 0)),
        ),
    )


def _model_prefix_count(model: PyTree) -> int:
    if hasattr(model, "num_base_prefix_tokens"):
        return int(model.num_base_prefix_tokens)
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)
    return _base_prefix_count(model)


def _has_prefix_metadata(model: PyTree) -> bool:
    return any(
        hasattr(model, name)
        for name in (
            "num_base_prefix_tokens",
            "num_prefix_tokens",
            "cls_token",
            "dist_token",
            "reg_tokens",
        )
    )


def _base_prefix_count(model: PyTree) -> int:
    count = 0
    for name in ("cls_token", "dist_token"):
        token = getattr(model, name, None)
        if token is not None:
            count += (
                int(token.shape[0]) if hasattr(token, "shape") and token.ndim > 1 else 1
            )
    reg_tokens = getattr(model, "reg_tokens", None)
    if reg_tokens is not None:
        count += int(reg_tokens.shape[0]) if hasattr(reg_tokens, "shape") else 1
    return count


def _call_head(
    head: eqx.Module, x: Any, *, key: jax.Array | None, inference: bool | None
):
    if not callable(head):
        raise TypeError(f"{type(head).__name__} is not callable.")
    callable_head = cast(Callable[..., object], head)
    try:
        return callable_head(x, key=key, inference=inference)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        return callable_head(x)


__all__ = (
    "AttentionPoolingProbe",
    "FeatureExtractor",
    "LinearProbe",
    "extract_features",
    "make_attention_pool_input_from_forward_features",
    "make_attention_pool_input_from_intermediates",
    "make_attention_pool_probe",
    "make_linear_probe",
)
