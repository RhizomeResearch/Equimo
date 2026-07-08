"""Feature-extraction wrappers for fine-tuning workflows."""

from __future__ import annotations

from typing import Any

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
    pool_features,
)
from .surgery import replace_head


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
    key: jax.Array | None = None,
    inference: bool | None = True,
    **kwargs,
) -> Any:
    """Call a model feature path and apply an optional pooling policy."""

    if hasattr(model, "features"):
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
    return pool_features(features, pool, key=key, **kwargs)


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
    cls_token = jnp.concatenate(cls_tokens, axis=-1)
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
    return LinearProbe(backbone, probe_head, pool=pool)


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

    head = AttentionPoolingClassifierHead(
        in_features,
        out_features,
        key=key,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
    )
    try:
        backbone = replace_head(backbone, IdentityHead())
    except ValueError:
        pass
    return AttentionPoolingProbe(
        backbone,
        head,
        n_last_blocks=n_last_blocks,
        prepend_cls_token=prepend_cls_token,
        l2_normalize_cls=l2_normalize_cls,
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
            raise ValueError("intermediate feature dict must contain x_norm_patchtokens.")
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
    if isinstance(features, dict):
        return "auto"
    if not eqx.is_array(features):
        return pool

    model_name = model.__class__.__name__.lower()
    if _is_audio_model(model, model_name):
        return "mean_frame"
    if _is_text_model(model):
        if hasattr(model, "pooler") or hasattr(model, "cls_token"):
            return "cls"
        return MeanTokenPool()
    if _is_convnet_model(model):
        return GlobalAveragePool()
    if _is_mae_like(model, model_name):
        return _mean_patch_pool_for_model(model)
    if _is_vit_like(model):
        if getattr(model, "global_pool", None) in {"avg", "mean", "mean_patch"}:
            return _mean_patch_pool_for_model(model)
        return "cls"
    if features.ndim <= 1:
        return "none"
    return "cls"


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
    return MeanPatchPool(
        num_prefix_tokens=_model_prefix_count(model),
        num_prompt_tokens=int(getattr(model, "num_prompt_tokens", 0)),
    )


def _cls_patch_mean_pool_for_model(model: PyTree) -> CLSPatchMeanPool:
    return CLSPatchMeanPool(
        num_prefix_tokens=_model_prefix_count(model),
        num_prompt_tokens=int(getattr(model, "num_prompt_tokens", 0)),
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
    try:
        return head(x, key=key, inference=inference)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        return head(x)


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
