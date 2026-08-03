# ty: ignore[invalid-assignment]
# Adapted from tfc-t0 and modified for JAX/Equinox; see NOTICE.
"""T0 patch-transformer time-series foundation model."""

__all__ = ["Forecast", "T0", "t0", "t0_alpha"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.factory import build_model_variant
from equimo.core.layers import BlockChunk, RMSNormGated
from equimo.registry import register_model
from equimo.time_series.layers import (
    PatchEncoder,
    ResidualMlp,
    T0Block,
    get_layer,
)


DEFAULT_MAX_HORIZON = 1024
_SCALER_EPS = 1e-1
_VALID = 0
_PAD = 1
_MISSING = 2
_WITHHELD = 4
_TARGET = 0
_FUTURE = 2


@dataclass(frozen=True)
class Forecast:
    """Quantile forecast returned by :meth:`T0.predict`."""

    quantiles: Float[Array, "*batch horizon quantiles"]
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> Float[Array, "*batch horizon"]:
        if 0.5 in self.quantile_levels:
            return self.quantiles[..., self.quantile_levels.index(0.5)]
        return _interpolate_quantiles(
            jnp.asarray((0.5,), dtype=jnp.float32),
            jnp.asarray(self.quantile_levels, dtype=jnp.float32),
            self.quantiles,
        )[..., 0]


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _interpolate_quantiles(query, levels, values):
    values = jnp.asarray(values, dtype=jnp.float32)
    levels = jnp.asarray(levels, dtype=jnp.float32)
    if levels.ndim == 1:
        levels = jnp.broadcast_to(levels, values.shape)
    if levels.shape != values.shape:
        raise ValueError("quantile levels must match values or be one-dimensional")
    query = jnp.asarray(query, dtype=jnp.float32)
    padded_levels = jnp.concatenate(
        (jnp.zeros_like(levels[..., :1]), levels, jnp.ones_like(levels[..., :1])),
        axis=-1,
    )
    padded_values = jnp.concatenate(
        (values[..., :1], values, values[..., -1:]), axis=-1
    )
    query_for_search = query.reshape((1,) * (levels.ndim - 1) + (query.shape[0], 1))
    upper = jnp.sum(padded_levels[..., None, :] <= query_for_search, axis=-1)
    upper = jnp.clip(upper, 1, padded_levels.shape[-1] - 1)
    lower = upper - 1
    lower_levels = jnp.take_along_axis(
        padded_levels[..., None, :], lower[..., None], axis=-1
    )[..., 0]
    upper_levels = jnp.take_along_axis(
        padded_levels[..., None, :], upper[..., None], axis=-1
    )[..., 0]
    lower_values = jnp.take_along_axis(
        padded_values[..., None, :], lower[..., None], axis=-1
    )[..., 0]
    upper_values = jnp.take_along_axis(
        padded_values[..., None, :], upper[..., None], axis=-1
    )[..., 0]
    weight = jnp.nan_to_num(
        (query - lower_levels) / (upper_levels - lower_levels), nan=0.0
    )
    return lower_values + weight * (upper_values - lower_values)


def _causal_stats(values, invalid):
    valid = ~invalid
    count = jnp.cumsum(valid.astype(values.dtype), axis=-1)
    safe_count = jnp.maximum(count, 1.0)
    masked = jnp.where(invalid, 0.0, values)
    mean = jnp.cumsum(masked, axis=-1) / safe_count
    previous_mean = jnp.concatenate(
        (jnp.zeros_like(mean[..., :1]), mean[..., :-1]), axis=-1
    )
    increment = (masked - previous_mean) * (masked - mean) * valid
    m2 = jnp.cumsum(increment, axis=-1)
    variance = m2 / jnp.maximum(count - 1.0, 1.0)
    return mean, jnp.sqrt(jnp.maximum(variance, 0.0) + _SCALER_EPS)


def _global_stats(values, invalid):
    valid = ~invalid
    count = jnp.sum(valid, axis=-1, keepdims=True).astype(values.dtype)
    safe_count = jnp.maximum(count, 1.0)
    masked = jnp.where(invalid, 0.0, values)
    mean = jnp.sum(masked, axis=-1, keepdims=True) / safe_count
    variance = jnp.sum(
        jnp.where(valid, (values - mean) ** 2, 0.0), axis=-1, keepdims=True
    )
    variance = variance / jnp.maximum(count, 2.0)
    scale = jnp.maximum(jnp.sqrt(variance), _SCALER_EPS)
    return jnp.broadcast_to(mean, values.shape), jnp.broadcast_to(scale, values.shape)


def _scale_input(values, mask, group_ids, variate_type):
    invalid = mask != _VALID
    non_padding = group_ids >= 0
    causal = non_padding & ((variate_type == _TARGET) | (variate_type == 1))
    future = non_padding & (variate_type == _FUTURE)
    loc = jnp.zeros_like(values)
    scale = jnp.ones_like(values)
    causal_loc, causal_scale = _causal_stats(values, invalid)
    global_loc, global_scale = _global_stats(values, invalid)
    loc = jnp.where(causal, causal_loc, loc)
    scale = jnp.where(causal, causal_scale, scale)
    loc = jnp.where(future, global_loc, loc)
    scale = jnp.where(future, global_scale, scale)
    scaled = jnp.arcsinh((values - loc) / scale)
    return scaled, loc, scale


def _rescale_predictions(predictions, loc, scale, model_patch_size):
    loc = loc[:, model_patch_size - 1 :: model_patch_size]
    scale = scale[:, model_patch_size - 1 :: model_patch_size]
    if loc.shape[1] != predictions.shape[1]:
        raise ValueError(
            f"scaler patches ({loc.shape[1]}) do not match model patches ({predictions.shape[1]})"
        )
    loc = loc[..., None, None]
    scale = scale[..., None, None]
    return jnp.sinh(predictions) * scale + loc


def _probability_masses(levels):
    boundaries = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=levels.dtype),
            levels,
            jnp.ones((1,), dtype=levels.dtype),
        )
    )
    masses = (boundaries[2:] - boundaries[:-2]) / 2
    return masses / masses.sum()


def _weighted_quantile(query, weights, samples):
    indices = jnp.argsort(samples, axis=-1)
    sorted_samples = jnp.take_along_axis(samples, indices, axis=-1)
    sorted_weights = jnp.take_along_axis(
        jnp.broadcast_to(weights, samples.shape), indices, axis=-1
    )
    levels = jnp.clip(jnp.cumsum(sorted_weights, axis=-1), 0.0, 1.0)
    return _interpolate_quantiles(query, levels, sorted_samples)


def _prepare_input(context, future_covariates, horizon):
    context = jnp.asarray(context, dtype=jnp.float32)
    if context.ndim == 1:
        context = context[None]
    if context.ndim not in (2, 3):
        raise ValueError(
            f"context must be [T], [B, T] or [B, V, T]; got shape {context.shape}"
        )
    batch_shape = context.shape[:2] if context.ndim == 3 else context.shape[:1]
    batch_size = context.shape[0]
    n_variates = context.shape[1] if context.ndim == 3 else 1
    context_length = context.shape[-1]
    target_values = context.reshape(-1, context_length)
    missing = jnp.isnan(target_values)
    target_values = jnp.nan_to_num(target_values, nan=0.0)
    target_mask = jnp.where(missing, _MISSING, _VALID).astype(jnp.int8)
    target_groups = jnp.repeat(jnp.arange(batch_size), n_variates)[:, None]
    target_groups = jnp.broadcast_to(target_groups, target_values.shape)
    target_types = jnp.full(target_values.shape, _TARGET, dtype=jnp.int32)

    if future_covariates is None:
        return (
            target_values,
            target_mask,
            target_groups,
            target_types,
            batch_shape,
            context_length,
        )

    future = jnp.asarray(future_covariates, dtype=jnp.float32)
    expected_length = context_length + horizon
    if (
        future.ndim != 3
        or future.shape[0] != batch_size
        or future.shape[2] != expected_length
    ):
        raise ValueError(
            f"future_covariates must be [B={batch_size}, F, T+horizon={expected_length}]; "
            f"got shape {future.shape}"
        )
    n_future = future.shape[1]
    future = future.reshape(-1, expected_length)
    future_missing = jnp.isnan(future)
    future = jnp.nan_to_num(future, nan=0.0)
    future_mask = jnp.where(future_missing, _MISSING, _VALID).astype(jnp.int8)
    future_groups = jnp.repeat(jnp.arange(batch_size), n_future)[:, None]
    future_groups = jnp.broadcast_to(future_groups, future.shape)
    future_types = jnp.full(future.shape, _FUTURE, dtype=jnp.int32)

    target_values = jnp.concatenate(
        (
            target_values,
            jnp.zeros((target_values.shape[0], horizon), dtype=jnp.float32),
        ),
        axis=-1,
    )
    target_mask = jnp.concatenate(
        (
            target_mask,
            jnp.full((target_mask.shape[0], horizon), _WITHHELD, dtype=jnp.int8),
        ),
        axis=-1,
    )
    target_groups = jnp.broadcast_to(target_groups[:, :1], target_values.shape)
    target_types = jnp.broadcast_to(target_types[:, :1], target_values.shape)
    return (
        jnp.concatenate((target_values, future), axis=0),
        jnp.concatenate((target_mask, future_mask), axis=0),
        jnp.concatenate((target_groups, future_groups), axis=0),
        jnp.concatenate((target_types, future_types), axis=0),
        batch_shape,
        context_length,
    )


def _prepare_rollout_buffer(
    values, mask, groups, types, prediction_length, context_length, patch_size
):
    pad_left = (-context_length) % patch_size
    forecast_width = _round_up(prediction_length, patch_size)
    known = min(values.shape[1] - context_length, forecast_width)
    target_rows = types[:, 0] == _TARGET
    future_rows = types[:, 0] == _FUTURE
    forecast_values = jnp.zeros((values.shape[0], forecast_width), dtype=values.dtype)
    forecast_mask = jnp.broadcast_to(
        jnp.where(target_rows[:, None], _WITHHELD, _PAD),
        (values.shape[0], forecast_width),
    ).astype(jnp.int8)
    if known:
        known_values = values[:, context_length : context_length + known]
        known_mask = mask[:, context_length : context_length + known]
        forecast_values = forecast_values.at[:, :known].set(
            jnp.where(future_rows[:, None], known_values, 0.0)
        )
        forecast_mask = forecast_mask.at[:, :known].set(
            jnp.where(future_rows[:, None], known_mask, _PAD)
        )
    row_groups = jnp.broadcast_to(groups[:, :1], (values.shape[0], forecast_width))
    row_types = jnp.broadcast_to(types[:, :1], (values.shape[0], forecast_width))
    pad_values = jnp.zeros((values.shape[0], pad_left), dtype=values.dtype)
    pad_mask = jnp.full((values.shape[0], pad_left), _PAD, dtype=jnp.int8)
    pad_groups = jnp.full((values.shape[0], pad_left), -1, dtype=groups.dtype)
    pad_types = jnp.full((values.shape[0], pad_left), -1, dtype=types.dtype)
    context = slice(0, context_length)
    return (
        jnp.concatenate((pad_values, values[:, context], forecast_values), axis=1),
        jnp.concatenate((pad_mask, mask[:, context], forecast_mask), axis=1),
        jnp.concatenate((pad_groups, groups[:, context], row_groups), axis=1),
        jnp.concatenate((pad_types, types[:, context], row_types), axis=1),
    )


def _expand_prediction_paths(values, mask, groups, types, n_paths):
    row_groups = groups[:, -1]
    row_types = types[:, -1]
    path_groups = (row_groups[:, None] * n_paths + jnp.arange(n_paths)).reshape(-1)
    path_types = jnp.repeat(row_types, n_paths)[:, None]
    return (
        jnp.repeat(values, n_paths, axis=0),
        jnp.repeat(mask, n_paths, axis=0),
        jnp.broadcast_to(path_groups[:, None], (path_groups.shape[0], values.shape[1])),
        jnp.broadcast_to(path_types, (path_types.shape[0], values.shape[1])),
    )


def _update_prediction_buffer(values, mask, groups, types, prediction, at):
    target_rows = types[:, -1] == _TARGET
    horizon = prediction.shape[1]
    path_values = jnp.transpose(prediction, (0, 2, 1)).reshape(-1, horizon)
    values = values.at[target_rows, at : at + horizon].set(path_values)
    mask = mask.at[target_rows, at : at + horizon].set(_VALID)
    return values, mask, groups, types


def _predict_step(model, values, mask, groups, types, horizon, key):
    scaled, loc, scale = _scale_input(values, mask, groups, types)
    predictions = model(
        scaled,
        mask,
        groups,
        types,
        key=key,
        inference=True,
    )
    predictions = _rescale_predictions(predictions, loc, scale, model.patch_size)
    context_patches = (values.shape[1] - horizon) // model.patch_size
    start = context_patches - 1
    return predictions[:, start : start + horizon // model.patch_size].reshape(
        predictions.shape[0], -1, predictions.shape[-1]
    )


@register_model("t0", modality="time_series")
class T0(eqx.Module):
    """T0 backbone for flattened variates and per-cell metadata."""

    patch_encoder: PatchEncoder
    blocks: tuple[BlockChunk, ...]
    out_norm: RMSNormGated
    decoder: ResidualMlp
    quantile_levels: tuple[float, ...] = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    group_every_n: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        embed_dim=512,
        num_layers=24,
        num_heads=8,
        mlp_hidden_dim=2048,
        patch_size=32,
        group_every_n=3,
        dropout=0.1,
        quantile_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        patch_encoder_layer="patchencoder",
        block_layer="t0block",
        decoder_layer="residualmlp",
        key: PRNGKeyArray,
    ):
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 2:
            raise ValueError("embed_dim / num_heads must be even for XPos")
        if group_every_n > 0 and num_layers % group_every_n:
            raise ValueError("group_every_n must divide num_layers")
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        quantile_levels = tuple(sorted(float(q) for q in quantile_levels))
        if not quantile_levels:
            raise ValueError(
                "quantile_levels must be a non-empty sequence of floats in (0, 1)"
            )
        for quantile in quantile_levels:
            if not 0.0 < quantile < 1.0:
                raise ValueError(f"each quantile must be in (0, 1); got {quantile}")
        patch_encoder_layer = cast(type[PatchEncoder], get_layer(patch_encoder_layer))
        block_layer = cast(type[T0Block], get_layer(block_layer))
        decoder_layer = cast(type[ResidualMlp], get_layer(decoder_layer))
        key_encoder, key_blocks, key_decoder = jr.split(key, 3)
        self.patch_encoder = patch_encoder_layer(embed_dim, patch_size, key=key_encoder)
        attention_types = [
            "group" if group_every_n > 0 and (i + 1) % group_every_n == 0 else "time"
            for i in range(num_layers)
        ]
        self.blocks = (
            BlockChunk(
                depth=num_layers,
                module=block_layer,
                module_kwargs={
                    "dim": embed_dim,
                    "num_heads": num_heads,
                    "hidden_dim": mlp_hidden_dim,
                    "dropout": dropout,
                    "attention_type": attention_types,
                },
                key=key_blocks,
            ),
        )
        self.out_norm = RMSNormGated(embed_dim, eps=1e-8)
        self.quantile_levels = quantile_levels
        self.decoder = decoder_layer(
            embed_dim,
            embed_dim,
            patch_size * len(self.quantile_levels),
            key=key_decoder,
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.group_every_n = group_every_n

    def _patch(self, values, mask, group_ids, variate_type):
        pad = (-values.shape[-1]) % self.patch_size
        if pad:
            values = jnp.pad(values, ((0, 0), (pad, 0)))
            mask = jnp.pad(mask, ((0, 0), (pad, 0)), constant_values=1)
            group_ids = jnp.pad(group_ids, ((0, 0), (pad, 0)), constant_values=-1)
            variate_type = jnp.pad(variate_type, ((0, 0), (pad, 0)), constant_values=-1)
        shape = (values.shape[0], -1, self.patch_size)
        return tuple(x.reshape(shape) for x in (values, mask, group_ids, variate_type))

    @staticmethod
    def _masks(group_ids, variate_type, mask):
        patch_groups = group_ids[:, :, 0]
        patch_types = variate_type[:, :, 0]
        valid = patch_groups >= 0
        same_doc = (
            (patch_groups[:, :, None] == patch_groups[:, None, :])
            & valid[:, :, None]
            & valid[:, None, :]
        )
        causal = jnp.tril(jnp.ones(same_doc.shape[-2:], dtype=bool))
        time_mask = jnp.where(
            (patch_types == 2)[:, :, None], same_doc, same_doc & causal
        )
        time_mask &= jnp.any(mask != 1, axis=-1)[:, None, :]
        groups_t, valid_t = patch_groups.T, valid.T
        group_mask = (
            (groups_t[:, :, None] == groups_t[:, None, :])
            & valid_t[:, :, None]
            & valid_t[:, None, :]
        )
        return time_mask, group_mask

    def _encode(
        self,
        values,
        mask,
        group_ids,
        variate_type,
        *,
        key_encoder,
        key_blocks,
        inference,
    ):
        values, mask, group_ids, variate_type = self._patch(
            values, mask, group_ids, variate_type
        )
        x = self.patch_encoder(
            values, mask, variate_type, key=key_encoder, inference=inference
        )
        time_mask, group_mask = self._masks(group_ids, variate_type, mask)
        x = self.blocks[0](
            x,
            time_attn_mask=time_mask,
            group_attn_mask=group_mask,
            key=key_blocks,
            inference=inference,
        )
        return self.out_norm(x)

    def features(
        self,
        values: Float[Array, "variates time"],
        mask: Int[Array, "variates time"],
        group_ids: Int[Array, "variates time"],
        variate_type: Int[Array, "variates time"],
        *,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "tokens embed_dim"]:
        """Return post-normalization latent tokens before quantile decoding.

        Tokens are flattened in variate-major, patch-minor order. The patch axis
        includes the left-padded leading patch when the input length is not
        divisible by ``patch_size``.
        """

        key_encoder, key_blocks, _ = jr.split(key, 3)
        x = self._encode(
            values,
            mask,
            group_ids,
            variate_type,
            key_encoder=key_encoder,
            key_blocks=key_blocks,
            inference=inference,
        )
        return x.reshape(-1, self.embed_dim)

    def __call__(
        self,
        values: Float[Array, "variates time"],
        mask: Int[Array, "variates time"],
        group_ids: Int[Array, "variates time"],
        variate_type: Int[Array, "variates time"],
        *,
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
    ) -> Float[Array, "variates patches patch_size quantiles"]:
        key_encoder, key_blocks, key_decoder = jr.split(key, 3)
        x = self._encode(
            values,
            mask,
            group_ids,
            variate_type,
            key_encoder=key_encoder,
            key_blocks=key_blocks,
            inference=inference,
        )
        x = self.decoder(x, key=key_decoder, inference=inference)
        x = x.reshape(*x.shape[:-1], self.patch_size, len(self.quantile_levels))
        first = x[..., :1]
        return jnp.concatenate(
            (first, first + jnp.cumsum(jax.nn.softplus(x[..., 1:]), axis=-1)), axis=-1
        )

    def predict(
        self,
        context,
        horizon: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        future_covariates=None,
    ) -> Forecast:
        """Forecast future values using TFC T0's scaling and rollout API."""
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        query_quantiles = tuple(float(q) for q in quantiles)
        if not query_quantiles:
            raise ValueError("quantiles must be non-empty")
        if any(not 0.0 < q < 1.0 for q in query_quantiles):
            raise ValueError(f"each quantile must be in (0, 1); got {query_quantiles}")
        if list(query_quantiles) != sorted(set(query_quantiles)):
            raise ValueError(
                "quantiles must be sorted ascending without duplicates; "
                f"got {list(query_quantiles)}"
            )
        if (
            DEFAULT_MAX_HORIZON < self.patch_size
            or DEFAULT_MAX_HORIZON % self.patch_size
        ):
            raise ValueError(
                f"max_horizon must be a positive multiple of patch_size ({self.patch_size})"
            )

        values, mask, groups, types, batch_shape, context_length = _prepare_input(
            context, future_covariates, horizon
        )
        buffer = _prepare_rollout_buffer(
            values,
            mask,
            groups,
            types,
            horizon,
            context_length,
            self.patch_size,
        )
        context_width = _round_up(context_length, self.patch_size)
        decoded_horizon = min(_round_up(horizon, self.patch_size), DEFAULT_MAX_HORIZON)
        key = jr.PRNGKey(42)
        target_rows = types[:, 0] == _TARGET
        initial_window = tuple(x[:, : context_width + decoded_horizon] for x in buffer)
        block = _predict_step(
            self,
            initial_window[0],
            initial_window[1],
            initial_window[2],
            initial_window[3],
            horizon=decoded_horizon,
            key=key,
        )
        query = jnp.asarray(query_quantiles, dtype=jnp.float32)
        prediction = _interpolate_quantiles(
            query,
            jnp.asarray(self.quantile_levels, dtype=jnp.float32),
            block,
        )[target_rows]

        if horizon > decoded_horizon:
            paths = _expand_prediction_paths(
                buffer[0], buffer[1], buffer[2], buffer[3], n_paths=len(query_quantiles)
            )
            path_target_rows = paths[3][:, -1] == _TARGET
            predicted_masses = _probability_masses(
                jnp.asarray(self.quantile_levels, dtype=jnp.float32)
            )
            query_masses = _probability_masses(query)
            weights = jnp.outer(predicted_masses, query_masses).reshape(-1)
            predictions = [prediction]
            decoded = decoded_horizon
            remaining = horizon - decoded_horizon
            while remaining > 0:
                previous_width = predictions[-1].shape[1]
                paths = _update_prediction_buffer(
                    paths[0],
                    paths[1],
                    paths[2],
                    paths[3],
                    predictions[-1],
                    at=context_width + decoded - previous_width,
                )
                step_horizon = min(
                    _round_up(remaining, self.patch_size), DEFAULT_MAX_HORIZON
                )
                window = tuple(
                    x[:, decoded : context_width + decoded + step_horizon]
                    for x in paths
                )
                block = _predict_step(
                    self,
                    window[0],
                    window[1],
                    window[2],
                    window[3],
                    horizon=step_horizon,
                    key=key,
                )[path_target_rows]
                target_count = prediction.shape[0]
                block = block.reshape(
                    target_count,
                    len(query_quantiles),
                    block.shape[1],
                    block.shape[2],
                )
                samples = jnp.transpose(block, (0, 2, 3, 1)).reshape(
                    target_count, block.shape[2], -1
                )
                predictions.append(_weighted_quantile(query, weights, samples))
                decoded += step_horizon
                remaining -= step_horizon
            prediction = jnp.concatenate(predictions, axis=1)[:, :horizon]
        else:
            prediction = prediction[:, :horizon]

        prediction = jnp.nan_to_num(
            prediction.astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        if len(batch_shape) == 2:
            prediction = prediction.reshape(
                batch_shape[0], batch_shape[1], horizon, len(query_quantiles)
            )
        else:
            prediction = prediction.reshape(
                batch_shape[0], horizon, len(query_quantiles)
            )
        return Forecast(prediction, query_quantiles)


_T0_BASE_CFG = {
    "embed_dim": 512,
    "num_layers": 24,
    "num_heads": 8,
    "mlp_hidden_dim": 2048,
    "patch_size": 32,
    "group_every_n": 3,
    "dropout": 0.1,
    "quantile_levels": (0.1, 0.25, 0.5, 0.75, 0.9),
}

_T0_REGISTRY = {
    "t0": (_T0_BASE_CFG, {}),
    "t0_alpha": (_T0_BASE_CFG, {}),
}

_T0_PRETRAINED_IDENTIFIERS = {
    "t0": "t0_alpha",
    "t0_alpha": "t0_alpha",
}


def _catalog_model_variants():
    """Return catalog metadata for the published T0-alpha backbone."""
    from equimo.catalog import (
        ModelInput,
        ModelProvenance,
        ModelVariant,
        PretrainedWeights,
    )

    variant = "t0_alpha"
    return (
        ModelVariant(
            key=f"time_series/{variant}",
            modality="time_series",
            family="t0",
            variant=variant,
            model_registry_key="t0",
            constructor=f"{__name__}.{variant}",
            inputs=(
                ModelInput(
                    name="values",
                    shape=("variates", "time"),
                    axes=("variates", "time"),
                    dtype="float32",
                    description="Flattened, preprocessed variate values.",
                ),
                ModelInput(
                    name="mask",
                    shape=("variates", "time"),
                    axes=("variates", "time"),
                    dtype="int8",
                    description="Per-cell T0 mask reason.",
                ),
                ModelInput(
                    name="group_ids",
                    shape=("variates", "time"),
                    axes=("variates", "time"),
                    dtype="int64",
                    description="Per-cell sample grouping identifiers.",
                ),
                ModelInput(
                    name="variate_type",
                    shape=("variates", "time"),
                    axes=("variates", "time"),
                    dtype="int64",
                    description="Per-cell target, historical, or future role.",
                ),
            ),
            pretrained=PretrainedWeights(available=True, identifier=variant),
            provenance=ModelProvenance(
                conversion="models/t0.py",
                reference=(
                    "tests/data/reference_provenance.json#t0_alpha_reference.npz"
                ),
            ),
            notes=(
                "This entry exposes the raw T0 backbone, not the upstream "
                "scaling and rollout predict API.",
            ),
            field_status=(
                ("inputs", "complete"),
                ("pretrained", "complete"),
                ("provenance", "complete"),
                ("notes", "complete"),
            ),
        ),
    )


def _build_t0(
    variant: str,
    *,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> T0:
    return build_model_variant(
        T0,
        _T0_REGISTRY,
        variant,
        pretrained=pretrained,
        inference_mode=inference_mode,
        key=key,
        pretrained_variants=frozenset(_T0_PRETRAINED_IDENTIFIERS),
        pretrained_identifiers=_T0_PRETRAINED_IDENTIFIERS,
        pretrained_label="T0",
        allow_pretrained_overrides=False,
        **overrides,
    )


def t0(
    pretrained: bool = False,
    inference_mode: bool = True,
    **kwargs,
) -> T0:
    """Build the default T0 model, loading T0-alpha when pretrained."""
    return _build_t0(
        "t0",
        pretrained=pretrained,
        inference_mode=inference_mode,
        **kwargs,
    )


def t0_alpha(
    pretrained: bool = False,
    inference_mode: bool = True,
    **kwargs,
) -> T0:
    """Build the published T0-alpha model."""
    return _build_t0(
        "t0_alpha",
        pretrained=pretrained,
        inference_mode=inference_mode,
        **kwargs,
    )
