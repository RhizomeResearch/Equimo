# ty: ignore[invalid-assignment]
# ty: ignore[invalid-return-type]
# ty: ignore[call-non-callable]
"""Plain mask decoder for frozen vision features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from equimo.core._prng import split_for_mode
from equimo.core.layers.attention import Attention, AttentionBlock
from equimo.core.layers.rotary import RotaryFactors, insert_rotary_identity
from equimo.utils import nearest_power_of_2_divisor
from equimo.vision.models.eomt import (
    MaskEmbedding,
    MaskPrediction,
    ScaleBlock,
    _build_upscale,
    _predict_query_masks,
)
from equimo.vision.segmentation import (
    _anneal_mask_probabilities,
    _query_attention_mask,
)


_MISSING = object()


@dataclass(frozen=True)
class PMTConfig:
    """PMD architecture and lateral normalization.

    ``small()`` is the default. ``base()`` and ``large()`` reproduce the
    respective encoder widths, attention heads, and taps of the author image
    configurations. All presets use six blocks and 100 semantic queries;
    ``PMTConfig.small(num_queries=200)`` selects the author COCO query count.

    The lateral normalization defaults to GroupNorm. For each feature level,
    its group count is ``nearest_power_of_2_divisor(dim, norm_max_group)``;
    the default maximum is 32. This normalizes one image independently and
    needs no mutable model state. Use ``norm_layer="batchnorm"`` for the
    original PMT architecture and its published decoder weights. That route
    requires explicit ``equinox.nn.State`` and a named ``pmt_batch`` vmap axis
    during training. Changing normalization creates a different architecture.
    """

    dim: int = 384
    num_heads: int = 6
    hidden_dim: int = 384
    taps: tuple[int, ...] = (2, 5, 8, 11)
    num_queries: int = 100
    num_blocks: int = 6
    norm_layer: str = "groupnorm"
    norm_max_group: int = 32
    norm_kwargs: tuple[tuple[str, float | bool], ...] | Mapping[str, float | bool] = ()
    masked_attention: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.norm_kwargs, Mapping):
            object.__setattr__(
                self, "norm_kwargs", tuple(sorted(self.norm_kwargs.items()))
            )
        if (
            self.dim <= 0
            or self.num_heads <= 0
            or self.dim % self.num_heads
            or self.dim // self.num_heads % 4
            or self.hidden_dim <= 0
            or self.num_queries <= 0
            or self.num_blocks <= 0
            or self.norm_max_group <= 0
        ):
            raise ValueError(
                "Invalid PMT width, heads, depth, queries, or norm groups."
            )
        if (
            not self.taps
            or tuple(sorted(set(self.taps))) != self.taps
            or self.taps[0] < 0
        ):
            raise ValueError(
                "PMT taps must be sorted, unique nonnegative block indices."
            )
        if self.norm_layer not in {
            "groupnorm",
            "batchnorm",
            "layernorm",
            "rmsnorm",
            "none",
        }:
            raise ValueError("Unsupported PMT lateral norm_layer.")
        allowed = {
            "groupnorm": {"eps", "channelwise_affine"},
            "batchnorm": {"eps", "momentum"},
            "layernorm": {"eps", "use_weight", "use_bias"},
            "rmsnorm": {"eps", "use_weight", "use_bias"},
            "none": set(),
        }[self.norm_layer]
        options = dict(self.norm_kwargs)
        if len(options) != len(self.norm_kwargs) or options.keys() - allowed:
            raise ValueError("Unsupported or repeated PMT normalization option.")

    @classmethod
    def small(cls, **overrides) -> PMTConfig:
        """Build the Small configuration; defaults to 100 semantic queries."""
        return replace(cls(), **overrides)

    @classmethod
    def base(cls, **overrides) -> PMTConfig:
        """Build the Base width, heads, and four 12-block taps."""
        return replace(cls(dim=768, num_heads=12, hidden_dim=768), **overrides)

    @classmethod
    def large(cls, **overrides) -> PMTConfig:
        """Build the Large width, heads, and four 24-block taps."""
        return replace(
            cls(dim=1024, num_heads=16, hidden_dim=1024, taps=(5, 11, 17, 23)),
            **overrides,
        )


class PMTFeatures(eqx.Module):
    """Cached encoder output before all trainable lateral normalization.

    Levels contain normalized encoder tokens in class/register/patch order.
    The encoder digest, input view, and static geometry identify the feature
    source; a cache must be discarded if any of them changes.
    """

    levels: tuple[jax.Array, ...]
    rotary: RotaryFactors | None
    grid_size: tuple[int, int] = eqx.field(static=True)
    patch_size: tuple[int, int] = eqx.field(static=True)
    prefix_tokens: tuple[str, ...] = eqx.field(static=True)
    taps: tuple[int, ...] = eqx.field(static=True)
    positional_configuration: tuple[tuple[str, str], ...] = eqx.field(static=True)
    backbone_id: str = eqx.field(static=True)
    backbone_digest: str = eqx.field(static=True)
    input_view: str = eqx.field(static=True)


class PMTOutput(eqx.Module):
    """Final and intermediate class/mask logits in decoder execution order."""

    final: MaskPrediction
    auxiliary: tuple[MaskPrediction, ...]


class PMTMaskState(eqx.Module):
    """Explicit per-block mask-retention probabilities and optimizer step."""

    probabilities: jax.Array
    step: jax.Array

    def __init__(self, probabilities: jax.Array, step: int | jax.Array = 0):
        probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
        if probabilities.ndim != 1:
            raise ValueError("PMT mask probabilities must be one-dimensional.")
        self.probabilities = probabilities
        self.step = jnp.asarray(step, dtype=jnp.int32)


def anneal_pmt_mask_state(
    step: int | jax.Array,
    start_steps: tuple[int, ...],
    end_steps: tuple[int, ...],
    *,
    power: float = 0.9,
) -> PMTMaskState:
    """Return author-style polynomial mask retention at an optimizer step."""
    return PMTMaskState(
        _anneal_mask_probabilities(step, start_steps, end_steps, power), step
    )


class ReferenceBatchNorm(eqx.nn.StatefulLayer):
    """Single-host counterpart of the author's lateral SyncBatchNorm.

    Call over one image inside ``jax.vmap(..., axis_name="pmt_batch")`` in
    training. Statistics cover batch and tokens, including prefix tokens.
    Inference uses the stored running mean and unbiased running variance.
    """

    weight: jax.Array
    bias: jax.Array
    mean_index: eqx.nn.StateIndex
    variance_index: eqx.nn.StateIndex
    count_index: eqx.nn.StateIndex
    eps: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-5, momentum: float = 0.1):
        if dim <= 0 or eps <= 0 or not 0 < momentum <= 1:
            raise ValueError("Invalid reference BatchNorm dimensions or options.")
        self.weight = jnp.ones((dim,), dtype=jnp.float32)
        self.bias = jnp.zeros((dim,), dtype=jnp.float32)
        self.mean_index = eqx.nn.StateIndex(jnp.zeros((dim,), dtype=jnp.float32))
        self.variance_index = eqx.nn.StateIndex(jnp.ones((dim,), dtype=jnp.float32))
        self.count_index = eqx.nn.StateIndex(jnp.asarray(0, dtype=jnp.int32))
        self.eps = eps
        self.momentum = momentum

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, *, inference: bool
    ) -> tuple[jax.Array, eqx.nn.State]:
        if x.ndim != 2 or x.shape[0] != self.weight.shape[0]:
            raise ValueError("Reference BatchNorm expects (channels, tokens).")
        value = x.astype(jnp.float32)
        if inference:
            mean = state.get(self.mean_index)
            variance = state.get(self.variance_index)
        else:
            if x.shape[1] * lax.axis_size("pmt_batch") < 2:
                raise ValueError("BatchNorm requires at least two observations.")
            mean = lax.pmean(jnp.mean(value, axis=1), "pmt_batch")
            variance = lax.pmean(
                jnp.mean(jnp.square(value - mean[:, None]), axis=1), "pmt_batch"
            )
            count = lax.psum(jnp.asarray(x.shape[1], dtype=jnp.float32), "pmt_batch")
            unbiased = variance * count / (count - 1)
            state = state.set(
                self.mean_index,
                (1 - self.momentum) * state.get(self.mean_index) + self.momentum * mean,
            )
            state = state.set(
                self.variance_index,
                (1 - self.momentum) * state.get(self.variance_index)
                + self.momentum * unbiased,
            )
            state = state.set(self.count_index, state.get(self.count_index) + 1)
        normalized = (value - mean[:, None]) * lax.rsqrt(variance[:, None] + self.eps)
        return (normalized * self.weight[:, None] + self.bias[:, None]).astype(
            x.dtype
        ), state


class PMDAttention(Attention):
    """Fused QKV weights with trainable Q/V bias and no key-bias leaf."""

    query_bias: jax.Array
    value_bias: jax.Array

    def __init__(self, dim: int, num_heads: int, *, key: jax.Array, **kwargs):
        kwargs.pop("qkv_bias", None)
        super().__init__(dim, num_heads, key=key, qkv_bias=False, **kwargs)
        self.query_bias = jnp.zeros((dim,))
        self.value_bias = jnp.zeros((dim,))

    def _project_qkv(self, x: jax.Array) -> jax.Array:
        result = super()._project_qkv(x)
        bias = jnp.concatenate(
            (self.query_bias, jnp.zeros_like(self.query_bias), self.value_bias)
        )
        return result + bias.astype(result.dtype)


class Lateral(eqx.Module):
    """One trainable normalization, residual MLP, and output scale."""

    norm: eqx.Module
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    scale: jax.Array
    norm_layer: str = eqx.field(static=True)

    def __init__(self, config: PMTConfig, *, key: jax.Array):
        dim = config.dim
        options = dict(config.norm_kwargs)
        if config.norm_layer == "groupnorm":
            self.norm = eqx.nn.GroupNorm(
                nearest_power_of_2_divisor(dim, config.norm_max_group), dim, **options
            )
        elif config.norm_layer == "batchnorm":
            self.norm = ReferenceBatchNorm(dim, **options)
        elif config.norm_layer == "layernorm":
            self.norm = eqx.nn.LayerNorm(dim, **options)
        elif config.norm_layer == "rmsnorm":
            self.norm = eqx.nn.RMSNorm(dim, **options)
        else:
            self.norm = eqx.nn.Identity()
        first, second = jr.split(key)
        self.fc1 = eqx.nn.Linear(dim, dim // 2, key=first)
        self.fc2 = eqx.nn.Linear(dim // 2, dim, key=second)
        self.scale = jnp.ones((dim,))
        self.norm_layer = config.norm_layer

    def __call__(
        self,
        tokens: jax.Array,
        state: eqx.nn.State | None,
        *,
        inference: bool,
    ) -> tuple[jax.Array, eqx.nn.State | None]:
        if self.norm_layer == "batchnorm":
            assert state is not None
            normalized, state = self.norm(tokens.T, state, inference=inference)
            normalized = normalized.T
        elif self.norm_layer in ("groupnorm", "none"):
            normalized = self.norm(tokens.T).T
        else:
            normalized = jax.vmap(self.norm)(tokens)
        projected = jax.vmap(self.fc1)(normalized)
        projected = jax.nn.gelu(projected, approximate=False)
        projected = jax.vmap(self.fc2)(projected)
        return (normalized + projected) * self.scale.astype(tokens.dtype), state


class PlainMaskDecoder(eqx.Module):
    """Decode retained frozen ViT features into query class and mask logits.

    GroupNorm (the default) has no normalization state: omit ``state`` to
    receive an output, or supply it to receive ``(output, state)``. Reference
    BatchNorm requires ``eqx.nn.make_with_state`` and a named training vmap.
    """

    lateral: tuple[Lateral, ...]
    queries: jax.Array
    blocks: tuple[AttentionBlock, ...]
    decoder_norm: eqx.nn.LayerNorm
    class_head: eqx.nn.Linear
    mask_head: MaskEmbedding
    upscale: tuple[ScaleBlock, ...]
    masked_attention: bool
    config: PMTConfig = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    patch_size: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        config: PMTConfig,
        *,
        num_classes: int,
        num_prefix_tokens: int,
        patch_size: tuple[int, int],
        key: jax.Array,
    ):
        if num_classes <= 0 or num_prefix_tokens < 0 or min(patch_size) < 8:
            raise ValueError("Invalid PMD classes, prefixes, or patch geometry.")
        if patch_size[0] != patch_size[1] or patch_size[0] & (patch_size[0] - 1):
            raise ValueError("PMD requires a square power-of-two patch size.")
        keys = jr.split(key, 5 + len(config.taps) + config.num_blocks)
        self.config = config
        self.masked_attention = config.masked_attention
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_size = patch_size
        self.lateral = tuple(
            Lateral(config, key=keys[i]) for i in range(len(config.taps))
        )
        offset = len(config.taps)
        self.queries = jr.normal(keys[offset], (config.num_queries, config.dim))
        self.blocks = tuple(
            AttentionBlock(
                config.dim,
                config.num_heads,
                key=keys[offset + 1 + i],
                mlp_ratio=config.hidden_dim / config.dim,
                init_values=1.0,
                qkv_bias=False,
                attn_layer=PMDAttention,
                eps=1e-5,
                act_layer="exactgelu",
            )
            for i in range(config.num_blocks)
        )
        tail = offset + 1 + config.num_blocks
        self.decoder_norm = eqx.nn.LayerNorm(config.dim, eps=1e-5)
        self.class_head = eqx.nn.Linear(config.dim, num_classes + 1, key=keys[tail])
        self.mask_head = MaskEmbedding(config.dim, key=keys[tail + 1])
        self.upscale = _build_upscale(config.dim, patch_size[0], key=keys[tail + 2])

    def is_stateful(self) -> bool:
        """Whether lateral BatchNorm requires mutable running statistics."""
        return self.config.norm_layer == "batchnorm"

    def _predict(self, tokens: jax.Array, grid: tuple[int, int]) -> MaskPrediction:
        return _predict_query_masks(
            tokens,
            grid,
            norm=self.decoder_norm,
            class_head=self.class_head,
            mask_head=self.mask_head,
            upscale=self.upscale,
            num_queries=self.config.num_queries,
            num_prefix_tokens=self.num_prefix_tokens,
        )

    def __call__(
        self,
        features: PMTFeatures,
        state: eqx.nn.State | None | object = _MISSING,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        mask_state: PMTMaskState | None = None,
    ) -> PMTOutput | tuple[PMTOutput, eqx.nn.State | None]:
        """Run PMD; explicit state gives the same tuple interface for every norm."""
        supplied = state is not _MISSING
        if self.is_stateful() and not isinstance(state, eqx.nn.State):
            raise ValueError("BatchNorm PMD needs state from eqx.nn.make_with_state.")
        if supplied and state is not None and not isinstance(state, eqx.nn.State):
            raise TypeError("PMD state must be an eqx.nn.State or None.")
        if not inference and key is None:
            raise ValueError("PMD training requires an explicit PRNG key.")
        if self.masked_attention and key is None:
            raise ValueError("Masked PMD attention requires an explicit PRNG key.")
        active_mask_state: PMTMaskState = (
            PMTMaskState(jnp.ones((self.config.num_blocks,)))
            if mask_state is None
            else mask_state
        )
        if active_mask_state.probabilities.shape != (self.config.num_blocks,):
            raise ValueError("PMD mask-state length must equal decoder depth.")
        if (
            len(features.levels) != len(self.lateral)
            or features.taps != self.config.taps
            or features.patch_size != self.patch_size
            or len(features.prefix_tokens) != self.num_prefix_tokens
        ):
            raise ValueError("PMD retained features disagree with decoder geometry.")
        grid = features.grid_size
        expected_shape = (self.num_prefix_tokens + grid[0] * grid[1], self.config.dim)
        if any(level.shape != expected_shape for level in features.levels):
            raise ValueError("PMD feature shape disagrees with prefix/grid metadata.")
        running_state: eqx.nn.State | None = (
            state if isinstance(state, eqx.nn.State) else None
        )
        lateral = []
        for layer, tokens in zip(self.lateral, features.levels, strict=True):
            result, running_state = layer(tokens, running_state, inference=inference)
            lateral.append(result)
        tokens = jnp.sum(jnp.stack(lateral), axis=0)
        tokens = jnp.concatenate((self.queries.astype(tokens.dtype), tokens), axis=0)
        rotary = features.rotary
        if rotary is not None:
            rotary = insert_rotary_identity(
                rotary, index=0, count=self.config.num_queries
            )
        keys = split_for_mode(
            key,
            self.config.num_blocks,
            inference=inference and not self.masked_attention,
        )
        auxiliary = []
        for index, block in enumerate(self.blocks):
            mask = None
            if self.masked_attention:
                prediction = self._predict(tokens, grid)
                auxiliary.append(prediction)
                mask = _query_attention_mask(
                    prediction.mask_logits,
                    grid,
                    active_mask_state.probabilities[index],
                    keys[index],
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            tokens = block(
                tokens,
                key=keys[index],
                inference=inference,
                rotary=rotary,
                attn_mask=mask,
            )
        output = PMTOutput(self._predict(tokens, grid), tuple(auxiliary))
        return (output, running_state) if supplied else output


__all__ = [
    "PMTConfig",
    "PMTFeatures",
    "PMTOutput",
    "PMTMaskState",
    "PlainMaskDecoder",
    "ReferenceBatchNorm",
    "anneal_pmt_mask_state",
]
