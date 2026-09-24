"""Encoder-only mask transformer for image segmentation."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from equimo.core._prng import split_for_mode
from equimo.core.layers.attention import AttentionBlock
from equimo.core.layers.norm import LayerNorm2d
from equimo.core.layers.rotary import insert_rotary_identity
from equimo.registry import register_model
from equimo.vision.models.vit import VisionTransformer
from equimo.vision.segmentation import (
    _anneal_mask_probabilities,
    _query_attention_mask,
)


class MaskPrediction(eqx.Module):
    """Unbatched query class and mask logits."""

    class_logits: jax.Array
    mask_logits: jax.Array


class EoMTOutput(eqx.Module):
    """Final prediction and auxiliary predictions in execution order."""

    final: MaskPrediction
    auxiliary: tuple[MaskPrediction, ...]


class EoMTMaskState(eqx.Module):
    """Per-block mask probabilities and optimizer-step position."""

    probabilities: jax.Array
    step: jax.Array

    def __init__(self, probabilities: jax.Array, step: int | jax.Array = 0):
        probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
        if probabilities.ndim != 1:
            raise ValueError("Mask probabilities must be a one-dimensional array.")
        self.probabilities = probabilities
        self.step = jnp.asarray(step, dtype=jnp.int32)


def anneal_mask_state(
    step: int | jax.Array,
    start_steps: tuple[int, ...],
    end_steps: tuple[int, ...],
    *,
    power: float = 0.9,
) -> EoMTMaskState:
    """Polynomially remove masked attention at declared optimizer steps."""
    return EoMTMaskState(
        _anneal_mask_probabilities(step, start_steps, end_steps, power), step
    )


class ScaleBlock(eqx.Module):
    """Twofold spatial upscaling used by the EoMT mask head."""

    conv1: eqx.nn.ConvTranspose2d
    conv2: eqx.nn.Conv2d
    norm: LayerNorm2d

    def __init__(self, channels: int, *, key: jax.Array):
        key1, key2 = jr.split(key)
        self.conv1 = eqx.nn.ConvTranspose2d(channels, channels, 2, stride=2, key=key1)
        self.conv2 = eqx.nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, use_bias=False, key=key2
        )
        self.norm = LayerNorm2d(channels, eps=1e-6)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(self.conv2(jax.nn.gelu(self.conv1(x), approximate=False)))


class MaskEmbedding(eqx.Module):
    """Three linear layers with exact GELU between them."""

    layers: tuple[eqx.nn.Linear, ...]

    def __init__(self, dim: int, *, key: jax.Array):
        self.layers = tuple(
            eqx.nn.Linear(dim, dim, key=subkey) for subkey in jr.split(key, 3)
        )

    def __call__(self, queries: jax.Array) -> jax.Array:
        x = queries
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(jax.vmap(layer)(x), approximate=False)
        return jax.vmap(self.layers[-1])(x)


def _build_upscale(
    dim: int, patch_size: int, *, key: jax.Array
) -> tuple[ScaleBlock, ...]:
    """Build the twofold upscaling blocks that bring patch features to stride 4."""
    count = max(1, patch_size.bit_length() - 3)
    return tuple(ScaleBlock(dim, key=subkey) for subkey in jr.split(key, count))


def _predict_query_masks(
    tokens: jax.Array,
    grid: tuple[int, int],
    *,
    norm: eqx.Module,
    class_head: eqx.nn.Linear,
    mask_head: MaskEmbedding,
    upscale: tuple[ScaleBlock, ...],
    num_queries: int,
    num_prefix_tokens: int,
) -> MaskPrediction:
    """Predict query class and mask logits from query/prefix/patch tokens."""
    normalized = jax.vmap(norm)(tokens)
    queries = normalized[:num_queries]
    patches = normalized[num_queries + num_prefix_tokens :]
    features = patches.T.reshape(tokens.shape[-1], *grid)
    for layer in upscale:
        features = layer(features)
    return MaskPrediction(
        class_logits=jax.vmap(class_head)(queries),
        mask_logits=jnp.einsum("qc,chw->qhw", mask_head(queries), features),
    )


@register_model("eomt", modality="vision")
class EoMT(eqx.Module):
    """Inject learned queries into the final blocks of a plain ViT."""

    backbone: VisionTransformer
    queries: jax.Array
    class_head: eqx.nn.Linear
    mask_head: MaskEmbedding
    upscale: tuple[ScaleBlock, ...]
    num_classes: int = eqx.field(static=True)
    num_queries: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    masked_attention: bool

    def __init__(
        self,
        backbone: VisionTransformer,
        num_classes: int,
        num_queries: int = 100,
        num_blocks: int = 4,
        *,
        key: jax.Array,
        masked_attention: bool = True,
    ):
        if not isinstance(backbone, VisionTransformer):
            raise TypeError("EoMT requires a VisionTransformer backbone.")
        if num_classes <= 0 or num_queries <= 0:
            raise ValueError("Class and query counts must be positive.")
        if not 1 <= num_blocks <= backbone.num_blocks:
            raise ValueError("Query blocks must be between 1 and backbone depth.")
        if not backbone.dynamic_img_size:
            raise ValueError("EoMT requires a dynamic image-size ViT.")
        if not isinstance(backbone.head, eqx.nn.Identity):
            raise ValueError("EoMT requires a feature-only ViT backbone.")
        for chunk in backbone.blocks:
            if (
                not isinstance(chunk.posemb, eqx.nn.Identity)
                or chunk.downsample is not None
            ):
                raise ValueError("EoMT requires plain transformer block chunks.")
            if any(
                not isinstance(block, AttentionBlock) for block in chunk.blocks or ()
            ):
                raise ValueError("EoMT requires plain attention blocks.")
        patch_size, patch_width = backbone.patch_embed.patch_size
        if patch_size != patch_width:
            raise ValueError("EoMT requires square patches.")
        if patch_size < 8 or patch_size & (patch_size - 1):
            raise ValueError("EoMT requires power-of-two patch size of at least 8.")

        query_key, class_key, mask_key, upscale_key = jr.split(key, 4)
        self.backbone = backbone
        self.queries = jr.normal(query_key, (num_queries, backbone.dim))
        self.class_head = eqx.nn.Linear(backbone.dim, num_classes + 1, key=class_key)
        self.mask_head = MaskEmbedding(backbone.dim, key=mask_key)
        self.upscale = _build_upscale(backbone.dim, patch_size, key=upscale_key)
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_blocks = num_blocks
        self.masked_attention = masked_attention

    def _predict(self, tokens: jax.Array, height: int, width: int) -> MaskPrediction:
        return _predict_query_masks(
            tokens,
            (height, width),
            norm=self.backbone.norm,
            class_head=self.class_head,
            mask_head=self.mask_head,
            upscale=self.upscale,
            num_queries=self.num_queries,
            num_prefix_tokens=self.backbone.num_prefix_tokens,
        )

    def _attention_mask(
        self,
        mask_logits: jax.Array,
        *,
        height: int,
        width: int,
        probability: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        return _query_attention_mask(
            mask_logits,
            (height, width),
            probability,
            key,
            num_prefix_tokens=self.backbone.num_prefix_tokens,
        )

    def _forward(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None,
        inference: bool,
        mask_state: EoMTMaskState | None,
        predict_final: bool,
        collect_trace: bool,
    ) -> tuple[EoMTOutput | None, jax.Array, tuple[jax.Array, ...]]:
        if x.ndim != 3:
            raise ValueError("EoMT expects an unbatched (C, H, W) image.")
        if key is None and self.masked_attention:
            raise ValueError("Masked attention requires an explicit PRNG key.")
        if not inference and key is None:
            raise ValueError("Training requires an explicit PRNG key.")
        if mask_state is None:
            mask_state = EoMTMaskState(jnp.ones((self.num_blocks,)))
        if mask_state.probabilities.shape != (self.num_blocks,):
            raise ValueError("Mask state length must equal query block count.")

        block_keys = split_for_mode(
            key,
            self.backbone.num_blocks + 2,
            inference=inference and not self.masked_attention,
        )
        tokens, height, width, rotary = self.backbone.prepare_tokens(
            x, key=block_keys[0], inference=inference
        )
        auxiliary = []
        trace = []
        insertion = self.backbone.num_blocks - self.num_blocks
        for index in range(self.backbone.num_blocks):
            if index == insertion:
                tokens = jnp.concatenate(
                    (self.queries.astype(tokens.dtype), tokens), axis=0
                )
                if rotary is not None:
                    rotary = insert_rotary_identity(
                        rotary, index=0, count=self.num_queries
                    )
            attention_mask = None
            if self.masked_attention and index >= insertion:
                if collect_trace:
                    trace.append(jax.vmap(self.backbone.norm)(tokens))
                prediction = self._predict(tokens, height, width)
                auxiliary.append(prediction)
                attention_mask = self._attention_mask(
                    prediction.mask_logits,
                    height=height,
                    width=width,
                    probability=mask_state.probabilities[index - insertion],
                    key=block_keys[index + 1],
                )
            tokens = self.backbone.block_at(index)(
                tokens,
                key=block_keys[index + 1],
                inference=inference,
                rotary=rotary,
                attn_mask=attention_mask,
            )
        final_tokens = jax.vmap(self.backbone.norm)(tokens)
        if collect_trace:
            trace.append(final_tokens)
        output = (
            EoMTOutput(self._predict(tokens, height, width), tuple(auxiliary))
            if predict_final
            else None
        )
        return output, final_tokens, tuple(trace)

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
        mask_state: EoMTMaskState | None = None,
    ) -> EoMTOutput:
        """Return query predictions; ``mask_state`` controls training masks."""
        output, _, _ = self._forward(
            x,
            key=key,
            inference=inference,
            mask_state=mask_state,
            predict_final=True,
            collect_trace=False,
        )
        assert output is not None
        return output

    def joint_features(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
        mask_state: EoMTMaskState | None = None,
    ) -> jax.Array:
        """Return final normalized joint tokens in query/prefix/patch order."""
        return self._forward(
            x,
            key=key,
            inference=inference,
            mask_state=mask_state,
            predict_final=False,
            collect_trace=False,
        )[1]

    def features(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
        mask_state: EoMTMaskState | None = None,
    ) -> jax.Array:
        """Return joint tokens; use ``joint_features`` for an explicit name."""
        return self.joint_features(
            x, key=key, inference=inference, mask_state=mask_state
        )

    def token_trace(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
        mask_state: EoMTMaskState | None = None,
    ) -> tuple[jax.Array, ...]:
        """Normalized joint tokens before each prediction, in output order."""
        return self._forward(
            x,
            key=key,
            inference=inference,
            mask_state=mask_state,
            predict_final=False,
            collect_trace=True,
        )[2]


def mask_free_eomt(model: EoMT, state: EoMTMaskState) -> EoMT:
    """Construct the terminal inference model after mask annealing has ended."""
    if state.probabilities.shape != (model.num_blocks,):
        raise ValueError("Mask state length must equal query block count.")
    if not bool(jnp.all(state.probabilities == 0)):
        raise ValueError("Mask-free inference requires terminal mask probabilities.")
    return eqx.tree_at(lambda m: m.masked_attention, model, False)


__all__ = [
    "EoMT",
    "EoMTMaskState",
    "EoMTOutput",
    "MaskPrediction",
    "ScaleBlock",
    "anneal_mask_state",
    "mask_free_eomt",
]
