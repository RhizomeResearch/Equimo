from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from equimo.core._prng import split_for_mode
from equimo.core.factory import build_model_variant
from equimo.core.intermediates import intermediate_indices
from equimo.core.layers.activation import get_act
from equimo.core.layers.attention import AttentionBlock
from equimo.registry import register_model

__all__ = [
    "CausalTextTransformer",
    "TextTransformerEncoder",
    "TransformerEncoderStack",
    "global_avg_pooling",
    "sam31_text",
]


def global_avg_pooling(
    inputs: Float[Array, "..."],
    compatible_paddings: Int[Array, "..."],
    pooling_dims: Sequence[int],
    epsilon: float = 1e-8,
):
    """Average unpadded tokens over the requested pooling dimensions."""

    output_dtype = inputs.dtype
    accumulation_dtype = (
        jnp.float32 if output_dtype in (jnp.bfloat16, jnp.float16) else output_dtype
    )
    valid_mask = jnp.asarray(1, dtype=accumulation_dtype) - jnp.asarray(
        compatible_paddings, dtype=accumulation_dtype
    )
    masked_inputs = inputs.astype(accumulation_dtype) * valid_mask
    inputs_sum = jnp.sum(masked_inputs, axis=pooling_dims)
    valid_count = jnp.sum(valid_mask, axis=pooling_dims)
    pooled = inputs_sum / (valid_count + jnp.asarray(epsilon, dtype=accumulation_dtype))
    return pooled.astype(output_dtype)


class TransformerEncoderStack(eqx.Module):
    """Stack of modality-neutral transformer blocks for token sequences."""

    blocks: Tuple[AttentionBlock, ...]

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        key: PRNGKeyArray,
        act_layer: Callable | str = jax.nn.gelu,
    ):
        keys = jr.split(key, depth)
        act_layer = get_act(act_layer)
        self.blocks = tuple(
            AttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                key=keys[i],
            )
            for i in range(depth)
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray | None = None,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
        *,
        attn_mask: Optional[Float[Array, ""]] = None,
        ffn_mask: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "seqlen dim"]:
        keys = split_for_mode(key, len(self.blocks), inference=inference)
        for block, block_key in zip(self.blocks, keys):
            x = block(
                x,
                mask=mask,
                attn_mask=attn_mask,
                ffn_mask=ffn_mask,
                inference=inference,
                key=block_key,
            )
        return x

    def intermediate_features(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray | None = None,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
        indices: Sequence[int] | None = None,
        n_last_blocks: int | None = None,
        *,
        attn_mask: Optional[Float[Array, ""]] = None,
        ffn_mask: Optional[Float[Array, ""]] = None,
    ) -> Tuple[Float[Array, "seqlen dim"], ...]:
        """Return selected native transformer block outputs."""

        wanted = intermediate_indices(
            len(self.blocks),
            indices=indices,
            n_last_blocks=n_last_blocks,
        )
        keys = split_for_mode(key, len(self.blocks), inference=inference)
        outputs = []
        for i, (block, block_key) in enumerate(zip(self.blocks, keys)):
            x = block(
                x,
                mask=mask,
                attn_mask=attn_mask,
                ffn_mask=ffn_mask,
                inference=inference,
                key=block_key,
            )
            if i in wanted:
                outputs.append(x)
        return tuple(outputs)


@register_model("text_transformer_encoder", modality="language")
class TextTransformerEncoder(eqx.Module):
    """Transformer text encoder that returns pooled sequence embeddings."""

    token_embedding: eqx.nn.Embedding
    transformer: TransformerEncoderStack
    ln_final: eqx.nn.LayerNorm

    dim: int = eqx.field(static=True)
    scale_sqrt_depth: bool = eqx.field(static=True)
    temperature: float = eqx.field(static=True, default=-1)

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
        vocab_size: int,
        *,
        key: PRNGKeyArray,
        scale_sqrt_depth: bool = True,
        act_layer: Callable | str = jax.nn.gelu,
        temperature: float = -1.0,
    ):
        key_emb, key_trans = jr.split(key, 2)
        self.dim = dim
        self.scale_sqrt_depth = scale_sqrt_depth
        self.temperature = temperature
        self.token_embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=dim,
            key=key_emb,
        )
        self.transformer = TransformerEncoderStack(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            key=key_trans,
        )
        self.ln_final = eqx.nn.LayerNorm(dim)

    def posemb(
        self,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        seq_len: Optional[int] = None,
        position: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if position is None:
            if seq_len is None:
                raise ValueError("If position is None, seq_len must be provided.")
            position = jnp.arange(seq_len, dtype=jnp.float32)
        elif position.ndim != 1:
            raise ValueError("position must be a 1D array.")

        num_timescales = self.dim // 2
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / jnp.maximum(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * jnp.exp(
            jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
        )
        scaled_time = position[:, None] * inv_timescales[None, :]
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
        if self.dim % 2 == 1:
            signal = jnp.pad(signal, ((0, 0), (0, 1)), mode="constant")
        return signal

    def _prepare_tokens(
        self, ids: jax.Array, padding_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        seq_len = ids.shape[0]
        x = jax.vmap(self.token_embedding)(ids)
        valid_mask = (padding_mask == 0).astype(x.dtype)
        if self.scale_sqrt_depth:
            x = x * jnp.asarray(self.dim**0.5, dtype=x.dtype)

        x = x + self.posemb(seq_len=seq_len).astype(x.dtype)
        return x, valid_mask

    def features(
        self,
        ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Float[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray | None = None,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        x, valid_mask = self._prepare_tokens(ids, padding_mask)
        x = self.transformer(
            x,
            mask=valid_mask[None, None, :],
            ffn_mask=valid_mask[:, None],
            inference=inference,
            key=key,
        )
        return jax.vmap(self.ln_final)(x)

    def intermediate_features(
        self,
        ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Float[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray | None = None,
        inference: Optional[bool] = None,
        indices: Sequence[int] | None = None,
        n_last_blocks: int | None = None,
    ) -> Tuple[Float[Array, "seqlen dim"], ...]:
        """Return selected native transformer block outputs."""

        x, valid_mask = self._prepare_tokens(ids, padding_mask)
        return self.transformer.intermediate_features(
            x,
            mask=valid_mask[None, None, :],
            ffn_mask=valid_mask[:, None],
            inference=inference,
            key=key,
            indices=indices,
            n_last_blocks=n_last_blocks,
        )

    def __call__(
        self,
        ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Float[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray | None = None,
        inference: Optional[bool] = None,
    ) -> Float[Array, "dim"]:
        x = self.features(ids, padding_mask, key=key, inference=inference)
        return global_avg_pooling(
            x, compatible_paddings=padding_mask[:, None], pooling_dims=[0]
        )


@register_model("causal_text_transformer", modality="language")
class CausalTextTransformer(eqx.Module):
    """CLIP-style text transformer that returns per-token features.

    Every token attends only to itself and earlier tokens, so its output never
    depends on what follows it. ``padding_mask`` uses ``True`` for padding and
    removes padded positions from the attention keys. Token id 0 is a real CLIP
    token, so padding cannot be inferred from the ids.
    """

    token_embedding: eqx.nn.Embedding
    position_embedding: Float[Array, "context_length dim"]
    transformer: TransformerEncoderStack
    ln_final: eqx.nn.LayerNorm
    projection: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
        vocab_size: int,
        context_length: int,
        *,
        key: PRNGKeyArray,
        out_dim: int | None = None,
        act_layer: Callable | str = jax.nn.gelu,
    ):
        key_emb, key_pos, key_trans, key_proj = jr.split(key, 4)
        self.token_embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=dim,
            key=key_emb,
        )
        self.position_embedding = 0.01 * jr.normal(key_pos, (context_length, dim))
        self.transformer = TransformerEncoderStack(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            key=key_trans,
        )
        self.ln_final = eqx.nn.LayerNorm(dim)
        self.projection = (
            eqx.nn.Identity()
            if out_dim is None
            else eqx.nn.Linear(dim, out_dim, key=key_proj)
        )

    def _attention_mask(
        self,
        padding_mask: Bool[Array, "seqlen"],  # noqa: F821
    ) -> Bool[Array, "seqlen seqlen"]:
        seq_len = padding_mask.shape[0]
        with jax.ensure_compile_time_eval():
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        return causal & (padding_mask == 0)[None, :]

    def features(
        self,
        token_ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Bool[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Float[Array, "seqlen out_dim"]:
        x = jax.vmap(self.token_embedding)(token_ids) + self.position_embedding
        x = self.transformer(
            x,
            attn_mask=self._attention_mask(padding_mask),
            inference=inference,
            key=key,
        )
        return jax.vmap(self.projection)(jax.vmap(self.ln_final)(x))

    def __call__(
        self,
        token_ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Bool[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Float[Array, "seqlen out_dim"]:
        return self.features(token_ids, padding_mask, key=key, inference=inference)


_CAUSAL_TEXT_BASE_CFG: dict = {
    "dim": 1024,
    "mlp_ratio": 4.0,
    "depth": 24,
    "num_heads": 16,
    "vocab_size": 49408,
    "context_length": 32,
    "act_layer": "exactgelu",
}

_CAUSAL_TEXT_REGISTRY: dict[str, tuple[dict, dict]] = {
    "sam31_text": (_CAUSAL_TEXT_BASE_CFG, {"out_dim": 256}),
}


def sam31_text(
    pretrained: bool = False,
    inference_mode: bool = True,
    **kwargs,
) -> CausalTextTransformer:
    """Build the SAM 3.1 text encoder."""
    return build_model_variant(
        CausalTextTransformer,
        _CAUSAL_TEXT_REGISTRY,
        "sam31_text",
        pretrained=pretrained,
        inference_mode=inference_mode,
        pretrained_label="SAM 3.1 text",
        allow_pretrained_overrides=False,
        **kwargs,
    )
