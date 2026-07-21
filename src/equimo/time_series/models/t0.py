# ty: ignore[invalid-assignment]
"""T0 patch-transformer time-series foundation model."""

__all__ = ["T0", "load_t0_weights", "t0", "t0_alpha"]

import json
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.conversion.utils import stringify_name
from equimo.core.layers import Attention, BlockChunk, Mlp, RMSNormGated, SwiGluFused
from equimo.registry import register_model


def _linear(layer: eqx.nn.Linear, x: Array) -> Array:
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


class ResidualMlp(eqx.Module):
    """Reference residual projection, backed by Equimo's Mlp."""

    mlp: Mlp
    residual_layer: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, *, key):
        key1, key2 = jr.split(key)
        self.mlp = Mlp(
            input_size,
            hidden_dim=hidden_size,
            out_dim=output_size,
            act_layer="relu",
            dropout_rate=0.0,
            key=key1,
        )
        self.residual_layer = eqx.nn.Linear(input_size, output_size, key=key2)

    def __call__(self, x, *, key, inference=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = self.mlp(flat, key=key, inference=inference)
        out += jax.vmap(self.residual_layer)(flat)
        return out.reshape(*shape[:-1], -1)


class PatchEncoder(eqx.Module):
    projection: ResidualMlp
    type_embeddings: eqx.nn.Embedding
    patch_size: int = eqx.field(static=True)

    def __init__(self, embed_dim, patch_size, *, key):
        key1, key2 = jr.split(key)
        self.projection = ResidualMlp(patch_size * 3, embed_dim, embed_dim, key=key1)
        self.type_embeddings = eqx.nn.Embedding(3, embed_dim, key=key2)
        self.patch_size = patch_size

    def __call__(self, values, mask, variate_type, *, key, inference=None):
        time = jnp.arange(self.patch_size, dtype=values.dtype) / self.patch_size
        time = jnp.broadcast_to(time, values.shape)
        validity = (mask == 0).astype(values.dtype)
        x = self.projection(
            jnp.concatenate((values, time, validity), axis=-1),
            key=key,
            inference=inference,
        )
        return x + self.type_embeddings.weight[jnp.maximum(variate_type[:, :, 0], 0)]


def _rotate_half(x: Array) -> Array:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)


def _xpos(q: Array, k: Array) -> tuple[Array, Array]:
    """Exact rotary-embedding-torch 0.8.x RoPE + XPos."""
    dim, seq_len = q.shape[-1], q.shape[-2]
    positions = jnp.arange(seq_len, dtype=q.dtype)
    frequencies = 1.0 / (10_000 ** (jnp.arange(0, dim, 2) / dim))
    angles = jnp.repeat(jnp.outer(positions, frequencies), 2, axis=-1)
    base = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
    power = (positions - (seq_len - 1) // 2) / 512.0
    scale = jnp.repeat(base[None] ** power[:, None], 2, axis=-1)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return (
        (q * cos + _rotate_half(q) * sin) * scale,
        (k * cos + _rotate_half(k) * sin) / scale,
    )


class AxisAttention(eqx.Module):
    """T0 axis routing and XPos around Equimo's Attention parameters."""

    attention: Attention
    attention_type: str = eqx.field(static=True)

    def __init__(self, dim, num_heads, dropout, attention_type, *, key):
        self.attention = Attention(
            dim,
            num_heads,
            qk_norm=True,
            norm_layer=RMSNormGated,
            eps=1e-8,
            attn_drop=dropout,
            proj_drop=dropout,
            key=key,
        )
        self.attention_type = attention_type

    def __call__(self, x, mask, *, key, inference=None):
        if self.attention_type == "group":
            x = jnp.swapaxes(x, 0, 1)
        key1, key2 = jr.split(key)
        attn = self.attention
        qkv = _linear(attn.qkv, x).reshape(
            *x.shape[:-1], 3, attn.num_heads, attn.head_dim
        )
        q, k, v = jnp.moveaxis(qkv, -3, 0)
        q, k, v = (jnp.swapaxes(value, -3, -2) for value in (q, k, v))
        q, k = attn.q_norm(q), attn.k_norm(k)  # ty: ignore[call-non-callable]
        if self.attention_type == "time":
            q, k = _xpos(q, k)
        weights = jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(attn.head_dim)
        weights = jnp.where(mask[..., None, :, :], weights, -jnp.inf)
        weights = jnp.nan_to_num(
            jax.nn.softmax(weights.astype(jnp.float32), axis=-1)
        ).astype(x.dtype)
        weights = attn.attn_drop(weights, key=key1, inference=inference)
        x = jnp.einsum("...hqk,...hkd->...hqd", weights, v)
        x = jnp.swapaxes(x, -3, -2).reshape(*x.shape[:-3], x.shape[-2], -1)
        x = _linear(attn.proj, x)
        x = attn.proj_drop(x, key=key2, inference=inference)
        return jnp.swapaxes(x, 0, 1) if self.attention_type == "group" else x


class T0Block(eqx.Module):
    attn_norm: RMSNormGated
    attn: AxisAttention
    ffn_norm: RMSNormGated
    ffn: SwiGluFused
    ffn_dropout: eqx.nn.Dropout
    attention_type: str = eqx.field(static=True)

    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        dropout,
        attention_type,
        *,
        key,
        drop_path=0.0,
    ):
        del drop_path
        key1, key2 = jr.split(key)
        self.attn_norm = RMSNormGated(dim, eps=1e-8)
        self.attn = AxisAttention(dim, num_heads, dropout, attention_type, key=key1)
        self.ffn_norm = RMSNormGated(dim, eps=1e-8)
        # SwiGluFused applies the LLaMA 2/3 contraction internally.
        self.ffn = SwiGluFused(
            dim, hidden_dim=hidden_dim * 3 // 2, dropout_rate=0.0, key=key2
        )
        self.ffn_dropout = eqx.nn.Dropout(dropout)
        self.attention_type = attention_type

    def __call__(
        self,
        x,
        *,
        time_attn_mask,
        group_attn_mask,
        key,
        inference=None,
    ):
        key1, key2, key3 = jr.split(key, 3)
        mask = time_attn_mask if self.attention_type == "time" else group_attn_mask
        x += self.attn(self.attn_norm(x), mask, key=key1, inference=inference)
        shape = x.shape
        flat = self.ffn_norm(x).reshape(-1, shape[-1])
        out = self.ffn(flat, key=key2, inference=inference).reshape(shape)
        return x + self.ffn_dropout(out, key=key3, inference=inference)


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
        key: PRNGKeyArray,
    ):
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if group_every_n > 0 and num_layers % group_every_n:
            raise ValueError("group_every_n must divide num_layers")
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        key_encoder, key_blocks, key_decoder = jr.split(key, 3)
        self.patch_encoder = PatchEncoder(embed_dim, patch_size, key=key_encoder)
        attention_types = [
            "group" if group_every_n > 0 and (i + 1) % group_every_n == 0 else "time"
            for i in range(num_layers)
        ]
        self.blocks = (
            BlockChunk(
                depth=num_layers,
                module=T0Block,
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
        self.quantile_levels = tuple(sorted(float(q) for q in quantile_levels))
        self.decoder = ResidualMlp(
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
        values, mask, group_ids, variate_type = self._patch(
            values, mask, group_ids, variate_type
        )
        key_encoder, key_blocks, key_decoder = jr.split(key, 3)
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
        x = self.out_norm(x)
        x = self.decoder(x, key=key_decoder, inference=inference)
        x = x.reshape(*x.shape[:-1], self.patch_size, len(self.quantile_levels))
        first = x[..., :1]
        return jnp.concatenate(
            (first, first + jnp.cumsum(jax.nn.softplus(x[..., 1:]), axis=-1)), axis=-1
        )


_T0_REGISTRY = {
    "t0": {},
    "t0_alpha": {
        "embed_dim": 512,
        "num_layers": 24,
        "num_heads": 8,
        "mlp_hidden_dim": 2048,
        "patch_size": 32,
        "group_every_n": 3,
        "dropout": 0.1,
        "quantile_levels": (0.1, 0.25, 0.5, 0.75, 0.9),
    },
}


def _checkpoint_name(name: str) -> str:
    name = name.replace("blocks.0.blocks.", "transformer.layers.")
    name = name.replace(
        "patch_encoder.projection.mlp.fc1.",
        "patch_encoder.projection.mlp.hidden_layer.",
    )
    name = name.replace(
        "patch_encoder.projection.mlp.fc2.",
        "patch_encoder.projection.mlp.output_layer.",
    )
    name = name.replace("decoder.mlp.fc1.", "decoder.mlp.hidden_layer.")
    name = name.replace("decoder.mlp.fc2.", "decoder.mlp.output_layer.")
    name = name.replace(".attn_norm.w", ".attention_block.norm.scale")
    name = name.replace(".attn.attention.qkv.", ".attention_block.attention.wQKV.")
    name = name.replace(".attn.attention.proj.", ".attention_block.attention.wO.")
    name = name.replace(
        ".attn.attention.q_norm.w", ".attention_block.attention.q_norm.scale"
    )
    name = name.replace(
        ".attn.attention.k_norm.w", ".attention_block.attention.k_norm.scale"
    )
    name = name.replace(".ffn_norm.w", ".norm.scale")
    name = name.replace(".ffn.w12.", ".mlp.0.")
    name = name.replace(".ffn.w3.", ".mlp.2.")
    return name.replace("out_norm.w", "transformer.out_norm.scale")


def _safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as file:
        if file.read(7) == b"version":
            raise ValueError(f"{path} is a Git LFS pointer, not a safetensors file")
        file.seek(0)
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    data_start = 8 + header_size
    arrays = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(
                f"Unsupported safetensors dtype {info['dtype']} for {name}"
            )
        start, stop = info["data_offsets"]
        arrays[name] = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=data_start + start,
            shape=tuple(info["shape"]),
        )
        if arrays[name].nbytes != stop - start:
            raise ValueError(f"Invalid safetensors offsets for {name}")
    return arrays


def load_t0_weights(model: T0, path: str | Path) -> T0:
    """Load a native TFC T0 ``model.safetensors`` checkpoint strictly."""
    state = _safetensors(Path(path))
    dynamic, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(dynamic)
    converted, used = [], set()
    for tree_path, leaf in flat:
        name = _checkpoint_name(stringify_name(tree_path))
        if name not in state:
            raise KeyError(f"No T0 checkpoint tensor for {name!r}")
        array = state[name]
        if tuple(array.shape) != tuple(leaf.shape):
            raise ValueError(
                f"{name}: expected {tuple(leaf.shape)}, got {tuple(array.shape)}"
            )
        converted.append(jnp.asarray(np.asarray(array)))
        used.add(name)
    leftover = set(state) - used - {"head.quantile_levels"}
    if leftover:
        raise KeyError(f"Unused T0 checkpoint tensors: {sorted(leftover)}")
    tree = jax.tree_util.tree_unflatten(treedef, converted)
    return eqx.nn.inference_mode(eqx.combine(tree, static), value=True)


def _build_t0(variant, *, pretrained=False, weights=None, key=None, **overrides):
    if key is None:
        key = jr.PRNGKey(42)
    model = T0(**(_T0_REGISTRY[variant] | overrides), key=key)
    if pretrained:
        if weights is None:
            raise ValueError(
                "pretrained=True requires weights=path/to/model.safetensors"
            )
        model = load_t0_weights(cast(T0, model), weights)
    return model


def t0(**kwargs) -> T0:
    return _build_t0("t0", **kwargs)


def t0_alpha(**kwargs) -> T0:
    return _build_t0("t0_alpha", **kwargs)
