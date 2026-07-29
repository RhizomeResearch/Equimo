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
from equimo.core.factory import build_model_variant
from equimo.core.layers import BlockChunk, RMSNormGated
from equimo.registry import register_model
from equimo.time_series.layers import (
    PatchEncoder,
    ResidualMlp,
    T0Block,
    get_layer,
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
        if group_every_n > 0 and num_layers % group_every_n:
            raise ValueError("group_every_n must divide num_layers")
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
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
        self.quantile_levels = tuple(sorted(float(q) for q in quantile_levels))
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

_T0_REGISTRY: dict[str, tuple[dict, dict]] = {
    "t0": (_T0_BASE_CFG, {}),
    "t0_alpha": (_T0_BASE_CFG, {}),
}

_T0_PRETRAINED_IDENTIFIERS = {"t0": "t0_alpha"}


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


def _build_t0(
    variant: str,
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
        pretrained_identifiers=_T0_PRETRAINED_IDENTIFIERS,
        **overrides,
    )


def t0(
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **kwargs,
) -> T0:
    return _build_t0(
        "t0",
        pretrained=pretrained,
        inference_mode=inference_mode,
        key=key,
        **kwargs,
    )


def t0_alpha(
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **kwargs,
) -> T0:
    return _build_t0(
        "t0_alpha",
        pretrained=pretrained,
        inference_mode=inference_mode,
        key=key,
        **kwargs,
    )
