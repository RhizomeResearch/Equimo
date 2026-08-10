# ty: ignore[unknown-argument]
# ty: ignore[invalid-assignment]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[call-non-callable]
from typing import Callable, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core._prng import split_for_mode
from equimo.core.layers.activation import get_act
from equimo.core.layers.dropout import DropPathAdd, split_drop_path
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.norm import LayerScale, get_norm, maybe_layer_scale
from equimo.core.layers.rotary import RotaryFactors, apply_rotary_qk
from equimo.core.layers._registry import make_get, make_register

_ATTN_REGISTRY: dict[str, type[eqx.Module]] = {}
_ATTN_BLOCK_REGISTRY: dict[str, type[eqx.Module]] = {}


register_attn = make_register(_ATTN_REGISTRY)


get_attn = make_get(_ATTN_REGISTRY)


register_attn_block = make_register(_ATTN_BLOCK_REGISTRY)


get_attn_block = make_get(_ATTN_BLOCK_REGISTRY)


def _apply_module_last_dim(module: eqx.Module, x: jax.Array) -> jax.Array:
    """Apply a vector module independently over every leading dimension."""

    leading_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    output = jax.vmap(module)(flat)
    return output.reshape(*leading_shape, output.shape[-1])


def _broadcast_attention_mask(
    mask: jax.Array,
    *,
    x_ndim: int,
    shape: tuple[int, ...],
) -> jax.Array:
    """Broadcast a nonzero-is-allowed mask to attention score shape."""

    mask = jnp.asarray(mask) != 0
    if mask.ndim == x_ndim:
        mask = mask[..., None, :, :]
    try:
        return jnp.broadcast_to(mask, shape)
    except ValueError as error:
        raise ValueError(
            f"Attention mask shape {mask.shape} is not broadcastable to "
            f"attention scores with shape {shape}."
        ) from error


@register_attn()
class Attention(eqx.Module):
    """Multi-head self attention over tensors shaped ``(..., seq, dim)``.

    Masks use nonzero/``True`` entries for allowed query-key pairs. They may be
    batch-aligned with shape ``(..., query, key)`` or directly broadcastable to
    ``(..., heads, query, key)``. Fully masked query rows receive zero attention
    probability mass.
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: eqx.Module
    k_norm: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: str | type[eqx.Module] = "layernorm",
        eps: float = 1e-5,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_qkv, key_proj = jr.split(key, 2)
        norm_layer = get_norm(norm_layer)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=key_qkv)
        self.proj = eqx.nn.Linear(dim, dim, use_bias=proj_bias, key=key_proj)
        self.q_norm = (
            norm_layer(self.head_dim, eps=eps) if qk_norm else eqx.nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, eps=eps) if qk_norm else eqx.nn.Identity()
        )
        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "... seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        mask: Optional[Array] = None,
        rotary: Optional[RotaryFactors] = None,
    ) -> Float[Array, "... seqlen dim"]:
        key1, key2 = split_for_mode(key, 2, inference=inference)

        qkv = _apply_module_last_dim(self.qkv, x).reshape(
            *x.shape[:-1], 3, self.num_heads, self.head_dim
        )
        q, k, v = jnp.moveaxis(qkv, -3, 0)
        q, k, v = (jnp.swapaxes(value, -3, -2) for value in (q, k, v))
        q = _apply_module_last_dim(self.q_norm, q)
        k = _apply_module_last_dim(self.k_norm, k)

        if rotary is not None:
            if rotary.feature_dim != self.head_dim:
                raise ValueError(
                    f"Rotary factor dimension ({rotary.feature_dim}) must equal "
                    f"head_dim ({self.head_dim})."
                )
            q, k = apply_rotary_qk(q, k, rotary)

        attn = (
            jnp.einsum("...hqd,...hkd->...hqk", q, k) / jnp.sqrt(self.head_dim)
        ).astype(jnp.float32)
        if mask is not None:
            allowed = _broadcast_attention_mask(
                mask,
                x_ndim=x.ndim,
                shape=attn.shape,
            )
            attn = jnp.where(allowed, attn, -jnp.inf)
        attn = jax.nn.softmax(attn, axis=-1)
        if mask is not None:
            attn = jnp.where(jnp.any(allowed, axis=-1, keepdims=True), attn, 0)
        attn = attn.astype(x.dtype)
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x = jnp.einsum("...hqk,...hkd->...hqd", attn, v)
        x = jnp.swapaxes(x, -3, -2).reshape(*x.shape[:-3], x.shape[-2], self.dim)
        x = _apply_module_last_dim(self.proj, x)
        return self.proj_drop(x, inference=inference, key=key2)


@register_attn_block()
class AttentionBlock(eqx.Module):
    """Pre-norm transformer block for sequence tensors.

    ``mask`` applies to both attention and the feed-forward layer by default.
    ``attn_mask`` and ``ffn_mask`` override it when those layers need masks with
    different axis layouts.
    """

    prenorm: eqx.Module
    postnorm: eqx.Module
    norm: eqx.Module
    ls1: LayerScale | eqx.nn.Identity
    ls2: LayerScale | eqx.nn.Identity
    attn: eqx.Module
    mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 4.0,
        drop_path: float | List[float] = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: str | Callable = "gelu",
        attn_layer: str | type[eqx.Module] = "attention",
        ffn_layer: str | type[eqx.Module] = "mlp",
        ffn_bias: bool = True,
        ffn_norm: bool = False,
        ffn_kwargs: dict = {},
        norm_layer: str | type[eqx.Module] = "layernorm",
        post_attention_norm: bool = False,
        init_values: float | None = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        key_attn, key_mlp = jr.split(key, 2)
        act_layer = get_act(act_layer)
        attn_layer = get_attn(attn_layer)
        ffn_layer = get_ffn(ffn_layer)
        norm_layer = get_norm(norm_layer)

        dr1, dr2 = split_drop_path(drop_path)

        self.prenorm = norm_layer(dim, eps=eps)
        self.postnorm = (
            norm_layer(dim, eps=eps) if post_attention_norm else eqx.nn.Identity()
        )
        self.norm = norm_layer(dim, eps=eps)
        self.attn = attn_layer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            eps=eps,
            key=key_attn,
        )
        self.mlp = ffn_layer(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if ffn_norm else None,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            eps=eps,
            key=key_mlp,
            **ffn_kwargs,
        )
        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)
        self.ls1 = maybe_layer_scale(dim, axis=1, init_values=init_values)
        self.ls2 = maybe_layer_scale(dim, axis=1, init_values=init_values)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        *,
        mask: Optional[Float[Array, ""]] = None,
        attn_mask: Optional[Float[Array, ""]] = None,
        ffn_mask: Optional[Float[Array, ""]] = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_mlp, key_dr1, key_dr2 = split_for_mode(
            key, 4, inference=inference
        )
        attn_mask = mask if attn_mask is None else attn_mask
        ffn_mask = mask if ffn_mask is None else ffn_mask
        attn_kwargs = {"mask": attn_mask} if attn_mask is not None else {}
        ffn_kwargs = {"mask": ffn_mask} if ffn_mask is not None else {}
        attn_kwargs = (
            attn_kwargs | {"rotary": kwargs["rotary"]}
            if kwargs.get("rotary") is not None
            else attn_kwargs
        )
        x = self.drop_path1(
            x,
            self.ls1(
                jax.vmap(self.postnorm)(
                    self.attn(
                        jax.vmap(self.prenorm)(x),
                        inference=inference,
                        key=key_attn,
                        **attn_kwargs,
                    )
                )
            ),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.ls2(
                self.mlp(
                    jax.vmap(self.norm)(x),
                    inference=inference,
                    key=key_mlp,
                    **ffn_kwargs,
                )
            ),
            inference=inference,
            key=key_dr2,
        )
        return x
