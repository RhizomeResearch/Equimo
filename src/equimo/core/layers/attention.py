import dataclasses
from typing import Callable, List, Literal, Optional, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core._prng import split_for_mode
from equimo.core.layers._apply import _apply_module_last_dim
from equimo.core.layers.activation import get_act
from equimo.core.layers.dropout import DropPathAdd, split_drop_path
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.norm import LayerScale, get_norm, maybe_layer_scale
from equimo.core.layers.rotary import RotaryFactors, apply_rotary, apply_rotary_qk
from equimo.core.layers._registry import make_get, make_register

_ATTN_REGISTRY: dict[str, type[eqx.Module]] = {}
_ATTN_BLOCK_REGISTRY: dict[str, type[eqx.Module]] = {}


register_attn = make_register(_ATTN_REGISTRY)


get_attn = make_get(_ATTN_REGISTRY)


register_attn_block = make_register(_ATTN_BLOCK_REGISTRY)


get_attn_block = make_get(_ATTN_BLOCK_REGISTRY)


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


AttentionImplementation = Literal["xla", "cudnn"]
_IMPLEMENTATIONS = ("xla", "cudnn")
# cuDNN's fused attention backward requires sequence lengths divisible by 64.
_CUDNN_SEQUENCE_MULTIPLE = 64

_Model = TypeVar("_Model")


def _check_implementation(implementation: str) -> None:
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError(
            f"implementation must be one of {_IMPLEMENTATIONS}, got {implementation!r}."
        )


def _check_cudnn_dropout(
    dropout: eqx.nn.Dropout | None, inference: Optional[bool]
) -> None:
    """Reject attention-probability dropout, which the fused kernel cannot apply."""
    if dropout is None:
        return
    if inference is None:
        inference = dropout.inference
    if not inference and dropout.p != 0:
        raise ValueError(
            "implementation='cudnn' does not support attention dropout. Set the "
            "attention dropout rate to 0, call with inference=True, or use "
            "implementation='xla'."
        )


def _fused_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    allowed: jax.Array | None = None,
    *,
    bias: jax.Array | None = None,
    key_value_seq_lengths: jax.Array | None = None,
    implementation: AttentionImplementation = "cudnn",
) -> jax.Array:
    """Fused dot-product attention over ``(..., heads, seq, head_dim)`` inputs.

    ``allowed`` broadcasts to ``(..., heads, query, key)``. Fully masked query rows
    return zeros, as in the unfused path. Keys and values may have fewer heads
    than queries if that number divides the query heads. ``implementation`` is
    passed to :func:`jax.nn.dot_product_attention`; ``"xla"`` lets CPU tests
    check the padding and masking.

    ``key_value_seq_lengths`` replaces ``allowed`` when the valid keys are packed
    at the front: it broadcasts to the leading shape ``q.shape[:-3]`` (a scalar
    for unbatched inputs) and gives the number of valid keys of each entry, so a
    variable key count needs no dense mask. Entries whose length is 0 return
    zeros.

    ``bias`` is added to the scaled logits and broadcasts to
    ``(..., heads, query, key)``. The kernels add it in the dtype of ``q``, so
    under cuDNN it is rounded to bfloat16 or float16 first.
    """
    if allowed is not None and key_value_seq_lengths is not None:
        raise ValueError("Pass either allowed or key_value_seq_lengths, not both.")
    if implementation == "cudnn" and q.dtype not in (jnp.bfloat16, jnp.float16):
        raise ValueError(
            "implementation='cudnn' requires bfloat16 or float16 attention inputs, "
            f"got {q.dtype}. Cast the model or use implementation='xla'."
        )
    *leading, heads, query_len, head_dim = q.shape
    key_len = k.shape[-2]
    padded_query = query_len + -query_len % _CUDNN_SEQUENCE_MULTIPLE
    padded_key = key_len + -key_len % _CUDNN_SEQUENCE_MULTIPLE

    def batch_first(x: jax.Array, length: int) -> jax.Array:
        x = jnp.swapaxes(x, -3, -2)
        x = x.reshape(-1, *x.shape[-3:])
        return jnp.pad(x, ((0, 0), (0, length - x.shape[1]), (0, 0), (0, 0)))

    def batch_scores(x: jax.Array) -> jax.Array:
        # Materialize every head: a broadcast head axis fails under vmap
        # (jax-ml/jax#34920). Padded queries and keys get zeros (False).
        x = jnp.broadcast_to(x, (*leading, heads, query_len, key_len))
        return jnp.pad(
            x.reshape(-1, heads, query_len, key_len),
            ((0, 0), (0, 0), (0, padded_query - query_len), (0, padded_key - key_len)),
        )

    q = batch_first(q, padded_query)
    k, v = batch_first(k, padded_key), batch_first(v, padded_key)
    batch = q.shape[0]
    if bias is not None:
        bias = batch_scores(jnp.asarray(bias, q.dtype))
    if allowed is None:
        if key_value_seq_lengths is None:
            lengths = jnp.full((batch,), key_len, jnp.int32)
        else:
            lengths = jnp.asarray(key_value_seq_lengths, jnp.int32)
            lengths = jnp.broadcast_to(lengths, tuple(leading)).reshape(batch)
        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            bias=bias,
            query_seq_lengths=jnp.full((batch,), query_len, jnp.int32),
            key_value_seq_lengths=lengths,
            implementation=implementation,
        )
        if key_value_seq_lengths is not None:
            # Without a valid key the kernels return an average of the values.
            out = jnp.where((lengths > 0)[:, None, None, None], out, 0)
    else:
        mask = batch_scores(allowed)
        out = jax.nn.dot_product_attention(
            q, k, v, bias=bias, mask=mask, implementation=implementation
        )
        # cuDNN returns the mean value for fully masked rows instead of zeros.
        has_key = jnp.swapaxes(jnp.any(mask, axis=-1), 1, 2)
        out = jnp.where(has_key[..., None], out, 0)
    out = out[:, :query_len].reshape(*leading, query_len, heads, head_dim)
    return jnp.swapaxes(out, -3, -2)


def with_attention_implementation(
    model: _Model, implementation: AttentionImplementation
) -> _Model:
    """Return ``model`` with every attention module using ``implementation``.

    ``"cudnn"`` runs core and tabular attention through NVIDIA cuDNN's fused
    kernel (:func:`jax.nn.dot_product_attention`), which computes logits in
    float32 without materializing them. It requires a CUDA GPU, bfloat16 or
    float16 inputs, and inactive attention dropout, and it cannot be exported
    with jax2onnx; switch back to ``"xla"`` before exporting. Parameters and
    checkpoint layouts are unchanged. The setting is static, so compiled
    functions retrace after switching.
    """
    _check_implementation(implementation)

    def is_attention(node) -> bool:
        return isinstance(node, eqx.Module) and any(
            field.name == "implementation" for field in dataclasses.fields(node)
        )

    def switch(node):
        if not is_attention(node):
            return node
        # tree_at cannot replace static fields, and copy.copy returns the same
        # module, so rebuild the instance as Equinox's unflattening does.
        switched = object.__new__(type(node))
        switched.__dict__.update(node.__dict__)
        object.__setattr__(switched, "implementation", implementation)
        return switched

    return jax.tree.map(switch, model, is_leaf=is_attention)


@register_attn()
class Attention(eqx.Module):
    """Multi-head self attention over tensors shaped ``(..., seq, dim)``.

    Masks use nonzero/``True`` entries for allowed query-key pairs. They may be
    batch-aligned with shape ``(..., query, key)`` or directly broadcastable to
    ``(..., heads, query, key)``. Fully masked query rows receive zero attention
    probability mass.

    ``implementation="cudnn"`` selects the fused kernel described in
    :func:`with_attention_implementation`.
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)

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
        implementation: AttentionImplementation = "xla",
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        _check_implementation(implementation)
        self.implementation = implementation

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

        qkv = self._project_qkv(x).reshape(
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

        allowed = None
        if mask is not None:
            allowed = _broadcast_attention_mask(
                mask,
                x_ndim=x.ndim,
                shape=(*q.shape[:-1], k.shape[-2]),
            )
        if self.implementation == "cudnn":
            _check_cudnn_dropout(self.attn_drop, inference)
            x = _fused_attention(q, k, v, allowed)
        else:
            attn = jnp.einsum("...hqd,...hkd->...hqk", q, k).astype(
                jnp.float32
            ) / jnp.sqrt(self.head_dim)
            attn = jax.nn.softmax(attn, axis=-1, where=allowed).astype(x.dtype)
            attn = self.attn_drop(attn, inference=inference, key=key1)
            x = jnp.einsum("...hqk,...hkd->...hqd", attn, v)

        x = jnp.swapaxes(x, -3, -2).reshape(*x.shape[:-3], x.shape[-2], self.dim)
        x = _apply_module_last_dim(self.proj, x)
        return self.proj_drop(x, inference=inference, key=key2)

    def _project_qkv(self, x: jax.Array) -> jax.Array:
        """Project tokens to queries, keys, and values."""
        return _apply_module_last_dim(self.qkv, x)


class CrossAttention(eqx.Module):
    """Multi-head attention from query tokens to separate context tokens.

    Queries come from ``x`` with shape ``(query, dim)``, keys from ``context``
    with shape ``(key, kv_dim)``, and values from ``values``, which defaults to
    ``context``. DETR-style attention needs them apart, because it adds position
    encodings to queries and keys only. All three are projected to
    ``internal_dim`` (``dim`` by default; SAM's two-way transformer halves it),
    and the attended values are projected back to ``dim``.

    ``rotary`` rotates the projected queries and ``key_rotary`` the projected
    keys, independently; values are never rotated. Unlike
    :class:`Attention`, each takes its own factors, since queries and keys can
    have different lengths.

    ``key_mask`` uses ``True`` for keys to attend to and broadcasts to
    ``(heads, query, key)``, so a ``(key,)`` mask applies to every query. Fully
    masked query rows attend to nothing, so their output is ``proj``'s bias.
    ``bias`` is added to the scaled logits and broadcasts to the same shape;
    the XLA path adds it in float32, the cuDNN kernel in the input dtype. The layer has no dropout, so
    ``key`` and ``inference`` are accepted only for the endpoint convention.

    ``implementation="cudnn"`` selects the fused kernel described in
    :func:`with_attention_implementation`.
    """

    dim: int = eqx.field(static=True)
    kv_dim: int = eqx.field(static=True)
    internal_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)

    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        kv_dim: int | None = None,
        internal_dim: int | None = None,
        implementation: AttentionImplementation = "xla",
    ):
        kv_dim = dim if kv_dim is None else kv_dim
        internal_dim = dim if internal_dim is None else internal_dim
        if internal_dim % num_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )
        _check_implementation(implementation)
        self.dim = dim
        self.kv_dim = kv_dim
        self.internal_dim = internal_dim
        self.num_heads = num_heads
        self.implementation = implementation

        key_q, key_k, key_v, key_proj = jr.split(key, 4)
        self.q = eqx.nn.Linear(dim, internal_dim, key=key_q)
        self.k = eqx.nn.Linear(kv_dim, internal_dim, key=key_k)
        self.v = eqx.nn.Linear(kv_dim, internal_dim, key=key_v)
        self.proj = eqx.nn.Linear(internal_dim, dim, key=key_proj)

    @property
    def head_dim(self) -> int:
        return self.internal_dim // self.num_heads

    def __call__(
        self,
        x: Float[Array, "query dim"],
        context: Float[Array, "key kv_dim"],
        *,
        values: Float[Array, "key kv_dim"] | None = None,
        rotary: RotaryFactors | None = None,
        key_rotary: RotaryFactors | None = None,
        key_mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Float[Array, "query dim"]:
        q = self._heads(self.q, x)
        k = self._heads(self.k, context)
        v = self._heads(self.v, context if values is None else values)
        if rotary is not None:
            q = apply_rotary(q, rotary)
        if key_rotary is not None:
            k = apply_rotary(k, key_rotary)

        allowed = None
        if key_mask is not None:
            allowed = jnp.broadcast_to(
                jnp.asarray(key_mask, dtype=bool),
                (self.num_heads, x.shape[0], context.shape[0]),
            )
        if self.implementation == "cudnn":
            out = _fused_attention(q, k, v, allowed, bias=bias)
        else:
            logits = jnp.einsum("hqd,hkd->hqk", q, k).astype(jnp.float32) / jnp.sqrt(
                self.head_dim
            )
            if bias is not None:
                logits = logits + jnp.asarray(bias, jnp.float32)
            weights = jax.nn.softmax(logits, axis=-1, where=allowed).astype(x.dtype)
            out = jnp.einsum("hqk,hkd->hqd", weights, v)

        out = jnp.swapaxes(out, 0, 1).reshape(x.shape[0], self.internal_dim)
        return jax.vmap(self.proj)(out)

    def _heads(self, linear: eqx.nn.Linear, tokens: jax.Array) -> jax.Array:
        """Project ``(seq, features)`` tokens to ``(heads, seq, head_dim)``."""
        projected = jax.vmap(linear)(tokens)
        projected = projected.reshape(tokens.shape[0], self.num_heads, self.head_dim)
        return jnp.swapaxes(projected, 0, 1)


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
