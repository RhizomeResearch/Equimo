# ty: ignore[invalid-assignment]
# Adapted from tfc-t0 and modified for JAX/Equinox; see NOTICE.
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from .registry import get_layer, register_layer
from .residual_mlp import ResidualMlp


@register_layer()
class PatchEncoder(eqx.Module):
    projection: ResidualMlp
    type_embeddings: eqx.nn.Embedding
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim,
        patch_size,
        *,
        projection_layer="residualmlp",
        key,
    ):
        key1, key2 = jr.split(key)
        projection_layer = cast(type[ResidualMlp], get_layer(projection_layer))
        self.projection = projection_layer(
            patch_size * 3, embed_dim, embed_dim, key=key1
        )
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
