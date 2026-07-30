# ty: ignore[invalid-assignment]
# Adapted from tfc-t0 and modified for JAX/Equinox; see NOTICE.
import equinox as eqx
import jax
import jax.random as jr

from equimo.core.layers import Mlp

from .registry import register_layer


@register_layer()
class ResidualMlp(eqx.Module):
    """Residual projection backed by Equimo's shared MLP."""

    mlp: Mlp
    residual_layer: eqx.nn.Linear

    def __init__(self, in_features, hidden_features, out_features, *, key):
        key1, key2 = jr.split(key)
        self.mlp = Mlp(
            in_features,
            hidden_dim=hidden_features,
            out_dim=out_features,
            act_layer="relu",
            dropout_rate=0.0,
            key=key1,
        )
        self.residual_layer = eqx.nn.Linear(in_features, out_features, key=key2)

    def __call__(self, x, *, key, inference=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = self.mlp(flat, key=key, inference=inference)
        out += jax.vmap(self.residual_layer)(flat)
        return out.reshape(*shape[:-1], -1)
