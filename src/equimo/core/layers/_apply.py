"""Apply vector modules over arrays of any rank."""

import equinox as eqx
import jax


def _apply_module_last_dim(module: eqx.Module, x: jax.Array) -> jax.Array:
    """Apply a vector module independently over every leading dimension."""

    leading_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    output = jax.vmap(module)(flat)
    return output.reshape(*leading_shape, output.shape[-1])
