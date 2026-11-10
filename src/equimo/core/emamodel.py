__all__ = ["update_ema_model"]

from typing import Any

import equinox as eqx
import jax

from .parameters import trainable_mask


@jax.jit
def _average_params(ema_params: Any, current_params: Any, decay: float) -> Any:
    return jax.tree_util.tree_map(
        lambda ema, new, trainable: (
            decay * ema + (1.0 - decay) * new if trainable else new
        ),
        ema_params,
        current_params,
        trainable_mask(ema_params),
    )


def update_ema_model(
    ema_model: eqx.Module, current_params: Any, decay: float = 0.9999
) -> eqx.Module:
    """Updates EMA model with new parameters using exponential moving average.

    Only trainable leaves (see :func:`equimo.core.parameters.trainable_mask`)
    are averaged. Other arrays, such as integer indices, boolean masks, and
    buffers, take their current values.

    Args:
        ema_model: The EMA model to update
        current_params: Array leaves of the training model, as returned by
            ``eqx.filter(model, eqx.is_array)``
        decay: EMA decay rate

    Returns:
        Updated EMA model
    """
    ema_params = eqx.filter(ema_model, eqx.is_array)
    return eqx.combine(_average_params(ema_params, current_params, decay), ema_model)
