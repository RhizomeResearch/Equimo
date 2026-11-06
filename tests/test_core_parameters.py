import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.core.layers.norm import LayerScale
from equimo.core.parameters import buffer, tagged, trainable_mask, weight_decay_mask


class _Block(eqx.Module):
    linear: eqx.nn.Linear
    post: eqx.nn.Sequential
    ls: LayerScale
    query_bias: jax.Array
    gain: jax.Array
    temperature: jax.Array
    A_log: jax.Array = tagged("ssm")
    basis: jax.Array = buffer()
    gate: jax.Array = buffer(when=lambda block: not block.gate_trainable)
    index: jax.Array
    gate_trainable: bool = eqx.field(static=True)


def _block(*, gate_trainable: bool) -> _Block:
    return _Block(
        linear=eqx.nn.Linear(2, 2, key=jr.PRNGKey(0)),
        post=eqx.nn.Sequential([eqx.nn.LayerNorm(2)]),
        ls=LayerScale(2),
        query_bias=jnp.zeros(2),
        gain=jnp.ones(2),
        temperature=jnp.ones(()),
        A_log=jnp.zeros(2),
        basis=jnp.ones((2, 2)),
        gate=jnp.ones((2, 2)),
        index=jnp.arange(2),
        gate_trainable=gate_trainable,
    )


@pytest.mark.parametrize("gate_trainable", [False, True])
def test_masks_follow_module_declarations(gate_trainable):
    block = _block(gate_trainable=gate_trainable)
    trainable = trainable_mask(block)
    decay = weight_decay_mask(block)

    assert trainable.basis is False
    assert trainable.index is False
    assert trainable.gate is gate_trainable
    assert trainable.A_log is True
    assert trainable.post.layers[0].weight is True

    assert decay.linear.weight is True
    assert decay.gate is gate_trainable
    assert not any(
        (
            decay.linear.bias,
            decay.post.layers[0].weight,
            decay.post.layers[0].bias,
            decay.ls.gamma,
            decay.query_bias,
            # Arrays with at most one axis, whatever their names.
            decay.gain,
            decay.temperature,
            decay.A_log,
            decay.basis,
            decay.index,
        )
    )


def test_weight_decay_mask_never_decays_buffers():
    decay = weight_decay_mask(_block(gate_trainable=False), no_decay_tags=())

    assert decay.A_log is True
    assert decay.ls.gamma is True
    assert decay.basis is False
    assert decay.gate is False
