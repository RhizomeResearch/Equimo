"""Two logical CPU devices: synchronized PMT BatchNorm against NumPy references."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from equimo.vision.models.pmd import ReferenceBatchNorm


def main():
    assert len(jax.devices()) == 2
    norm, initial = eqx.nn.make_with_state(ReferenceBatchNorm)(
        2, axis_name=("pmt_batch", "devices")
    )
    samples = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) / 3
    weights = np.random.default_rng(29).normal(size=samples.shape).astype(np.float32)

    def local(x, valid, state):
        return jax.vmap(
            lambda value, keep: norm(value, state, inference=False, example_valid=keep),
            axis_name="pmt_batch",
            out_axes=(0, None),
        )(x, valid)

    mapped = jax.pmap(local, axis_name="devices", in_axes=(0, 0, None))
    for valid in (
        np.array([[True, False], [True, True]]),
        np.array([[True, True], [False, False]]),
        np.zeros((2, 2), bool),
    ):
        output, state = mapped(jnp.array(samples), jnp.array(valid), initial)
        expected = np.zeros_like(samples)
        expected_gradient = np.zeros_like(samples)
        if valid.any():
            selected = samples[valid].transpose(1, 0, 2).reshape(2, -1)
            mean = selected.mean(axis=1)
            variance = selected.var(axis=1)
            expected[valid] = (samples[valid] - mean[None, :, None]) / np.sqrt(
                variance[None, :, None] + 1e-5
            )
            g = weights[valid].transpose(1, 0, 2).reshape(2, -1)
            centered = selected - mean[:, None]
            dx = (
                g
                - g.mean(axis=1)[:, None]
                - centered
                * (g * centered).mean(axis=1)[:, None]
                / (variance[:, None] + 1e-5)
            ) / np.sqrt(variance[:, None] + 1e-5)
            expected_gradient[valid] = dx.reshape(2, -1, 3).transpose(1, 0, 2)
            expected_mean = 0.1 * mean
            expected_var = 0.9 + 0.1 * selected.var(axis=1, ddof=1)
        else:
            expected_mean, expected_var = np.zeros(2), np.ones(2)
        np.testing.assert_allclose(output, expected, atol=2e-6)
        np.testing.assert_allclose(
            state.get(norm.mean_index),
            np.broadcast_to(expected_mean, (2, 2)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            state.get(norm.variance_index),
            np.broadcast_to(expected_var, (2, 2)),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            state.get(norm.count_index), np.full(2, int(valid.any()))
        )
        gradient = jax.jit(
            jax.grad(
                lambda x: jnp.sum(
                    mapped(x, jnp.array(valid), initial)[0] * jnp.array(weights)
                )
            )
        )(jnp.array(samples))
        np.testing.assert_allclose(gradient, expected_gradient, atol=2e-6)
    print("Distributed moments, gradients, and state counters match NumPy references.")


if __name__ == "__main__":
    main()
