"""One synthetic EoMT update with native assignment and mask losses.

Run from the repository root with ``uv run python examples/query_segmentation_training.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from equimo.finetune.vision import eomt_full_finetune
from equimo.vision.models import EoMT, VisionTransformer
from equimo.vision.query_training import (
    QueryLossConfig,
    QueryTargets,
    prepare_query_loss,
    query_segmentation_loss,
)


def main():
    encoder = VisionTransformer(
        img_size=16,
        in_channels=3,
        dim=8,
        patch_size=8,
        num_heads=2,
        depths=[2],
        num_classes=0,
        dynamic_img_size=True,
        key=jr.key(0),
    )
    model = EoMT(encoder, num_classes=2, num_queries=3, num_blocks=1, key=jr.key(1))
    plan = eomt_full_finetune(model)
    image = jr.normal(jr.key(2), (3, 16, 16))
    first = jnp.indices((16, 16))[1] < 8
    targets = QueryTargets(
        jnp.array([0, 1]),
        jnp.stack((first, ~first)),
        jnp.ones(2, bool),
        jnp.ones((16, 16), bool),
    )
    config = QueryLossConfig(matching_num_points=32, loss_num_points=32)

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss(trainable, forward_key, sampling_key):
        output = plan.combine(trainable)(image, key=forward_key, inference=False)
        context = prepare_query_loss(output, targets, key=sampling_key, config=config)
        return query_segmentation_loss(output, targets, context).total

    value, gradients = loss(plan.trainable, jr.key(3), jr.key(4))
    if not bool(jnp.isfinite(value)):
        raise RuntimeError("Invalid loss; update rejected.")
    # This example owns its simple SGD update; the library owns no optimizer.
    updates = jax.tree.map(lambda gradient: -1e-3 * gradient, gradients)
    updated = plan.combine(eqx.apply_updates(plan.trainable, updates))
    assert not bool(jnp.array_equal(updated.queries, model.queries))
    print(
        f"Finite query loss {float(value):.6f}; encoder and query parameters updated."
    )


if __name__ == "__main__":
    main()
