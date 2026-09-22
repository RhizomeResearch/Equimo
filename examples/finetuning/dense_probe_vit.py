"""Pointwise spatial probe over a local patch-feature backbone."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


class PatchBackbone(eqx.Module):
    dim: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    head: eqx.Module

    def __init__(self):
        self.dim = 4
        self.num_prefix_tokens = 1
        self.head = eqx.nn.Identity()

    def features(self, image):
        height, width = image.shape[-2:]
        patches = jnp.moveaxis(image, 0, -1).reshape(height * width, 3)
        fourth_channel = jnp.mean(patches, axis=-1, keepdims=True)
        prefix = jnp.zeros((1, self.dim), dtype=image.dtype)
        return jnp.concatenate((prefix, jnp.concatenate((patches, fourth_channel), -1)))

    def feature_metadata(self, image, *, endpoint, endpoint_options):
        del endpoint, endpoint_options
        height, width = image.shape[-2:]
        return {
            "input_size": (height, width),
            "patch_size": (1, 1),
            "patch_padding": (0, 0),
            "grid_size": (height, width),
            "prefix_tokens": ("cls",),
            "tokens_include_prefix": True,
            "endpoint_normalization": "none",
        }


spec = eqft.FeatureSpec(
    endpoint="features",
    output_layout="BNC",
    token_selection="patches",
    pooling=None,
    return_metadata=True,
)
probe = eqft.vision.make_dense_probe(
    PatchBackbone(),
    in_features=4,
    out_features=3,
    key=jr.PRNGKey(0),
    feature_spec=spec,
)

image = jnp.ones((3, 2, 3), dtype=jnp.float32)
batch = jnp.stack((image, image * 2))
print(probe(image).shape)  # (3, 2, 3)
print(jax.vmap(probe)(batch).shape)  # (2, 3, 2, 3)
