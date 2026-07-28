"""Field-free forward-pass mixins shared by staged vision models.

These mixins define methods only — no fields and no ``__init__`` — so mixing
them into a concrete ``eqx.Module`` leaves its PyTree structure, and therefore
its saved-checkpoint signature, unchanged.
"""

from typing import TYPE_CHECKING, Any, Optional, Sequence

import jax
import jax.random as jr
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core.intermediates import intermediate_indices


class DenseStageFeatures:
    """Forward pass for dense (channel-first) staged models.

    Requires ``blocks``, ``dropout``, ``norm``, and ``head`` attributes.
    """

    # Type-checking-only declarations: these must not be runtime annotations,
    # or Equinox would collect them as dataclass fields on subclasses.
    if TYPE_CHECKING:
        blocks: Any
        dropout: Any
        norm: Any
        head: Any

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "dim height width"]:
        key_drop, *key_blocks = jr.split(key, len(self.blocks) + 1)

        for blk, key_blk in zip(self.blocks, key_blocks):
            x = blk(x, inference=inference, key=key_blk)
        x = self.dropout(x, inference=inference, key=key_drop)

        return x

    def intermediate_features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        indices: Sequence[int] | None = None,
        n_last_blocks: int | None = None,
        **kwargs,
    ) -> tuple[Float[Array, "dim height width"], ...]:
        """Return selected native stage outputs."""

        wanted = intermediate_indices(
            len(self.blocks),
            indices=indices,
            n_last_blocks=n_last_blocks,
        )
        _, *key_blocks = jr.split(key, len(self.blocks) + 1)
        outputs = []
        for i, (blk, key_blk) in enumerate(zip(self.blocks, key_blocks)):
            x = blk(x, inference=inference, key=key_blk)
            if i in wanted:
                outputs.append(x)
        return tuple(outputs)

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:  # noqa: F821
        x = self.features(x, inference=inference, key=key)
        x = self.norm(x.mean((1, 2)))
        x = self.head(x)

        return x


class TokenStemFeatures:
    """Forward pass for token models with a patch stem and dropout.

    Requires ``patch_embed``, ``pos_drop``, ``blocks``, ``norm``, and
    ``head`` attributes.
    """

    if TYPE_CHECKING:
        patch_embed: Any
        pos_drop: Any
        blocks: Any
        norm: Any
        head: Any

    def features(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        key_pd, *keys = jr.split(key, 1 + len(self.blocks))

        x = self.patch_embed(x)
        x = self.pos_drop(x, inference=inference, key=key_pd)
        for i, blk in enumerate(self.blocks):
            x = blk(x, inference=inference, key=keys[i])

        return x

    def intermediate_features(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        indices: Sequence[int] | None = None,
        n_last_blocks: int | None = None,
    ) -> tuple[Float[Array, "..."], ...]:
        """Return selected native stage outputs."""

        wanted = intermediate_indices(
            len(self.blocks),
            indices=indices,
            n_last_blocks=n_last_blocks,
        )
        key_pd, *keys = jr.split(key, 1 + len(self.blocks))
        x = self.patch_embed(x)
        x = self.pos_drop(x, inference=inference, key=key_pd)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, inference=inference, key=keys[i])
            if i in wanted:
                outputs.append(x)
        return tuple(outputs)

    def __call__(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        """Features, token-mean pooling, and classification head."""

        x = self.features(x, inference=inference, key=key)
        x = jax.vmap(self.norm)(x)
        x = reduce(x, "s d -> d", "mean")
        x = self.head(x)

        return x
