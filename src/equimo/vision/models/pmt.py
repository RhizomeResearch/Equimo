# ty: ignore[invalid-assignment]
# ty: ignore[invalid-return-type]
# ty: ignore[unresolved-attribute]
"""Plain mask transformer with a deterministic frozen vision encoder."""

from __future__ import annotations

import equinox as eqx
import jax

from equimo.registry import register_model
from equimo.vision._encoder_identity import encoder_array_digest
from equimo.vision.models.pmd import (
    _MISSING,
    PMTConfig,
    PMTFeatures,
    PMTMaskState,
    PMTOutput,
    PlainMaskDecoder,
)
from equimo.vision.models.vit import VisionTransformer


@register_model("pmt", modality="vision")
class PMT(eqx.Module):
    """Compose a frozen ViT with a trainable plain mask decoder.

    The default ``PMTConfig.small()`` uses 384-wide Small features, six
    decoder blocks, 100 queries, and lateral GroupNorm with at most 32
    power-of-two groups. GroupNorm is independent of batch size: a direct
    ``model(image, key=key)`` call returns ``PMTOutput`` without model state.
    ``PMTConfig.base()``, ``PMTConfig.large()``, and configuration overrides
    select other source architectures.

    ``norm_layer="batchnorm"`` reproduces the author's lateral architecture.
    Construct that model with ``eqx.nn.make_with_state(PMT)`` and pass its state
    on each call. Training must vmap across real images with
    ``axis_name="pmt_batch"``. Normalization runs after ``encode`` so cached
    frozen features can be reused while the decoder learns.
    """

    backbone: VisionTransformer
    head: PlainMaskDecoder
    backbone_id: str = eqx.field(static=True)
    backbone_digest: str
    input_view: str = eqx.field(static=True)
    class_ontology: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        backbone: VisionTransformer,
        num_classes: int,
        *,
        key: jax.Array,
        config: PMTConfig | None = None,
        backbone_id: str = "unbound",
        input_view: str = "native",
        class_ontology: tuple[str, ...] | None = None,
    ):
        if not isinstance(backbone, VisionTransformer):
            raise TypeError("PMT requires a VisionTransformer backbone.")
        config = PMTConfig.small() if config is None else config
        if num_classes <= 0 or config.dim != backbone.dim:
            raise ValueError(
                "PMT class count and encoder width must agree with config."
            )
        if config.taps[-1] != backbone.num_blocks - 1:
            raise ValueError("PMT final tap must select the last encoder block.")
        if not backbone.dynamic_img_size or backbone.local_pos_embed is None:
            raise ValueError(
                "PMT requires a dynamic ViT with spatial rotary positions."
            )
        if not isinstance(backbone.head, eqx.nn.Identity):
            raise ValueError("PMT requires a feature-only backbone.")
        if any(chunk.downsample is not None for chunk in backbone.blocks):
            raise ValueError("PMT requires a plain transformer backbone.")
        attention = backbone.block_at(0).attn
        if attention.num_heads != config.num_heads:
            raise ValueError("PMT attention heads disagree with the backbone.")
        if not backbone_id or not input_view:
            raise ValueError("PMT backbone_id and input_view must be nonempty.")
        ontology = (
            tuple(str(index) for index in range(num_classes))
            if class_ontology is None
            else tuple(class_ontology)
        )
        if len(ontology) != num_classes or len(set(ontology)) != num_classes:
            raise ValueError("PMT class ontology must have unique ordered class names.")
        self.backbone = backbone
        self.head = PlainMaskDecoder(
            config,
            num_classes=num_classes,
            num_prefix_tokens=backbone.num_prefix_tokens,
            patch_size=backbone.patch_embed.patch_size,
            key=key,
        )
        self.backbone_id = backbone_id
        self.backbone_digest = encoder_array_digest(backbone)
        self.input_view = input_view
        self.class_ontology = ontology

    def is_stateful(self) -> bool:
        """Whether the selected lateral normalization has mutable state."""
        return self.head.is_stateful()

    def encode(self, image: jax.Array) -> PMTFeatures:
        """Return reusable, encoder-normalized features before lateral layers.

        The encoder always runs with stochastic operations disabled. Inputs
        must already use the selected backbone's channel-first preprocessing.
        """
        if image.ndim != 3:
            raise ValueError("PMT expects one (channels, height, width) image.")
        config = self.head.config
        metadata = self.backbone.feature_metadata(
            image,
            endpoint="intermediate_features",
            endpoint_options={"indices": config.taps, "apply_norm": True},
        )
        levels = self.backbone.intermediate_features(
            image, key=None, inference=True, indices=config.taps, apply_norm=True
        )
        _, height, width, rotary = self.backbone.prepare_tokens(
            image, key=None, inference=True
        )
        if metadata["grid_size"] != (height, width):
            raise ValueError("PMT patch-grid metadata disagrees with encoder output.")
        return PMTFeatures(
            levels=tuple(jax.lax.stop_gradient(level) for level in levels),
            rotary=rotary,
            grid_size=metadata["grid_size"],
            patch_size=metadata["patch_size"],
            prefix_tokens=metadata["prefix_tokens"],
            taps=metadata["layer_indices"],
            positional_configuration=metadata["positional_configuration"],
            backbone_id=self.backbone_id,
            backbone_digest=self.backbone_digest,
            input_view=self.input_view,
        )

    def encoder_features(self, image: jax.Array) -> jax.Array:
        """Return the last normalized frozen-encoder tap in prefix/patch order."""
        return self.encode(image).levels[-1]

    def features(
        self,
        image: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> jax.Array:
        """Return frozen encoder tokens; use ``encoder_features`` explicitly."""
        del key, inference
        return self.encoder_features(image)

    def decode(
        self,
        features: PMTFeatures,
        state: eqx.nn.State | None | object = _MISSING,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        mask_state: PMTMaskState | None = None,
    ) -> PMTOutput | tuple[PMTOutput, eqx.nn.State | None]:
        """Decode retained features with the selected normalization state."""
        if (
            features.backbone_id != self.backbone_id
            or features.backbone_digest != self.backbone_digest
            or features.input_view != self.input_view
            or features.positional_configuration
            != self.backbone._feature_position_configuration()
        ):
            raise ValueError("PMT features disagree with encoder or input view.")
        return self.head(
            features, state, inference=inference, key=key, mask_state=mask_state
        )

    def __call__(
        self,
        image: jax.Array,
        state: eqx.nn.State | None | object = _MISSING,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        mask_state: PMTMaskState | None = None,
    ) -> PMTOutput | tuple[PMTOutput, eqx.nn.State | None]:
        """Return mask/class predictions; state is optional with GroupNorm."""
        return self.decode(
            self.encode(image),
            state,
            inference=inference,
            key=key,
            mask_state=mask_state,
        )


def mask_free_pmt(model: PMT, mask_state: PMTMaskState) -> PMT:
    """Disable intermediate mask predictions after all masks anneal away."""
    if mask_state.probabilities.shape != (model.head.config.num_blocks,):
        raise ValueError("PMT mask-state length must equal decoder depth.")
    if not bool(jax.numpy.all(mask_state.probabilities == 0)):
        raise ValueError("Mask-free PMT requires terminal mask probabilities.")
    return eqx.tree_at(lambda candidate: candidate.head.masked_attention, model, False)


__all__ = ["PMT", "mask_free_pmt"]
