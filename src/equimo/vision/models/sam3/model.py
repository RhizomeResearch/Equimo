"""SAM 3.1 model and factory.

The converted weights are distributed under the SAM License; its
acknowledgement and trade-control terms apply to users of those weights. The
code is under Equimo's license.
"""

from __future__ import annotations

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.factory import build_model_variant
from equimo.registry import register_model
from equimo.vision.models.sam3.detector import (
    ConceptDetector,
    ConceptPrompt,
    DetectorOutput,
)
from equimo.vision.models.sam3.encoder import ImageFeatures, TriNeck, ViTDetTrunk
from equimo.vision.models.sam3.interactive import (
    InteractiveHead,
    InteractiveOutput,
    InteractivePrompt,
)
from equimo.vision.models.sam3.tracker import (
    Association,
    CommitPlan,
    Sam3Tracker,
    TrackerState,
    TrackOutput,
    associate,
)


@register_model("sam3", modality="vision")
class Sam3(eqx.Module):
    """SAM 3.1 image encoder, concept detector, interactive head and tracker.

    Every endpoint runs on one unbatched sample; batch with ``jax.vmap``. The
    port has no dropout, so ``key`` and ``inference`` only follow the endpoint
    convention. Text comes in as features from the separate text model.
    """

    trunk: ViTDetTrunk
    neck: TriNeck
    detector: ConceptDetector
    interactive: InteractiveHead
    tracker: Sam3Tracker

    def __init__(
        self,
        *,
        key: PRNGKeyArray,
        in_channels: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_features: int,
        window_size: int,
        global_blocks: tuple[int, ...],
        dim: int,
        head_num_heads: int,
        ffn_features: int,
        geometry_depth: int,
        fusion_depth: int,
        decoder_depth: int,
        num_queries: int,
        two_way_depth: int,
        memory_depth: int,
        memory_heads: int,
        num_slots: int,
        memory_start_channels: int,
        memory_fuser_depth: int,
    ):
        key_trunk, key_neck, key_detector, key_interactive, key_tracker = jr.split(
            key, 5
        )
        grid = img_size // patch_size
        self.trunk = ViTDetTrunk(
            key=key_trunk,
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_features=mlp_features,
            window_size=window_size,
            global_blocks=global_blocks,
        )
        self.neck = TriNeck(embed_dim, dim, key=key_neck)
        self.detector = ConceptDetector(
            dim,
            head_num_heads,
            ffn_features,
            key=key_detector,
            geometry_depth=geometry_depth,
            fusion_depth=fusion_depth,
            decoder_depth=decoder_depth,
            num_queries=num_queries,
        )
        self.interactive = InteractiveHead(
            dim,
            head_num_heads,
            ffn_features,
            two_way_depth,
            grid,
            img_size,
            key=key_interactive,
        )
        self.tracker = Sam3Tracker(
            dim,
            head_num_heads,
            ffn_features,
            key=key_tracker,
            memory_depth=memory_depth,
            memory_heads=memory_heads,
            decoder_depth=two_way_depth,
            num_slots=num_slots,
            grid=grid,
            image_size=img_size,
            memory_start_channels=memory_start_channels,
            memory_fuser_depth=memory_fuser_depth,
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Float[Array, "dim grid grid"]:
        """Return the detector-level feature map at the trunk's resolution."""
        return self.neck.detector.levels[2](self.trunk(x))

    def encode_image(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray | None = None,
        inference: bool = True,
        *,
        checkpoint_trunk: bool = False,
    ) -> ImageFeatures:
        """Return the three feature pyramids of one image.

        ``checkpoint_trunk`` rematerializes the trunk blocks under autodiff,
        trading compute for the memory fine-tuning through the trunk needs.
        """
        return self.neck(self.trunk(x, checkpoint=checkpoint_trunk))

    def detect(
        self,
        features: ImageFeatures,
        prompt: ConceptPrompt,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> DetectorOutput:
        """Find every instance of the prompted concept."""
        return self.detector(features, prompt)

    def predict_interactive(
        self,
        features: ImageFeatures,
        prompt: InteractivePrompt,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> InteractiveOutput:
        """Segment one object from points, a box or a mask."""
        return self.interactive(features, prompt)

    def propagate(
        self,
        state: TrackerState,
        features: ImageFeatures,
        frame_index: Int[Array, ""],
        key: PRNGKeyArray | None = None,
        inference: bool = True,
        *,
        num_frames: int | None = None,
    ) -> TrackOutput:
        """Track every bucket into a frame (experimental).

        ``num_frames`` is the clip length, ``None`` when it is unknown. It only
        scales the temporal encoding of object pointers.
        """
        return self.tracker.propagate(
            state, features, frame_index, num_frames=num_frames
        )

    def associate(
        self,
        detections: DetectorOutput,
        tracks: TrackOutput,
        state: TrackerState,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Association:
        """Compute the flags the host policy decides from (experimental)."""
        return associate(detections, tracks, state)

    def commit(
        self,
        state: TrackerState,
        features: ImageFeatures,
        tracks: TrackOutput,
        plan: CommitPlan,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> TrackerState:
        """Apply the host's plan and store the frame's memory (experimental)."""
        return self.tracker.commit(self.interactive, state, features, tracks, plan)


_SAM31_BASE_CFG: dict = {
    "in_channels": 3,
    "img_size": 1008,
    "patch_size": 14,
    "embed_dim": 1024,
    "depth": 32,
    "num_heads": 16,
    "mlp_features": 4736,
    "window_size": 24,
    "global_blocks": (7, 15, 23, 31),
    "dim": 256,
    "head_num_heads": 8,
    "ffn_features": 2048,
    "geometry_depth": 3,
    "fusion_depth": 6,
    "decoder_depth": 6,
    "num_queries": 200,
    "two_way_depth": 2,
    "memory_depth": 4,
    "memory_heads": 8,
    "num_slots": 16,
    "memory_start_channels": 4,
    "memory_fuser_depth": 2,
}

_SAM3_REGISTRY: dict[str, tuple[dict, dict]] = {
    "sam31_multiplex": (_SAM31_BASE_CFG, {}),
}


def sam31_multiplex(
    pretrained: bool = False,
    inference_mode: bool = True,
    **kwargs,
) -> Sam3:
    """Build SAM 3.1 at its published configuration.

    Text prompts need the separate ``sam31_text`` model. The weights are under
    the SAM License.
    """
    return build_model_variant(
        Sam3,
        _SAM3_REGISTRY,
        "sam31_multiplex",
        pretrained=pretrained,
        inference_mode=inference_mode,
        pretrained_label="SAM 3.1",
        allow_pretrained_overrides=False,
        **kwargs,
    )


__all__ = ["Sam3", "sam31_multiplex"]
