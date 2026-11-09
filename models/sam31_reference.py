"""Generate SAM 3.1 reference fixtures from the pinned upstream implementation.

This maintainer tool runs on a CUDA GPU in the environment described by
``models/sam31_reference_requirements.txt``, because the reference hard-codes
CUDA in several constructors and forward passes. Native Equimo code and the
offline tests only read its outputs. ``TINY_CONFIG`` is importable without torch
so that the tests build the matching Equimo model.

Every fixture holds ``state.<torch key>`` with the reference's parameters,
``inputs.<component>.<argument>`` and ``outputs.<component>.<part>`` with what
each module received and returned, named after the Equimo modules. The float32
reference has autocast and TF32 off, FlashAttention off, the math attention
kernel in memory attention, the ViT MLP as plain fc1, exact GELU and fc2, and
memory stored in float32. The bfloat16 reference is the shipped pipeline with
FlashAttention off. ``torch.compile`` is off in every run.

Random weights give the tiny detector no detection above any threshold and the
tiny tracker no usable mask, so the tiny clip replaces the policy's inputs with
a scripted scenario of disjoint per-object masks. The real detector and tracker
still run and are recorded. Masks either coincide or are disjoint, which keeps
every compared overlap away from its threshold.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import numpy as np


REFERENCE_REPOSITORY = "https://github.com/facebookresearch/sam3"
REFERENCE_REVISION = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
CHECKPOINT_NAME = "sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"

# Reference builder values for the full model (sam3/model_builder.py).
FULL_MULTIPLEX_COUNT = 16
FULL_MAX_OBJECTS = 16

# The tests build the matching Equimo model from this table.
TINY_CONFIG: dict[str, Any] = {
    "img_size": 56,
    "pretrain_img_size": 28,
    "patch_size": 14,
    "embed_dim": 16,
    "depth": 2,
    "num_heads": 2,
    "mlp_features": 32,
    "window_size": 2,
    "global_blocks": (1,),
    "dim": 16,
    "head_num_heads": 2,
    "ffn_features": 32,
    "geometry_depth": 1,
    "fusion_depth": 1,
    "decoder_depth": 2,
    "num_queries": 4,
    "two_way_depth": 1,
    "memory_depth": 1,
    "memory_heads": 2,
    "num_slots": 2,
    "text_width": 16,
    "text_depth": 1,
    "text_heads": 2,
    "context_length": 8,
    "num_merges": 512,
    "memory_start_channels": 1,
    "max_objects": 4,
    "num_frames": 20,
    "source_size": (40, 60),
    "seed": 31,
}

CONCEPT = "yellow school bus"
TOKENIZER_CORPUS = (
    "a photo of a cat",
    "YELLOW School BUS",
    "Crème brûlée à la carte",
    "two 🐈 and one 🐕",
    "fish &amp; chips",
    "it's the dog's toy, isn't it?",
    "  many   spaces\tand\nnewlines  ",
    "a very long prompt " * 8,
    "",
)

# Component outputs the image fixture keeps; interactive prompts are stored as
# ``outputs.interactive.point`` and ``outputs.interactive.box``.
IMAGE_RECORDED_OUTPUTS = (
    "outputs.trunk",
    "outputs.neck",
    "outputs.text",
    "outputs.detector.geometry",
    "outputs.detector.head",
    "outputs.detector.joint",
    "outputs.interactive.point",
    "outputs.interactive.box",
)
IMAGE_MEAN_ABSOLUTE_ERROR = 5e-4
# Several boxes of the fixture's image land a query within the threshold margin.
# This one clears every margin and keeps several detections.
EXEMPLAR_OBJECT = 19

# Fixed by the design; the 1e-3 margin keeps numerically legitimate differences
# from flipping a comparison.
THRESHOLD_MARGIN = 1e-3
DETECTION_THRESHOLDS = (0.4, 0.5, 0.65, 0.8)
TRACK_PROBABILITY_THRESHOLD = 0.8
OVERLAP_THRESHOLDS = (0.1, 0.5)
OCCLUSION_IOU_THRESHOLD = 0.7
BOUNDARY_MARGIN = 0.025
DECISION_TYPES = (
    "added",
    "removed_unmatched",
    "reconditioned",
    "suppressed",
    "confirmed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_margin(value: float, threshold: float, site: str) -> None:
    """Fail when a compared value lies too close to its threshold."""
    if abs(value - threshold) < THRESHOLD_MARGIN:
        raise ValueError(
            f"{site}: {value:.6f} is within {THRESHOLD_MARGIN} of {threshold}; "
            "this input cannot serve as a reference."
        )


def check_no_ties(values: np.ndarray, site: str) -> None:
    """Fail when a sort could order equal values differently across libraries."""
    if np.unique(values).size != values.size:
        raise ValueError(f"{site}: tied values {np.sort(values).tolist()}.")


def flatten(prefix: str, value: Any) -> dict[str, np.ndarray]:
    """Flatten nested module inputs and outputs into named NumPy arrays."""
    import torch

    if value is None or isinstance(value, (str, bool, int, float)):
        return {}
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu() if value.is_floating_point() else value
        return {prefix: array.detach().cpu().numpy()}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    elif hasattr(value, "tensors"):
        return flatten(prefix, value.tensors)
    else:
        return {}
    flat: dict[str, np.ndarray] = {}
    for key, item in items:
        flat |= flatten(f"{prefix}.{key}", item)
    return flat


class Recorder:
    """Save what named modules received and returned on their first call."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self._seen: set[str] = set()
        self._modules: list[Any] = []

    def watch(self, name: str, module: Any) -> None:
        # The neck's forward is called directly, which skips registered hooks.
        forward = module.forward

        def recording_forward(*args: Any, **kwargs: Any) -> Any:
            output = forward(*args, **kwargs)
            if name not in self._seen:
                self._seen.add(name)
                self.arrays |= flatten(f"inputs.{name}.args", args)
                self.arrays |= flatten(f"inputs.{name}.kwargs", kwargs)
                self.arrays |= flatten(f"outputs.{name}", output)
            return output

        module.forward = recording_forward
        self._modules.append(module)

    def missing(self, expected: Iterator[str]) -> list[str]:
        return sorted(set(expected) - self._seen)

    def close(self) -> None:
        for module in self._modules:
            del module.forward


def component_modules(model: Any) -> dict[str, Any]:
    """Map Equimo component names to the reference modules that implement them."""
    tracker = model.tracker.model
    vision = model.detector.backbone.vision_backbone
    transformer = model.detector.transformer
    return {
        "trunk": vision.trunk,
        "neck": vision,
        "text": model.detector.backbone.language_backbone,
        "detector.geometry": model.detector.geometry_encoder,
        "detector.head.fusion": transformer.encoder,
        "detector.head.decoder": transformer.decoder,
        "detector.head.scoring": model.detector.dot_prod_scoring,
        "detector.head.segmentation": model.detector.segmentation_head,
        "interactive.prompt_encoder": tracker.interactive_sam_prompt_encoder,
        "interactive.mask_decoder": tracker.interactive_sam_mask_decoder,
        "tracker.mask_decoder": tracker.sam_mask_decoder,
        "tracker.memory_encoder": tracker.maskmem_backbone,
        "tracker.memory_attention": tracker.transformer.encoder,
    }


def check_reference_revision(reference_dir: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(
            f"Expected reference revision {REFERENCE_REVISION}, got {revision}."
        )


def seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def use_deterministic_kernels() -> None:
    import torch

    # torchvision compiles roi_align on GPU regardless of the reference's options.
    # Eager kernels are reproducible, and Triton needs /sbin/ldconfig, absent on NixOS.
    torch._dynamo.config.disable = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def float32_reference() -> Iterator[None]:
    """Run the reference in float32, undoing its bfloat16 storage and ViT MLP cast."""
    import sam3.model.decoder as reference_decoder
    import sam3.model.sam3_multiplex_base  # noqa: F401
    import torch
    from sam3.model.vitdet import Mlp
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Importing the reference turns TF32 on again, for convolutions and matmul alike.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def plain_mlp(self: Any, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(self.norm(x)))

    original_mlp = Mlp.forward
    original_to = torch.Tensor.to

    def keep_float32(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        args = tuple(torch.float32 if arg is torch.bfloat16 else arg for arg in args)
        if kwargs.get("dtype") is torch.bfloat16:
            kwargs["dtype"] = torch.float32
        return original_to(self, *args, **kwargs)

    def math_kernel(_backend: SDPBackend) -> Any:
        return sdpa_kernel(SDPBackend.MATH)

    original_kernel = reference_decoder.sdpa_kernel
    Mlp.forward = plain_mlp
    torch.Tensor.to = keep_float32
    # Memory attention requests FlashAttention 2, which rejects float32.
    reference_decoder.sdpa_kernel = math_kernel
    try:
        yield
    finally:
        Mlp.forward = original_mlp
        torch.Tensor.to = original_to
        reference_decoder.sdpa_kernel = original_kernel


def exit_autocast(model: Any) -> None:
    """Close the bfloat16 context the tracker wrapper opens for its whole lifetime."""
    model.tracker.bf16_context.__exit__(None, None, None)


def write_tiny_vocabulary(source: Path, output: Path) -> None:
    """Write the header and first merges of the CLIP vocabulary, reproducibly."""
    lines = gzip.decompress(source.read_bytes()).decode("utf-8").split("\n")
    # A trailing newline would become an empty merge and shift the special tokens.
    text = "\n".join(lines[: TINY_CONFIG["num_merges"] + 1])
    output.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))


def tiny_vocabulary_size() -> int:
    return 2 * 256 + TINY_CONFIG["num_merges"] + 2


@contextmanager
def tiny_interactive_heads() -> Iterator[None]:
    """Shrink the mask decoder sizes the reference hard-codes for the full model."""
    import sam3.model.video_tracking_multiplex as tracking

    config = TINY_CONFIG
    original_transformer = tracking.TwoWayTransformer
    original_decoder = tracking.MaskDecoder
    original_multiplex = tracking.MultiplexMaskDecoder

    def transformer(**kwargs: Any) -> Any:
        kwargs |= {
            "depth": config["two_way_depth"],
            "mlp_dim": config["ffn_features"],
            "num_heads": config["head_num_heads"],
        }
        return original_transformer(**kwargs)

    def small_iou_head(decoder: Callable[..., Any]) -> Callable[..., Any]:
        return lambda **kwargs: decoder(
            **(kwargs | {"iou_head_hidden_dim": config["dim"]})
        )

    tracking.TwoWayTransformer = transformer
    tracking.MaskDecoder = small_iou_head(original_decoder)
    tracking.MultiplexMaskDecoder = small_iou_head(original_multiplex)
    try:
        yield
    finally:
        tracking.TwoWayTransformer = original_transformer
        tracking.MaskDecoder = original_decoder
        tracking.MultiplexMaskDecoder = original_multiplex


def build_tiny_tracker(device: str) -> Any:
    """Assemble the multiplex tracker the 3.1 builder creates, at the tiny sizes."""
    from sam3.model.decoder import (
        DecoupledTransformerDecoderLayerv2,
        SimpleRoPEAttention,
        TransformerEncoderDecoupledCrossAttention,
    )
    from sam3.model.memory import (
        CXBlock,
        SimpleFuser,
        SimpleMaskDownSampler,
        SimpleMaskEncoder,
    )
    from sam3.model.multiplex_utils import MultiplexController
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.model_misc import TransformerWrapper
    from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo
    from sam3.model.vl_combiner import TriHeadVisionOnly

    config = TINY_CONFIG
    dim = config["dim"]
    grid = config["img_size"] // config["patch_size"]
    position = PositionEmbeddingSine(
        num_pos_feats=dim,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=config["img_size"],
    )
    # 1152 / 1008 of the input image, as in the full builder.
    interpolation = config["img_size"] * 8 // 7
    downsampler = SimpleMaskDownSampler(
        embed_dim=dim,
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[interpolation, interpolation],
        multiplex_count=config["num_slots"],
        starting_out_chan=config["memory_start_channels"],
        input_channel_multiplier=2,
    )
    fuser_block = CXBlock(
        dim=dim,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    memory = SimpleMaskEncoder(
        out_dim=dim,
        in_dim=dim,
        position_encoding=position,
        mask_downsampler=downsampler,
        fuser=SimpleFuser(layer=fuser_block, num_layers=1),
    )
    self_attention = SimpleRoPEAttention(
        d_model=dim,
        num_heads=config["memory_heads"],
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[grid, grid],
    )
    cross_attention = SimpleRoPEAttention(
        d_model=dim,
        num_heads=config["memory_heads"],
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[grid, grid],
        rope_k_repeat=True,
    )
    layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=dim,
        num_heads=config["memory_heads"],
        dropout=0.1,
        dim_feedforward=config["ffn_features"],
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention,
        cross_attention_rope=cross_attention,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=dim,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=layer,
        num_layers=config["memory_depth"],
        use_act_checkpoint=False,
        batch_first=True,
    )
    tri_neck = build_tiny_neck()
    backbone = TriHeadVisionOnly(visual=tri_neck, n_features=dim, scalp=0)
    controller = MultiplexController(
        multiplex_count=config["num_slots"],
        eval_multiplex_count=config["num_slots"],
    )
    arguments = tiny_tracker_arguments()
    with tiny_interactive_heads():
        model = Sam3VideoTrackingMultiplexDemo(
            backbone=backbone,
            transformer=TransformerWrapper(encoder=encoder, decoder=None, d_model=dim),
            maskmem_backbone=memory,
            multiplex_controller=controller,
            **arguments,
        )
    model.to(device=device)
    del model.backbone
    model.backbone = None
    return model


def tiny_tracker_arguments() -> dict[str, Any]:
    """Tracker options of the 3.1 builder, at the tiny image size."""
    config = TINY_CONFIG
    return {
        "image_size": config["img_size"],
        "backbone_stride": config["patch_size"],
        "num_maskmem": 7,
        "use_high_res_features_in_sam": True,
        "use_obj_ptrs_in_encoder": True,
        "max_obj_ptrs_in_encoder": 16,
        "add_tpos_enc_to_obj_ptrs": True,
        "proj_tpos_enc_in_obj_ptrs": True,
        "use_mlp_for_obj_ptr_proj": True,
        "pred_obj_scores": True,
        "pred_obj_scores_mlp": True,
        "fixed_no_obj_ptr": True,
        "use_no_obj_ptr": True,
        "use_linear_no_obj_ptr": True,
        "no_obj_embed_spatial": True,
        "sincos_tpos_enc": True,
        "multimask_output_in_sam": True,
        "multimask_output_for_tracking": True,
        "multimask_min_pt_num": 0,
        "multimask_max_pt_num": 1,
        "use_multimask_token_for_obj_ptr": True,
        "num_multimask_outputs": 3,
        "apply_sigmoid_to_mask_logits_for_mem_enc": True,
        "sigmoid_scale_for_mem_enc": 2.0,
        "sigmoid_bias_for_mem_enc": -1.0,
        "non_overlap_masks_for_mem_enc": False,
        "add_output_suppression_embeddings": True,
        "add_object_conditional_embeddings": False,
        "condition_as_mask_input": True,
        "condition_as_mask_input_fg": 1.0,
        "condition_as_mask_input_bg": 0.0,
        "use_maskmem_tpos_v2": True,
        "save_image_features": True,
        "randomness_fix": True,
        "use_mask_input_as_output_without_sam": True,
        "directly_add_no_mem_embed": True,
        "iou_prediction_use_sigmoid": False,
        "forward_backbone_per_frame_for_eval": True,
        "offload_output_to_cpu_for_eval": False,
        "trim_past_non_cond_mem_for_eval": False,
        "max_cond_frames_in_attn": 4,
        "is_dynamic_model": True,
        "sam_mask_decoder_extra_args": {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        "compile_all_components": False,
        "use_memory_selection": False,
    }


def build_tiny_neck() -> Any:
    from sam3.model.necks import Sam3TriViTDetNeck
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.vitdet import ViT

    config = TINY_CONFIG
    trunk = ViT(
        img_size=config["img_size"],
        pretrain_img_size=config["pretrain_img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_features"] / config["embed_dim"],
        norm_layer="LayerNorm",
        drop_path_rate=0.0,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=config["global_blocks"],
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=config["window_size"],
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
    )
    position = PositionEmbeddingSine(
        num_pos_feats=config["dim"],
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=config["img_size"],
    )
    return Sam3TriViTDetNeck(
        trunk=trunk,
        position_encoding=position,
        d_model=config["dim"],
        scale_factors=[4.0, 2.0, 1.0],
    )


def build_tiny_detector(vocabulary: Path) -> Any:
    """Assemble the 3.1 detector at the tiny sizes, with the builder's options."""
    from torch import nn

    from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer
    from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
    from sam3.model.geometry_encoders import SequenceGeometryEncoder
    from sam3.model.maskformer_segmentation import (
        PixelDecoder,
        UniversalSegmentationHead,
    )
    from sam3.model.model_misc import (
        DotProductScoring,
        MLP,
        MultiheadAttention,
        TransformerWrapper,
    )
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.text_encoder_ve import VETextEncoder
    from sam3.model.tokenizer_ve import SimpleTokenizer
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    config = TINY_CONFIG
    dim = config["dim"]
    heads = config["head_num_heads"]
    ffn = config["ffn_features"]

    def attention(batch_first: bool = False, dropout: float = 0.1) -> Any:
        return MultiheadAttention(
            num_heads=heads, dropout=dropout, embed_dim=dim, batch_first=batch_first
        )

    fusion_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=dim,
        dim_feedforward=ffn,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=attention(batch_first=True),
        cross_attention=attention(batch_first=True),
    )
    fusion = TransformerEncoderFusion(
        layer=fusion_layer,
        num_layers=config["fusion_depth"],
        d_model=dim,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=False,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=dim,
        dim_feedforward=ffn,
        dropout=0.1,
        cross_attention=attention(),
        n_heads=heads,
        use_text_cross_attention=True,
    )
    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=config["decoder_depth"],
        num_queries=config["num_queries"],
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=dim,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=config["img_size"],
        stride=config["patch_size"],
        use_act_checkpoint=False,
        presence_token=True,
    )
    prompt_mlp = MLP(
        input_dim=dim,
        hidden_dim=ffn,
        output_dim=dim,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(dim),
    )
    scoring = DotProductScoring(d_model=dim, d_proj=dim, prompt_mlp=prompt_mlp)
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3, interpolation_mode="nearest", hidden_dim=dim
    )
    segmentation = UniversalSegmentationHead(
        hidden_dim=dim,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=False,
        cross_attend_prompt=attention(dropout=0.0),
        pixel_decoder=pixel_decoder,
    )
    geometry_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=dim,
        dim_feedforward=ffn,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=attention(),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=attention(),
    )
    geometry = SequenceGeometryEncoder(
        pos_enc=PositionEmbeddingSine(
            num_pos_feats=dim, normalize=True, scale=None, temperature=10000
        ),
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=dim,
        num_layers=config["geometry_depth"],
        layer=geometry_layer,
        use_act_ckpt=False,
        add_cls=True,
        add_post_encode_proj=True,
    )
    text_encoder = VETextEncoder(
        tokenizer=SimpleTokenizer(bpe_path=str(vocabulary)),
        d_model=dim,
        width=config["text_width"],
        heads=config["text_heads"],
        layers=config["text_depth"],
        context_length=config["context_length"],
        vocab_size=tiny_vocabulary_size(),
    )
    return Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=SAM3VLBackboneTri(
            scalp=0, visual=build_tiny_neck(), text=text_encoder
        ),
        transformer=TransformerWrapper(encoder=fusion, decoder=decoder, d_model=dim),
        segmentation_head=segmentation,
        semantic_segmentation_head=None,
        input_geometry_encoder=geometry,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )


def tracking_arguments(max_objects: int, image_size: int) -> dict[str, Any]:
    """Policy options of ``build_sam3_multiplex_video_predictor``."""
    return {
        "score_threshold_detection": 0.4,
        "det_nms_thresh": 0.1,
        "det_nms_use_iom": True,
        "assoc_iou_thresh": 0.1,
        "new_det_thresh": 0.65,
        "hotstart_delay": 15,
        "hotstart_unmatch_thresh": 8,
        "hotstart_dup_thresh": 8,
        "suppress_unmatched_only_within_hotstart": False,
        "suppress_overlapping_based_on_recent_occlusion_threshold": 0.7,
        "suppress_det_close_to_boundary": True,
        "fill_hole_area": 0,
        "recondition_every_nth_frame": 16,
        "use_iom_recondition": True,
        "iom_thresh_recondition": 0.5,
        "masklet_confirmation_enable": True,
        "reconstruction_bbox_iou_thresh": -1,
        "reconstruction_bbox_det_score": 0.8,
        "max_num_objects": max_objects,
        "postprocess_batch_size": 16,
        "use_batched_grounding": True,
        "batched_grounding_batch_size": 16,
        "max_num_kboxes": 0,
        "sprinkle_removal_area": 0,
        "is_multiplex": True,
        "image_size": image_size,
        "image_mean": (0.5, 0.5, 0.5),
        "image_std": (0.5, 0.5, 0.5),
        "compile_model": False,
    }


def build_tiny_model(vocabulary: Path, device: str) -> Any:
    """Build the tiny tracking model on the CPU RNG, then move it to ``device``."""
    import torch
    from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )

    config = TINY_CONFIG
    seed_everything(config["seed"])
    tracker = Sam3MultiplexPredictorWrapper(
        model=build_tiny_tracker(device),
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )
    detector = build_tiny_detector(vocabulary)
    model = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=detector,
        **tracking_arguments(config["max_objects"], config["img_size"]),
    )
    # A random object head would mark about half of the masks as absent.
    with torch.no_grad():
        for decoder in (
            model.tracker.model.sam_mask_decoder,
            model.tracker.model.interactive_sam_mask_decoder,
        ):
            decoder.pred_obj_score_head.layers[-1].bias.fill_(5.0)
    return model.to(device).eval()


def initial_state(
    model: Any,
    images: Any,
    device: str,
    source_size: tuple[int, int],
    is_image_only: bool = False,
) -> dict[str, Any]:
    """The state ``init_state`` builds, for frames already held in memory."""
    import torch

    state: dict[str, Any] = {
        "image_size": model.image_size,
        "num_frames": len(images),
        "device": torch.device(device),
        "orig_height": source_size[0],
        "orig_width": source_size[1],
        "constants": {},
    }
    model._construct_initial_input_batch(state, images)
    state["sam2_inference_states"] = []
    state["tracker_metadata"] = {}
    state["feature_cache"] = {}
    state["cached_frame_outputs"] = {}
    state["is_image_only"] = is_image_only
    state["action_history"] = []
    return state


def object_mask(size: int, object_id: int) -> np.ndarray:
    """The scripted mask of one object: its own block, disjoint from the others."""
    side = size // 4
    left = (object_id % 4) * side
    mask = np.zeros((size, size), bool)
    mask[side // 2 : side // 2 + side, left : left + side] = True
    return mask


def scripted_detections(
    frame: int, track_ids: list[int], size: int
) -> list[tuple[np.ndarray, float]]:
    """Detections of the tiny clip as (mask, probability), highest first.

    Objects 0, 1 and 2 start on frame 0 in one state of two buckets, and object 3
    joins the second bucket on frame 5. Object 0 keeps matching, so it is
    confirmed and reconditioned on frame 16. Object 1 is never detected again and
    is removed as unmatched on frame 8. Object 2 stops being detected after
    frame 5 and is removed as unmatched on frame 13. Object 3 keeps matching but
    weakly shares a detection with object 0, so it is removed as a duplicate on
    frame 13, which releases the second bucket. A fifth new detection on frame 6
    goes over the object limit and is dropped.
    """
    if frame == 0:
        return [
            (object_mask(size, 0), 0.91),
            (object_mask(size, 1), 0.86),
            (object_mask(size, 2), 0.84),
        ]
    detections = [(object_mask(size, 0), 0.93)]
    if 2 in track_ids and frame <= 5:
        detections.append((object_mask(size, 2), 0.92))
    if 3 in track_ids:
        detections.append((object_mask(size, 3), 0.90))
        detections.append((weakly_shared_mask(size), 0.71))
    if frame == 5:
        detections.append((object_mask(size, 3), 0.77))
    if frame == 6:
        detections.append((dropped_detection_mask(size), 0.75))
    return sorted(detections, key=lambda item: -item[1])


def weakly_shared_mask(size: int) -> np.ndarray:
    """A detection touching objects 0 and 3 with an overlap of 3/16 each.

    A strong match to two tracks marks the detection as ambiguous and hides it
    from the duplicate count, so only an overlap between 0.1 and 0.5 counts.
    """
    side = size // 4
    top = side // 2
    mask = np.zeros((size, size), bool)
    mask[top, :3] = True
    mask[top, 3 * side : 3 * side + 3] = True
    mask[top + side + 1, :] = True
    return mask


def dropped_detection_mask(size: int) -> np.ndarray:
    """A block disjoint from every object, for the detection over the limit."""
    side = size // 4
    mask = np.zeros((size, size), bool)
    mask[2 * side : 3 * side, side : 2 * side] = True
    return mask


def scripted_tracks(track_ids: list[int], size: int) -> np.ndarray:
    """Track logits of the tiny clip: every live object on its own block."""
    masks = np.full((len(track_ids), size, size), -8.0, np.float32)
    for row, object_id in enumerate(track_ids):
        masks[row][object_mask(size, object_id)] = 8.0
    return masks


def pad_detections(
    detections: list[tuple[np.ndarray, float]], queries: int, size: int, device: str
) -> tuple[dict[str, Any], Any]:
    """Pack scripted detections like the reference's sorted per-frame ``det_out``."""
    import torch

    masks = np.zeros((queries, size, size), np.float32) - 8.0
    scores = np.linspace(0.2, 0.05, queries, dtype=np.float32)
    boxes = np.tile(np.array([0.3, 0.3, 0.7, 0.7], np.float32), (queries, 1))
    for query, (mask, probability) in enumerate(detections):
        masks[query] = np.where(mask, 8.0, -8.0)
        scores[query] = probability
        rows, columns = np.nonzero(mask)
        if rows.size:
            boxes[query] = (
                columns.min() / size,
                rows.min() / size,
                (columns.max() + 1) / size,
                (rows.max() + 1) / size,
            )
    keep = scores > 0.4
    det_out = {
        "mask": torch.from_numpy(masks).to(device),
        "scores": torch.from_numpy(scores).to(device),
        "bbox": torch.from_numpy(boxes).to(device),
    }
    return det_out, torch.from_numpy(keep).to(device)


STATE_OUTPUTS = ("maskmem_features", "obj_ptr", "pred_masks", "object_score_logits")


def frame_output(state: dict[str, Any], frame: int) -> tuple[str, dict[str, Any]]:
    """Return where a state stores this frame, and the stored output."""
    outputs = state["output_dict"]
    storage = (
        "cond_frame_outputs"
        if frame in outputs["cond_frame_outputs"]
        else "non_cond_frame_outputs"
    )
    return storage, outputs[storage][frame]


class DecisionTrace:
    """Record the policy's inputs and decisions for every planned frame."""

    def __init__(self, model: Any, scripted: bool, assert_margins: bool) -> None:
        self.model = model
        self.scripted = scripted
        self.assert_margins = assert_margins
        self.frames: list[dict[str, Any]] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.removal_reasons: dict[int, str] = {}
        self._prefix = ""
        self._suppressed: list[int] = []
        self._reconditioned_pairs: list[list[int]] = []
        self._planning = model.run_tracker_update_planning_phase
        self._hotstart = model._process_hotstart_gpu
        self._execution = model.run_tracker_update_execution_phase
        self._recondition = model._recondition_masklets
        self._suppress = model._get_objects_to_suppress_based_on_most_recently_occluded
        model.run_tracker_update_planning_phase = self.plan
        model._process_hotstart_gpu = self.hotstart
        model.run_tracker_update_execution_phase = self.execute
        model._recondition_masklets = self.recondition
        model._get_objects_to_suppress_based_on_most_recently_occluded = self.suppress

    def hotstart(self, **kwargs: Any) -> Any:
        to_remove, to_suppress, metadata = self._hotstart(**kwargs)
        if metadata.get("N_obj", 0) > 0:
            ids = kwargs["tracker_metadata_prev"]["obj_ids_all_gpu"]
            unmatched = metadata["consecutive_unmatch_count"].cpu().numpy()
            for index in np.nonzero(to_remove.cpu().numpy())[0]:
                reason = (
                    "removed_unmatched"
                    if unmatched[index] >= self.model.hotstart_unmatch_thresh
                    else "removed_duplicate"
                )
                self.removal_reasons[int(ids[index])] = reason
        return to_remove, to_suppress, metadata

    def recondition(
        self,
        frame_idx: int,
        det_out: dict,
        matches: dict[int, int],
        states: list[Any],
        metadata: dict,
        scores: Any,
        masks: Any,
    ) -> Any:
        ids = metadata["obj_ids_all_gpu"].tolist()
        probabilities = scores.float().sigmoid().cpu().numpy()
        self._reconditioned_pairs = sorted(
            [int(obj), int(query)]
            for obj, query in matches.items()
            if probabilities[ids.index(obj)] > TRACK_PROBABILITY_THRESHOLD
        )
        return self._recondition(
            frame_idx, det_out, matches, states, metadata, scores, masks
        )

    def suppress(
        self,
        masks: Any,
        last_occluded: Any,
        obj_ids: np.ndarray,
        frame_idx: int | None = None,
        reverse: bool = False,
    ) -> Any:
        to_suppress = self._suppress(masks, last_occluded, obj_ids, frame_idx, reverse)
        self._suppressed = sorted(
            int(obj_ids[index]) for index in np.nonzero(to_suppress.cpu().numpy())[0]
        )
        return to_suppress

    def check_margins(
        self, scores: np.ndarray, overlap: np.ndarray, boxes: np.ndarray
    ) -> None:
        check_no_ties(scores, "detection scores")
        for score in scores:
            for threshold in DETECTION_THRESHOLDS:
                check_margin(float(score), threshold, "detection probability")
        for value in overlap.ravel():
            for threshold in OVERLAP_THRESHOLDS:
                check_margin(float(value), threshold, "mask overlap")
        for box in boxes:
            for center in ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2):
                check_margin(float(center), BOUNDARY_MARGIN, "box center")
                check_margin(float(center), 1 - BOUNDARY_MARGIN, "box center")

    def plan(self, **kwargs: Any) -> Any:
        import torch
        from sam3.model.sam3_video_base import mask_iom

        frame = kwargs["frame_idx"]
        prior = kwargs["tracker_metadata_prev"]
        tracks = kwargs["tracker_low_res_masks_global"]
        track_scores = kwargs["tracker_obj_scores_global"]
        track_ids = [int(obj) for obj in prior["obj_ids_all_gpu"]]
        self._prefix = f"frames.{frame:03d}.pass{self._pass(frame)}"
        self._suppressed = []
        self._reconditioned_pairs = []
        if self.scripted:
            self.record_raw_tracks(frame, tracks, track_scores, kwargs)
            # The outputs and memory encoder read this same tensor afterwards.
            tracks[:] = torch.from_numpy(
                scripted_tracks(track_ids, tracks.shape[-1])
            ).to(tracks.device)
            self.script_stored_masks(frame, tracks, track_ids, kwargs)
            detections = scripted_detections(frame, track_ids, tracks.shape[-1])
            scripted, keep = pad_detections(
                detections,
                TINY_CONFIG["num_queries"],
                tracks.shape[-1],
                tracks.device,
            )
            # The execution phase and the outputs read this same dictionary.
            kwargs["det_out"].clear()
            kwargs["det_out"].update(scripted)
            kwargs["det_keep"] = keep
        det_out, keep = kwargs["det_out"], kwargs["det_keep"]
        scores = det_out["scores"].float().cpu().numpy()
        kept = keep.cpu().numpy().astype(bool)
        probabilities = track_scores.float().sigmoid().cpu().numpy()
        if tracks.shape[0] and det_out["mask"].shape[0]:
            overlap = mask_iom(
                det_out["mask"] > 0,
                torch.nn.functional.interpolate(
                    tracks[None].float(),
                    size=det_out["mask"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
                > 0,
            )
            overlap = overlap.cpu().numpy()
        else:
            overlap = np.zeros((scores.size, tracks.shape[0]), np.float32)
        if self.assert_margins:
            self.check_margins(
                scores[kept], overlap[kept], det_out["bbox"][keep].cpu().numpy()
            )
            for probability in probabilities:
                check_margin(
                    float(probability), TRACK_PROBABILITY_THRESHOLD, "track probability"
                )

        prefix = self._prefix
        self.arrays[f"{prefix}.det_scores"] = scores.astype(np.float32)
        self.arrays[f"{prefix}.det_keep"] = kept
        self.arrays[f"{prefix}.det_boxes"] = det_out["bbox"].cpu().numpy()
        self.arrays[f"{prefix}.det_masks"] = (
            det_out["mask"].float().cpu().numpy().astype(np.float32)
        )
        self.arrays[f"{prefix}.track_masks"] = tracks.float().cpu().numpy()
        self.arrays[f"{prefix}.track_score_logits"] = track_scores.float().cpu().numpy()

        plan, metadata = self._planning(**kwargs)
        self.frames.append(self.summarize(frame, track_ids, plan, metadata))
        return plan, metadata

    def _pass(self, frame: int) -> int:
        return sum(record["frame"] == frame for record in self.frames)

    def record_raw_tracks(
        self, frame: int, tracks: Any, track_scores: Any, kwargs: dict[str, Any]
    ) -> None:
        """Store what the tracker computed, before the script replaces its masks."""
        prefix = self._prefix
        self.arrays[f"{prefix}.raw_track_masks"] = tracks.float().cpu().numpy()
        self.arrays[f"{prefix}.raw_track_score_logits"] = (
            track_scores.float().cpu().numpy()
        )
        for index, state in enumerate(kwargs["tracker_states_local"]):
            _, output = frame_output(state, frame)
            self.arrays[f"{prefix}.state{index}.raw_obj_ptr"] = (
                output["obj_ptr"].float().cpu().numpy()
            )

    def script_stored_masks(
        self, frame: int, tracks: Any, track_ids: list[int], kwargs: dict[str, Any]
    ) -> None:
        """Replace each state's stored masks, which the re-encoding path reads."""
        for state in kwargs["tracker_states_local"]:
            _, output = frame_output(state, frame)
            rows = [track_ids.index(int(obj)) for obj in state["obj_ids"]]
            output["pred_masks"][:] = tracks[rows][:, None].to(
                output["pred_masks"].device
            )

    def summarize(
        self, frame: int, track_ids: list[int], plan: dict, metadata: dict
    ) -> dict[str, Any]:
        new_objects = plan["new_det_obj_ids"].tolist()
        queries = plan["new_det_fa_inds"].tolist()
        removed = sorted(int(obj) for obj in plan["obj_ids_newly_removed"])
        reasons = [
            self.removal_reasons.get(obj, "removed_unmatched") for obj in removed
        ]
        status = metadata["rank0_metadata"].get("masklet_confirmation", {})
        return {
            "frame": frame,
            "track_ids": track_ids,
            "added": [[int(o), int(q)] for o, q in zip(new_objects, queries)],
            "removed_unmatched": [
                o for o, r in zip(removed, reasons) if r == "removed_unmatched"
            ],
            "removed_duplicate": [
                o for o, r in zip(removed, reasons) if r == "removed_duplicate"
            ],
            "reconditioned": sorted(int(o) for o in plan["reconditioned_obj_ids"]),
            "reconditioned_pairs": self._reconditioned_pairs,
            "suppressed": self._suppressed,
            "dropped": int(plan["num_obj_dropped_due_to_limit"]),
            "confirmation": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in status.items()
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        states = self._execution(**kwargs)
        record = self.frames[-1]
        record["states"] = [[int(obj) for obj in state["obj_ids"]] for state in states]
        if self.scripted:
            record["state_details"] = [
                self.record_state(index, state, record["frame"])
                for index, state in enumerate(states)
            ]
        return states

    def record_state(
        self, index: int, state: dict[str, Any], frame: int
    ) -> dict[str, Any]:
        """Store the memory and outputs a state holds for this frame."""
        storage, output = frame_output(state, frame)
        for name in STATE_OUTPUTS:
            self.arrays[f"{self._prefix}.state{index}.{name}"] = (
                output[name].float().cpu().numpy()
            )
        return {
            "storage": storage,
            "obj_ids": [int(obj) for obj in state["obj_ids"]],
            "conditioning_objects": sorted(
                int(obj) for obj in output["conditioning_objects"]
            ),
            "assignments": [
                [int(entry) for entry in bucket]
                for bucket in state["multiplex_state"].assignments
            ],
        }


def render_outputs(
    results: Iterator[tuple[int, dict[str, Any]]], prefix: str
) -> dict[str, np.ndarray]:
    """Store the per-frame masks the reference emits."""
    arrays: dict[str, np.ndarray] = {}
    for frame, output in results:
        key = f"{prefix}.{frame:03d}"
        arrays[f"{key}.object_ids"] = np.asarray(output["out_obj_ids"], np.int64)
        arrays[f"{key}.scores"] = np.asarray(output["out_probs"], np.float32)
        arrays[f"{key}.boxes"] = np.asarray(output["out_boxes_xywh"], np.float32)
        arrays[f"{key}.masks"] = np.asarray(output["out_binary_masks"], bool)
    return arrays


def add_box_prompt_cases(model: Any, recorder: Recorder, device: str) -> None:
    """Call the interactive head with a box; no 3.1 pipeline sends it one."""
    import torch

    tracker = model.tracker.model
    size = TINY_CONFIG["img_size"]
    boxes = torch.tensor([[[0.2, 0.3], [0.7, 0.8]]], device=device) * size
    labels = torch.tensor([[2, 3]], device=device)
    with torch.inference_mode():
        sparse, dense = tracker.interactive_sam_prompt_encoder(
            points=(boxes, labels), boxes=None, masks=None
        )
    recorder.arrays |= flatten("inputs.interactive.box_prompt.points", boxes)
    recorder.arrays |= flatten("inputs.interactive.box_prompt.labels", labels)
    recorder.arrays |= flatten("outputs.interactive.box_prompt.sparse", sparse)
    recorder.arrays |= flatten("outputs.interactive.box_prompt.dense", dense)


def watch_joint_scores(model: Any, recorder: Recorder) -> None:
    """Record the first joint logits and boxes the detector computes."""
    update = model.detector._update_scores_and_boxes
    recorded = False

    def recording_update(out: dict, *args: Any, **kwargs: Any) -> Any:
        nonlocal recorded
        result = update(out, *args, **kwargs)
        if not recorded:
            recorded = True
            names = ("pred_logits", "pred_boxes", "presence_logit_dec")
            recorder.arrays |= flatten(
                "outputs.detector.joint", {name: out[name] for name in names}
            )
        return result

    model.detector._update_scores_and_boxes = recording_update


def add_geometry_prompt_cases(model: Any, recorder: Recorder, device: str) -> None:
    """Encode exemplar points and boxes, with and without padded entries."""
    import torch
    from sam3.model.geometry_encoders import Prompt

    grid = TINY_CONFIG["img_size"] // TINY_CONFIG["patch_size"]
    image = "inputs.detector.geometry.kwargs"
    features = [torch.from_numpy(recorder.arrays[f"{image}.img_feats.0"]).to(device)]
    positions = [
        torch.from_numpy(recorder.arrays[f"{image}.img_pos_embeds.0"]).to(device)
    ]
    boxes = torch.tensor(
        [[[0.4, 0.5, 0.3, 0.4]], [[0.2, 0.3, 0.1, 0.1]]], device=device
    )
    points = torch.tensor([[[0.3, 0.6]], [[0.7, 0.2]]], device=device)
    labels = torch.tensor([[1], [0]], device=device, dtype=torch.long)
    cases = {
        "full": torch.tensor([[False, False]], device=device),
        "padded": torch.tensor([[False, True]], device=device),
    }
    for name, padding in cases.items():
        prompt = Prompt(
            box_embeddings=boxes,
            box_mask=padding,
            box_labels=labels,
            point_embeddings=points,
            point_mask=padding,
            point_labels=labels,
        )
        with torch.inference_mode():
            tokens, mask = model.detector.geometry_encoder(
                geo_prompt=prompt,
                img_feats=features,
                img_sizes=[(grid, grid)],
                img_pos_embeds=positions,
            )
        prefix = f"inputs.detector.geometry_prompt.{name}"
        recorder.arrays |= flatten(f"{prefix}.boxes", boxes)
        recorder.arrays |= flatten(f"{prefix}.points", points)
        recorder.arrays |= flatten(f"{prefix}.labels", labels)
        recorder.arrays |= flatten(f"{prefix}.padding", padding)
        recorder.arrays |= flatten(
            f"outputs.detector.geometry_prompt.{name}", (tokens, mask)
        )


def add_mask_nms_case(recorder: Recorder, device: str) -> None:
    """Run the reference's IoM mask NMS on overlapping random rectangles."""
    import torch
    from sam3.model.sam3_multiplex_detector_utils import nms_masks

    random = np.random.default_rng(41)
    size = 16
    masks = np.zeros((12, size, size), np.float32) - 1.0
    for mask in masks:
        top, left = random.integers(0, size - 4, 2)
        height, width = random.integers(3, 12, 2)
        mask[top : top + height, left : left + width] = 1.0
    probabilities = random.uniform(0.2, 0.95, 12).astype(np.float32)
    check_no_ties(probabilities, "mask NMS case")
    keep = nms_masks(
        torch.from_numpy(probabilities).to(device),
        torch.from_numpy(masks).to(device),
        prob_threshold=0.4,
        iou_threshold=0.1,
        nms_use_iom=True,
    )
    recorder.arrays["inputs.nms.probabilities"] = probabilities
    recorder.arrays["inputs.nms.masks"] = masks
    recorder.arrays["outputs.nms.keep"] = keep.cpu().numpy()
    batched_keep = nms_masks(
        torch.from_numpy(probabilities).to(device)[None],
        torch.from_numpy(masks).to(device)[None],
        prob_threshold=0.4,
        iou_threshold=0.1,
        nms_use_iom=True,
    )
    recorder.arrays["outputs.nms.keep_batched"] = batched_keep[0].cpu().numpy()


def add_antialias_resize_cases(recorder: Recorder, device: str) -> None:
    """Run torch's antialiased bilinear resize at the sizes the tracker uses."""
    import torch

    random = np.random.default_rng(43)
    cases = {
        "input_to_source": ((64, 64), (40, 60)),
        "source_to_mask": ((40, 60), (16, 16)),
        "upsample": ((16, 16), (40, 60)),
        "mixed": ((16, 64), (40, 40)),
    }
    for name, (source, target) in cases.items():
        values = random.normal(size=(2, *source)).astype(np.float32)
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(values[:, None]).to(device),
            size=target,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0]
        recorder.arrays[f"inputs.antialias.{name}"] = values
        recorder.arrays[f"outputs.antialias.{name}"] = resized.cpu().numpy()


def run_tiny_clip(model: Any, images: Any, device: str) -> dict[str, np.ndarray]:
    """Track a concept through the clip, then add one object with points."""
    import torch

    config = TINY_CONFIG
    state = initial_state(model, images, device, config["source_size"])
    arrays: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        model.add_prompt(state, frame_idx=0, text_str=CONCEPT)
        arrays |= render_outputs(
            model.propagate_in_video(state, start_frame_idx=0, reverse=False),
            "emitted",
        )
        point_frame = config["num_frames"] // 2
        points = torch.tensor([[0.4, 0.5], [0.8, 0.2]], dtype=torch.float32)
        labels = torch.tensor([1, 0], dtype=torch.int32)
        object_id = 1 + max(
            int(obj) for obj in state["tracker_metadata"]["obj_id_to_score"]
        )
        arrays["inputs.points.coordinates"] = points.numpy()
        arrays["inputs.points.labels"] = labels.numpy()
        arrays["inputs.points.frame"] = np.asarray(point_frame, np.int64)
        arrays["inputs.points.object_id"] = np.asarray(object_id, np.int64)
        model.add_prompt(
            state,
            frame_idx=point_frame,
            points=points,
            point_labels=labels,
            obj_id=object_id,
        )
        arrays |= render_outputs(
            model.propagate_in_video(state, start_frame_idx=point_frame, reverse=False),
            "emitted_points",
        )
        arrays |= point_track_arrays(state, object_id)
    return arrays


def point_track_arrays(state: dict[str, Any], object_id: int) -> dict[str, np.ndarray]:
    """Store the singleton state a new object added with points is tracked in."""
    point_state = next(
        inner
        for inner in state["sam2_inference_states"]
        if inner["obj_ids"] == [object_id]
    )
    arrays: dict[str, np.ndarray] = {}
    for storage in ("cond_frame_outputs", "non_cond_frame_outputs"):
        for frame, output in point_state["output_dict"][storage].items():
            prefix = f"points.{frame:03d}"
            arrays[f"{prefix}.is_cond"] = np.asarray(storage == "cond_frame_outputs")
            for name in STATE_OUTPUTS:
                arrays[f"{prefix}.{name}"] = output[name].float().cpu().numpy()
    return arrays


def save_fixture(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted keys and NumPy's fixed archive timestamps make the output reproducible.
    np.savez_compressed(path, **{key: arrays[key] for key in sorted(arrays)})


def versions() -> dict[str, str]:
    import timm
    import torch
    import torchvision

    return {
        "numpy": np.__version__,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "timm": timm.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def provenance(path: Path, command: str, seed: int, input_shape: list[int]) -> dict:
    """The ``reference_provenance.json`` entry of a generated fixture."""
    with np.load(path, allow_pickle=False) as fixture:
        schema = {
            key: {"dtype": str(fixture[key].dtype), "shape": list(fixture[key].shape)}
            for key in sorted(fixture.files)
        }
    return {
        "arrays": schema,
        "comparison": {"atol": 1e-5, "metric": "allclose", "rtol": 1e-5},
        "environment": versions(),
        "generator": command,
        "input_shape": input_shape,
        "seed": seed,
        "sha256": sha256_file(path),
        "tolerance": 1e-5,
        "upstream": {
            "checkpoint": "none (random weights)",
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
        },
    }


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def generate_tokenizer(reference_dir: Path, output_dir: Path) -> None:
    """Write the tiny vocabulary and the reference tokenizer's ids for it."""
    from sam3.model.tokenizer_ve import SimpleTokenizer

    check_reference_revision(reference_dir)
    vocabulary = output_dir / "tiny_clip_bpe.txt.gz"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tiny_vocabulary(
        reference_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz", vocabulary
    )
    tokenizer = SimpleTokenizer(bpe_path=str(vocabulary))
    context_length = 32
    ids = tokenizer(list(TOKENIZER_CORPUS), context_length=context_length)
    write_json(
        output_dir / "clip_tokenizer_reference.json",
        {
            "context_length": context_length,
            "corpus": list(TOKENIZER_CORPUS),
            "ids": ids.tolist(),
            "reference_revision": REFERENCE_REVISION,
            "vocabulary_sha256": sha256_file(vocabulary),
            "vocabulary_size": tokenizer.vocab_size,
        },
    )


def blocky_mask(random: np.random.Generator, size: int, block: int) -> np.ndarray:
    """Noise repeated in blocks, which compresses well at full mask resolution."""
    coarse = random.normal(size=(size // block, size // block)).astype(np.float32)
    return coarse.repeat(block, axis=0).repeat(block, axis=1)[None]


def operator_cases() -> dict[str, np.ndarray]:
    """Inputs on the map border, sub-pixel, and at the model's resize sizes."""
    random = np.random.default_rng(23)
    features = random.normal(size=(3, 72, 72)).astype(np.float32)
    boxes = np.asarray(
        [
            [10.0, 12.0, 30.5, 40.25],
            [0.0, 0.0, 71.5, 71.5],
            [70.25, 70.5, 72.0, 72.0],
            [33.1, 20.2, 33.6, 20.9],
            [-2.0, 5.0, 9.0, 74.0],
        ],
        np.float32,
    )
    points = np.asarray(
        [[0.0, 0.0], [-1.0, -1.0], [1.0, 1.0], [0.37, -0.81], [1.2, 0.5]],
        np.float32,
    )
    return {
        "inputs.features": features,
        "inputs.boxes": boxes,
        "inputs.points": points,
        "inputs.mask_1008": blocky_mask(random, 1008, 28),
        "inputs.mask_288": blocky_mask(random, 288, 8),
    }


def generate_operators(output_dir: Path, device: str) -> dict:
    """Run torch and torchvision on the operators Equimo reimplements."""
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align

    use_deterministic_kernels()
    arrays = operator_cases()
    features = torch.from_numpy(arrays["inputs.features"]).to(device)[None]
    boxes = torch.from_numpy(arrays["inputs.boxes"]).to(device)
    points = torch.from_numpy(arrays["inputs.points"]).to(device)[None, None]
    mask_1008 = torch.from_numpy(arrays["inputs.mask_1008"]).to(device)[None]
    mask_288 = torch.from_numpy(arrays["inputs.mask_288"]).to(device)[None]
    arrays["outputs.roi_align"] = (
        roi_align(
            features, [boxes], output_size=7, spatial_scale=1.0, sampling_ratio=-1
        )
        .cpu()
        .numpy()
    )
    arrays["outputs.grid_sample"] = (
        F.grid_sample(features, points, mode="bilinear", align_corners=False)[0, :, 0]
        .T.cpu()
        .numpy()
    )
    resizes = {
        "resize_1008_1152": (mask_1008, 1152, False),
        "resize_1008_1152_antialias": (mask_1008, 1152, True),
        "resize_288_1008": (mask_288, 1008, False),
    }
    for seed, (name, (mask, size, antialias)) in enumerate(sorted(resizes.items())):
        output = F.interpolate(
            mask,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=antialias,
        )
        for part, values in seeded_subset(output.cpu().numpy(), 4096, seed).items():
            arrays[f"outputs.{name}.{part}"] = values
    path = output_dir / "vision_ops_reference.npz"
    save_fixture(path, arrays)
    entry = provenance(
        path,
        "python models/sam31_reference.py operators --output-dir /path/to/output",
        23,
        list(arrays["inputs.features"].shape),
    )
    entry["upstream"]["checkpoint"] = "none (operator fixture)"
    return {path.name: entry}


def tiny_frames(device: str) -> tuple[Any, np.ndarray]:
    """Normalized frames, from a generator independent of the model's seed."""
    import torch

    config = TINY_CONFIG
    size = config["img_size"]
    random = np.random.default_rng(config["seed"])
    pixels = random.integers(
        0, 256, (config["num_frames"], 3, size, size), dtype=np.uint8
    )
    normalized = pixels.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(normalized).to(device), pixels


def generate_tiny(reference_dir: Path, output_dir: Path, device: str) -> dict:
    """Generate the tiny random-weight fixture and its decision trace."""
    check_reference_revision(reference_dir)
    use_deterministic_kernels()
    generate_tokenizer(reference_dir, output_dir)
    vocabulary = output_dir / "tiny_clip_bpe.txt.gz"
    with float32_reference():
        model = build_tiny_model(vocabulary, device)
        exit_autocast(model)
        state = {
            f"state.{name}": value.detach().float().cpu().numpy()
            if value.is_floating_point()
            else value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        }
        recorder = Recorder()
        modules = component_modules(model)
        for name, module in modules.items():
            recorder.watch(name, module)
        watch_joint_scores(model, recorder)
        trace = DecisionTrace(model, scripted=True, assert_margins=True)
        images, pixels = tiny_frames(device)
        arrays = run_tiny_clip(model, images, device)
        add_box_prompt_cases(model, recorder, device)
        add_geometry_prompt_cases(model, recorder, device)
        add_mask_nms_case(recorder, device)
        add_antialias_resize_cases(recorder, device)
        recorder.close()
    if missing := recorder.missing(iter(modules)):
        raise RuntimeError(f"Components never ran: {missing}")
    arrays |= state | recorder.arrays | trace.arrays
    arrays["inputs.frames"] = pixels
    path = output_dir / "sam3_tiny_reference.npz"
    save_fixture(path, arrays)
    write_json(
        path.with_suffix(".json"),
        {
            "concept": CONCEPT,
            "config": TINY_CONFIG,
            "decisions": trace.frames,
            "reference_revision": REFERENCE_REVISION,
        },
    )
    return {
        path.name: provenance(
            path,
            "python models/sam31_reference.py tiny "
            "--reference-dir /path/to/sam3 --output-dir /path/to/output",
            TINY_CONFIG["seed"],
            [3, TINY_CONFIG["img_size"], TINY_CONFIG["img_size"]],
        )
    }


def build_full_model(checkpoint: Path, device: str) -> Any:
    """Build the shipped 3.1 predictor without FlashAttention or real-valued RoPE."""
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint digest {actual}, expected {CHECKPOINT_SHA256}.")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint),
        max_num_objects=FULL_MAX_OBJECTS,
        multiplex_count=FULL_MULTIPLEX_COUNT,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    return predictor.model.to(device).eval()


@contextmanager
def reference_precision(precision: str, build: Callable[[], Any]) -> Iterator[Any]:
    """Build a model in the requested precision and keep that precision active."""
    if precision == "bfloat16":
        yield build()
        return
    import torch

    with float32_reference():
        model = build()
        exit_autocast(model)
        # Building the video predictor turns TF32 on again.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield model


def seeded_subset(array: np.ndarray, count: int, seed: int) -> dict[str, np.ndarray]:
    """A fixed subset of a large output, stored with its indices."""
    if array.size <= count:
        return {"values": array}
    random = np.random.default_rng(seed)
    indices = np.sort(random.choice(array.size, count, replace=False))
    return {"indices": indices, "values": array.ravel()[indices]}


def normalized_frame(pixels: np.ndarray, device: str) -> Any:
    """Scale a resized uint8 image with the reference's mean and std of 0.5."""
    import torch

    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0
    return ((tensor - 0.5) / 0.5).to(device)


def resized_image(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize a single image as the reference's image loader does."""
    from PIL import Image

    with Image.open(path) as image:
        return (
            np.array(image.convert("RGB").resize((size, size))),
            (image.height, image.width),
        )


def resized_video_frame(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize a decoded frame as the reference's video loader does."""
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read image {path}.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return (
        cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC),
        rgb.shape[:2],
    )


def generate_image(
    checkpoint: Path, image: Path, concept: str, output_dir: Path, precision: str
) -> dict:
    """Record every image component's outputs on one real image with full weights."""
    device = "cuda"
    use_deterministic_kernels()
    with reference_precision(
        precision, partial(build_full_model, checkpoint, device)
    ) as model:
        recorder = Recorder()
        for name, module in component_modules(model).items():
            recorder.watch(name, module)
        watch_joint_scores(model, recorder)
        DecisionTrace(model, scripted=False, assert_margins=True)
        resized, source = resized_image(image, 1008)
        images = normalized_frame(resized, device)[None]
        state = initial_state(model, images, device, source, is_image_only=True)
        arrays = run_single_image(model, state, concept)
        recorder.close()
        arrays |= concept_tokens(model, concept)
        interactive = run_interactive_prompts(model, images, device, source, arrays)
        arrays |= run_exemplars_alone(model, images, device, source, arrays)
    subset: dict[str, np.ndarray] = {"inputs.image": resized}
    recorded = recorder.arrays | interactive
    # The first text batch holds the concept, then the placeholder names.
    recorded["outputs.text.0"] = recorded["outputs.text.0"][:1]
    for name in ("outputs.text.1", "outputs.text.2"):
        recorded[name] = recorded[name][:, :1]
    kept = (name for name in recorded if name.startswith(IMAGE_RECORDED_OUTPUTS))
    for seed, name in enumerate(sorted(kept)):
        for part, values in seeded_subset(recorded[name], 4096, seed).items():
            subset[f"{name}.{part}"] = values
    subset |= arrays
    suffix = "" if precision == "float32" else f"_{precision}"
    path = output_dir / f"sam31_image_reference{suffix}.npz"
    save_fixture(path, subset)
    return {path.name: image_provenance(path, concept)}


def concept_tokens(model: Any, concept: str) -> dict[str, np.ndarray]:
    """Store the ids the reference tokenizer gives the concept, and their padding."""
    language = model.detector.backbone.language_backbone
    tokens = language.tokenizer([concept], context_length=language.context_length)
    return {
        "inputs.text.token_ids": tokens.cpu().numpy(),
        "inputs.text.padding_mask": (tokens == 0).cpu().numpy(),
    }


def run_exemplars_alone(
    model: Any,
    images: Any,
    device: str,
    source: tuple[int, int],
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Prompt with one box and no text, on the object ``EXEMPLAR_OBJECT``.

    With no text, the 3.1 predictor leaves its placeholder string in the text batch
    and encodes that, so the placeholder's token ids are stored as well.
    """
    import torch

    boxes = arrays["emitted.000.boxes"][EXEMPLAR_OBJECT : EXEMPLAR_OBJECT + 1]
    state = initial_state(model, images, device, source, is_image_only=True)
    with torch.inference_mode():
        _, output = model.add_prompt(
            state,
            frame_idx=0,
            boxes_xywh=boxes,
            box_labels=np.ones(1, np.int64),
        )
    tokens = concept_tokens(model, "<text placeholder>")
    return render_outputs(iter([(0, output)]), "emitted.exemplar") | {
        "inputs.exemplar.boxes_xywh": boxes,
        "inputs.exemplar.text.token_ids": tokens["inputs.text.token_ids"],
        "inputs.exemplar.text.padding_mask": tokens["inputs.text.padding_mask"],
    }


def run_interactive_prompts(
    model: Any,
    images: Any,
    device: str,
    source: tuple[int, int],
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run the interactive head once for a point and once for box corners.

    Both prompts sit on the first emitted object, so they do real work.
    """
    import torch

    x, y, width, height = arrays["emitted.000.boxes"][0]
    center = (x + width / 2, y + height / 2)
    prompts = {
        "point": ([center], [1]),
        "box": ([(x, y), (x + width, y + height)], [2, 3]),
    }
    tracker = model.tracker.model
    recorded: dict[str, np.ndarray] = {}
    for name, (points, labels) in prompts.items():
        coordinates = torch.tensor(points, dtype=torch.float32)
        point_labels = torch.tensor(labels, dtype=torch.int32)
        arrays[f"inputs.interactive.{name}.points"] = coordinates.numpy()
        arrays[f"inputs.interactive.{name}.labels"] = point_labels.numpy()
        recorder = Recorder()
        recorder.watch(
            "interactive.prompt_encoder", tracker.interactive_sam_prompt_encoder
        )
        recorder.watch("interactive.mask_decoder", tracker.interactive_sam_mask_decoder)
        state = initial_state(model, images, device, source, is_image_only=True)
        with torch.inference_mode():
            model.add_prompt(
                state,
                frame_idx=0,
                points=coordinates,
                point_labels=point_labels,
                obj_id=0,
            )
        recorder.close()
        recorded |= {
            key.replace("outputs.interactive.", f"outputs.interactive.{name}."): value
            for key, value in recorder.arrays.items()
            if key.startswith("outputs.")
        }
    return recorded


def run_single_image(model: Any, state: dict, concept: str) -> dict[str, np.ndarray]:
    import torch

    with torch.inference_mode():
        _, output = model.add_prompt(state, frame_idx=0, text_str=concept)
    return render_outputs(iter([(0, output)]), "emitted")


def image_provenance(path: Path, concept: str) -> dict:
    """The provenance of a full-weight fixture, compared by mean absolute error."""
    entry = provenance(
        path,
        "python models/sam31_reference.py image --checkpoint /path/to/sam3.1_multiplex.pt "
        f"--image /path/to/image --concept {json.dumps(concept)} "
        "--output-dir /path/to/output",
        0,
        [3, 1008, 1008],
    )
    entry["comparison"] = {
        "atol": IMAGE_MEAN_ABSOLUTE_ERROR,
        "metric": "mean_absolute_error",
    }
    entry["tolerance"] = IMAGE_MEAN_ABSOLUTE_ERROR
    entry["upstream"]["checkpoint"] = CHECKPOINT_SHA256
    return entry


def generate_video(
    checkpoint: Path, frames_dir: Path, concept: str, output_dir: Path, precision: str
) -> None:
    """Record the decision trace and per-frame masks of a clip with full weights."""
    import torch

    device = "cuda"
    use_deterministic_kernels()
    paths = sorted(frames_dir.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No JPEG frames in {frames_dir}.")
    with reference_precision(
        precision, partial(build_full_model, checkpoint, device)
    ) as model:
        trace = DecisionTrace(model, scripted=False, assert_margins=True)
        loaded = [resized_video_frame(path, 1008) for path in paths]
        images = torch.stack([normalized_frame(pixels, device) for pixels, _ in loaded])
        state = initial_state(model, images, device, loaded[0][1])
        with torch.inference_mode():
            model.add_prompt(state, frame_idx=0, text_str=concept)
            arrays = render_outputs(
                model.propagate_in_video(state, start_frame_idx=0, reverse=False),
                "emitted",
            )
    check_decision_coverage(trace.frames)
    save_fixture(output_dir / f"sam31_video_reference_{precision}.npz", arrays)
    write_json(
        output_dir / f"sam31_video_reference_{precision}.json",
        {
            "concept": concept,
            "decisions": trace.frames,
            "reference_revision": REFERENCE_REVISION,
        },
    )


def check_decision_coverage(frames: list[dict[str, Any]]) -> None:
    """Fail when a clip does not exercise every decision type."""
    seen = {kind for record in frames for kind in DECISION_TYPES if record.get(kind)}
    if missing := set(DECISION_TYPES) - seen:
        raise ValueError(f"The clip never triggers: {sorted(missing)}.")


def require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The reference hard-codes CUDA; run this on a machine with a GPU."
        )
    return "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tokenizer", "operators", "tiny"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", required=True, type=Path)
        if name != "operators":
            command.add_argument("--reference-dir", required=True, type=Path)
    for name in ("image", "video"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--concept", required=True)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument(
            "--precision", choices=("float32", "bfloat16"), default="float32"
        )
        if name == "image":
            command.add_argument("--image", required=True, type=Path)
        else:
            command.add_argument("--frames-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "tokenizer":
        generate_tokenizer(args.reference_dir, args.output_dir)
        return
    device = require_cuda()
    if args.command == "operators":
        entries = generate_operators(args.output_dir, device)
    elif args.command == "tiny":
        entries = generate_tiny(args.reference_dir, args.output_dir, device)
    elif args.command == "image":
        entries = generate_image(
            args.checkpoint, args.image, args.concept, args.output_dir, args.precision
        )
    else:
        generate_video(
            args.checkpoint,
            args.frames_dir,
            args.concept,
            args.output_dir,
            args.precision,
        )
        return
    manifest = args.output_dir / "provenance_fragment.json"
    existing = json.loads(manifest.read_text()) if manifest.is_file() else {}
    write_json(manifest, existing | entries)


if __name__ == "__main__":
    main()
