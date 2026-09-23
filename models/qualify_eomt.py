"""Generate compact predictions from a pinned EoMT author checkout.

This optional maintainer tool needs PyTorch and timm. Native Equimo inference
and the offline regression tests do not import either package.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import numpy as np


SOURCE_REVISION = "7bd19ddd621c5c6adedcd260458a34783cd4a45f"


def generate(author_dir: Path, output: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(author_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"Expected EoMT author revision {SOURCE_REVISION}, got {revision}."
        )

    import timm
    import torch

    sys.path.insert(0, str(author_dir))
    try:
        from models.eomt import EoMT
    finally:
        sys.path.pop(0)

    torch.set_num_threads(1)
    torch.manual_seed(19)
    backbone = timm.models.vision_transformer.VisionTransformer(
        img_size=16,
        patch_size=8,
        in_chans=3,
        num_classes=0,
        global_pool="",
        embed_dim=8,
        depth=2,
        num_heads=2,
        reg_tokens=1,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.register_buffer(
                "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    model = EoMT(Encoder(), num_classes=3, num_q=2, num_blocks=1).eval()
    captured_masks = []
    captured_tokens = []
    reference_attention_mask = model._attn_mask
    reference_predict = model._predict

    def capture_attention_mask(tokens, mask_logits, index):
        mask = reference_attention_mask(tokens, mask_logits, index)
        captured_masks.append(mask[0].detach().numpy())
        return mask

    model._attn_mask = capture_attention_mask

    def capture_prediction(tokens):
        captured_tokens.append(tokens[0].detach().numpy())
        return reference_predict(tokens)

    model._predict = capture_prediction
    image = torch.rand(1, 3, 16, 16)
    fixture = {
        f"state.{name}": value.detach().numpy()
        for name, value in model.state_dict().items()
    }
    fixture["image"] = image[0].numpy()
    with torch.no_grad():
        spatial_probe = torch.rand(1, 8, 2, 2)
        fixture["upscale_probe.input"] = spatial_probe[0].numpy()
        fixture["upscale_probe.conv1"] = (
            model.upscale[0].conv1(spatial_probe)[0].numpy()
        )
        fixture["upscale_probe.output"] = model.upscale(spatial_probe)[0].numpy()
        masked_masks, masked_classes = model(image)
        fixture["masked.attention.0"] = captured_masks[0]
        for index, tokens in enumerate(captured_tokens):
            fixture[f"masked.tokens.{index}"] = tokens
        captured_tokens.clear()
        model.masked_attn_enabled = False
        final_masks, final_classes = model(image)
        for index, tokens in enumerate(captured_tokens):
            fixture[f"unmasked.tokens.{index}"] = tokens
    for label, masks, classes in (
        ("masked", masked_masks, masked_classes),
        ("unmasked", final_masks, final_classes),
    ):
        for index, (mask, cls) in enumerate(zip(masks, classes, strict=True)):
            fixture[f"{label}.mask.{index}"] = mask[0].numpy()
            fixture[f"{label}.class.{index}"] = cls[0].numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **fixture)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "author_repository": "https://github.com/tue-mps/eomt",
                "author_revision": SOURCE_REVISION,
                "fixture": "tiny timm ViT with pinned author EoMT forward",
                "torch": torch.__version__,
                "timm": timm.__version__,
                "image_shape": [3, 16, 16],
                "query_count": 2,
                "query_blocks": 1,
                "classes": 3,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.author_dir, args.output)
