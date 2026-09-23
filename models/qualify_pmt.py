"""Generate a compact PMD parity fixture from the pinned author implementation.

Requires an explicit local checkout of the author repository and the optional
PyTorch/Transformers reference environment. Native Equimo tests read the
generated NumPy archive without importing either framework.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import numpy as np


SOURCE_REVISION = "0e803722aa5737a242b383dec1b90c2c66b86baa"


def generate(author_dir: Path, output: Path) -> None:
    """Save random initialized reference weights and execution intermediates."""
    revision = subprocess.check_output(
        ["git", "-C", str(author_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"Expected PMT author revision {SOURCE_REVISION}, got {revision}."
        )
    import torch

    sys.path.insert(0, str(author_dir / "image"))
    try:
        from models.layers import DINOv3ViTRopePositionEmbedding
        from models.pmd import PlainMaskDecoder
    finally:
        sys.path.pop(0)

    torch.set_num_threads(1)
    torch.manual_seed(19)
    decoder = PlainMaskDecoder(
        embed_dim=8,
        hidden_dim=8,
        num_prefix_tokens=2,
        grid_size=(2, 3),
        patch_size=(8, 8),
        num_classes=3,
        num_q=2,
        num_blocks=2,
        masked_attn_enabled=True,
        interaction_indices=[0, 1],
        num_heads=2,
    ).eval()
    norms = torch.nn.ModuleList([torch.nn.SyncBatchNorm(8) for _ in range(2)]).eval()
    with torch.no_grad():
        for index, norm in enumerate(norms):
            norm.weight.copy_(torch.linspace(0.8, 1.2, 8) + 0.1 * index)
            norm.bias.copy_(torch.linspace(-0.2, 0.2, 8) + 0.1 * index)
            norm.running_mean.copy_(torch.linspace(-0.3, 0.3, 8))
            norm.running_var.copy_(torch.linspace(0.8, 1.3, 8))
            norm.num_batches_tracked.fill_(3)

    levels = [torch.randn(1, 8, 8) for _ in range(2)]
    rope_module = DINOv3ViTRopePositionEmbedding(
        hidden_size=8, num_attention_heads=2, image_size=16, patch_size=8
    ).eval()
    reference: dict[str, np.ndarray] = {}
    for name, value in decoder.state_dict().items():
        reference[f"decoder.{name}"] = value.detach().numpy()
    for index, norm in enumerate(norms):
        for name, value in norm.state_dict().items():
            reference[f"norm.{index}.{name}"] = value.detach().numpy()
    for index, level in enumerate(levels):
        reference[f"level.{index}"] = level[0].numpy()

    with torch.no_grad():
        train_norm = torch.nn.SyncBatchNorm(8)
        train_norm.load_state_dict(norms[0].state_dict())
        train_norm.train()
        training_batch = torch.randn(2, 8, 8)
        reference["batchnorm_train.input"] = training_batch.numpy()
        reference["batchnorm_train.output"] = train_norm(training_batch).numpy()
        reference["batchnorm_train.running_mean"] = train_norm.running_mean.numpy()
        reference["batchnorm_train.running_var"] = train_norm.running_var.numpy()
        reference["batchnorm_train.num_batches_tracked"] = (
            train_norm.num_batches_tracked.numpy()
        )

        rope = rope_module(torch.zeros(1, 3, 16, 24))
        reference["rope.cos"] = rope[0].numpy()
        reference["rope.sin"] = rope[1].numpy()
        normalized = [
            norm(level.permute(0, 2, 1)).permute(0, 2, 1)
            for norm, level in zip(norms, levels, strict=True)
        ]
        for index, value in enumerate(normalized):
            reference[f"normalized.{index}"] = value[0].numpy()
        projected = decoder._lateral_projections_forward(
            [value.clone() for value in normalized]
        )
        for index, value in enumerate(projected):
            reference[f"lateral.{index}"] = value[0].numpy()
        reference["fused"] = torch.stack(projected).sum(dim=0)[0].numpy()

        original_forward = decoder.block_forward
        traces: list[np.ndarray] = []

        def capture_block(tokens, block, mask, pe):
            value = original_forward(tokens, block, mask, pe)
            traces.append(value[0].numpy())
            return value

        decoder.block_forward = capture_block
        for mode in ("masked", "unmasked"):
            traces.clear()
            decoder.masked_attn_enabled = mode == "masked"
            masks, classes = decoder([value.clone() for value in normalized], rope)
            for index, value in enumerate(traces):
                reference[f"{mode}.tokens.{index}"] = value
            for index, (mask, labels) in enumerate(zip(masks, classes, strict=True)):
                reference[f"{mode}.mask.{index}"] = mask[0].numpy()
                reference[f"{mode}.class.{index}"] = labels[0].numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **reference)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "author_repository": "https://github.com/tue-mps/pmt",
                "author_revision": SOURCE_REVISION,
                "fixture": "random PMD with two frozen-feature levels and BatchNorm",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "image_shape": [3, 16, 24],
                "token_shape": [8, 8],
                "query_count": 2,
                "decoder_blocks": 2,
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
    arguments = parser.parse_args()
    generate(arguments.author_dir, arguments.output)
