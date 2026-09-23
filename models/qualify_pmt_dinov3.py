"""Generate intermediate DINOv3 Small tensors for frozen-decoder qualification.

Requires a pinned local Hugging Face snapshot and optional PyTorch/Transformers
reference environment. The installed Equimo tests only read the NumPy output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import numpy as np


SOURCE_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
SOURCE_CHECKPOINT_SHA256 = (
    "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
)
TAPS = (2, 5, 8, 11)


def generate(snapshot: Path, output: Path) -> None:
    """Save final-normalized intermediate tokens from a pinned model snapshot."""
    if (
        snapshot.name != SOURCE_REVISION
        or not (snapshot / "model.safetensors").is_file()
    ):
        raise ValueError(f"Expected the local DINOv3 snapshot {SOURCE_REVISION}.")
    checkpoint_sha256 = hashlib.sha256(
        (snapshot / "model.safetensors").read_bytes()
    ).hexdigest()
    if checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("DINOv3 snapshot weights disagree with the pinned checkpoint.")
    import torch
    import transformers

    torch.set_num_threads(1)
    model = transformers.AutoModel.from_pretrained(
        snapshot, local_files_only=True
    ).eval()
    image = np.random.default_rng(23).standard_normal((3, 32, 48)).astype(np.float32)
    with torch.no_grad():
        result = model(torch.from_numpy(image)[None], output_hidden_states=True)
        tensors = {"image": image}
        for tap in TAPS:
            tensors[f"tap_{tap}"] = model.norm(result.hidden_states[tap + 1])[0].numpy()
        np.testing.assert_allclose(
            tensors["tap_11"], result.last_hidden_state[0].numpy(), atol=1e-6
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **tensors)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "repository": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "revision": SOURCE_REVISION,
                "checkpoint_sha256": checkpoint_sha256,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "taps": TAPS,
                "image_shape": image.shape,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.snapshot, arguments.output)
