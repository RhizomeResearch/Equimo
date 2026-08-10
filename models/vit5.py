"""Generate the small upstream ViT-5 rotary regression fixture."""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == script_dir:
        sys.path.pop(0)

import argparse
from pathlib import Path

import numpy as np

REVISION = "b7b0796062f4b51ae00523cb77c7117ae2dff536"


def generate_reference(output_dir: Path) -> Path:
    """Reproduce ``VisionRotaryEmbedding`` from the pinned official source."""
    height, width, dim = 2, 3, 8
    pt_seq_len, theta = 14, 10_000.0
    rng = np.random.default_rng(42)
    x = rng.standard_normal((height * width, 2, 2 * dim)).astype(np.float32)

    frequencies = 1.0 / (
        theta ** (np.arange(0, dim, 2, dtype=np.float32)[: dim // 2] / dim)
    )
    positions_h = np.arange(height, dtype=np.float32) / height * pt_seq_len
    positions_w = np.arange(width, dtype=np.float32) / width * pt_seq_len
    frequencies_h = np.repeat(np.outer(positions_h, frequencies), 2, axis=-1)
    frequencies_w = np.repeat(np.outer(positions_w, frequencies), 2, axis=-1)
    angles_h = np.broadcast_to(
        frequencies_h[:, None], (height, width, frequencies_h.shape[-1])
    )
    angles_w = np.broadcast_to(
        frequencies_w[None], (height, width, frequencies_w.shape[-1])
    )
    angles = np.concatenate((angles_h, angles_w), axis=-1).reshape(height * width, -1)
    sin = np.sin(angles).astype(np.float32)
    cos = np.cos(angles).astype(np.float32)

    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = np.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)
    output = x * cos[:, None] + rotated * sin[:, None]

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vit5_rope_reference.npz"
    np.savez(
        path,
        x=x,
        sin=sin,
        cos=cos,
        output=output,
        height=np.asarray(height),
        width=np.asarray(width),
        dim=np.asarray(dim),
        pt_seq_len=np.asarray(pt_seq_len),
        theta=np.asarray(theta),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate_reference(args.output_dir))


if __name__ == "__main__":
    main()
