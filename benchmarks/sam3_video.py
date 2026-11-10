"""Measure SAM 3.1 video tracking memory and latency on one GPU.

Run from the devenv shell with the CUDA wheels, for example:

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 \\
    uv run --with "jax[cuda12]==0.11.2" python benchmarks/sam3_video.py \\
    --archive-dir /path/to/archives --fixture /path/to/sam31_video_reference.tar.lz4

``--archive-dir`` holds ``sam31_multiplex.tar.lz4`` and ``sam31_text.tar.lz4``.
The clip and concept come from the hosted video fixture. Buckets are padded to
``--buckets``: the state and attention arrays have the shapes of a clip with that
many used buckets, so peak memory is representative, and time is a lower bound
because empty buckets attend to no keys.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from equimo.core.layers.attention import with_attention_implementation
from equimo.language import sam31_text
from equimo.serialization import load_weights
from equimo.vision.models.sam3 import (
    ConceptPrompt,
    pad_tracker_buckets,
    sam31_multiplex,
)
from equimo.vision.models.sam3.video import Sam3VideoTracker

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

from cases.sam31_video_fixture import load_fixture  # noqa: E402

WARMUP_FRAMES = 3


def _bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def _peak_gigabytes() -> float:
    stats = jax.local_devices()[0].memory_stats()
    return stats["peak_bytes_in_use"] / 1024**3


def measure(vision, text, reference, buckets: int, frames: int) -> dict:
    token_ids = jnp.asarray(reference["inputs.text.token_ids"][0], jnp.int32)
    padding_mask = jnp.asarray(reference["inputs.text.padding_mask"][0])
    pixels = jnp.asarray(reference["inputs.frames"][:frames], jnp.float32) / 255.0
    clip = jnp.moveaxis((pixels - 0.5) / 0.5, -1, 1).astype(jnp.bfloat16)
    tracker = Sam3VideoTracker(
        vision,
        num_frames=clip.shape[0],
        source_size=tuple(int(side) for side in reference["inputs.source_size"]),
    )
    tracker.set_prompt(
        ConceptPrompt.create(
            text_features=_bfloat16(text.features(token_ids, padding_mask)),
            text_padding_mask=padding_mask,
            dim=256,
        )
    )
    if buckets > 1:
        tracker.policy._grow(buckets)
        tracker._state = pad_tracker_buckets(tracker._state, buckets)
    times = []
    for index, frame in enumerate(clip):
        started = time.perf_counter()
        tracker.push_frame(frame)
        times.append(time.perf_counter() - started)
        jax.block_until_ready(tracker._state)
        if index == WARMUP_FRAMES:
            warm = list(times)
    return {
        "buckets": buckets,
        "frames": clip.shape[0],
        "peak_gigabytes": _peak_gigabytes(),
        "seconds_per_frame_median": float(np.median(times[WARMUP_FRAMES:])),
        "compile_seconds_first_frames": float(sum(warm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--implementation", choices=("xla", "cudnn"), default="xla")
    parser.add_argument("--buckets", default="1,2,4")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    vision = load_weights(
        sam31_multiplex(), path=args.archive_dir / "sam31_multiplex.tar.lz4"
    )
    vision = _bfloat16(vision)
    if args.implementation == "cudnn":
        vision = with_attention_implementation(vision, "cudnn")
    text = _bfloat16(
        load_weights(sam31_text(), path=args.archive_dir / "sam31_text.tar.lz4")
    )
    reference = load_fixture(args.fixture)
    for buckets in (int(item) for item in args.buckets.split(",")):
        record = {
            "implementation": args.implementation,
            **measure(vision, text, reference, buckets, args.frames),
        }
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
