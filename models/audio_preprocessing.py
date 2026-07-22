"""Generate pinned raw-waveform AST preprocessing references."""

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)

import numpy as np

from equimo.audio import get_ast_preprocessing_spec

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
VARIANTS = (
    "ast_base_patch16_audioset_10_10_0_4593",
    "ast_base_patch16_speechcommands_v2_10_10_0_9812",
)


def deterministic_waveform() -> np.ndarray:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    return (
        0.35 * np.sin(2 * np.pi * 440 * time) + 0.15 * np.sin(2 * np.pi * 997 * time)
    ).astype(np.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate AST preprocessing and end-to-end parity references."
    )
    parser.add_argument("variants", nargs="*", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.variants = args.variants or list(VARIANTS)
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    for variant in args.variants:
        spec = get_ast_preprocessing_spec(variant)
        path = args.output_dir / f"{variant}_raw_audio_reference.npz"
        print(
            f"{variant}: model={spec.checkpoint} revision={spec.checkpoint_revision} "
            f"output={path}"
        )
    if args.dry_run:
        return

    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    waveform = deterministic_waveform()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        spec = get_ast_preprocessing_spec(variant)
        extractor = ASTFeatureExtractor.from_pretrained(
            spec.checkpoint, revision=spec.checkpoint_revision
        )
        model = ASTForAudioClassification.from_pretrained(
            spec.checkpoint, revision=spec.checkpoint_revision
        ).eval()
        input_values = extractor(
            waveform, sampling_rate=spec.sample_rate, return_tensors="pt"
        ).input_values
        with torch.no_grad():
            features = model.audio_spectrogram_transformer(
                input_values=input_values
            ).last_hidden_state
            logits = model(input_values=input_values).logits
        path = args.output_dir / f"{variant}_raw_audio_reference.npz"
        np.savez(
            path,
            input_values=input_values[0].numpy().astype(np.float32),
            cls_token=features[0, 0].numpy().astype(np.float32),
            dist_token=features[0, 1].numpy().astype(np.float32),
            logits=logits[0].numpy().astype(np.float32),
        )
        print(f"Saved AST raw-audio reference to {path}")


if __name__ == "__main__":
    main()
