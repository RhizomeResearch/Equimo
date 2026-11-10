"""Measure SAM 3.1 image inference and fine-tuning on one GPU.

Run from the devenv shell with the CUDA wheels, for example:

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 \\
    uv run --with "jax[cuda12]==0.11.2" python benchmarks/sam3_image.py \\
    lora --archive-dir /path/to/archives

``--archive-dir`` holds ``sam31_multiplex.tar.lz4`` and ``sam31_text.tar.lz4``.
The image and concept come from the reference fixture, so no other input is needed.
"""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import equimo.finetune as eqft
from equimo.core.layers.attention import with_attention_implementation
from equimo.language import sam31_text
from equimo.serialization import load_weights
from equimo.vision.models.sam3 import ConceptPrompt, Sam3, sam31_multiplex

FIXTURE = Path(__file__).parents[1] / "tests" / "data" / "sam31_image_reference.npz"
TRAINING_STEPS = 50
LEARNING_RATE = 1e-4
LORA_RANK = 8


def _bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def _peak_gigabytes() -> float:
    stats = jax.local_devices()[0].memory_stats()
    return stats["peak_bytes_in_use"] / 1024**3


def _load_inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        pixels = jnp.asarray(fixture["inputs.image"], jnp.float32) / 255.0
        tokens = jnp.asarray(fixture["inputs.text.token_ids"][0], jnp.int32)
        padding = jnp.asarray(fixture["inputs.text.padding_mask"][0])
    image = jnp.moveaxis((pixels - 0.5) / 0.5, -1, 0).astype(jnp.bfloat16)
    return image, tokens, padding


def _detect(
    model: Sam3, image: jax.Array, prompt: ConceptPrompt, checkpoint_trunk: bool = False
):
    return model.detect(
        model.encode_image(image, checkpoint_trunk=checkpoint_trunk), prompt
    )


def _loss(model: Sam3, image: jax.Array, prompt: ConceptPrompt) -> jax.Array:
    """Pull the masks of the 16 highest-scoring queries towards uncertainty."""
    output = _detect(model, image, prompt, checkpoint_trunk=True)
    logits = output.mask_logits[jnp.argsort(-output.logits)[:16]]
    log_positive = jax.nn.log_sigmoid(logits)
    log_negative = jax.nn.log_sigmoid(-logits)
    return -0.5 * jnp.mean(log_positive + log_negative)


def _measure_inference(model: Sam3, image: jax.Array, prompt: ConceptPrompt) -> dict:
    detect = eqx.filter_jit(_detect)
    started = time.perf_counter()
    jax.block_until_ready(detect(model, image, prompt))
    compile_seconds = time.perf_counter() - started
    times = []
    for _ in range(5):
        started = time.perf_counter()
        jax.block_until_ready(detect(model, image, prompt))
        times.append(time.perf_counter() - started)
    return {
        "compile_seconds": compile_seconds,
        "latency_ms_median": 1000 * float(np.median(times)),
        "peak_gigabytes": _peak_gigabytes(),
    }


def _train(
    model: Sam3, image: jax.Array, prompt: ConceptPrompt, spec: eqft.TrainableSpec
) -> dict:
    plan = eqft.prepare_finetune(model, trainable=spec)
    trainable = plan.trainable

    def loss_of(trainable: eqx.Module) -> jax.Array:
        return _loss(plan.combine(trainable), image, prompt)

    @eqx.filter_jit
    def step(trainable: eqx.Module) -> tuple[eqx.Module, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_of)(trainable)
        updated = jax.tree.map(
            lambda value, gradient: (
                value - LEARNING_RATE * gradient.astype(value.dtype)
                if eqx.is_inexact_array(value)
                else value
            ),
            trainable,
            grads,
        )
        return updated, loss

    losses = []
    started = time.perf_counter()
    for _ in range(TRAINING_STEPS):
        trainable, loss = step(trainable)
        losses.append(float(loss))
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "finite": bool(all(np.isfinite(losses))),
        "seconds_per_step": (time.perf_counter() - started) / TRAINING_STEPS,
        "trainable_parameters": plan.report.trainable_params,
        "peak_gigabytes": _peak_gigabytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("inference", "head", "lora"))
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument(
        "--implementation", choices=("xla", "cudnn"), default="xla", dest="attention"
    )
    args = parser.parse_args()

    # Load and cast on the CPU so the peak reports this workload, not the float32 archive.
    with jax.default_device(jax.devices("cpu")[0]):
        model = _bfloat16(
            load_weights(
                sam31_multiplex(), path=args.archive_dir / "sam31_multiplex.tar.lz4"
            )
        )
        text = load_weights(sam31_text(), path=args.archive_dir / "sam31_text.tar.lz4")
        image, tokens, padding = _load_inputs()
        text_features = eqx.filter_jit(text.features)(tokens, padding)
        prompt = ConceptPrompt.create(
            text_features=text_features.astype(jnp.bfloat16),
            text_padding_mask=padding,
            dim=256,
        )
    del text
    gpu = jax.devices("gpu")[0]
    model, image, prompt = jax.device_put((model, image, prompt), gpu)
    if args.attention == "cudnn":
        model = with_attention_implementation(model, "cudnn")

    if args.mode == "inference":
        result = _measure_inference(model, image, prompt)
    elif args.mode == "head":
        result = _train(model, image, prompt, eqft.TrainableSpec(mode="head"))
    else:
        model = eqft.apply_lora(
            model,
            eqft.LoRAConfig(
                rank=LORA_RANK,
                target=eqft.TargetSpec(
                    tags_any=(
                        "attention.q",
                        "attention.k",
                        "attention.v",
                        "attention.proj",
                    ),
                    exclude=("interactive", "tracker"),
                ),
            ),
            key=jr.PRNGKey(0),
        )
        result = _train(
            model,
            image,
            prompt,
            eqft.TrainableSpec(mode="peft", method_name="lora", train_head=False),
        )
    print(json.dumps({"mode": args.mode, **result}, indent=2))


if __name__ == "__main__":
    main()
