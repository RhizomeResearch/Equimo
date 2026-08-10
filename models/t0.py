"""Convert and numerically validate the official T0-alpha checkpoint."""
# Adapted from tfc-t0 and modified for JAX/Equinox; see NOTICE.

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == script_dir:
        sys.path.pop(0)

import argparse
import dataclasses
import hashlib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from equimo.conversion.utils import stringify_name
from equimo.serialization import save_model
from equimo.timeseries.models.t0 import T0, _T0_REGISTRY


IDENTIFIER = "t0_alpha"
MODEL_ID = "theforecastingcompany/t0-alpha"
REVISION = "f8727c2357e0d81f1d9f56fe3aaac43068b5fc72"
CHECKPOINT_SHA256 = "16c030d3fd70f06dc4238e9a8356e9b5a631d07f80f1bc76ba539991aed5897f"
REFERENCE_PATH = Path("t0_alpha_reference.npz")
DEFAULT_OUTPUT_DIR = Path("/tmp/equimo-references")
DEFAULT_SAVE_DIR = Path("~/.cache/equimo/t0").expanduser()
RTOL = 2e-4
ATOL = 2e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_name(name: str) -> str:
    name = name.replace("blocks.0.blocks.", "transformer.layers.")
    name = name.replace(
        "patch_encoder.projection.mlp.fc1.",
        "patch_encoder.projection.mlp.hidden_layer.",
    )
    name = name.replace(
        "patch_encoder.projection.mlp.fc2.",
        "patch_encoder.projection.mlp.output_layer.",
    )
    name = name.replace("decoder.mlp.fc1.", "decoder.mlp.hidden_layer.")
    name = name.replace("decoder.mlp.fc2.", "decoder.mlp.output_layer.")
    name = name.replace(".attn_norm.w", ".attention_block.norm.scale")
    name = name.replace(".attn.attention.qkv.", ".attention_block.attention.wQKV.")
    name = name.replace(".attn.attention.proj.", ".attention_block.attention.wO.")
    name = name.replace(
        ".attn.attention.q_norm.w", ".attention_block.attention.q_norm.scale"
    )
    name = name.replace(
        ".attn.attention.k_norm.w", ".attention_block.attention.k_norm.scale"
    )
    name = name.replace(".ffn_norm.w", ".norm.scale")
    name = name.replace(".ffn.w12.", ".mlp.0.")
    name = name.replace(".ffn.w3.", ".mlp.2.")
    return name.replace("out_norm.w", "transformer.out_norm.scale")


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _model_config(torch_model) -> dict:
    upstream = dataclasses.asdict(torch_model.config)
    if upstream.pop("scaler_use_arcsinh") is not True:
        raise ValueError(
            "T0-alpha must use the published arcsinh scaler configuration."
        )
    upstream["quantile_levels"] = tuple(upstream["quantile_levels"])

    base_cfg, variant_cfg = _T0_REGISTRY[IDENTIFIER]
    expected = base_cfg | variant_cfg
    if upstream != expected:
        raise ValueError(
            "Upstream T0-alpha configuration does not match Equimo's registered "
            f"architecture: expected {expected}, got {upstream}."
        )
    return upstream


def convert_torch_to_equimo(model: T0, torch_model) -> T0:
    state = torch_model.state_dict()
    dynamic, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(dynamic)

    converted = []
    used = set()
    for tree_path, leaf in flat:
        equimo_name = stringify_name(tree_path)
        torch_name = _checkpoint_name(equimo_name)
        if torch_name not in state:
            raise KeyError(
                f"No T0 checkpoint tensor maps to Equimo parameter {equimo_name!r}; "
                f"expected {torch_name!r}."
            )
        array = _to_numpy(state[torch_name])
        if array.dtype != np.float32:
            raise ValueError(
                f"{torch_name}: expected float32 checkpoint data, got {array.dtype}."
            )
        if tuple(array.shape) != tuple(leaf.shape):
            raise ValueError(
                f"{torch_name}: expected shape {tuple(leaf.shape)}, got {array.shape}."
            )
        converted_leaf = jnp.asarray(array)
        np.testing.assert_array_equal(np.asarray(converted_leaf), array)
        converted.append(converted_leaf)
        used.add(torch_name)

    quantile_name = "head.quantile_levels"
    if quantile_name not in state:
        raise KeyError(f"Missing T0 checkpoint tensor {quantile_name!r}.")
    np.testing.assert_array_equal(
        _to_numpy(state[quantile_name]),
        np.asarray(model.quantile_levels, dtype=np.float32),
    )
    used.add(quantile_name)

    leftover = set(state) - used
    if leftover:
        raise KeyError(f"Unconverted T0 checkpoint tensors: {sorted(leftover)}")

    converted_tree = jax.tree_util.tree_unflatten(treedef, converted)
    return eqx.nn.inference_mode(
        eqx.combine(converted_tree, static),
        value=True,
    )


def make_fixture(torch, TimeSeries, *, seed: int):
    rng = np.random.default_rng(seed)
    context = rng.standard_normal((2, 2, 64)).astype(np.float32)
    future = rng.standard_normal((2, 1, 97)).astype(np.float32)

    context[0, 0, 5] = np.nan
    context[0, 1, 31:34] = np.nan
    context[1, 0, 63] = np.nan
    future[1, 0, 70] = np.nan

    series = TimeSeries.from_array(
        torch.from_numpy(context),
        torch.from_numpy(future),
    )
    arrays = {
        "values": _to_numpy(series.variates),
        "mask": _to_numpy(series.mask),
        "group_ids": _to_numpy(series.group_ids),
        "variate_type": _to_numpy(series.variate_type),
    }
    return series, arrays


def _assert_exact(name: str, jax_value, torch_value) -> None:
    np.testing.assert_array_equal(_to_numpy(jax_value), _to_numpy(torch_value))
    print(f"{name}: exact")


def _assert_close(name: str, jax_value, torch_value) -> None:
    jax_array = _to_numpy(jax_value)
    torch_array = _to_numpy(torch_value)
    if jax_array.shape != torch_array.shape:
        raise ValueError(
            f"{name}: Equimo shape {jax_array.shape} != upstream {torch_array.shape}."
        )
    delta = np.abs(jax_array.astype(np.float64) - torch_array.astype(np.float64))
    mae = float(np.mean(delta))
    max_error = float(np.max(delta))
    print(f"{name}: mae={mae:.3e} max={max_error:.3e}")
    np.testing.assert_allclose(
        jax_array,
        torch_array,
        rtol=RTOL,
        atol=ATOL,
        err_msg=name,
    )


def validate_stages(model: T0, torch_model, series, arrays: dict, *, seed: int):
    import torch

    from t0.mask import compute_patch_attention_mask

    key = jax.random.PRNGKey(seed)
    values, mask, group_ids, variate_type = (
        jnp.asarray(arrays[name])
        for name in ("values", "mask", "group_ids", "variate_type")
    )
    jax_patches = model._patch(values, mask, group_ids, variate_type)

    padded = torch_model.patcher.pad(series)
    torch_value_patches = torch_model.patcher.patch(padded.variates)
    torch_mask_patches = torch_model.patcher.patch(padded.mask)
    torch_type_patches = torch_model.patcher.patch(padded.variate_type)
    torch_group_patches = torch_model.patcher.patch(padded.group_ids)

    for name, jax_value, torch_value in zip(
        ("patched values", "patched mask", "patched groups", "patched types"),
        jax_patches,
        (
            torch_value_patches,
            torch_mask_patches,
            torch_group_patches,
            torch_type_patches,
        ),
    ):
        _assert_exact(name, jax_value, torch_value)

    jax_x = model.patch_encoder(
        jax_patches[0],
        jax_patches[1],
        jax_patches[3],
        key=key,
        inference=True,
    )
    with torch.inference_mode():
        torch_x = torch_model.patch_encoder(
            torch_value_patches,
            torch_mask_patches,
            torch_type_patches,
        )
    _assert_close("patch encoder", jax_x, torch_x)

    jax_time_mask, jax_group_mask = model._masks(
        jax_patches[2],
        jax_patches[3],
        jax_patches[1],
    )
    torch_patch_groups = torch_group_patches[:, :, 0]
    torch_patch_types = torch_type_patches[:, :, 0]
    attendable = compute_patch_attention_mask(torch_mask_patches)
    padding_mask = ~attendable if not bool(attendable.all()) else None
    mask_builder = torch_model.transformer.mask_builder
    torch_time_mask = mask_builder.build_time_mask(
        torch_patch_groups,
        torch_patch_types,
        padding_mask,
    )
    torch_group_mask = mask_builder.expand_group_mask(
        mask_builder.build_group_mask(torch_patch_groups)
    )
    _assert_exact("time mask", jax_time_mask, torch_time_mask.squeeze(1))
    _assert_exact("group mask", jax_group_mask, torch_group_mask.squeeze(1))

    _, jax_blocks = model.blocks[0].intermediate_features(
        jax_x,
        time_attn_mask=jax_time_mask,
        group_attn_mask=jax_group_mask,
        key=key,
        inference=True,
        indices=tuple(range(model.num_layers)),
    )
    torch_blocks = []
    with torch.inference_mode():
        for layer in torch_model.transformer.layers:
            torch_x = layer(
                torch_x,
                time_attn_mask=torch_time_mask,
                group_attn_mask=torch_group_mask,
            )
            torch_blocks.append(torch_x)

    if len(jax_blocks) != len(torch_blocks):
        raise ValueError(
            f"Expected {len(torch_blocks)} Equimo block outputs, got {len(jax_blocks)}."
        )
    for index, (jax_block, torch_block) in enumerate(
        zip(jax_blocks, torch_blocks, strict=True)
    ):
        attention_type = model.blocks[0].blocks[index].attention_type
        _assert_close(
            f"transformer block {index:02d} ({attention_type})", jax_block, torch_block
        )

    jax_norm = model.out_norm(jax_blocks[-1])
    with torch.inference_mode():
        torch_norm = torch_model.transformer.out_norm(torch_blocks[-1])
    _assert_close("transformer output norm", jax_norm, torch_norm)

    jax_decoded = model.decoder(jax_norm, key=key, inference=True).reshape(
        *jax_norm.shape[:-1],
        model.patch_size,
        len(model.quantile_levels),
    )
    with torch.inference_mode():
        torch_decoded = torch_model.decoder(torch_norm).unflatten(
            -1,
            (torch_model.patch_size, torch_model.head.n_quantiles),
        )
        torch_quantiles = torch_model.head(torch_decoded)
        torch_output = torch_model(series)
    _assert_close("decoder", jax_decoded, torch_decoded)

    first = jax_decoded[..., :1]
    jax_quantiles = jnp.concatenate(
        (
            first,
            first + jnp.cumsum(jax.nn.softplus(jax_decoded[..., 1:]), axis=-1),
        ),
        axis=-1,
    )
    _assert_close("quantile head", jax_quantiles, torch_quantiles)
    _assert_close(
        "public backbone forward",
        model(
            values,
            mask,
            group_ids,
            variate_type,
            key=key,
            inference=True,
        ),
        torch_output,
    )
    return torch_output


def save_reference(path: Path, arrays: dict, output) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, output=_to_numpy(output))
    print(f"Saved T0-alpha reference to {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert and validate the official T0-alpha checkpoint."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved work without importing Torch or accessing the Hub.",
    )
    args = parser.parse_args(argv)
    args.save_dir = args.save_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    reference_path = args.output_dir / REFERENCE_PATH
    archive_path = args.save_dir / f"{IDENTIFIER}.tar.lz4"
    print(
        f"{IDENTIFIER}: model={args.model_id} revision={args.revision} "
        f"checkpoint_sha256={args.checkpoint_sha256} seed={args.seed} "
        f"reference={reference_path} archive={archive_path}"
    )
    if args.dry_run:
        return

    try:
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
        from t0 import T0Forecaster
        from t0.data import TimeSeries
    except ImportError as exc:
        raise ImportError(
            "T0 conversion requires the `reference` dependency group; run "
            "`uv sync --group dev --group reference`."
        ) from exc

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                allow_patterns=("config.json", "model.safetensors"),
            )
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "T0-alpha is gated. Accept its Hugging Face terms and authenticate "
            "with `hf auth login` or HF_TOKEN before running this script."
        ) from exc

    checkpoint_path = snapshot / "model.safetensors"
    checkpoint_sha256 = _sha256(checkpoint_path)
    print(f"Source checkpoint SHA-256: {checkpoint_sha256}")
    if (
        args.checkpoint_sha256 is not None
        and checkpoint_sha256 != args.checkpoint_sha256
    ):
        raise ValueError(
            "T0-alpha checkpoint checksum mismatch: "
            f"expected {args.checkpoint_sha256}, got {checkpoint_sha256}."
        )

    torch_model = T0Forecaster.from_pretrained(str(snapshot)).cpu().eval()
    model_config = _model_config(torch_model)
    series, arrays = make_fixture(torch, TimeSeries, seed=args.seed)
    with torch.inference_mode():
        official_output = torch_model(series)

    if args.references_only:
        save_reference(reference_path, arrays, official_output)
        return

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = T0(**model_config, key=jax.random.PRNGKey(args.seed))
        model = convert_torch_to_equimo(model, torch_model)
        official_output = validate_stages(
            model,
            torch_model,
            series,
            arrays,
            seed=args.seed,
        )
        saved_path = save_model(
            args.save_dir / IDENTIFIER,
            model,
            model_config,
            torch_hub_cfg={
                "hf_model": args.model_id,
                "revision": args.revision,
                "checkpoint_sha256": checkpoint_sha256,
            },
            compression=True,
        )

    print(f"Native archive SHA-256: {_sha256(saved_path)}")
    save_reference(reference_path, arrays, official_output)


if __name__ == "__main__":
    main()
