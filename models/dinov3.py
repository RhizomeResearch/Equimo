# ruff: noqa: E402
import gc
import os
import re
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import GetAttrKey, SequenceKey

import equimo.vision.models as em
from equimo.conversion.utils import (
    convert_torch_to_equinox,
    expand_torch_tensor,
    stringify_name,
)
from equimo.serialization import load_weights, save_model

DIR = Path("~/.cache/torch/hub/dinov3").expanduser()
DINOV3_LOCAL_ROPE_CFG = {
    "strategy": "period",
    "base": 100.0,
    "normalize_coords": "separate",
    "rescale_coords": 2.0,
    "dtype": "float32",
    "periods_dtype": "float32",
}


def compare(j, t) -> float:
    j = np.array(j)
    if hasattr(t, "detach"):
        t = t.squeeze().detach().cpu().numpy()
    return float(np.mean(np.abs(j - t)))


def _expected_rope_freqs(cfg: dict) -> jax.Array:
    num_heads = cfg["num_heads"]
    if isinstance(num_heads, list):
        num_heads = num_heads[0]
    D_head = cfg["dim"] // num_heads
    return DINOV3_LOCAL_ROPE_CFG["base"] ** (
        2.0 * jnp.arange(D_head // 4, dtype=jnp.float32) / float(D_head // 2)
    )


def _assert_rope_freqs(model, cfg: dict, name: str) -> None:
    rope = model.local_pos_embed.patch_rope
    expected = _expected_rope_freqs(cfg)
    if rope.freqs.dtype != jnp.float32:
        raise AssertionError(f"{name}: RoPE freqs dtype is {rope.freqs.dtype}")
    if not np.allclose(np.asarray(rope.freqs), np.asarray(expected), rtol=0, atol=0):
        err = float(np.max(np.abs(np.asarray(rope.freqs - expected))))
        raise AssertionError(f"{name}: RoPE freqs differ from fp32 periods: {err}")


def _fix_torch_rope_periods(torch_model, cfg: dict, name: str) -> None:
    torch_model.rope_embed._init_weights()
    periods = torch_model.rope_embed.periods.detach().cpu().numpy()
    expected = np.asarray(_expected_rope_freqs(cfg))
    if not np.allclose(periods, expected, rtol=0, atol=0):
        err = float(np.max(np.abs(periods - expected)))
        raise AssertionError(
            f"{name}: Torch RoPE periods differ from fp32 periods: {err}"
        )


def _tree_get(tree, path: tuple):
    leaf = tree
    for key in path:
        if isinstance(key, GetAttrKey):
            leaf = getattr(leaf, key.name)
        elif isinstance(key, SequenceKey):
            leaf = leaf[key.idx]
        else:
            raise TypeError(f"Unsupported tree path key: {key!r}")
    return leaf


def _convert_param_path(path: tuple, replace_cfg: dict[str, str]) -> str:
    param_path = stringify_name(path)
    param_path = re.sub(r"\.scale|\.kernel", ".weight", param_path)
    for old, new in replace_cfg.items():
        param_path = param_path.replace(old, new)
    return param_path


def _stream_torch_state_into_equinox(
    model,
    checkpoint_path: str,
    *,
    replace_cfg: dict[str, str],
    expand_cfg: dict[str, list],
    squeeze_cfg: dict[str, int | None],
    torch_whitelist: list[str],
    jax_whitelist: list[str],
    torch,
):
    state = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    used: set[str] = set()
    params = eqx.filter(model, eqx.is_array)
    flat, _ = jax.tree_util.tree_flatten_with_path(params)

    for path, leaf in flat:
        param_path = _convert_param_path(path, replace_cfg)
        if param_path not in state:
            if param_path not in jax_whitelist:
                raise AttributeError(
                    f"{param_path} ({leaf.shape}) not found in PyTorch checkpoint."
                )
            continue

        torch_param = state[param_path]
        if param_path in expand_cfg:
            torch_param = expand_torch_tensor(torch_param, *expand_cfg[param_path])
        if param_path in squeeze_cfg:
            torch_param = torch.squeeze(torch_param, dim=squeeze_cfg[param_path])
        if leaf.shape != torch_param.shape:
            raise ValueError(
                f"`{param_path}`: expected shape ({leaf.shape}) does not match "
                f"its PyTorch checkpoint ({torch_param.shape})."
            )

        array = np.asarray(torch_param.detach().cpu().numpy())
        new_leaf = jnp.asarray(array)
        new_leaf.block_until_ready()
        model = eqx.tree_at(
            lambda tree, path=path: _tree_get(tree, path),
            model,
            new_leaf,
        )
        used.add(param_path)
        del torch_param, array, new_leaf
        gc.collect()

    leftovers = {
        name
        for name in set(state) - used - {"rope_embed.periods"}
        if not name.endswith(".attn.qkv.bias_mask")
    }
    leftovers -= set(torch_whitelist)
    if leftovers:
        raise AttributeError(
            f"PyTorch checkpoint contains unconverted parameters: {sorted(leftovers)}"
        )
    del state
    gc.collect()
    return eqx.nn.inference_mode(model, True)


def _torch_x_prenorm(
    *,
    torch,
    torch_hub_cfg: dict,
    cfg: dict,
    name: str,
    arr: np.ndarray,
) -> np.ndarray:
    torch_model = torch.hub.load(
        torch_hub_cfg["repo_or_dir"],
        torch_hub_cfg["model"],
        source=torch_hub_cfg["source"],
        pretrained=False,
        weights=torch_hub_cfg["weights"],
    )
    state = torch.load(torch_hub_cfg["weights"], map_location="cpu", mmap=True)
    torch_model.load_state_dict(state, strict=True)
    del state
    _fix_torch_rope_periods(torch_model, cfg, name)
    torch_model.eval()

    torch_arr = torch.tensor(arr).unsqueeze(0).float()
    with torch.no_grad():
        torch_features = (
            torch_model.forward_features(torch_arr)["x_prenorm"]
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
    del torch_model, torch_arr
    gc.collect()
    return torch_features


weights = {
    # LVD
    "dinov3_vits16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            )
        ).expanduser()
    ),
    "dinov3_vits16plus_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
            )
        ).expanduser()
    ),
    "dinov3_vitb16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            )
        ).expanduser()
    ),
    "dinov3_vitl16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            )
        ).expanduser()
    ),
    "dinov3_vith16plus_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            )
        ).expanduser()
    ),
    "dinov3_vit7b16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
            )
        ).expanduser()
    ),
    # SAT
    "dinov3_vitl16_pretrain_sat493m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            )
        ).expanduser()
    ),
    "dinov3_vit7b16_pretrain_sat493m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
            )
        ).expanduser()
    ),
}

configs = {
    "dinov3_vits16_pretrain_lvd1689m": {
        "dim": 384,
        "num_heads": 6,
        "depths": [12],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
    },
    "dinov3_vits16plus_pretrain_lvd1689m": {
        "dim": 384,
        "num_heads": 6,
        "depths": [12],
        "reg_tokens": 4,
        "mlp_ratio": 6.0,
        "ffn_layer": "swiglu",
    },
    "dinov3_vitb16_pretrain_lvd1689m": {
        "dim": 768,
        "num_heads": 12,
        "depths": [12],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
    },
    "dinov3_vitl16_pretrain_lvd1689m": {
        "dim": 1024,
        "num_heads": 16,
        "depths": [24],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
    },
    "dinov3_vith16plus_pretrain_lvd1689m": {
        "dim": 1280,
        "num_heads": 20,
        "depths": [32],
        "reg_tokens": 4,
        "mlp_ratio": 6.0,
        "ffn_layer": "swiglu",
    },
    "dinov3_vit7b16_pretrain_lvd1689m": {
        "dim": 4096,
        "num_heads": 32,
        "depths": [40],
        "reg_tokens": 4,
        "mlp_ratio": 3.0,
        "untie_global_and_local_cls_norm": True,
        "ffn_layer": "swiglu",
        "ffn_kwargs": {"align_to": 64},
        "qkv_bias": False,
    },
    "dinov3_vitl16_pretrain_sat493m": {
        "dim": 1024,
        "num_heads": 16,
        "depths": [24],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
        "untie_global_and_local_cls_norm": True,
    },
    "dinov3_vit7b16_pretrain_sat493m": {
        "dim": 4096,
        "num_heads": 32,
        "depths": [40],
        "reg_tokens": 4,
        "mlp_ratio": 3.0,
        "untie_global_and_local_cls_norm": True,
        "ffn_layer": "swiglu",
        "ffn_kwargs": {"align_to": 64},
        "qkv_bias": False,
    },
}

citr = iter(configs.items())
name, config = next(citr)


def main():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("`torch` not available") from exc

    key = jax.random.PRNGKey(42)
    dinov3_config = {
        "img_size": 224,
        "in_channels": 3,
        "patch_size": 16,
        "num_classes": 0,
        "use_mask_token": True,
        "use_global_pos_embed": False,
        "use_local_pos_embed": True,
        "reg_tokens": 4,
        "init_values": 1e-5,
        "eps": 1e-5,
        "dynamic_img_size": True,
        "act_layer": "exactgelu",
        "local_pos_embed_config_patch": DINOV3_LOCAL_ROPE_CFG,
    }
    rng = np.random.default_rng(42)
    missing_weights = [
        name for name, path in weights.items() if not Path(path).exists()
    ]
    if missing_weights:
        missing = ", ".join(missing_weights)
        raise FileNotFoundError(f"Missing DINOv3 Torch checkpoints: {missing}")

    resume = os.environ.get("EQUIMO_DINOV3_RESUME") == "1"

    for name, config in configs.items():
        print(f"Converting {name}...")

        cfg = dinov3_config | config
        save_path = Path(f"~/.cache/equimo/dinov3/{name}").expanduser()
        archive_path = save_path.with_name(f"{save_path.name}.tar.lz4")
        if resume and archive_path.exists():
            print(f"{name}: archive exists, skipping because EQUIMO_DINOV3_RESUME=1")
            continue

        torch_name = "_".join(name.split("_")[:-2])
        torch_hub_cfg = {
            "repo_or_dir": str(DIR / "dinov3"),
            "model": torch_name,
            "source": "local",
            "weights": weights[name],
        }
        # model = torch.hub.load(**torch_hub_cfg)

        replace_cfg = {
            "reg_tokens": "storage_tokens",
            "blocks.0.blocks": "blocks",
            ".prenorm.": ".norm1.",
            ".norm.": ".norm2.",
        }
        expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
        squeeze_cfg = {
            "pos_embed": 0,
            "cls_token": 0,
            "storage_tokens": 0,
        }
        torch_whitelist = []
        jax_whitelist = ["local_pos_embed.patch_rope.freqs", "pos_embed.periods"]

        arr = rng.standard_normal((3, cfg["img_size"], cfg["img_size"])).astype(
            np.float32
        )
        jax_arr = jnp.array(arr)

        if cfg["dim"] >= 4096:
            torch_features = _torch_x_prenorm(
                torch=torch,
                torch_hub_cfg=torch_hub_cfg,
                cfg=cfg,
                name=name,
                arr=arr,
            )
            dinov3 = em.VisionTransformer(
                **cfg,
                key=key,
            )
            dinov3 = _stream_torch_state_into_equinox(
                dinov3,
                weights[name],
                replace_cfg=replace_cfg,
                expand_cfg=expand_cfg,
                squeeze_cfg=squeeze_cfg,
                torch_whitelist=torch_whitelist,
                jax_whitelist=jax_whitelist,
                torch=torch,
            )
        else:
            dinov3 = em.VisionTransformer(
                **cfg,
                key=key,
            )
            dinov3, torch_model = convert_torch_to_equinox(
                dinov3,
                replace_cfg,
                expand_cfg,
                squeeze_cfg,
                torch_whitelist,
                jax_whitelist,
                strict=True,
                torch_hub_cfg=torch_hub_cfg,
                return_torch=True,
            )
            dinov3 = eqx.nn.inference_mode(dinov3, True)
            _fix_torch_rope_periods(torch_model, cfg, name)
            with torch.no_grad():
                torch_features = torch_model.forward_features(
                    torch.tensor(arr).unsqueeze(0).float()
                )["x_prenorm"]
            del torch_model
            gc.collect()

        _assert_rope_freqs(dinov3, cfg, name)

        err = compare(
            dinov3.features(jax_arr, inference=True, key=key),
            torch_features,
        )
        assert err < 5e-4, f"{name} conversion error: {err}"
        print(f"{name} conversion MAE: {err:.3e}")

        save_model(
            save_path,
            dinov3,
            cfg,
            torch_hub_cfg,
            compression=True,
        )
        del dinov3
        gc.collect()

        loaded = em.VisionTransformer(
            **cfg,
            key=key,
        )
        loaded = load_weights(loaded, path=archive_path)
        loaded = eqx.nn.inference_mode(loaded, True)
        _assert_rope_freqs(loaded, cfg, name)
        archive_err = compare(
            loaded.features(jax_arr, inference=True, key=key),
            torch_features,
        )
        assert archive_err < 5e-4, f"{name} archive error: {archive_err}"
        print(f"{name} archive MAE: {archive_err:.3e}")


if __name__ == "__main__":
    main()
