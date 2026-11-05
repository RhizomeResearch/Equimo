# equimo.conversion

## Overview

Utilities for porting PyTorch weights into Equimo's Equinox models. They walk the Equinox parameter tree, derive a
PyTorch-style dotted name for each array, apply caller-supplied renames and reshapes, copy matching PyTorch tensors, and
return the model in inference mode. The maintainer conversion scripts under the repository's `models/` directory build
on these helpers. The package `__init__.py` is empty.

## Interface

- `convert_torch_to_equinox(jax_model, replace_cfg={}, expand_cfg={}, squeeze_cfg={}, torch_whitelist=[], jax_whitelist=[], strict=True, source="torchhub", torch_hub_cfg=None, torch_model=None, timm_cfg=None, return_torch=False)`
  \[utils.py:convert_torch_to_equinox\]: returns the converted model, or `(model, torch_model.eval())` when
  `return_torch=True`.
- `convert_params_from_torch(jax_model, replace_cfg, expand_cfg, squeeze_cfg, torch_whitelist, jax_whitelist, strict=True, source="torchhub", ...)`
  \[utils.py:convert_params_from_torch\]: returns the converted array PyTree (and the Torch model when
  `return_torch=True`).
- `stringify_name(path) -> str` \[utils.py:stringify_name\]: joins `GetAttrKey` names, `SequenceKey` indices, and
  `DictKey` keys with dots; any other path entry raises `TypeError`.
- `torch_parameter_name(path, replace_cfg) -> str` \[utils.py:torch_parameter_name\]: the PyTorch name a JAX parameter
  converts from.
- `expand_torch_tensor(tensor, pos, n)` \[utils.py:expand_torch_tensor\]: adds `n` singleton axes `"before"` or
  `"after"`.

## Invariants

- `source` selects the Torch model: `"torchhub"` needs `torch_hub_cfg`, `"timm"` needs `timm_cfg`, `"custom"` needs
  `torch_model`; a missing argument raises `ValueError`. Any other `source` raises `ValueError`, even when a
  `torch_model` is given [utils.py:convert_params_from_torch].
- Parameter names come from `stringify_name`, then every whole `scale` or `kernel` path component becomes `weight`
  (`proj.scale_x` and `conv_kernel` are unchanged), then every `replace_cfg` substring replacement applies in order
  [utils.py:torch_parameter_name].
- Candidates come only from `torch_model.named_parameters()`; buffers such as BatchNorm running statistics are never
  matched, so JAX arrays that correspond to buffers count as missing [utils.py:convert_params_from_torch].
- `expand_cfg[name]` is `(pos, n)` passed to `expand_torch_tensor`; `squeeze_cfg[name]` is the `torch.squeeze` dim;
  after reshaping, a shape mismatch always raises `ValueError` [utils.py:convert_params_from_torch].
- With `strict=True`, a JAX parameter missing from PyTorch raises `AttributeError` unless it is in `jax_whitelist`, and
  a leftover PyTorch parameter raises `AttributeError` unless it is in `torch_whitelist`; non-strict or whitelisted
  missing parameters keep their original JAX values [utils.py:convert_params_from_torch].
- Converted arrays take the dtype of the JAX parameter they replace; tensors NumPy cannot represent directly are
  converted through `float()` first [utils.py:convert_params_from_torch].
- Only array leaves are converted; static structure is recombined unchanged and the result is put in
  `eqx.nn.inference_mode(..., value=True)` [utils.py:convert_torch_to_equinox].

## Side Effects

- Lazily imports `torch` and `timm`, raising `ImportError` pointing to the `torch` extra when missing
  [utils.py:convert_params_from_torch].
- `source="torchhub"` calls `torch.hub.load` and `source="timm"` calls `timm.create_model`, which may download code and
  weights [utils.py:convert_params_from_torch].
- Logs every converted, missing, and unconverted parameter through `loguru` [utils.py:convert_params_from_torch].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_conversion_utils.py tests/test_conversion_optional.py -q`
