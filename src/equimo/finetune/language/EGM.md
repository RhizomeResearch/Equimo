# equimo.finetune.language

## Overview

Language-specific entry points for the fine-tuning subsystem. The package adds text semantic tags on top of the
canonical parameter tags and provides recipe presets for LoRA, prefix tuning, and projection heads on text encoders. All
work is delegated to the parent package (`src/equimo/finetune/`) and its `peft` subpackage.

## Interface

- `language_tags_for_path(path, leaf=None) -> frozenset[str]`: canonical tags plus `text.*` tags
  [selectors.py:language_tags_for_path]
- `make_tag_tree(model)`: `ParamInfo` tree built with the language tagger [selectors.py:make_tag_tree]
- `resolve_target(model, target, *, allow_empty=False) -> tuple[ParamInfo, ...]` and
  `resolve_target_paths(...) -> tuple[str, ...]`: `TargetSpec` resolution using language tags
  [selectors.py:resolve_target] [selectors.py:resolve_target_paths]
- `lora_encoder(model, *, key, rank=8, alpha=16.0)`: LoRA on tags `attention.qkv` and `attention.proj` via `apply_lora`
  [recipes.py:lora_encoder]
- `prefix_encoder(model, *, key, num_prefix_tokens=16) -> PrefixTunedModel`: wraps `apply_prefixes` with `PrefixConfig`
  [recipes.py:prefix_encoder]
- `projection_head(in_features, out_features, *, key) -> ProjectionHead` [recipes.py:projection_head]
- Package exports: the three recipe functions plus the `recipes` and `selectors` submodules
  [\_\_init\_\_.py:\_\_all\_\_]

## Invariants

- Language tags are a superset of `canonical_tags_for_path(path, leaf)`: path parts `token_embed`, `pos_embed`, and
  `blocks` add `text.embedding`/`text.token`, `text.position`, and `text.block`; canonical tags `attention.qkv`,
  `attention.proj`, `mlp.fc1`, `mlp.fc2`, and `head` add the matching `text.` prefixed tag (`head` maps to
  `text.projection_head`) [selectors.py:language_tags_for_path]
- `resolve_target`, `resolve_target_paths`, and `make_tag_tree` always pass `tagger=language_tags_for_path` to the
  shared implementations in `finetune/selectors.py` and `finetune/tags.py` [selectors.py:resolve_target]
  [selectors.py:resolve_target_paths] [selectors.py:make_tag_tree]
- The LoRA target set of `lora_encoder` is fixed to `TargetSpec(tags_any=("attention.qkv", "attention.proj"))`
  [recipes.py:lora_encoder]

## Side Effects

- None: functions return new modules or PyTrees and consume only the PRNG keys passed in [recipes.py:lora_encoder]
  [recipes.py:prefix_encoder]

## Verification

- Run: `devenv shell -- uv run pytest tests/finetune/test_language_recipes.py tests/finetune/test_tags.py -q`
