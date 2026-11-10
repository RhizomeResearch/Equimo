# equimo.finetune.audio

## Overview

Audio-specific entry points for the fine-tuning subsystem. The package adds audio semantic tags on top of the canonical
parameter tags, and provides thin recipe presets that build heads, pooling, linear probes, adapters, and LoRA for
AST-like audio encoders. Every function delegates to the generic machinery in the parent package
(`src/equimo/finetune/`) and its `peft` subpackage; nothing here owns parameters or training state of its own.

## Interface

- `audio_tags_for_path(path, leaf=None) -> frozenset[str]`: canonical tags plus `audio.patch_embed`,
  `audio.transformer.block`, `audio.tagging_head`, and `audio.pooling` [selectors.py:audio_tags_for_path]
- `make_tag_tree(model)`: `ParamInfo` tree built with the audio tagger [selectors.py:make_tag_tree]
- `resolve_target(model, target, *, allow_empty=False) -> tuple[ParamInfo, ...]` and
  `resolve_target_paths(...) -> tuple[str, ...]`: `TargetSpec` resolution using audio tags [selectors.py:resolve_target]
  [selectors.py:resolve_target_paths]
- `linear_probe_ast(model, *, in_features, out_features, key, pool="mean_frame") -> LinearProbe`: wraps
  `make_linear_probe` [recipes.py:linear_probe_ast]
- `multilabel_tagging_head(in_features, out_features, *, key) -> MultiLabelHead` [recipes.py:multilabel_tagging_head]
- `mean_frame_pool() -> MeanFramePool` [recipes.py:mean_frame_pool]
- `ctc_head(in_features, vocab_size, *, key, blank_id=0) -> CTCHead` [recipes.py:ctc_head]
- `adapter_ast(model, *, key, bottleneck=64, placement="after_mlp")`: applies `AdapterConfig` adapters via
  `apply_adapters` [recipes.py:adapter_ast]
- `lora_ast(model, *, key, rank=8, alpha=16.0, target=("attention.qkv", "attention.proj"))`: applies `LoRAConfig` with
  `TargetSpec(tags_any=target)` via `apply_lora` [recipes.py:lora_ast]
- Package exports: the six recipe functions plus the `recipes` and `selectors` submodules [\_\_init\_\_.py:\_\_all\_\_]

## Invariants

- Audio tags are a superset of `canonical_tags_for_path(path, leaf)`; extra tags are added only when the path parts
  contain `patch_embed`, `blocks`, `head`, or `norm` [selectors.py:audio_tags_for_path]
- `resolve_target`, `resolve_target_paths`, and `make_tag_tree` always pass `tagger=audio_tags_for_path` to the shared
  implementations in `finetune/selectors.py` and `finetune/tags.py`; resolution semantics (including the `allow_empty`
  behavior) are those of the shared functions [selectors.py:resolve_target] [selectors.py:resolve_target_paths]
  [selectors.py:make_tag_tree]
- Recipe presets take explicit PRNG keys and only forward arguments to the shared constructors; defaults are
  `pool="mean_frame"`, `bottleneck=64`, `placement="after_mlp"`, `rank=8`, `alpha=16.0`, and `blank_id=0`
  [recipes.py:linear_probe_ast] [recipes.py:adapter_ast] [recipes.py:lora_ast] [recipes.py:ctc_head]
- `multilabel_tagging_head` and `ctc_head` are documented as raw-logit heads [recipes.py:multilabel_tagging_head]
  [recipes.py:ctc_head]
- `lora_ast` resolves `target` with the canonical tagger that `apply_lora` defaults to, not the audio tagger, so
  `target=("audio.transformer.block",)` matches nothing and raises `ValueError` [recipes.py:lora_ast]
  [../peft/lora.py:apply_lora] [../selectors.py:resolve_target]

## Side Effects

- None: functions return new modules or PyTrees and consume only the PRNG keys passed in [recipes.py:adapter_ast]
  [recipes.py:lora_ast]

## Verification

- Run: `devenv shell -- uv run pytest tests/finetune/test_audio_recipes.py tests/finetune/test_tags.py -q`
