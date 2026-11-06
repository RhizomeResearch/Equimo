# equimo.core

## Overview

This is the code that every modality shares. The direct files provide:

- PRNG helpers that make inference possible without a key;
- an EMA parameter update;
- semantic tags for parameter paths, buffer and tag field markers, and the trainability and weight-decay masks built
  from them;
- a config-registry model factory that can also load pretrained weights;
- validation of the indices used to select intermediate features.

The subdirectories hold the shared layers, the numerical kernels and the Deep Equilibrium layers. The vision, language,
audio and tabular packages build on all of these.

## Interface

`equimo.core` itself re-exports only `update_ema_model` ([`__init__.py`](./__init__.py)). Import everything else from
its submodule.

- `[emamodel.py:update_ema_model]`: `update_ema_model(ema_model, current_params, decay=0.9999)`, a `jax.jit`-compiled
  function. It returns a new module whose trainable leaves are `decay * ema + (1 - decay) * new`.
- `[tags.py:canonical_tags_for_path]`: `canonical_tags_for_path(path, leaf=None) -> frozenset[str]` infers tags such as
  `norm` and `embedding.position` from a parameter path, `bias` for a last path part named `bias` or ending in `_bias`,
  and `vector` when *leaf* is an array with at most one axis. `[tags.py:CANONICAL_TAGS]` lists the canonical tags,
  `Tagger` is the `(path, leaf) -> Iterable[str]` type, `[tags.py:infer_depth]` derives block indices, and
  `[tags.py:key_path_to_path]` converts a JAX key path to the path form. `equimo.finetune` re-exports
  `canonical_tags_for_path`, `CANONICAL_TAGS`, and `infer_depth`; `key_path_to_path` is available from
  `equimo.finetune.paths`.
- `[parameters.py:tagged]`: `tagged(*tags, when=None, **field_kwargs)` declares an Equinox field whose leaves carry
  `tags`, optionally only when `when(module)` is true, and raises `ValueError` without tags. `[parameters.py:buffer]` is
  `tagged("buffer", ...)`: an array the module stores but never trains.
- `[parameters.py:module_tags]`, `[parameters.py:buffer_mask]`, `[parameters.py:trainable_mask]`: return the tree with
  each leaf replaced by its module-given tags, by whether it is a buffer, and by whether it is a floating array outside
  buffers. `trainable_mask` is an Equinox filter specification.
- `[parameters.py:weight_decay_mask]`:
  `weight_decay_mask(tree, *, no_decay_tags=DEFAULT_NO_DECAY_TAGS, tagger=canonical_tags_for_path)` returns whether each
  leaf decays. It can be passed directly as an optax `mask`. `[parameters.py:DEFAULT_NO_DECAY_TAGS]` lists biases,
  norms, layer scales, state-space dynamics (`ssm`), arrays with at most one axis (`vector`), position embeddings, and
  learned class, register, distillation, and mask tokens.
- `[factory.py:build_model_variant]`:
  `build_model_variant(model_cls, registry, variant, *, pretrained=False, inference_mode=True, key=None, deepcopy_cfg=False, pretrained_variants=None, pretrained_identifiers=None, pretrained_label=None, allow_pretrained_overrides=True, **overrides)`.
  - It builds `model_cls(**(base_cfg | variant_cfg | overrides), key=key)`, where `registry[variant]` is the pair
    `(base_cfg, variant_cfg)`.
  - When `pretrained=True`, it then calls `equimo.serialization.load_weights`.
- `[intermediates.py:intermediate_indices]`:
  `intermediate_indices(total, *, indices=None, n_last_blocks=None) -> frozenset[int]`. It returns every index by
  default, the last `n_last_blocks` indices, or the given `indices` with negative values wrapped.
  `BlockChunk.intermediate_features` uses it.
- `[_prng.py:split_for_mode]`: `split_for_mode(key, num=2, *, inference)`. It returns `(key,) * num` without splitting
  when `inference is True`, and `jr.split(key, num)` otherwise. This is the key-splitting helper that model forward
  passes use.
- `[_prng.py:fold_in_for_mode]`: returns `key` unchanged when `inference is True` or `key is None`, and
  `jr.fold_in(key, data)` otherwise.
- `[_prng.py:default_key_for_mode]`: returns `key` when it is set or when `inference is True`, and `jr.PRNGKey(seed)`
  otherwise.

Subdirectories:

- [layers/EGM.md](./layers/EGM.md): attention (XLA and opt-in cuDNN), rotary embeddings, FFNs, norms, drop-path, the
  Mamba2 mixer, and generic containers, with string-keyed layer registries and `get_layer`.
- [ops/EGM.md](./ops/EGM.md): state-space scan kernels, and softmax/log-softmax/cross-entropy computed in at least
  float32 with custom JVPs that stay accurate at high confidence.
- [implicit/EGM.md](./implicit/EGM.md): Deep Equilibrium layers (`DEQBlock`, `DEQCell`) solved with `banax`. Their
  injectors, stabilizers and stack strategies are pluggable through registries.

## Invariants

- `[_prng.py:split_for_mode]`: `key=None` is accepted only when `inference is True`. Otherwise it raises `ValueError`.
  `inference=None` counts as stochastic. Forward passes therefore run without a key only under static inference.
- `[emamodel.py:update_ema_model]`: it averages only trainable leaves (`trainable_mask`). Other array leaves, such as
  integer indices and buffers, take the current value, and non-array leaves are kept. `current_params` must have the
  same tree structure as the filtered EMA parameters. It returns a new module and mutates nothing.
- `[parameters.py:weight_decay_mask]`: a leaf decays only when it is trainable and none of its tags (path tags plus
  `module_tags`) is in `no_decay_tags`. Buffers never decay, even with `no_decay_tags=()`.
- `[parameters.py:module_tags]`: a leaf held directly by a class in `_NORM_REGISTRY` gets `norm` unless its own field
  declares tags, so norms nested in containers such as `eqx.nn.Sequential` are found whatever their field name. A
  `tagged` field gives its tags to every leaf below it.
- `[parameters.py:buffer]`: the marker only affects masks; forward passes must still read buffers through
  `jax.lax.stop_gradient`. Values that follow from static configuration are computed at use instead of being stored.
- `[factory.py:build_model_variant]`: it raises these errors:
  - `KeyError` when `variant` is unknown;
  - `ValueError` when `pretrained=True` and `variant` is not in `pretrained_variants`;
  - `ValueError` when `pretrained=True`, `allow_pretrained_overrides=False` and overrides are given.
- `[factory.py:build_model_variant]`: on configs and keys:
  - Overrides take precedence over `variant_cfg`, which takes precedence over `base_cfg`.
  - The merge builds a new top-level dict but shares nested containers with the registry. `deepcopy_cfg=True` makes a
    deep copy for models that mutate nested config.
  - `key=None` becomes `jax.random.PRNGKey(42)`.
- `[intermediates.py:intermediate_indices]`: it raises `ValueError` in these cases:
  - both `indices` and `n_last_blocks` are given;
  - `total < 0`;
  - `n_last_blocks` is outside `[1, total]`;
  - an index is out of range after negative indices are wrapped.

## Side Effects

- `[factory.py:build_model_variant]`: with `pretrained=True`, it lazily imports `equimo.serialization.load_weights`,
  which fetches and deserializes the checkpoint (network and disk I/O).
- `[factory.py:build_model_variant]`: model constructors can mutate nested config containers that are shared with the
  registry unless `deepcopy_cfg=True`.
- Subpackages fill module-global layer and component registries when imported. See [layers/EGM.md](./layers/EGM.md) and
  [implicit/EGM.md](./implicit/EGM.md).

## Verification

- Run:
  `devenv shell -- uv run pytest tests/test_emamodel.py tests/test_core_parameters.py tests/finetune/test_tags.py tests/test_model_factory.py tests/test_prng_free_inference.py tests/test_models.py::test_vit_intermediate_features_last_block_matches_features -q`
- `tests/test_emamodel.py` covers `emamodel.py`, `tests/test_core_parameters.py` covers `parameters.py`,
  `tests/test_model_factory.py` covers `factory.py`, and `tests/test_prng_free_inference.py` covers `_prng.py`.
  `tags.py` is covered through `tests/finetune/test_tags.py`, and its `vector` rule through `weight_decay_mask` in
  `tests/test_core_parameters.py`. `intermediates.py` has no direct test: the ViT
  `intermediate_features(..., n_last_blocks=2)` test exercises it indirectly, and its `ValueError` cases are untested.
- The subdirectories each have their own Verification commands.
