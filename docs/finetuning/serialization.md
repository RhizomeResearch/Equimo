# Serialization

`save_delta` writes a `FineTuneBundle` containing method metadata,
architecture hash, schema version, target paths, shapes, and method state.
The `.eqft` file is a pickle-free LZ4-compressed archive containing JSON
metadata and Equinox-serialized array leaves.

```python
bundle = eqft.save_delta(model, "delta.eqft", method="lora")
loaded = eqft.load_delta(base_model, "delta.eqft")
```

The spec-style order is also accepted:

```python
bundle = eqft.save_delta("delta.eqft", model, base_model, spec)
loaded = eqft.load_delta("delta.eqft", base_model)
```

Loading checks architecture hashes and target shapes. Incompatible bases raise
a method-specific error.

Use `save_finetune_bundle` and `load_finetune_bundle` when you already have a
`FineTuneBundle`. Bundle metadata includes parameter counts, dtype summary,
target paths, a base checkpoint hash, mergeability, and optional user metadata.

An executable [`FeatureSpec`](feature_specs.md) can be stored with the delta:

```python
bundle = eqft.save_delta(
    model,
    "delta.eqft",
    method="lora",
    feature_spec=feature_spec,
)
```

The versioned feature-spec codec preserves preprocessing identity and rejects
unknown fields or values instead of silently changing extraction behavior.

Calibration artifact sets use a separate versioned format with the same
pickle-free array encoding:

```python
eqft.save_calibration_artifacts("calibration.eqft", artifacts)
artifacts = eqft.load_calibration_artifacts("calibration.eqft")
```

Saving and loading validate logical IDs, statistic shapes, checkpoint hashes,
data fingerprints, accumulation dtypes, and reduction metadata. See
[Calibration collectors](calibration_collectors.md) for the numerical
contracts.

Supported delta methods: `lora`, `dora`, `adapter`, `prompt`, `prefix`,
`scale_shift`, `ia3`, and `vera`.
