# Calibration Collectors

Equimo provides prototype, optimizer-neutral collectors for the statistics used
by EVA and RegMean. The functions are pure state transitions: the caller owns
the model calls, activation extraction, batches, iteration, JIT boundary,
devices, and any distributed orchestration.

## Contracts

One sample is one row after flattening every leading dimension of a supplied
value. Each update requires exactly one of:

- a boolean `sample_mask`, shaped like the value without its final feature
  dimension; or
- a `sample_count` equal to the number of already-filtered rows.

Activation logical IDs are the target module paths expected by EVA, such as
`blocks.0.attn.proj`. RegMean logical IDs identify the corresponding matrix
weight. A schema binds each ID to its input feature width before any values are
collected.

The collector kinds have distinct finalization contracts:

| Kind | Centering | Default denominator | Orientation | Consumer |
|---|---:|---:|---|---|
| `activation_covariance` | configurable | `N` (`sample_mean`) | features by features | EVA |
| `activation_svd` | configurable | `N` (`sample_mean`) | right vectors are rank by features | EVA |
| `input_covariance` | no | none (`sum`) | input features by input features | RegMean |

`input_covariance` therefore produces `X.T @ X`, matching RegMean's
`out_in` weight multiplication. Centered activation covariance uses the
population denominator `N`; a centered singleton is the zero matrix. SVD
finalization rejects a requested rank larger than the number of effective
samples, where centering consumes one degree of freedom.

## Caller-owned collection

```python
import equimo.finetune as eqft

state = eqft.initialize_calibration_collector(
    kind="activation_svd",
    logical_parameter_dims={
        "blocks.0.attn.proj": 768,
        "blocks.1.attn.proj": 768,
    },
    base_checkpoint_hash="sha256:...",
    data_fingerprint="sha256:preprocessing-and-calibration-data",
    centered=True,
    rank=16,
)

for batch in calibration_batches:  # owned by the caller
    named_inputs, valid_mask = extract_named_inputs(model, batch)
    state = eqft.update_calibration_collector(
        state,
        named_inputs,
        sample_mask=valid_mask,
    )

artifacts = eqft.finalize_calibration_collector(state)
eqft.save_calibration_artifacts("eva-calibration.eqft", artifacts)
```

The extraction function in this example is intentionally application-owned;
Equimo does not install hooks or call the model. A caller may JIT `update` and
may combine compatible partial states with
`combine_calibration_collectors`. Combining rejects different logical schemas,
checkpoint hashes, preprocessing/data fingerprints, statistic kinds,
centering, dtypes, ranks, normalizations, or reduction metadata.

For RegMean, finalized `input_covariance` artifacts can be passed directly as
covariance leaves:

```python
merged = eqft.regmean_merge(
    [model_a, model_b],
    [covariance_artifacts_a, covariance_artifacts_b],
)
```

## Cost and prototype decision

For an input width `d`, each logical ID retains a count, a length-`d` mean, and
a `d` by `d` accumulator. State memory is therefore `O(d²)` regardless of the
requested SVD rank; no samples are retained. With float32 accumulation the
array storage is `4 * (1 + d + d²)` bytes per ID. SVD finalization is also
full-rank work before truncation. Callers should account for this cost before
collecting wide layers.

Deterministic float32 tests at widths 2–4 compare streaming, chunked, combined,
masked, eager, and JIT updates to direct calculations at `1e-6` absolute
tolerance. The protocol is a **go for extension as a prototype**: two
covariance-based consumers share the state transition without giving Equimo a
training loop. The public shapes and numerical conventions are explicit, but
future Fisher, quantization-residual, low-rank/sketched, sharded, or distributed
collectors must remain separate until their mathematics and reduction
semantics are tested. `distributed_reduction` is caller-declared provenance;
the collector does not perform a collective operation.
