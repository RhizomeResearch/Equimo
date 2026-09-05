# Performance audit

This audit targets redundant computation and host-side traversal while preserving
Equimo's numerical tolerances, shapes, dtypes, PRNG behavior, public signatures,
PyTree structure, and checkpoint formats. Measurements below are CPU results,
not accelerator performance claims.

## Changes

| Area | Change | Preserved behavior |
| --- | --- | --- |
| Tabular `InContextAttention` | Project keys and values from training rows only. | All rows remain queries; scaling, training/test head selection, and output projection are unchanged. |
| `PartialFormerBlock` | Construct the inverse patch permutation with an integer scatter instead of a second sort. | The primary stable ranking, foreground/background split, spatial restoration, and PRNG sequence are unchanged. |
| RegMean Cholesky solver | Use triangular solves after Cholesky instead of factoring both triangular matrices again. | Regularization, solver selection, output dtype, and plain transpose semantics are unchanged. |
| Fine-tuning selectors | Match each parameter's ancestor paths against the selected modules. | Predicate call order, leaf order, depth filters, exclusions, and empty-selection handling are unchanged. |
| PEFT merge/unmerge | Replace all wrappers of one family with one `eqx.tree_at` call. | Outer-wrapper traversal boundaries, callback order, family order, and no-match identity are unchanged. |

Selector descendant matching no longer scans every parameter for each matched
module. Wrapper replacement no longer rebuilds the entire model once per wrapper.
These improvements concern model setup and merge/unmerge overhead; they do not
change the arithmetic performed by the adapters during training.

## Measurements

Baseline: source at `a4b3a18b` (`chore: misc simplifications`). Environment:
Python 3.12.5, JAX 0.9.2, Equinox 0.13.6, Linux x86-64, one JAX CPU device,
x64 disabled. The paired run used CPU affinity 0–3 and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

Each compiled function receives model arrays as runtime arguments. The harness
records lowering/compilation separately, performs three warm-ups, synchronizes
every result, and reports the median of three rounds of 20 measurements. Slow
host-only workloads use three measurements per round. Paired runs alternate
which implementation executes first and check output parity before timing.

| Workload | Baseline ms | Optimized ms | Speedup |
| --- | ---: | ---: | ---: |
| Attention: 1,024 rows, 128 training, width 128, full KV heads, float32 | 1.906 | 1.654 | 1.15× |
| Attention: 4,096 rows, 128 training, width 256, one test KV head, float32 | 21.093 | 16.152 | 1.31× |
| Same larger attention case, bfloat16 | 32.742 | 27.497 | 1.19× |
| Inverse permutation only: 1,024 patches, 4 tokens/patch, width 64 | 0.172 | 0.079 | 2.18× |
| Full PartialFormer block: 28×28 tokens, width 32, patch size 2 | 4.248 | 4.144 | 1.03× |
| RegMean: 64×64 system, 32 right-hand sides | 0.099 | 0.049 | 2.04× |
| RegMean: 256×256 system, 128 right-hand sides | 2.067 | 0.755 | 2.74× |
| Resolve linear selector over 1,024 small modules | 243.538 | 47.091 | 5.17× |
| Merge 64 small LoRA wrappers | 164.601 | 22.903 | 7.19× |
| Merge 256 small LoRA wrappers | 3,228.788 | 85.475 | 37.77× |

The inverse-permutation comparison measures the two restoration algorithms
directly. Its speedup does not describe the whole attention block: the complete
14×14 block was approximately unchanged (0.681 versus 0.691 ms), and the larger
block improved modestly. Selector and wrapper cases repeat a small module at
distinct tree paths to isolate traversal cost; larger weights can reduce the
relative benefit.

Across the mixed-row attention cases, latency improved by approximately 5–23%.
The larger attention case reduced compiled arithmetic from 2.815 to 1.775 billion
FLOPs (approximately 37%). Compiled temporary memory was unchanged for all
attention cases. RegMean temporary memory fell from 50,436 to 32,836 bytes for the
64-dimensional case and from 790,020 to 524,356 bytes for the 256-dimensional case.
Compiler memory estimates are not process RSS or measured device peak memory.

Other CPU-heavy processes were active during this audit. Separate before/after
processes produced inconsistent timings even for unchanged controls, so the table
uses alternating measurements within one process. One all-training attention
control initially appeared slower; a targeted three-round rerun with 200
measurements per round measured 0.384 versus 0.381 ms, with identical arithmetic
and temporary memory. No reproducible latency or temporary-memory regression
above 5% remained in the representative cases. Compilation timings are recorded
by the harness but are too variable here to claim compilation improvements.

### Reproduction

Capture a source snapshot before editing; it must contain the `equimo/` package.
Use the same dependency environment for both implementations. For example:

```bash
mkdir -p .cache/performance-baseline
cp -R src/equimo .cache/performance-baseline/
```

After applying the change, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run benchmarks/performance.py \
  --baseline-source .cache/performance-baseline \
  --output .cache/performance-results.json
```

For an already-applied change, supply a trusted copy of the baseline revision's
`src/` directory instead. The paired loader executes the selected baseline
functions and their imports in isolated namespaces, using the current model
classes and unchanged helpers. It is specific to this audit's unchanged model
layouts; it is not a general comparison tool for incompatible revisions.

Omit `--baseline-source` to measure only the installed source. Use `--only`
with `attention`, `partialformer`, `regmean`, `selectors`, or `wrappers` to narrow
the run. `--rounds` and `--repeats` control sampling. On Linux, CPU affinity can be
held fixed with `taskset -c <available CPUs>` before `uv run`; choose CPUs valid
for the current host. JSON output includes per-round latencies, compilation
times, FLOP estimates, temporary memory, and dependency/device information.

## Audit coverage and deferred candidates

The source review covered shared layers and scans, modality entry points,
vision convolution/attention/position/wavelet paths, fine-tuning traversal and
linear algebra, preprocessing, conversion, and checkpoint I/O. Targeted
experiments were used for the candidates above; this was not an exhaustive
profiler run of every registered model.

| Area | Assessment |
| --- | --- |
| Core attention; language, audio, and time-series transformers | Preserve explicit softmax precision, masks, fully masked rows, and dropout. Fused-attention substitution needs accelerator measurements and separate parity evidence. |
| Core scans | SSD constructs a quadratic matrix over chunks, but recurrence reformulation changes floating-point evaluation order. Defer until long-sequence, initial-state, and gradient parity are established. |
| Grouped tabular attention | Avoiding repeated KV heads produced mixed compiler results. One common bfloat16 case increased temporary memory by approximately 28%; this rewrite was not retained. |
| Linear projections and model stacks | Existing `vmap` projections already lower to batched linear algebra; Python loops and reshapes alone are not evidence of runtime overhead. No blanket replacements. |
| Vision convolution, wavelets, and position embeddings | Haar transforms already use shared polyphase arithmetic. Shape-dependent carrier-token indexing runs at trace time. No additional measured change was selected. |
| PRNG handling | Shared helpers already remove unnecessary key work during static deterministic inference. Preserve training splits and folding order. |
| Fine-tuning numerical adapters | Preserve existing dense/factored choices and adapter reconstruction semantics. This pass changes traversal and the Cholesky solve only. |
| Audio/image preprocessing and conversion | Keep checkpoint-faithful preprocessing and optional integrations intact. No network-dependent performance experiment was needed. |
| Serialization and checkpoint I/O | Existing bounded streaming and checksum verification are compatibility and integrity requirements. No format, caching, or validation shortcuts were introduced. |

## Verification

The 29 new preservation cases passed against the baseline before implementation.
After implementation, all 129 focused attention, TabPFN, selector, RegMean,
wrapper-mapping, and LoRA-lifecycle tests passed. Coverage includes JIT, `vmap`,
float32/bfloat16 outputs and gradients where supported, ranked-patch ties,
foreground split extremes, root/nested selectors, callback order, and mixed-family
serialization. RegMean additionally checks normalized linear-system residuals
and non-positive-definite NaN behavior.

Separate baseline/candidate comparisons matched all 120 PartialFormer output,
input-gradient, and parameter-gradient arrays across float32/bfloat16 training
and inference with fixed dropout keys. Both changed attention paths also passed
output and gradient checks using sharded inputs across two virtual CPU devices.
The cached TabPFN classifier passed its upstream-reference parity test. A complex
RegMean comparison confirmed preservation of the existing plain-transpose behavior.

Repository-wide Ruff lint and format checks passed, including the benchmark
directory. `ty check src` exited successfully with seven pre-existing unused-ignore
warnings in unrelated optional-import code.

The full pytest run finished with **1,788 passed, 7 skipped, 9 deselected, and one
failure** in 729.81 seconds. The failing wheel-metadata test used a pre-existing
`dist/` wheel whose embedded README differs from the repository's baseline README;
this change does not modify README or packaging metadata. A fresh
`uv build --no-sources` into a temporary directory passed all **three distribution
tests**, including installation/imports in a clean environment. The unmodified
distribution test module's artifact-directory constant was pointed at that build
in an isolated test process, preserving the existing `dist/` artifacts. Running
pytest against those older artifacts will continue to report the metadata mismatch
until they are rebuilt.

Accelerator timing and physical multi-device performance remain unmeasured.
