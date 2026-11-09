# Performance audit

This audit targets redundant computation and host-side traversal while preserving Equimo's numerical tolerances, shapes,
dtypes, PRNG behavior, public signatures, PyTree structure, and checkpoint formats. The audit's measurements are CPU
results, not accelerator performance claims; the softmax and fused-attention sections report GPU measurements.

## Changes

| Area                         | Change                                                                                    | Preserved behavior                                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Tabular `InContextAttention` | Project keys and values from training rows only.                                          | All rows remain queries; scaling, training/test head selection, and output projection are unchanged.           |
| `PartialFormerBlock`         | Construct the inverse patch permutation with an integer scatter instead of a second sort. | The primary stable ranking, foreground/background split, spatial restoration, and PRNG sequence are unchanged. |
| RegMean Cholesky solver      | Use triangular solves after Cholesky instead of factoring both triangular matrices again. | Regularization, solver selection, output dtype, and plain transpose semantics are unchanged.                   |
| Fine-tuning selectors        | Match each parameter's ancestor paths against the selected modules.                       | Predicate call order, leaf order, depth filters, exclusions, and empty-selection handling are unchanged.       |
| PEFT merge/unmerge           | Replace all wrappers of one family with one `eqx.tree_at` call.                           | Outer-wrapper traversal boundaries, callback order, family order, and no-match identity are unchanged.         |

Selector descendant matching no longer scans every parameter for each matched module. Wrapper replacement no longer
rebuilds the entire model once per wrapper. These improvements concern model setup and merge/unmerge overhead; they do
not change the arithmetic performed by the adapters during training.

## Measurements

Baseline: source at `a4b3a18b` (`chore: misc simplifications`). Environment: Python 3.12.5, JAX 0.9.2, Equinox 0.13.6,
Linux x86-64, one JAX CPU device, x64 disabled. The paired run used CPU affinity 0–3 and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

Each compiled function receives model arrays as runtime arguments. The harness records lowering/compilation separately,
performs three warm-ups, synchronizes every result, and reports the median of three rounds of 20 measurements. Slow
host-only workloads use three measurements per round. Paired runs alternate which implementation executes first and
check output parity before timing.

| Workload                                                                  | Baseline ms | Optimized ms | Speedup |
| ------------------------------------------------------------------------- | ----------: | -----------: | ------: |
| Attention: 1,024 rows, 128 training, width 128, full KV heads, float32    |       1.906 |        1.654 |   1.15× |
| Attention: 4,096 rows, 128 training, width 256, one test KV head, float32 |      21.093 |       16.152 |   1.31× |
| Same larger attention case, bfloat16                                      |      32.742 |       27.497 |   1.19× |
| Inverse permutation only: 1,024 patches, 4 tokens/patch, width 64         |       0.172 |        0.079 |   2.18× |
| Full PartialFormer block: 28×28 tokens, width 32, patch size 2            |       4.248 |        4.144 |   1.03× |
| RegMean: 64×64 system, 32 right-hand sides                                |       0.099 |        0.049 |   2.04× |
| RegMean: 256×256 system, 128 right-hand sides                             |       2.067 |        0.755 |   2.74× |
| Resolve linear selector over 1,024 small modules                          |     243.538 |       47.091 |   5.17× |
| Merge 64 small LoRA wrappers                                              |     164.601 |       22.903 |   7.19× |
| Merge 256 small LoRA wrappers                                             |   3,228.788 |       85.475 |  37.77× |

The inverse-permutation comparison measures the two restoration algorithms directly. Its speedup does not describe the
whole attention block: the complete 14×14 block was approximately unchanged (0.681 versus 0.691 ms), and the larger
block improved modestly. Selector and wrapper cases repeat a small module at distinct tree paths to isolate traversal
cost; larger weights can reduce the relative benefit.

Across the mixed-row attention cases, latency improved by approximately 5–23%. The larger attention case reduced
compiled arithmetic from 2.815 to 1.775 billion FLOPs (approximately 37%). Compiled temporary memory was unchanged for
all attention cases. RegMean temporary memory fell from 50,436 to 32,836 bytes for the 64-dimensional case and from
790,020 to 524,356 bytes for the 256-dimensional case. Compiler memory estimates are not process RSS or measured device
peak memory.

Other CPU-heavy processes were active during this audit. Separate before/after processes produced inconsistent timings
even for unchanged controls, so the table uses alternating measurements within one process. One all-training attention
control initially appeared slower; a targeted three-round rerun with 200 measurements per round measured 0.384 versus
0.381 ms, with identical arithmetic and temporary memory. No reproducible latency or temporary-memory regression above
5% remained in the representative cases. Compilation timings are recorded by the harness but are too variable here to
claim compilation improvements.

### Reproduction

Capture a source snapshot before editing; it must contain the `equimo/` package. Use the same dependency environment for
both implementations. For example:

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

For an already-applied change, supply a trusted copy of the baseline revision's `src/` directory instead. The paired
loader executes the selected baseline functions and their imports in isolated namespaces, using the current model
classes and unchanged helpers. It is specific to this audit's unchanged model layouts; it is not a general comparison
tool for incompatible revisions.

Omit `--baseline-source` to measure only the installed source. Use `--only` with `attention`, `partialformer`,
`regmean`, `selectors`, or `wrappers` to narrow the run. `--rounds` and `--repeats` control sampling. On Linux, CPU
affinity can be held fixed with `taskset -c <available CPUs>` before `uv run`; choose CPUs valid for the current host.
JSON output includes per-round latencies, compilation times, FLOP estimates, temporary memory, and dependency/device
information.

## Audit coverage and deferred candidates

The source review covered shared layers and scans, modality entry points, vision convolution/attention/position/wavelet
paths, fine-tuning traversal and linear algebra, preprocessing, conversion, and checkpoint I/O. Targeted experiments
were used for the candidates above; this was not an exhaustive profiler run of every registered model.

| Area                                                          | Assessment                                                                                                                                                                                          |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core attention; language, audio, and time-series transformers | Preserve explicit softmax precision, masks, fully masked rows, and dropout. Fused cuDNN attention is opt-in; see [Fused attention](#fused-attention).                                               |
| Core scans                                                    | SSD constructs a quadratic matrix over chunks, but recurrence reformulation changes floating-point evaluation order. Defer until long-sequence, initial-state, and gradient parity are established. |
| Grouped tabular attention                                     | Avoiding repeated KV heads produced mixed compiler results. One common bfloat16 case increased temporary memory by approximately 28%; this rewrite was not retained.                                |
| Linear projections and model stacks                           | Existing `vmap` projections already lower to batched linear algebra; Python loops and reshapes alone are not evidence of runtime overhead. No blanket replacements.                                 |
| Vision convolution, wavelets, and position embeddings         | Haar transforms already use shared polyphase arithmetic. Shape-dependent carrier-token indexing runs at trace time. No additional measured change was selected.                                     |
| PRNG handling                                                 | Shared helpers already remove unnecessary key work during static deterministic inference. Preserve training splits and folding order.                                                               |
| Fine-tuning numerical adapters                                | Preserve existing dense/factored choices and adapter reconstruction semantics. This pass changes traversal and the Cholesky solve only.                                                             |
| Audio/image preprocessing and conversion                      | Keep checkpoint-faithful preprocessing and optional integrations intact. No network-dependent performance experiment was needed.                                                                    |
| Serialization and checkpoint I/O                              | Existing bounded streaming and checksum verification are compatibility and integrity requirements. No format, caching, or validation shortcuts were introduced.                                     |

## Verification

The 29 new preservation cases passed against the baseline before implementation. After implementation, all 129 focused
attention, TabPFN, selector, RegMean, wrapper-mapping, and LoRA-lifecycle tests passed. Coverage includes JIT, `vmap`,
float32/bfloat16 outputs and gradients where supported, ranked-patch ties, foreground split extremes, root/nested
selectors, callback order, and mixed-family serialization. RegMean additionally checks normalized linear-system
residuals and non-positive-definite NaN behavior.

Separate baseline/candidate comparisons matched all 120 PartialFormer output, input-gradient, and parameter-gradient
arrays across float32/bfloat16 training and inference with fixed dropout keys. Both changed attention paths also passed
output and gradient checks using sharded inputs across two virtual CPU devices. The cached TabPFN classifier passed its
upstream-reference parity test. A complex RegMean comparison confirmed preservation of the existing plain-transpose
behavior.

Repository-wide Ruff lint and format checks passed, including the benchmark directory. `ty check src` exited
successfully with seven pre-existing unused-ignore warnings in unrelated optional-import code.

The full pytest run finished with **1,788 passed, 7 skipped, 9 deselected, and one failure** in 729.81 seconds. The
failing wheel-metadata test used a pre-existing `dist/` wheel whose embedded README differs from the repository's
baseline README; this change does not modify README or packaging metadata. A fresh `uv build --no-sources` into a
temporary directory passed all **three distribution tests**, including installation/imports in a clean environment. The
unmodified distribution test module's artifact-directory constant was pointed at that build in an isolated test process,
preserving the existing `dist/` artifacts. Running pytest against those older artifacts will continue to report the
metadata mismatch until they are rebuilt.

Accelerator timing and physical multi-device performance remain unmeasured.

## SAM 3.1 image inference and fine-tuning

Measured on an RTX 4080 (16 GB) with `benchmarks/sam3_image.py`, in bfloat16 at 1008 x 1008 on one image, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=1.0`. Weights load on the CPU so that peak
memory reports the workload. Training runs 50 SGD steps with a loss on the 16 highest-scoring queries' mask logits; the
loss stayed finite and decreased in every run that completed.

| Workload                                     | Attention | Trunk rematerialized | Peak GPU memory | Step or call time |
| -------------------------------------------- | --------- | -------------------- | --------------: | ----------------: |
| Image inference (encode and detect)          | XLA       | no                   |          2.8 GB |            124 ms |
| Image inference (encode and detect)          | cuDNN     | no                   |          1.4 GB |             82 ms |
| Head-only fine-tuning (detector head)        | XLA       | yes                  |          7.0 GB |            0.84 s |
| Head-only fine-tuning (detector head)        | cuDNN     | yes                  |          2.0 GB |            0.69 s |
| LoRA on trunk and detector attention, rank 8 | XLA       | no                   |   out of memory |                 - |
| LoRA on trunk and detector attention, rank 8 | XLA       | yes                  |          7.3 GB |            1.32 s |
| LoRA on trunk and detector attention, rank 8 | cuDNN     | no                   |          8.4 GB |            1.02 s |
| LoRA on trunk and detector attention, rank 8 | cuDNN     | yes                  |          2.2 GB |            1.06 s |

Rematerialization is `encode_image(x, checkpoint_trunk=True)`, which the benchmark's training steps pass. LoRA with XLA
attention and no rematerialization fails trying to allocate 15.4 GB. Run
`benchmarks/sam3_image.py {inference,head,lora} --archive-dir DIR --implementation {xla,cudnn}` from the devenv shell
with the CUDA wheels.

## Softmax precision

These GPU measurements choose where Equimo uses the corrected derivative of `equimo.core.ops.softmax`. Environment:
NVIDIA RTX 4080, JAX 0.9.2 with CUDA 12, Equinox 0.13.6. Errors are relative L2 errors against float64 evaluations of
the same rounded parameters and inputs. Times are medians of 40 compiled calls after five warm-ups.

| Softmax                                                                                             | Computation                      | Reason                                                                                 |
| --------------------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------- |
| Attention between tokens                                                                            | Float32 logits, `jax.nn.softmax` | The corrected derivative added 5-10% to ViT training steps, without a measurable gain. |
| Attention pooling, attention-pooling heads, adapter fusion, tabular decoder, `"softmax"` activation | Float32, `equimo.core.ops`       | Under 0.5% of a training step; stays accurate when every example is confident (below). |

A pooling head has one softmax row per example, so when every row is confident, its whole gradient depends on the
dominant entry. Attention pooling over 64 bfloat16 examples, each with one planted dominant token, gave these query
gradient errors:

| Minimum / median logit gap | Model dtype, `jax.nn.softmax` | Model dtype, corrected | Float32, `jax.nn.softmax` | Float32, corrected |
| -------------------------: | ----------------------------: | ---------------------: | ------------------------: | -----------------: |
|                    5.4 / 8 |                          2.1% |                   2.1% |                     0.26% |              0.26% |
|                   9.4 / 12 |                          2.7% |                   2.1% |                     0.22% |              0.22% |
|                  17.4 / 20 |                        4,654% |                   4.1% |                     0.50% |              0.28% |
|                  27.4 / 30 |                        1.4e8% |                   5.8% |                       99% |              0.38% |

With random inputs, float32 computation reduced the attention-pooling gradient error from 1.6-5.8% to 0.17-0.28%, and
the corrected derivative changed nothing, because the least confident examples dominate the gradient. At the
attention-pooling head, adapter fusion, and tabular decoder, bfloat16 rounding of the projections dominated: no softmax
variant changed their errors by more than about 10%, and producing their logits directly in float32 gave mixed results.

Cost of the corrected derivative over float32 `jax.nn.softmax`, for bfloat16 training steps:

| Workload                                            | Float32 `jax.nn.softmax` | Float32, corrected |
| --------------------------------------------------- | -----------------------: | -----------------: |
| Attention-pooling head, batch 64, 197 tokens        |          2.10 ms, 74 MiB |    2.19 ms, 92 MiB |
| Attention-pooling head, batch 64, 577 tokens        |         6.69 ms, 220 MiB |   7.15 ms, 271 MiB |
| Attention pooling, batch 64, 577 tokens             |                  0.30 ms |            0.35 ms |
| Adapter fusion, batch 64, 197 tokens, four adapters |                  6.64 ms |            6.47 ms |
| Tabular decoder, 4,096 training and 1,024 test rows |                  0.89 ms |            0.84 ms |
| TabPFN, 1,024 training and 512 test rows            |                  29.1 ms |            29.1 ms |

For comparison, a bfloat16 ViT-B/16 training step at batch 64 took 117.5 ms with 5,377 MiB of temporary memory. In
float32 models, the corrected TabPFN step took 50.0 ms against 52.7 ms.

Float32 models were limited by matrix multiplication rather than softmax. With JAX's default precision, the
attention-pooling head's float32 gradient error was 0.6-2.5%; under `jax.default_matmul_precision("highest")` it was
0.001-0.02%, because the default runs float32 matrix multiplications as TF32 on this GPU.

## Fused attention

`equimo.core.layers.with_attention_implementation(model, "cudnn")` runs core `Attention`, prefix-tuned attention, and
tabular `Attention`, `CrossAttention`, and `InContextAttention` through NVIDIA cuDNN's fused kernel, via
`jax.nn.dot_product_attention`. The kernel computes float32 logits without storing the attention matrix. It is opt-in
because it requires a CUDA GPU, bfloat16 or float16 inputs, and inactive attention dropout, because jax2onnx cannot
export it, and because it did not speed up TabPFN below. Switching back with `"xla"` restores the default path; neither
direction changes parameters or checkpoints.

Environment: NVIDIA RTX 4080, JAX 0.11.2 with CUDA 12, Equinox 0.13.6. Times are medians of 30 compiled bfloat16
training steps after five warm-ups; memory is XLA's temporary-buffer estimate.

| Workload                                          |                 XLA |               cuDNN |
| ------------------------------------------------- | ------------------: | ------------------: |
| ViT-S/16, 224 px, batch 128                       |  92.5 ms, 5,414 MiB |  84.1 ms, 4,468 MiB |
| ViT-B/16, 224 px, batch 64                        | 121.4 ms, 5,396 MiB | 113.1 ms, 4,449 MiB |
| ViT-B/16, 448 px, batch 16                        | 160.7 ms, 9,324 MiB | 124.9 ms, 4,117 MiB |
| EoMT on ViT-S/16, 512 px, batch 8, masked queries |  75.0 ms, 4,948 MiB |  62.0 ms, 2,923 MiB |
| TabPFN, 1,024 training and 512 test rows          |  30.1 ms, 2,541 MiB |  31.6 ms, 1,288 MiB |

Accuracy did not change. Against float64, bfloat16 attention layers of width 384 and 768 at 197 and 577 tokens, with and
without a causal mask, had 0.49-0.66% relative error in outputs and gradients with either implementation, because
bfloat16 projections dominate. Float32 layers had 0.03-0.06%.

The backward kernel requires sequence lengths divisible by 64, so Equimo pads queries and keys and excludes the padding.
Masks are expanded to every head, because a broadcast head axis fails under `vmap` (jax-ml/jax#34920), and fully masked
query rows return zeros, as in the XLA path. SHSA is not covered, because its queries and keys are narrower than its
values. ConvAttention is not covered either: at the short sequences of 224-pixel LowFormer inputs, the fused kernel
slowed its training step, for example from 0.20 to 0.37 ms for four heads over 196 tokens at batch 64.
