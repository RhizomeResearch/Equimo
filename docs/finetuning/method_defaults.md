# Method Defaults

| Method | Default |
|---|---|
| Linear head | trunc-normal 0.02 weight init, zero bias |
| MLP head | 2 layers, GELU, hidden dim equals input dim |
| Linear probe | frozen backbone, trainable head |
| Full FT LLRD | decay 0.75, metadata only |
| LoRA | rank 8, alpha 16, zero-B identity init |
| rsLoRA | alpha over square-root rank scaling |
| PiSSA | rank 16, truncated SVD LoRA initialization |
| LoRA+ | `lora_A` and `lora_B` labels only |
| static AdaLoRA | rank mask metadata, external scheduling |
| DoRA | rank 8, alpha 16, base weight norm magnitude |
| VeRA | rank 256, frozen random bases, trainable scale vectors |
| BitFit | bias leaves plus head, mask only |
| Adapter | reduction factor 16, after-MLP placement, zero-up identity init |
| AdaptFormer | bottleneck 64, parallel branch, zero-up init |
| Parallel adapters | residual-sum side branch |
| AdapterFusion | attention fusion over named adapter banks |
| Prompt tuning (`PromptConfig`) | 10 tokens, shallow, std 0.02 |
| Soft prompts | 20 text prompt tokens, shallow |
| Deep prompts | 10 prompt tokens per layer |
| Prefix tuning | 16 prefix tokens, deep, projected K/V prefix state |
| Scale/shift | scale 1, shift 0 |
| IA3 | scaling vector 1 |
| Continued SSL | LoRA r8/a16 plus last-block unfreeze metadata |
| L2-SP (`L2SPConfig`) | alpha 1e-3, beta 0, sum reduction, half-scaled paper objective |
| Feature distillation | MSE over 50/100 percent layers |
| Mixout | p 0.1 anchored to pretrained weights |
| EWC | diagonal Fisher penalty from supplied statistics |
| WiSE-FT | alpha 0.5, head excluded by default |
| TIES | density 0.20, disjoint mean merge |
| DARE | drop rate 0.90 with rescaling |
| Model breadcrumbs | drop bottom 5 percent and top 1 percent deltas |
| Fisher merge | diagonal Fisher-weighted averaging |
| RegMean | ridge 1e-5, external input covariances |
| SAM/ASAM | metadata only, optimizer remains external |

## Method fidelity notes

Default-config fidelity relative to the reference implementations cited in
[references](references.md). "paper-exact" means the default configuration
reproduces the paper's method; "reference implementation" means it follows the
authors' released code; "safe default" means Equimo deliberately deviates for
identity-safe initialization or robustness.

| Method | Fidelity of defaults | Primary references | Notable default deviation |
|---|---|---|---|
| AdaptFormer | paper-exact | Chen et al. 2022 | none with `AdaptFormerConfig.paper_chen2022()` |
| Bottleneck adapters (Houlsby) | safe default | Houlsby et al. 2019; adapter-bert | Kaiming down / zero up init instead of small truncated-normal for both projections |
| AdapterFusion | reference implementation | Pfeiffer et al. 2021 | none |
| VPT (deep and shallow) | reference implementation | Jia et al. 2022 | none |
| Soft prompts | safe default | Lester et al. 2021 | shallow input-prepended prompts |
| P-tuning v2 | safe default | Liu et al. 2022 | none for default depth="all" |
| Prefix tuning | reference implementation | Li and Liang 2021 | projected K/V prefix state |
| IA3 | reference implementation | Liu et al. 2022 (T-Few) | none |
| SSF (scale/shift) | reference implementation | Lian et al. 2022 | none |
| LoRA | safe default (`lora.equimo_default`) / reference for Q,V-only targeting | Hu et al. 2021 | default targets qkv+proj rather than the paper's Q,V-only |
| rsLoRA | reference implementation | Kalajdzievski 2023 | alpha over sqrt(rank) scaling |
| PiSSA | reference implementation with full SVD; experimental with randomized SVD | Meng et al. 2024 | exactness requires `svd="full"`, `niter=0` |
| LoRA-FA | reference implementation | Zhang et al. 2026 (v3, corrected B-gradient) | frozen-A custom VJP |
| VeRA | safe default | Kopiczko et al. 2024 | shape-compatible frozen random bases; exactness requires a pinned seed |
| DoRA | paper-exact (`paper_equation`) / reference (NVlabs) | Liu et al. 2024 | magnitude taken over the merged weight norm |
| EVA initializer | reference implementation | Paischer et al. 2024 | requires activation-SVD calibration artifacts |
| FourierFT | reference implementation | Gao et al. 2024 | exactness requires an explicit frequency seed |
| OFT | reference implementation | Qiu et al. 2023 | Cayley parameterization |
| BOFT | reference implementation | Liu et al. 2024 | butterfly-factored Cayley; requires explicit block size |
| RandLoRA | reference implementation | Albert et al. 2025 | serialized frozen random bases |
