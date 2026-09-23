---
license: other
---

# Equimo pretrained models

This repository hosts pretrained artifacts converted for use with
[Equimo](https://github.com/clementpoiret/equimo). The artifacts do not share a
single license: each model family remains subject to its upstream terms. The
Equimo source-code license does not replace or override those model licenses.

> **Check before use.** Upstream licenses can evolve, so the bundled copies may be outdated.
> Before using a pretrained model, follow its **Current license**
> link and check the latest version. If you find an outdated license here,
> please open an [issue or pull request](https://github.com/clementpoiret/equimo/issues).

The checked date for each family is listed below; ConvNeXt entries were added
on **2026-09-05**.
See the repository [NOTICE](NOTICE) for attribution and conversion notices.

<!-- pretrained-license-table:begin -->
| Family | Identifier prefix | Count | License | Bundled snapshot | Model/source | Current license | Checked |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| ast | `ast_` | 2 | BSD 3-Clause | [snapshot](LICENSES/pretrained/ast-BSD-3-Clause.txt) | [AST model](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | [upstream license](https://github.com/YuanGongND/ast/blob/master/LICENSE) | 2026-08-21 |
| convnext | `convnext_` | 40 | Apache 2.0 | [snapshot](LICENSES/pretrained/convnext-Apache-2.0.txt) | [ConvNeXt checkpoints](https://huggingface.co/timm/convnext_atto.d2_in1k) | [upstream license](https://huggingface.co/timm/convnext_atto.d2_in1k) | 2026-09-05 |
| convnextv2 | `convnextv2_` | 26 | Creative Commons Attribution NonCommercial 4.0 International | [snapshot](LICENSES/pretrained/convnextv2-CC-BY-NC-4.0.txt) | [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) | [upstream license](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE) | 2026-09-05 |
| dinov2 | `dinov2_` | 8 | Apache 2.0 | [snapshot](LICENSES/pretrained/dinov2-Apache-2.0.txt) | [DINOv2](https://github.com/facebookresearch/dinov2) | [upstream license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | 2026-08-21 |
| dinov3 | `dinov3_` | 8 | DINOv3 License | [snapshot](LICENSES/pretrained/dinov3-License.md) | [DINOv3](https://github.com/facebookresearch/dinov3) | [upstream license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) | 2026-08-21 |
| eupe | `eupe_` | 6 | FAIR Noncommercial Research License v1 | [snapshot](LICENSES/pretrained/eupe-FAIR-Noncommercial-Research-License.md) | [EUPE](https://github.com/facebookresearch/EUPE) | [upstream license](https://github.com/facebookresearch/EUPE/blob/main/LICENSE.md) | 2026-08-21 |
| siglip2 | `siglip2_` | 15 | Apache 2.0 | [snapshot](LICENSES/pretrained/siglip2-Apache-2.0.txt) | [SigLIP 2 model](https://huggingface.co/google/siglip2-base-patch16-256) | [current license declaration](https://huggingface.co/google/siglip2-base-patch16-256) | 2026-08-21 |
| t0 | `t0_` | 1 | Apache 2.0 | [snapshot](LICENSES/pretrained/t0-alpha-Apache-2.0.txt) | [T0-alpha model](https://huggingface.co/theforecastingcompany/t0-alpha) | [upstream license](https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE) | 2026-08-21 |
| tabpfn | `tabpfn_` | 8 | TABPFN-3 License v1.0 | [snapshot](LICENSES/pretrained/tabpfn-3-License-v1.0.txt) | [TabPFN-3 model](https://huggingface.co/Prior-Labs/tabpfn_3) | [upstream license](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE) | 2026-08-21 |
| tips | `tips_` | 12 | Creative Commons Attribution 4.0 International | [snapshot](LICENSES/pretrained/tips-CC-BY-4.0.txt) | [TIPS](https://github.com/google-deepmind/tips) | [upstream license statement](https://github.com/google-deepmind/tips/blob/main/README.md#license-and-disclaimer) | 2026-08-21 |
<!-- pretrained-license-table:end -->

The family prefixes and counts cover the 126 pretrained identifiers currently
registered by Equimo. Existing checkpoint paths remain under
`models/default/<family>/`.

## LingBot-Vision

The upstream [Small](https://huggingface.co/robbyant/lingbot-vision-vit-small),
[Base](https://huggingface.co/robbyant/lingbot-vision-vit-base),
[Large](https://huggingface.co/robbyant/lingbot-vision-vit-large), and
[Giant](https://huggingface.co/robbyant/lingbot-vision-vit-giant) checkpoint
cards declare Apache 2.0. The [bundled license snapshot](LICENSES/pretrained/lingbot-vision-Apache-2.0.txt)
comes from the upstream [LingBot-Vision source](https://github.com/robbyant/lingbot-vision/blob/151e46321bae4399f8568829f190c7bdec216b49/LICENSE).
Equimo converts the checkpoint parameters to its JAX/Equinox layout and archive
format; the converted archives retain the upstream terms. The [current upstream
license](https://github.com/robbyant/lingbot-vision/blob/main/LICENSE) and model
cards were checked on 2026-09-23.

## ConvNeXt conversion validation

The 40 ConvNeXt V1 and 26 ConvNeXt V2 archives were converted from pinned timm
checkpoints using exact GELU. Each passed float32 feature and classifier-output
comparisons at its native input resolution with seeds 42 and 43, followed by
an exact output comparison after archive reload. Validation requires mean
absolute error below `5e-4` and elementwise `atol=5e-4`, `rtol=1e-4`.
The [conversion report](convnext-conversion.json) records upstream commits,
archive SHA-256 digests, and measured errors for every checkpoint.

## TIPS tokenizer material

`models/tokenizers/sentencepiece_tips.model` is upstream TIPS tokenizer
material and is included in the TIPS attribution and CC BY 4.0 license scope.
The TIPS software license is Apache 2.0, while its checkpoint and other
non-software materials are published under CC BY 4.0.

## Redistribution and use

The files in `LICENSES/pretrained/` are dated snapshots supplied for notice
and redistribution purposes. They are not legal advice and do not replace the
current upstream terms. Some licenses restrict commercial use, require
attribution or notices, or impose conditions on modified materials. Users are
responsible for checking and complying with the terms applicable to the
specific artifacts they use.
