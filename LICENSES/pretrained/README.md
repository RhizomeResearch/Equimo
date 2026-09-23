# Pretrained-model licenses

Pretrained weights available through Equimo remain subject to their upstream
licenses. This directory bundles the license text associated with every
pretrained model family known to Equimo on the date shown below. These copies
are provided for notice and redistribution purposes; they do not replace the
upstream terms and are not legal advice.

> **Check before use.** Upstream licenses can evolve, so the bundled copies may be outdated.
> Before using a pretrained model, follow its **Current license**
> link and check the latest version. If you find an outdated license here,
> please open an issue or pull request.

The family prefixes and counts cover all identifiers in
`equimo._pretrained.PRETRAINED_ARCHIVE_SHA256`. The checked date records when
the linked upstream terms and bundled snapshot were last reviewed; it does not
guarantee that the terms are still current.

<!-- pretrained-license-table:begin -->
| Family | Identifier prefix | Count | License | Bundled snapshot | Model/source | Current license | Checked |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| ast | `ast_` | 2 | BSD 3-Clause | [snapshot](ast-BSD-3-Clause.txt) | [AST model](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | [upstream license](https://github.com/YuanGongND/ast/blob/master/LICENSE) | 2026-08-21 |
| convnext | `convnext_` | 40 | Apache 2.0 | [snapshot](convnext-Apache-2.0.txt) | [ConvNeXt checkpoints](https://huggingface.co/timm/convnext_atto.d2_in1k) | [upstream license](https://huggingface.co/timm/convnext_atto.d2_in1k) | 2026-09-05 |
| convnextv2 | `convnextv2_` | 26 | Creative Commons Attribution NonCommercial 4.0 International | [snapshot](convnextv2-CC-BY-NC-4.0.txt) | [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) | [upstream license](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE) | 2026-09-05 |
| dinov2 | `dinov2_` | 8 | Apache 2.0 | [snapshot](dinov2-Apache-2.0.txt) | [DINOv2](https://github.com/facebookresearch/dinov2) | [upstream license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | 2026-08-21 |
| dinov3 | `dinov3_` | 8 | DINOv3 License | [snapshot](dinov3-License.md) | [DINOv3](https://github.com/facebookresearch/dinov3) | [upstream license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) | 2026-08-21 |
| eupe | `eupe_` | 6 | FAIR Noncommercial Research License v1 | [snapshot](eupe-FAIR-Noncommercial-Research-License.md) | [EUPE](https://github.com/facebookresearch/EUPE) | [upstream license](https://github.com/facebookresearch/EUPE/blob/main/LICENSE.md) | 2026-08-21 |
| lingbot | `lingbot_` | 4 | Apache 2.0 | [snapshot](lingbot-vision-Apache-2.0.txt) | [LingBot-Vision checkpoints](https://huggingface.co/collections/robbyant/lingbot-vision) | [upstream license](https://github.com/robbyant/lingbot-vision/blob/main/LICENSE) | 2026-09-23 |
| siglip2 | `siglip2_` | 15 | Apache 2.0 | [snapshot](siglip2-Apache-2.0.txt) | [SigLIP 2 model](https://huggingface.co/google/siglip2-base-patch16-256) | [current license declaration](https://huggingface.co/google/siglip2-base-patch16-256) | 2026-08-21 |
| t0 | `t0_` | 1 | Apache 2.0 | [snapshot](t0-alpha-Apache-2.0.txt) | [T0-alpha model](https://huggingface.co/theforecastingcompany/t0-alpha) | [upstream license](https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE) | 2026-08-21 |
| tabpfn | `tabpfn_` | 8 | TABPFN-3 License v1.0 | [snapshot](tabpfn-3-License-v1.0.txt) | [TabPFN-3 model](https://huggingface.co/Prior-Labs/tabpfn_3) | [upstream license](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE) | 2026-08-21 |
| tips | `tips_` | 12 | Creative Commons Attribution 4.0 International | [snapshot](tips-CC-BY-4.0.txt) | [TIPS](https://github.com/google-deepmind/tips) | [upstream license statement](https://github.com/google-deepmind/tips/blob/main/README.md#license-and-disclaimer) | 2026-08-21 |
<!-- pretrained-license-table:end -->

## LingBot-Vision

Equimo provides `lingbot_vits16`, `lingbot_vitb16`, `lingbot_vitl16`, and
`lingbot_vitg16` constructors and a checkpoint converter. The upstream
[Small](https://huggingface.co/robbyant/lingbot-vision-vit-small),
[Base](https://huggingface.co/robbyant/lingbot-vision-vit-base),
[Large](https://huggingface.co/robbyant/lingbot-vision-vit-large), and
[Giant](https://huggingface.co/robbyant/lingbot-vision-vit-giant) model cards
each declare Apache 2.0 for their weights. The [bundled snapshot](lingbot-vision-Apache-2.0.txt)
is the upstream LingBot-Vision [LICENSE](https://github.com/robbyant/lingbot-vision/blob/151e46321bae4399f8568829f190c7bdec216b49/LICENSE)
at the source revision used by `models/lingbot.py`. The [current upstream
license](https://github.com/robbyant/lingbot-vision/blob/main/LICENSE) and model
cards were checked on 2026-09-23.

Equimo converts the upstream parameters to its JAX/Equinox layout and archive
format. The converted archives remain subject to the upstream Apache 2.0 terms.
Check the current upstream license and the relevant model card before use or
redistribution.

For SigLIP 2, the model card declares `apache-2.0`; its bundled legal text was
copied from the Apache-2.0-licensed
[`big_vision` project](https://github.com/google-research/big_vision/blob/main/LICENSE),
which publishes the upstream implementation. TIPS licenses software under
Apache 2.0 but licenses all other materials, including its checkpoint
materials, under CC BY 4.0; this inventory therefore bundles the CC BY 4.0
legal code for TIPS pretrained weights.

ConvNeXt V1 checkpoint licenses are declared by their upstream timm model
cards; the Apache 2.0 snapshot comes from timm. The ConvNeXt V2 snapshot
contains both its MIT software license and its CC BY-NC 4.0 model-weight
license. Converted V2 weights remain subject to the latter. Equimo converted
all 66 checkpoints to the JAX/Equinox parameter layout and archive format.

## Attribution and modification notices

### TabPFN-3

The upstream license requires this notice to accompany a distribution:

> The TABPFN-3 Model is licensed by Prior Labs GmbH under the TABPFN-3
> Non-Commercial License.  
> Copyright © Prior Labs GmbH 2026.  
> THE SERVICES ARE PROVIDED FREE OF CHARGE: COMPANY SHALL NOT BE LIABLE FOR
> DAMAGES RESULTING FROM SLIGHT NEGLIGENCE. LIABILITY FOR GROSS NEGLIGENCE AND
> INTENTIONAL MISCONDUCT REMAINS UNAFFECTED.

Equimo's pretrained TabPFN-3 archives are derivatives: upstream checkpoint
parameters were converted to Equimo's JAX/Equinox parameter layout and archive
format and are used with Equimo's JAX/Equinox implementation. They are not an
official Prior Labs product and are not represented as endorsed, approved, or
validated by Prior Labs.

### TIPS

TIPS checkpoint materials are Copyright 2025 DeepMind Technologies Limited and
are made available under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
Equimo converted the upstream vision and text checkpoint parameters to its
JAX/Equinox parameter layout and archive format. The upstream source is the
[`google-deepmind/tips` project](https://github.com/google-deepmind/tips), which
states that it is not an official Google product. No endorsement by the
upstream authors is implied.

### T0-alpha

Equimo converted the upstream T0-alpha checkpoint to its JAX/Equinox parameter
layout and archive format. The Apache-2.0 snapshot in this directory covers the
pretrained checkpoint; the T0-derived source implementation has separate
attribution in Equimo's source distribution.

### Other converted checkpoints

Equimo-hosted pretrained archives for the other families likewise repackage or
convert upstream parameters for Equimo's JAX/Equinox implementations. Consult
each current license above for all conditions that apply to use,
redistribution, publication, attribution, and modified materials.
