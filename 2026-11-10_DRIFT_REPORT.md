# Design and contract drift report

Compares `docs/designs/sam3.md` and all 26 `EGM.md` contracts with the source at revision 39d197d8. Findings are ordered
by severity within each document, and each one ends with the suggested direction of the fix.

Findings marked **[run]** were reproduced by executing the code on CPU. All others come from reading both sides. All
1,501 `[file.py:symbol]` citations, all relative links, and every test path named in a contract resolve. Thirteen
citations in the audio, language, tabular and timeseries contracts are written from the repository root rather than from
the contract's directory.

Not checked:

- Claims about upstream SAM 3.1 source, checkpoint contents, Hugging Face hosting, and numerical parity.
- GPU-only behavior, including cuDNN kernels and GPU scatter determinism.
- Whether each contract's verification command passes. The test suite was not run.
- Runtime keyless execution of every registered model. Endpoint signatures and key guards were read instead.
- Pre-2.8 checkpoint and hash compatibility against a real older bundle.
- The per-subpackage summaries at `src/equimo/EGM.md:157-171`, which were checked only for the names they mention, and
  the key-handling exceptions at `src/equimo/EGM.md:180-187`, which were checked by search.
- The preset wrappers at `src/equimo/vision/models/vit.py:1359-1677`, which were compared by definition and sampled
  signature, and `ConvNeXt` pretrained membership. `FasterViT` behavior past its `vmap` error could not be exercised.

## docs/designs/sam3.md

- **The `sam3` ONNX export is described as a failing case with no skip or xfail, but the test now skips it on the
  affected jax2onnx releases.** Lines 1178-1179, 1234-1235 and 1280-1282 say it stays in the matrix "with no xfail" and
  that `pytest` passes apart from that case. `tests/test_onnx_export.py:28-34,122-123` skips `mlla`, `partialformer` and
  `sam3` on jax2onnx 0.17.1 and 0.18.0. Fix: update the document.
- **The RoPE section places the axis transpose in the encoder and says `VisionRoPE` does not change, but the transpose
  is a new `VisionRoPE` method.** Lines 342-345 say so, and line 255 of the design disagrees with them.
  `VisionRoPE.width_first_factors` (`src/equimo/vision/layers/posemb.py:1081`) does the exchange, and
  `src/equimo/vision/models/sam3/encoder.py:80,84` and `src/equimo/vision/models/sam3/tracker.py:320` only call it. Fix:
  update the document.
- **The file-by-file plan omits the files that received the generic blocks moved out of the sam3 package.** Lines
  1018-1047 and 1240-1241 list only `attention.py`, `core/layers/__init__.py`, `tags.py`, `vision/ops.py` and their
  tests, and lines 1227-1228 call the `vision/layers/EGM.md` edit independent of SAM 3.1. The blocks now live in
  `core/layers/ffn.py:261`, `core/layers/norm.py:19`, `core/ops/probability.py:101`,
  `vision/layers/posemb.py:72-122,1081`, `vision/layers/convolution.py:41` and `vision/segmentation.py:305-375`. Fix:
  update the document.
- **The parameter breakdown misstates the detector and tracker sizes for the model's actual fields.** **[run]** Lines
  257-259 give detector heads 33M and tracker 20M. `eqx.filter_eval_shape(sam31_multiplex)` gives `detector` 32.15M,
  `interactive` 4.22M and `tracker` 15.41M. 20M only fits if `interactive`, which the image path uses, counts as
  tracker. Fix: update the document.
- **The regeneration command is missing a required argument.** Lines 1182-1184 give
  `uv run python models/generate_model_contract_references.py`, but `models/generate_model_contract_references.py:55`
  requires `--source-revision`, as `docs/testing.md:79-80` shows. Fix: update the document.

## src/equimo/EGM.md

- **The one-line `core` summary leaves out `inverse_sigmoid`, which the latest refactor moved into `core.ops`.** Lines
  154-155 list "float32-safe softmax/cross-entropy and SSM kernels". `src/equimo/core/ops/__init__.py:3` re-exports it.
  Fix: update the document.
- **`SeekableReader` is public in a contracted module and named in no contract.** Line 62 lists the other public names
  of `_checkpoint_limits.py`, but not `SeekableReader`. Fix: update the document.

## src/equimo/conversion/EGM.md

No drift detected.

## src/equimo/core/EGM.md

- **The Overview says the modalities build on all the core subdirectories, but only `vision` uses the Deep Equilibrium
  layers and `timeseries` is left out.** Lines 14-15 say "the vision, language, audio and tabular packages build on all
  of these". `src/equimo/vision/models/deq.py:32` is the only import of `equimo.core.implicit`, and
  `src/equimo/timeseries/models/t0.py:17` also builds on `core`. Fix: update the document.
- **The `ops` summary does not cover `inverse_sigmoid`, which falls outside its "custom JVPs" description.** Lines 72-73
  describe only scan kernels and custom-JVP softmaxes and losses. `src/equimo/core/ops/probability.py:101-108` computes
  `inverse_sigmoid` strictly in float32 with no custom JVP. Fix: update the document.

## src/equimo/core/ops/EGM.md

No drift detected.

## src/equimo/core/implicit/EGM.md

No drift detected.

## src/equimo/core/layers/EGM.md

- **`layer_norm` is not re-exported, but the contract does not mark it, although it says it marks every such name.**
  Lines 21-22 state the convention and lines 111-112 list `layer_norm` unmarked (compare lines 106 and 115).
  `src/equimo/core/layers/__init__.py:1-44,71-79` neither exports nor imports it. Fix: update the document.
- **Any-rank application is described without its helper or that helper's limit.** Lines 91-94 and 111-112 describe
  application over the last axis of any rank, and `_apply.py` is named nowhere. `src/equimo/core/layers/_apply.py:6-13`
  flattens the leading axes and `vmap`s the module on the array alone, so no `key`, `inference` or state can reach it.
  Fix: update the document.

## src/equimo/vision/EGM.md

- **The verification command does not run the dedicated tests for the documented mask operations.** Lines 62-70 document
  the IoM and IoU matrices, the NMS variants, `boxes_from_masks` and `remove_overlaps`, and line 152 lists the tests to
  run. Those functions are tested only in `tests/test_segmentation_masks.py`, `tests/test_sam3.py` and
  `tests/test_sam3_reference.py`, none of which line 152 includes. Fix: update the document.

## src/equimo/vision/query_training/EGM.md

No drift detected.

## src/equimo/vision/layers/EGM.md

- **The key-default invariant names four modules that require `key`, but ten more also require it.** Lines 144-145 name
  `WindowedAttention`, `MMSA`, `PWSEDownsampler` and `PosCNN2D`. `src/equimo/vision/layers/attention.py` also declares a
  required `key` on `HATBlock` (`:366`), `SHMA` (`:707`), `SHMABlock` (`:830`), `SQA` (`:1342`),
  `LinearAngularAttention` (`:1660`), `RFAttention` (`:1801`), `RFAttentionBlock` (`:1907`), `ConvAttention` (`:2015`),
  `ConvAttentionBlock` (`:2120`) and `LowFormerBlock` (`:2223`). Calling any of these with `inference=True` and no key
  raises `TypeError`. Fix: update the document.
- **The `norm_max_group` invariant is false for `SHSA`, whose pre-norm always uses one group.** **[run]** Lines 173-175
  name `HWDConv` as the only exception. `src/equimo/vision/layers/attention.py:528` builds `eqx.nn.GroupNorm(1, pdim)`
  regardless of `norm_max_group`. Fix: update the document.
- **`upsample_twofold` also requires a biased convolution, which the contract does not state.** **[run]** Lines 78-79
  give no precondition. `src/equimo/vision/layers/convolution.py:51` asserts `conv.bias is not None`, so a
  `use_bias=False` convolution raises `AssertionError`. Fix: update the document.

## src/equimo/vision/models/EGM.md

- **`FasterViT` fails on every forward endpoint once it has a third stage, so the documented HAT carrier-token path
  never runs.** **[run]** Lines 107 and 114 list it among backbones with `features`, `intermediate_features` and
  `__call__`. `src/equimo/vision/models/fastervit.py:355` makes stage index 2 and later `hatblock` whatever `hat` is,
  and `fastervit.py:255-259` passes a four-entry `in_axes` to `jax.vmap` with two positional arguments. A four-stage
  model raised `ValueError` from `features` and `__call__`, with `hat` either true or false. Fix: change the code.
- **`SHViT.__call__` raises with the default `norm_layer`.** **[run]** Lines 110-114 list it with no exception.
  `src/equimo/vision/models/shvit.py:309` runs `jax.vmap(self.norm)(x)` on a `[C, H, W]` map, so the norm receives
  `(H, W)` and raises `ValueError`. `features` works. Fix: change the code.
- **The `LowFormer` presets raise at 224 pixels because `att_strides` keeps its stem entry while the other lists drop
  theirs.** **[run]** Lines 107, 114 and 179 describe the presets and list only `widths`, `depths` and `block_types` as
  popped. `src/equimo/vision/models/lowformer.py:202-203` zips the popped `depths` with the unpopped `att_strides`, so
  `[2, 2, 2, 2, 1]` at line 310 gives the last stage stride 2 and never uses the final 1. `lowformer_backbone_b0` raises
  a 7 versus 8 broadcast `TypeError` at 224 and runs at 256. Fix: change the code.
- **`lowformer_backbone_b0..b3` cannot be called with defaults, unlike the thin preset factories the contract
  describes.** **[run]** Lines 9-10 and 107 describe them with no required argument.
  `src/equimo/vision/models/lowformer.py:124` makes `attention_type` required and keyword-only, and the registry at line
  307 does not set it. Fix: decide between adding it to the registry config and documenting the required argument.
- **`ReduceFormer.intermediates` is a public method no contract describes.** Line 114 lists only the three standard
  endpoints. `src/equimo/vision/models/reduceformer.py:217` returns the stem output followed by every stage output. Fix:
  update the document.
- **Three public classes are named in no contract.** `MaskEmbedding` (`eomt.py`), `TokenInitializer` (`fastervit.py`)
  and `BasicBlock` (`shvit.py`) are public top-level classes in contracted modules. Fix: update the document, or make
  them private.

## src/equimo/vision/models/sam3/EGM.md

- **The contract documents `AssociationPolicy` but never says the driver exposes it as `Sam3VideoTracker.policy`.**
  Lines 70-79 list the driver's methods and `decisions`, and lines 86-93 give no way to reach the driver's instance.
  `src/equimo/vision/models/sam3/video.py:850` sets the public attribute, `docs/designs/sam3.md:296` lists it, and
  `tests/test_sam3_video.py:905` reads it. Fix: update the document.
- **Fifteen public building blocks are named in no contract, and fourteen of them are also absent from the design.**
  `GeometryBlock`, `FusionBlock`, `FusionEncoder`, `BoxRelativeBias`, `ConceptDecoder`, `DotProductScoring`,
  `PixelStage` and `SegmentationHead` (`detector.py`), `PyramidLevel`, `FeaturePyramid` and `TriNeck` (`encoder.py`),
  and `TwoWayBlock`, `MaskUpsampler`, `MaskDecoder` and `InteractiveHead` (`interactive.py`) have public names. Only
  `MaskDecoder` appears in the design. Fix: update the document, or make them private.

## src/equimo/finetune/EGM.md

- **`extract_features` with `FeatureSpec(pooling="attention")` returns float32 for bfloat16 features, unlike the sibling
  attention poolers.** **[run]** Lines 190-193 say `AttentionPoolingClassifierHead` casts probabilities back and
  `AttentionPool` returns the feature dtype, and cite `_attention_pool_explicit` with no different output dtype.
  `src/equimo/finetune/feature_extraction.py:1094-1098` computes `weights` in `working_dtype` and returns the sum with
  no cast back. This changed in 7a1c950. Fix: decide between casting back to the feature dtype, which matches
  `AttentionPool` and the `AGENTS.md` softmax rule, and documenting the promotion.
- **A bundle from `task_vector_bundle` that records a logical-ID hash cannot be loaded under any `method`, so the stated
  load failure names the wrong condition.** **[run]** Lines 85-89 say delta loading raises `FineTuneBundleError` for a
  `method` without a registered codec. `src/equimo/finetune/merging.py:652` records the vector's hash, and
  `src/equimo/finetune/serialization.py:218` runs the lineage check before the codec lookup at `:219`. The bundle has no
  entries or target paths, so `serialization.py:1490-1496` raises a logical-ID mismatch for both `method="lora"` and
  `method="task_vector"`. Fix: update the document.
- **The "accepted but currently unused" list omits `BreadcrumbsConfig.require_same_base_hash`.** Lines 225-227 list the
  comparable unused flags. `src/equimo/finetune/merging.py:107` declares it, and `breadcrumbs_task_vector` at
  `merging.py:453-485` never reads it. Fix: update the document.
- **`model_block_depth` and `supported_methods_text` are public names in no contract.** They are in `_blocks.py` and
  `_serialization_codecs.py`. Fix: update the document, or make them private.

## src/equimo/finetune/audio/EGM.md

No drift detected.

## src/equimo/finetune/language/EGM.md

No drift detected.

## src/equimo/finetune/vision/EGM.md

No drift detected.

## src/equimo/finetune/peft/EGM.md

- **`apply_eva_lora` raises `ValueError` on re-application instead of skipping already-wrapped targets.** **[run]**
  Lines 127-128 say the LoRA family skips a target that names or lies inside its own wrapper type.
  `src/equimo/finetune/peft/lora.py:1101` validates activation artifacts for every resolved path before the
  `inside_wrapper` skip at `lora.py:1118`, so a wrapped model's `...qkv.base`, `...qkv.lora_A` and `...qkv.lora_B` paths
  raise `Missing EVA activation artifacts` at `lora.py:2181-2182`. Fix: decide between moving the skip ahead of artifact
  validation and documenting EVA as an exception.
- **The "accepted but currently unused" list omits several config fields with no effect.** Lines 202-203 list three
  groups. These are also inert: `LoRAConfig.weight_layout` (`lora.py:66,1919`), `EVAInitializerConfig.svd` and
  `allocation` (`lora.py:174,176`), `ParallelAdapterConfig.branch` and `fusion` (`adapters.py:68-69`),
  `AdaptFormerConfig.placement` (`adapters.py:77`), and `OrthogonalAdapterConfig.eps`, discarded at `adapters.py:1437`.
  **[run]** Changing `eps` from 1e-6 to 1.0 gave identical output. `PiSSAConfig.svd` and `niter` are only validated and
  recorded (`lora.py:1921-1924,1981-1983`), and `_pissa_init` always runs a full SVD (`lora.py:2000`). Fix: update the
  document.
- **`apply_adapter_fusion` changes which adapters are active, and the contract does not say so.** Lines 70-71 only name
  it. With a single active adapter, `_fusion_active_names` returns every name at the site (`adapters.py:1657-1664`), and
  `adapters.py:838` stores that set as active. Sites match by path parts only (`adapters.py:1289-1300`), so a
  `SerialAdapterBlock` at a block path never matches `after_mlp`, and no matching site raises `ValueError`
  (`adapters.py:823-827`). Fix: update the document.
- **The public `AdaptFormerConfig` presets are undocumented, although the `ScaleShiftConfig` presets are listed.** Line
  65 names the config and line 79 enumerates the scale/shift presets. `adapters.py:86-112` defines `safe_default` and
  `paper_chen2022`, which sets `activation="relu"`, `scale_init=0.1` and `scale_trainable=False`. Fix: update the
  document.
- **Six public helpers in `_common.py` are named in no contract.** `apply_last_axis`, `split_optional_key`,
  `linear_module_path`, `target_linear_paths`, `target_mentions_qkv_segment` and `selected_qkv_segment_names`. The
  module is not re-exported. Fix: update the document, or make them private.

## src/equimo/language/EGM.md

No drift detected.

## src/equimo/audio/EGM.md

No drift detected.

## src/equimo/audio/layers/EGM.md

No drift detected.

## src/equimo/audio/models/EGM.md

No drift detected.

## src/equimo/tabular/EGM.md

No drift detected.

## src/equimo/tabular/layers/EGM.md

No drift detected.

## src/equimo/tabular/models/EGM.md

No drift detected.

## src/equimo/timeseries/EGM.md

No drift detected.

## src/equimo/timeseries/layers/EGM.md

No drift detected.

## src/equimo/timeseries/models/EGM.md

No drift detected.
