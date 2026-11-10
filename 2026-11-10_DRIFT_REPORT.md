# SAM 3.1 and contract drift report

Compares `docs/designs/sam3.md` and all 26 `EGM.md` contracts with the working copy, line by line. Line numbers refer to
the working copy. Each item ends with the suggested direction of the fix.

Findings marked **[run]** were reproduced by executing the code on CPU. All others come from reading both sides. Nothing
in `2026-11-08_DRIFT_REPORT.md` remains open except part of its reference-citation item, repeated below. All 1,026
`[file.py:symbol]` citations resolve (the three that a plain AST walk misses are a PEP 695 alias and two re-exports the
contract describes correctly), and every test file named by a verification command exists.

Not checked: the appendix's "Reference" column against upstream SAM 3.1 source, Hugging Face contents and digests,
numerical parity, GPU-only paths, jax2onnx beyond the `sam3` case, and whether each verification command passes. The
bodies of the older backbones (ConvNeXt, FasterViT, IFormer, LowFormer, Mlla, MobileNet, PartialFormer, ReduceFormer,
SHViT, AttNet) were compared by signature and search, not body by body.

## docs/designs/sam3.md

- **A required check fails.** Lines 1123, 1171 and 1215 keep `sam3` in the ONNX matrix and require `pytest` to pass.
  **[run]** `tests/test_onnx_export.py -k sam3` fails with
  `TypeError: concatenate() got multiple values for argument 'axis'` inside jax2onnx, and the case has no xfail
  (`tests/test_onnx_export.py:108`). `AGENTS.md` records the failure, the design does not. Fix: decide between an xfail
  with the upstream reference and recording it in the design.
- **Where full-weight parity runs is contradictory.** Lines 92-93 and the table at 560-565 say the scheduled job; lines
  1217-1218 say "local, on the RTX 4080, because the scheduled job has no GPU runner". `.gitlab-ci.yml:98-112` runs the
  whole of `tests/test_sam3_reference.py` on CPU, video included, and `EQUIMO_REQUIRE_REFERENCE_CACHE=1` makes a missing
  archive fail. `docs/testing.md:46-50` says both. Fix: decide.
- **`detect` produces no semantic map.** Lines 369-370 say the head gives mask logits "and a one-channel semantic map".
  `SegmentationHead.__call__` (`detector.py:758-773`) never uses `semantic_proj`, and `DetectorOutput` has no such
  field. Only the module method `semantic_logits` computes it, for tests. Fix: decide between an output field and
  stating it is module-level only.
- **One cross-attention parent breaks the naming rule.** Lines 310-311 name cross-attention parents `cross_attn`.
  `TwoWayTransformer.final_attn` (`interactive.py:332,355`) is a cross-attention that `core/tags.py:216-222` does not
  tag, so `attention.*` selectors skip it in both mask decoders. Fix: decide between adding `final_attn` to the tag set
  and documenting the exception. Renaming would change checkpoint paths.
- **Birth-frame rendering differs on the prompt frame.** Lines 460-463 say an object renders from its detector mask on
  its birth frame, ordered by detection probability. Objects born in the first pass of the prompt frame render from
  `last_masks`, with priority `sigmoid(+/-10)` (`video.py:956-966,1076,1084-1086`), so ties go to the lower ID. Line 466
  also describes the first pass as "nothing hidden", yet it emits nothing. The reference may behave this way, which was
  not confirmed. Fix: decide after checking `build_outputs`.
- **Re-encoding does more than described.** Lines 452-454 say existing rows keep their propagated masks.
  `tracker.py:1136-1144` also replaces reconditioned rows, and `_masks_through_source_size` (`tracker.py:606-618`)
  empties any pixel claimed by another mask of the state before the highest-logit step. Fix: update the design.
- **Resizing is not `jax.image.resize`.** Lines 242-243 and 994-1000 say resizing uses it directly and list only
  `resize_bilinear`. `ops.py:144,177` add custom `resize_bilinear` and `resize_bilinear_antialias`, used by
  `decoding.py`, `tracker.py` and `video.py`; `tracker.py:208` still calls `jax.image.resize`, and the antialiasing test
  sits in `tests/test_sam3.py:1902`. Fix: update the design.
- **Margins are not asserted at every threshold.** Lines 550-551 and 871.
  `models/sam31_reference.py:1118-1131,1181-1187` checks detection probability, detection-track IoM, box centers and
  track probability only. Occlusion IoU (`OCCLUSION_IOU_THRESHOLD` at line 126 is unused), area fraction 0.3, stability
  0.98 and NMS IoM are not checked, and `video-exemplar` sets `assert_margins=False` (line 2027). Fix: decide between
  adding the checks and narrowing the claim.
- **The clip's coverage check is narrower than stated.** Line 870 promises "an occlusion and every decision type but
  occlusion suppression". `DECISION_TYPES` and `CLIP_DECISION_TYPES` (`models/sam31_reference.py:128-138`) omit
  `removed_duplicate`, and nothing asserts an occlusion. Fix: update the design or extend the check.
- **Replay tests differ from the plan.** Lines 1094-1096 plan a policy-only replay of recorded device outputs and a tiny
  network driven by the reference's decisions, both in `tests/test_sam3_video.py`. That file reads no reference fixture.
  The only replay is `tests/test_sam3.py:2133`, which runs Equimo's own policy. Fix: update the design or split the
  test.
- **The exemplar placeholder check never runs in CI.** `tests/test_sam3_reference.py:159-163` skips without `ftfy`,
  which only the `clip` extra provides, and `test:reference-parity` does not install it. Fix: add the extra to the job.
- **The text model ignores `padding_mask`.** Lines 288-289 say modules turn it into a key mask.
  `language/models.py:323-337` accepts it and never reads it. The provenance note records this as intended. Fix: update
  the design.
- **A mask prompt changes interactive selection.** Lines 142-145, 387 and 954 count labels only. `interactive.py:638`
  and `tracker.py:1015` use `(num_labels > 1) | has_mask`. Fix: update the design.
- **Memory attention wording.** Line 505 says projections are called from `_attend_packed_memory`, but `q` and `k` are
  called in `MemoryAttentionBlock.__call__` (`tracker.py:264-270`). Lines 433-434 say position encodings go through a
  projection; they are added unprojected. Fix: update the design.
- **Compilation requirement is incomplete.** Lines 881-882 omit that `propagate` also compiles per distinct
  `min(num_frames, 16)` (`video.py:845-849`). No test varies `source_size`, and no test enforces the 59 MB per bucket at
  line 878. Fix: update the design, and decide on tests.
- **Model dtype exceptions.** Lines 304 and 516-517 do not note that interactive and tracker mask logits are float32
  (`interactive.py:490-500,605-616`). `FramePrediction` is a NumPy dataclass with variable `K` (`video.py:108-121`), not
  the "Equinox modules with fixed shapes" of line 288. Fix: update the design.
- **Reconditioned slots carry the conditioning flag.** Line 935 says the frame "stays non-conditioning". That holds per
  frame, but `video.py:1026` sets `conditioning | add | recondition` per slot and feeds it to memory encoding. Fix:
  distinguish frame from slot.
- **Provenance `measured` values are mostly absent.** Lines 1112-1113 promise them for every new `.npz`; only
  `sam31_image_reference.npz` has them. Fix: decide.
- **Stale status lines.** Line 122-123 still calls the antialiasing fixture future work, but
  `tests/test_vision_ops.py:278` has it. Line 872 says "author's reading" while the provenance note says legal review
  took place. Line 869 stays open and no bfloat16 reference fixture exists; the `test_bfloat16_*` tests compare Equimo
  with Equimo. Fix: update the statuses.
- **Reference citations are still partial.** Line 1209 promises a location for each threshold. `_NMS_IOM` has no
  `det_nms_thresh` citation, and lines 916-919 omit `video_tracking_multiplex_demo.py`, used at lines 940 and 964. Fix:
  update the comments and the appendix.
- **Thresholds are split across modules.** Lines 1052-1054 put them in `video.py`; device thresholds are in
  `tracker.py:61-70`. Fix: update the design.
- **Public surface and behavior the design omits.** `Sam3VideoTracker.decisions` and `.policy`,
  `ConceptPrompt.without_exemplars`, `encode_concept` (also untiered in `docs/stability.md`), `empty_state`,
  `CommitPlan.empty`, `remove_tracker_slot`, `detection_keep`, `associate_tracks`, `decode_object`. Also: `push_frame`
  errors (`video.py:892-897`), an unused `object_id` is accepted and advances the counter (`video.py:247-258`), point
  tracks are not counted against `max_objects` (`video.py:316`, against lines 70 and 394), and
  `ConceptPrompt.create(source_size=)` converts xyxy boxes (`detector.py:79-80,141-157`). Fix: add them or point to the
  contract.
- **Files and fixtures outside the plan.** `tests/data/sam31_video_exemplar_reference.npz` with its harness command and
  test (also absent from the table at lines 557-565), `benchmarks/sam3_image.py`, `benchmarks/sam3_video.py`,
  `AGENTS.md`, `tests/ci_shards.py`, the extra `.gitlab-ci.yml` changes, `finetune/EGM.md`, `finetune/peft/EGM.md`,
  `vision/layers/EGM.md`, and about 189 lines in `tests/test_conversion_cli.py` where line 1080 plans two tuple
  additions. `models/validate_references.py` changed although lines 1113-1114 say it needs no change. Fix: update the
  design.

## src/equimo/vision/models/sam3/EGM.md

- Lines 114-116 say every cross-attention parent is tagged. `final_attn` is not (see the design section), and
  `MemoryEncoder.fuser` holds blocks outside any `blocks` field (`tracker.py:170`).
- Lines 92 and 159-160: spatial keys also contain projected memory (`tracker.py:267-271`), and queries add the projected
  current image (`tracker.py:264-266`).
- Lines 169-170 repeat the design's re-encoding wording. `commit` also starts `stored` from `tracks.mask_logits`
  (`tracker.py:1143`), so the suppression stated at line 166 does not reach re-encoded memory. Compare with the
  reference before deciding.
- Line 73: `PointTrack` scores are the sigmoid of its object-score logit, not "detection probability at birth"
  (`video.py:788,799`).
- Line 128: attention softmax runs in float32 only on the XLA path; cuDNN requires bfloat16 or float16.
- Line 57: `CommitPlan.empty` "changes nothing", but `commit` still writes memory, ring buffers and `last_masks`
  (`tracker.py:1182-1193`).
- Line 176 delegates to the `video.py` docstring, which is wrong twice. Line 12 says "at most 15 frames old", but the
  test is `frame - birth < 15` (line 338) and the design says 14. Line 13's overlap count uses `matches`, which zeroes
  the strongest duplicates (`tracker.py:707-711,733`).
- Omitted behavior: exemplars apply only to the prompt frame (`video.py:867,876,901`); the `CommitPlan` host obligations
  (`tracker.py:407-411`); `Association` semantics (`tracker.py:382-387`); NMS and pre-overlap boxes in `decode_concepts`
  (`decoding.py:146,155,159`); the raises in `push_frame`, `add_object`, prompt maxima, and `dim % 8`.
- Unlisted public members: `commit_point_object`, `commit_point_step`, `FrameObservation` fields, `AssociationPolicy`
  helpers, `iom_matrix`, `iou_matrix`, `boxes_from_masks`, `remove_overlaps`, `MultiplexOutput` fields. It does not say
  that `associate_tracks`, `detection_keep` and `remove_tracker_slot` are not package exports.
- The verification command omits `tests/test_sam3_reference.py`.

## src/equimo/vision/models/EGM.md

- Lines 169-170 contradict line 141: "Unlike `VisionTransformer`" is wrong, because both raise `TypeError` on unknown
  keywords (`vit.py:161-217`, `parcae.py:661-664`).
- Lines 151-153 omit PMT's square power-of-two patch of at least 8 (`pmd.py:390-393`), `num_classes > 0`, and no
  downsampling chunks (`pmt.py:61,73`).
- Line 101: `DEQ.features` also returns a tuple (`deq.py:368,379`).
- Lines 59-60: `intermediate_features` entries 0 and 1 are `stop_gradient` (`parcae.py:1461,1520,1612-1616`).
- Lines 72-74 and 82: `auxiliary` is empty when masked attention is off (`eomt.py:277-281`, `pmd.py:502-504`).
- Omitted: `PartialFormer.features(return_qa=...)`, `PMTFeatures` fields.

## src/equimo/vision/EGM.md

- Lines 90-91 overstate validation: `_validate_logits` guards only three helpers, and `resize_mask_logits` accepts empty
  `Q` (`segmentation.py:45,108,217`).
- Line 115's keyless-inference exceptions omit `VisionParcae` (`models/parcae.py:1480-1486`).
- Lines 103-105 omit `kind` from the compared configuration (`pmt_checkpoint.py:88`).
- Lines 29-38 omit `resize_bilinear_antialias` (`ops.py:177`), and "Both operations" at line 87 is ambiguous with four.

## src/equimo/vision/query_training/EGM.md

- Line 90: `LossTerm.value` divides by 1 on an empty denominator and returns the numerator (`_loss.py:164`), which is
  zero only for terms the library builds. Fix: decide.
- Line 47: "unweighted" terms still apply per-query `no_object_weight` (`_loss.py:419-426`).
- Lines 40, 42, 47 show `config` as required; it defaults to `QueryLossConfig()` (`_loss.py:258,304,372`).

## src/equimo/core/EGM.md

- **Bug.** Lines 74-76 say non-array leaves are kept and `current_params` comes from filtering. **[run]**
  `update_ema_model` raises `TypeError` with `eqx.filter(model, eqx.is_array)` on any module holding `eqx.nn.Dropout`,
  `DropPathAdd` or a callable leaf, because `emamodel.py:11` uses `jax.jit` instead of `eqx.filter_jit`.
  `tests/test_emamodel.py` covers only array and static fields. Fix: update the code and test a module with dropout.
- Line 73 says forward passes run without a key only under static inference, and `AGENTS.md` says `split_for_mode`
  raises otherwise. The convolution blocks call `default_key_for_mode` first and silently use `PRNGKey(0)` in training
  (`convolution.py:125,232,431,507,906`), as `vision/layers/EGM.md:124-125` documents. Fix: decide.
- Line 28: `CANONICAL_TAGS` is not complete (`tags.py` also emits `block.N`, `backbone`, `weight`, `lora`, `adapter`).
  Line 29: `infer_depth` falls back to the stage index. Line 37: "floating array" also admits complex. Line 86: the
  error applies only when `pretrained_variants` is set.
- Omitted: `Path` in `tags.__all__`, and the factory arguments `inference_mode`, `pretrained_identifiers`,
  `pretrained_label`.

## src/equimo/core/layers/EGM.md

- **Bug.** Lines 178-181 allow `key=None` under static inference. **[run]** `WindowedSequence.__call__` raises
  `TypeError` because `generic.py:201` stacks the `None` keys. Fix: update the code.
- Line 183: at `p == 1` both dropout layers return before the key check (`dropout.py:95-96,160-161`).
- Lines 159-170 omit the cuDNN head-width limits that `AGENTS.md` states, and `attention.py:124-128` validates only
  dtype.
- Lines 188-189: `RMSNormGated` multiplies its gate in the input dtype before casting (`norm.py:63-66`).
- Line 17: activation is outside the "all core registries" lookup (`generic.py:28-35`). Lines 21-22: `make_register`,
  `make_get` and `AttentionImplementation` are unmarked. Lines 201-202: the cast back is in the callers
  (`rotary.py:142,163`). Line 90 omits `DINOHead`'s activations. Line 112 omits `split_drop_path`'s `AssertionError`.
- Omitted: unknown constructor keywords are silently swallowed (`attention.py:255,477`, `ffn.py:106,198,295,392`,
  `mamba.py:107`); `AttentionBlock` with `mask` and `ffn_layer="swiglu"` raises `TypeError`; `Residual` reuses one key
  for the module and drop path (`generic.py:127,135`); `LayerResolver` and the `AttentionBlock` constructor.

## src/equimo/core/ops/EGM.md

- Lines 61-62 say the cumulative sums of `A` are promoted to float32. They are computed in `A`'s dtype (`scan.py:166`)
  and only the rounded result is cast (`:169`), which weakens low-precision accuracy. Fix: update the code.
- Lines 47-48: `softmax_cross_entropy_with_logit_targets` returns the common promoted dtype (`probability.py:199,203`).

## src/equimo/core/implicit/EGM.md

- Lines 82 and 87: with float32 parameters and lower-precision input, the output is silently promoted
  (`injectors.py:221-222`, `stabilizers.py:161-162,220-221`). No low-precision test exists. Fix: decide.
- Line 100: `random` draws even under inference (`deq.py:84`), and `DEQBlock` reaches only `mixed` (`deq.py:228`).
- Lines 25-26: `aux["z0"]` is the caller's argument, not the solver's start (`deq.py:246`).
- Omitted: the component constructors, `learnable` (line 80), and `DampedProject`'s group rule (line 84).

## src/equimo/vision/layers/EGM.md

- **Bug.** Lines 122-123 allow a `None` key under inference. **[run]** windowed `SHMA` raises `TypeError`
  (`attention.py:735`), and so does `iformer_m_faster` end to end. `iformer_l_faster` uses the same setting. Fix: update
  the code, and add a windowed case to `tests/cases/model_cases.py`.
- Line 134 says norms run in float32; the q and k norms are computed in the model dtype, then cast
  (`attention.py:1679-1680`). Fix: update the code.
- Line 153: `HWDConv` hard-codes 32 and has no `norm_max_group` (`wavelet.py:204-205`). Line 33: `__all__` holds no
  registry, and lines 33-35 omit `get_conv`, `register_conv`, `update_ghostnet`, `finalize_ghostnet`. Line 124:
  `SingleConvBlock` substitutes a key only when `dropout.p > 0`.
- Line 52: the returned `ct` is from before the joint attention (`attention.py:464-484`). Not compared with upstream.
- Omitted: `BlockChunk` and `WindowedSequence` resolve strings with the core registry, so vision `register_attn` does
  not reach them; `CompositeVisionRoPE.get_factors`; `DoubleConvBlock` is residual only at stride 1;
  `RFAttentionBlock(residual=False)` discards the context branch (`attention.py:1910-1918`).
- Stale docstrings: `downsample.py:42-44` says simple mode ignores the activation but `:98-99` applies it, and
  `wavelet.py:122` names a function that does not exist.

## src/equimo/finetune/EGM.md

- **Bug.** Line 128 says ViT block depth uses execution order across chunks. `continued_ssl.py:80` compares chunk-local
  depth with execution-order depths. **[run]** On `depths=[2, 2]`, `unfreeze_policy="last"` selects no block (it raises
  when nothing else is trainable) and `"first"` unfreezes block 0 of both chunks; `depths=[4]` is correct.
  `recipes.py:367-373` does this properly. Fix: update the code and test a multi-chunk model.
- **Bug.** `extract_subtree` promises "only leaves selected by `target`". **[run]** It also returns `head.weight` and
  `head.bias`, because `train_head` defaults to true (`surgery.py:183-188`). Fix: update the code.
- **Bug.** `task_vector_bundle` is documented as portable (line 79). **[run]** `save_finetune_bundle` raises `TypeError`
  on its module delta. Delta loading also rejects unregistered methods (`serialization.py:1970-1974`). Fix: decide.
- Line 161 overstates merging. `fisher_merge` and `regmean_merge` have no `include_head` and merge head leaves
  (`merging.py:488-579`). Hash checks hold only on `TaskVector` paths, the others compare architecture only with a
  config, and `fisher_merge` never checks.
- Line 149: attention pooling computes `tokens @ query` in the model dtype before `softmax`
  (`feature_extraction.py:1091-1096`). Fix: update the code or narrow the claim.
- Line 109: `recipes.py:70`, `continued_ssl.py:75` and `peft/bitfit.py:43` use path tags only, so the `head_plus_norm()`
  recipe and `mode="head_plus_norm"` can disagree. Real-world impact not enumerated. Fix: update the code.
- Line 104: a `freeze` target that matches nothing raises, and `full_ft_llrd` and `partial_ft_last_k_blocks` default to
  `freeze_patch_embed=True`, so they raise on models with no `patch_embed` path (`masks.py:49-50,146-150`,
  `recipes.py:288,312`). Fix: decide.
- Line 149: `ContrastiveProjectionHead` L2-normalizes by default, and identity and projection heads return features, not
  logits. Line 170: only the logical-ID table and quantization and preprocessing fingerprints are checked on load
  (`serialization.py:1178-1187,1489-1503`). Line 206: `greedy_soup` makes `n` or `2n-1` calls and orders candidates
  differently with a config. Line 158: the error is raised in `_pool_declared_features`, not the cited function.
- Omitted: `method_name` semantics, `surgical` and `partial` defaults (`masks.py:91-106,117,170-171,186-221`);
  `TrainableReport.mergeable` is false for methods that have merge functions, and `adapter_params` omits DoRA and VeRA
  (`inspection.py:225-250`); `extract_features` preprocessing checks and pool heuristics; `ewc_loss` includes buffers;
  `adapter_norm_loss` omits several adapter tags; `DepthAxis`, `TargetKind`, `TrainableMode`, `LeafFilter`,
  `SelectorPredicate`, `CalibrationCollectorKind`, `CovarianceNormalization`.
- `feature_extraction.__all__` lacks `FeatureResult` and `FeatureLevelMetadata`, which the contract lists there.
- Config fields that nothing reads: `WiSEFTConfig.mask`, `UniformSoupConfig.mask`, `TaskVectorConfig.mask`,
  `FisherMergeConfig.require_statistics`, `RegMeanConfig.require_input_covariances`, `HeadPlusNormConfig.norm_types`,
  `bn_stats_policy`. `DeltaCodec.merge_for_save` is never called.

## src/equimo/finetune/peft/EGM.md

- **Bug.** Line 119 says already-wrapped modules of the same type are skipped. The guards cannot fire, because targets
  come from `weight` and `bias` leaves and a wrapper has none (`_common.py:65-80`). **[run]** Applying twice raises
  `TypeError` for LoRA at apply time, `TypeError` for DoRA at forward, and `AttributeError` for IA3, and VeRA silently
  nests (29 to 37 array leaves). Read only: scale/shift raises and `apply_adapters` stacks another adapter. Fix: decide,
  repairing the guards is the likely direction.
- Line 121: only LoRA and RandLoRA zero rows outside selected q/k/v segments, and only with
  `target_kind="projection_segment"` (`lora.py:288,301,598,1840`). IA3 ignores it, and LoRA-FA, AdaLoRA, FourierFT and
  EVA change all rows. Fix: update the contract or reject in code.
- Line 110 says only AdaLoRA rejects dropout, but `apply_lora_fa` and `apply_randlora` silently ignore `config.dropout`.
  `apply_adalora` ignores `fan_in_fan_out`.
- Line 121: `apply_eva_lora` always builds `LoRALinear` (`lora.py:1142`). Lines 131-133: `ScaleShiftWrapper` has no
  dropout and does not forward `None`. Line 135: prefix logits are exactly float32, narrowing float64
  (`prefix.py:172-174`). Line 109: only the LoRA family checks rank. Line 119: `apply_scale_shift` wraps any module with
  `weight`, `bias` or `scale`, and `apply_ia3` and `apply_orthogonal_adapters` take no key.
- Fields nothing reads: `LoRAPlusLabelConfig.label_A`, `label_B`, `label_base`, `VeRAConfig.seed_required`,
  `DoRAConfig.magnitude_axis`, and `train_head` on the IA3, scale/shift, prefix, prompt and AdaptFormer configs.
- Omitted: `SerialAdapterBlock` is built only by delta loading; `PrefixAttention.state` is a non-buffer float leaf
  overwritten each call; `depth="shallow"` keeps the first attention only; line 18's list omits `LoRAPlusLabelConfig`
  and `CalibrationSpec`.

## src/equimo/finetune/vision/EGM.md

- Omitted: `DenseProbe` accepts a non-`LinearHead` head at construction and fails later at `self.head.linear`
  (`vision/dense.py:88`), and user-supplied head widths are validated (`surgery.py:347-367`).

## src/equimo/finetune/audio/EGM.md

- Omitted: `lora_ast` uses the canonical tagger (`audio/recipes.py:91-95`), so `target=("audio.transformer.block",)`
  raises.

## src/equimo/finetune/language/EGM.md

- Omitted: `text.token` is emitted (`language/selectors.py:21`) but is not in `CANONICAL_TAGS`, and several listed
  `text.*` tags are never emitted.

## src/equimo/EGM.md

- Line 73 claims identifiers prevent injection. **[run]** `_validate_identifier("abc\n")` passes, because `$` matches
  before a trailing newline (`registry.py:8`, `serialization.py:92`). Fix: use `fullmatch`.
- Line 162 says modules are unbatched. `T0.__call__` takes flattened samples with `group_ids`, and `T0.predict` takes
  batches (`timeseries/models/t0.py:177-193,499-526`). Fix: add the exception.
- Line 78: member order is only written, not enforced on read (`serialization.py:217-271,314`).
- Line 174: tabular registries are a separate implementation (`tabular/layers/registry.py:12-77`). Fix: update the
  contract or migrate.
- Lines 164-167 omit other key fallbacks (`tabular/layers/blocks.py:48-56`, `tabular/layers/mlp.py:16`,
  `finetune/peft/prompts.py:238-239`).
- Lines 57-65 misstate the consumers of the private modules: `_checkpoint_limits` (`vision/pmt_checkpoint.py:14` imports
  only two names), `_io`, `_legacy_layouts`, `_pretrained`.
- Line 188: only `language/__init__.py` registers anything on import; the `.models` and `.layers` subpackages fill the
  others. Lines 192-193: PMT's `psum` runs on every non-inference call (`vision/models/pmd.py:209-262`).
- Lines 183-185 and 48: `repository=` and `url=` reach arbitrary hosts, including `http://` for tokenizers
  (`language/tokenizers.py:91`). Line 122: the sidecar is rewritten only when missing or stale
  (`serialization.py:500-502`). Line 55: several `utils` helpers have no importers. Lines 16-17: `language` has no
  layers of its own. Lines 146-147 omit `resize_bilinear` and `resize_bilinear_antialias`.
- Omitted: `save_model` accepts `model.v1.tar.lz4`, while loaders require suffixes of exactly `[".tar", ".lz4"]`
  (`serialization.py:376,630,722`), so that name is written but not decompressed on load (read only, decide);
  `inspect_checkpoint` writes `.extracted/` and a lock beside the archive; `{model}` in the URL is the identifier's
  first `_` segment.

## src/equimo/conversion/EGM.md

- Omitted: `expand_torch_tensor` raises `ValueError` for an invalid `pos` (`utils.py:49-52`), expand runs before squeeze
  (`:154-157`), and `source="custom"` still imports `timm` (`:87-94`).

## src/equimo/language/EGM.md

- Omitted: `temperature` is stored but never read (`models.py:142,155,160`), so line 12 implies an effect it does not
  have; `sam31_text(pretrained=True, <override>)` raises; `url` takes precedence over `path` (`tokenizers.py:76-80`);
  `ClipBpeTokenizer(path=)` must be gzip (`:190`).

## src/equimo/audio/EGM.md

- Line 33: the range check runs on the mono mix, so stereo channels of +2.0 and -2.0 pass (`io.py:235-243`). Fix:
  decide.
- Lines 28-32: `__post_init__` does not check several fields, including `snip_edges`, `energy_floor` and
  `preemphasis_coefficient` (`io.py:65-125`). Fix: update the contract or tighten the code.

## src/equimo/audio/models/EGM.md

- Lines 33-34: `pos_embed` is cast and added after concatenation, using the patch embedding's dtype (`ast.py:257-263`).
- Lines 35-36: `cls_patch_mean` includes the CLS token (`utils.py:276`).
- Omitted: the constructor discards `**kwargs` (`ast.py:101`), so a mistyped factory override is ignored.

## src/equimo/audio/layers/EGM.md

- Lines 25-26: a non-2D input raises `ValueError`, not `AssertionError` (`patch.py:78`).
- Lines 14 and 17: the parameter is `module`, not `name_or_cls` (`core/layers/_registry.py:59`).

## src/equimo/tabular/EGM.md

- Lines 22-23: registration is not required, because the constructor accepts classes (`models/tabpfn.py:93-96`,
  `layers/registry.py:67-68`).

## src/equimo/tabular/models/EGM.md

- Omitted: `n_train` must be a concrete Python int (`layers/preprocessing.py:62-68`), although the catalog describes it
  as an int32 scalar input; `**kwargs` on the forward methods are ignored (`tabpfn.py:222,263`).

## src/equimo/tabular/layers/EGM.md

- Line 14: the parameter is `module`, not `name_or_cls`.
- Line 71: with `drop_path > 0` outside inference, `key=None` raises `RuntimeError` (`core/layers/dropout.py:162-165`).
- Omitted: `AttentionDecoder` discards `mlp_ratio` and `act_layer`, so `SoftmaxScaling` always uses `exactgelu`
  (`decoder.py:95-97`, `tabpfn.py:199-200`), and `RegressionDecoder` discards extra keywords (`decoder.py:162-164`).

## src/equimo/timeseries/EGM.md

No drift detected.

## src/equimo/timeseries/models/EGM.md

- Lines 38-40 say fully padded key patches are excluded. Validity comes from each patch's first cell, so any left-padded
  leading patch, even partly padded, is excluded as query and key (`t0.py:413-418,424-440`). Fix: update the contract.

## src/equimo/timeseries/layers/EGM.md

- Line 13: the parameter is `module`, not `name_or_cls` (`registry.py:10`).

## AGENTS.md

Not a requested document; these came up while checking the contracts against its invariants.

- Softmax rule: `vision/segmentation.py:111,226` and `vision/query_training/_loss.py:285` are not token attention yet
  use `jax.nn.softmax`. `WindowedAttention` and `MMSA` scale in the model dtype before casting
  (`vision/layers/attention.py:161-163,1242-1246`).
- Key-splitting rule: a parent key is reused after being split (`vision/layers/attention.py:1502,1509`,
  `vision/layers/convolution.py:233,240`).
