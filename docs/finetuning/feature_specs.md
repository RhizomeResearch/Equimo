# Executable feature specifications

`FeatureSpec` makes feature extraction reproducible instead of relying on model
class names and inferred pooling. Pass it to `extract_features`,
`FeatureExtractor`, `LinearProbe`, or `make_linear_probe`:

```python
spec = eqft.FeatureSpec(
    endpoint="features",
    output_layout="BTC",
    token_selection="all",
    pooling="mean_token",
    mask_field="padding_mask",
    preprocessing_fingerprint="sha256:...",
)

features = eqft.extract_features(
    model,
    token_ids,
    padding_mask,
    feature_spec=spec,
    observed_preprocessing_fingerprint="sha256:...",
)
```

When a spec is present, it controls endpoint traversal, selection, pooling,
normalization, and aggregation. `pool` may be left at its default `"auto"` or
set to the same policy as the spec; a contradictory value is rejected. When no
spec is present, the existing native/heuristic route remains available for
third-party models.

## Field contract

- `endpoint` is an exact callable path. `features`, `forward_features`, and
  `__call__` select the corresponding native endpoint. Dotted attributes and
  integer tuple/list components are supported for directly callable layer
  paths. Missing or non-callable paths are errors; there is no explicit-spec
  endpoint inference.
- `output_layout` describes the endpoint array. Batched ranks and Equimo's
  corresponding one-example ranks are both accepted: `BNC`/`BTC` use sequence
  axis `N`/`T` and final feature axis `C`; `BCT` uses channel axis `C` and final
  time axis `T`; `BCHW` uses channel axis `C` and spatial axes `H,W`; `BC` is an
  already-read-out feature matrix. Rank and mask shapes are validated.
- `token_selection` supports `all`, `cls`, `patches`, `frames`, and
  `last_valid`. `cls` and `last_valid` produce a feature vector and therefore
  cannot be followed by another pool. Portable `custom` selection is rejected
  because the schema has no callback identity to serialize.
- `pooling` supports `none`, `native`, `cls`, `cls_patch_mean`, `global_avg`,
  `mean_token`, `mean_patch`, `mean_frame`, `attention`, `gem`, and
  `last_token`. `native` requires `model.global_pool` and executes that declared
  readout on native normalized feature dictionaries when available.
- `mask_field` names one endpoint argument. Its polarity is fixed: zero means
  valid and nonzero means padding/excluded. Its shape must equal the feature
  tensor with the feature axis removed. Masked mean and last-valid operations
  return a zero feature vector for an all-padding example, including under JIT.
- `exclude_prompt_tokens` controls whether patch selection and patch/native
  aggregate pooling omit declared prompt tokens. Base prefix/register tokens
  are always excluded from patch reductions. For non-patch operations the flag
  is retained as contract metadata but does not alter the result.
- `normalize` applies `l2` or `standardize` only across the resolved feature
  axis. Sensitive arithmetic uses float32 locally and returns the endpoint
  dtype.
- `layer_aggregation` accepts `{"method": "last"}`, `"mean"`, `"concat"`, or
  `"separate"`.
  It applies to a non-empty tuple/list returned by the endpoint before token
  selection and pooling. Concatenation uses the resolved feature axis.
- `endpoint_options` serializes intermediate endpoint arguments. The portable
  options are `indices`, `n_last_blocks`, and `apply_norm`; layer selectors are
  mutually exclusive. A runtime argument cannot override a serialized option.
- `layer_aggregation={"method": "separate"}` applies selection, pooling, and
  feature normalization independently to every retained endpoint level.
- `return_metadata=True` returns a `FeatureResult`. Its `features` member holds
  the usual array or tuple of arrays, and `levels` records immutable layer,
  width, prefix-token, normalization, positional, input, padding, and patch-grid
  metadata. The default remains the existing array return.
- `preprocessing_fingerprint`, when present, must match either
  `observed_preprocessing_fingerprint=` at extraction time or a
  `model.preprocessing_fingerprint` attribute. Missing and mismatched observed
  values are errors.

`BCHW` supports only `all` with no pooling, `global_avg`, or `gem`. `BC`
supports only `all` with no pooling. Other contradictory selection, pooling,
mask, and aggregation combinations fail when the spec is constructed.

## Built-in and third-party integrations

Built-in integrations should publish an exact endpoint and layout. Transformer
vision and AST readouts can use `endpoint="forward_features"`, layout `BNC`,
and `pooling="native"`; padded language encoders should name their padding-mask
argument; already-read-out tabular predictions should use layout `BC` with no
pooling rather than treating rows as text tokens.

Every built-in registered model family has a characterized explicit contract:

| Registry family | Endpoint | Layout | Selection and pooling |
| --- | --- | --- | --- |
| `vit`, `vision_parcae` | `forward_features` | `BNC` | `cls` with no pool, or `patches` with `mean_patch` |
| `ast` | `forward_features` | `BNC` | `frames` with `mean_frame`, or `all` with `native` |
| `fastervit`, `partialformer`, `mlla`, `vssd` | `features` | `BNC` | `all` with `global_avg` |
| `attnet`, `convnext`, `iformer`, `lowformer`, `mobilenetv3`, `reduceformer`, `shvit` | `features` | `BCHW` | `all` with `global_avg` |
| `text_transformer_encoder` | `features` | `BTC` | `all` with `mean_token` and `mask_field="padding_mask"` |
| `tabpfn` | `__call__` | `BC` | `all` with no pooling |
| `deq` | `intermediate_features` | `BCHW` | `all` with `global_avg` and `layer_aggregation={"method": "last"}` |

The DEQ contract passes `n_last_blocks=1` to extraction because its
`features` endpoint intentionally returns both the feature map and solver
diagnostics. TabPFN callers must keep `n_train` static when JIT-compiling, just
as they must for a direct model call. Named sizes and pretrained variants share
their registered family implementation; configurations that remove a class
token, such as SigLIP-style ViTs, support patch selection but reject `cls`
selection explicitly.

Spatial token models that support metadata publish
`feature_metadata(*args, endpoint=..., endpoint_options=...)`. The hook must
return exact input, patch, padding, grid, prefix ordering, endpoint
normalization, and positional configuration. Token extraction rejects a
spatial metadata request when this contract is absent. Native `BCHW` endpoints
derive their grid directly from the declared spatial axes. A patch grid is
always row-major and is never recovered from the square root of a token count.

Third-party integrations without a stable contract can omit `feature_spec` and
continue to use the compatibility heuristics. Once they publish a spec, invalid
declarations are not silently redirected to that fallback.

For a base DINOv2 ViT-S model, the normalized class-token and normalized
patch-mean contracts can be expressed without consumer-side dictionary
traversal or pooling:

```python
final_cls = eqft.FeatureSpec(
    endpoint="forward_features",
    output_layout="BNC",
    token_selection="cls",
    pooling=None,
    normalize="none",
    preprocessing_fingerprint=preprocessing_id,
)
final_patch_mean = eqft.FeatureSpec(
    endpoint="forward_features",
    output_layout="BNC",
    token_selection="patches",
    pooling="mean_patch",
    normalize="l2",
    exclude_prompt_tokens=True,
    preprocessing_fingerprint=preprocessing_id,
)
```

For DINOv3-S/16 classification, the explicit default recipe uses the final
normalized class token and therefore returns 384 features. Pooling stays a
caller-visible choice across models:

```python
final_cls = eqft.FeatureSpec(
    "forward_features", "BNC", "cls", None
)
final_patches = eqft.FeatureSpec(
    "forward_features", "BNC", "patches", None, return_metadata=True
)
patch_mean = eqft.FeatureSpec(
    "forward_features", "BNC", "patches", "mean_patch"
)
cls_and_patch_mean = eqft.FeatureSpec(
    "forward_features", "BNC", "all", "cls_patch_mean"
)
intermediate_levels = eqft.FeatureSpec(
    "intermediate_features",
    "BNC",
    "all",
    None,
    layer_aggregation={"method": "separate"},
    endpoint_options={"indices": (2, 5, 8, 11), "apply_norm": True},
    return_metadata=True,
)
```

`apply_norm=True` applies the encoder's final normalization separately at each
selected intermediate level. `normalize="l2"` or `"standardize"` remains a
subsequent feature operation, and both stages are recorded independently.

`forward_features` publishes normalized patch tokens after removing the base
class/register prefix. Prompt-tuned wrappers expose their own endpoint contract;
the example above characterizes the unwrapped DINOv2 endpoint.

## Serialization

`FineTuneBundle.feature_spec` uses a versioned, strict codec. Version 2 stores
endpoint options and metadata-return behavior; version 1 remains readable with
their disabled defaults. Pass
`feature_spec=` to `save_delta` to store the executable contract and bind its
preprocessing fingerprint into bundle lineage. Unknown codec versions, fields,
layouts, selections, pools, normalizations, and aggregation values are rejected
on load.
