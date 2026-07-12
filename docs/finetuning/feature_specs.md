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
- `layer_aggregation` accepts `{"method": "last"}`, `"mean"`, or `"concat"`.
  It applies to a non-empty tuple/list returned by the endpoint before token
  selection and pooling. Concatenation uses the resolved feature axis.
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

Third-party integrations without a stable contract can omit `feature_spec` and
continue to use the compatibility heuristics. Once they publish a spec, invalid
declarations are not silently redirected to that fallback.

## Serialization

`FineTuneBundle.feature_spec` uses a versioned, strict codec. Pass
`feature_spec=` to `save_delta` to store the executable contract and bind its
preprocessing fingerprint into bundle lineage. Unknown codec versions, fields,
layouts, selections, pools, normalizations, and aggregation values are rejected
on load.
