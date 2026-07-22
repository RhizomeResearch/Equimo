# Experimental model variant catalog

## Decision

Proceed with a bounded catalog prototype in `equimo.catalog`; do not declare
the API stable until more variant families have been migrated. Catalog keys use
the separate lowercase `<modality>/<variant>` namespace, for example
`vision/dinov2_vits14_reg`. This avoids collisions with the existing model-class
registry. A bare variant name is accepted only when it resolves uniquely.

Descriptors are immutable and contain stable data only. Identity fields are
complete by contract, while `field_status` records whether inputs, pretrained
metadata, provenance, and notes are `complete`, `experimental`, or
`unavailable`. Deterministic listing order is lexical by full catalog key;
duplicate full keys are rejected after lowercase normalization. Duplicate bare
variant names are permitted across modalities but require a full key at query
time.

Each modality owns a private `_catalog_model_variants()` provider beside its
authoritative variant registry. The provider looks up the representative key
on every query and derives input dimensions and checkpoint identifiers from
that registry. The central catalog contains only provider module paths, not a
second constructor manifest. `create_model()` resolves the descriptor's string
constructor path and delegates to the existing public variant factory.

## Authority inventory

| Modality | Variant authority | Constructor source | Pretrained identifier source | Input contract source | Conversion/reference evidence | Current gap |
| --- | --- | --- | --- | --- | --- | --- |
| Vision | `_VIT_REGISTRY` | matching factory and `_build_vit()` in `vision/models/vit.py` | `_build_vit()` uses the variant key; README advertises availability | `in_channels` and `img_size` from merged registry config | `models/torch_models.py`; `dinov2_vits14_reg_reference.npz` entry in the provenance manifest | Only `dinov2_vits14_reg` is catalog-covered |
| Audio | `_AST_REGISTRY` | matching factory and `_build_ast()` in `audio/models/ast.py` | `_AST_PRETRAINED_VARIANTS` | `input_tdim` and `input_fdim` from merged registry config | `models/ast.py`; AudioSet fixture entry in the provenance manifest | Only the AudioSet variant is catalog-covered; raw-waveform preprocessing is out of scope |
| Tabular | `_TABPFN_REGISTRY` | matching factory and `_build_tabpfn()` in `tabular/models/tabpfn.py` | `_TABPFN_PRETRAINED_IDENTIFIERS`, falling back to the variant key | symbolic row/column inputs plus classification bounds from merged config | `models/tabpfn3.py`; default classifier fixture entry in the provenance manifest | Only the default classifier is catalog-covered |

All 23 concrete identifiers in the README pretrained examples map to a private
variant authority, and none collide after lowercase normalization. The README's
EUPE statement is a family-level availability claim rather than an additional
concrete identifier. Twenty concrete legacy identifiers remain intentionally
outside this prototype and are labeled as such in the README.

## API boundary and migration

`list_models()` supports filtering by modality, family, and pretrained
availability. `model_info()` accepts a full key or an unambiguous bare variant,
and unknown names include deterministic suggestions. `create_model()` delegates
to the existing factory and never loads a checkpoint unless the caller passes
`pretrained=True`. Listing and metadata queries instantiate no models, perform
no network access, and import no conversion-only dependencies.

Follow-up migrations should add one modality-owned provider entry at a time,
with a reference/provenance pointer and README drift coverage in the same
change. Once coverage is broad enough to test the descriptor against additional
families and missing-checkpoint cases, maintainers can decide whether to
stabilize this module or revise the contract.

Rejected alternatives:

- Sharing the model-class key namespace: a class key such as `vit` does not
  identify a configuration and can be ambiguous by modality.
- A central hand-written variant manifest: it would duplicate private registry
  keys and drift independently.
- Storing factories or instantiated models in descriptors: callables are not
  serializable, and instances would make discovery expensive and side-effectful.
- Reading conversion scripts or test manifests at import time: those files are
  repository evidence, not installed runtime dependencies.
