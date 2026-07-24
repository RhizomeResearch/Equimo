# Equimo stability policy

Equimo follows Semantic Versioning beginning with 2.0.0. A stable release means
the library contract is versioned; it does not imply that every research model
is suitable for production use.

## Stable public surface

The following are covered by the v2 compatibility contract:

- documented, non-underscore exports from `equimo.core`, `equimo.vision`,
  `equimo.language`, `equimo.audio`, `equimo.tabular`, `equimo.finetune`,
  `equimo.serialization`, and `equimo.conversion`;
- public constructor and call signatures, including keyword names;
- documented tensor shapes, axis conventions, output dtypes, explicit PRNG-key
  behavior, and Equinox PyTree structure;
- model-checkpoint format version 1, fine-tuning bundle schema version 1, and
  calibration format version 1;
- existing model archives uploaded for the Equimo 2.0 alpha series.

Backward-incompatible changes to that surface require a new major release.
Features may be deprecated in a minor release and removed in the next major
release. Correctness, security, and data-loss fixes may change erroneous
behavior in a minor or patch release and will be called out in the changelog.

## Explicitly experimental surface

`equimo.catalog` is a bounded prototype and remains experimental until its
coverage and descriptor contract are declared stable. Its module and catalog
documentation identify this exception. Private names beginning with an
underscore, maintainer scripts under `models/`, tests, generated reference
artifacts, and repository-internal registries are not public API.

Optional integrations remain optional. Their public Equimo wrappers are stable,
but the supported Python range may be narrower when an upstream dependency does
not publish wheels. Equimo documents and tests those exceptions explicitly.

## Checkpoint compatibility and integrity

New model archives contain a format/version marker, model class and structural
signature, and SHA-256 weight digest. Readers validate these fields before
deserializing. Fine-tuning and calibration archives likewise validate their
manifest and serialized-array digest. All archive readers enforce member and
size limits.

Compressed model saves are byte-identical for the same model, metadata, and
Equimo/JAX/Equinox versions. The complete archive digest identifies packaging;
the embedded weight digest identifies Equinox's serialized parameter stream.
`inspect_checkpoint` validates local archives or directories without
deserializing weights. Its `verified` flag covers versioned metadata and
parameter-stream integrity; model compatibility is additionally checked when a
model is supplied.

Schema-less model and fine-tuning archives created during the 2.0 alpha series
remain readable throughout the v2 release line. Their internal checksum cannot
be reconstructed, so readers identify that compatibility path. Public
inspection rejects them by default and returns an explicitly unverified result
only when `allow_legacy=True`; `load_weights` continues to read them with a
warning. Equimo v1 archives and the removed v1 `load_model` API are outside this
contract.

Each Equimo release pins its built-in Hugging Face model repository to an
immutable revision and embeds the Git LFS SHA-256 of every uploaded archive.
An explicit archive SHA-256 can also be supplied to `load_weights` for
deployments that maintain their own digest manifest.

## Supported environments

Core CI runs the complete suite on Python 3.12, 3.13, and 3.14, plus a
lowest-direct-dependency environment on Python 3.12. Optional extras are tested
on every supported Python version with upstream binary wheels. The authoritative
commands and job matrix are in `.gitlab-ci.yml`.
