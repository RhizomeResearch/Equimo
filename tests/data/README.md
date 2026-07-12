# Numerical reference fixtures

The `.npz` files in this directory are checked-in outputs from pinned upstream
models. Their commands, upstream revisions, input seeds and shapes, locked
library versions, expected schemas, comparison tolerances, and fixture hashes
are recorded in `reference_provenance.json`.

The EUPE fixture retains the original generator's deterministic RNG offset: a
256×256 image was sampled before its 224×224 input. The manifest records this as
`rng_discard_shape`, and `models/torch_models.py` reproduces that sequence.

## Environment

Create the locked maintainer environment from the repository root:

```bash
uv sync --locked --group dev --group reference
```

The reference group covers the tracked generators' published dependencies.
TIPS and EUPE still require their separately cloned upstream source trees; local
paths must never be written into the provenance file.

## Generate and review

Never generate directly into `tests/data`. Use the exact command recorded for
the fixture, replacing documented `/path/to/...` placeholders, and target a
temporary directory such as `/tmp/equimo-references`.

First validate the committed set and its provenance:

```bash
uv run python models/validate_references.py
```

Then compare a complete candidate set:

```bash
uv run python models/validate_references.py \
  --candidate-dir /tmp/equimo-references
```

The comparison checks fixture coverage, SHA-256, keys, shapes, dtypes, maximum
absolute delta, and mean absolute delta. A delta beyond the recorded tolerance
stops the review. Inspect every reported delta and the upstream release notes
before replacing a fixture. Replacement is a deliberate human action; the
validator never writes into this directory.

When a fixture is accepted, update its generation command, revision,
checkpoint checksum (when published), environment versions, schema, tolerance,
and new fixture SHA-256 in `reference_provenance.json` in the same change.
