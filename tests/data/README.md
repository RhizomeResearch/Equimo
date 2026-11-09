# Numerical reference fixtures

The `.npz` files in this directory are checked-in outputs from pinned upstream models. Their commands, upstream
revisions, input seeds and shapes, locked library versions, expected schemas, comparison tolerances, and fixture hashes
are recorded in `reference_provenance.json`.

`model_output_references.json` is a separate, small Equimo-only regression baseline for the tiny production model-family
matrix. It detects unintended local numerical changes; it is not evidence of PyTorch/upstream parity and is therefore
never listed in `reference_provenance.json`.

The EUPE fixture retains the original generator's deterministic RNG offset: a 256×256 image was sampled before its
224×224 input. The manifest records this as `rng_discard_shape`, and `models/torch_models.py` reproduces that sequence.

## Environment

Create the locked maintainer environment from the repository root:

```bash
uv sync --locked --group dev --group reference --extra audio
```

The reference group covers the tracked generators' published dependencies. The `audio` extra supplies the exact
TorchAudio filter-bank backend used by the pinned AST preprocessing references. Generating the T0-alpha reference
additionally requires accepting the model's Hugging Face access terms and authenticating with `hf auth login` or
`HF_TOKEN`. TIPS and EUPE still require their separately cloned upstream source trees; local paths must never be written
into the provenance file.

## Generate and review

Never generate directly into `tests/data`. Use the exact command recorded for the fixture, replacing documented
`/path/to/...` placeholders, and target a temporary directory such as `/tmp/equimo-references`.

First validate the committed set and its provenance:

```bash
uv run python models/validate_references.py
```

Then compare a complete candidate set:

```bash
uv run python models/validate_references.py \
  --candidate-dir /tmp/equimo-references
```

The comparison checks fixture coverage, SHA-256, keys, shapes, dtypes, maximum absolute delta, and mean absolute delta.
A delta beyond the recorded tolerance stops the review. Inspect every reported delta and the upstream release notes
before replacing a fixture. Replacement is a deliberate human action; the validator never writes into this directory.

When a fixture is accepted, update its generation command, revision, checkpoint checksum (when published), environment
versions, schema, tolerance, and new fixture SHA-256 in `reference_provenance.json` in the same change.

## SAM 3.1

`models/sam31_reference.py` generates the SAM 3.1 fixtures on a CUDA GPU, because the reference hard-codes CUDA in
several constructors and forward passes. It runs in its own environment, separate from `uv.lock` because the reference
requires NumPy below 2. Create it and clone the pinned reference outside any synced path:

```bash
uv venv --python 3.12 /path/to/environment
uv pip install --python /path/to/environment/bin/python -r models/sam31_reference_requirements.txt
git clone https://github.com/facebookresearch/sam3 /path/to/sam3
git -C /path/to/sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
```

Generate into a staging directory, not `tests/data`:

```bash
/path/to/environment/bin/python models/sam31_reference.py tokenizer --reference-dir /path/to/sam3 --output-dir /path/to/output
/path/to/environment/bin/python models/sam31_reference.py operators --output-dir /path/to/output
/path/to/environment/bin/python models/sam31_reference.py tiny --reference-dir /path/to/sam3 --output-dir /path/to/output
```

Each fixture must regenerate byte-identically. The `operators` and `tiny` commands also write
`provenance_fragment.json`, the entries to copy into `reference_provenance.json`. The tokenizer command needs no GPU.

The tiny clip's decision trace is in `sam3_tiny_reference.json`. Its detection and track masks are a fixed script,
because random weights produce nothing the association policy would act on; the real detector and tracker still run and
are recorded under `inputs.*` and `outputs.*`.
