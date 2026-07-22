# Equimo Contributor Instructions

## Project Overview

Equimo is a Python 3.12+ research library implementing multimodal models in
JAX and Equinox. Vision is the most complete modality; language, audio, and
tabular modules are also public. Treat numerical behavior, array shapes, public
imports, and checkpoint compatibility as part of the API.

## Repository Map

- `src/equimo/`: installable package, organized into `core`, `vision`,
  `language`, `audio`, `tabular`, `finetune`, `serialization`, and `conversion`.
- `tests/`: pytest suite. `tests/finetune/` covers the fine-tuning subsystem;
  `tests/data/` contains numerical reference artifacts.
- `models/`: maintainer conversion and reference-generation scripts, not the
  installed model package.
- `examples/`: runnable usage and fine-tuning examples.
- `docs/` and `README.md`: user-facing documentation.
- `.gitlab-ci.yml`: authoritative CI commands and supported CI environment.

## Environment and Commands

Use `uv` from the repository root. CI uses Python 3.12.

```bash
uv sync --group dev
```

Run a focused test while iterating:

```bash
uv run pytest tests/path/to/test_file.py -q
uv run pytest tests/path/to/test_file.py::test_name -q
```

Before declaring a code change complete, run the relevant focused tests and the
full CI-equivalent checks:

```bash
uv run ruff check src tests examples models
uv run ruff format --check src tests examples models
uv run ty check src
uv run pytest
```

Apply formatting with `uv run ruff format <changed paths>`. Do not broaden a
change merely to fix unrelated pre-existing failures. If a full check cannot be
run, report exactly what was and was not verified.

## Change Workflow

- Read the relevant implementation, neighboring modules, tests, and public
  exports before editing. Follow existing local patterns.
- Keep changes scoped to the request. Do not combine fixes with unrelated
  refactors, formatting, renames, or cleanup.
- Preserve user changes already present in the worktree.
- For a bug fix, first add or identify a test that fails for the reported
  behavior, then implement the smallest fix that makes it pass.
- Add or update tests for changed behavior. Prefer deterministic, small tensors
  and fixed PRNG keys; assert shapes, dtypes, values, and finiteness as relevant.
- Update public exports, docstrings, examples, and user documentation when a
  public API or documented behavior changes.
- Do not commit, push, publish, create releases, or regenerate large reference
  artifacts unless explicitly requested.

## JAX and Equinox Conventions

- Keep computation JAX-native and compatible with transformations such as
  `jax.jit`, `jax.vmap`, and automatic differentiation. Avoid host-side NumPy
  conversions or Python data-dependent control flow in traced paths.
- Pass PRNG keys explicitly and split them deterministically. Do not introduce
  hidden global randomness.
- Follow the repository's unbatched module convention unless the surrounding
  API says otherwise; use `jax.vmap` for batching.
- Models may later be sharded across multiple devices. Keep model code compatible
  with JAX sharding and avoid assumptions that arrays or parameters live on a
  single device.
- Preserve Equinox PyTree structure and static-field semantics. Do not mutate
  modules or arrays in place.
- Models must work in low precision, including `bfloat16` and potentially lower
  precision formats. Preserve existing dtype behavior; perform numerically
  sensitive operations in sufficient precision with explicit, local casts, then
  return results in the expected dtype instead of silently promoting the model.
- Use the repository registries and modality boundaries instead of duplicating
  shared layers or adding cross-modality imports without a demonstrated need.

## API and Dependency Discipline

- Preserve public signatures and import paths unless the task explicitly calls
  for a breaking change. Do not add compatibility aliases for renamed
  parameters unless explicitly requested.
- Keep optional integrations optional. Guard their imports and provide an
  actionable error when the corresponding extra is missing.
- Add dependencies only when necessary. Declare runtime or optional dependencies
  in `pyproject.toml`, development-only tools in the `dev` dependency group, and
  update `uv.lock` with `uv`; do not hand-edit the lockfile.
- Never add credentials, tokens, private model URLs, or machine-specific absolute
  paths. Keep tests offline and deterministic unless network behavior is the
  explicit subject of the test.

## Dimension Naming

Use axis names that match the operation domain:

- Linear layers and heads use `features`, `in_features`, and `out_features`.
- Token or embedding-width modules use `dim` only when that width is preserved.
- Token or embedding-width modules that change width use `in_dim` and `out_dim`.
- Image/audio-to-token projection modules use `embed_dim` for the token output width.
- Convolution modules use `channels` when the channel count is preserved.
- Convolution modules that can change channel count use `in_channels` and `out_channels`.
- Mixed-domain modules name both sides explicitly, for example `in_channels` to `embed_dim`.

Do not mix input `dim` with `out_channels` in one constructor. Do not add
compatibility aliases for old parameter names unless explicitly requested.
