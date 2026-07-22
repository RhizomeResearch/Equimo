# Contributing to Equimo

Equimo supports Python 3.12 and newer. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) before setting
up a development checkout.

## Set up the repository

Clone the repository and install the locked development environment:

```bash
git clone https://github.com/clementpoiret/equimo.git
cd equimo
uv sync --locked --group dev
```

This installs Equimo together with pytest, Ruff, ty, and pre-commit. Activate
the repository hooks once per checkout:

```bash
uv run pre-commit install
```

If you use [`devenv`](devenv.nix), `devenv shell` can provide Python 3.12 and
`uv` and sync the environment on entry. It is an optional shell environment;
`uv` and `uv.lock` remain the canonical dependency resolver and lockfile.

## Verify a change

Format changed Python files while iterating:

```bash
uv run ruff format <changed paths>
```

Run the verification ladder from the repository root:

```bash
# Check formatting and linting
uv run ruff format --check src tests examples models
uv run ruff check src tests examples models

# Type-check the package
uv run ty check src

# Run the most relevant test while iterating
uv run pytest tests/path/to/test_file.py::test_name -q

# Run the full test suite
uv run pytest -q -p no:cacheprovider
```

The full test suite took about eight minutes in the audited environment, so
prefer targeted tests during iteration. Before submitting a change, also run
all configured hooks:

```bash
uv run pre-commit run --all-files
```

The formatting, linting, type-checking, and full-suite steps cover the checks
run by [GitLab CI](.gitlab-ci.yml). A successful change has no formatting,
lint, type, or test failures.

## Follow the contributor conventions

The repository's binding implementation and review conventions are in
[`AGENTS.md`](AGENTS.md). In particular, dimension names must describe their
domain: linear widths use `features`, token widths use `dim`, convolutional
widths use `channels`, and image/audio token projections use `embed_dim`.
Modules that change a width use explicit `in_` and `out_` names. Do not add
aliases for old parameter names unless the change explicitly requests them.

Keep changes focused, preserve public signatures and checkpoint behavior, and
add deterministic tests for changed behavior. When fixing a bug, first add or
identify a test that reproduces it.
