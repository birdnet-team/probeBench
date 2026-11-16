# AGENTS.md

Guidance for coding agents working in this repository. It is the single source of
truth for agent instructions; `CLAUDE.md` just points here. It holds only what the
code does not already say.

Personal or machine-specific preferences (which virtualenv, where the data lives,
whether an agent may commit) belong in an untracked `CLAUDE.local.md` /
`AGENTS.local.md`, which `.gitignore` covers.

## Setup

`pip install -e ".[dev]"` gets the package, the `birdnet` package (BirdNET V3.0 on
ONNX), pytest and ruff. BirdNET V2.4 needs TensorFlow: `.[birdnet-tf]`.

**Data and models download on first use.** `probebench setup` / `run` fetch the
datasets into the data root (`--data`, else `$PROBEBENCH_DATA`, else `./data`,
which `.gitignore` excludes); the `birdnet` package caches its models under
`$BIRDNET_APP_DATA`. The test suite is offline: it builds synthetic tasks and uses
the mel baseline, so it needs neither.

## Before handing work back

Run `ruff check`, `ruff format --check` and `pytest`. **The ruff version is pinned
in two places** (the `tests` extra and `.github/workflows/ci.yml`); bump both.

## Architecture

- `tasks.py`: the task format (`task.json` + `manifest.parquet`) and loading.
- `datasets/`: one module per source dataset (download locations + `prepare`),
  registered in `datasets/__init__.py`.
- `models/`: adapters (`name`, `sample_rate`, `window_s`, `embed`), registered by
  `@register`; `from_spec` also takes `torchscript:` and `py:` specs.
- `embed.py` (windows, pooling, cache), `probe.py` (the recipe), `report.py`
  (table, PR figure), `cli.py`.

## Conventions

- **The recipe is fixed.** Every published number comes from the same windowing,
  pooling and probe (see README). A change to it invalidates `results/` and needs
  all models rerun; do not add per-model or per-task tuning.
- **Splits are grouped.** A new task groups by recording, session, site or
  individual (`grouped_split`, `per_class_group_split`); random clip-level splits
  leak and inflate scores. Say in the task description what the split is grouped by.
- **Name models by what produced the numbers.** A results file records the model
  file or release; never present numbers of one checkpoint under another's name
  (e.g. the `birdnet` package's "3.0" is not necessarily the V3.0 release used in
  `results/`).
- Published numbers use three decimals and American English.
- No machine-local paths in tracked files; manifests store paths relative to the
  task folder.
- User-facing changes get a `CHANGELOG.md` entry under `[Unreleased]` (Keep a
  Changelog), one or two sentences of cause and effect, citing the PR as a markdown
  link.
- Comments: short and current-state only (a constraint, a non-obvious why); no
  history.
