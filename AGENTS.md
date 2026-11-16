# AGENTS.md

Guidance for coding agents working in this repository. It is the single source of
truth for agent instructions; `CLAUDE.md` just points here. It holds only what the
code does not already say.

Personal or machine-specific preferences (which virtualenv, where the data lives,
whether an agent may commit) belong in an untracked `CLAUDE.local.md` /
`AGENTS.local.md`, which `.gitignore` covers.

## Setup

`pip install -e ".[dev]"` gets the package, onnxruntime (BirdNET+ V3.0),
ai-edge-litert (BirdNET V2.4), pytest and ruff. The base install needs neither
runtime; each adapter imports its runtime when it is built.

**Data and models download on first use.** `probebench setup` / `run` fetch the
datasets into the data root (`--data`, else `$PROBEBENCH_DATA`, else `./data`,
which `.gitignore` excludes); the BirdNET adapters fetch their model files from
the Zenodo records into `<data>/models/` (MD5-checked). The test suite is offline:
it builds synthetic tasks and uses the mel baseline, so it needs neither.

## Before handing work back

Run `ruff check`, `ruff format --check` and `pytest`. **The ruff version is pinned
in two places** (the `tests` extra and `.github/workflows/ci.yml`); bump both.

## Architecture

- `tasks.py`: the task format (`task.json` + `manifest.parquet`) and loading.
- `datasets/`: one module per source dataset (download locations + `prepare`),
  registered in `datasets/__init__.py`.
- `models/`: adapters (`name`, `sample_rate`, `window_s`, `embed`), registered by
  `@register` (factories get the models folder); `from_spec` also takes
  `torchscript:` and `py:` specs. `birdnet_zenodo.py` pins the Zenodo record and
  file of each BirdNET model; a model change is a new record, not a new runtime.
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
  file or release (DOI); never present numbers of one checkpoint under another's
  name. Results computed with another build of a model (e.g. a TorchScript or
  SavedModel export) are published under the Zenodo file only after its embeddings
  were shown to match. The embedding cache is keyed on the adapter's `provenance`.
- Published numbers use three decimals and American English.
- No machine-local paths in tracked files; manifests store paths relative to the
  task folder.
- User-facing changes get a `CHANGELOG.md` entry under `[Unreleased]` (Keep a
  Changelog), one or two sentences of cause and effect, citing the PR as a markdown
  link.
- Comments: short and current-state only (a constraint, a non-obvious why); no
  history.
