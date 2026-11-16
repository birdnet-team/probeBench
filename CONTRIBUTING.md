# Contributing

Thanks for helping. New tasks and model adapters are the most useful
contributions.

## Development setup

```sh
pip install -e ".[dev]"
ruff check && ruff format --check && pytest
```

The tests are offline and take a few seconds.

## Adding a task

1. Add a module under `src/probebench/datasets/` with a `DATASETS` list: where to
   download the data (public URLs or a Zenodo record via `zenodo(...)`) and a
   `prepare(raw_dir, tasks_root)` function that writes the task with `write_task`.
2. List the module in `datasets/__init__.py`.
3. Group the split by recording, session, site or individual, and say so in the
   task description. Keep the license and citation in the task config.
4. Run the mel baseline and at least one learned model on it, and add the numbers
   to the pull request.

Datasets that need a login or a manual license step are welcome too: document the
manual download in the module docstring and have `prepare` read the files from
`<data>/raw/<name>/`.

## Adding a model

Write an adapter (`name`, `sample_rate`, `window_s`, `embed`) as described in the
README. If it should be built in, add a module under `src/probebench/models/`,
register it with `@register("short-name")`, and import it in `_load_builtins`.
Optional dependencies go into a new extra in `pyproject.toml`, imported inside the
adapter so the rest of the package works without them.

## Publishing results

Results in `results/` come from the fixed recipe on all tasks. A pull request that
adds a model's results includes the `results/<model>.json` from `probebench run`
(its `_meta` names the model file or release) and states the hardware-independent
settings used.

## Pull requests

Keep them focused, describe what changed and why, and add a `CHANGELOG.md` entry
for user-facing changes. CI runs ruff and pytest.
