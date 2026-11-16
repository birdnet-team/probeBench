# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- First version: 15 tasks from public datasets with download and prepare steps,
  grouped splits, the linear-probe recipe (mean + max pooled window embeddings,
  standardized, class-balanced logistic regression with C chosen on val), adapters
  for TorchScript models and a mel baseline, the `probebench` command line, and
  reference results for BirdNET+ V3.0, BirdNET V2.4 and the mel baseline.
- The BirdNET adapters download their model files from Zenodo on first use
  (BirdNET+ V3.0 ONNX, doi:10.5281/zenodo.24162518, run with onnxruntime; BirdNET
  V2.4 TFLite, doi:10.5281/zenodo.15050749, run with ai-edge-litert) instead of
  depending on the `birdnet` package, so the benchmark pins exact model files and
  the base install stays small (extras `v3`, `v24`, `all`). The embedding cache is
  keyed on the model file, so a different checkpoint under the same name is
  embedded again.
- Perch 2.0 adapter (`perch-v2`): Google's TensorFlow SavedModel from Kaggle Models,
  downloaded anonymously on first use (GPU or CPU variant), 5 s windows, 1,536-value
  embedding; extra `perch`. Perch 2.0 results for all 15 tasks are in `results/`.
- Project scaffolding: LICENSE.md with the data and model license scope,
  CITATION.cff, code of conduct, security and acceptable-use policies,
  pre-commit hooks, and the README's license, citation, funding and partner sections.
