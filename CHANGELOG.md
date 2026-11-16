# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- First version: 15 tasks from public datasets with download and prepare steps,
  grouped splits, the linear-probe recipe (mean + max pooled window embeddings,
  standardized, class-balanced logistic regression with C chosen on val), adapters
  for BirdNET V3.0 and V2.4 (through the `birdnet` package), TorchScript models and
  a mel baseline, the `probebench` command line, and reference results for
  BirdNET+ V3.0 (12K release), BirdNET V2.4 and the mel baseline.
