# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

diploSHIC is a deep learning tool for identifying hard and soft selective sweeps in population genomic data. It uses a CNN (convolutional neural network) to classify genomic windows into five categories: hard sweeps, soft sweeps, hard-linked, soft-linked, and neutral.

## Build & Development Commands

diploSHIC uses `uv` for environment management. The project venv lives at `.venv/` in the repo root.

```bash
# Create venv and install in development mode (from repo root)
uv venv --python 3.12
uv pip install -e ".[dev]"

# Activate the venv (or prefix commands with .venv/bin/)
source .venv/bin/activate

# Lint
ruff check diploshic/

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_regression.py::test_import -v
```

The regression test (`test_regression`) is skipped unless `exampleApplication/` test data is present in the repo root.

## Architecture

### CLI Entry Point

`diploshic/diploSHIC` (no `.py` extension) is the main CLI script. It uses argparse subcommands and dispatches to helper scripts via `subprocess`. The five modes are:
- **fvecSim** — generate feature vectors from ms-format simulation output (dispatches to `makeFeatureVecsForSingleMs*.py`)
- **fvecVcf** — generate feature vectors from VCF files (dispatches to `makeFeatureVecsForChrArmFromVcf*.py`)
- **makeTrainingSets** — consolidate fvec files into balanced training sets (dispatches to `makeTrainingSets.py`)
- **train** — train the CNN using TensorFlow/Keras
- **predict** — classify new data with a trained model

The `--diploid` flag on fvecSim/fvecVcf selects between diploid and haploid (ogSHIC) feature vector scripts.

### Core Modules (in `diploshic/`)

- **`fvTools.py`** — Main computation engine. Reads FASTA/ms/VCF data and computes population genetic summary statistics (SFS, LD, diversity, etc.) across sub-windows. Contains BLAS-optimized R² matrix calculations (`fast_r2_for_ld`, `fast_r2_matrix_diploid`, `fast_r2_matrix_haploid`).
- **`numba_stats.py`** — Numba JIT-compiled replacements for former C extensions. Provides `compute_r2_matrix`, `zns`, `omega`, `pairwise_diffs`, `get_haplotype_freq_spec`. Wrapper functions at module level maintain the old C extension API.
- **`msTools.py`** — Parser for Hudson's ms coalescent simulator output format. Supports sequential reading of large files and gzip.
- **`misc.py`** — Utility for confusion matrix display (sklearn-compatible).

### Package Exports

`diploshic/__init__.py` star-imports from `fvTools`, `msTools`, and `numba_stats`, making all functions available at the `diploshic` package level.

### Feature Vector Scripts

The `makeFeatureVecsFor*.py` scripts are the workhorses that compute feature vectors for genomic windows. There are four variants along two axes:
- **Input format**: `SingleMs` (simulation) vs `ChrArmFromVcf` (empirical VCF)
- **Ploidy**: `Diploid` vs `ogSHIC` (haploid/original SHIC)

## Code Style

- Ruff linter with line length 127, targeting Python 3.10+
- Many legacy lint rules are suppressed (star imports, unused variables, etc.) — see `pyproject.toml [tool.ruff.lint]`
- CI runs on Python 3.10, 3.11, 3.12

## Key Technical Details

- The Numba stats module uses `@njit` and `@njit(parallel=True)` decorators. First-call JIT compilation is expected.
- R² (linkage disequilibrium) calculations are performance-critical and use scipy BLAS (`dgemm`) for matrix multiplication where possible.
- Feature vectors (`.fvec` files) are tab-delimited text with a header row. Training/testing data is bundled in `diploshic/training/` and `diploshic/testing/`.
