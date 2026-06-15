# runDRO Supplementary Package

## Purpose

This package provides one reviewer-facing runner for the runDRO paper-table experiments. It rebuilds likelihood PCs, creates likelihood-PC adversarial splits, runs Ours and DRO-SPN, creates random perturbation splits, and writes table-facing results to CSV.

## Prerequisites

- Python >= 3.10; Python >= 3.11 is recommended when available
- C++17 compiler
- Cython >= 3.0
- numpy
- scipy
- Torch
- PyJuice
- matplotlib
- Gurobi / gurobipy
- valid Gurobi license

## Installation

Install the local `fastcircuits` package from this directory:

```bash
python -m pip install -e .
```

PyJuice, Torch, and Gurobi/gurobipy are external prerequisites. PyJuice's latest PyPI release may not be usable for this pipeline; use a project environment with a verified working PyJuice version. No exact PyJuice pin is asserted here.

## DEBD Layout

The runner expects package-relative DEBD files:

```text
DEBD/datasets/<id>/<id>.train.data
DEBD/datasets/<id>/<id>.valid.data
DEBD/datasets/<id>/<id>.test.data
```

It runs the seven paper datasets: `nltcs`, `msnbc`, `plants`, `bnetflix`, `dna`, `tmovie`, and `bbc`.

## One-Command Execution

Start from a package copy without preexisting experiment output directories, then run:

```bash
python run_all_experiments.py \
  --ours-iters <owner-confirmed-iters> \
  --output-csv outputs/results.csv \
  --output-latex outputs/table.txt
```

The runner defaults to `h = 1 3 5`, `eta_theta = 1e-3`, `eta_phi = 1e-3`, and `theta_num_samples = 2000`.

## Smoke Check

```bash
python run_all_experiments.py --smoke
```

`--smoke` only validates package composition, package-relative paths, and CSV generation. It is not paper-result reproduction.

## CSV Output

The default CSV path is:

```text
outputs/results.csv
```

The schema is:

```text
dataset,h,method,T,T_a,T_r
```

## Runtime Caveat

The full run trains all methods for seven datasets and three h values. Runtime depends on hardware, PyJuice, Gurobi, and dataset size.

## sample-DRO Caveat

Ours/sample-DRO uses stochastic sampling by default because `THETA_SEED=None` in the upstream implementation. Exact paper-value reproduction may require owner-confirmed `--ours-iters` and any other final settings.
