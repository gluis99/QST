# Quantum State Tomography (QST)

Project work for the SCIQIS course. We reconstruct a quantum state's density
matrix from simulated homodyne (quadrature) measurements at several phases,
and study how reconstruction accuracy depends on the measurement and
algorithm parameters.

The main focus of this project is **maximum-likelihood estimation (MLE)**
via the iterative RρR algorithm, which we implement, validate, and
benchmark in detail. As a secondary point of comparison, we also implement
a **Bayesian mean estimate (BME)** and contrast its behaviour with MLE.

## What's implemented

- **Homodyne data simulation** (`utilities/artificial_samples.py`,
  `utilities/states.py`): generate simulated quadrature samples
  x^(θ) for a chosen quantum state (Fock, cat, squeezed vacuum, ON, and
  photon-subtracted variants) at a set of measurement angles θ.
- **MLE reconstruction** (`utilities/MLE_class.py`): iterative RρR
  algorithm that reconstructs a density matrix ρ from binned quadrature
  data, with configurable Fock-space cutoff, bin count/edges, and
  stopping criteria (fixed iterations, fidelity target, or fidelity-change
  setpoint).
- **Bayesian mean estimate**: reconstruction via posterior sampling /
  averaging over density matrices, compared against MLE on the same
  simulated data. *(in progress)*
- **Benchmarking** (`utilities/benchmarking.py`, `utilities/obtain_data.py`):
  sweep MLE reconstruction fidelity and convergence speed over the number
  of histogram bins, number of measurement angles, and samples per angle,
  across a library of target states (cat, ON, squeezed vacuum) and
  squeezing levels. Results are computed with seeded RNGs, checkpointed to
  HDF5 as they're produced, and resumable if interrupted.
- **Interactive tools** (`Notebooks/interactive.ipynb`,
  `utilities/state_selector_interactive.py`): pick a target state and its
  parameters via widgets, visualize its Wigner function, and run MLE
  reconstruction on simulated data from it.
- **Plotting** (`utilities/plotting.py`): density matrix heatmaps, Wigner
  functions, and fidelity-vs-iteration convergence curves.

## Repository structure

```
Notebooks/
  interactive.ipynb              interactive state selection + MLE reconstruction
  MLE_reconstruction_interactive.ipynb   walkthrough of the MLE algorithm
  benchmark_nb.ipynb              exploration of saved benchmark results
utilities/
  states.py                       target quantum states (Fock, cat, sqv, ON, ...)
  artificial_samples.py           simulated homodyne sampling
  wavefunctions_projectors.py     quadrature wavefunctions and bin projectors
  MLE_class.py                    MLE (RρR) reconstruction algorithm
  benchmarking.py                 parameter sweeps for MLE reconstruction
  obtain_data.py                  script: run full benchmark suite, save to HDF5
  state_selector_interactive.py   ipywidgets UI for interactive.ipynb
  plotting.py                     density matrix / Wigner / fidelity plots
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment and dependencies. Install uv, clone this repository, and run:

```bash
uv sync
```

This creates or updates the project's `.venv` environment using the
dependencies listed in `pyproject.toml` and the locked versions in
`uv.lock`.

### Run in VS Code

The supported notebooks are:

- `Notebooks/interactive.ipynb` for interactive state reconstruction.
- `Notebooks/benchmark_nb.ipynb` for exploring benchmark results.

1. Open one of the supported notebooks in VS Code.
2. Select the Python interpreter from `.venv` when prompted.
3. Select the `QST (.venv)` notebook kernel.
4. Run the notebook cells from top to bottom.

To verify the selected environment from a terminal, run:

```bash
uv run python --version
uv run which python       # Linux and macOS
uv run where python       # Windows
```

## Generate Benchmark Data

The benchmark data-generation script lives with the reusable Python
modules:

```bash
uv run python utilities/obtain_data.py --output benchmark_results_10runs.h5
```

This sweeps MLE reconstruction fidelity and iteration count over bin
counts, angle counts, and samples-per-angle for a library of target states
(cat, ON, squeezed vacuum states). Runs are seeded for reproducibility and
checkpointed after each state, so an interrupted run can be resumed by
re-running the same command with the same `--output` file.

The generated HDF5 result files are kept local and are not committed to
the repository.

## Roadmap

- [x] Simulated homodyne data generation
- [x] MLE (RρR) reconstruction
- [x] MLE benchmarking across bin count, angle count, and sample size
- [ ] Bayesian mean estimate implementation and comparison against MLE
- [ ] Neural-network-based reconstruction (stretch goal, may be dropped
      depending on time)
