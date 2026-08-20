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
- **Bayesian mean estimate**: reconstruction via Metropolis-Hastings
  posterior sampling over density matrices, compared against MLE on the
  same simulated data. *(`Notebooks/BME_homodyne.ipynb`)*
- **Benchmarking** (`utilities/benchmarking.py`, `utilities/obtain_data.py`):
  sweep MLE reconstruction fidelity and convergence speed over the number
  of histogram bins, number of measurement angles, and samples per angle,
  across a library of target states (cat, ON, squeezed vacuum) and
  squeezing levels. Results are computed with seeded RNGs, checkpointed to
  HDF5 as they're produced, and resumable if interrupted.
- **Interactive tools** (`Notebooks/interactive.ipynb`,
  `utilities/state_selector_interactive.py`,
  `utilities/algorithm_interactive.py`): pick a target state and its
  parameters via widgets and preview its Wigner function, then run MLE
  reconstruction on simulated data from it, with live plots of the
  homodyne samples, the convergence, and the reconstructed Wigner
  function.
- **Plotting** (`utilities/plotting.py`): density matrix heatmaps, Wigner
  functions, fidelity-vs-iteration convergence curves, and homodyne
  sample histograms.

## Repository structure

```
Notebooks/
  interactive.ipynb              interactive state selection + MLE reconstruction
  MLE_reconstruction_interactive.ipynb   walkthrough of the MLE algorithm
  benchmark_nb.ipynb              exploration of saved benchmark results
  BME_homodyne.ipynb              draft: Bayesian mean estimation via Metropolis-Hastings
utilities/
  states.py                       target quantum states (Fock, cat, sqv, ON, ...)
  artificial_samples.py           simulated homodyne sampling
  wavefunctions_projectors.py     quadrature wavefunctions and bin projectors
  MLE_class.py                    MLE (RρR) reconstruction algorithm
  benchmarking.py                 parameter sweeps for MLE reconstruction
  obtain_data.py                  script: run full benchmark suite, save to HDF5
  state_selector_interactive.py   ipywidgets UI: pick a target state and parameters
  algorithm_interactive.py        ipywidgets UI: sample, reconstruct, and plot results
  plotting.py                     density matrix / Wigner / fidelity / homodyne plots
bme_homodyne/
  __init__.py                     
  diagnostics.py                  BME optimization file to optimize the Metropolis-Hastings Algorithm
  homodyne.py                     Simulate the homodyne data same as MLE
  likelihood.py                   for pre-computing the likelihood and storing before going to MH algorithm
  mcmc.py                         Implements the Metropolis-Hastings Algorithm
  priors.py                       choice of priors, includes local haar random step
  states.py                       contains the relevant states we need to estimate
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
- [x] Bayesian mean estimate implementation and comparison against MLE.
      A draft (`Notebooks/BME_homodyne.ipynb`) reconstructs a state with
      Metropolis-Hastings sampling, it depends on `bme_homodyne` folder
