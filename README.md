# Quantum State Tomography (QST)

This repository contains our project work for the SCIQIS course. We study how to reconstruct a quantum state from quadrature measurements obtained through homodyne detection, including simulated measurements.

The reconstruction can be used to estimate the state's density matrix and Wigner function. We plan to compare several reconstruction methods, including maximum-likelihood estimation, Bayesian mean estimation, and neural-network-based approaches.

For each method, we will benchmark the measurement and algorithm parameters required for accurate reconstruction while minimizing resource use.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. Install uv, clone this repository, and run:

```bash
uv sync
```

This creates or updates the project's `.venv` environment using the dependencies listed in `pyproject.toml` and the locked versions in `uv.lock`.

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

The benchmark data-generation script lives with the reusable Python modules:

```bash
uv run python utilities/obtain_data.py --output benchmark_results_10runs.h5
```

The generated HDF5 result files are kept local and are not committed to the repository.