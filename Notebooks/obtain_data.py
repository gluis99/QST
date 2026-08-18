import sys
import time
from pathlib import Path

import h5py
import numpy as np
import qutip as q
from tqdm.auto import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utilities.functions import ON_state, cat, fock
from utilities.artificial_samples import sample_homodyne
from utilities.benchmarking import diff_bins, diff_samples

# Simulation settings
x_lim = (-5, 5)
x_points = 1000
x_vec = np.linspace(x_lim[0], x_lim[1], x_points)
cutoff = 50
default_n_angles = 4
default_bins = 20
max_iter = 1000
fid_threshold = 1e-5
rho_0 = np.zeros((cutoff, cutoff))
rho_0[0, 0] = 1.0

default_n_samples = 1000
n_bins_list = np.arange(1, 30)
n_angles_list = np.arange(2, 20)
n_samples_list = np.arange(100, 1000, 100)
seed = 12345
# Independent measurement realizations + reconstructions per configuration
n_repeats = 20


def benchmark_bins(target_state, rng):
    """Benchmark reconstructions while varying the number of histogram bins.

    Each repeat draws an independent homodyne sample set (via the shared,
    continuously-advancing rng) and is swept over all bin counts, giving
    n_repeats independent fidelity/iteration values per bin count.
    """
    fidelity = np.empty((n_repeats, len(n_bins_list)))
    iterations = np.empty((n_repeats, len(n_bins_list)), dtype=np.int64)
    sweep_runtime = np.empty(n_repeats)

    for r in range(n_repeats):
        _, samples = sample_homodyne(
            target_state,
            n_angles=default_n_angles,
            n_samples_theta=default_n_samples,
            bin_data=False,
            rng=rng,
        )
        start = time.perf_counter()
        runs = diff_bins(
            n_bins_array=n_bins_list,
            xlims=x_lim,
            x_points=x_points,
            samples=samples,
            max_iter=max_iter,
            threshold=fid_threshold,
            rho_init=rho_0,
        )
        sweep_runtime[r] = time.perf_counter() - start
        for j, run in enumerate(runs):
            fidelity[r, j] = q.fidelity(target_state, q.Qobj(run.rho_out))
            iterations[r, j] = len(run.step_fidelities)

    return {"fidelity": fidelity, "iterations": iterations, "sweep_runtime": sweep_runtime}


def benchmark_samples(target_state, rng):
    """Benchmark reconstructions while varying angles and samples per angle.

    Each (n_angles, n_samples) configuration is reconstructed n_repeats times
    from independent draws, so mean/std/median fidelity and runtime can be
    computed per configuration instead of relying on a single realization.
    """
    configurations = [(n_angles, n_samples)
                      for n_angles in n_angles_list
                      for n_samples in n_samples_list]
    n_configs = len(configurations)
    fidelity = np.empty((n_repeats, n_configs))
    iterations = np.empty((n_repeats, n_configs), dtype=np.int64)
    runtime = np.empty((n_repeats, n_configs))

    for r in range(n_repeats):
        for j, (n_angles, n_samples) in enumerate(configurations):
            start = time.perf_counter()
            run = diff_samples(
                n_angle_samples=[(n_angles, n_samples)],
                state=target_state,
                x_vec=x_vec,
                n_bins=default_bins,
                max_iter=max_iter,
                threshold=fid_threshold,
                rho_init=rho_0,
                rng=rng,
            )[0]
            runtime[r, j] = time.perf_counter() - start
            fidelity[r, j] = q.fidelity(target_state, q.Qobj(run.rho_out))
            iterations[r, j] = len(run.step_fidelities)

    return {"fidelity": fidelity, "iterations": iterations, "runtime": runtime}


def build_states():
    """Return target states with labels and parameters for HDF5 metadata."""
    states = [
        {
            "family": "Fock",
            "label": f"n_{number}",
            "target_state": fock(cutoff, number),
            "fock_number": number,
        }
        for number in range(5)
    ]

    for parity_name, parity in (("even", 0), ("odd", 1)):
        for alpha in np.arange(0.0, 3.5, 0.5):
            target_state = fock(cutoff, 1) if alpha == 0.0 and parity else cat(cutoff, parity, alpha)
            states.append(
                {
                    "family": "cat",
                    "label": f"alpha_{alpha:.1f}_{parity_name}",
                    "target_state": target_state,
                    "alpha": alpha,
                    "parity": parity_name,
                }
            )

    for parity_name, parity in (("plus", 0), ("minus", 1)):
        for number in range(1, 5):
            states.append(
                {
                    "family": "ON",
                    "label": f"n_{number}_{parity_name}",
                    "target_state": ON_state(cutoff, number, parity),
                    "on_number": number,
                    "parity": parity_name,
                }
            )
    return states


def init_metadata(h5, states):
    """Write settings and reproducibility metadata (only for a fresh file)."""
    families = sorted({state["family"] for state in states})
    h5.attrs["state_families"] = np.asarray(families, dtype=h5py.string_dtype())
    h5.attrs["x_lim"] = x_lim
    h5.attrs["x_points"] = x_points
    h5.attrs["cutoff"] = cutoff
    h5.attrs["default_n_angles"] = default_n_angles
    h5.attrs["default_n_samples"] = default_n_samples
    h5.attrs["default_bins"] = default_bins
    h5.attrs["max_iter"] = max_iter
    h5.attrs["fid_threshold"] = fid_threshold
    h5.attrs["seed"] = seed
    h5.attrs["n_repeats"] = n_repeats
    h5.create_dataset("rho_0", data=rho_0)
    h5.create_dataset("n_bins_values", data=n_bins_list)
    h5.create_dataset("n_angles_values", data=n_angles_list)
    h5.create_dataset("n_samples_values", data=n_samples_list)
    # Full (n_angles, n_samples) grid: column j of the "samples" fidelity/
    # iterations/runtime arrays corresponds to sample_configurations[j].
    sample_configurations = np.asarray(
        [(n_angles, n_samples) for n_angles in n_angles_list for n_samples in n_samples_list]
    )
    h5.create_dataset("sample_configurations", data=sample_configurations)
    h5.create_group("bins")
    h5.create_group("samples")
    h5.create_group("target_states")


def state_done(h5, state):
    """Check whether a state's results were already written (for resuming)."""
    return all(
        state["label"] in h5[benchmark_name].get(state["family"], {})
        for benchmark_name in ("bins", "samples")
    )


def save_target_state(h5, state):
    """Persist the target density matrix itself, so it can be reloaded without
    depending on the generator functions (fock, cat, ON_state) that built it."""
    group = h5.require_group("target_states").require_group(state["family"])
    if state["label"] in group:
        return
    target_state = state["target_state"]
    rho = q.ket2dm(target_state).full() if target_state.isket else target_state.full()
    group.create_dataset(state["label"], data=rho, compression="gzip")
    h5.flush()


def save_state_results(h5, benchmark_name, state, result):
    """Write one state's results and flush to disk immediately."""
    benchmark_group = h5[benchmark_name]
    state_group = benchmark_group.require_group(state["family"]).create_group(state["label"])
    for name, value in state.items():
        if name != "target_state":
            state_group.attrs[name] = value
    # Store raw per-repeat arrays (shape: n_repeats x n_configs) so any
    # summary statistic (mean, std, median, ...) can be recomputed later.
    for array_name, array in result.items():
        state_group.create_dataset(array_name, data=array, compression="gzip")
    h5.flush()


def main():
    states = build_states()
    # Independent child RNG per state (derived from the master seed) so that
    # skipping already-computed states on resume doesn't change other states' draws.
    child_seeds = np.random.SeedSequence(seed).spawn(len(states))
    output_path = root / "benchmark_results.h5"

    # Resume support: reopen an existing (partially completed) file and skip
    # states that were already computed and flushed to disk.
    mode = "a" if output_path.exists() else "w"
    with h5py.File(output_path, mode) as h5:
        if "bins" not in h5:
            init_metadata(h5, states)

        for state, child_seed in zip(tqdm(states, desc="Benchmarking", unit="state"), child_seeds):
            save_target_state(h5, state)
            if state_done(h5, state):
                continue
            tqdm.write(f"Running {state['family']}: {state['label']}")
            rng = np.random.default_rng(child_seed)
            target_state = state["target_state"]
            save_state_results(h5, "bins", state, benchmark_bins(target_state, rng))
            save_state_results(h5, "samples", state, benchmark_samples(target_state, rng))


if __name__ == "__main__":
    main()

