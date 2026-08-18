# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import h5py

import qutip as q
from tqdm.notebook import tqdm

####################################################################
# Import utilities
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utilities.state_selector_interactive import state
from utilities.functions import cat, sqv, pssqv, sq_cat, fock
from utilities.artificial_samples import sample_homodyne
from utilities.benchmarking import diff_bins, diff_samples

# Set quadrature range parameters
x_lim = (-5, 5)
x_points = 1000
x_vec = np.linspace(x_lim[0], x_lim[1], x_points)

cutoff = 50

default_n_angles = 4
default_n_samples = 1000
default_bins = 20

max_iter = 1000
fid_threshold = 1e-12
# Initial state vacuum
rho_0 = np.zeros((cutoff, cutoff))
rho_0[0, 0] = 1.0

def benchmark_bins(state, n_bins_list, 
                   n_angles=default_n_angles, n_samples=default_n_samples, x_vec=x_vec,
                   max_iter=max_iter, fid_threshold=fid_threshold, rho_0=rho_0):
    _, samples = sample_homodyne(state, n_angles=n_angles, 
                                 n_samples_theta=n_samples, bin_data=False)

    results = diff_bins(n_bins_array=n_bins_list, xlims=x_lim, x_points=x_points, samples=samples,
                    max_iter=max_iter, threshold=fid_threshold, rho_init=rho_0)
    output = []
    for n_bin, result in zip(n_bins_list, results):
        rho_out = result.rho_out
        fid_list = result.fidelities
        fidelity = q.fidelity(state, q.Qobj(rho_out))
        n_iter = len(fid_list)
        output.append((n_bin, fidelity, n_iter))
    return output

def benchmark_samples(state, n_angles_list, n_points_list,
                      n_bins=default_bins, x_vec=x_vec, 
                      max_iter=max_iter, fid_threshold=fid_threshold, rho_0=rho_0):
    output = []
    for n_angles, n_samples in zip(n_angles_list, n_points_list):
        _, samples = sample_homodyne(state, n_angles=n_angles, n_samples_theta=n_samples)
        result = diff_samples(n_angle_samples=[(n_angles, n_samples)], state=state, x_vec=x_vec, n_bins=n_bins,
                              max_iter=max_iter, threshold=fid_threshold, rho_init=rho_0)[0]
        rho_out = result.rho_out
        fid_list = result.fidelities
        fidelity = q.fidelity(state, q.Qobj(rho_out))
        n_iter = len(fid_list)
        output.append((n_angles, n_samples, fidelity, n_iter))
    return output

###################################################################
states = {f"fock_{n}": fock(cutoff, n) for n in range(5)}

n_bins_list = np.arange(1, 40)
n_angles_list = np.arange(2, 20)
n_samples_list = np.arange(100, 1000, 100)

results_bins = {}
results_samples = {}
for state_name, target_state in states.items():
    results_bins[state_name] = benchmark_bins(target_state, n_bins_list)
    results_samples[state_name] = benchmark_samples(
        target_state, n_angles_list, n_samples_list
    )

# Save numerical results and the settings required to reproduce them.
output_path = root / "benchmark_results.h5"
with h5py.File(output_path, "w") as h5:
    h5.attrs["state_family"] = "Fock"
    h5.attrs["x_lim"] = x_lim
    h5.attrs["x_points"] = x_points
    h5.attrs["cutoff"] = cutoff
    h5.attrs["default_n_angles"] = default_n_angles
    h5.attrs["default_n_samples"] = default_n_samples
    h5.attrs["default_bins"] = default_bins
    h5.attrs["max_iter"] = max_iter
    h5.attrs["fid_threshold"] = fid_threshold
    h5.create_dataset("rho_0", data=rho_0)

    bins_group = h5.create_group("bins")
    samples_group = h5.create_group("samples")
    for state_name in states:
        fock_number = int(state_name.removeprefix("fock_"))

        state_bins = bins_group.create_group(state_name)
        state_bins.attrs["state_type"] = "Fock"
        state_bins.attrs["fock_number"] = fock_number
        state_bins.create_dataset(
            "results",
            data=np.asarray(
                results_bins[state_name],
                dtype=[("n_bins", "i8"), ("fidelity", "f8"), ("iterations", "i8")],
            ),
            compression="gzip",
        )

        state_samples = samples_group.create_group(state_name)
        state_samples.attrs["state_type"] = "Fock"
        state_samples.attrs["fock_number"] = fock_number
        state_samples.create_dataset(
            "results",
            data=np.asarray(
                results_samples[state_name],
                dtype=[
                    ("n_angles", "i8"),
                    ("n_samples", "i8"),
                    ("fidelity", "f8"),
                    ("iterations", "i8"),
                ],
            ),
            compression="gzip",
        )

