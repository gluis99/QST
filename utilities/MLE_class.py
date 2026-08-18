import numpy as np
import qutip as q
from dataclasses import dataclass, field
from typing import List

from .functions import proj_xvec, proj_bins

DEFAULT_CUTOFF = 15
DEFAULT_MAX_ITER = 100
DEFAULT_SP_THRESHOLD = 1e-8

# Data class for storing the results of an MLE run
@dataclass
class RunResult:
    
    rho_out: np.ndarray
    fidelities: List[float]
    states: List[np.ndarray] = field(default_factory=list)


#######################################################################
# Class implementing the maximum likelihood estimation (MLE) algorithm
# for quantum state tomography given homodyne measurement data.

class MLE:
    def __init__(self, data, N_bins=30, initial_rho=None, N_cutoff=None,
                 x_lims=(-5, 5), x_points=200, bin_edges=None):
        # Raw data: array data[a, i] = x^(theta[a])_i
        if isinstance(data, list):
            self.data = np.array(data)
        self.data = data

        self.thetas = np.linspace(0, np.pi, len(self.data), endpoint=False)

        self.N_angles = len(self.thetas)

        self.N_bins = N_bins
        self.x_vec = np.linspace(x_lims[0], x_lims[1], x_points)

        # Set initial state for MLE algorithm
        self.set_initial_state(state=initial_rho, cutoff=N_cutoff)

        self.point_projectors = proj_xvec(self.cutoff, self.x_vec)

        self.set_binning(N_bins=self.N_bins, bin_edges=bin_edges)

    #######################################################################
    # Set initial state for MLE algorithm
    def set_initial_state(self, state, cutoff=None):
        if state is None and cutoff is None:
            cutoff = DEFAULT_CUTOFF
        if cutoff is None:
            cutoff = state.shape[0]
        elif state is None:
            state = np.eye(cutoff) / cutoff

        # If state given as ket, convert to density matrix
        if isinstance(state, q.Qobj):
            if state.isket:
                state = q.ket2dm(state)
            rho_init = state.full()
        # If state given as density matrix (numpy array)
        else:
            rho_init = np.array(state, dtype=complex)

        # Normalize the density matrix
        rho_init = rho_init / np.trace(rho_init)
        self.rho_init = rho_init
        self.cutoff = cutoff

    # Set binning parameters
    def set_binning(self, N_bins=None, bin_edges=None):
        x_lims = (self.x_vec[0], self.x_vec[-1])
        self.N_bins = N_bins
        if bin_edges is None:
            bin_edges = np.linspace(x_lims[0], x_lims[1], N_bins + 1, endpoint=True)
        self.bin_edges = bin_edges

        bin_widths = np.diff(self.bin_edges)
        # Compute histogram of hits for each angle from data and bin edges
        hits = []
        for i, _ in enumerate(self.thetas):
            hist, _ = np.histogram(self.data[i], bins=self.bin_edges, density=True)
            hits.append(hist)
        # Transform hits from density into probabilities
        self.hits_prob = np.array(hits) * bin_widths[None, :]
        # Compute projectors onto the bins
        self.proj_bins = proj_bins(self.cutoff, self.point_projectors, self.x_vec, 
                                   self.bin_edges, self.N_bins, self.thetas)

    def new_data(self, data):
        # Raw data: array data[a, i] = x^(theta[a])_i
        if isinstance(data, list):
            self.data = np.array(data)
        self.data = data

        self.thetas = np.linspace(0, np.pi, len(self.data), endpoint=False)

        self.N_angles = len(self.thetas)
        # New data requires new histogram and projectors
        self.set_binning(N_bins=self.N_bins)

    #######################################################################
    # Compute R(rho) operator
    def R(self, rho_current=None):
        # set density matrix, if none provided-> initial
        if rho_current is None:
            rho_current = self.rho_init
        
        # Compute traces tr{proj_j rho} for each angle and bin
        traces = np.einsum('abnm,mn->ab', self.proj_bins, rho_current)

        # Compute inverse of traces, avoiding division by zero
        traces_inv = np.where(traces != 0, 1.0 / traces, 0.0)
            
        R_op = np.einsum('abnm,ab,ab->nm', 
                         self.proj_bins, traces_inv, self.hits_prob)
        return R_op/self.N_angles
    
    # Compute one iteration: rho' = R(rho) rho R(rho)
    def one_iteration(self, rho_iter=None):
        if rho_iter is None:
            rho_iter = self.rho_init
        R_op = self.R(rho_iter)
        rho_prime = np.einsum('nm,ml,lk->nk', R_op, rho_iter, R_op)
        return rho_prime/np.trace(rho_prime)

    #######################################################################
    # Apply algorithm
    def run(self, rho_init=None, store_states=False,
            stop_condition=lambda i, fid: False, max_iter=DEFAULT_MAX_ITER, 
            verbose=False):
        rho_current = rho_init if rho_init is not None else self.rho_init
        fidelities = []
        states = [rho_current.copy()] if store_states else []

        for i in range(max_iter):
            rho_next = self.one_iteration(rho_current)
            fid = q.fidelity(q.Qobj(rho_current), q.Qobj(rho_next))**2
            fidelities.append(fid)

            if store_states:
                states.append(rho_next.copy())
            if verbose:
                print(f"Iteration {i+1}: Fidelity = {fid:.6f}")

            rho_current = rho_next
            if stop_condition(i, fid):
                break

        return RunResult(rho_out=rho_current, fidelities=fidelities, states=states)

    def run_ntimes(self, n_iterations, **kw):
        return self.run(max_iter=n_iterations, **kw)

    def run_fid(self, fid_target=1.0, max_iter=DEFAULT_MAX_ITER, **kw):
        return self.run(stop_condition=lambda i, fid: fid >= fid_target, max_iter=max_iter, **kw)

    def run_setpoint(self, threshold=DEFAULT_SP_THRESHOLD, max_iter=DEFAULT_MAX_ITER, **kw):
        return self.run(stop_condition=lambda i, fid: 1 - fid <= threshold, max_iter=max_iter, **kw)
