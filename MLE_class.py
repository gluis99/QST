import numpy as np
import math
from scipy.special import gammaln, hermite

import qutip as q

# Calculate the quadrature-basis wavefunction <q_0=x|n>
def x_n(n, x):
    pref = 1.0 / (np.pi**0.25 * math.sqrt((2.0**n) * math.factorial(n)))
    return hermite(n)(x) * np.exp(-0.5 * x**2) * pref

# Calculate the quadrature-basis wavefunction <q_theta=x|n>=<q_0=x|n> * exp(i n theta)
def qtheta_n(n, theta, x):
    return np.exp(1j * n * theta) * x_n(n, x)

# Calculate point projectors {|q_theta=x_i><q_theta=x_i|}_i for x_i in xvec
def proj_xvec(cutoff, xvec):
    fock_index = np.arange(cutoff)
    # Obtain wf_x[n,i]=<q_0=x_i|n>
    wf_x = np.array([qtheta_n(n, 0.0, xvec) for n in fock_index])
    # Obtain point_projectors[i,n,m]=|q_0=x_i><q_0=x_i|
    point_projectors = np.einsum('ni,mi->inm', wf_x, np.conj(wf_x))

    return point_projectors

# Calculate projectors onto the bins {integral_j|q_theta=x_j><q_theta=x_j|}_j
def proj_bins(cutoff,point_projectors, x_vec, bin_edges, N_bins, thetas):
    # Determine which bin each x_vec value belongs to
    bin_idx = np.digitize(x_vec, bin_edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, N_bins - 1)
    # Obtain point_projectors[i,n,m]=|q_0=x_i><q_0=x_i|
    point_projectors = proj_xvec(cutoff, x_vec)
    # Build array of projectors (without phase factors): 
    # proj_bins[b,n,m]= 1/N_b sum_{i in box b} |q_0=x_i><q_0=x_i|
    proj_bins = np.zeros((N_bins, cutoff, cutoff), dtype=complex)
    np.add.at(proj_bins, bin_idx, point_projectors)
    proj_bins /= np.bincount(bin_idx, minlength=N_bins)[:, None, None]

    n = np.arange(cutoff)
    # Obtain correct phase factors exp(i (n-m)theta)
    phase = np.exp(1j * thetas[:, None, None] * (n[:, None] - n[None, :]))
    # Apply phase factors: array of projectors (with phase factors)
    # proj_bins[b,n,m]=1/N_b sum_{i in box b} |q_theta=x_i><q_theta=x_i|
    proj_j = phase[:, None, :, :] * proj_bins[None, :, :, :]

    return proj_j


#######################################################################
# Implement the maximum likelihood estimation (MLE) algorithm for quantum state tomography
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
            cutoff = 20
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
            
        R_op = np.einsum('abnm,ab,ab->nm', self.proj_bins, traces_inv, self.hits_prob) / self.N_angles
        return R_op

    #######################################################################
    # Apply algorithm

    def one_iteration(self, rho_current=None):
        if rho_current is None:
            rho_current = self.rho_init
        R_op = self.R(rho_current)
        rho_next = np.einsum('nm,mk,kl->nl', R_op, rho_current, R_op)
        return rho_next / np.trace(rho_next)

    def run_MLE_ntimes(self, rho_init=None, n_iterations=None, verbose=False):
        if rho_init is None:
            rho_init = self.rho_init

        rho_current = rho_init
        fidelities = []

        for i in range(n_iterations):
            rho_next = self.one_iteration(rho_current)
            rho_next /= np.trace(rho_next)

            fid = q.fidelity(q.Qobj(rho_current), q.Qobj(rho_next))**2
            fidelities.append(fid)

            if verbose:
                print(f"Iteration {i+1}: Fidelity = {fid:.6f}")

            rho_current = rho_next
        return rho_current, fidelities

    def run_MLE_fid(self, rho_init=None, fid_target=1.0, max_iter=1000, verbose=False):
        if rho_init is None:
            rho_init = self.rho_init
        
        rho_current = rho_init
        fidelities = []

        for i in range(max_iter):
            rho_next = self.one_iteration(rho_current)
            rho_next /= np.trace(rho_next)

            fid = q.fidelity(q.Qobj(rho_current), q.Qobj(rho_next))**2
            fidelities.append(fid)

            if verbose:
                print(f"Iteration {i+1}: Fidelity = {fid:.6f}")

            rho_current = rho_next

            if fid >= fid_target:
                rho_current
                break

        return rho_current, fidelities

