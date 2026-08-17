import numpy as np
import math
from dataclasses import dataclass
from scipy.special import hermite

import qutip as q


def x_n(n, x):
    """Quadrature-basis wavefunction <q_0=x|n>."""
    pref = 1.0 / (np.pi**0.25 * math.sqrt((2.0**n) * math.factorial(n)))
    return hermite(n)(x) * np.exp(-0.5 * x**2) * pref

def qtheta_n(n, theta, x):
    """Quadrature-basis wavefunction at angle theta."""
    return np.exp(1j * n * theta) * x_n(n, x)

def proj_xvec(cutoff, xvec):
    """Point projectors |q_0=x_i><q_0=x_i| on the x-grid."""
    fock_index = np.arange(cutoff)
    wf_x = np.array([qtheta_n(n, 0.0, xvec) for n in fock_index])
    return np.einsum('ni,mi->inm', wf_x, np.conj(wf_x))

def proj_bins(cutoff, point_projectors, x_vec, bin_edges, N_bins, thetas):
    """Bin projectors including the angle-dependent phase factors."""

    # Determine which bin each x_vec value belongs to.
    bin_idx = np.digitize(
        x_vec,
        bin_edges[1:-1],
        right=False
    )

    bin_idx = np.clip(
        bin_idx,
        0,
        N_bins - 1
    )

    # Build array of projectors (without phase factors):
    #
    # proj_bins[b,n,m] = 1/N_b sum_{i in box b} |q_0=x_i><q_0=x_i|
    proj_bins_theta0 = np.zeros(
        (N_bins, cutoff, cutoff),
        dtype=complex
    )

    np.add.at(
        proj_bins_theta0,
        bin_idx,
        point_projectors
    )

    # Number of x-grid points in each bin
    bin_counts = np.bincount(
        bin_idx,
        minlength=N_bins
    )

    # Avoid silently producing NaNs for empty bins
    if np.any(bin_counts == 0):
        empty_bins = np.where(bin_counts == 0)[0]
        raise ValueError(
            f"Some bins contain no x_vec points: {empty_bins}"
        )

    proj_bins_theta0 /= bin_counts[:, None, None]

    # Obtain correct phase factors exp(i (n-m)theta)
    n = np.arange(cutoff)

    phase = np.exp(1j * thetas[:, None, None] 
                   * (n[None, :, None] - n[None, None, :]))

    # Apply phase factors:
    #
    # proj_j[a,b,n,m]
    # = 1/N_b sum_{i in box b}
    #   |q_theta[a]=x_i><q_theta[a]=x_i|
    proj_j = (
        phase[:, None, :, :]
        * proj_bins_theta0[None, :, :, :]
    )

    return proj_j

def hit_probabilities(data, bin_edges):
    """Convert empirical histogram densities to bin probabilities."""
    bin_widths = np.diff(bin_edges)
    hits = []

    for row in data:
        hist, _ = np.histogram(row, bins=bin_edges, density=True)
        hits.append(hist)

    return np.array(hits) * bin_widths[None, :]


@dataclass
class TomographyOperators:
    """Cached angle/bin-resolved tomography data for one binning."""
    proj_bins: np.ndarray
    hits_prob: np.ndarray
    N_angles: int
    N_bins: int
    cutoff: int
    bin_edges: np.ndarray


class MLE:
    """Maximum-likelihood quantum state tomography from homodyne data."""

    def __init__(
        self,
        data,
        N_bins=30,
        initial_rho=None,
        N_cutoff=None,
        x_lims=(-5, 5),
        x_points=200,
        bin_edges=None
    ):

        self.data = np.asarray(data)

        if self.data.ndim != 2:
            raise ValueError(
                "data must be a 2D array with shape "
                "(N_angles, N_shots)"
            )

        # Measurement angles
        self.thetas = np.linspace(
            0,
            np.pi,
            len(self.data),
            endpoint=False
        )

        self.N_angles = len(self.thetas)

        # Quadrature grid used to approximate the bin projectors
        self.x_vec = np.linspace(
            x_lims[0],
            x_lims[1],
            x_points
        )

        # Set initial state for MLE algorithm
        self.set_initial_state(
            state=initial_rho,
            cutoff=N_cutoff
        )

        # Calculate point projectors only once.
        #
        # These do not depend on the binning, so they can be reused
        # whenever set_binning() is called.
        self.point_projectors = proj_xvec(
            self.cutoff,
            self.x_vec
        )

        # Set initial binning
        self.set_binning(
            N_bins=N_bins,
            bin_edges=bin_edges
        )

    def set_initial_state(self, state=None, cutoff=None):

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
            rho_init = np.array(
                state,
                dtype=complex
            )

        if rho_init.shape != (cutoff, cutoff):
            raise ValueError(
                f"Initial state has shape {rho_init.shape}, "
                f"but cutoff is {cutoff}."
            )

        # Normalize the density matrix
        rho_init = self._normalize_density_matrix(
            rho_init,
            name="Initial density matrix"
        )

        self.rho_init = rho_init
        self.cutoff = cutoff

    def set_binning(self, N_bins=None, bin_edges=None):

        # If explicit bin edges are provided, determine N_bins from them
        if bin_edges is not None:

            bin_edges = np.asarray(
                bin_edges,
                dtype=float
            )

            if bin_edges.ndim != 1:
                raise ValueError(
                    "bin_edges must be a one-dimensional array."
                )

            if len(bin_edges) < 2:
                raise ValueError(
                    "bin_edges must contain at least two values."
                )

            if np.any(np.diff(bin_edges) <= 0):
                raise ValueError(
                    "bin_edges must be strictly increasing."
                )

            if N_bins is not None and N_bins != len(bin_edges) - 1:
                raise ValueError(
                    "N_bins does not match the number of bin edges."
                )

            N_bins = len(bin_edges) - 1

        # Otherwise construct equally spaced bin edges
        else:

            if N_bins is None:
                N_bins = getattr(
                    self,
                    "N_bins",
                    30
                )

            if N_bins <= 0:
                raise ValueError(
                    "N_bins must be positive."
                )

            x_lims = (
                self.x_vec[0],
                self.x_vec[-1]
            )

            bin_edges = np.linspace(
                x_lims[0],
                x_lims[1],
                N_bins + 1,
                endpoint=True
            )

        self.N_bins = N_bins
        self.bin_edges = bin_edges

        # Recompute only the quantities that depend on the binning.
        self.hits_prob = hit_probabilities(
            self.data,
            self.bin_edges
        )

        # Compute projectors onto the bins
        self.proj_bins = proj_bins(
            self.cutoff,
            self.point_projectors,
            self.x_vec,
            self.bin_edges,
            self.N_bins,
            self.thetas
        )

        # Store the current complete tomography setup
        self.ops = TomographyOperators(
            proj_bins=self.proj_bins,
            hits_prob=self.hits_prob,
            N_angles=self.N_angles,
            N_bins=self.N_bins,
            cutoff=self.cutoff,
            bin_edges=self.bin_edges.copy()
        )

    def _normalize_density_matrix(self, rho, name="state"):
        rho = np.asarray(rho, dtype=complex)
        trace = np.trace(rho)

        if abs(trace) <= 1e-12:
            raise ValueError(f"{name} has zero trace.")

        return rho / np.real(trace)

    def _compute_fidelity(self, rho_a, rho_b):
        return q.fidelity(q.Qobj(rho_a), q.Qobj(rho_b)) ** 2


    def compute_R_operator(self, rho_current=None):

        # Set density matrix, if none provided -> initial
        if rho_current is None:
            rho_current = self.rho_init

        # Compute traces tr{proj_j rho} for each angle and bin
        traces = np.einsum(
            'abnm,mn->ab',
            self.proj_bins,
            rho_current
        )

        # Theoretical probabilities should be real.
        # Remove tiny numerical imaginary parts.
        traces = traces.real

        # Compute inverse of traces, avoiding division by zero.
        # Use a small positive floor to prevent unstable blow-ups from
        # tiny negative/zero theoretical probabilities arising from finite
        # numerical precision.
        traces = np.clip(traces, 1e-12, None)
        traces_inv = 1.0 / traces

        R_op = np.einsum(
            'abnm,ab,ab->nm',
            self.proj_bins,
            traces_inv,
            self.hits_prob
        )

        R_op /= self.N_angles

        # R(rho) should be Hermitian
        R_op = 0.5 * (
            R_op + np.conj(R_op.T)
        )

        return R_op

    def update_once(self, rho_iter=None):

        if rho_iter is None:
            rho_iter = self.rho_init

        R_op = self.compute_R_operator(rho_iter)

        rho_next = np.einsum(
            'nm,ml,lk->nk',
            R_op,
            rho_iter,
            R_op
        )

        rho_next = self._normalize_density_matrix(
            rho_next,
            name="Updated density matrix"
        )
        rho_next = 0.5 * (rho_next + np.conj(rho_next.T))

        return rho_next


    def run_mle(self, rho_init=None, store_states=False,
                stop_condition=lambda i, fid: False,
                max_iter=100, verbose=False):

        rho_current = (rho_init if rho_init is not None else self.rho_init)
        rho_current = np.array(rho_current, dtype=complex)
        rho_current = self._normalize_density_matrix(
            rho_current,
            name="Current density matrix"
        )

        fidelities = []

        if store_states:
            states = [rho_current.copy()]

        for i in range(max_iter):

            rho_next = self.update_once(rho_current)
            fid = self._compute_fidelity(rho_current, rho_next)

            if store_states:
                states.append(
                    rho_next.copy()
                )

            fidelities.append(fid)

            if verbose:
                print(
                    f"Iteration {i + 1}: "
                    f"Fidelity = {fid:.6f}"
                )

            rho_current = rho_next

            if stop_condition(i, fid):
                break

        if store_states:
            return (
                rho_current,
                fidelities,
                states
            )

        return (
            rho_current,
            fidelities
        )

    def run_iterations(self, n_iterations, **kw):
        return self.run_mle(max_iter=n_iterations, **kw)

    def run_until_fidelity(self, fid_target=1.0, max_iter=1000, **kw):
        return self.run_mle(
            stop_condition=lambda i, fid: (fid >= fid_target),
            max_iter=max_iter,
            **kw
        )

    def run_until_convergence(self, threshold=1e-6, max_iter=1000, **kw):
        return self.run_mle(
            stop_condition=lambda i, fid: (1 - fid <= threshold),
            max_iter=max_iter,
            **kw
        )