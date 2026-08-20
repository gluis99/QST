"""Quadrature wavefunctions and homodyne sampling."""

import math
import numpy as np
from scipy.special import hermite


def x_n(n, x):
    """Harmonic-oscillator position wavefunction <x|n>."""
    pref = 1.0 / (
        np.pi ** 0.25 * math.sqrt((2.0 ** n) * math.factorial(n))
    )
    return hermite(n)(x) * np.exp(-0.5 * x ** 2) * pref


def qtheta_n(n, theta, x):
    """Quadrature wavefunction <x_theta|n> in the notebook convention."""
    return np.exp(1j * n * theta) * x_n(n, x)


def _rho_array(state):
    if isinstance(state, np.ndarray):
        return np.asarray(state)
    return state.full() if state.isoper else state.proj().full()


def prob_marginal(state, theta, xvec):
    """Evaluate the quadrature probability p(x|theta) on ``xvec``."""
    rho = _rho_array(state)
    N = rho.shape[0]
    xvec = np.asarray(xvec)

    wf = np.array(
        [[qtheta_n(n, theta, x) for x in xvec] for n in range(N)],
        dtype=complex,
    )

    p = np.real(np.einsum("ij,ik,kj->j", wf, rho, np.conj(wf)))
    return np.clip(p, 0, None)


def sample_homodyne(
    state,
    n_angles,
    n_samples_theta,
    bin_data=False,
    x_vec=None,
    n_bins=30,
    rng=None,
):
    """Generate homodyne samples at equally spaced angles in [0, pi)."""
    if x_vec is None:
        x_vec = np.linspace(-5, 5, 200)
    x_vec = np.asarray(x_vec)

    if rng is None:
        rng = np.random.default_rng()

    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)
    samples = []

    for theta in thetas:
        p = prob_marginal(state, theta, x_vec)
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            raise ValueError("Invalid quadrature probability.")
        p = p / p_sum

        x_samples = rng.choice(x_vec, size=n_samples_theta, p=p)

        if bin_data:
            counts, bin_edges = np.histogram(
                x_samples,
                bins=n_bins,
                range=(x_vec[0], x_vec[-1]),
            )
            samples.append((counts, bin_edges))
        else:
            samples.append(x_samples)

    return thetas, samples
