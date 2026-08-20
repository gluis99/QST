"""POVM-effect construction and homodyne likelihoods."""

import numpy as np
from .homodyne import qtheta_n


def make_effects(N, thetas, bin_edges, x):
    """Precompute quadrature-bin effects E[angle, bin, :, :]."""
    n_bins = len(bin_edges) - 1
    effects = np.zeros((len(thetas), n_bins, N, N), dtype=complex)

    for a, theta in enumerate(thetas):
        qtheta = np.array(
            [[qtheta_n(n, theta, xi) for xi in x] for n in range(N)],
            dtype=complex,
        )

        for j in range(n_bins):
            if j == n_bins - 1:
                mask = (x >= bin_edges[j]) & (x <= bin_edges[j + 1])
            else:
                mask = (x >= bin_edges[j]) & (x < bin_edges[j + 1])

            x_bin = x[mask]
            q_bin = qtheta[:, mask]

            if len(x_bin) > 1:
                integrand = np.einsum(
                    "ix,jx->xij",
                    np.conj(q_bin),
                    q_bin,
                )
                effects[a, j] = np.trapezoid(integrand, x_bin, axis=0)

    return effects


def probabilities_from_rho(rho, effects):
    """Return p(theta, bin) = Tr[rho E(theta, bin)]."""
    probabilities = np.real(np.einsum("ij,abji->ab", rho, effects))
    return np.clip(probabilities, 0, None)


def fast_log_likelihood(rho, effects, counts, min_probability=1e-15):
    """Multinomial log-likelihood for binned homodyne data."""
    probabilities = probabilities_from_rho(rho, effects)
    probabilities = np.clip(probabilities, min_probability, None)
    return float(np.sum(counts * np.log(probabilities)))
