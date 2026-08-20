"""Hilbert-Schmidt prior and purification-space proposals."""

import numpy as np


def sample_hilbert_prior(N, rng=None):
    """Sample the Hilbert-Schmidt induced measure."""
    if rng is None:
        rng = np.random.default_rng()

    A = (
        rng.normal(size=(N, N))
        + 1j * rng.normal(size=(N, N))
    ) / np.sqrt(2)

    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    return rho


def rho_from_A(A):
    """Map a square complex matrix A to rho=A A†/Tr(A A†)."""
    rho = A @ A.conj().T
    return rho / np.trace(rho).real


def local_haar_step(A, rng=None, delta_std=0.15):
    """Make the local symmetric proposal used by the supplied notebook."""
    if rng is None:
        rng = np.random.default_rng()

    psi = A.reshape(-1).copy()
    D = len(psi)
    i, j = rng.integers(0, D, size=2)
    delta = rng.normal(0.0, delta_std)

    if i == j:
        psi[i] *= np.exp(1j * delta)
    else:
        c, s = np.cos(delta), np.sin(delta)
        ai, aj = psi[i], psi[j]
        if i < j:
            psi[i] = c * ai + 1j * s * aj
            psi[j] = 1j * s * ai + c * aj
        else:
            psi[i] = c * ai + s * aj
            psi[j] = -s * ai + c * aj

    return psi.reshape(A.shape)

def haar_unitary(N, rng=None):
    """Generate an N x N Haar-random unitary matrix."""
    
    if rng is None:
        rng = np.random.default_rng()

    Z = (
        rng.normal(size=(N, N))
        + 1j * rng.normal(size=(N, N))
    ) / np.sqrt(2)

    Q, R = np.linalg.qr(Z)

    # Fix phases so that Q is Haar distributed
    phases = np.diag(R)
    phases = np.where(
        np.abs(phases) > 0,
        phases / np.abs(phases),
        1.0
    )

    Q = Q @ np.diag(np.conj(phases))

    return Q

def haar_step(A, rng=None, delta_std=0.15):
    
    if rng is None:
        rng = np.random.default_rng()

    N = A.shape[0]

    U = haar_unitary(
        N,
        rng=rng
    )

    A_proposed = U @ A

    return A_proposed


