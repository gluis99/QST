import numpy as np
import math
from scipy.special import gammaln, hermite

import qutip as q

# Two mode unitary operator beamsplitter (T)
def BS_unitary(N, T):
    # Two-mode annihilation operators in the tensor-product space
    a = q.tensor(q.destroy(N), q.qeye(N))
    b = q.tensor(q.qeye(N), q.destroy(N))
    theta = np.arccos(np.sqrt(T))
    BS = theta * (a.dag() * b - a * b.dag())
    return BS.expm()

########################################################################
# Useful quantum states

# Fock state |n>
def fock(N, n):
    return q.basis(N, n)

# Cat state
def cat(N, k, alpha: complex, 
             abs_alpha: float = None, phase: float = None):
    if alpha is None:
        if abs_alpha is None:
            return q.basis(N, 0)
        elif phase is None:
            phase = 0
        alpha = abs_alpha * np.exp(1j*phase)
    
    n = np.arange(N)

    # Only even n for k=0, only odd n for k=1
    mask = (n % 2 == k)

    coeffs = np.zeros(N, dtype=complex)

    # log(sqrt(n!)) = 0.5 log(n!)
    coeffs[mask] = (alpha ** n[mask]
        * np.exp(-0.5 * abs(alpha)**2- 0.5 * gammaln(n[mask] + 1)))

    return q.Qobj(coeffs, dims=[[N], [1]]).unit()

# Squeezed vacuum state
def sqv(N, r, theta=0):
    if r == 0:
        return q.basis(N, 0)

    n = np.arange(N)
    t = np.tanh(r)

    # Only even n for squeezed vacuum
    mask = (n % 2 == 0)
    j = n[mask] // 2

    log_coeff = (gammaln(2*j + 1) - j * np.log(2)- gammaln(j + 1)
                  + j * np.log(abs(t)) - 0.5 * gammaln(n[mask] + 1))
    coeffs = np.zeros(N, dtype=complex)
    coeffs[mask] = ((-np.sign(t))**j * np.exp(1j * j * theta) * np.exp(log_coeff))

    return q.Qobj(coeffs, dims=[[N], [1]]).unit()

# Photon subtracted squeezed vacuum state (PSSQV)
def pssqv(N, r, m):
    if r == 0:
        return q.basis(N, 0)

    n = np.arange(N)
    k = n + m
    mask = (k % 2 == 0)
    j = k[mask] // 2

    t = np.tanh(r)

    log_coeff = (gammaln(2*j + 1) - j * np.log(2)- gammaln(j + 1)
                  - 0.5 * gammaln(n[mask] + 1) + j * np.log(abs(t)))

    coeffs = np.zeros(N)

    coeffs[mask] = ((-np.sign(t))**j * np.exp(log_coeff))

    return q.Qobj(coeffs, dims=[[N], [1]]).unit()

# 0n state |0> +- |n>
def ON_state(N, n, parity):
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0
    state[n] = (-1)**parity
    return q.Qobj(state, dims=[[N], [1]]).unit()

# Single photon subtracted squeezed vacuum state (PSSQV(m=1))
def pssqv_realistic(N, r, T):
    two_mode_state = BS_unitary(N, T) * (sqv(N, r) & q.basis(N, 0))
    # POVM element of photon detection
    P_on = q.qeye(N) - q.basis(N, 0).proj()

    final_state = ((q.qeye(N) & P_on) * two_mode_state).ptrace(0)
    return final_state.unit()

def sq_cat(N, k, alpha, r):
    Sq = q.squeeze(N, r)
    cat_state = cat(N, k, alpha)
    return Sq * cat_state

