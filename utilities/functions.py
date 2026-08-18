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

def sq_cat(N, r_1, r_2=None, n_2=1):
    if r_2 is None:
        r_2 = -r_1
    two_mode_state = BS_unitary(N, 0.5) * (sqv(N, r_1) & sqv(N, r_2))
    # POVM element of photon detection
    P_n2 = q.basis(N, n_2).proj()
    final_state = ((q.qeye(N) & P_n2) * two_mode_state).ptrace(0)
    return final_state.unit()

################################################################################

# Calculate the quadrature-basis wavefunction <q_0=x|n>
def x_n(n, x):
    pref = 1.0 / (np.pi**0.25 * math.sqrt((2.0**n) * math.factorial(n)))
    return hermite(n)(x) * np.exp(-0.5 * x**2) * pref

def x_n_range(fock_indices, x_vec):
    wf_x = np.array([x_n(n, x_vec) for n in fock_indices])
    return wf_x

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
    # Build array of projectors (without phase factors): 
    # proj_bins[b,n,m]= 1/N_b sum_{i in box b} |q_0=x_i><q_0=x_i|
    proj_bins = np.zeros((N_bins, cutoff, cutoff), dtype=complex)
    np.add.at(proj_bins, bin_idx, point_projectors)
    proj_bins /= np.bincount(bin_idx, minlength=N_bins)[:, None, None]

    n = np.arange(cutoff)
    # Obtain correct phase factors exp(i (n-m)theta)
    phase = np.exp(1j * thetas[:, None, None] * (n[None,:] - n[:,None]))
    # Apply phase factors: array of projectors (with phase factors)
    # proj_bins[b,n,m]=1/N_b sum_{i in box b} |q_theta=x_i><q_theta=x_i|
    proj_j = phase[:, None, :, :] * proj_bins[None, :, :, :]

    return proj_j

# Build quadrature marginal distribution p(q_theta=x|state)
def prob_marginal(state, wf_q):
    # Accept either ket or density operator
    rho = q.ket2dm(state).full() if state.isket else state.full()
    # Compute p_x = <q_theta=x|rho|q_theta=x>
    p_x = np.real(np.einsum('ij,ik,kj->j', wf_q, rho, np.conj(wf_q)))
    return p_x
