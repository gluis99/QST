import numpy as np
import math
from scipy.special import hermite

import qutip as q


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

######################################################################################

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
