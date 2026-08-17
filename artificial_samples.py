import numpy as np
import math
from scipy.special import gammaln, hermite
import qutip as q

# Useful quantum states
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
def sqv(N, r):
    n = np.arange(N)
    t = np.tanh(r)

    # Only even n for squeezed vacuum
    mask = (n % 2 == 0)
    j = n[mask] // 2

    log_coeff = (gammaln(2*j + 1) - j * np.log(2)- gammaln(j + 1)
                  + j * np.log(abs(t)) - 0.5 * gammaln(n[mask] + 1))

    coeffs = np.zeros(N)

    coeffs[mask] = ((-np.sign(t))**j * np.exp(log_coeff))

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

# Two mode unitary operator beamsplitter (T)
def BS_unitary(N, T):
    # Two-mode annihilation operators in the tensor-product space
    a = q.tensor(q.destroy(N), q.qeye(N))
    b = q.tensor(q.qeye(N), q.destroy(N))
    theta = np.arccos(np.sqrt(T))
    BS = theta * (a.dag() * b - a * b.dag())
    return BS.expm()

# Single photon subtracted squeezed vacuum state (PSSQV(m=1))
def spssqv(N, r, T):
    two_mode_state = BS_unitary(N, T) * (sqv(N, r) & q.basis(N, 0))
    # POVM element of photon detection
    P_on = q.qeye(N) - q.basis(N, 0).proj()

    final_state = ((q.qeye(N) & P_on) * two_mode_state).ptrace(0)
    return final_state.unit()

#######################################################################
# Calculate quadrature-basis wavefunction <q_0=x|n>
def x_n(n, x):
    pref = 1.0 / (np.pi**0.25 * math.sqrt((2.0**n) * math.factorial(n)))
    return hermite(n)(x) * np.exp(-0.5 * x**2) * pref

# Calculate quadrature-basis wavefunction <q_theta=x|n>=<q_0=x|n> * exp(i n theta)
def qtheta_n(n, theta, x):
    return np.exp(1j * n * theta) * x_n(n, x)

# Build quadrature marginal distribution p(q_theta=x)
def prob_marginal(state, theta, xvec):
    # Accept either ket or density operator
    rho = q.ket2dm(state).full() if state.isket else state.full()
    N = rho.shape[0]
    wf_x = np.array([[qtheta_n(n, theta, x) for x in xvec] for n in range(N)], dtype=complex)
    p_x = np.real(np.einsum('ij,ik,kj->j', wf_x, rho, np.conj(wf_x)))
    return p_x


def sample_homodyne(state, n_angles, n_samples_theta, 
                    bin_data=True, x_vec=np.linspace(-5, 5, 200), n_bins=30):
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)
    # b_samples: list of  salmples [x^theta_i]
    # If bin_data is True: binned [hits bin, bin edges]
    b_samples = []
    for i, theta in enumerate(thetas):
        p =  prob_marginal(state, theta, x_vec)
        samples_theta = np.random.choice(x_vec, size=n_samples_theta, p=p / p.sum())
        if bin_data:
            hist, bin_edges = np.histogram(samples_theta, bins=n_bins, 
                                           range=(x_vec[0], x_vec[-1]), density=True)
            b_samples.append((hist, bin_edges))
        else:
            b_samples.append(samples_theta)
    return thetas, b_samples


