import numpy as np
import math
from scipy.special import gammaln, hermite
import qutip as q

from .functions import x_n_range

#######################################################################
# Build quadrature marginal distribution p(q_theta=x)
def prob_marginal(state, wf_q):
    # Accept either ket or density operator
    rho = q.ket2dm(state).full() if state.isket else state.full()
    # Compute p_x = <q_theta=x|rho|q_theta=x>
    p_x = np.real(np.einsum('ij,ik,kj->j', wf_q, rho, np.conj(wf_q)))
    return p_x


def sample_homodyne(state, n_angles, n_samples_theta, 
                    bin_data=False, x_vec=np.linspace(-5, 5, 200), n_bins=30):
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)

    fock_indices = np.arange(state.shape[0])
    # Obtain wf_x[n,i]=<q_0=x_i|n>
    wf_q0 = x_n_range(fock_indices, x_vec)
    # Obtain phase factors phase_factors[a,n]=exp(i *n*theta[a])
    phase_factors = np.exp(1j * np.einsum('i,j->ji', fock_indices, thetas))
    # Obtain wf_q[a,n,i]=<q_theta[a]=x_i|n>
    wf_qs = np.einsum('an,ni->ani', phase_factors, wf_q0)

    # b_samples: list of  salmples [x^theta_i]
    # If bin_data is True: binned [hits bin, bin edges]
    b_samples = []
    for a in range(n_angles):
        wf_q = wf_qs[a]
        p =  prob_marginal(state, wf_q)
        samples_theta = np.random.choice(x_vec, size=n_samples_theta, p=p / p.sum())
        if bin_data:
            hist, bin_edges = np.histogram(samples_theta, bins=n_bins, 
                                           range=(x_vec[0], x_vec[-1]), density=True)
            b_samples.append((hist, bin_edges))
        else:
            b_samples.append(samples_theta)
    return thetas, b_samples


