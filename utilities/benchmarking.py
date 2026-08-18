import numpy as np

from .MLE_class import MLE
#from .MLE_class_faster import MLE as MLE_faster
import qutip as q

from .functions import x_n_range

# Run MLE algorithm for different parameters and return a list of RunResult outputs
# Different number of bins
def diff_bins(n_bins_array, xlims, x_points, samples,
              max_iter=1000, threshold=1e-6,
              rho_init=None, cutoff=None):
    if rho_init is None:
        cutoff = cutoff or 10
        rho_init = np.eye(cutoff) / cutoff
    elif cutoff is not None:
        cutoff = min(cutoff, rho_init.shape[0])
        rho_init = rho_init[:cutoff, :cutoff]

    outputs = []
    first = True
    # Loop over different numbers of bins
    for n_bins in n_bins_array:
        # Re-bin existing MLE object on subsequent iterations
        if first:
            MaxLik = MLE(data=samples, N_bins=n_bins,
                         initial_rho=rho_init,
                         x_lims=(xlims[0], xlims[1]), x_points=x_points)
            first = False
        else:
            MaxLik.set_binning(N_bins=n_bins)
        outputs.append(MaxLik.run_setpoint(threshold=threshold, max_iter=max_iter))
    return outputs


# Different number of x values
def diff_x_values(xlims, n_xvalues, samples, n_bins,
                   max_iter=1000, threshold=1e-6,
                   rho_init=None, cutoff=None):
    if rho_init is None:
        cutoff = cutoff or 10
        rho_init = np.eye(cutoff) / cutoff
    elif cutoff is not None:
        cutoff = min(cutoff, rho_init.shape[0])
        rho_init = rho_init[:cutoff, :cutoff]

    outputs = []
    # Loop over different numbers of x points
    for n_x in n_xvalues:
        # x_points changes the quadrature grid, so the projectors must be rebuilt from scratch each time
        MaxLik = MLE(data=samples, N_bins=n_bins,
                     initial_rho=rho_init,
                     x_lims=(xlims[0], xlims[1]), x_points=n_x)
        outputs.append(MaxLik.run_setpoint(threshold=threshold, max_iter=max_iter))
    return outputs


# Different sample sizes: n_angle_samples[k]=(a_k,s_k): a angles with s samples per angle for k=0,...,K-1
def diff_samples(n_angle_samples, state, x_vec=np.linspace(-5, 5, 1000), n_bins=20,
                 max_iter=1000, threshold=1e-6,
                 rho_init=None):

    # Density matrix of the state
    rho = q.ket2dm(state).full() if state.isket else state.full()

    # Vector of Fock indices n
    fock_indices = np.arange(state.shape[0])
    # Obtain wf_x[n,i]=<q_0=x_i|n>
    wf_q0 = x_n_range(fock_indices, x_vec)

    if rho_init is None:
        cutoff = state.shape[0]
        rho_init = np.eye(cutoff) / cutoff
    
    outputs = []
    first = True
    # Loop over different sample sizes
    for n_angles, n_samples in n_angle_samples:
        # Obtainthetas[a]=pi*a/n_angles
        thetas = np.linspace(0, np.pi, n_angles, endpoint=False)
        # Obtain phase_factors[a,n]=exp(i *n*theta[a])
        phase_factors = np.exp(1j * np.einsum('n,a->an', fock_indices, thetas))
        # Obtain wf_q[a,n,i]=<q_theta[a]=x_i|n>
        wf_qs = np.einsum('an,ni->ani', phase_factors, wf_q0)

        # Obtain samples: list of  salmples [x^theta_i]
        samples = []
        for a in range(n_angles):
            wf_q = wf_qs[a]
            # Compute p_x = <q_theta=x|rho|q_theta=x>
            p_x = np.real(np.einsum('ij,ik,kj->j', wf_q, rho, np.conj(wf_q)))
            samples_theta = np.random.choice(x_vec, size=n_samples, p=p_x / p_x.sum())
            samples.append(samples_theta)

        # Run algorithm and store outputs
        # Create MLE object on first iteration
        if first:
            MaxLik = MLE(data=samples,N_bins=n_bins,
                         initial_rho=rho_init, 
                         x_lims=(x_vec[0], x_vec[-1]), x_points=len(x_vec))
            outputs.append(MaxLik.run_setpoint(threshold=threshold, max_iter=max_iter))
            first = False
        else:
            MaxLik.new_data(samples)
            outputs.append(MaxLik.run_setpoint(threshold=threshold, max_iter=max_iter))
    return outputs

