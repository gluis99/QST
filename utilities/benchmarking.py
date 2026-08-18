import numpy as np

from .MLE_class import MLE
#from .MLE_class_faster import MLE as MLE_faster
import qutip as q

from .functions import x_n_range

# Run MLE algorithm for different parameters and return the fidelities, number of iterations and final states
# Different number of bins
def diff_bins(N_bins_array, xlims, x_points, samples, 
                max_iter=1000, threshold=1e-6,
                rho_init=None, cutoff=None,target_state=None,
                faster=False):
    if rho_init is None and cutoff is None:
        cutoff = 10
        rho_init = np.eye(cutoff) / cutoff
    elif rho_init is None:
        rho_init = np.eye(cutoff) / cutoff
    elif cutoff is None:
        cutoff = rho_init.shape[0]
    else:
        cutoff = min(cutoff, rho_init.shape[0])
        rho_init = rho_init[:cutoff, :cutoff]

    # Create MLE object
    if faster:
        ML_estimator = MLE_faster(data=samples, N_bins=N_bins_array[0], 
                                  x_lims=(xlims[0], xlims[1]), x_points=x_points,
                                  initial_rho=rho_init)
    else:
        ML_estimator = MLE(data=samples, N_bins=N_bins_array[0], 
                            x_lims=(xlims[0], xlims[1]), x_points=x_points,
                            initial_rho=rho_init)

    fidelities_arrays = []
    n_iter_array = []
    final_states = []

    fidelities = []

    for n_bins in N_bins_array:
        # Run MLE algorithm
        ML_estimator.set_binning(N_bins=n_bins)
        rho_final, fidelities = ML_estimator.run_setpoint(threshold=threshold, max_iter=max_iter)
        n_iter = len(fidelities)

        fidelities_arrays.append(fidelities)

        if target_state is not None:
            fidelity = q.fidelity(q.Qobj(rho_final), q.Qobj(target_state))**2
        fidelities.append(fidelity)
        n_iter_array.append(n_iter)
        final_states.append(rho_final)
    if target_state is not None:
        return (np.array(fidelities_arrays), np.array(n_iter_array), 
                np.array(final_states), fidelities)
    return np.array(fidelities_arrays), np.array(n_iter_array), np.array(final_states)

# Different number of x values
def diff_x_values(xlims, n_xvalues, samples, 
                    n_bins,
                    max_iter=1000, threshold=1e-6,
                    rho_init=None, cutoff=None,target_state=None):
    if rho_init is None and cutoff is None:
        cutoff = 10
        rho_init = np.eye(cutoff) / cutoff
    elif rho_init is None:
        rho_init = np.eye(cutoff) / cutoff
    elif cutoff is None:
        cutoff = rho_init.shape[0]
    else:
        cutoff = min(cutoff, rho_init.shape[0])
        rho_init = rho_init[:cutoff, :cutoff]

    fidelities_arrays = []
    n_iter_array = []
    final_states = []
    
    fidelities = []

    for n_x in n_xvalues:
        ML_estimator = MLE(data=samples, N_bins=n_bins, 
              x_lims=(xlims[0], xlims[1]), x_points=n_x,
              initial_rho=rho_init)
        
        rho_final, fidelities = ML_estimator.run_setpoint(threshold=threshold, max_iter=max_iter)
        n_iter = len(fidelities)

        fidelities_arrays.append(fidelities)

        if target_state is not None:
            fidelity = q.fidelity(q.Qobj(rho_final), q.Qobj(target_state))**2
            fidelities.append(fidelity)
        n_iter_array.append(n_iter)
        final_states.append(rho_final)
    if target_state is not None:
        return (np.array(fidelities_arrays), np.array(n_iter_array), 
                np.array(final_states), fidelities)
    return np.array(fidelities_arrays), np.array(n_iter_array), np.array(final_states)

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

