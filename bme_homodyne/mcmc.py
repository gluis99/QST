"""Metropolis-Hastings Bayesian mean estimation."""

import numpy as np
import qutip as q

from .likelihood import fast_log_likelihood
from .priors import rho_from_A, local_haar_step
from .states import fidelity


def metropolis_hastings_bme(
    effects,
    counts,
    N,
    burn_in=2000,
    n_samples=2000,
    thin=5,
    delta_std=0.15,
    target_acceptance=0.60,
    adapt_interval=100,
    rng=None,
    rho_true=None,
    store_chain=True,
):
    """Run the MH BME chain from the supplied notebook.

    The chain starts at I/N and uses a purification parameterization
    rho=A A†/Tr(A A†). During burn-in only, the local proposal width
    is adapted toward ``target_acceptance``.
    """
    if rng is None:
        rng = np.random.default_rng()

    A_current = np.eye(N, dtype=complex) / np.sqrt(N)
    # A_current = np.zeros((N, N), dtype=complex)
    # A_current[0, 0] = 1.0
    rho_current = rho_from_A(A_current)
    logL_current = fast_log_likelihood(rho_current, effects, counts)

    total_steps = burn_in + n_samples * thin

    posterior_samples = []
    sampled_iterations = []
    sample_fidelity = []
    running_fidelity = []
    log_likelihood_trace = []
    acceptance_trace = []
    proposal_scale_trace = []

    rho_sum = np.zeros((N, N), dtype=complex)
    accepted_total = 0
    accepted_postburnin = 0
    window_accepted = 0
    postburnin_steps = 0

    for step in range(total_steps):
        A_proposed = local_haar_step(A_current, rng, delta_std)
        rho_proposed = rho_from_A(A_proposed)
        logL_proposed = fast_log_likelihood(rho_proposed, effects, counts)

        log_alpha = logL_proposed - logL_current
        accepted = np.log(rng.random()) < min(0.0, log_alpha)

        if accepted:
            A_current = A_proposed
            rho_current = rho_proposed
            logL_current = logL_proposed
            accepted_total += 1
            window_accepted += 1
            if step >= burn_in:
                accepted_postburnin += 1

        if step < burn_in and (step + 1) % adapt_interval == 0:
            window_rate = window_accepted / adapt_interval
            if window_rate < target_acceptance - 0.05:
                delta_std *= 0.80
            elif window_rate > target_acceptance + 0.05:
                delta_std *= 1.20
            delta_std = float(np.clip(delta_std, 1e-4, 2.0))
            window_accepted = 0

        log_likelihood_trace.append(logL_current)
        acceptance_trace.append(accepted)
        proposal_scale_trace.append(delta_std)

        if step >= burn_in and (step - burn_in) % thin == 0:
            sample = rho_current.copy()
            if store_chain:
                posterior_samples.append(sample)

            rho_sum += sample
            postburnin_steps += 1

            rho_running_mean = rho_sum / postburnin_steps
            rho_running_mean = (rho_running_mean + rho_running_mean.conj().T) / 2
            rho_running_mean /= np.trace(rho_running_mean).real

            sampled_iterations.append(step + 1)

            if rho_true is not None:
                sample_fidelity.append(
                    fidelity(rho_true, q.Qobj(sample))
                )
                running_fidelity.append(
                    fidelity(rho_true, q.Qobj(rho_running_mean))
                )

    if store_chain:
        posterior_samples = np.asarray(posterior_samples)
        rho_bme = np.mean(posterior_samples, axis=0)
    else:
        posterior_samples = None
        rho_bme = rho_sum / postburnin_steps

    rho_bme = (rho_bme + rho_bme.conj().T) / 2
    rho_bme /= np.trace(rho_bme).real

    postburnin_total_steps = n_samples * thin

    info = {
        "acceptance_rate_total": accepted_total / total_steps,
        "acceptance_rate_postburnin": accepted_postburnin / max(1, postburnin_total_steps),
        "final_delta_std": delta_std,
        "iterations": np.asarray(sampled_iterations),
        "sample_fidelity": np.asarray(sample_fidelity),
        "running_fidelity": np.asarray(running_fidelity),
        "log_likelihood": np.asarray(log_likelihood_trace),
        "accepted": np.asarray(acceptance_trace),
        "proposal_scale": np.asarray(proposal_scale_trace),
        "initial_state": np.eye(N) / N,
        "burn_in": burn_in,
        "thin": thin,
        "n_samples": n_samples,
        "target_acceptance": target_acceptance,
        "prior_measure": "Hilbert-Schmidt induced measure",
    }
    return rho_bme, posterior_samples, info
