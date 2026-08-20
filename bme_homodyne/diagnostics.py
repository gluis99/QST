import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import qutip as q
from IPython.display import display, clear_output

from .mcmc import metropolis_hastings_bme
from .states import fidelity


def interactive_mh(
    effects,
    counts,
    N_CUTOFF,
    rho_true=None,
):
    """
    Interactive Metropolis-Hastings diagnostic.

    Returns
    -------
    rho_me : np.ndarray
        Final Bayesian mean estimate.

    chain : np.ndarray
        Posterior samples.

    diag : dict
        MCMC diagnostics.
    """

    # ========================================================
    # WIDGETS
    # ========================================================

    delta_widget = widgets.FloatSlider(
        value=0.15,
        min=0.01,
        max=0.80,
        step=0.01,
        description=r"$\delta_{\rm std}$:",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )

    burnin_widget = widgets.IntSlider(
        value=2000,
        min=500,
        max=15000,
        step=500,
        description="Burn-in:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )

    samples_widget = widgets.IntSlider(
        value=1500,
        min=500,
        max=10000,
        step=500,
        description="Samples:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )

    thin_widget = widgets.IntSlider(
        value=5,
        min=1,
        max=30,
        step=1,
        description="Thin:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )

    run_button = widgets.Button(
        description="Run MH",
        button_style="primary",
        icon="play",
    )

    output = widgets.Output()

    # ========================================================
    # STORAGE
    # ========================================================

    result = {
        "rho_me": None,
        "chain": None,
        "diag": None,
    }

    # ========================================================
    # CALLBACK
    # ========================================================

    def run_mh(_=None):

        with output:

            clear_output(wait=True)

            print("Running MH diagnostic...\n")

            print(
                f"delta_std = {delta_widget.value:.3f}"
            )

            print(
                f"burn_in   = {burnin_widget.value}"
            )

            print(
                f"samples   = {samples_widget.value}"
            )

            print(
                f"thin      = {thin_widget.value}"
            )

            diagnostic_rng = np.random.default_rng(9876)

            # ------------------------------------------------
            # RUN MH
            # ------------------------------------------------

            rho_me, chain, diag = metropolis_hastings_bme(
                effects=effects,
                counts=counts,
                N=N_CUTOFF,

                burn_in=burnin_widget.value,
                n_samples=samples_widget.value,
                thin=thin_widget.value,

                delta_std=delta_widget.value,
                target_acceptance=0.60,
                adapt_interval=100,

                rng=diagnostic_rng,
                rho_true=rho_true,
                store_chain=True,
            )

            # ------------------------------------------------
            # STORE RESULTS
            # ------------------------------------------------

            result["rho_me"] = rho_me
            result["chain"] = chain
            result["diag"] = diag

            # ------------------------------------------------
            # FIDELITY
            # ------------------------------------------------

            if rho_true is not None:

                F = fidelity(
                    rho_true,
                    q.Qobj(rho_me)
                )

                print()
                print(f"BME fidelity = {F:.6f}")

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            print(
                f"Final acceptance rate = "
                f"{diag['acceptance_rate_postburnin']:.4f}"
            )

            print(
                f"Final proposal scale = "
                f"{diag['final_delta_std']:.5f}"
            )

            # =================================================
            # PLOTS
            # =================================================

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(10, 10)
            )

            # -------------------------------------------------
            # FIDELITY
            # -------------------------------------------------

            if rho_true is not None:

                axes[0].plot(
                    diag["iterations"],
                    diag["sample_fidelity"],
                    alpha=0.25,
                    linewidth=1,
                    label="sample"
                )

                axes[0].plot(
                    diag["iterations"],
                    diag["running_fidelity"],
                    linewidth=2.5,
                    label="running BME"
                )

                axes[0].axhline(
                    F,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"final = {F:.4f}"
                )

                axes[0].set_ylim(0, 1.02)

            axes[0].set_ylabel("Fidelity")
            axes[0].set_title("Fidelity convergence")
            axes[0].legend()

            # -------------------------------------------------
            # ACCEPTANCE
            # -------------------------------------------------

            accepted = diag["accepted"].astype(float)

            window = min(
                200,
                max(20, len(accepted) // 20)
            )

            axes[1].plot(
                accepted,
                alpha=0.2,
                linewidth=0.5
            )

            if len(accepted) >= window:

                moving = np.convolve(
                    accepted,
                    np.ones(window) / window,
                    mode="valid"
                )

                axes[1].plot(
                    np.arange(
                        window - 1,
                        len(accepted)
                    ),
                    moving,
                    linewidth=2
                )

            axes[1].axhline(
                0.60,
                linestyle="--",
                linewidth=1.5,
                label="target = 0.60"
            )

            axes[1].set_ylabel("Acceptance")
            axes[1].set_title(
                "Acceptance-rate diagnostic"
            )
            axes[1].legend()

            # -------------------------------------------------
            # LOG LIKELIHOOD
            # -------------------------------------------------

            axes[2].plot(
                diag["log_likelihood"],
                linewidth=1
            )

            axes[2].axvline(
                diag["burn_in"],
                linestyle="--",
                linewidth=1.5,
                label="burn-in"
            )

            axes[2].set_xlabel("MH step")
            axes[2].set_ylabel(r"$\log L$")
            axes[2].set_title(
                "Likelihood trace"
            )
            axes[2].legend()

            fig.tight_layout()

            plt.show()

    # ========================================================
    # BUTTON
    # ========================================================

    run_button.on_click(run_mh)

    # ========================================================
    # DISPLAY
    # ========================================================

    display(
        widgets.VBox([
            delta_widget,
            burnin_widget,
            samples_widget,
            thin_widget,
            run_button,
            output,
        ])
    )

    # Return a container that will be populated
    return result