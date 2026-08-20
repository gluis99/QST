"""Example CV states and fidelity utilities."""

import math
import numpy as np
import qutip as q
from scipy.special import gammaln
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from IPython.display import display, clear_output

def cat(N, k, alpha: complex = None, abs_alpha=None, phase=None):
    """Parity-selected coherent-state cat component.

    Parameters
    ----------
    N : int
        Fock-space cutoff.
    k : int
        Parity selector: 0 for even, 1 for odd.
    alpha : complex, optional
        Coherent-state amplitude.
    abs_alpha, phase : float, optional
        Alternative polar representation if ``alpha`` is None.
    """
    if alpha is None:
        if abs_alpha is None:
            return q.basis(N, 0)
        if phase is None:
            phase = 0
        alpha = abs_alpha * np.exp(1j * phase)

    n = np.arange(N)
    mask = (n % 2 == k)
    coeffs = np.zeros(N, dtype=complex)
    coeffs[mask] = (
        alpha ** n[mask]
        * np.exp(-0.5 * abs(alpha) ** 2
                 - 0.5 * gammaln(n[mask] + 1))
    )
    return q.Qobj(coeffs, dims=[[N], [1]]).unit()


def squeezed_vacuum(N, r):
    """Squeezed vacuum state with squeezing parameter ``r``."""
    n = np.arange(N)
    t = np.tanh(r)
    if t == 0:
        return q.basis(N, 0)

    mask = (n % 2 == 0)
    j = n[mask] // 2
    log_coeff = (
        gammaln(2 * j + 1)
        - j * np.log(2)
        - gammaln(j + 1)
        + j * np.log(abs(t))
        - 0.5 * gammaln(n[mask] + 1)
    )

    coeffs = np.zeros(N)
    coeffs[mask] = (-np.sign(t)) ** j * np.exp(log_coeff)
    return q.Qobj(coeffs, dims=[[N], [1]]).unit()


def fidelity(rho1, rho2):
    """Squared Uhlmann fidelity, matching the supplied notebook."""
    return float(q.fidelity(rho1, rho2) ** 2)


def select_state(N_CUTOFF=30):
    """
    Interactive selection of the target quantum state.

    The user selects the state and its parameters, previews the
    Wigner function, and confirms the choice.

    Returns
    -------
    state : qutip.Qobj
        Confirmed pure quantum state.
    """

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    result = {
        "state": None,
        "rho_true": None,
        "name": None,
        "confirmed": False,
    }

    # --------------------------------------------------------
    # Widgets
    # --------------------------------------------------------

    state_dropdown = widgets.Dropdown(
        options=["Cat", "Squeezed Vacuum"],
        value="Cat",
        description="State:",
        layout=widgets.Layout(width="250px"),
    )

    parity_widget = widgets.ToggleButtons(
        options=[("Even", 0), ("Odd", 1)],
        value=0,
        description="Parity:",
    )

    alpha_widget = widgets.FloatSlider(
        value=2.0,
        min=0.2,
        max=4.0,
        step=0.1,
        description=r"|α|:",
        continuous_update=False,
    )

    phase_widget = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=2*np.pi,
        step=0.1,
        description="Phase:",
        continuous_update=False,
    )

    r_widget = widgets.FloatSlider(
        value=0.8,
        min=0.0,
        max=1.25,
        step=0.01,
        description="r:",
        continuous_update=False,
    )

    confirm_button = widgets.Button(
        description="Confirm State",
        button_style="success",
        icon="check",
    )

    output = widgets.Output()

    # --------------------------------------------------------
    # Build state
    # --------------------------------------------------------

    def build_state():

        if state_dropdown.value == "Cat":

            psi = cat(
                N_CUTOFF,
                parity_widget.value,
                abs_alpha=alpha_widget.value,
                phase=phase_widget.value,
            )

            name = (
                f"{'Even' if parity_widget.value == 0 else 'Odd'} "
                f"Cat "
                f"(|α|={alpha_widget.value:.2f}, "
                f"ϕ={phase_widget.value:.2f})"
            )

        else:

            psi = squeezed_vacuum(
                N_CUTOFF,
                r_widget.value,
            )

            name = (
                f"Squeezed Vacuum "
                f"(r={r_widget.value:.2f})"
            )

        return psi, name

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    def update_preview(*args):

        with output:

            clear_output(wait=True)

            psi, name = build_state()

            rho = q.ket2dm(psi)

            x = np.linspace(-5.5, 5.5, 200)

            W = q.wigner(
                rho,
                x,
                x,
            )

            fig, ax = plt.subplots(
                figsize=(6, 5)
            )

            norm= mcolors.TwoSlopeNorm(vcenter=0, vmax=np.max(W)+0.1, vmin=np.min(W)-0.1)
            im = ax.contourf(
                x,
                x,
                W,
                levels=100,
                cmap="RdBu_r", norm=norm
            )

            fig.colorbar(
                im,
                ax=ax,
                label=r"$W(x,p)$",
            )

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p$")
            ax.set_title(name)
            ax.set_aspect("equal")

            plt.tight_layout()
            plt.show()

    # --------------------------------------------------------
    # Confirm
    # --------------------------------------------------------

    def confirm_state(_):

        psi, name = build_state()

        result["state"] = psi
        result["rho_true"] = q.ket2dm(psi)
        result["name"] = name
        result["confirmed"] = True

        with output:

            clear_output(wait=True)

            print(f"Confirmed: {name}")

            # Plot final confirmed Wigner function
            x = np.linspace(-5.5, 5.5, 200)

            W = q.wigner(
                result["rho_true"],
                x,
                x,
            )

            fig, ax = plt.subplots(
                figsize=(6, 5)
            )

            norm= mcolors.TwoSlopeNorm(vcenter=0, vmax=np.max(W)+0.1, vmin=np.min(W)-0.1)
            
            im = ax.contourf(
                x,
                x,
                W,
                levels=100,
                cmap="RdBu_r", norm =norm
            )

            fig.colorbar(
                im,
                ax=ax,
                label=r"$W(x,p)$",
            )

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p$")
            ax.set_title(
                f"Confirmed: {name}"
            )

            ax.set_aspect("equal")

            plt.tight_layout()
            plt.show()

    confirm_button.on_click(confirm_state)

    # --------------------------------------------------------
    # Show / hide controls
    # --------------------------------------------------------

    def refresh_controls(*args):

        if state_dropdown.value == "Cat":

            parity_widget.layout.display = ""
            alpha_widget.layout.display = ""
            phase_widget.layout.display = ""

            r_widget.layout.display = "none"

        else:

            parity_widget.layout.display = "none"
            alpha_widget.layout.display = "none"
            phase_widget.layout.display = "none"

            r_widget.layout.display = ""

        update_preview()

    # --------------------------------------------------------
    # Observers
    # --------------------------------------------------------

    state_dropdown.observe(
        refresh_controls,
        names="value",
    )

    parity_widget.observe(
        update_preview,
        names="value",
    )

    alpha_widget.observe(
        update_preview,
        names="value",
    )

    phase_widget.observe(
        update_preview,
        names="value",
    )

    r_widget.observe(
        update_preview,
        names="value",
    )

    # --------------------------------------------------------
    # Initial preview
    # --------------------------------------------------------

    refresh_controls()

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display(
        widgets.VBox([
            state_dropdown,
            parity_widget,
            alpha_widget,
            phase_widget,
            r_widget,
            confirm_button,
            output,
        ])
    )

    # Return the result container
    return result