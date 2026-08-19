import numpy as np
import matplotlib.pyplot as plt
import qutip as q

from ipywidgets import (
    FloatLogSlider,
    IntSlider,
    ToggleButtons,
    Button,
    VBox,
    HBox,
    Layout,
    Output,
    HTML,
    Tab,
)
from IPython.display import clear_output, display

from utilities.artificial_samples import sample_homodyne
from utilities.MLE_class import MLE
from utilities.plotting import plot_fidelities, plot_Wigner

import utilities.state_selector_interactive as state_selector


# =====================================================================
# Global state
# =====================================================================

# Result of the last reconstruction
_last_result = None
_last_mle = None
_last_target = None

# ---------------------------------------------------------------------
# Sampling data
#
# _current_samples:
#     Original UNBINNED homodyne samples.
#     These are passed to MLE.
#
# _current_binned_samples:
#     Visualization-only representation of the same samples.
#     These are recalculated whenever N_bins changes.
#
# _current_thetas:
#     LO phases associated with the samples.
#
# _current_sampling_state:
#     Target state used to generate the current samples.
# ---------------------------------------------------------------------

_current_thetas = None
_current_samples = None
_current_binned_samples = None
_current_bin_edges = None
_current_sampling_state = None


def result():
    """Return the RunResult from the last reconstruction run."""
    return _last_result


def mle():
    """Return the MLE object from the last reconstruction run."""
    return _last_mle


# =====================================================================
# Sampling parameters
# =====================================================================

x_range_label = HTML()

n_angles_slider = IntSlider(
    value=4,
    min=1,
    max=32,
    step=1,
    description='Angles:',
    continuous_update=False,
)

n_samples_slider = IntSlider(
    value=500,
    min=100,
    max=4000,
    step=100,
    description='Samples/angle:',
    style={'description_width': '110px'},
    continuous_update=False,
)

n_bins_slider = IntSlider(
    value=20,
    min=1,
    max=100,
    step=1,
    description='Bins:',
    continuous_update=False,
)

generate_sampling_button = Button(
    description='Generate samples',
    button_style='info',
    icon='refresh',
    layout=Layout(width='180px'),
)

sampling_status = HTML(
    value='<i>No samples generated yet.</i>'
)

sampling_box = VBox([
    x_range_label,
    n_angles_slider,
    n_samples_slider,
    n_bins_slider,
    generate_sampling_button,
    sampling_status,
])


def _update_x_range_label(change=None):
    """Update the label showing the x range inherited from state selector."""

    xmin, xmax = state_selector.x_range_slider.value
    npoints = state_selector.x_points_slider.value

    x_range_label.value = (
        f'<i>'
        f'x range: [{xmin:.1f}, {xmax:.1f}], '
        f'{npoints} points '
        f'(from Plot settings above)'
        f'</i>'
    )


# =====================================================================
# Sampling data / binning
# =====================================================================

def _bin_samples(samples, x_range, n_bins):
    """
    Bin the unbinned homodyne samples for visualization.

    Parameters
    ----------
    samples : ndarray
        Shape (n_angles, n_samples_per_angle).

    x_range : tuple
        (xmin, xmax).

    n_bins : int
        Number of bins.

    Returns
    -------
    binned_samples : ndarray
        Shape (n_angles, n_bins).

        Each row contains the normalized probability density
        in each x bin.

    bin_edges : ndarray
        Shape (n_bins + 1,).

    Notes
    -----
    This does NOT modify the original samples.

    The returned binned data is only for visualization.
    """

    xmin, xmax = x_range

    bin_edges = np.linspace(
        xmin,
        xmax,
        n_bins + 1,
    )

    binned_samples = np.zeros(
        (samples.shape[0], n_bins),
        dtype=float,
    )

    bin_width = bin_edges[1] - bin_edges[0]

    for i, samples_theta in enumerate(samples):

        counts, _ = np.histogram(
            samples_theta,
            bins=bin_edges,
        )

        # Convert counts to probability density.
        #
        # Integral of the histogram over x is then approximately 1.
        binned_samples[i] = (
            counts
            / (
                len(samples_theta)
                * bin_width
            )
        )

    return binned_samples, bin_edges


# =====================================================================
# Homodyne plot
# =====================================================================

homodyne_output = Output()

homodyne_box = VBox([
    homodyne_output,
])


def _create_homodyne_plot(
    thetas,
    samples,
    binned_samples,
    bin_edges,
    xrange,
):
    """
    Construct the interactive homodyne visualization.

    The raw samples are shown in the upper panel.

    The lower panel shows the binned marginal distribution for
    the selected LO phase.

    Parameters
    ----------
    thetas : ndarray
        LO phases.

    samples : ndarray
        Original unbinned samples.

    binned_samples : ndarray
        Histogram probability densities.

    bin_edges : ndarray
        Histogram bin edges.

    xrange : tuple
        x-axis range.
    """

    theta_slider = IntSlider(
        value=0,
        min=0,
        max=len(thetas) - 1,
        step=1,
        description='LO phase:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=Layout(width='500px'),
    )

    output = Output()

    theta_width = 0.025

    # Fixed RNG so jitter does not change when moving the slider.
    rng = np.random.default_rng(42)

    def plot_homodyne(theta_index=0):

        theta_selected = thetas[theta_index]

        x_selected = samples[theta_index]

        probability_selected = (
            binned_samples[theta_index]
        )

        with output:

            clear_output(wait=True)

            fig = plt.figure(
                figsize=(10, 7),
                constrained_layout=True,
            )

            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[3, 2],
            )

            # =====================================================
            # Top plot: all homodyne measurements
            # =====================================================

            ax1 = fig.add_subplot(gs[0])

            theta_scatter = np.repeat(
                thetas,
                samples.shape[1],
            )

            x_scatter = samples.flatten()

            theta_jitter = rng.uniform(
                -theta_width,
                theta_width,
                size=theta_scatter.shape,
            )

            ax1.scatter(
                theta_scatter + theta_jitter,
                x_scatter,
                s=4,
                alpha=0.12,
                linewidths=0,
            )

            # -----------------------------------------------------
            # Selected LO phase
            # -----------------------------------------------------

            selected_theta_jitter = rng.uniform(
                -theta_width,
                theta_width,
                size=len(x_selected),
            )

            ax1.scatter(
                theta_selected + selected_theta_jitter,
                x_selected,
                s=10,
                alpha=0.55,
                linewidths=0,
            )

            ax1.axvline(
                theta_selected,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
            )

            ax1.set_xlabel(
                r'LO phase $\theta$',
                fontsize=13,
            )

            ax1.set_ylabel(
                r'Quadrature $x_\theta$',
                fontsize=13,
            )

            ax1.set_title(
                'Homodyne Measurements',
                fontsize=14,
            )

            # =====================================================
            # Bottom plot: BINNED marginal distribution
            # =====================================================

            ax2 = fig.add_subplot(gs[1])

            # -----------------------------------------------------
            # Draw the actual binned distribution
            #
            # The values are probability densities and therefore
            # the integral of the histogram is approximately 1.
            # -----------------------------------------------------

            bin_width = (
                bin_edges[1]
                - bin_edges[0]
            )

            bin_centers = (
                0.5
                * (
                    bin_edges[:-1]
                    + bin_edges[1:]
                )
            )

            ax2.bar(
                bin_centers,
                probability_selected,
                width=bin_width,
                align='center',
                alpha=0.75,
                edgecolor='none',
            )

            # -----------------------------------------------------
            # Optional smooth KDE for comparison
            # -----------------------------------------------------

            x_kde = np.linspace(
                xrange[0],
                xrange[1],
                500,
            )

            std = np.std(x_selected)
            n = len(x_selected)

            bandwidth = (
                1.06
                * std
                * n ** (-1 / 5)
            )

            bandwidth = max(
                bandwidth,
                1e-3,
            )

            kde = np.zeros_like(
                x_kde
            )

            # Chunked calculation to avoid huge arrays
            chunk_size = 500

            for start in range(
                0,
                len(x_selected),
                chunk_size,
            ):

                x_chunk = x_selected[
                    start:start + chunk_size
                ]

                kde += np.sum(
                    np.exp(
                        -0.5
                        * (
                            (
                                x_kde[:, None]
                                - x_chunk[None, :]
                            )
                            / bandwidth
                        ) ** 2
                    ),
                    axis=1,
                )

            kde /= (
                len(x_selected)
                * bandwidth
                * np.sqrt(2 * np.pi)
            )

            ax2.plot(
                x_kde,
                kde,
                linewidth=2.5,
                label='KDE',
            )

            ax2.set_xlabel(
                r'Quadrature $x_\theta$',
                fontsize=13,
            )

            ax2.set_ylabel(
                r'$P(x_\theta)$',
                fontsize=13,
            )

            ax2.set_title(
                rf'Binned Marginal Distribution at '
                rf'$\theta={theta_selected:.3f}$ rad '
                rf'({len(probability_selected)} bins)',
                fontsize=14,
            )

            ax2.set_xlim(
                xrange
            )

            ax2.grid(
                alpha=0.2,
            )

            ax2.legend()

            display(fig)

            plt.close(fig)

    # -------------------------------------------------------------
    # Slider callback
    # -------------------------------------------------------------

    def update_plot(change):

        plot_homodyne(
            change['new']
        )

    theta_slider.observe(
        update_plot,
        names='value',
    )

    # Initial plot
    plot_homodyne(0)

    return VBox([
        theta_slider,
        output,
    ])


# =====================================================================
# Generate samples
# =====================================================================

def _generate_sampling_preview(button=None):
    """
    Generate a new homodyne experiment.

    This regenerates the actual random samples.

    N_bins determines the visualization binning but does NOT alter
    the stored raw samples.
    """

    global _current_thetas
    global _current_samples
    global _current_binned_samples
    global _current_bin_edges
    global _current_sampling_state

    target_state = state_selector.state()

    # -------------------------------------------------------------
    # Check target state
    # -------------------------------------------------------------

    if target_state is None:

        sampling_status.value = (
            '<span style="color: red;">'
            'Select and confirm a target state above first.'
            '</span>'
        )

        with homodyne_output:

            clear_output(wait=True)

            display(
                HTML(
                    '<i>'
                    'No target state selected.'
                    '</i>'
                )
            )

        return

    # -------------------------------------------------------------
    # x range
    # -------------------------------------------------------------

    xmin, xmax = (
        state_selector.x_range_slider.value
    )

    npoints = (
        state_selector.x_points_slider.value
    )

    sampling_status.value = (
        '<i>Generating homodyne samples...</i>'
    )

    # -------------------------------------------------------------
    # Generate UNBINNED samples
    # -------------------------------------------------------------

    try:

        thetas, samples = sample_homodyne(
            state=target_state,
            n_angles=n_angles_slider.value,
            n_samples_theta=n_samples_slider.value,
            bin_data=False,
            x_vec=np.linspace(
                xmin,
                xmax,
                npoints,
            ),
        )

    except Exception as exc:

        sampling_status.value = (
            '<span style="color: red;">'
            f'Error generating samples: {exc}'
            '</span>'
        )

        with homodyne_output:

            clear_output(wait=True)

        return

    thetas = np.asarray(thetas)
    samples = np.asarray(samples)

    # -------------------------------------------------------------
    # Store ORIGINAL unbinned samples
    # -------------------------------------------------------------

    _current_thetas = thetas
    _current_samples = samples
    _current_sampling_state = target_state.copy()

    # -------------------------------------------------------------
    # Bin the samples for visualization
    # -------------------------------------------------------------

    (
        _current_binned_samples,
        _current_bin_edges,
    ) = _bin_samples(
        samples=samples,
        x_range=(xmin, xmax),
        n_bins=n_bins_slider.value,
    )

    # -------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------

    widget = _create_homodyne_plot(
        thetas=_current_thetas,
        samples=_current_samples,
        binned_samples=_current_binned_samples,
        bin_edges=_current_bin_edges,
        xrange=(xmin, xmax),
    )

    # -------------------------------------------------------------
    # Display
    # -------------------------------------------------------------

    with homodyne_output:

        clear_output(wait=True)

        display(widget)

    total_samples = (
        n_angles_slider.value
        * n_samples_slider.value
    )

    sampling_status.value = (
        f'<b>Sampling data ready.</b> '
        f'{n_angles_slider.value} angles × '
        f'{n_samples_slider.value} samples/angle '
        f'= {total_samples:,} raw measurements. '
        f'Displayed using {n_bins_slider.value} bins.'
    )


# =====================================================================
# Update only the binning
# =====================================================================

def _update_sampling_bins(change=None):
    """
    Re-bin the EXISTING raw samples.

    This does not generate a new random experiment.

    This is the callback used when N_bins changes.
    """

    global _current_binned_samples
    global _current_bin_edges

    # No samples yet
    if _current_samples is None:
        sampling_status.value = (
            '<i>'
            'Set the sampling parameters and generate samples.'
            '</i>'
        )
        return

    xmin, xmax = (
        state_selector.x_range_slider.value
    )

    # -------------------------------------------------------------
    # Re-bin existing samples
    # -------------------------------------------------------------

    (
        _current_binned_samples,
        _current_bin_edges,
    ) = _bin_samples(
        samples=_current_samples,
        x_range=(xmin, xmax),
        n_bins=n_bins_slider.value,
    )

    # -------------------------------------------------------------
    # Recreate visualization
    # -------------------------------------------------------------

    widget = _create_homodyne_plot(
        thetas=_current_thetas,
        samples=_current_samples,
        binned_samples=_current_binned_samples,
        bin_edges=_current_bin_edges,
        xrange=(xmin, xmax),
    )

    with homodyne_output:

        clear_output(wait=True)

        display(widget)

    total_samples = (
        _current_samples.shape[0]
        * _current_samples.shape[1]
    )

    sampling_status.value = (
        f'<b>Sampling data ready.</b> '
        f'{total_samples:,} raw measurements. '
        f'Displayed using {n_bins_slider.value} bins.'
    )


# =====================================================================
# Invalidate sampling data
# =====================================================================

def _invalidate_sampling(change=None):
    """
    Invalidate the current experiment when a parameter changes that
    changes the actual simulated measurement data.
    """

    global _current_thetas
    global _current_samples
    global _current_binned_samples
    global _current_bin_edges
    global _current_sampling_state

    _current_thetas = None
    _current_samples = None
    _current_binned_samples = None
    _current_bin_edges = None
    _current_sampling_state = None

    sampling_status.value = (
        '<i>'
        'Sampling parameters changed. '
        'Generate new samples.'
        '</i>'
    )

    with homodyne_output:

        clear_output(wait=True)


# =====================================================================
# Algorithm parameters: initial state
# =====================================================================

initial_state_toggle = ToggleButtons(
    options=[
        ('Fock state', 'fock'),
        ('Maximally mixed', 'mixed'),
    ],
    description='Initial state:',
    style={'description_width': '110px'},
)

initial_fock_n_slider = IntSlider(
    value=0,
    min=0,
    max=49,
    step=1,
    description='n:',
)

initial_mixed_nmax_slider = IntSlider(
    value=10,
    min=1,
    max=49,
    step=1,
    description='n_max:',
)

initial_state_panels = {
    'fock': VBox([
        initial_fock_n_slider
    ]),

    'mixed': VBox([
        initial_mixed_nmax_slider
    ]),
}

initial_state_param_box = VBox([
    initial_state_panels['fock']
])

initial_state_box = VBox([
    initial_state_toggle,
    initial_state_param_box,
])


# =====================================================================
# Algorithm parameters: run mode
# =====================================================================

run_mode_toggle = ToggleButtons(
    options=[
        ('Fixed iterations', 'fixed'),
        ('Run until convergence', 'convergence'),
    ],
    description='Run mode:',
    style={'description_width': '110px'},
)

n_iterations_slider = IntSlider(
    value=20,
    min=1,
    max=1000,
    step=1,
    description='Iterations:',
)

max_iter_slider = IntSlider(
    value=100,
    min=1,
    max=1000,
    step=1,
    description='Max iter:',
)

threshold_slider = FloatLogSlider(
    value=1e-8,
    base=10,
    min=-12,
    max=-2,
    step=0.5,
    description='Threshold:',
    style={'description_width': '110px'},
)

run_mode_panels = {
    'fixed': VBox([
        n_iterations_slider
    ]),

    'convergence': VBox([
        max_iter_slider,
        threshold_slider,
    ]),
}

run_mode_param_box = VBox([
    run_mode_panels['fixed']
])

run_mode_box = VBox([
    run_mode_toggle,
    run_mode_param_box,
])


# =====================================================================
# Complete algorithm parameter box
# =====================================================================

algorithm_box = VBox([
    HTML('<b>Initial state</b>'),
    initial_state_box,

    HTML('<br><b>Run mode</b>'),
    run_mode_box,
])


# =====================================================================
# Run button + status
# =====================================================================

run_button = Button(
    description='Run reconstruction',
    button_style='success',
    icon='play',
)

run_label = HTML(
    value='<i>Generate samples before running reconstruction.</i>'
)


# =====================================================================
# Reconstruction outputs
# =====================================================================

reconstructed_output = Output()

convergence_output = Output()

convergence_scale_box = HBox([
    state_selector.fidelity_xscale_dropdown,
    state_selector.fidelity_yscale_dropdown,
])

convergence_box = VBox([
    convergence_scale_box,
    convergence_output,
])

explorer_slider = IntSlider(
    value=0,
    min=0,
    max=0,
    step=1,
    description='Iteration:',
)

explorer_output = Output()

explorer_box = VBox([
    explorer_slider,
    explorer_output,
])

run_output = Tab(
    children=[
        reconstructed_output,
        convergence_box,
        explorer_box,
    ]
)

run_output.set_title(
    0,
    'Reconstructed state',
)

run_output.set_title(
    1,
    'Convergence',
)

run_output.set_title(
    2,
    'Iteration explorer',
)


# =====================================================================
# Complete interface
# =====================================================================

interface = VBox([

    # -------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------

    HTML('<h3>Sampling Parameters</h3>'),

    sampling_box,

    HTML('<h4>Simulated Homodyne Data</h4>'),

    homodyne_box,

    HTML('<hr>'),

    # -------------------------------------------------------------
    # Algorithm
    # -------------------------------------------------------------

    HTML('<h3>Algorithm Parameters</h3>'),

    algorithm_box,

    HTML('<hr>'),

    # -------------------------------------------------------------
    # Run
    # -------------------------------------------------------------

    HBox([
        run_button,
        run_label,
    ]),

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    run_output,
])


# =====================================================================
# Callbacks
# =====================================================================

def _update_initial_state_panel(change):
    """Show the parameters for the selected initial state."""

    initial_state_param_box.children = [
        initial_state_panels[change['new']]
    ]


def _update_run_mode_panel(change):
    """Show parameters for the selected run mode."""

    run_mode_param_box.children = [
        run_mode_panels[change['new']]
    ]


# =====================================================================
# Initial density matrix
# =====================================================================

def _build_initial_rho(cutoff):
    """
    Build the initial density matrix selected by the user.
    """

    if initial_state_toggle.value == 'fock':

        n = min(
            initial_fock_n_slider.value,
            cutoff - 1,
        )

        rho = np.zeros(
            (cutoff, cutoff),
            dtype=complex,
        )

        rho[n, n] = 1.0

        return rho

    else:

        n_max = min(
            initial_mixed_nmax_slider.value,
            cutoff,
        )

        rho = np.zeros(
            (cutoff, cutoff),
            dtype=complex,
        )

        rho[:n_max, :n_max] = (
            np.eye(n_max)
            / n_max
        )

        return rho


# =====================================================================
# x vector
# =====================================================================

def _current_xy_vec():
    """
    Return the x vector specified by the state selector.
    """

    xmin, xmax = (
        state_selector.x_range_slider.value
    )

    npoints = (
        state_selector.x_points_slider.value
    )

    return np.linspace(
        xmin,
        xmax,
        npoints,
    )


# =====================================================================
# Convergence plot
# =====================================================================

def _update_convergence_plot(change=None):

    if _last_result is None:
        return

    with convergence_output:

        clear_output(wait=True)

        plot_fidelities(
            _last_result.step_fidelities,
            title='Fidelity between consecutive iterations',
            xscale=(
                state_selector
                .fidelity_xscale_dropdown
                .value
            ),
            yscale=(
                state_selector
                .fidelity_yscale_dropdown
                .value
            ),
        )


# =====================================================================
# Wigner display
# =====================================================================

def _display_wigner(
    output,
    rho,
    title,
    info_html,
):
    """
    Display a Wigner function inside an Output widget.
    """

    xvec = _current_xy_vec()

    with output:

        clear_output(wait=True)

        display(
            HTML(info_html)
        )

        ax = plot_Wigner(
            q.Qobj(rho),
            xvec=xvec,
            pvec=xvec,
            title=title,
            colorbar=True,
        )

        plt.tight_layout()

        display(
            ax.figure
        )

        plt.close(
            ax.figure
        )


# =====================================================================
# Iteration explorer
# =====================================================================

def _update_explorer(change=None):

    if (
        _last_result is None
        or _last_target is None
    ):
        return

    idx = explorer_slider.value

    rho = _last_result.states[idx]

    fid_target = (
        q.fidelity(
            q.Qobj(rho),
            _last_target,
        )
        ** 2
    )

    fid_prev = (
        _last_result.step_fidelities[idx - 1]
        if idx > 0
        else 0.0
    )

    info = (
        f'<b>Fidelity to target:</b> '
        f'{fid_target:.6f}'
        f' &nbsp;|&nbsp; '
        f'<b>Fidelity to previous:</b> '
        f'{fid_prev:.6f}'
    )

    _display_wigner(
        explorer_output,
        rho,
        f'Iteration {idx}',
        info,
    )


# =====================================================================
# Run reconstruction
# =====================================================================

def run_reconstruction(button=None):

    global _last_result
    global _last_mle
    global _last_target

    # -------------------------------------------------------------
    # Check target state
    # -------------------------------------------------------------

    target_state = state_selector.state()

    if target_state is None:

        run_label.value = (
            '<span style="color: red;">'
            'Select and confirm a target state above first.'
            '</span>'
        )

        return

    # -------------------------------------------------------------
    # Check sampling data
    # -------------------------------------------------------------

    if _current_samples is None:

        run_label.value = (
            '<span style="color: red;">'
            'Generate the homodyne samples first.'
            '</span>'
        )

        return

    # -------------------------------------------------------------
    # Start
    # -------------------------------------------------------------

    run_label.value = '<i>Running...</i>'

    cutoff = target_state.shape[0]

    xmin, xmax = (
        state_selector.x_range_slider.value
    )

    npoints = (
        state_selector.x_points_slider.value
    )

    x_vec = np.linspace(
        xmin,
        xmax,
        npoints,
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # MLE receives the ORIGINAL UNBINNED samples.
    #
    # N_bins is passed separately to MLE.
    # -------------------------------------------------------------

    unbinned_samples = _current_samples

    # -------------------------------------------------------------
    # Initial state
    # -------------------------------------------------------------

    initial_rho = _build_initial_rho(
        cutoff
    )

    # -------------------------------------------------------------
    # Create MLE object
    # -------------------------------------------------------------

    MaxLik = MLE(
        data=unbinned_samples,
        N_bins=n_bins_slider.value,
        initial_rho=initial_rho,
        x_lims=(xmin, xmax),
        x_points=len(x_vec),
    )

    # -------------------------------------------------------------
    # Run algorithm
    # -------------------------------------------------------------

    if run_mode_toggle.value == 'fixed':

        run = MaxLik.run_ntimes(
            n_iterations_slider.value,
            store_states=True,
        )

    else:

        run = MaxLik.run_setpoint(
            threshold=threshold_slider.value,
            max_iter=max_iter_slider.value,
            store_states=True,
        )

    # -------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------

    _last_result = run
    _last_mle = MaxLik
    _last_target = target_state

    # -------------------------------------------------------------
    # Status
    # -------------------------------------------------------------

    n_iter_done = len(
        run.step_fidelities
    )

    final_fid = (
        run.step_fidelities[-1]
        if run.step_fidelities
        else float('nan')
    )

    run_label.value = (
        f'<b>Done.</b> '
        f'{n_iter_done} iterations, '
        f'final step fidelity '
        f'{final_fid:.6g}'
    )

    # -------------------------------------------------------------
    # Reconstructed state
    # -------------------------------------------------------------

    final_fid_target = (
        q.fidelity(
            q.Qobj(run.rho_out),
            target_state,
        )
        ** 2
    )

    _display_wigner(
        reconstructed_output,
        run.rho_out,
        'Reconstructed state',
        (
            f'<b>Fidelity to target:</b> '
            f'{final_fid_target:.6f}'
        ),
    )

    # -------------------------------------------------------------
    # Convergence
    # -------------------------------------------------------------

    _update_convergence_plot()

    # -------------------------------------------------------------
    # Iteration explorer
    # -------------------------------------------------------------

    explorer_slider.max = (
        len(run.states) - 1
    )

    explorer_slider.value = (
        explorer_slider.max
    )

    _update_explorer()


# =====================================================================
# Register callbacks
# =====================================================================

# -------------------------------------------------------------
# Generate new sampling experiment
# -------------------------------------------------------------

generate_sampling_button.on_click(
    _generate_sampling_preview
)

# These parameters change the actual experiment.
n_angles_slider.observe(
    _invalidate_sampling,
    names='value',
)

n_samples_slider.observe(
    _invalidate_sampling,
    names='value',
)

# -------------------------------------------------------------
# N_bins ONLY changes the representation of the existing samples.
# -------------------------------------------------------------

n_bins_slider.observe(
    _update_sampling_bins,
    names='value',
)

# -------------------------------------------------------------
# x-range / x-points affect the sampling experiment
# -------------------------------------------------------------

state_selector.x_range_slider.observe(
    _invalidate_sampling,
    names='value',
)

state_selector.x_points_slider.observe(
    _invalidate_sampling,
    names='value',
)

# -------------------------------------------------------------
# Initial state UI
# -------------------------------------------------------------

initial_state_toggle.observe(
    _update_initial_state_panel,
    names='value',
)

# -------------------------------------------------------------
# Run mode UI
# -------------------------------------------------------------

run_mode_toggle.observe(
    _update_run_mode_panel,
    names='value',
)

# -------------------------------------------------------------
# Iteration explorer
# -------------------------------------------------------------

explorer_slider.observe(
    _update_explorer,
    names='value',
)

# -------------------------------------------------------------
# x range label
# -------------------------------------------------------------

state_selector.x_range_slider.observe(
    _update_x_range_label,
    names='value',
)

state_selector.x_points_slider.observe(
    _update_x_range_label,
    names='value',
)

# -------------------------------------------------------------
# Convergence plot
# -------------------------------------------------------------

state_selector.fidelity_xscale_dropdown.observe(
    _update_convergence_plot,
    names='value',
)

state_selector.fidelity_yscale_dropdown.observe(
    _update_convergence_plot,
    names='value',
)

# -------------------------------------------------------------
# Run
# -------------------------------------------------------------

run_button.on_click(
    run_reconstruction
)


# =====================================================================
# Public display function
# =====================================================================

def show():
    """
    Display the complete maximum-likelihood reconstruction interface.
    """

    _update_x_range_label()

    display(interface)