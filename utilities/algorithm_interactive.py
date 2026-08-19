import numpy as np
import matplotlib.pyplot as plt
import qutip as q

from ipywidgets import (
    FloatSlider, FloatLogSlider, IntSlider,
    ToggleButtons, Button, VBox, HBox, Layout, Output, HTML, Tab
)
from IPython.display import clear_output, display

from utilities.artificial_samples import sample_homodyne
from utilities.MLE_class import MLE
from utilities.plotting import plot_fidelities, plot_Wigner

import utilities.state_selector_interactive as state_selector

# The result of the last run lives here once "Run reconstruction" is clicked.
_last_result = None
_last_mle = None
_last_target = None


def result():
    """Return the RunResult from the last reconstruction run (or None)."""
    return _last_result


def mle():
    """Return the MLE object from the last reconstruction run (or None)."""
    return _last_mle


###################################################################
# Widgets: sampling

# x range/points are not set here; they are read from the state selector's
# "Plot settings" (state_selector.x_range_slider / x_points_slider) at run time.
x_range_label = HTML()

n_angles_slider = IntSlider(value=4, min=1, max=32, step=1, description='Angles:')
n_samples_slider = IntSlider(value=1000, min=100, max=10000, step=100,
                              description='Samples/angle:',
                              style={'description_width': '110px'})
n_bins_slider = IntSlider(value=30, min=2, max=200, step=1, description='Bins:')

sampling_box = VBox([x_range_label, n_angles_slider, n_samples_slider, n_bins_slider])


def _update_x_range_label(change=None):
    xmin, xmax = state_selector.x_range_slider.value
    npoints = state_selector.x_points_slider.value
    x_range_label.value = (
        f'<i>x range: [{xmin:.1f}, {xmax:.1f}], {npoints} points '
        f'(from Plot settings above)</i>'
    )

###################################################################
# Widgets: initial state for the algorithm

initial_state_toggle = ToggleButtons(
    options=[('Fock state', 'fock'), ('Maximally mixed', 'mixed')],
    description='Initial state:', style={'description_width': '110px'},
)

initial_fock_n_slider = IntSlider(value=0, min=0, max=49, step=1, description='n:')
initial_mixed_nmax_slider = IntSlider(value=15, min=1, max=100, step=1, description='n_max:')

initial_state_panels = {
    'fock': VBox([initial_fock_n_slider]),
    'mixed': VBox([initial_mixed_nmax_slider]),
}

initial_state_param_box = VBox([initial_state_panels['fock']])
initial_state_box = VBox([initial_state_toggle, initial_state_param_box])

###################################################################
# Widgets: run mode

run_mode_toggle = ToggleButtons(
    options=[('Fixed iterations', 'fixed'), ('Run until convergence', 'convergence')],
    description='Run mode:', style={'description_width': '110px'},
)

n_iterations_slider = IntSlider(value=50, min=1, max=1000, step=1, description='Iterations:')
max_iter_slider = IntSlider(value=300, min=1, max=2000, step=1, description='Max iter:')
threshold_slider = FloatLogSlider(value=1e-8, base=10, min=-12, max=-2, step=0.5,
                                   description='Threshold:',
                                   style={'description_width': '110px'})

run_mode_panels = {
    'fixed': VBox([n_iterations_slider]),
    'convergence': VBox([max_iter_slider, threshold_slider]),
}

run_mode_param_box = VBox([run_mode_panels['fixed']])
run_mode_box = VBox([run_mode_toggle, run_mode_param_box])

###################################################################
# Run button + status + output

run_button = Button(description='Run reconstruction', button_style='success', icon='play')
run_label = HTML(value='<i>Not run yet.</i>')

reconstructed_output = Output()
convergence_output = Output()
explorer_slider = IntSlider(value=0, min=0, max=0, step=1, description='Iteration:')
explorer_output = Output()
explorer_box = VBox([explorer_slider, explorer_output])

run_output = Tab(children=[reconstructed_output, convergence_output, explorer_box])
run_output.set_title(0, 'Reconstructed state')
run_output.set_title(1, 'Convergence')
run_output.set_title(2, 'Iteration explorer')

interface = VBox([
    HTML('<b>Sampling</b>'), sampling_box,
    HTML('<b>Initial state</b>'), initial_state_box,
    HTML('<b>Run mode</b>'), run_mode_box,
    HBox([run_button, run_label]),
    run_output,
])


###################################################################
# Callbacks

def _update_initial_state_panel(change):
    initial_state_param_box.children = [initial_state_panels[change['new']]]


def _update_run_mode_panel(change):
    run_mode_param_box.children = [run_mode_panels[change['new']]]


def _build_initial_rho(cutoff):
    if initial_state_toggle.value == 'fock':
        n = min(initial_fock_n_slider.value, cutoff - 1)
        rho = np.zeros((cutoff, cutoff), dtype=complex)
        rho[n, n] = 1.0
        return rho
    else:
        n_max = min(initial_mixed_nmax_slider.value, cutoff)
        rho = np.zeros((cutoff, cutoff), dtype=complex)
        rho[:n_max, :n_max] = np.eye(n_max) / n_max
        return rho


def _current_xy_vec():
    xmin, xmax = state_selector.x_range_slider.value
    npoints = state_selector.x_points_slider.value
    return np.linspace(xmin, xmax, npoints)


def _display_wigner(output, rho, title, info_html):
    xvec = _current_xy_vec()
    with output:
        clear_output(wait=True)
        display(HTML(info_html))
        ax = plot_Wigner(q.Qobj(rho), xvec=xvec, pvec=xvec, title=title, colorbar=True)
        plt.tight_layout()
        display(ax.figure)
        plt.close(ax.figure)


def _update_explorer(change=None):
    if _last_result is None or _last_target is None:
        return
    idx = explorer_slider.value
    rho = _last_result.states[idx]
    fid_target = q.fidelity(q.Qobj(rho), _last_target) ** 2
    fid_prev = _last_result.step_fidelities[idx - 1] if idx > 0 else 0.0
    info = (f'<b>Fidelity to target:</b> {fid_target:.6f} &nbsp;|&nbsp; '
            f'<b>Fidelity to previous:</b> {fid_prev:.6f}')
    _display_wigner(explorer_output, rho, f'Iteration {idx}', info)


def run_reconstruction(button=None):
    global _last_result, _last_mle, _last_target

    target_state = state_selector.state()
    if target_state is None:
        run_label.value = '<span style="color: red;">Select and confirm a target state above first.</span>'
        return

    run_label.value = '<i>Running...</i>'

    cutoff = target_state.shape[0]
    xmin, xmax = state_selector.x_range_slider.value
    npoints = state_selector.x_points_slider.value
    x_vec = np.linspace(xmin, xmax, npoints)

    # Simulate homodyne samples for the confirmed target state
    _, samples = sample_homodyne(
        target_state,
        n_angles=n_angles_slider.value,
        n_samples_theta=n_samples_slider.value,
        x_vec=x_vec,
    )

    initial_rho = _build_initial_rho(cutoff)

    MaxLik = MLE(
        data=samples,
        N_bins=n_bins_slider.value,
        initial_rho=initial_rho,
        x_lims=(xmin, xmax),
        x_points=len(x_vec),
    )

    if run_mode_toggle.value == 'fixed':
        run = MaxLik.run_ntimes(n_iterations_slider.value, store_states=True)
    else:
        run = MaxLik.run_setpoint(
            threshold=threshold_slider.value,
            max_iter=max_iter_slider.value,
            store_states=True,
        )

    _last_result = run
    _last_mle = MaxLik
    _last_target = target_state

    n_iter_done = len(run.step_fidelities)
    final_fid = run.step_fidelities[-1] if run.step_fidelities else float('nan')
    run_label.value = (f'<b>Done.</b> {n_iter_done} iterations, '
                        f'final step fidelity {final_fid:.6g}')

    final_fid_target = q.fidelity(q.Qobj(run.rho_out), target_state) ** 2
    _display_wigner(
        reconstructed_output, run.rho_out, 'Reconstructed state',
        f'<b>Fidelity to target:</b> {final_fid_target:.6f}',
    )

    with convergence_output:
        clear_output(wait=True)
        plot_fidelities(run.step_fidelities,
                         title='Fidelity between consecutive iterations')

    explorer_slider.max = len(run.states) - 1
    explorer_slider.value = explorer_slider.max
    _update_explorer()


run_button.on_click(run_reconstruction)
initial_state_toggle.observe(_update_initial_state_panel, names='value')
run_mode_toggle.observe(_update_run_mode_panel, names='value')
explorer_slider.observe(_update_explorer, names='value')
state_selector.x_range_slider.observe(_update_x_range_label, names='value')
state_selector.x_points_slider.observe(_update_x_range_label, names='value')


def show():
    _update_x_range_label()
    display(interface)