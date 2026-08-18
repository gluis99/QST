import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import (
    FloatSlider, IntSlider, Dropdown, Button, Accordion,
    HBox, VBox, Layout, FloatRangeSlider, Output, HTML
)
from IPython.display import clear_output, display

from utilities.functions import fock, cat, sqv, ON_state
from utilities.plotting import plot_Wigner

dB_to_r = lambda dB: dB * np.log(10) / 20  # Convert squeezing dB to squeezing parameter r
r_to_dB = lambda r: 20 * r / np.log(10)  # Convert squeezing parameter r to dB

# The confirmed state lives here once "Confirm" is clicked.
_selected_state = None
_selected_params = None


def state():
    """Return the state that was confirmed in the interface (or None)."""
    return _selected_state


def state_params():
    """Return the parameters that produced the confirmed state (or None)."""
    return _selected_params

###################################################################
# Widgets for state selection

fock_cutoff_slider = IntSlider(value=50, min=10, max=100, 
                               step=1, description='Fock cutoff:')

state_type_dropdown = Dropdown(options=[('Fock State', 'fock'),
                                      ('Cat State', 'cat'),
                                      ('Squeezed Vacuum', 'sqv'),
                                      ('ON State', 'on')],
                                value='fock', description='State:')

# Fock state parameters n
fock_n_slider = IntSlider(value=1, min=0, max=fock_cutoff_slider.value-1, 
                          step=1, description='N:')

# Cat state parameters
cat_amplitude_slider = FloatSlider(value=1.0, min=0.1, max=3.0, 
                                   step=0.1, description='|α|:')
cat_phase_slider = FloatSlider(value=0.0, min=0.0, max=2*np.pi, step=0.1, 
                               description='θ [rad]:')
cat_parity_dropdown = Dropdown(options=[('Even', 0), ('Odd', 1)], 
                               value=0, description='Parity:')

# Squeezed vacuum state parameters
squeezing_dB_slider = FloatSlider(value=3.0, min=-12.0, max=12.0, 
                                  step=0.1, description='r [dB]:')
squeezing_phase_slider = FloatSlider(value=0.0, min=0.0, max=2*np.pi, 
                                     step=0.1, description='θ [rad]:')

# ON state parameters
on_n_slider = IntSlider(value=2, min=1, max=fock_cutoff_slider.value-1, 
                        step=1, description='N:')
on_parity_dropdown = Dropdown(options=[('Even', 0), ('Odd', 1)], 
                              value=0, description='Parity:')

# Subwidgets for each state type
fock_state_params = VBox([fock_n_slider])
cat_state_params = VBox([cat_amplitude_slider, cat_phase_slider, cat_parity_dropdown])
squeezing_state_params = VBox([squeezing_dB_slider, squeezing_phase_slider])
on_state_params = VBox([on_n_slider, on_parity_dropdown])

parameter_panels = {
    'fock': fock_state_params,
    'cat': cat_state_params,
    'sqv': squeezing_state_params,
    'on': on_state_params
}

parameter_box = VBox([fock_state_params])

x_range_slider = FloatRangeSlider(
    value=(-5.0, 5.0), min=-10.0, max=10.0, step=0.1,
    readout_format='.1f', description='Range:', continuous_update=False,
)

x_points_slider = IntSlider(
    value=200, min=50, max=1000, step=10,
    description='Points:', continuous_update=False,
)

wigner_output = Output()

confirm_button = Button(description='Confirm state', button_style='success', icon='check')
confirm_label = HTML(value='<i>No state confirmed yet.</i>')

LEFT_COLUMN_WIDTH_PX = 380
ACCORDION_MARGIN_PX = 20
ACCORDION_WIDTH_PX = 420
ROW_WIDTH_PX = LEFT_COLUMN_WIDTH_PX + ACCORDION_MARGIN_PX + ACCORDION_WIDTH_PX

# Wigner figure width matches the row above; height keeps the same aspect ratio.
WIGNER_ASPECT_RATIO = 7.2 / 5.6
WIGNER_FIG_WIDTH_IN = ROW_WIDTH_PX / 100
WIGNER_FIG_HEIGHT_IN = WIGNER_FIG_WIDTH_IN / WIGNER_ASPECT_RATIO

plot_settings_box = VBox([x_range_slider, x_points_slider])
plot_settings_accordion = Accordion(
    children=[plot_settings_box],
    layout=Layout(margin=f'0 0 0 {ACCORDION_MARGIN_PX}px', width=f'{ACCORDION_WIDTH_PX}px')
)
plot_settings_accordion.set_title(0, 'Plot settings')
plot_settings_accordion.selected_index = None

left_column = VBox(
    [fock_cutoff_slider, state_type_dropdown, parameter_box],
    layout=Layout(width=f'{LEFT_COLUMN_WIDTH_PX}px')
)
state_selector = HBox(
    [left_column, plot_settings_accordion],
    layout=Layout(align_items='flex-start')
)
confirm_box = HBox([confirm_button, confirm_label])

interface = VBox([state_selector, wigner_output, confirm_box])


# Compute selected state
def get_selected_state():
    N = fock_cutoff_slider.value
    state_type = state_type_dropdown.value

    if state_type == 'fock':
        n = fock_n_slider.value
        return fock(N, n)

    elif state_type == 'cat':
        amplitude = cat_amplitude_slider.value
        phase = cat_phase_slider.value
        parity = cat_parity_dropdown.value

        alpha = amplitude * np.exp(1j * phase)

        return cat(N, parity, alpha)

    elif state_type == 'sqv':
        r_dB = squeezing_dB_slider.value
        r = dB_to_r(r_dB)
        theta = squeezing_phase_slider.value

        return sqv(N, r, theta)

    elif state_type == 'on':
        n = on_n_slider.value
        parity = on_parity_dropdown.value

        return ON_state(N, n, parity)

# Get parameters for the selected state
def get_selected_params():
    N = fock_cutoff_slider.value
    state_type = state_type_dropdown.value

    if state_type == 'fock':
        return {'fock_cutoff': N, 'state_type': state_type, 'n': fock_n_slider.value}

    elif state_type == 'cat':
        return {'fock_cutoff': N, 'state_type': state_type,
                'amplitude': cat_amplitude_slider.value,
                'phase': cat_phase_slider.value,
                'parity': cat_parity_dropdown.value}

    elif state_type == 'sqv':
        return {'fock_cutoff': N, 'state_type': state_type,
                'squeezing_dB': squeezing_dB_slider.value,
                'phase': squeezing_phase_slider.value}

    elif state_type == 'on':
        return {'fock_cutoff': N, 'state_type': state_type,
                'n': on_n_slider.value,
                'parity': on_parity_dropdown.value}


def update_parameter_panel(change):
    parameter_box.children = [parameter_panels[change['new']]]

# Update maximum N depending on the Fock cutoff
def update_n_max(change):
    new_max = change['new'] - 1
    fock_n_slider.max = new_max
    on_n_slider.max = new_max

# Update plot for selected state
def update_wigner(change=None):
    with wigner_output:
        clear_output(wait=True)

        state = get_selected_state()

        xmin, xmax = x_range_slider.value
        npoints = x_points_slider.value

        xvec = np.linspace(xmin, xmax, npoints)
        pvec = np.linspace(xmin, xmax, npoints)

        plot_Wigner(
            state,
            xvec=xvec,
            pvec=pvec,
            title=f"{state_type_dropdown.label} Wigner function",
            fig_size=(WIGNER_FIG_WIDTH_IN, WIGNER_FIG_HEIGHT_IN),
            colorbar=True
        )

        plt.tight_layout()
        plt.show()


def confirm_state(button=None):
    global _selected_state, _selected_params

    _selected_state = get_selected_state()
    _selected_params = get_selected_params()

    confirm_label.value = f"<b>Confirmed:</b> {state_type_dropdown.label}"


# Set up observers for widgets

state_type_dropdown.observe(update_parameter_panel, names='value')
fock_cutoff_slider.observe(update_n_max, names='value')

widgets_to_observe = [
    fock_cutoff_slider,
    fock_n_slider,
    cat_amplitude_slider,
    cat_phase_slider,
    cat_parity_dropdown,
    squeezing_dB_slider,
    squeezing_phase_slider,
    on_n_slider,
    on_parity_dropdown,
    state_type_dropdown,
    x_range_slider,
    x_points_slider,
]

for widget in widgets_to_observe:
    widget.observe(update_wigner, names='value')

confirm_button.on_click(confirm_state)


def show():
    display(interface)
    update_wigner()