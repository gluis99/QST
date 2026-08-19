import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import qutip as q

def plot_dm(state, cutoff=None,
            fig_size=(8, 6), title='Density Matrix', cmap='viridis', vlim=None,
            scaling='linear', v_log_min=1e-15, with_imag=False):
    if cutoff is None:
        cutoff = state.shape[0]
    state_cutoff = len(state)
    if cutoff>state_cutoff:
        print(f'Warning:cutoff {cutoff} larger than state dimension {state_cutoff}.')
        cutoff = state_cutoff
    rho = state.full()[:cutoff, :cutoff]
    rho_real = np.abs(rho.real)
    rho_imag = np.abs(rho.imag)

    if scaling=='log':
        rho_real = np.where(rho_real>v_log_min, np.log10(rho_real), np.log10(v_log_min))
        rho_imag = np.where(rho_imag>v_log_min, np.log10(rho_imag), np.log10(v_log_min))

    if with_imag:
        fig, axs = plt.subplots(1, 2, figsize=fig_size)
        im0 = axs[0].imshow(rho_real, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        axs[0].set_title(r'$|Re(\rho_{nm})|$')
        axs[0].set_xlabel(r'$n$')
        axs[0].set_ylabel(r'$m$')
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(rho_imag, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        axs[1].set_title(r'$|Im(\rho_{nm})|$')
        axs[1].set_xlabel(r'$n$')
        axs[1].set_ylabel(r'$m$')
        fig.colorbar(im1, ax=axs[1])
    else:
        plt.figure(figsize=fig_size)
        im = plt.imshow(rho_real, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        plt.title(title)
        plt.xlabel(r'$n$')
        plt.ylabel(r'$m$')
        plt.colorbar(im)
    plt.show()

def plot_Wigner(
        state,
        xvec=np.linspace(-5, 5, 200),
        pvec=np.linspace(-5, 5, 200),
        title=None,
        cmap='RdBu_r',
        vlim=None,
        fig_size=(8, 6),
        ax=None,
        colorbar=False
    ):

    # Calculate the Wigner function
    W = q.wigner(state, xvec, pvec)

    # Create a meshgrid for plotting
    X, P = np.meshgrid(xvec, pvec)

    # Set the colormap
    if vlim is None:
        m = np.max(np.abs(W))
        vmin, vmax = -m, m
    else:
        vmin, vmax = vlim

    norm = TwoSlopeNorm(
        vmin=vmin,
        vcenter=0.0,
        vmax=vmax
    )

    # Create figure/axes if one wasn't supplied
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    contour = ax.contourf(
        X, P, W,
        levels=100,
        cmap=cmap,
        norm=norm
    )

    ax.set_xlabel(r'Quadrature $x$')
    ax.set_ylabel(r'Quadrature $p$')

    if title:
        ax.set_title(title)

    if colorbar:
        fig.colorbar(contour, ax=ax)

    return ax

def plot_fidelities(fidelities,
                    fig_size=(8, 6), xscale='linear', yscale='linear',
                    title='Fidelity vs Iteration',
                    color='blue', marker='o', linestyle='-',
                    linewidth=2, markersize=6, with_grid=True):

    # Convert input to a NumPy array
    fidelities = np.asarray(fidelities)
    # Iteration numbers
    iterations = np.arange(1, len(fidelities) + 1)

    # Validate axis scales
    if xscale not in ('linear', 'log'):
        raise ValueError("xscale must be either 'linear' or 'log'.")

    if yscale not in ('linear', 'log'):
        raise ValueError("yscale must be either 'linear' or 'log'.")

    # Log axes require strictly positive values
    if xscale == 'log' and np.any(iterations <= 0):
        raise ValueError("Logarithmic x-axis requires positive iteration values.")

    if yscale == 'log' and np.any(fidelities <= 0):
        raise ValueError("Logarithmic y-axis requires all fidelity values to be positive.")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the original data -- do NOT manually take log10
    ax.plot(iterations, fidelities, color=color, marker=marker,
        linestyle=linestyle, linewidth=linewidth, markersize=markersize)

    # Set axis scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    ax.set_title(title)

    # Grid
    if with_grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Keep layout clean
    fig.tight_layout()

    plt.show()

    return fig, ax

#########################################################
# Utilities for homodyne sampling and plotting
def plot_homodyne(
    state,
    thetas,
    samples,
    x_vec,
    bins=200,
    theta_index=0,
    figsize=(12, 9),
    theta_width=0.025,
    random_seed=42,
):

    # ==============================================================
    # Convert inputs
    # ==============================================================

    thetas = np.asarray(thetas)
    samples = np.asarray(samples)
    x_vec = np.asarray(x_vec)

    # ==============================================================
    # Validate inputs
    # ==============================================================

    if samples.ndim != 2:
        raise ValueError(
            "samples must be a 2D array with shape "
            "(n_angles, n_samples_per_angle)."
        )

    if len(thetas) != samples.shape[0]:
        raise ValueError(
            "The number of thetas must match the first dimension "
            "of samples."
        )

    if not 0 <= theta_index < len(thetas):
        raise ValueError(
            f"theta_index must be between 0 and "
            f"{len(thetas) - 1}, got {theta_index}."
        )

    if bins < 1:
        raise ValueError(
            "bins must be >= 1."
        )

    # ==============================================================
    # Selected marginal
    # ==============================================================

    theta_selected = thetas[theta_index]
    x_selected = samples[theta_index]

    # ==============================================================
    # Figure
    # ==============================================================

    fig = plt.figure(
        figsize=figsize,
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3, 2],
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ==============================================================
    # Top plot: all homodyne measurements
    # ==============================================================

    theta_scatter = np.repeat(
        thetas,
        samples.shape[1],
    )

    x_scatter = samples.flatten()

    # Fixed jitter for reproducible visualization
    rng = np.random.default_rng(random_seed)

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

    # ==============================================================
    # Highlight selected theta
    # ==============================================================

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

    # ==============================================================
    # Bottom plot: marginal distribution
    # ==============================================================

    ax2.hist(
        x_selected,
        bins=bins,
        density=True,
        alpha=0.75,
        label='Samples',
    )

    # ==============================================================
    # KDE
    # ==============================================================

    x_kde = np.linspace(
        x_vec[0],
        x_vec[-1],
        len(x_vec),
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

    kde = np.zeros_like(x_kde)

    # Chunked calculation to avoid creating a huge
    # x_kde × samples array.
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

    # ==============================================================
    # Labels
    # ==============================================================

    ax2.set_xlabel(
        r'Quadrature $x_\theta$',
        fontsize=13,
    )

    ax2.set_ylabel(
        r'$P(x_\theta)$',
        fontsize=13,
    )

    ax2.set_title(
        rf'Marginal Distribution at '
        rf'$\theta={theta_selected:.3f}$ rad',
        fontsize=14,
    )

    ax2.set_xlim(
        x_vec[0],
        x_vec[-1],
    )

    ax2.grid(
        alpha=0.2,
    )

    ax2.legend()

    return fig, (ax1, ax2)
