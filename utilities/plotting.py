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
                    fig_size=(8, 6), scaling='linlog', title='Fidelity vs Iteration',
                    color='blue', marker='o', linestyle='-', linewidth=2, markersize=6,
                    with_gritd=True):
    plt.figure(figsize=fig_size)

    if scaling == 'loglog':
        plt.loglog(fidelities, color=color, marker=marker, 
                   linestyle=linestyle, linewidth=linewidth, markersize=markersize)
    elif scaling == 'linlog' or scaling == 'linear':
        if scaling == 'linlog':
            fidelities = np.log10(fidelities)
        plt.plot(fidelities, color=color, marker=marker, 
                 linestyle=linestyle, linewidth=linewidth, markersize=markersize)
    else:
        raise ValueError("Invalid scaling option. Choose 'loglog', 'linlog', or 'linear'.")
    plt.xlabel('Iteration')
    plt.ylabel('Fidelity')
    plt.title(title)
    if with_gritd:
        plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show()
