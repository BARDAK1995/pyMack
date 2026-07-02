"""
Publication-quality visualization for LST results.

All plots use consistent styling with minimum font sizes:
labels >= 14pt, ticks >= 12pt, titles >= 16pt.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#: pyMack house style. Applied on demand (never at import) -- call
#: :func:`apply_plot_style` yourself, or let the figure helpers do it.
PLOT_STYLE = {
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

_style_applied = False


def apply_plot_style(force=False):
    """Apply pyMack's publication style to matplotlib's global rcParams.

    Called automatically by the figure helpers in this module; call it
    directly to style your own figures the same way. Importing
    :mod:`pymack.plotting` never touches global state by itself.
    """
    global _style_applied
    if _style_applied and not force:
        return
    rcParams.update(PLOT_STYLE)
    _style_applied = True

# Color palette
COLORS = {
    'primary': '#2563EB',
    'secondary': '#DC2626',
    'tertiary': '#059669',
    'quaternary': '#D97706',
    'neutral': '#6B7280',
    'unstable': '#DC2626',
    'stable': '#2563EB',
    'neutral_curve': '#1F2937',
}


def plot_eigenspectrum(eigenvalues, title='Eigenspectrum', save_path=None,
                       highlight_idx=None):
    """Plot eigenvalues in the complex plane.

    Parameters
    ----------
    eigenvalues : array
        Complex eigenvalues (c for temporal, α for spatial).
    title : str
        Plot title.
    save_path : str, optional
        File path to save figure.
    highlight_idx : int or array, optional
        Indices of eigenvalues to highlight.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(eigenvalues.real, eigenvalues.imag,
               s=25, c=COLORS['primary'], alpha=0.6, edgecolors='none',
               label='Eigenvalues')

    if highlight_idx is not None:
        hi = np.atleast_1d(highlight_idx)
        ax.scatter(eigenvalues[hi].real, eigenvalues[hi].imag,
                   s=80, c=COLORS['secondary'], marker='*', zorder=5,
                   label='Selected mode')

    ax.axhline(0, color='k', lw=0.8, ls='-')
    ax.set_xlabel(r'$c_r$' if 'temporal' in title.lower() else r'$\alpha_r$')
    ax.set_ylabel(r'$c_i$' if 'temporal' in title.lower() else r'$\alpha_i$')
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved: {save_path}')
    plt.close(fig)
    return fig


def plot_growth_rate(omega, sigma, title='Spatial Growth Rate',
                     save_path=None, labels=None):
    """Plot growth rate σ vs frequency ω.

    Parameters
    ----------
    omega : array or list of arrays
        Frequency values.
    sigma : array or list of arrays
        Growth rates (-α_i).
    labels : list of str, optional
        Legend labels for multiple curves.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    if not isinstance(omega, list):
        omega = [omega]
        sigma = [sigma]

    colors = [COLORS['primary'], COLORS['secondary'],
              COLORS['tertiary'], COLORS['quaternary']]

    for i, (om, sig) in enumerate(zip(omega, sigma)):
        label = labels[i] if labels else None
        ax.plot(om, sig, color=colors[i % len(colors)], label=label)

    ax.axhline(0, color='k', lw=0.8, ls='-')
    ax.fill_between(omega[0], 0, np.maximum(sigma[0], 0),
                     alpha=0.08, color=COLORS['unstable'])

    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\sigma = -\alpha_i$')
    ax.set_title(title)
    if labels:
        ax.legend(loc='best')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved: {save_path}')
    plt.close(fig)
    return fig


def plot_neutral_curve(Re_arr, omega_arr, sigma_map, title='Neutral Stability Curve',
                       save_path=None, Ma=None):
    """Plot neutral curve as contour in (Re, ω) space.

    Parameters
    ----------
    Re_arr, omega_arr : arrays
        Parameter arrays.
    sigma_map : 2D array
        Growth rate map (Re × ω).
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    Re_g, Om_g = np.meshgrid(Re_arr, omega_arr, indexing='ij')

    # Filled contour for growth rate
    levels = np.linspace(np.nanmin(sigma_map), np.nanmax(sigma_map), 25)
    cf = ax.contourf(Re_g, Om_g, sigma_map, levels=levels,
                     cmap='RdBu_r', extend='both')
    cb = fig.colorbar(cf, ax=ax, label=r'$\sigma = -\alpha_i$', shrink=0.85)

    # Neutral curve (σ = 0)
    try:
        cs = ax.contour(Re_g, Om_g, sigma_map, levels=[0],
                        colors=COLORS['neutral_curve'], linewidths=2.5)
        ax.clabel(cs, fmt=r'$\sigma=0$', fontsize=12)
    except ValueError:
        pass

    subtitle = f'Ma = {Ma}' if Ma else ''
    ax.set_xlabel(r'$Re_{\delta^*}$')
    ax.set_ylabel(r'$\omega$')
    ax.set_title(f'{title}\n{subtitle}' if subtitle else title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved: {save_path}')
    plt.close(fig)
    return fig


def plot_nfactor(Re_arr, N_vals, title='N-Factor', save_path=None,
                 labels=None, frequencies=None):
    """Plot N-factor curves.

    Parameters
    ----------
    Re_arr : array or list
    N_vals : array or list
    labels : list of str
    frequencies : list of float
        Corresponding frequencies for labeling.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    if not isinstance(Re_arr, list):
        Re_arr = [Re_arr]
        N_vals = [N_vals]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Re_arr)))

    for i, (re, nv) in enumerate(zip(Re_arr, N_vals)):
        label = labels[i] if labels else (f'ω={frequencies[i]:.3f}' if frequencies else None)
        ax.plot(re, nv, color=colors[i], label=label)

    # N=9 transition line
    ax.axhline(9, color=COLORS['secondary'], ls='--', lw=1.5,
               alpha=0.7, label='N = 9 (transition)')

    ax.set_xlabel(r'$Re_{\delta^*}$')
    ax.set_ylabel(r'$N$')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved: {save_path}')
    plt.close(fig)
    return fig


def plot_eigenfunction(y, phi, labels=None, title='Eigenfunction',
                       save_path=None):
    """Plot eigenfunction profiles.

    Parameters
    ----------
    y : array
        Wall-normal coordinate.
    phi : dict or list of arrays
        Eigenfunction components. If dict, keys are labels.
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    if isinstance(phi, dict):
        items = list(phi.items())
    else:
        items = [(f'Mode {i}', p) for i, p in enumerate(phi)]

    colors = [COLORS['primary'], COLORS['secondary'],
              COLORS['tertiary'], COLORS['quaternary']]

    for i, (name, p) in enumerate(items):
        c = colors[i % len(colors)]
        axes[0].plot(np.abs(p), y, color=c, label=f'|{name}|')
        axes[1].plot(np.angle(p, deg=True), y, color=c, label=f'∠{name}')

    axes[0].set_xlabel('Amplitude')
    axes[0].set_ylabel(r'$y / \delta^*$')
    axes[0].set_title('Magnitude')
    axes[0].legend(loc='best', fontsize=11)

    axes[1].set_xlabel('Phase [deg]')
    axes[1].set_title('Phase')
    axes[1].legend(loc='best', fontsize=11)

    fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved: {save_path}')
    plt.close(fig)
    return fig


def plot_baseflow(y, profiles, title='Mean Flow Profiles', save_path=None):
    """Plot base flow profiles (U, T, ρ).

    Parameters
    ----------
    y : array
        Wall-normal coordinate.
    profiles : dict
        Keys: 'U', 'T', 'rho' etc.
    """
    apply_plot_style()
    n_plots = sum(1 for k in ['U', 'T', 'rho', 'mu'] if k in profiles)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 6), sharey=True)
    if n_plots == 1:
        axes = [axes]

    plot_info = [
        ('U', r'$\bar{U}/U_e$', COLORS['primary']),
        ('T', r'$\bar{T}/T_e$', COLORS['secondary']),
        ('rho', r'$\bar{\rho}/\rho_e$', COLORS['tertiary']),
        ('mu', r'$\bar{\mu}/\mu_e$', COLORS['quaternary']),
    ]

    idx = 0
    for key, xlabel, color in plot_info:
        if key in profiles:
            axes[idx].plot(profiles[key], y, color=color, lw=2)
            axes[idx].set_xlabel(xlabel)
            if idx == 0:
                axes[idx].set_ylabel(r'$y / \delta^*$')
            idx += 1

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved: {save_path}')
    plt.close(fig)
    return fig
