"""A parameter sweep with `pymack.sweep` -- one call, one stability map.

The per-point API (`pm.temporal_mode`) solves ONE (alpha, Re) point. This
example solves a whole GRID of points in one call to `temporal_sweep`, and
plots the resulting temporal growth-rate map with its neutral curve
(omega_i = 0).

This runs entirely on the public CPU backend. See docs/SWEEP_API.md for the
full API reference, the `seed_map`
provenance contract, and the determinism/meta contract this sweep publishes.

Run:  python examples/04_parameter_sweep.py --backend cpu     (~15 s)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

try:
    import pymack as pm
except ModuleNotFoundError:  # running from a repo checkout without install
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import pymack as pm

from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep

# Matplotlib font-size floor used throughout pyMack figures.
LABEL_FS, TICK_FS, TITLE_FS = 14, 12, 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--backend', default='cpu', choices=('cpu', 'auto'),
        help="sweep backend ('auto' resolves to the public CPU path)")
    args = parser.parse_args()

    # 1. Same Mach-6 self-similar flat-plate base flow as examples 01-03.
    boundary_layer = pm.flat_plate(Ma=6.0)

    # 2. A small (alpha_L, R) grid around the documented worked example
    #    (alpha_L = 0.174, R = 5500) -- one temporal_sweep call replaces 60
    #    individual pm.temporal_mode calls. `y_max` is set explicitly to the
    #    same boundary-layer-scaled domain height `pm.temporal_mode` defaults
    #    to (`10 * delta*/L*`) -- temporal_sweep's own default (a fixed
    #    height) is tuned for the deployed driver it reproduces bit-for-bit
    #    (docs/SWEEP_API.md), not for this profile.
    #
    #    The `CBand` window (0.885, 0.96) isolates the single continuous
    #    second-mode branch from a second, less-unstable branch that also
    #    lives in this alpha range -- `pymack.sweep`'s per-family selection
    #    is the batched analog of a phase-speed *window* filter (see
    #    docs/SWEEP_API.md, "Mode families"), so an overly wide window can
    #    admit more than one physical branch and the argmax selection can
    #    jump between them from one grid point to the next. Two disjoint
    #    `CBand` families (as in `validation/test_sweep_cpu_backend.py`) is
    #    the general way to track several branches at once.
    y_max = 10.0 * delta_star_over_lstar(boundary_layer)
    alphas = np.linspace(0.02, 0.20, 10)
    res_values = np.geomspace(3000.0, 8000.0, 6)
    mack_band = CBand(0.885, 0.96, label='Mack')

    result = temporal_sweep(
        boundary_layer, alphas, res_values,
        Ma=6.0, N=100, y_max=y_max, operator='ozgen_2d',
        families=(mack_band,),
        backend=args.backend, cpu_workers=4,
    )
    family = result.families[0]

    # 3. The determinism/provenance contract: every artifact is
    #    self-describing (docs/SWEEP_API.md, "Determinism and the meta
    #    contract").
    meta = result.meta
    print(f"backend={meta['backend']}  precision={meta['precision']}  "
          f"tile_size={meta['tile_size']}  cpu_workers={meta['cpu_workers']}")
    print(f"pymack {meta['versions']['pymack']}  "
          f"numpy {meta['versions']['numpy']}  "
          f"scipy {meta['versions']['scipy']}")
    print(f"wall_time_s = {meta['wall_time_s']:.3f}   "
          f"converged = {int(family.converged.sum())}/{family.converged.size}"
          f"   n_failed_points = {meta['n_failed_points']}")
    print(f"seed_map legend: {meta['seed_codes']}")

    result.to_npz('example04_sweep.npz')
    print('wrote example04_sweep.npz')

    # 4. Stability map: filled contours of temporal growth rate omega_i,
    #    with the neutral curve (omega_i = 0) overlaid.
    Re_grid, alpha_grid = np.meshgrid(result.Res, result.alphas)
    omega_i = family.omega_i

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cf = ax.contourf(Re_grid, alpha_grid, omega_i, levels=15, cmap='RdBu_r')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r'$\omega_i$  (temporal growth rate)', fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    if np.nanmin(omega_i) < 0.0 < np.nanmax(omega_i):
        ax.contour(Re_grid, alpha_grid, omega_i, levels=[0.0], colors='k',
                   linewidths=1.5)
    ax.set_xlabel(r'$R = \sqrt{Re_x}$', fontsize=LABEL_FS)
    ax.set_ylabel(r'$\alpha_L$', fontsize=LABEL_FS)
    ax.set_title('Mach-6 second mode: temporal stability map (CPU backend)',
                fontsize=TITLE_FS)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    fig.tight_layout()
    fig.savefig('example04_stability_map.png', dpi=150)
    print('wrote example04_stability_map.png')


if __name__ == '__main__':
    main()
