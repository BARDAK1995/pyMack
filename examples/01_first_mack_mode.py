"""Find your first Mack mode -- the pyMack quickstart.

Solves the documented worked example (Mach 6 flat plate, R = 5500,
alpha_L = 0.174) both ways:

* temporal: real wavenumber in, complex phase speed out;
* spatial:  real frequency in, complex wavenumber out.

Run:  python examples/01_first_mack_mode.py
"""

import matplotlib.pyplot as plt
import numpy as np

try:
    import pymack as pm
except ModuleNotFoundError:  # running from a repo checkout without install
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import pymack as pm

# 1. A self-similar compressible flat-plate base flow (adiabatic wall).
boundary_layer = pm.flat_plate(Ma=6.0)

# 2. Temporal problem: is a wave with alpha_L = 0.174 unstable at R = 5500?
mode = pm.temporal_mode(boundary_layer, alpha=0.174, Re=5500)
print(mode)
print(f'  phase speed c_r        = {mode.phase_speed:.4f}  (Mack mode: ~ 1 - 1/Ma = {1 - 1/6:.3f})')
print(f'  temporal growth omega_i = {mode.growth_rate:+.3e}  ({"unstable" if mode.unstable else "stable"})')

# 3. The same mode, spatially: fix the frequency, ask for the wavenumber.
spatial = pm.spatial_mode(boundary_layer, omega=mode.omega.real, Re=5500)
print(spatial)
print(f'  spatial growth sigma   = {spatial.sigma:+.3e} per L*')

# 4. Eigenfunctions: the second mode's signature is a wall-confined pressure.
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for field, label in ((mode.u, '|u|'), (mode.v, '|v|'), (mode.T, '|T|'), (mode.p, '|p|')):
    amp = np.abs(field)
    ax.plot(amp / amp.max(), mode.y, label=label)
ax.set_ylim(0, 30)
ax.set_xlabel('normalized amplitude')
ax.set_ylabel('y / L*')
ax.set_title(f'Mack-mode eigenfunctions, c = {mode.eigenvalue:.4f}')
ax.legend()
fig.tight_layout()
fig.savefig('example01_eigenfunctions.png', dpi=150)
print('wrote example01_eigenfunctions.png')
