"""Fixed-frequency spatial growth and the N-factor.

A disturbance of fixed physical frequency rides downstream through the
growing boundary layer: stable at first, amplified through the second-mode
band, stable again beyond it. Integrating the positive part of its spatial
growth gives the N-factor of the e^N transition-prediction method.

Pattern worth copying: solve ONE station unguided (pyMack then applies both
discrete-mode acceptance tests automatically), and continue outward from it
with warm-started guesses -- fast, and it stays on the physical branch.

Run:  python examples/02_growth_curve_and_n_factor.py     (~1 minute)
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

boundary_layer = pm.flat_plate(Ma=6.0)

F = 1.0e-4                                    # reduced frequency F = omega_L / R
stations = np.linspace(1150.0, 2150.0, 21)    # R = sqrt(Re_x)
i_seed = 8                                    # a mid-band station, R ~ 1550

# 1. Seed solve: no guess, so both discrete-mode tests run automatically.
seed = pm.spatial_mode(boundary_layer, omega=F * stations[i_seed],
                       Re=stations[i_seed], N=100)
print('seed station:', seed)

# 2. Continue outward from the seed, warm-starting each neighbour.
modes = {i_seed: seed}
for direction in (+1, -1):
    guess = seed.alpha
    for i in range(i_seed + direction, -1 if direction < 0 else len(stations),
                   direction):
        R = stations[i]
        modes[i] = pm.spatial_mode(boundary_layer, omega=F * R, Re=R, N=100,
                                   alpha_guess=guess)
        guess = modes[i].alpha

sigma = np.array([modes[i].sigma for i in range(len(stations))])
for R, s in zip(stations, sigma):
    print(f'R = {R:6.0f}   sigma_L = {s:+.3e}')

# 3. N-factor: integrate positive growth along R.
result = pm.integrate_n_factor({'sigma': sigma, 'Re': stations})
print(f"\npeak N over the F = {F:.0e} band: {result['N'].max():.2f}")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))
ax1.plot(stations, sigma * 1e3, 'o-')
ax1.axhline(0.0, color='k', lw=0.8)
ax1.set_ylabel(r'$\sigma_L \times 10^3$')
ax1.set_title(f'Mach-6 second mode at fixed frequency F = {F:.0e}')
ax2.plot(result['path'], result['N'], 's-')
ax2.set_xlabel(r'$R = \sqrt{Re_x}$')
ax2.set_ylabel('N-factor')
fig.tight_layout()
fig.savefig('example02_n_factor.png', dpi=150)
print('wrote example02_n_factor.png')
