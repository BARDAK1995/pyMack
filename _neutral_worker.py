"""Worker: compute c_i map for one Mach number, save as .npy.

Usage: python -u _neutral_worker.py <Mach>

Computes c_i(Re_L, alpha_L) on a grid using get_max_ci
(convergence-filtered two-mode detection). Results saved
for later contour plotting.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from chapters.ozgen_kircali_2008.ozgen import (
    make_profile, get_y_delta_star, get_max_ci,
    get_ci_incompressible, BlasiusProfile
)

Ma = float(sys.argv[1])
yds = get_y_delta_star(Ma)
print(f'M={Ma}, yds={yds:.4f}', flush=True)

# Grid setup — higher resolution for clean contours
alpha_L_max = min(0.40, 4.0 / max(yds, 1.0))

if Ma < 0.05:
    n_Re, n_alpha = 50, 50
    Re_L_arr = np.linspace(200, 10000, n_Re)
    alpha_L_arr = np.linspace(0.01, 0.30, n_alpha)
    N_eig = 100
else:
    n_Re, n_alpha = 45, 45
    Re_L_arr = np.linspace(50, 5000, n_Re)
    alpha_L_arr = np.linspace(0.003, alpha_L_max, n_alpha)
    N_eig = 80

print(f'Grid: {n_Re}x{n_alpha}, Re=[{Re_L_arr[0]:.0f},{Re_L_arr[-1]:.0f}], '
      f'aL=[{alpha_L_arr[0]:.4f},{alpha_L_arr[-1]:.4f}], N={N_eig}', flush=True)

# Profile once
if Ma < 0.05:
    bf = BlasiusProfile()
else:
    bf = make_profile(Ma, Re=2000)

ci_map = np.full((n_Re, n_alpha), np.nan)

for i, Re_L in enumerate(Re_L_arr):
    Re_d = Re_L * yds
    for j, alpha_L in enumerate(alpha_L_arr):
        alpha_d = alpha_L * yds
        if Ma < 0.05:
            ci_map[i, j] = get_ci_incompressible(bf, alpha_d, Re_d, N=N_eig, y_max=40.0)
        else:
            max_ci, _, _ = get_max_ci(bf, alpha_d, Re_d, Ma, N=N_eig)
            ci_map[i, j] = max_ci
    if (i + 1) % 5 == 0:
        print(f'  {i+1}/{n_Re}', flush=True)

outpath = os.path.join('chapters', 'ozgen_kircali_2008', 'neutral_data', f'map_M{Ma:.0f}.npz')
np.savez(outpath, Re_L=Re_L_arr, alpha_L=alpha_L_arr, ci_map=ci_map)
print(f'M={Ma}: Saved {outpath}', flush=True)
