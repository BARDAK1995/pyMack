"""Recompute the full spatial spectrum for the B&M (1992) M4.5 branch case.

Mirrors validation/test_malik1990_case6_anchor.py setup, but at the
Balakumar & Malik (1992) / Xi-Ren-Fu Case 1 condition:
    M = 4.5, T0 = 311 K  -> T_edge = 311 / (1 + 0.2*4.5^2) = 61.58 K,
    Pr = 0.72, gamma = 1.4, adiabatic, Sutherland S = 110.33 K,
    Re = 1000, omega = 0.2, beta = 0, N = 120, lambda_mu_ratio = 1.2,
    length_scale = 'L_star'.

Saves the full eigenvalue array to spectrum.npz for plotting.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYMACK_NO_BANNER", "1")

import numpy as np

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_spatial_full_spectrum

HERE = os.path.dirname(os.path.abspath(__file__))

MA = 4.5
RE_L = 1000.0
OMEGA_L = 0.2
BETA = 0.0
PR = 0.72
GAMMA = 1.4
T0_K = 311.0
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)   # 61.58 K
SUTHERLAND_S_K = 198.6 / 1.8                            # 110.33 K
N = 120

print(f"T_edge = {T_EDGE_K:.4f} K")

t_rec = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
profile = CompressibleBlasiusProfile(
    Ma=MA,
    T_edge=T_EDGE_K,
    T_wall=t_rec,
    gamma=GAMMA,
    Pr=PR,
    wall_bc="adiabatic",
    viscosity_model="sutherland",
    sutherland_S=SUTHERLAND_S_K,
    n_points=3000,
    eta_max=20.0,
)

print("solving full spectrum ...")
alphas, modes, y = solve_spatial_full_spectrum(
    profile, OMEGA_L, RE_L, MA, PR, GAMMA,
    N=N, y_max=40.0, wall_bc="isothermal",
    length_scale="L_star", lambda_mu_ratio=1.2,
)
alphas = np.asarray(alphas)
print(f"got {alphas.size} eigenvalues")

# phase speed c = omega / alpha_r
c = OMEGA_L / alphas.real

np.savez(
    os.path.join(HERE, "spectrum.npz"),
    alphas=alphas,
    c=c,
    omega=OMEGA_L,
    Re=RE_L,
    Ma=MA,
    N=N,
)

# Locate the discrete second mode near the published value
PUB = 0.220 - 0.003091j
band = (c > 0.85) & (c < 0.97) & (np.abs(alphas.imag) < 0.05)
cand = alphas[band]
if cand.size:
    disc = complex(cand[np.argmin(np.abs(cand - PUB))])
    print(f"discrete second mode alpha = {disc.real:.7f} {disc.imag:+.7f}i  c={OMEGA_L/disc.real:.5f}")
else:
    print("no discrete-mode candidate found")
print("saved spectrum.npz")
