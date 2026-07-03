"""Validation gate: Malik (1990) Fig. 4 -- M=10 second-mode EIGENFUNCTION.

Opens the eigenVECTOR-validation axis (every other anchor checks eigenVALUES).
Malik Fig. 4 plots the disturbance eigenfunctions (u_hat, T_hat, p_hat vs y) of
the SAME case as Test Case 4 (M=10, R=2000, cooled wall T_w/T_adb=0.1,
alpha=0.105, beta=0). Its signature is a sharp temperature-fluctuation peak near
the boundary-layer edge (T_hat_r ~ 1 at y ~ 13), dwarfing the velocity and
pressure fluctuations -- the hallmark of the hypersonic Mack (second) mode.

    M. R. Malik, J. Comput. Phys. 86:376 (1990), Fig. 4 (source scan:
    refPapers/NewPapers/figures/malik1990_fig4_eigenfunctions_M10.png).

pyMack's second-mode eigenfunction (normalized to unit peak |T_hat|) reproduces
that structure: |T_hat| peaks at y = 13.3 (Malik ~13 ~ delta*), with the correct
shape (negative near the wall, sharp positive peak at the edge, T_hat_i dipping
just past the peak). Cross-variable amplitudes (u_hat, p_hat relative to T_hat)
depend on each code's per-variable non-dimensionalization, so the physically
robust, convention-independent checks are the mode STRUCTURE: the temperature
peak location and its dominance over the velocity fluctuation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_temporal_compressible

MA = 10.0
R_L = 2000.0
ALPHA = 0.105
PR = 0.7
GAMMA = 1.4
T0_K = 4200.0 * 5.0 / 9.0
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)
T_REC_K = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
T_WALL_K = 0.1 * T_REC_K
SUTHERLAND_S_K = 198.6 / 1.8


@pytest.fixture(scope="module")
def case4_eigenfunction():
    prof = CompressibleBlasiusProfile(
        Ma=MA, T_edge=T_EDGE_K, T_wall=T_WALL_K, gamma=GAMMA, Pr=PR,
        wall_bc="isothermal", viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K, n_points=4000, eta_max=40.0,
    )
    c, modes, y = solve_temporal_compressible(
        prof, ALPHA, R_L, MA, PR, GAMMA, N=200, y_max=75.0,
        wall_bc="isothermal", length_scale="L_star", lambda_mu_ratio=1.2,
    )
    c = np.asarray(c); modes = np.asarray(modes); y = np.asarray(y)
    n = len(y)
    band = np.where((c.real > 0.85) & (c.real < 0.98) & (np.abs(c.imag) < 0.05))[0]
    idx = band[np.argmin(np.abs(c[band] - (0.9284 + 0.0193j)))]
    phi = modes[:, idx]
    u, v, T, p = phi[0:n], phi[n:2 * n], phi[2 * n:3 * n], phi[3 * n:4 * n]
    scale = T[int(np.argmax(np.abs(T)))]          # normalize to unit peak |T|
    return dict(c=complex(c[idx]), y=y, u=u / scale, v=v / scale,
                T=T / scale, p=p / scale)


def test_temperature_peak_at_boundary_layer_edge(case4_eigenfunction):
    """|T_hat| peaks near y ~ 13 (delta*), matching Malik Fig. 4."""
    ef = case4_eigenfunction
    y = ef["y"]; T = ef["T"]
    y_peak = y[int(np.argmax(np.abs(T)))]
    assert 11.0 < y_peak < 15.0, f"T_hat peak at y={y_peak:.2f}, expected ~13"
    # normalized peak is real and ~1 (unit-peak normalization)
    ipk = int(np.argmax(np.abs(T)))
    assert abs(T[ipk].real - 1.0) < 1e-6 and abs(T[ipk].imag) < 1e-6


def test_temperature_fluctuation_dominates(case4_eigenfunction):
    """The M=10 second mode is temperature-dominated: |T_hat| >> |u_hat|."""
    ef = case4_eigenfunction
    assert np.abs(ef["T"]).max() > 3.0 * np.abs(ef["u"]).max()


def test_second_mode_identity(case4_eigenfunction):
    """The extracted mode is the Case-4 second mode (c ~ 0.928, amplified)."""
    c = case4_eigenfunction["c"]
    assert abs(c.real - 0.9284) < 5.0e-3
    assert c.imag > 0.0
