"""Layer-4b validation gate: Malik (1990) Test Case 3 temporal eigenvalue anchor.

The OBLIQUE (3-D) compressible companion to the Case-4/6 anchors. Case 3 is the
cleanest-converging case in Malik's paper -- all four of his schemes
(2FD/4CD/SDSP/MDSP) agree on the eigenvalue to five significant figures.

    M. R. Malik, "Numerical methods for hypersonic boundary layer stability",
    J. Comput. Phys. 86(2), 376-413 (1990):

        Test Case 3 (Table I):  M = 2.5, R = 3000, adiabatic wall,
                    T0 = 600 degR (stagnation), delta* = 4.2578, y_i = 6,
                    Sutherland S = 198.6 degR = 110.33 K, Pr = 0.7, gamma = 1.4,
                    Mack length scale l = sqrt(nu_e x/u_e), R = sqrt(Re_x)

        Fixed real alpha = 0.06, beta = 0.10  (Table IV, OBLIQUE wave)
        omega = 0.0367340 + 0.0005840 i   (all four schemes converge, N+1 = 61)
                                          (first/oblique mode, c = omega/alpha ~ 0.612)

Provenance: dimensional conditions (T0 = 600 degR, adiabatic wall) from Malik's
Table I (refPapers/NewPapers/figures/malik1990_table1_test_cases.png); eigenvalue
from Table IV (..._table4_case3_eigenvalue.png). Previously logged
non-reproducible ("unknown T0/gas"); adding the source paper supplied the
conditions and unlocked it.

Formulation mapping mirrors the Case-6 gate (lambda_mu_ratio=1.2, isothermal
perturbation BC on the adiabatic mean flow, length_scale='L_star'). Because
beta != 0 this uses the oblique/3-D temporal solver
``solve_temporal_compressible_3d`` (wave-aligned state, eigenvalue c = omega/alpha).

pyMack 3-D temporal EVP (alpha=0.06, beta=0.10), isothermal disturbance BC,
N=200, y_max=55:

    omega = 0.0367346 + 0.0005841 i  vs Malik 0.0367340 + 0.0005840 i
    -> omega_r deviation ~6e-7 (0.002%), omega_i deviation ~1e-7 (0.02%) --
       essentially exact, matching Malik's MDSP scheme to 5-6 sig figs.

(An earlier version mistakenly used the ADIABATIC disturbance BC, giving the
looser 0.0367093 + 0.0005699 i = 0.067%/2.4%; the isothermal T'|_wall=0 BC is
the one Malik used and it tightens the match dramatically -- a BC correction,
not a fit.) Gate runs at N=200 with abs tol 1e-4 on omega_r and relative tol 6%
on the (small) omega_i.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_temporal_compressible_3d

MALIK_OMEGA = 0.0367340 + 0.0005840j

MA = 2.5
R_L = 3000.0
ALPHA = 0.06
BETA = 0.10
PR = 0.7
GAMMA = 1.4
T0_K = 600.0 * 5.0 / 9.0                                 # 600 degR -> 333.33 K
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)    # 148.15 K
SUTHERLAND_S_K = 198.6 / 1.8                             # 110.33 K
TOL_R = 1.0e-4
TOL_I_REL = 0.06


@pytest.fixture(scope="module")
def malik_case3_profile():
    t_rec = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
    return CompressibleBlasiusProfile(
        Ma=MA, T_edge=T_EDGE_K, T_wall=t_rec, gamma=GAMMA, Pr=PR,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K, n_points=3000, eta_max=25.0,
    )


def test_malik_case3_oblique_temporal_eigenvalue(malik_case3_profile):
    c, _, _ = solve_temporal_compressible_3d(
        malik_case3_profile, ALPHA, BETA, R_L, MA, PR, GAMMA,
        N=200, y_max=55.0, wall_bc="isothermal",
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    c = np.asarray(c)
    c_target = MALIK_OMEGA / ALPHA        # ~ 0.6122 + 0.00973 i
    band = c[(c.real > 0.55) & (c.real < 0.68) & (np.abs(c.imag) < 0.05)]
    assert band.size, "no oblique-mode candidate found at Malik Case 3 conditions"
    c_mode = complex(band[np.argmin(np.abs(band - c_target))])
    omega = ALPHA * c_mode

    assert abs(omega.real - MALIK_OMEGA.real) < TOL_R, (
        f"omega_r = {omega.real:.7f} vs Malik {MALIK_OMEGA.real:.7f}"
    )
    assert abs(omega.imag - MALIK_OMEGA.imag) / abs(MALIK_OMEGA.imag) < TOL_I_REL, (
        f"omega_i = {omega.imag:.7f} vs Malik {MALIK_OMEGA.imag:.7f}"
    )
    assert omega.imag > 0.0        # amplified, as Malik reports


def test_malik_case3_adiabatic_mean_flow_is_hot(malik_case3_profile):
    """Sanity: the M=2.5 insulated wall recovers a warm wall (Tw/Te ~ 2.0)."""
    wall = malik_case3_profile(0.0)
    assert 1.8 < wall["T"] < 2.3
