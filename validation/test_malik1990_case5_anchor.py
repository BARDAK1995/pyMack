"""Layer-4b validation gate: Malik (1990) Test Case 5 temporal eigenvalue anchor.

Malik's deliberately SEVERE M=10 test: a near-neutral hypersonic second mode
whose tiny growth rate (omega_i ~ 1.5e-4) is scheme-sensitive -- Malik singles
it out to stress-test a solver's resolution. pyMack nonetheless nails it.

    M. R. Malik, "Numerical methods for hypersonic boundary layer stability",
    J. Comput. Phys. 86(2), 376-413 (1990):

        Test Case 5 (Table I):  M = 10, R = 1000, adiabatic wall,
                    T0 = 4200 degR (stagnation), delta* = 31.679, y_i = 32,
                    Sutherland S = 198.6 degR = 110.33 K, Pr = 0.7, gamma = 1.4,
                    Mack length scale l = sqrt(nu_e x/u_e), R = sqrt(Re_x)

        Fixed real alpha = 0.12, beta = 0  (Table VI)
        omega = 0.1158647 + 0.0001529 i   (converged, 4CD, N+1 = 81)
        (Malik's independent IVM/RK4 cross-check: 0.1158627 + 0.0001557 i)
        (second mode, c = omega/alpha ~ 0.966)

Provenance: dimensional conditions (T0 = 4200 degR, adiabatic) from Malik's
Table I (refPapers/NewPapers/figures/malik1990_table1_test_cases.png); eigenvalue
from Table VI (..._table6_case5_eigenvalue.png). The adiabatic M=10 mean flow is
very thick (delta* ~ 31.7), so a large domain / high resolution are needed.

pyMack temporal EVP (alpha = 0.12), isothermal disturbance BC (T'|_wall=0 --
Malik's convention; the mean wall is adiabatic but the perturbation T' vanishes
at the wall), N=280, y_max=140:

    omega = 0.1158623 + 0.0001557 i  vs Malik 0.1158647 + 0.0001529 i
    -> omega_r deviation ~2.4e-6 (0.002%) and omega_i within ~3e-6 (1.83%),
       landing right on Malik's own IVM/RK4 cross-check (0.0001557 i).

(An earlier version mistakenly used the ADIABATIC disturbance BC, giving
0.1158463 + 0.0001499 i = 0.016%/2.0%; switching to Malik's isothermal
disturbance BC is a correction, not a fit, and tightens omega_r ten-fold. This
M=10 mode is edge-peaked, so the wall-T BC has modest leverage -- unlike the
near-wall Ma & Zhong second mode where the same fix moved 3.4% -> 1.0%.) Gate
runs at N=280 with abs tol 1e-4 on omega_r and relative tol 8% on the tiny
omega_i.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_temporal_compressible

MALIK_OMEGA = 0.1158647 + 0.0001529j

MA = 10.0
R_L = 1000.0
ALPHA = 0.12
PR = 0.7
GAMMA = 1.4
T0_K = 4200.0 * 5.0 / 9.0                                # 2333.33 K (stagnation)
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)    # 111.11 K
SUTHERLAND_S_K = 198.6 / 1.8                             # 110.33 K
TOL_R = 1.0e-4
TOL_I_REL = 0.08


@pytest.fixture(scope="module")
def malik_case5_profile():
    t_rec = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
    return CompressibleBlasiusProfile(
        Ma=MA, T_edge=T_EDGE_K, T_wall=t_rec, gamma=GAMMA, Pr=PR,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K, n_points=6000, eta_max=70.0,
    )


def test_malik_case5_temporal_eigenvalue(malik_case5_profile):
    c, _, _ = solve_temporal_compressible(
        malik_case5_profile, ALPHA, R_L, MA, PR, GAMMA,
        N=280, y_max=140.0, wall_bc="isothermal",
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    c = np.asarray(c)
    c_target = MALIK_OMEGA / ALPHA        # ~ 0.9655 + 0.001274 i
    band = c[(c.real > 0.90) & (c.real < 1.0) & (np.abs(c.imag) < 0.03)]
    assert band.size, "no second-mode candidate found at Malik Case 5 conditions"
    c_mode = complex(band[np.argmin(np.abs(band - c_target))])
    omega = ALPHA * c_mode

    assert abs(omega.real - MALIK_OMEGA.real) < TOL_R, (
        f"omega_r = {omega.real:.7f} vs Malik {MALIK_OMEGA.real:.7f}"
    )
    assert abs(omega.imag - MALIK_OMEGA.imag) / abs(MALIK_OMEGA.imag) < TOL_I_REL, (
        f"omega_i = {omega.imag:.7f} vs Malik {MALIK_OMEGA.imag:.7f}"
    )
    assert abs(c_mode.real - 0.9655) < 5.0e-3     # second-mode phase speed
    assert omega.imag > 0.0                       # amplified (near-neutral, but > 0)


def test_malik_case5_adiabatic_mean_flow_is_hot(malik_case5_profile):
    """Sanity: the adiabatic M=10 wall is very hot (Tw/Te ~ 17.7)."""
    wall = malik_case5_profile(0.0)
    assert 16.5 < wall["T"] < 18.5
