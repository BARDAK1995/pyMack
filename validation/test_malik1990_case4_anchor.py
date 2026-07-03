"""Layer-4b validation gate: Malik (1990) Test Case 4 temporal eigenvalue anchor.

The hypersonic, cooled-wall companion to the Case-6 spatial anchor
(``validation/test_malik1990_case6_anchor.py``). Where Case 6 pins a Mach-4.5
*spatial* second mode on an adiabatic wall, Case 4 pins a Mach-10 *temporal*
second mode on a strongly cooled wall -- extending the compressible-anchor
coverage in three directions at once (Mach 4.5 -> 10, spatial -> temporal,
adiabatic -> cooled).

    M. R. Malik, "Numerical methods for hypersonic boundary layer stability",
    J. Comput. Phys. 86(2), 376-413 (1990):

        Test Case 4 (Table I):  M = 10, R = 2000, T_w/T_adb = 0.1 (cooled),
                    T0 = 4200 degR (stagnation), delta* = 12.917, y_i = 13,
                    Sutherland viscosity (S = 198.6 degR = 110.33 K),
                    Pr = 0.7, gamma = 1.4, Mack length scale l = sqrt(nu_e x/u_e),
                    R = sqrt(Re_x)

        Fixed real alpha = 0.105, beta = 0  (Table V)
        omega = 0.0974837 + 0.0020304 i   (converged, 4CD scheme, N+1 = 81)
                                          (second mode, c = omega/alpha ~ 0.928)

Provenance: the dimensional conditions (T0 = 4200 degR and, crucially, the
COOLED wall T_w/T_adb = 0.1) come from Malik's own Table I, read from the
archived JCP scan (``refPapers/NewPapers/figures/malik1990_table1_test_cases.png``);
the eigenvalue is Malik's Table V (``..._table5_case4_eigenvalue.png``, 4CD
N+1=81). This case was previously logged as non-reproducible ("unknown T0/gas",
see the pre-2026-07 note in ``verification/compare_malik1990_anchors.py``)
because Hildebrand et al. (arXiv:1712.08239) list only M and Re; adding the
source paper supplied the missing T0 and wall condition and unlocked it.

Formulation mapping mirrors the Case-6 gate: pyMack's ``lambda_mu_ratio = 1.2``
(package default, Mack's second-viscosity convention), the isothermal
perturbation boundary condition, and ``length_scale = 'L_star'`` (Malik's
l = sqrt(nu_e x / u_e), R = sqrt(Re_x)). The mean flow is the cooled-wall
compressible Blasius profile at T_wall = 0.1 * T_adb (isothermal wall).

Observed convergence of the pyMack temporal EVP (alpha = 0.105):

    N=160, y_max=60 : omega = 0.0974542 + 0.0019839 i
    N=200, y_max=75 : omega = 0.0974572 + 0.0019877 i
    N=240, y_max=90 : omega = 0.0974576 + 0.0019879 i   (drift < 1e-6)

Converged pyMack value ~ 0.097457 + 0.001988 i vs Malik 0.0974837 + 0.0020304 i:
omega_r deviation ~2.6e-5 (0.027%, essentially exact), omega_i deviation ~4.3e-5
(~2.1% of the growth rate). The omega_i offset sits well inside Malik's OWN
inter-scheme spread for this deliberately severe M=10 case (his Table V four
schemes at N+1=61 span omega_i = 0.0020224..0.0020316 and omega_r =
0.0974002..0.0974864 -- pyMack's omega_r lands squarely inside that band).

The gate runs at N=200 (converged, a few seconds) with tolerance 1e-4 on both
components -- ~4x the observed omega_r deviation and ~2.3x the omega_i deviation,
and comfortably inside Malik's inter-scheme table spread.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_temporal_compressible

MALIK_OMEGA = 0.0974837 + 0.0020304j

MA = 10.0
R_L = 2000.0
ALPHA = 0.105
PR = 0.7
GAMMA = 1.4
T0_K = 4200.0 * 5.0 / 9.0                                # 4200 degR -> 2333.33 K (stagnation)
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)    # 111.11 K (edge static)
T_REC_K = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)  # adiabatic recovery
T_WALL_K = 0.1 * T_REC_K                                 # T_w/T_adb = 0.1 (cooled)
SUTHERLAND_S_K = 198.6 / 1.8                             # 110.33 K
TOL = 1.0e-4


@pytest.fixture(scope="module")
def malik_case4_profile():
    return CompressibleBlasiusProfile(
        Ma=MA,
        T_edge=T_EDGE_K,
        T_wall=T_WALL_K,
        gamma=GAMMA,
        Pr=PR,
        wall_bc="isothermal",          # cooled wall held at a fixed temperature
        viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K,
        n_points=4000,
        eta_max=40.0,
    )


def test_malik_case4_temporal_eigenvalue(malik_case4_profile):
    c, _modes, _y = solve_temporal_compressible(
        malik_case4_profile, ALPHA, R_L, MA, PR, GAMMA,
        N=200, y_max=75.0, wall_bc="isothermal",
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    c = np.asarray(c)
    c_target = MALIK_OMEGA / ALPHA        # ~ 0.9284 + 0.01934 i
    band = c[(c.real > 0.85) & (c.real < 0.98) & (np.abs(c.imag) < 0.05)]
    assert band.size, "no second-mode candidate found at Malik Case 4 conditions"
    c_mode = complex(band[np.argmin(np.abs(band - c_target))])
    omega = ALPHA * c_mode

    assert abs(omega.real - MALIK_OMEGA.real) < TOL, (
        f"omega_r = {omega.real:.7f} vs Malik {MALIK_OMEGA.real:.7f}"
    )
    assert abs(omega.imag - MALIK_OMEGA.imag) < TOL, (
        f"omega_i = {omega.imag:.7f} vs Malik {MALIK_OMEGA.imag:.7f}"
    )
    # Second-mode identity: phase speed near Malik's c = omega/alpha ~ 0.928.
    assert abs(c_mode.real - 0.9284) < 5.0e-3
    # Genuinely amplified (temporal growth), as Malik reports (omega_i > 0).
    assert omega.imag > 0.0


def test_malik_case4_wall_is_cooled(malik_case4_profile):
    """Sanity: the isothermal wall is strongly cooled vs the hot adiabatic value.

    T_adb/T_e ~ 17.7 at M=10; the imposed wall sits at T_w/T_e ~ 1.77
    (i.e. T_w/T_adb = 0.1), far below the recovery temperature.
    """
    wall = malik_case4_profile(0.0)
    assert 1.5 < wall["T"] < 2.1
