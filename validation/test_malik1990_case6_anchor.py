"""Layer-4b validation gate: Malik (1990) Test Case 6 spatial eigenvalue anchor.

The canonical tabulated benchmark for compressible spatial LST:

    M. R. Malik, "Numerical methods for hypersonic boundary layer stability",
    J. Comput. Phys. 86(2), 376-413 (1990), Table IX (multi-domain spectral
    collocation, local method, N+1=61):

        Test Case 6:  M_e = 4.5, R = 1500, omega = 0.23, beta = 0,
                      insulated (adiabatic) wall, T0 = 1100 degR = 611.11 K
                      (T_e = T0 / (1 + 0.2 M^2) = 121.01 K),
                      Sutherland viscosity (S = 198.6 degR = 110.33 K),
                      constant Prandtl number 0.7, gamma = 1.4,
                      Mack length scale l = sqrt(nu_e x / u_e), R = sqrt(Re_x)

        alpha = 0.2534048 - 0.0024921 i        (second mode, c ~ 0.9076)

Source provenance (verified June 2026): the original archived JCP scan
(Table IX read at high resolution), cross-checked digit-for-digit against
Xi, Ren & Fu (arXiv:2006.05970, Table 3) and to 4-5 digits against
Hildebrand et al. (arXiv:1712.08239, Table I). Note Tumin (2007) recomputed
this case with a different formulation and obtained alpha_i = -0.0027738
(~11% from Malik) — the literature spread on alpha_i for this case is larger
than pyMack's deviation from Malik's printed value.

Formulation mapping established empirically (see docs/VALIDATION_STRATEGY.md):
Malik's operator corresponds to pyMack's ``lambda_mu_ratio = 1.2`` (the
package default, Mack's second-viscosity convention — NOT Stokes) with the
standard isothermal perturbation boundary condition applied on the adiabatic
mean flow. Observed convergence of the pyMack companion QEP:

    N=100: alpha = 0.2533856 - 0.0025138i   (|d| ~ 2e-5)
    N=120: alpha = 0.2533998 - 0.0024898i   (|d| ~ 5e-6, 2e-6)
    N=150: alpha = 0.2534010 - 0.0024935i   (|d| ~ 4e-6, 1e-6)

The gate runs at N=120 (~1 s) with tolerance 5e-5 on both components — 10x
the observed deviation and well inside Malik's own inter-method table spread.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_spatial

MALIK_ALPHA = 0.2534048 - 0.0024921j

MA = 4.5
R_L = 1500.0
OMEGA_L = 0.23
PR = 0.7
GAMMA = 1.4
T0_K = 611.11
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)   # 121.01 K
SUTHERLAND_S_K = 198.6 / 1.8                            # 110.33 K
TOL = 5.0e-5


@pytest.fixture(scope="module")
def malik_profile():
    # T_wall is a placeholder for the adiabatic base flow (zero heat flux is
    # imposed by the BVP); the recovery estimate gives a sane initial guess.
    t_rec = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
    return CompressibleBlasiusProfile(
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


def test_malik_case6_spatial_eigenvalue(malik_profile):
    alphas, _modes, _y = solve_spatial(
        malik_profile, OMEGA_L, R_L, MA, PR, GAMMA,
        N=120, y_max=40.0, wall_bc="isothermal",
        target_alpha=OMEGA_L / 0.9076, n_modes=10,
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    c = OMEGA_L / alphas.real
    cand = alphas[(c > 0.85) & (c < 0.97) & (np.abs(alphas.imag) < 0.05)]
    assert cand.size, "no second-mode candidate found at Malik Case 6 conditions"
    alpha = complex(cand[np.argmin(np.abs(cand - MALIK_ALPHA))])

    assert abs(alpha.real - MALIK_ALPHA.real) < TOL, (
        f"alpha_r = {alpha.real:.7f} vs Malik {MALIK_ALPHA.real:.7f}"
    )
    assert abs(alpha.imag - MALIK_ALPHA.imag) < TOL, (
        f"alpha_i = {alpha.imag:.7f} vs Malik {MALIK_ALPHA.imag:.7f}"
    )
    # Second-mode identity: phase speed near Malik's c = omega/alpha_r ~ 0.9076.
    assert abs(OMEGA_L / alpha.real - 0.9076) < 5.0e-3


def test_malik_case6_adiabatic_mean_flow_is_hot(malik_profile):
    """Sanity: the insulated-wall mean flow recovers a hot wall (Tw/Te ~ 4.4)."""
    wall = malik_profile(0.0)
    assert 4.0 < wall["T"] < 4.8
