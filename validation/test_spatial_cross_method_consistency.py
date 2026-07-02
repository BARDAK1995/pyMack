"""Layer-4a validation gates: internal cross-method spatial consistency.

pyMack contains two independently-implemented spatial stability operators plus
several solution routes through the main operator:

- ``pymack.dense`` — dense companion QEP, own Lees–Dorodnitsyn base
  flow, **Stokes** viscous coefficients (lambda/mu = 0) hardcoded;
- ``pymack.solver`` family — companion QEP (shift-invert and full-spectrum),
  Newton on the temporal EVP, the temporal+Gaster+Newton pipeline, and Muller
  refinement, all on ``CompressibleBlasiusProfile``.

Agreement between the two *independent operators* validates discretization,
base flow, and physics assembly; agreement *within* the main-solver family
validates the eigensolvers and root-refinement logic. Tolerances are set at
~3–5x the deltas measured in the June 2026 cross-method study (see
docs/VALIDATION_STRATEGY.md, Layer 4a).

Known systematic, deliberately pinned here rather than hidden:
``pymack.solver`` defaults to ``lambda_mu_ratio=1.2`` (Mack's second-viscosity
choice) while ``pymack_dense`` hardcodes Stokes (0.0); at the canonical point
this shifts sigma_L by ~9%. Cross-operator checks therefore pass
``lambda_mu_ratio=0.0``; ``solve_spatial_muller`` (no such parameter, fixed at
the 1.2 default) is checked against the QEP evaluated at 1.2.

All points: Mach 6 nitrogen, T_e = 300 K, Tw/Te = 5.88, isothermal wall,
Sutherland S = 111 K, Pr = 0.72, gamma = 1.4, Mack L* scale (omega_L = F R_L).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import CompressibleBlasiusProfile
from pymack.dense import (
    DenseBaseFlowConfig,
    DenseGasModel,
    DenseLSTConfig,
    prepare_dense_case,
    solve_mack_branch,
)
from pymack.solver import (
    solve_spatial,
    solve_spatial_from_temporal,
    solve_spatial_muller,
    solve_spatial_newton,
)

MA, T_EDGE, TW_OVER_TE = 6.0, 300.0, 5.88
PR, GAMMA, SUTHERLAND_S = 0.72, 1.4, 111.0
PHASE_MIN, PHASE_MAX = 0.86, 0.97

# (R_L, F) comparison points. The third sits below the lower neutral branch
# (damped) to exercise sign behaviour, not just peak growth. R_L kept below
# ~2000: the dense backend's default ny=31 grid is documented as
# under-resolved beyond that (observed d_sigma ~ 2.5e-3 at R_L=2500).
POINT_AMPLIFIED = (800.0, 2.0e-4)
POINT_SECOND_LOBE = (1500.0, 1.0e-4)
POINT_DAMPED = (500.0, 2.0e-4)

# Evidence-based tolerances (absolute, L* units): ~3-5x observed deltas.
TOL_CROSS_OPERATOR_AR = 1.5e-3   # observed |d alpha_r| = 2.2e-4 (pt 1)
TOL_CROSS_OPERATOR_SI = 7.5e-4   # observed |d sigma|  = 1.0e-4 (pt 1), 1.6e-5 (pt 2)
TOL_NEAR_NEUTRAL_SI = 1.5e-3     # observed 4.3e-4 at the damped point
TOL_WITHIN_FAMILY = 5.0e-5       # observed <= 1e-6 (Newton/Gaster/Muller vs QEP)


@pytest.fixture(scope="module")
def profile():
    return CompressibleBlasiusProfile(
        Ma=MA,
        T_wall=TW_OVER_TE * T_EDGE,
        T_edge=T_EDGE,
        gamma=GAMMA,
        Pr=PR,
        wall_bc="isothermal",
        viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S,
        n_points=3000,
        eta_max=16.0,
    )


@pytest.fixture(scope="module")
def dense_case():
    gas = DenseGasModel(
        gamma=GAMMA,
        prandtl=PR,
        viscosity_law="sutherland",
        sutherland_S_K=SUTHERLAND_S,
        T_edge_K=T_EDGE,
    )
    base_cfg = DenseBaseFlowConfig(
        mach_edge=MA, Tw_Te=TW_OVER_TE, eta_max=16.0, eta_nodes=80,
        bvp_tol=1.0e-4, adiabatic=False,
    )
    lst_cfg = DenseLSTConfig(ny=31, y_max=30.0)
    base, y, D, base_grid = prepare_dense_case(gas, base_cfg, lst_cfg)
    return gas, lst_cfg, y, D, base_grid


def _dense_alpha(dense_case, R_L: float, F: float) -> complex:
    gas, lst_cfg, y, D, base_grid = dense_case
    rows = solve_mack_branch(F, np.array([R_L]), y, D, base_grid, gas, lst_cfg)
    row = rows[0]
    alpha = complex(row["alpha_real"], row["alpha_imag"])
    assert np.isfinite(alpha.real) and np.isfinite(alpha.imag), (
        f"dense backend failed to track the branch at R={R_L}, F={F}: {row}"
    )
    return alpha


def _qep_alpha(profile, R_L: float, F: float, *, lambda_mu_ratio: float = 0.0,
               N: int = 80) -> complex:
    omega = F * R_L
    alphas, _modes, _y = solve_spatial(
        profile, omega, R_L, MA, PR, GAMMA,
        N=N, y_max=30.0, wall_bc="isothermal",
        target_alpha=omega / 0.92, n_modes=10,
        length_scale="L_star", lambda_mu_ratio=lambda_mu_ratio,
    )
    c = omega / alphas.real
    cand = alphas[(c > PHASE_MIN) & (c < PHASE_MAX) & (np.abs(alphas.imag) < 0.4)]
    assert cand.size, f"no QEP candidate in the second-mode phase window at R={R_L}, F={F}"
    return complex(cand[np.argmin(cand.imag)])


def test_independent_operators_agree_amplified_point(profile, dense_case):
    """Dense (Stokes) vs main-solver QEP (Stokes) at the canonical M6 point."""
    R_L, F = POINT_AMPLIFIED
    a_dense = _dense_alpha(dense_case, R_L, F)
    a_qep = _qep_alpha(profile, R_L, F)
    assert abs(a_dense.real - a_qep.real) < TOL_CROSS_OPERATOR_AR
    assert abs(a_dense.imag - a_qep.imag) < TOL_CROSS_OPERATOR_SI
    # Both must agree the point is amplified (sigma = -Im(alpha) > 0).
    assert a_dense.imag < 0.0 and a_qep.imag < 0.0


def test_independent_operators_agree_second_lobe_point(profile, dense_case):
    R_L, F = POINT_SECOND_LOBE
    a_dense = _dense_alpha(dense_case, R_L, F)
    a_qep = _qep_alpha(profile, R_L, F)
    assert abs(a_dense.real - a_qep.real) < TOL_CROSS_OPERATOR_AR
    assert abs(a_dense.imag - a_qep.imag) < TOL_CROSS_OPERATOR_SI
    assert a_dense.imag < 0.0 and a_qep.imag < 0.0


def test_independent_operators_agree_damped_point(profile, dense_case):
    """Below the lower neutral branch both operators must report decay."""
    R_L, F = POINT_DAMPED
    a_dense = _dense_alpha(dense_case, R_L, F)
    a_qep = _qep_alpha(profile, R_L, F)
    assert a_dense.imag > 0.0, "dense backend should report a damped mode here"
    assert a_qep.imag > 0.0, "QEP should report a damped mode here"
    assert abs(a_dense.imag - a_qep.imag) < TOL_NEAR_NEUTRAL_SI


def test_newton_matches_qep(profile):
    """Newton on the temporal EVP vs companion QEP — different numerical route,
    same operator: must agree to eigensolver precision."""
    R_L, F = POINT_AMPLIFIED
    omega = F * R_L
    a_qep = _qep_alpha(profile, R_L, F)
    alpha, converged = solve_spatial_newton(
        profile, omega, R_L, MA, PR, GAMMA,
        alpha_guess=a_qep, c_target=omega / a_qep.real + 0j,
        N=80, y_max=30.0, tol=1e-8, max_iter=20,
        length_scale="L_star", lambda_mu_ratio=0.0,
    )
    assert converged
    assert abs(alpha.real - a_qep.real) < TOL_WITHIN_FAMILY
    assert abs(alpha.imag - a_qep.imag) < TOL_WITHIN_FAMILY


def test_gaster_pipeline_matches_qep(profile):
    """Fully-automatic temporal -> Gaster -> Newton pipeline vs the QEP root."""
    R_L, F = POINT_AMPLIFIED
    omega = F * R_L
    a_qep = _qep_alpha(profile, R_L, F)
    alpha, converged = solve_spatial_from_temporal(
        profile, omega, R_L, MA, PR, GAMMA,
        N=80, y_max=30.0, length_scale="L_star", lambda_mu_ratio=0.0,
    )
    assert converged
    assert abs(alpha.real - a_qep.real) < TOL_WITHIN_FAMILY
    assert abs(alpha.imag - a_qep.imag) < 1.0e-4  # Gaster-seeded; observed 4e-7


def test_muller_matches_qep_at_its_fixed_lambda(profile):
    """Muller has no lambda_mu_ratio parameter (fixed package default 1.2):
    it must reproduce the QEP evaluated at 1.2. This also pins the documented
    Stokes-vs-1.2 systematic so a silent default change breaks the suite."""
    R_L, F = POINT_AMPLIFIED
    omega = F * R_L
    a_qep_12 = _qep_alpha(profile, R_L, F, lambda_mu_ratio=1.2)
    alpha, converged, sigma_min = solve_spatial_muller(
        profile, omega, R_L, MA, PR, GAMMA,
        alpha_guess=a_qep_12, N=80, y_max=30.0,
        tol=1e-8, max_iter=30, length_scale="L_star",
    )
    assert converged
    assert sigma_min < 1e-8
    assert abs(alpha.real - a_qep_12.real) < TOL_WITHIN_FAMILY
    assert abs(alpha.imag - a_qep_12.imag) < TOL_WITHIN_FAMILY
    # And the systematic itself: Stokes vs 1.2 differ by a real, finite amount.
    a_qep_00 = _qep_alpha(profile, R_L, F, lambda_mu_ratio=0.0)
    assert abs(a_qep_12.imag - a_qep_00.imag) > 1.0e-4
