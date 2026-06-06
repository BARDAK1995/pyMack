"""Validate the reduced collocation operator against Mack Appendix A."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.equations import (
    DEFAULT_LAMBDA_MU_RATIO,
    momentum_viscous_coefficients,
    transport_conductivity_data,
    transport_temperature_derivatives,
)
from pymack.baseflow import make_ozgen_profile
from pymack.mack_conditions import make_mack_profile
from pymack.mack_shooting import (
    _sample_scaled_baseflow,
    mack_first_order_matrix_3d,
    mack_first_order_matrix_6,
    temporal_shooting_wall_matrix_3d,
    temporal_shooting_wall_matrix_6,
)
from pymack.solver import _assemble_temporal_compressible_3d_evp


def reduced_first_order_matrix_3d(
    baseflow,
    y,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
):
    """Recover Mack's first-order form from the reduced 3D collocation equations."""
    bf = _sample_scaled_baseflow(baseflow, np.array([y]), length_scale)

    U = complex(bf['U'][0])
    DU = complex(bf['dU'][0])
    D2U = complex(bf['d2U'][0])
    T = complex(bf['T'][0])
    DT = complex(bf['dT'][0])
    D2T = complex(bf['d2T'][0])
    rho = complex(bf['rho'][0])
    mu = complex(bf['mu'][0])
    dmu = complex(bf['dmu'][0])

    dmu_dT_v, d2mu_dT2_v = transport_temperature_derivatives(bf)
    dmu_dT = complex(dmu_dT_v[0])
    d2mu_dT2 = complex(d2mu_dT2_v[0])

    kappa_v, dkappa_v, dkappa_dT_v, d2kappa_dT2_v, needs_pr_prefactor = (
        transport_conductivity_data(bf, Pr)
    )
    kappa = complex(kappa_v[0])
    dkappa = complex(dkappa_v[0])
    dkappa_dT = complex(dkappa_dT_v[0])
    d2kappa_dT2 = complex(d2kappa_dT2_v[0])

    (
        x_alpha2_coeff,
        cross_grad_coeff,
        y_laplacian_coeff,
        y_u_algebraic_coeff,
    ) = momentum_viscous_coefficients(lambda_mu_ratio)

    k = float(np.hypot(alpha, beta))
    k2 = k**2
    gm1 = gamma - 1.0
    Ma2 = Ma**2
    rho_inv = 1.0 / rho

    visc = rho_inv / Re
    cond = rho_inv / Re if not needs_pr_prefactor else rho_inv / (Pr * Re)
    diss = gm1 * Ma2 * rho_inv / Re
    phase = alpha * U - alpha * c

    if k2 <= 0.0:
        raise ValueError('at least one of alpha or beta must be non-zero')

    inv_q = 1.0 / (visc * mu)
    inv_t = 1.0 / (cond * kappa)

    A = np.zeros((8, 8), dtype=complex)

    def z3_prime(z):
        z1, _, z3, z4, z5, _, _, _ = z
        return (
            -1j * z1
            + (DT / T) * z3
            - 1j * gamma * Ma2 * phase * z4
            + 1j * phase / T * z5
        )

    def z2_prime(z, z3p):
        z1, z2, z3, z4, z5, z6, _, _ = z
        return inv_q * (
            1j * phase * z1
            + visc * x_alpha2_coeff * k2 * mu * z1
            + alpha * DU * z3
            - 1j * k2 * visc * (cross_grad_coeff * mu * z3p + dmu * z3)
            - visc * dmu * z2
            - visc * (
                alpha * dmu_dT * D2U * z5
                + alpha * d2mu_dT2 * DU * DT * z5
                + alpha * dmu_dT * DU * z6
            )
            + 1j * k2 * rho_inv * z4
        )

    def z8_prime(z):
        _, _, z3, _, z5, z6, z7, z8 = z
        return inv_q * (
            1j * phase * z7
            + visc * k2 * mu * z7
            - beta * DU * z3
            - visc * dmu * z8
            + visc * (
                beta * dmu_dT * D2U * z5
                + beta * d2mu_dT2 * DU * DT * z5
                + beta * dmu_dT * DU * z6
            )
        )

    def z6_prime(z):
        _, z2, z3, z4, z5, z6, _, z8 = z
        spanwise_coupling = (
            2.0 * beta * mu * DU * diss / k2 * z8
            if include_spanwise_dissipation_coupling
            else 0.0
        )
        return inv_t * (
            -2.0 * alpha * mu * DU * diss / k2 * z2
            + (DT - 2j * alpha * diss * mu * DU) * z3
            + spanwise_coupling
            + (
                1j * phase
                - cond * dkappa_dT * D2T
                - cond * d2kappa_dT2 * DT * DT
                - diss * dmu_dT * DU * DU
                + k2 * cond * kappa
            ) * z5
            - 2.0 * cond * dkappa * z6
            - 1j * gm1 * Ma2 * phase * rho_inv * z4
        )

    def z4_prime(z, z3p):
        z1, z2, z3, z4, z5, z6, _, _ = z

        ratio = DT / T
        ratio_prime = D2T / T - (DT / T) ** 2
        phase_prime = alpha * DU
        phase_over_t_prime = phase_prime / T - phase * DT / (T * T)

        z3pp_without_z4p = (
            -1j * z2
            + ratio_prime * z3
            + ratio * z3p
            - 1j * gamma * Ma2 * phase_prime * z4
            + 1j * phase_over_t_prime * z5
            + 1j * phase / T * z6
        )
        z3pp_z4p_coeff = -1j * gamma * Ma2 * phase

        coeff_z4p = Re / mu - y_laplacian_coeff * z3pp_z4p_coeff
        rhs = (
            -1j * (
                cross_grad_coeff * z2
                + y_u_algebraic_coeff * dmu / mu * z1
            )
            + 1j * Re * phase / (mu * T) * z3
            - y_laplacian_coeff * z3pp_without_z4p
            - y_laplacian_coeff * dmu / mu * z3p
            + k2 * z3
            - 1j * alpha * dmu_dT * DU / mu * z5
        )
        return -rhs / coeff_z4p

    for j in range(8):
        z = np.zeros(8, dtype=complex)
        z[j] = 1.0

        z3p = z3_prime(z)
        z4p = z4_prime(z, z3p)
        A[0, j] = z[1]
        A[1, j] = z2_prime(z, z3p)
        A[2, j] = z3p
        A[3, j] = z4p
        A[4, j] = z[5]
        A[5, j] = z6_prime(z)
        A[6, j] = z[7]
        A[7, j] = z8_prime(z)

    return A


def check_case(ma, re_l, alpha, beta, c, y, tol=1e-10):
    profile = make_mack_profile(ma)
    A_reduced = reduced_first_order_matrix_3d(
        profile, y, alpha, beta, c, re_l, ma, 0.72, 1.4, length_scale='L_star'
    )
    A_mack = mack_first_order_matrix_3d(
        profile, y, alpha, beta, c, re_l, ma, 1.4, length_scale='L_star'
    )
    return np.max(np.abs(A_reduced - A_mack))


def test_reduced_operator_matches_appendix_a():
    cases = [
        (1.3, 500.0, 0.075, 0.075, 0.6100864888364268 + 0.12417131483672231j, 2.0),
        (4.5, 1500.0, 0.050, 0.050 * np.tan(np.deg2rad(60.0)),
         0.88630640521563 + 0.02343324359758292j, 3.5),
        (10.0, 1500.0, 0.040, 0.040 * np.tan(np.deg2rad(55.0)),
         0.14514727826255208 + 0.016774491888309546j, 2.5),
    ]
    for case in cases:
        error = check_case(*case)
        assert error < 1e-8, (
            f'reduced/operator mismatch exceeds tolerance for case {case}: '
            f'{error:.3e}'
        )


def test_temporal_evp_continuity_uses_u_minus_c_phase():
    """The finite-domain EVP continuity row must use alpha*(U-c), not alpha*(U+c)."""
    profile = make_mack_profile(2.2)
    alpha = 0.055
    beta = 0.010
    Ma = 2.2
    gamma = 1.4

    _, B, _, _, n, _, _, bf = _assemble_temporal_compressible_3d_evp(
        profile,
        alpha,
        beta,
        500.0,
        Ma,
        0.72,
        gamma,
        N=12,
        y_max=8.0,
        length_scale='L_star',
    )

    row = 3
    assert B[row, 3 * n + row] == -1j * alpha / bf['T'][row]
    assert B[row, 4 * n + row] == 1j * alpha * gamma * Ma**2


def test_reduced_operator_matches_appendix_a_ozgen_profile():
    """The same local first-order reduction should hold on the shared Ozgen profile."""
    profile = make_ozgen_profile(
        4.5,
        T_edge=250.0,
        Re_delta_star=800.0,
        n_points=3000,
        eta_max=40.0,
    )
    alpha = 0.08
    beta = alpha * np.tan(np.deg2rad(60.0))
    c = 0.2509073892488834 + 0.14718738090518793j

    errors = [
        np.max(np.abs(
            reduced_first_order_matrix_3d(
                profile, y, alpha, beta, c, 800.0, 4.5, 0.72, 1.4,
                length_scale='L_star',
            )
            - mack_first_order_matrix_3d(
                profile, y, alpha, beta, c, 800.0, 4.5, 1.4,
                length_scale='L_star',
            )
        ))
        for y in (40.0, 120.0)
    ]
    assert max(errors) < 1e-8, (
        'reduced/operator mismatch exceeds tolerance on Ozgen shared profile: '
        f'{max(errors):.3e}'
    )


def test_exact_shooting_supports_mack_sixth_order_coupling_switch():
    """Mack's 6th-order 3D approximation should drop only Appendix-A a68."""
    profile = make_mack_profile(2.2)
    alpha = 0.055
    beta = alpha * np.tan(np.deg2rad(60.0))
    c = 0.55 + 0.012j

    full = mack_first_order_matrix_3d(
        profile,
        3.0,
        alpha,
        beta,
        c,
        500.0,
        2.2,
        1.4,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=True,
    )
    sixth = mack_first_order_matrix_3d(
        profile,
        3.0,
        alpha,
        beta,
        c,
        500.0,
        2.2,
        1.4,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=False,
    )
    scaled = mack_first_order_matrix_3d(
        profile,
        3.0,
        alpha,
        beta,
        c,
        500.0,
        2.2,
        1.4,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=True,
        spanwise_dissipation_coupling_scale=2.5,
    )
    reduced_sixth = reduced_first_order_matrix_3d(
        profile,
        3.0,
        alpha,
        beta,
        c,
        500.0,
        2.2,
        0.72,
        1.4,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=False,
    )

    diff = full - sixth
    nonzero = np.argwhere(np.abs(diff) > 1e-12)
    assert nonzero.tolist() == [[5, 7]]
    assert abs(full[5, 7]) > 0.0
    assert sixth[5, 7] == 0.0
    assert np.max(np.abs(sixth - reduced_sixth)) < 1e-8

    scaled_diff = scaled - full
    scaled_nonzero = np.argwhere(np.abs(scaled_diff) > 1e-12)
    assert scaled_nonzero.tolist() == [[5, 7]]
    assert abs(scaled[5, 7] - 2.5 * full[5, 7]) < 1e-12


def test_primary_sixth_order_wall_determinant_factorizes_from_a68_zero_system():
    """The true sixth-order determinant should be the primary block of a68=0."""
    profile = make_mack_profile(1.6, condition='table_10_1')
    alpha = 0.070
    beta = alpha * np.tan(np.deg2rad(55.0))
    c = 0.45 + 0.02j
    Re = 500.0
    Ma = 1.6

    A6 = mack_first_order_matrix_6(
        profile,
        2.5,
        alpha,
        beta,
        c,
        Re,
        Ma,
        1.4,
        length_scale='L_star',
    )
    A8_sixth = mack_first_order_matrix_3d(
        profile,
        2.5,
        alpha,
        beta,
        c,
        Re,
        Ma,
        1.4,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=False,
    )
    assert np.max(np.abs(A6 - A8_sixth[:6, :6])) < 1e-12

    y_max = 8.0
    wall6 = temporal_shooting_wall_matrix_6(
        profile,
        alpha,
        beta,
        c,
        Re,
        Ma,
        0.72,
        1.4,
        y_max,
        length_scale='L_star',
        wall_bc='adiabatic',
        method='ivp',
    )
    wall8 = temporal_shooting_wall_matrix_3d(
        profile,
        alpha,
        beta,
        c,
        Re,
        Ma,
        0.72,
        1.4,
        y_max,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=False,
        wall_bc='adiabatic',
        method='ivp',
    )

    assert wall6.shape == (3, 3)
    assert wall8.shape == (4, 4)
    matrix_scale = max(np.max(np.abs(wall8[:3, :3])), 1.0)
    assert np.max(np.abs(wall6 - wall8[:3, :3])) / matrix_scale < 1e-6
    assert np.max(np.abs(wall8[:3, 3])) / matrix_scale < 1e-12

    det8 = np.linalg.det(wall8)
    factored_det8 = np.linalg.det(wall8[:3, :3]) * wall8[3, 3]
    scale = max(abs(det8), abs(factored_det8), 1.0)
    assert abs(det8 - factored_det8) / scale < 1e-6


if __name__ == '__main__':
    test_reduced_operator_matches_appendix_a()
    test_reduced_operator_matches_appendix_a_ozgen_profile()
    test_exact_shooting_supports_mack_sixth_order_coupling_switch()
    test_primary_sixth_order_wall_determinant_factorizes_from_a68_zero_system()
    print('Reduced operator reproduces the Eq. 8.9 / Appendix-A first-order matrix.')
