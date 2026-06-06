"""Validation of Mack Appendix-A/B helper utilities."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.asymptotic import (
    mack_freestream_characteristic_values,
    mack_freestream_decay_basis,
    mack_freestream_subspace_residual,
)
from pymack.mack_conditions import make_mack_profile
from pymack.mack_shooting import mack_first_order_matrix_3d
from pymack.scales import rescale_baseflow_derivatives


class UniformFreestreamProfile:
    """Synthetic uniform freestream profile for exact Appendix-B checks."""

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        zeros = np.zeros_like(y)
        ones = np.ones_like(y)
        return {
            'U': ones,
            'dU': zeros,
            'd2U': zeros,
            'T': ones,
            'dT': zeros,
            'd2T': zeros,
            'mu': ones,
            'dmu': zeros,
            'dmu_dT': zeros,
            'd2mu_dT2': zeros,
            'kappa': ones,
            'dkappa': zeros,
            'dkappa_dT': zeros,
            'd2kappa_dT2': zeros,
            'Pr_local': np.full_like(y, 0.72),
        }


def test_appendix_b_decay_basis():
    alpha = 0.05
    beta = 0.03
    c = 0.9 + 0.01j
    Re = 1500.0
    Ma = 4.5
    Pr = 0.72
    gamma = 1.4

    lambdas = mack_freestream_characteristic_values(alpha, beta, c, Re, Ma, Pr, gamma)
    assert abs(lambdas['lambda_3'] - lambdas['lambda_7']) < 1e-12

    basis, labels = mack_freestream_decay_basis(alpha, beta, c, Re, Ma, Pr, gamma)
    assert basis.shape == (8, 4)
    assert labels == ('lambda_1', 'lambda_3', 'lambda_5', 'lambda_7')

    coeffs = np.array([1.0 + 0.0j, -0.7 + 0.1j, 0.2 - 0.3j, 0.9 + 0.5j])
    state = basis @ coeffs
    residual = mack_freestream_subspace_residual(state, basis)
    assert residual < 1e-12


def test_appendix_b_decay_basis_satisfies_uniform_first_order_matrix():
    """Each Appendix-B decay vector should satisfy the uniform freestream ODE."""
    alpha = 0.05
    beta = 0.03
    c = 0.9 + 0.01j
    Re = 1500.0
    Ma = 4.5
    Pr = 0.72
    gamma = 1.4

    profile = UniformFreestreamProfile()
    A_inf = mack_first_order_matrix_3d(profile, 0.0, alpha, beta, c, Re, Ma, gamma)
    basis, labels = mack_freestream_decay_basis(alpha, beta, c, Re, Ma, Pr, gamma)
    lambdas = mack_freestream_characteristic_values(alpha, beta, c, Re, Ma, Pr, gamma)

    for idx, label in enumerate(labels):
        residual = A_inf @ basis[:, idx] - lambdas[label] * basis[:, idx]
        assert np.linalg.norm(residual) < 1e-10


def test_appendix_a_temperature_gradient_factor():
    profile = make_mack_profile(4.5)
    y = 4.0
    alpha = 0.05
    beta = alpha * np.tan(np.deg2rad(60.0))
    c = 0.9 + 0.01j
    Re = 1500.0
    Ma = 4.5
    gamma = 1.4

    A = mack_first_order_matrix_3d(
        profile, y, alpha, beta, c, Re, Ma, gamma, length_scale='L_star'
    )
    bf = profile(np.array([y / profile._delta_star]))
    bf = rescale_baseflow_derivatives(bf, profile._delta_star, target_scale='L_star')
    expected = -2.0 * bf['dkappa_dT'][0] * bf['dT'][0] / bf['kappa'][0]
    assert abs(A[5, 5] - expected) < 1e-10


if __name__ == '__main__':
    test_appendix_b_decay_basis()
    test_appendix_b_decay_basis_satisfies_uniform_first_order_matrix()
    test_appendix_a_temperature_gradient_factor()
    print('Appendix-A/B helper tests passed.')
