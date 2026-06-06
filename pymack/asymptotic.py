"""
Freestream asymptotic relations from Mack Appendix B.

These helpers expose the decay/growth roots of the compressible temporal
stability system in the uniform freestream. They are the foundation needed to
replace crude finite-domain truncation with proper boundedness conditions.
"""

from __future__ import annotations

import cmath

import numpy as np
from scipy import linalg


def _stable_square_root(value):
    """Return the principal square root with non-negative real part."""
    root = cmath.sqrt(value)
    if root.real < 0.0:
        root = -root
    return root


def _decaying_characteristic_mask(values, atol=1e-12):
    """Return the mask for characteristic values with bounded far-field behavior."""
    values = np.asarray(values, dtype=complex)
    return (values.real < -atol) | ((np.abs(values.real) <= atol) & (values.imag < 0.0))


def _normalized_null_vector(matrix):
    """Return the smallest-singular-vector null approximation of ``matrix``."""
    _, _, vh = linalg.svd(matrix)
    vector = vh[-1, :].conj()
    scale = np.max(np.abs(vector))
    if scale > 0.0:
        vector = vector / scale
    return vector


def mack_freestream_b_coefficients(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=1.2,
    U1=1.0,
    W1=0.0,
):
    """Return Mack Appendix-B freestream coefficients for a temporal wave."""
    k2 = alpha**2 + beta**2
    phase = alpha * U1 + beta * W1 - alpha * c
    d = float(lambda_mu_ratio)
    E1 = Re + 1j * (2.0 / 3.0) * (2.0 + d) * gamma * Ma**2 * phase

    b11 = k2 + 1j * Re * phase
    b12 = 1j * k2 * (Re + 1j * (1.0 + d) * gamma * Ma**2 * phase / 3.0)
    b13 = -(1.0 + 2.0 * d) * k2 * phase

    b22 = k2 - (Re / E1) * (
        gamma * Ma**2
        - (2.0 / 3.0) * (2.0 + d) * Pr * (gamma - 1.0) * Ma**2 * phase
    )
    b23 = (Re / E1) * (1.0 - (2.0 / 3.0) * (2.0 + d) * Pr) * phase

    b32 = -1j * (gamma - 1.0) * Ma**2 * Pr * phase
    b33 = k2 + 1j * Pr * Re * phase

    return {
        'b11': b11,
        'b12': b12,
        'b13': b13,
        'b22': b22,
        'b23': b23,
        'b32': b32,
        'b33': b33,
        'E1': E1,
        'phase': phase,
        'k2': k2,
    }


def mack_uniform_freestream_matrix(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=1.2,
    U1=1.0,
    W1=0.0,
    T1=1.0,
    mu1=1.0,
):
    """Return the exact uniform-freestream first-order matrix."""
    k2 = alpha**2 + beta**2
    phase = alpha * U1 + beta * W1 - alpha * c
    d = float(lambda_mu_ratio)
    E1 = Re / mu1 + 1j * (2.0 / 3.0) * (2.0 + d) * gamma * Ma**2 * phase

    A = np.zeros((8, 8), dtype=complex)
    A[0, 1] = 1.0
    A[1, 0] = 1j * Re * phase / (mu1 * T1) + k2
    A[1, 3] = 1j * Re * k2 / mu1 - (1.0 + 2.0 * d) * k2 * gamma * Ma**2 * phase / 3.0
    A[1, 4] = (1.0 + 2.0 * d) * k2 * phase / (3.0 * T1)
    A[2, 0] = -1j
    A[2, 3] = -1j * gamma * Ma**2 * phase
    A[2, 4] = 1j * phase / T1
    A[3, 1] = -1j / E1
    A[3, 2] = (-k2 - 1j * Re * phase / (mu1 * T1)) / E1
    A[3, 5] = 1j * 2.0 * (2.0 + d) * phase / (3.0 * T1 * E1)
    A[4, 5] = 1.0
    A[5, 3] = -1j * Re * Pr * (gamma - 1.0) * Ma**2 * phase / mu1
    A[5, 4] = 1j * Re * Pr * phase / (mu1 * T1) + k2
    A[6, 7] = 1.0
    A[7, 6] = 1j * Re * phase / (mu1 * T1) + k2
    return A


def _stable_freestream_branches(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=1.2,
    U1=1.0,
    W1=0.0,
    T1=1.0,
    mu1=1.0,
):
    """Return the stable non-shear and shear characteristic values."""
    A_inf = mack_uniform_freestream_matrix(
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        lambda_mu_ratio=lambda_mu_ratio,
        U1=U1,
        W1=W1,
        T1=T1,
        mu1=mu1,
    )
    eigvals = np.linalg.eigvals(A_inf)
    stable_vals = np.asarray(eigvals[_decaying_characteristic_mask(eigvals)], dtype=complex)
    if stable_vals.size != 4:
        raise ValueError(
            f'expected four stable freestream roots, found {stable_vals.size}'
        )

    phase = alpha * U1 + beta * W1 - alpha * c
    shear_target = -_stable_square_root(alpha**2 + beta**2 + 1j * Re * phase / (mu1 * T1))
    shear_order = np.argsort(np.abs(stable_vals - shear_target))
    shear_vals = stable_vals[shear_order[:2]]
    lambda_3 = np.mean(shear_vals)

    keep = np.ones(stable_vals.size, dtype=bool)
    keep[shear_order[:2]] = False
    remaining = stable_vals[keep]
    if remaining.size != 2:
        raise ValueError('expected two non-shear stable freestream roots')

    remaining = remaining[np.argsort(np.abs(remaining))]
    lambda_1 = remaining[0]
    lambda_5 = remaining[1]
    return lambda_1, lambda_3, lambda_5


def mack_freestream_characteristic_values(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=1.2,
    U1=1.0,
    W1=0.0,
):
    """Return the Appendix-B characteristic values ``lambda_1`` ... ``lambda_8``."""
    lambda_1, lambda_3, lambda_5 = _stable_freestream_branches(
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        lambda_mu_ratio=lambda_mu_ratio,
        U1=U1,
        W1=W1,
    )

    lambdas = {
        'lambda_1': lambda_1,
        'lambda_2': -lambda_1,
        'lambda_3': lambda_3,
        'lambda_4': -lambda_3,
        'lambda_5': lambda_5,
        'lambda_6': -lambda_5,
        'lambda_7': lambda_3,
        'lambda_8': -lambda_3,
    }
    return lambdas


def mack_freestream_decay_values(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=1.2,
    U1=1.0,
    W1=0.0,
):
    """Return the four characteristic values with negative real part."""
    lambdas = mack_freestream_characteristic_values(
        alpha, beta, c, Re, Ma, Pr, gamma,
        lambda_mu_ratio=lambda_mu_ratio, U1=U1, W1=W1,
    )
    return {
        name: value
        for name, value in lambdas.items()
        if value.real < 0.0 or (abs(value.real) < 1e-12 and value.imag < 0.0)
    }


def mack_freestream_decay_basis(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=1.2,
    U1=1.0,
    W1=0.0,
):
    """Return Mack's bounded freestream basis in first-order variables.

    The returned columns span the Appendix-B subspace of bounded solutions for
    the first-order state

        Z = [Z1, DZ1, v, p/(gamma*Ma^2), T, DT, Z7, DZ7]^T.

    The four columns correspond to the decaying characteristic values
    ``lambda_1``, ``lambda_3``, ``lambda_5``, and ``lambda_7``.
    """
    lambdas = mack_freestream_characteristic_values(
        alpha, beta, c, Re, Ma, Pr, gamma,
        lambda_mu_ratio=lambda_mu_ratio, U1=U1, W1=W1,
    )
    A_inf = mack_uniform_freestream_matrix(
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        lambda_mu_ratio=lambda_mu_ratio,
        U1=U1,
        W1=W1,
    )

    columns = []
    labels = []

    for name in ('lambda_1', 'lambda_3', 'lambda_5', 'lambda_7'):
        lam = lambdas[name]

        if name in {'lambda_1', 'lambda_5'}:
            column = _normalized_null_vector(A_inf - lam * np.eye(8))
        elif name == 'lambda_3':
            A3 = -1j / lam
            column = np.array(
                [1.0, lam, A3, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=complex,
            )
        else:
            column = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, lam],
                dtype=complex,
            )

        columns.append(column)
        labels.append(name)

    return np.column_stack(columns), tuple(labels)


def mack_freestream_subspace_residual(state, basis):
    """Return the relative distance of ``state`` from the decay subspace."""
    state = np.asarray(state, dtype=complex).reshape(-1)
    basis = np.asarray(basis, dtype=complex)

    if basis.ndim != 2:
        raise ValueError('basis must be a 2D array')
    if basis.shape[0] != state.shape[0]:
        raise ValueError('state length must match basis row count')

    state_norm = linalg.norm(state)
    if state_norm < 1e-30:
        return 0.0

    coeffs, _, _, _ = linalg.lstsq(basis, state)
    residual = state - basis @ coeffs
    return float(linalg.norm(residual) / state_norm)
