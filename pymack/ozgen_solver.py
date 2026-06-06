"""Ozgen & Kircali 2D temporal EVP solver.

This module assembles the two-dimensional temporal system directly from
Ozgen & Kircali (2008), Eqs. 2.12, 2.13, 2.15, 2.16, and 2.17.  It is kept
separate from :mod:`lst.solver` because that solver uses the Mack-style
enthalpy/pressure energy row, while Ozgen's paper advances a temperature
equation with explicit dilatation work.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .spectral import chebyshev_D, physical_derivatives
from .equations import transport_conductivity_data, transport_temperature_derivatives
from .scales import delta_star_over_lstar, rescale_baseflow_derivatives
from .solver import _temperature_wall_operator


def _scaled_problem(baseflow, y, D1, D2, length_scale):
    if length_scale == 'delta_star':
        return baseflow(y), D1, D2
    if length_scale != 'L_star':
        raise ValueError("length_scale must be 'delta_star' or 'L_star'")
    delta_over_l = delta_star_over_lstar(baseflow)
    bf = baseflow(y / delta_over_l)
    return (
        rescale_baseflow_derivatives(bf, delta_over_l, target_scale='L_star'),
        D1,
        D2,
    )


def solve_temporal_ozgen_2d(
    baseflow,
    alpha,
    Re,
    Ma,
    Pr=0.72,
    gamma=1.4,
    *,
    N=128,
    y_max=None,
    L=None,
    wall_bc='isothermal',
    length_scale='delta_star',
):
    """Solve Ozgen's 2D temporal compressible EVP ``A phi = c B phi``.

    The state is ``[u, v, T, p]`` where ``p`` is pressure divided by
    ``gamma Ma^2``.  Inputs are in the selected length scale.
    """
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)
    bf, D1, D2 = _scaled_problem(baseflow, y, D1, D2, length_scale)

    n = len(y)
    I = np.eye(n)
    alpha = float(alpha)
    ia = 1j * alpha
    a2 = alpha**2

    U = np.asarray(bf['U'])
    dU = np.asarray(bf['dU'])
    d2U = np.asarray(bf['d2U'])
    T = np.asarray(bf['T'])
    dT = np.asarray(bf['dT'])
    d2T = np.asarray(bf['d2T'])
    rho = np.asarray(bf['rho'])
    mu = np.asarray(bf['mu'])
    dmu = np.asarray(bf['dmu'])

    dmu_dT_v, d2mu_dT2_v = transport_temperature_derivatives(bf)
    kappa_v, _dkappa_v, dkappa_dT_v, d2kappa_dT2_v, _ = (
        transport_conductivity_data(bf, Pr)
    )
    pr_local = np.asarray(bf.get('Pr_local', np.full_like(T, Pr)))

    Ub = np.diag(U)
    dUb = np.diag(dU)
    d2Ub = np.diag(d2U)
    dTb = np.diag(dT)
    d2Tb = np.diag(d2T)
    rhob_inv = np.diag(1.0 / rho)
    Tb_inv = np.diag(1.0 / T)
    mub = np.diag(mu)
    dmub = np.diag(dmu)
    dmu_dT = np.diag(dmu_dT_v)
    d2mu_dT2 = np.diag(d2mu_dT2_v)
    kappab_inv = np.diag(1.0 / kappa_v)
    dkappa_dT = np.diag(dkappa_dT_v)
    d2kappa_dT2 = np.diag(d2kappa_dT2_v)

    visc = rhob_inv / Re
    cond = np.diag(gamma * mu / (pr_local * Re * rho))
    diss = np.diag(gamma * (gamma - 1.0) * Ma**2 / (Re * rho))

    nn = 4 * n
    A = np.zeros((nn, nn), dtype=complex)
    B = np.zeros((nn, nn), dtype=complex)

    def blk(i, j):
        return (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))

    # Continuity plus equation of state:
    # i alpha u + Dv - T'/T v + i alpha (U-c)(gamma M^2 p - T_hat/T)=0
    A[blk(0, 0)] = ia * I
    A[blk(0, 1)] = D1 - Tb_inv @ dTb
    A[blk(0, 2)] = -ia * Ub @ Tb_inv
    A[blk(0, 3)] = ia * gamma * Ma**2 * Ub
    B[blk(0, 2)] = -ia * Tb_inv
    B[blk(0, 3)] = ia * gamma * Ma**2 * I

    # Streamwise momentum, Eq. 2.12 at beta=0, divided by rho.
    A[blk(1, 0)] = (
        ia * Ub
        - visc @ (mub @ D2 + dmub @ D1)
        + (4.0 / 3.0) * a2 * visc @ mub
    )
    A[blk(1, 1)] = (
        dUb
        - ia * visc @ ((1.0 / 3.0) * mub @ D1 + dmub)
    )
    A[blk(1, 2)] = -visc @ (
        dmu_dT @ dUb @ D1
        + dmu_dT @ d2Ub
        + d2mu_dT2 @ dUb @ dTb
    )
    A[blk(1, 3)] = ia * rhob_inv
    B[blk(1, 0)] = ia * I

    # Wall-normal momentum, Eq. 2.13 at beta=0, divided by rho.
    A[blk(2, 0)] = -ia * visc @ (
        (1.0 / 3.0) * mub @ D1
        - (2.0 / 3.0) * dmub
    )
    A[blk(2, 1)] = (
        ia * Ub
        - visc @ ((4.0 / 3.0) * mub @ D2 + (4.0 / 3.0) * dmub @ D1)
        + a2 * visc @ mub
    )
    A[blk(2, 2)] = -ia * visc @ (dmu_dT @ dUb)
    A[blk(2, 3)] = rhob_inv @ D1
    B[blk(2, 1)] = ia * I

    # Temperature equation, Eq. 2.15, divided by rho.
    conduction_operator = (
        D2
        - a2 * I
        + kappab_inv @ dkappa_dT @ (2.0 * dTb @ D1 + d2Tb)
        + kappab_inv @ d2kappa_dT2 @ dTb @ dTb
    )
    A[blk(3, 0)] = (
        (gamma - 1.0) * rhob_inv @ (ia * I)
        - diss @ (2.0 * mub @ dUb @ D1)
    )
    A[blk(3, 1)] = (
        dTb
        + (gamma - 1.0) * rhob_inv @ D1
        - diss @ (2j * alpha * mub @ dUb)
    )
    A[blk(3, 2)] = (
        ia * Ub
        - cond @ conduction_operator
        - diss @ (dmu_dT @ dUb @ dUb)
    )
    B[blk(3, 2)] = ia * I

    wall = n - 1
    free = 0
    for var in range(2):
        for loc in (wall, free):
            row = var * n + loc
            A[row, :] = 0.0
            B[row, :] = 0.0
            A[row, row] = 1.0

    temp_slice = slice(2 * n, 3 * n)
    temp_wall_row = 2 * n + wall
    temp_free_row = 2 * n + free
    A[temp_wall_row, :] = 0.0
    B[temp_wall_row, :] = 0.0
    A[temp_wall_row, temp_slice] = _temperature_wall_operator(D1, n, wall_bc)
    A[temp_free_row, :] = 0.0
    B[temp_free_row, :] = 0.0
    A[temp_free_row, temp_free_row] = 1.0

    eigenvalues, eigenvectors = linalg.eig(A, B)
    valid = np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]
    physical = (
        (eigenvalues.real > -0.5)
        & (eigenvalues.real < 1.5)
        & (np.abs(eigenvalues.imag) < 0.5)
    )
    eigenvalues = eigenvalues[physical]
    eigenvectors = eigenvectors[:, physical]
    idx = np.argsort(-eigenvalues.imag)
    return eigenvalues[idx], eigenvectors[:, idx], y
