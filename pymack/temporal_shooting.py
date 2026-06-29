"""2D temporal shoot-search utilities (compressible LST).

A shooting/root-search alternative to the spectral temporal solver. The
first-order system implemented here (the ``ozgen_*`` helpers below) follows the
printed equations of Ozgen & Kircali (2008), Eqs. 22-28, for 2-D disturbances
over a 2-D mean flow -- kept separate from the Mack Appendix-A shooter
(:mod:`pymack.mack_shooting`) because that uses its own variable set and
viscous coefficients. Shooting for LST eigenvalues is a standard technique
(Mack 1984); this is a re-implementation, not a new method.

References: Ozgen & Kircali (2008), Theor. Comput. Fluid Dyn.;
Mack (1984), AGARD-R-709.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .mack_shooting import _sample_scaled_baseflow


def ozgen_first_order_matrix_2d(
    baseflow,
    y,
    alpha,
    c,
    Re,
    Ma,
    gamma=1.4,
    length_scale='L_star',
):
    """Return the 6x6 first-order matrix from Ozgen Eqs. 22-28 at beta=0."""
    bf = _sample_scaled_baseflow(baseflow, np.array([y]), length_scale)

    U = complex(bf['U'][0])
    DU = complex(bf['dU'][0])
    D2U = complex(bf['d2U'][0])
    T = complex(bf['T'][0])
    DT = complex(bf['dT'][0])
    mu = complex(bf['mu'][0])
    dmu_dT = complex(bf['dmu_dT'][0])
    d2mu_dT2 = complex(bf['d2mu_dT2'][0])
    dkappa = complex(bf['dkappa'][0])
    dkappa_dT = complex(bf['dkappa_dT'][0])
    d2kappa_dT2 = complex(bf['d2kappa_dT2'][0])

    alpha = float(alpha)
    k2 = alpha**2
    phase = alpha * (U - complex(c))
    mu_T_over_mu = dmu_dT / mu
    C = Re / (mu * T) + 1j * (4.0 / 3.0) * gamma * Ma**2 * phase

    A = np.zeros((6, 6), dtype=complex)

    # X1' = X2
    A[0, 1] = 1.0

    # X2', Eq. 23 at beta=W=0.
    A[1, 0] = 1j * Re * phase / (mu * T) + k2
    A[1, 1] = -mu_T_over_mu * DT
    A[1, 2] = Re * alpha * DU / T - 1j * k2 * (1.0 / T + mu_T_over_mu)
    A[1, 3] = 1j * k2 * Re * phase / (mu * T)
    A[1, 4] = (
        k2 * phase / T
        - mu_T_over_mu * alpha * DU
        - (d2mu_dT2 / mu) * DT * alpha * DU
    )
    A[1, 5] = -mu_T_over_mu * alpha * DU

    # X3', Eq. 24.
    A[2, 0] = -1j
    A[2, 2] = DT / T
    A[2, 3] = -1j * gamma * Ma**2 * phase
    A[2, 4] = 1j * phase / T

    # X4', Eq. 25 divided by C.
    A[3, 0] = 1j * (2.0 * dmu_dT * DT + (4.0 / 3.0) * DT / T) / C
    A[3, 1] = -1j / C
    A[3, 2] = (
        (4.0 / 3.0) * DT / T
        - k2
        + 1j * Re * phase / (mu * T)
        + (4.0 / 3.0) * mu_T_over_mu * DT / T
    ) / C
    A[3, 3] = (
        -1j
        * (4.0 / 3.0)
        * gamma
        * Ma**2
        * DT
        * (phase / T + alpha * DU * mu_T_over_mu * phase)
    ) / C
    A[3, 4] = (
        1j
        * (
            (4.0 / (3.0 * T)) * alpha * DU
            + mu_T_over_mu * DT * alpha * DU
            + mu_T_over_mu * alpha * D2U
        )
    ) / C
    A[3, 5] = 1j * (4.0 / (3.0 * T)) * phase / C

    # X5' = X6
    A[4, 5] = 1.0

    # X6', Eq. 28 at beta=W=0.
    A[5, 0] = -(gamma - 1.0) * alpha * Ma**2 * alpha * DU
    A[5, 2] = (
        alpha * Re / (mu * T)
        - 2.0 * (gamma - 1.0) * alpha * Ma**2 * alpha * DU
    )
    A[5, 3] = -(gamma - 1.0) * alpha * Re * Ma**2 * phase / (mu * T)
    A[5, 4] = (
        k2
        + 1j * alpha * Re * phase / mu
        - (dkappa + d2kappa_dT2 * DT**2) / mu
        - (gamma - 1.0) * alpha * Ma**2 * dmu_dT * DU**2 / mu
    )
    A[5, 5] = -2.0 * dkappa_dT * DT / mu

    return A


def ozgen_freestream_decay_basis_2d(baseflow, alpha, c, Re, Ma, gamma=1.4, y_max=40.0, length_scale='L_star'):
    """Return three freestream eigenvectors that decay as y -> infinity."""
    A_inf = ozgen_first_order_matrix_2d(
        baseflow,
        y_max,
        alpha,
        c,
        Re,
        Ma,
        gamma=gamma,
        length_scale=length_scale,
    )
    eigvals, eigvecs = np.linalg.eig(A_inf)
    decaying = np.where(eigvals.real < 0.0)[0]
    if len(decaying) < 3:
        decaying = np.argsort(eigvals.real)[:3]
    else:
        decaying = decaying[np.argsort(eigvals[decaying].real)[:3]]
    return eigvecs[:, decaying], eigvals[decaying]


def integrate_ozgen_bounded_basis_2d(
    baseflow,
    alpha,
    c,
    Re,
    Ma,
    *,
    gamma=1.4,
    y_max=40.0,
    length_scale='L_star',
    n_steps=800,
):
    """Integrate Ozgen's three bounded 2D solutions from freestream to wall."""
    basis, _ = ozgen_freestream_decay_basis_2d(
        baseflow,
        alpha,
        c,
        Re,
        Ma,
        gamma=gamma,
        y_max=y_max,
        length_scale=length_scale,
    )
    Y, _ = np.linalg.qr(basis)
    y_grid = np.linspace(float(y_max), 0.0, int(n_steps) + 1)

    for y0, y1 in zip(y_grid[:-1], y_grid[1:]):
        h = y1 - y0
        ym = 0.5 * (y0 + y1)
        A0 = ozgen_first_order_matrix_2d(
            baseflow, y0, alpha, c, Re, Ma, gamma=gamma, length_scale=length_scale)
        Am = ozgen_first_order_matrix_2d(
            baseflow, ym, alpha, c, Re, Ma, gamma=gamma, length_scale=length_scale)
        A1 = ozgen_first_order_matrix_2d(
            baseflow, y1, alpha, c, Re, Ma, gamma=gamma, length_scale=length_scale)
        k1 = A0 @ Y
        k2 = Am @ (Y + 0.5 * h * k1)
        k3 = Am @ (Y + 0.5 * h * k2)
        k4 = A1 @ (Y + h * k3)
        Y = Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Y, _ = np.linalg.qr(Y)

    return Y


def ozgen_temporal_wall_matrix_2d(
    baseflow,
    alpha,
    c,
    Re,
    Ma,
    *,
    gamma=1.4,
    y_max=40.0,
    length_scale='L_star',
    n_steps=800,
):
    """Return the 3x3 wall matrix for X1(0)=X3(0)=X5(0)=0."""
    wall_basis = integrate_ozgen_bounded_basis_2d(
        baseflow,
        alpha,
        c,
        Re,
        Ma,
        gamma=gamma,
        y_max=y_max,
        length_scale=length_scale,
        n_steps=n_steps,
    )
    return wall_basis[[0, 2, 4], :]


def ozgen_temporal_sigma_min_2d(
    baseflow,
    alpha,
    c,
    Re,
    Ma,
    *,
    gamma=1.4,
    y_max=40.0,
    length_scale='L_star',
    n_steps=800,
):
    """Return the smallest singular value of Ozgen's 2D wall matrix."""
    wall_matrix = ozgen_temporal_wall_matrix_2d(
        baseflow,
        alpha,
        c,
        Re,
        Ma,
        gamma=gamma,
        y_max=y_max,
        length_scale=length_scale,
        n_steps=n_steps,
    )
    return float(np.linalg.svd(wall_matrix, compute_uv=False)[-1])


def solve_ozgen_temporal_mode_2d_sigma_min(
    baseflow,
    alpha,
    c_guess,
    Re,
    Ma,
    *,
    gamma=1.4,
    y_max=40.0,
    length_scale='L_star',
    n_steps=800,
    c_real_bounds=(0.0, 1.2),
    c_imag_bounds=(-0.2, 0.2),
    xatol=1e-7,
    fatol=1e-8,
    max_iter=120,
):
    """Minimize Ozgen's 2D wall singular value over complex phase speed."""
    history = []

    def objective(x):
        c_val = complex(x[0], x[1])
        if (
            c_val.real < c_real_bounds[0]
            or c_val.real > c_real_bounds[1]
            or c_val.imag < c_imag_bounds[0]
            or c_val.imag > c_imag_bounds[1]
        ):
            return 1e6
        sigma = ozgen_temporal_sigma_min_2d(
            baseflow,
            alpha,
            c_val,
            Re,
            Ma,
            gamma=gamma,
            y_max=y_max,
            length_scale=length_scale,
            n_steps=n_steps,
        )
        history.append((c_val, sigma))
        return sigma

    result = minimize(
        objective,
        x0=np.array([complex(c_guess).real, complex(c_guess).imag], dtype=float),
        method='Nelder-Mead',
        options={
            'maxiter': int(max_iter),
            'xatol': float(xatol),
            'fatol': float(fatol),
        },
    )
    c_opt = complex(result.x[0], result.x[1])
    sigma = ozgen_temporal_sigma_min_2d(
        baseflow,
        alpha,
        c_opt,
        Re,
        Ma,
        gamma=gamma,
        y_max=y_max,
        length_scale=length_scale,
        n_steps=n_steps,
    )
    return c_opt, sigma, bool(result.success), history
