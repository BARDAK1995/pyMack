"""2D temporal shoot-search utilities (compressible LST).

A shooting/root-search alternative to the spectral temporal solver, following
Ozgen & Kircali (2008), Eqs. 2.20-2.28: the four disturbance equations are
recast as a first-order system in the six-component state
``(X1..X6) = (alpha*u, alpha*u', v, p, T, T')``, the three freestream-decaying
solutions are marched to the wall, and the eigenvalue is the ``c`` that makes
the 3x3 wall matrix singular.

The 6x6 coefficient matrix itself is built by delegating to the validated
Mack Appendix-A implementation (:func:`pymack.mack_shooting.
mack_first_order_matrix_6`) at ``beta = 0`` with ``lambda_mu_ratio = 0``
(Stokes closure) -- at those settings the Appendix-A system is algebraically
identical to Ozgen's printed 2-D equations, term for term.  Keeping a single
source of truth for the matrix avoids maintaining two transcriptions of the
same operator.  Shooting for LST eigenvalues is a standard technique
(Mack 1984); this is a re-implementation, not a new method.

References: Ozgen & Kircali (2008), Theor. Comput. Fluid Dyn.;
Mack (1984), AGARD-R-709.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .mack_shooting import mack_first_order_matrix_6


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
    """Return the 6x6 first-order matrix of Ozgen Eqs. 2.21-2.28 at beta=0.

    Delegates to :func:`pymack.mack_shooting.mack_first_order_matrix_6` with
    ``beta = 0`` and ``lambda_mu_ratio = 0`` (Stokes), which reproduces
    Ozgen's printed 2-D system exactly -- with one deliberate correction:
    the X1 coefficient of the printed Eq. (2.24) reads
    ``+i(2 mu dmu/dT T' + (4/3) T'/T)``, but re-deriving the pressure row
    from Eqs. (2.13) and (2.23) gives ``-i(2 (1/mu) dmu/dT T' + (4/3) T'/T)``
    (sign and mu-placement); the corrected form is used here and is what
    drives the wall matrix singular at the spectral solver's eigenvalue.
    """
    return mack_first_order_matrix_6(
        baseflow,
        y,
        float(alpha),
        0.0,
        c,
        Re,
        Ma,
        gamma,
        lambda_mu_ratio=0.0,
        length_scale=length_scale,
    )


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
