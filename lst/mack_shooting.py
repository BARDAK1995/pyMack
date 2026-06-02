"""Appendix-A/B shooting utilities for Mack's compressible temporal problem."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .asymptotic import mack_freestream_decay_basis
from .equations import DEFAULT_LAMBDA_MU_RATIO
from .scales import delta_star_over_lstar, rescale_baseflow_derivatives


def _sample_scaled_baseflow(baseflow, y, length_scale):
    """Sample the mean flow at a scalar/array ``y`` for the requested scale."""
    y = np.asarray(y, dtype=float)
    if length_scale == 'delta_star':
        return baseflow(y)
    if length_scale != 'L_star':
        raise ValueError("length_scale must be 'delta_star' or 'L_star'")

    delta_over_l = delta_star_over_lstar(baseflow)
    bf = baseflow(y / delta_over_l)
    return rescale_baseflow_derivatives(bf, delta_over_l, target_scale='L_star')


def _freestream_sigma(baseflow, Pr, length_scale):
    """Return the freestream Prandtl number consistent with the mean flow."""
    bf = _sample_scaled_baseflow(baseflow, np.array([50.0]), length_scale)
    if 'Pr_local' in bf:
        return complex(bf['Pr_local'][0])
    return complex(Pr)


def _wall_condition_rows_3d(wall_bc):
    """Return the first-order wall rows for isothermal/adiabatic walls."""
    if wall_bc == 'isothermal':
        return (0, 2, 4, 6)
    if wall_bc == 'adiabatic':
        return (0, 2, 5, 6)
    raise ValueError("wall_bc must be 'isothermal' or 'adiabatic'")


def _wall_condition_rows_6(wall_bc):
    """Return the primary sixth-order wall rows."""
    if wall_bc == 'isothermal':
        return (0, 2, 4)
    if wall_bc == 'adiabatic':
        return (0, 2, 5)
    raise ValueError("wall_bc must be 'isothermal' or 'adiabatic'")


def mack_first_order_matrix_3d(
    baseflow,
    y,
    alpha,
    beta,
    c,
    Re,
    Ma,
    gamma,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
):
    """Return Mack Appendix-A first-order matrix for a 2D mean flow."""
    bf = _sample_scaled_baseflow(baseflow, np.array([y]), length_scale)

    U = complex(bf['U'][0])
    DU = complex(bf['dU'][0])
    D2U = complex(bf['d2U'][0])
    T = complex(bf['T'][0])
    DT = complex(bf['dT'][0])
    D2T = complex(bf['d2T'][0])
    mu = complex(bf['mu'][0])
    dmu_dT = complex(bf['dmu_dT'][0])
    d2mu_dT2 = complex(bf['d2mu_dT2'][0])
    kappa = complex(bf['kappa'][0])
    dkappa_dT = complex(bf['dkappa_dT'][0])
    d2kappa_dT2 = complex(bf['d2kappa_dT2'][0])
    sigma = complex(bf.get('Pr_local', np.array([0.72]))[0])

    phase = alpha * U - alpha * c
    k2 = alpha**2 + beta**2
    d = float(lambda_mu_ratio)

    A = np.zeros((8, 8), dtype=complex)

    # Appendix A.1
    A[0, 1] = 1.0

    # Appendix A.2
    A[1, 0] = 1j * Re * phase / (mu * T) + k2
    A[1, 1] = -(dmu_dT / mu) * DT
    A[1, 2] = (
        Re * alpha * DU / (mu * T)
        - 1j * k2 * (dmu_dT / mu) * DT
        - 1j * (1.0 + 2.0 * d) * k2 * DT / (3.0 * T)
    )
    A[1, 3] = 1j * Re * k2 / mu - (1.0 + 2.0 * d) * k2 * gamma * Ma**2 * phase / 3.0
    A[1, 4] = (
        (1.0 + 2.0 * d) * k2 * phase / (3.0 * T)
        - (dmu_dT / mu) * alpha * D2U
        - (d2mu_dT2 / mu) * DT * alpha * DU
    )
    A[1, 5] = -(dmu_dT / mu) * alpha * DU

    # Appendix A.3
    A[2, 0] = -1j
    A[2, 2] = DT / T
    A[2, 3] = -1j * gamma * Ma**2 * phase
    A[2, 4] = 1j * phase / T

    # Appendix A.4/A.5
    E = Re / mu + 1j * (2.0 / 3.0) * (2.0 + d) * gamma * Ma**2 * phase
    A[3, 0] = -1j / E * (2.0 * dmu_dT * DT / mu + 2.0 * (2.0 + d) * DT / (3.0 * T))
    A[3, 1] = -1j / E
    # The printed Appendix-A pressure row drops one DT factor in a43 and a
    # 1/T factor in a46. Re-deriving Z4' directly from Eq. 8.9b and the
    # reduced collocation operator recovers the forms used here.
    A[3, 2] = (
        -k2
        + 2.0 * (2.0 + d) * dmu_dT * DT**2 / (3.0 * T * mu)
        + 2.0 * (2.0 + d) * D2T / (3.0 * T)
        - 1j * Re * phase / (mu * T)
    ) / E
    A[3, 3] = (
        -1j * 2.0 * (2.0 + d) * gamma * Ma**2 / (3.0 * E)
        * (phase * dmu_dT * DT / mu + alpha * DU + DT * phase / T)
    )
    A[3, 4] = (
        1j / E
        * (
            dmu_dT * alpha * DU / mu
            + 2.0 * (2.0 + d) / 3.0
            * (dmu_dT * DT * phase / (mu * T) + alpha * DU / T)
        )
    )
    A[3, 5] = 1j * 2.0 * (2.0 + d) * phase / (3.0 * T * E)

    # Appendix A.6
    A[4, 5] = 1.0

    # Appendix A.7
    A[5, 1] = -2.0 * sigma * (gamma - 1.0) * Ma**2 * alpha * DU / k2
    A[5, 2] = Re * sigma * DT / (mu * T) - 1j * 2.0 * sigma * (gamma - 1.0) * Ma**2 * alpha * DU
    A[5, 3] = -1j * Re * sigma * (gamma - 1.0) * Ma**2 * phase / mu
    A[5, 4] = (
        1j * Re * sigma * phase / (mu * T)
        + k2
        - D2T * dkappa_dT / kappa
        - DT**2 * d2kappa_dT2 / kappa
        - sigma * (gamma - 1.0) * Ma**2 * dmu_dT * DU**2 / mu
    )
    A[5, 5] = -2.0 * dkappa_dT * DT / kappa
    if include_spanwise_dissipation_coupling:
        A[5, 7] = (
            float(spanwise_dissipation_coupling_scale)
            * 2.0
            * sigma
            * (gamma - 1.0)
            * Ma**2
            * beta
            * DU
            / k2
        )

    # Appendix A.8/A.9
    A[6, 7] = 1.0
    A[7, 2] = -Re * beta * DU / (mu * T)
    A[7, 4] = dmu_dT * beta * D2U / mu + d2mu_dT2 * DT * beta * DU / mu
    A[7, 5] = dmu_dT * beta * DU / mu
    A[7, 6] = 1j * Re * phase / (mu * T) + k2
    A[7, 7] = -(dmu_dT / mu) * DT

    return A


def mack_first_order_matrix_6(
    baseflow,
    y,
    alpha,
    beta,
    c,
    Re,
    Ma,
    gamma,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
):
    """Return Mack's primary sixth-order first-order matrix.

    The sixth-order approximation drops Appendix-A ``a68`` and solves only the
    first six equations. The passive normal-vorticity pair is not part of the
    eigenvalue determinant.
    """
    A8 = mack_first_order_matrix_3d(
        baseflow,
        y,
        alpha,
        beta,
        c,
        Re,
        Ma,
        gamma,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        include_spanwise_dissipation_coupling=False,
    )
    return A8[:6, :6]


def mack_freestream_decay_basis_6(
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Return the three bounded Appendix-B columns for the primary 6x6 system."""
    basis8, labels8 = mack_freestream_decay_basis(
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        lambda_mu_ratio=lambda_mu_ratio,
    )
    keep = [i for i, label in enumerate(labels8) if label != 'lambda_7']
    return basis8[:6, keep], tuple(labels8[i] for i in keep)


def integrate_bounded_basis_3d(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    rtol=1e-8,
    atol=1e-10,
    method='ivp',
    n_steps=600,
):
    """Integrate the four bounded Appendix-B solutions from ``y_max`` to the wall."""
    sigma = _freestream_sigma(baseflow, Pr, length_scale)
    basis, _ = mack_freestream_decay_basis(
        alpha, beta, c, Re, Ma, sigma, gamma,
        lambda_mu_ratio=lambda_mu_ratio,
    )

    if method == 'qr':
        Y, _ = np.linalg.qr(basis)
        y_grid = np.linspace(float(y_max), 0.0, int(n_steps) + 1)

        for y0, y1 in zip(y_grid[:-1], y_grid[1:]):
            h = y1 - y0
            ym = 0.5 * (y0 + y1)

            A0 = mack_first_order_matrix_3d(
                baseflow, y0, alpha, beta, c, Re, Ma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
                spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
            )
            Am = mack_first_order_matrix_3d(
                baseflow, ym, alpha, beta, c, Re, Ma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
                spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
            )
            A1 = mack_first_order_matrix_3d(
                baseflow, y1, alpha, beta, c, Re, Ma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
                spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
            )

            k1 = A0 @ Y
            k2 = Am @ (Y + 0.5 * h * k1)
            k3 = Am @ (Y + 0.5 * h * k2)
            k4 = A1 @ (Y + h * k3)
            Y = Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Y, _ = np.linalg.qr(Y)

        return Y, {'success': True, 'method': 'qr', 'n_steps': int(n_steps)}

    def rhs(y, flat_state):
        Y = flat_state.reshape(8, 4)
        A = mack_first_order_matrix_3d(
            baseflow, y, alpha, beta, c, Re, Ma, gamma,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
        )
        return (A @ Y).reshape(-1)

    sol = solve_ivp(
        rhs,
        (float(y_max), 0.0),
        basis.reshape(-1),
        method='DOP853',
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    wall_basis = sol.y[:, -1].reshape(8, 4)
    return wall_basis, sol


def integrate_bounded_basis_6(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    rtol=1e-8,
    atol=1e-10,
    method='ivp',
    n_steps=600,
):
    """Integrate the three bounded primary sixth-order solutions to the wall."""
    sigma = _freestream_sigma(baseflow, Pr, length_scale)
    basis, _ = mack_freestream_decay_basis_6(
        alpha,
        beta,
        c,
        Re,
        Ma,
        sigma,
        gamma,
        lambda_mu_ratio=lambda_mu_ratio,
    )

    if method == 'qr':
        Y, _ = np.linalg.qr(basis)
        y_grid = np.linspace(float(y_max), 0.0, int(n_steps) + 1)

        for y0, y1 in zip(y_grid[:-1], y_grid[1:]):
            h = y1 - y0
            ym = 0.5 * (y0 + y1)

            A0 = mack_first_order_matrix_6(
                baseflow, y0, alpha, beta, c, Re, Ma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
            )
            Am = mack_first_order_matrix_6(
                baseflow, ym, alpha, beta, c, Re, Ma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
            )
            A1 = mack_first_order_matrix_6(
                baseflow, y1, alpha, beta, c, Re, Ma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
            )

            k1 = A0 @ Y
            k2 = Am @ (Y + 0.5 * h * k1)
            k3 = Am @ (Y + 0.5 * h * k2)
            k4 = A1 @ (Y + h * k3)
            Y = Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Y, _ = np.linalg.qr(Y)

        return Y, {'success': True, 'method': 'qr', 'n_steps': int(n_steps)}

    def rhs(y, flat_state):
        Y = flat_state.reshape(6, 3)
        A = mack_first_order_matrix_6(
            baseflow, y, alpha, beta, c, Re, Ma, gamma,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
        )
        return (A @ Y).reshape(-1)

    sol = solve_ivp(
        rhs,
        (float(y_max), 0.0),
        basis.reshape(-1),
        method='DOP853',
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    wall_basis = sol.y[:, -1].reshape(6, 3)
    return wall_basis, sol


def temporal_shooting_residual_3d(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
):
    """Return the wall determinant residual for the bounded 3D temporal problem."""
    wall_basis, _ = integrate_bounded_basis_3d(
        baseflow,
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
        spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
        method=method,
        n_steps=n_steps,
    )
    wall_matrix = wall_basis[_wall_condition_rows_3d(wall_bc), :]
    return np.linalg.det(wall_matrix)


def temporal_shooting_residual_6(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
):
    """Return the wall determinant residual for Mack's primary sixth-order system."""
    wall_matrix = temporal_shooting_wall_matrix_6(
        baseflow,
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=n_steps,
    )
    return np.linalg.det(wall_matrix)


def temporal_shooting_wall_matrix_3d(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
):
    """Return the 4x4 wall boundary matrix for the bounded temporal problem."""
    wall_basis, _ = integrate_bounded_basis_3d(
        baseflow,
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
        spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
        method=method,
        n_steps=n_steps,
    )
    return wall_basis[_wall_condition_rows_3d(wall_bc), :]


def temporal_shooting_wall_matrix_6(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
):
    """Return the 3x3 wall boundary matrix for Mack's sixth-order system."""
    wall_basis, _ = integrate_bounded_basis_6(
        baseflow,
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        method=method,
        n_steps=n_steps,
    )
    return wall_basis[_wall_condition_rows_6(wall_bc), :]


def temporal_shooting_sigma_min_3d(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
):
    """Return the smallest singular value of the wall boundary matrix."""
    wall_matrix = temporal_shooting_wall_matrix_3d(
        baseflow,
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
        spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=n_steps,
    )
    return np.linalg.svd(wall_matrix, compute_uv=False)[-1]


def temporal_shooting_sigma_min_6(
    baseflow,
    alpha,
    beta,
    c,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
):
    """Return the smallest singular value of the sixth-order wall matrix."""
    wall_matrix = temporal_shooting_wall_matrix_6(
        baseflow,
        alpha,
        beta,
        c,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=n_steps,
    )
    return np.linalg.svd(wall_matrix, compute_uv=False)[-1]


def _muller_step_complex(z0, z1, z2, f0, f1, f2):
    """One complex Muller's-method step."""
    h0 = z1 - z0
    h1 = z2 - z1
    d0 = (f1 - f0) / h0
    d1 = (f2 - f1) / h1
    a = (d1 - d0) / (h1 + h0)
    b = d1 + a * h1
    disc = np.sqrt(b**2 - 4.0 * a * f2)
    denom1 = b + disc
    denom2 = b - disc
    denom = denom1 if abs(denom1) > abs(denom2) else denom2
    return z2 - 2.0 * f2 / denom


def solve_temporal_mode_3d_shooting(
    baseflow,
    alpha,
    beta,
    c_guess,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    wall_bc='isothermal',
    tol=1e-8,
    max_iter=20,
    method='qr',
    n_steps=600,
):
    """Refine a temporal eigenvalue by Appendix-A/B bounded shooting."""
    z2 = complex(c_guess)
    z1 = z2 + (1e-3 - 2e-3j)
    z0 = z2 - (1e-3 + 2e-3j)

    def residual(c_val):
        return temporal_shooting_residual_3d(
            baseflow,
            alpha,
            beta,
            c_val,
            Re,
            Ma,
            Pr,
            gamma,
            y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=n_steps,
        )

    f0 = residual(z0)
    f1 = residual(z1)
    f2 = residual(z2)

    history = [(z0, f0), (z1, f1), (z2, f2)]

    for _ in range(max_iter):
        if abs(f2) < tol:
            return z2, True, history
        try:
            z_new = _muller_step_complex(z0, z1, z2, f0, f1, f2)
        except Exception:
            z_new = z2 + (1e-4 + 1e-4j)
        f_new = residual(z_new)
        z0, z1, z2 = z1, z2, z_new
        f0, f1, f2 = f1, f2, f_new
        history.append((z2, f2))

    return z2, abs(f2) < 1e-4, history


def solve_temporal_mode_3d_shooting_sigma_min(
    baseflow,
    alpha,
    beta,
    c_guess,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
    c_real_bounds=None,
    c_imag_bounds=None,
    out_of_bounds_penalty=1e6,
):
    """Minimize the wall-matrix smallest singular value over complex c."""

    history = []

    def objective(x):
        c_val = complex(x[0], x[1])
        penalty = 0.0
        if c_real_bounds is not None:
            lo, hi = c_real_bounds
            if c_val.real < lo:
                penalty += float(out_of_bounds_penalty) * (lo - c_val.real + 1.0)
            elif c_val.real > hi:
                penalty += float(out_of_bounds_penalty) * (c_val.real - hi + 1.0)
        if c_imag_bounds is not None:
            lo, hi = c_imag_bounds
            if c_val.imag < lo:
                penalty += float(out_of_bounds_penalty) * (lo - c_val.imag + 1.0)
            elif c_val.imag > hi:
                penalty += float(out_of_bounds_penalty) * (c_val.imag - hi + 1.0)
        if penalty > 0.0:
            history.append((c_val, penalty))
            return penalty

        sigma_min = temporal_shooting_sigma_min_3d(
            baseflow,
            alpha,
            beta,
            c_val,
            Re,
            Ma,
            Pr,
            gamma,
            y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=n_steps,
        )
        history.append((c_val, sigma_min))
        return sigma_min

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
    sigma_min = temporal_shooting_sigma_min_3d(
        baseflow,
        alpha,
        beta,
        c_opt,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
        spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=n_steps,
    )
    return c_opt, sigma_min, bool(result.success), history


def continue_temporal_mode_3d_shooting_sigma_min(
    case_sequence,
    baseflow_builder=None,
    *,
    Pr=0.72,
    gamma=1.4,
    initial_c,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    include_spanwise_dissipation_coupling=True,
    spanwise_dissipation_coupling_scale=1.0,
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
    polish_with_determinant=True,
):
    """Track one exact first-order temporal root across a sequence of cases."""
    tracked = []
    c_seed = complex(initial_c)

    for case in case_sequence:
        case_data = dict(case)
        alpha = float(case_data['alpha'])
        beta = float(case_data['beta'])
        Re = float(case_data['Re'])
        Ma = float(case_data['Ma'])
        Pr_case = float(case_data.get('Pr', Pr))
        gamma_case = float(case_data.get('gamma', gamma))
        y_max = float(case_data['y_max'])
        n_steps_case = int(case_data.get('n_steps', n_steps))

        baseflow = case_data.get('baseflow')
        if baseflow is None:
            if baseflow_builder is None:
                raise ValueError('baseflow_builder is required when case has no baseflow')
            baseflow = baseflow_builder(case_data)

        c_opt, sigma_min, converged, history = solve_temporal_mode_3d_shooting_sigma_min(
            baseflow,
            alpha,
            beta,
            c_seed,
            Re,
            Ma,
            Pr_case,
            gamma_case,
            y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=n_steps_case,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )

        c_final = c_opt
        det_converged = False
        det_history = []
        if polish_with_determinant:
            c_final, det_converged, det_history = solve_temporal_mode_3d_shooting(
                baseflow,
                alpha,
                beta,
                c_opt,
                Re,
                Ma,
                Pr_case,
                gamma_case,
                y_max,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
                spanwise_dissipation_coupling_scale=spanwise_dissipation_coupling_scale,
                wall_bc=wall_bc,
                method=method,
                n_steps=max(n_steps_case, 800),
            )

        tracked.append({
            'case': case_data,
            'c_sigma_min': c_opt,
            'c_final': c_final,
            'omega_i': alpha * c_final.imag,
            'sigma_min': sigma_min,
            'sigma_min_converged': converged,
            'determinant_converged': det_converged,
            'sigma_min_history': history,
            'determinant_history': det_history,
        })
        c_seed = c_final

    return tracked


def solve_temporal_mode_6_shooting(
    baseflow,
    alpha,
    beta,
    c_guess,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    tol=1e-8,
    max_iter=20,
    method='qr',
    n_steps=600,
):
    """Refine a temporal eigenvalue with Mack's primary sixth-order system."""
    z2 = complex(c_guess)
    z1 = z2 + (1e-3 - 2e-3j)
    z0 = z2 - (1e-3 + 2e-3j)

    def residual(c_val):
        return temporal_shooting_residual_6(
            baseflow,
            alpha,
            beta,
            c_val,
            Re,
            Ma,
            Pr,
            gamma,
            y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=n_steps,
        )

    f0 = residual(z0)
    f1 = residual(z1)
    f2 = residual(z2)

    history = [(z0, f0), (z1, f1), (z2, f2)]

    for _ in range(max_iter):
        if abs(f2) < tol:
            return z2, True, history
        try:
            z_new = _muller_step_complex(z0, z1, z2, f0, f1, f2)
        except Exception:
            z_new = z2 + (1e-4 + 1e-4j)
        f_new = residual(z_new)
        z0, z1, z2 = z1, z2, z_new
        f0, f1, f2 = f1, f2, f_new
        history.append((z2, f2))

    return z2, abs(f2) < 1e-4, history


def solve_temporal_mode_6_shooting_sigma_min(
    baseflow,
    alpha,
    beta,
    c_guess,
    Re,
    Ma,
    Pr,
    gamma,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
):
    """Minimize the primary sixth-order wall-matrix singular value."""

    history = []

    def objective(x):
        c_val = complex(x[0], x[1])
        sigma_min = temporal_shooting_sigma_min_6(
            baseflow,
            alpha,
            beta,
            c_val,
            Re,
            Ma,
            Pr,
            gamma,
            y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=n_steps,
        )
        history.append((c_val, sigma_min))
        return sigma_min

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
    sigma_min = temporal_shooting_sigma_min_6(
        baseflow,
        alpha,
        beta,
        c_opt,
        Re,
        Ma,
        Pr,
        gamma,
        y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=n_steps,
    )
    return c_opt, sigma_min, bool(result.success), history


def continue_temporal_mode_6_shooting_sigma_min(
    case_sequence,
    baseflow_builder=None,
    *,
    Pr=0.72,
    gamma=1.4,
    initial_c,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
    polish_with_determinant=True,
):
    """Track one temporal root with Mack's primary sixth-order system."""
    tracked = []
    c_seed = complex(initial_c)

    for case in case_sequence:
        case_data = dict(case)
        alpha = float(case_data['alpha'])
        beta = float(case_data['beta'])
        Re = float(case_data['Re'])
        Ma = float(case_data['Ma'])
        Pr_case = float(case_data.get('Pr', Pr))
        gamma_case = float(case_data.get('gamma', gamma))
        y_max = float(case_data['y_max'])
        n_steps_case = int(case_data.get('n_steps', n_steps))

        baseflow = case_data.get('baseflow')
        if baseflow is None:
            if baseflow_builder is None:
                raise ValueError('baseflow_builder is required when case has no baseflow')
            baseflow = baseflow_builder(case_data)

        c_opt, sigma_min, converged, history = solve_temporal_mode_6_shooting_sigma_min(
            baseflow,
            alpha,
            beta,
            c_seed,
            Re,
            Ma,
            Pr_case,
            gamma_case,
            y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=n_steps_case,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )

        c_final = c_opt
        det_converged = False
        det_history = []
        if polish_with_determinant:
            c_final, det_converged, det_history = solve_temporal_mode_6_shooting(
                baseflow,
                alpha,
                beta,
                c_opt,
                Re,
                Ma,
                Pr_case,
                gamma_case,
                y_max,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                wall_bc=wall_bc,
                method=method,
                n_steps=max(n_steps_case, 800),
            )

        tracked.append({
            'case': case_data,
            'c_sigma_min': c_opt,
            'c_final': c_final,
            'omega_i': alpha * c_final.imag,
            'sigma_min': sigma_min,
            'sigma_min_converged': converged,
            'determinant_converged': det_converged,
            'sigma_min_history': history,
            'determinant_history': det_history,
        })
        c_seed = c_final

    return tracked
