"""
Eigenvalue solver with mode filtering and tracking.

Temporal solver for Orr-Sommerfeld (incompressible validation)
and spatial solver for the compressible stability equations.
"""

import numpy as np
from scipy import linalg

from .asymptotic import (
    mack_freestream_decay_basis,
    mack_freestream_subspace_residual,
)
from .spectral import chebyshev_points, chebyshev_D, physical_derivatives
from .scales import delta_star_over_lstar, rescale_baseflow_derivatives, sample_baseflow
from .equations import (
    DEFAULT_LAMBDA_MU_RATIO,
    assemble_orr_sommerfeld,
    assemble_compressible_matrices,
    momentum_viscous_coefficients,
    transport_conductivity_data,
    transport_temperature_derivatives,
)


def _scaled_compressible_problem(baseflow, y, D1, D2, length_scale):
    """Sample the base flow and rescale derivatives for the requested length."""
    return sample_baseflow(baseflow, y, length_scale), D1, D2


def _top_first_order_state_3d(mode, D1, n, alpha, beta, Ma, gamma, boundary_index=0):
    """Convert a collocation eigenvector into Mack's first-order top state."""
    k = float(np.hypot(alpha, beta))
    if k <= 0.0:
        raise ValueError('At least one of alpha or beta must be non-zero')

    q = mode[0:n]
    v = mode[n:2*n]
    s = mode[2*n:3*n]
    T = mode[3*n:4*n]
    p = mode[4*n:5*n]

    q_y = D1[boundary_index, :] @ q
    T_y = D1[boundary_index, :] @ T
    s_y = D1[boundary_index, :] @ s

    return np.array(
        [
            k * q[boundary_index],
            k * q_y,
            v[boundary_index],
            p[boundary_index],
            T[boundary_index],
            T_y,
            k * s[boundary_index],
            k * s_y,
        ],
        dtype=complex,
    )


def temporal_freestream_leakage_3d(
    eigenvalues,
    eigenvectors,
    D1,
    alpha,
    beta,
    Re,
    Ma,
    Pr,
    gamma,
    Pr_freestream=None,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    boundary_index=0,
):
    """Measure how far each mode is from Mack's bounded freestream subspace."""
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.size == 0:
        return np.array([], dtype=float)

    n = D1.shape[0]
    scores = np.empty(eigenvalues.shape[0], dtype=float)
    sigma = Pr if Pr_freestream is None else Pr_freestream

    for j, c in enumerate(eigenvalues):
        try:
            basis, _ = mack_freestream_decay_basis(
                alpha, beta, c, Re, Ma, sigma, gamma,
                lambda_mu_ratio=lambda_mu_ratio,
            )
            top_state = _top_first_order_state_3d(
                eigenvectors[:, j], D1, n, alpha, beta, Ma, gamma,
                boundary_index=boundary_index,
            )
            scores[j] = mack_freestream_subspace_residual(top_state, basis)
        except Exception:
            scores[j] = np.inf

    return scores


def temperature_wall_operator(D1, n, wall_bc):
    """Return the wall operator for the thermal boundary condition."""
    wall = n - 1
    row = np.zeros(n, dtype=complex)
    if wall_bc == 'isothermal':
        row[wall] = 1.0
        return row
    if wall_bc == 'adiabatic':
        row[:] = D1[wall, :]
        return row
    raise ValueError("wall_bc must be 'isothermal' or 'adiabatic'")


def _assemble_spatial_qep(
    baseflow,
    omega,
    Re,
    Ma,
    Pr,
    gamma,
    N,
    y_max,
    L,
    wall_bc,
    length_scale,
    lambda_mu_ratio,
):
    """Assemble the spatial quadratic EVP after boundary conditions.

    The spatial problem is
        (C0 + alpha*C1 + alpha**2*C2) phi = 0.
    This helper is shared by shift-invert and full-spectrum paths so branch
    discovery changes do not silently change the operator.
    """
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)
    bf, D1, D2 = _scaled_compressible_problem(baseflow, y, D1, D2, length_scale)

    C0, C1, C2 = assemble_compressible_matrices(
        D1, D2, y, bf, omega, Re, Ma, Pr, gamma,
        lambda_mu_ratio=lambda_mu_ratio)

    n = len(y)

    # Apply BCs: no-slip plus either isothermal or adiabatic thermal wall BC.
    wall = n - 1
    free = 0
    for var in range(2):
        for loc in [wall, free]:
            row = var * n + loc
            C0[row, :] = 0
            C1[row, :] = 0
            C2[row, :] = 0
            C0[row, row] = 1.0

    temp_slice = slice(2 * n, 3 * n)
    temp_wall_row = 2 * n + wall
    temp_free_row = 2 * n + free
    C0[temp_wall_row, :] = 0
    C1[temp_wall_row, :] = 0
    C2[temp_wall_row, :] = 0
    C0[temp_wall_row, temp_slice] = temperature_wall_operator(D1, n, wall_bc)
    C0[temp_free_row, :] = 0
    C1[temp_free_row, :] = 0
    C2[temp_free_row, :] = 0
    C0[temp_free_row, temp_free_row] = 1.0

    return C0, C1, C2, y


def _spatial_companion_matrices(C0, C1, C2):
    """Return companion matrices for the spatial quadratic EVP."""
    nn = C0.shape[0]
    LL = np.zeros((2 * nn, 2 * nn), dtype=complex)
    RR = np.zeros((2 * nn, 2 * nn), dtype=complex)

    LL[:nn, :nn] = -C1
    LL[:nn, nn:] = -C0
    LL[nn:, :nn] = np.eye(nn)

    RR[:nn, :nn] = C2
    RR[nn:, nn:] = np.eye(nn)
    return LL, RR


def solve_temporal_os(baseflow, alpha, Re, N=128, y_max=40.0, L=None):
    """Solve the temporal Orr-Sommerfeld problem."""
    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)

    bf = baseflow(y)
    build_evp = assemble_orr_sommerfeld(D1, D2, y, bf, Re)
    A, B = build_evp(alpha)

    eigenvalues, eigenvectors = linalg.eig(A, B)

    valid = np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    phys = ((np.abs(eigenvalues.real) < 1.5) &
            (np.abs(eigenvalues.imag) < 1.0))
    eigenvalues = eigenvalues[phys]
    eigenvectors = eigenvectors[:, phys]

    idx = np.argsort(-eigenvalues.imag)
    return eigenvalues[idx], eigenvectors[:, idx], y


def solve_spatial(baseflow, omega, Re, Ma, Pr, gamma, N=128,
                  y_max=None, L=None, wall_bc='isothermal',
                  target_alpha=None, n_modes=20,
                  length_scale='delta_star',
                  lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Solve the spatial compressible stability problem.

    Uses shift-invert companion linearization to find eigenvalues
    near a target in the complex alpha plane.

    Parameters
    ----------
    target_alpha : complex, optional
        Shift for shift-invert. If None, uses omega/c_phase with c_phase
        estimated from the Mach number.
    n_modes : int
        Number of eigenvalues to return near the target.
    """
    # Estimate target if not given
    if target_alpha is None:
        # Second mode phase speed: c ~ 1 - 1/Ma for Ma > 3
        # First mode: c ~ 0.4
        if Ma > 3:
            c_2nd = 1.0 - 1.0 / Ma
            target_alpha = omega / c_2nd + 0j
        else:
            target_alpha = omega / 0.4 + 0j

    C0, C1, C2, y = _assemble_spatial_qep(
        baseflow, omega, Re, Ma, Pr, gamma, N, y_max, L, wall_bc,
        length_scale, lambda_mu_ratio,
    )
    nn = C0.shape[0]
    LL, RR = _spatial_companion_matrices(C0, C1, C2)

    # Shift-invert: (LL - sigma*RR)^{-1} @ RR has eigenvalues 1/(alpha - sigma)
    sigma = target_alpha
    A_shift = LL - sigma * RR

    try:
        # LU factorize and solve
        lu, piv = linalg.lu_factor(A_shift)
        B_inv = linalg.lu_solve((lu, piv), RR)

        # Eigenvalues of B_inv are mu = 1/(alpha - sigma)
        mu_vals, mu_vecs = linalg.eig(B_inv)

        # Convert back: alpha = sigma + 1/mu
        valid = np.abs(mu_vals) > 1e-15
        mu_vals = mu_vals[valid]
        mu_vecs = mu_vecs[:, valid]

        alphas = sigma + 1.0 / mu_vals

        # Extract phi (lower half of companion eigenvector)
        phi_all = mu_vecs[nn:, :]

    except linalg.LinAlgError:
        # Fallback: direct solve without shift-invert
        eigenvalues, eigenvectors = linalg.eig(LL, RR)
        valid = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 200)
        alphas = eigenvalues[valid]
        phi_all = eigenvectors[nn:, valid]

    # Filter physical modes
    phys = ((alphas.real > 1e-3) & (alphas.real < 100) &
            (np.abs(alphas.imag) < 5.0))
    alphas = alphas[phys]
    phi_all = phi_all[:, phys]

    # Sort by distance to target
    dist = np.abs(alphas - target_alpha)
    idx = np.argsort(dist)[:min(n_modes, len(alphas))]
    alphas = alphas[idx]
    modes = phi_all[:, idx]

    # Re-sort by alpha_i (most unstable first)
    idx2 = np.argsort(alphas.imag)
    return alphas[idx2], modes[:, idx2], y


def solve_spatial_full_spectrum(
    baseflow,
    omega,
    Re,
    Ma,
    Pr,
    gamma,
    N=64,
    y_max=None,
    L=None,
    wall_bc='isothermal',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    max_abs_alpha=100.0,
    max_abs_alpha_i=5.0,
    residual_tol=None,
):
    """Solve the full spatial companion spectrum for branch discovery.

    Unlike :func:`solve_spatial`, this does not shift-invert around a target.
    It returns every physical-looking alpha from the companion QEP, which is
    useful when a branch tracker must not miss the Mack/S root because the
    local shift landed on a nearby acoustic or spurious branch.
    """
    C0, C1, C2, y = _assemble_spatial_qep(
        baseflow, omega, Re, Ma, Pr, gamma, N, y_max, L, wall_bc,
        length_scale, lambda_mu_ratio,
    )
    nn = C0.shape[0]
    LL, RR = _spatial_companion_matrices(C0, C1, C2)

    eigenvalues, eigenvectors = linalg.eig(
        LL,
        RR,
        overwrite_a=True,
        overwrite_b=True,
        check_finite=False,
    )
    valid = (
        np.isfinite(eigenvalues)
        & (eigenvalues.real > 1.0e-8)
        & (np.abs(eigenvalues) < max_abs_alpha)
        & (np.abs(eigenvalues.imag) < max_abs_alpha_i)
    )
    alphas = eigenvalues[valid]
    modes = eigenvectors[nn:, valid]

    if residual_tol is not None:
        alphas, modes = _filter_with_residual(
            alphas,
            modes,
            C0,
            C1,
            C2,
            nn,
            omega,
            Re,
            Ma,
            tol=float(residual_tol),
        )

    idx = np.argsort(alphas.imag)
    return alphas[idx], modes[:, idx], y


def _filter_with_residual(eigenvalues, phi_all, C0, C1, C2, nn,
                          omega, Re, Ma, tol=1e-4):
    """Filter eigenvalues using QEP residual.

    For each eigenvalue alpha and eigenvector phi, compute:
        r = ||(C0 + alpha*C1 + alpha^2*C2) * phi|| / ||phi||
    Physical modes have small residuals; spurious modes have large ones.
    """
    valid = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 200)
    idx_valid = np.where(valid)[0]

    good_alphas = []
    good_modes = []

    for i in idx_valid:
        alpha = eigenvalues[i]
        phi = phi_all[:, i]

        # Basic bounds: downstream-propagating, reasonable range
        if alpha.real < 1e-3 or alpha.real > 100:
            continue
        if np.abs(alpha.imag) > 5.0:
            continue

        phi_norm = np.linalg.norm(phi)
        if phi_norm < 1e-30:
            continue

        # Compute QEP residual
        Lphi = (C0 + alpha * C1 + alpha**2 * C2) @ phi
        residual = np.linalg.norm(Lphi) / phi_norm

        # Normalize residual by matrix scale
        mat_scale = (np.linalg.norm(C0, 'fro') +
                     np.abs(alpha) * np.linalg.norm(C1, 'fro') +
                     np.abs(alpha)**2 * np.linalg.norm(C2, 'fro')) / nn
        rel_residual = residual / max(mat_scale, 1e-30)

        if rel_residual < tol:
            good_alphas.append(alpha)
            good_modes.append(phi)

    if len(good_alphas) == 0:
        return np.array([]), np.array([]).reshape(nn, 0)

    alphas = np.array(good_alphas)
    modes = np.column_stack(good_modes)

    # Sort by alpha_i ascending (most unstable first)
    idx = np.argsort(alphas.imag)
    return alphas[idx], modes[:, idx]


def solve_temporal_compressible(baseflow, alpha, Re, Ma, Pr, gamma,
                                N=128, y_max=None, L=None,
                                wall_bc='isothermal',
                                length_scale='delta_star',
                                lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Solve the temporal compressible stability problem: A*phi = c*B*phi.

    Given real alpha, find complex c = omega/alpha.
    This is a LINEAR generalized EVP (no companion needed).

    Parameters
    ----------
    baseflow : callable
        Profile object.
    alpha : float
        Real streamwise wavenumber.
    Re, Ma, Pr, gamma : float
        Flow parameters.
    N : int
        Chebyshev intervals.
    y_max : float, optional
        Domain height.

    Returns
    -------
    c : array
        Complex phase speeds (filtered, sorted by c_i descending).
    modes : array
        Eigenvectors (4n columns).
    y : array
        Grid points.
    """
    from .equations import assemble_compressible_matrices

    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)
    bf, D1, D2 = _scaled_compressible_problem(baseflow, y, D1, D2, length_scale)

    A, B = _assemble_temporal_2d_evp(
        bf, y, D1, D2, alpha, Re, Ma, Pr, gamma,
        wall_bc=wall_bc, lambda_mu_ratio=lambda_mu_ratio,
    )

    # Solve A*phi = c*B*phi
    eigenvalues, eigenvectors = linalg.eig(A, B)

    # Filter
    valid = np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    phys = ((eigenvalues.real > -0.5) & (eigenvalues.real < 1.5) &
            (np.abs(eigenvalues.imag) < 0.5))
    eigenvalues = eigenvalues[phys]
    eigenvectors = eigenvectors[:, phys]

    idx = np.argsort(-eigenvalues.imag)
    return eigenvalues[idx], eigenvectors[:, idx], y


def _assemble_temporal_2d_evp(bf, y, D1, D2, alpha, Re, Ma, Pr, gamma,
                              wall_bc='isothermal',
                              lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Assemble the 2D temporal compressible EVP operators (Mack enthalpy form).

    Assembly stage of :func:`solve_temporal_compressible`, extracted unchanged
    so operator-probing callers can obtain the generalized eigenvalue problem
    ``A*phi = c*B*phi`` from the discretized inputs (sampled base flow ``bf``,
    grid ``y``, physical derivative matrices ``D1``/``D2``).  Returns the pair
    ``(A, B)`` with boundary-condition rows applied, exactly as handed to the
    QZ solver.
    """
    n = len(y)
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Mean flow
    Ub = np.diag(bf['U'])
    dUb = np.diag(bf['dU'])
    d2Ub = np.diag(bf['d2U'])
    Tb = np.diag(bf['T'])
    dTb_diag = np.diag(bf['dT'])
    d2Tb_diag = np.diag(bf['d2T'])
    mub = np.diag(bf['mu'])
    dmub = np.diag(bf['dmu'])

    T_v = bf['T']
    rho_v = bf['rho']
    mu_v = bf['mu']
    dT_v = bf['dT']
    dU_v = bf['dU']

    rhoI = np.diag(1.0 / rho_v)
    TI = np.diag(1.0 / T_v)

    # Transport derivatives come directly from the mean-flow model when
    # available; this keeps the stability coefficients consistent with the
    # viscosity law used to build the profile.
    dmu_dT_v, d2mu_dT2_v = transport_temperature_derivatives(bf)
    dmu_dT = np.diag(dmu_dT_v)
    d2mu_dT2 = np.diag(d2mu_dT2_v)
    (
        x_alpha2_coeff,
        cross_grad_coeff,
        y_laplacian_coeff,
        y_u_algebraic_coeff,
    ) = momentum_viscous_coefficients(lambda_mu_ratio)

    kappa_v, dkappa_v, dkappa_dT_v, d2kappa_dT2_v, needs_pr_prefactor = (
        transport_conductivity_data(bf, Pr)
    )
    kappab = np.diag(kappa_v)
    dkappab = np.diag(dkappa_v)
    dkappa_dT = np.diag(dkappa_dT_v)
    d2kappa_dT2 = np.diag(d2kappa_dT2_v)

    gm1 = gamma - 1.0
    Ma2 = Ma**2
    ia = 1j * alpha
    a2 = alpha**2

    visc = rhoI / Re
    cond = rhoI / Re if not needs_pr_prefactor else rhoI / (Pr * Re)
    diss = gm1 * Ma2 * rhoI / Re

    def blk(i, j):
        return (slice(i*n, (i+1)*n), slice(j*n, (j+1)*n))

    nn = 4 * n
    A = np.zeros((nn, nn), dtype=complex)
    B = np.zeros((nn, nn), dtype=complex)

    # ====== A matrix (terms without c) ======

    # Continuity: i*alpha*u + Dv - (DT/T)*v + i*alpha*U*(gMa^2*p - T/T) = A part
    A[blk(0, 0)] = ia * I
    A[blk(0, 1)] = D1 - TI @ dTb_diag
    A[blk(0, 2)] = -ia * Ub @ TI
    A[blk(0, 3)] = ia * gamma * Ma2 * Ub

    # x-mom: i*alpha*U*u + DU*v + i*alpha*p/rho - visc*[viscous terms]
    A[blk(1, 0)] = (ia * Ub
                     - visc @ (mub @ D2 + dmub @ D1)
                     + x_alpha2_coeff * a2 * visc @ mub)
    A[blk(1, 1)] = dUb - ia * visc @ (cross_grad_coeff * mub @ D1 + dmub)
    A[blk(1, 2)] = -visc @ (dmu_dT @ d2Ub + d2mu_dT2 @ dUb @ dTb_diag
                             + dmu_dT @ dUb @ D1)
    A[blk(1, 3)] = ia * rhoI

    # y-mom: i*alpha*U*v + Dp/rho - visc*[viscous terms]
    A[blk(2, 0)] = -ia * visc @ (
        cross_grad_coeff * mub @ D1 + y_u_algebraic_coeff * dmub
    )
    A[blk(2, 1)] = (ia * Ub
                     - visc @ (
                         y_laplacian_coeff * mub @ D2
                         + y_laplacian_coeff * dmub @ D1
                     )
                     + a2 * visc @ mub)
    A[blk(2, 2)] = -ia * visc @ (dmu_dT @ dUb)
    A[blk(2, 3)] = rhoI @ D1

    # Energy (Form 1): i*alpha*U*T + DT*v - (gm1)*Ma^2*i*alpha*U*p/rho
    #   - cond*[kappa*(D^2 - alpha^2)T + ...] - diss*[2*mu*DU*(Du + i*alpha*v)]
    A[blk(3, 0)] = -diss @ (2.0 * mub @ np.diag(dU_v) @ D1)
    A[blk(3, 1)] = (np.diag(dT_v)
                     - 2j * alpha * diss @ (mub @ np.diag(dU_v)))
    A[blk(3, 2)] = (ia * Ub
                     - cond @ (kappab @ D2 + 2.0 * dkappab @ D1
                               + dkappa_dT @ d2Tb_diag
                               + d2kappa_dT2 @ dTb_diag @ dTb_diag)
                     - diss @ (dmu_dT @ dUb @ dUb)
                     + a2 * cond @ kappab)
    A[blk(3, 3)] = -ia * gm1 * Ma2 * Ub @ rhoI

    # ====== B matrix (coefficient of c, from -c*B part) ======

    # Continuity: i*alpha*(U-c)*(gamma*Ma^2*p - T/T)
    B[blk(0, 2)] = -ia * TI
    B[blk(0, 3)] = ia * gamma * Ma2 * I

    # x-mom: c*[i*alpha*u] → B[xmom,u] = i*alpha*I
    B[blk(1, 0)] = ia * I

    # y-mom: c*[i*alpha*v] → B[ymom,v] = i*alpha*I
    B[blk(2, 1)] = ia * I

    # Energy: c*[i*alpha*T - (gm1)*Ma^2*i*alpha*p/rho]
    B[blk(3, 2)] = ia * I
    B[blk(3, 3)] = -ia * gm1 * Ma2 * rhoI

    # ====== Boundary conditions ======
    wall = n - 1
    free = 0
    for var in range(2):  # u, v
        for loc in [wall, free]:
            row = var * n + loc
            A[row, :] = 0
            B[row, :] = 0
            A[row, row] = 1.0

    temp_slice = slice(2 * n, 3 * n)
    temp_wall_row = 2 * n + wall
    temp_free_row = 2 * n + free
    A[temp_wall_row, :] = 0
    B[temp_wall_row, :] = 0
    A[temp_wall_row, temp_slice] = temperature_wall_operator(D1, n, wall_bc)
    A[temp_free_row, :] = 0
    B[temp_free_row, :] = 0
    A[temp_free_row, temp_free_row] = 1.0

    return A, B


def assemble_temporal_compressible_3d_evp(
    baseflow,
    alpha,
    beta,
    Re,
    Ma,
    Pr,
    gamma,
    N=128,
    y_max=None,
    L=None,
    include_spanwise_dissipation_coupling=True,
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Assemble the 3D temporal generalized EVP before boundary conditions."""
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    alpha = float(alpha)
    beta = float(beta)
    k = float(np.hypot(alpha, beta))
    if k <= 0.0:
        raise ValueError('At least one of alpha or beta must be non-zero')

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)
    bf, D1, D2 = _scaled_compressible_problem(baseflow, y, D1, D2, length_scale)

    n = len(y)
    I = np.eye(n)

    Ub = np.diag(bf['U'])
    dUb = np.diag(bf['dU'])
    d2Ub = np.diag(bf['d2U'])
    dTb_diag = np.diag(bf['dT'])
    d2Tb_diag = np.diag(bf['d2T'])
    mub = np.diag(bf['mu'])
    dmub = np.diag(bf['dmu'])

    T_v = bf['T']
    rho_v = bf['rho']
    dT_v = bf['dT']
    dU_v = bf['dU']

    rhoI = np.diag(1.0 / rho_v)
    TI = np.diag(1.0 / T_v)

    dmu_dT_v, d2mu_dT2_v = transport_temperature_derivatives(bf)
    dmu_dT = np.diag(dmu_dT_v)
    d2mu_dT2 = np.diag(d2mu_dT2_v)
    (
        x_alpha2_coeff,
        cross_grad_coeff,
        y_laplacian_coeff,
        y_u_algebraic_coeff,
    ) = momentum_viscous_coefficients(lambda_mu_ratio)

    kappa_v, dkappa_v, dkappa_dT_v, d2kappa_dT2_v, needs_pr_prefactor = (
        transport_conductivity_data(bf, Pr)
    )
    kappab = np.diag(kappa_v)
    dkappab = np.diag(dkappa_v)
    dkappa_dT = np.diag(dkappa_dT_v)
    d2kappa_dT2 = np.diag(d2kappa_dT2_v)

    gm1 = gamma - 1.0
    Ma2 = Ma**2
    ia = 1j * alpha
    ik = 1j * k
    k2 = k**2

    alpha_over_k = alpha / k
    beta_over_k = beta / k

    visc = rhoI / Re
    cond = rhoI / Re if not needs_pr_prefactor else rhoI / (Pr * Re)
    diss = gm1 * Ma2 * rhoI / Re

    def blk(i, j):
        return (slice(i*n, (i+1)*n), slice(j*n, (j+1)*n))

    nn = 5 * n
    A = np.zeros((nn, nn), dtype=complex)
    B = np.zeros((nn, nn), dtype=complex)

    # Continuity
    A[blk(0, 0)] = ik * I
    A[blk(0, 1)] = D1 - TI @ dTb_diag
    A[blk(0, 3)] = -ia * Ub @ TI
    A[blk(0, 4)] = ia * gamma * Ma2 * Ub

    B[blk(0, 3)] = -ia * TI
    B[blk(0, 4)] = ia * gamma * Ma2 * I

    # Momentum along the wave vector (q equation)
    A[blk(1, 0)] = (
        ia * Ub
        - visc @ (mub @ D2 + dmub @ D1)
        + x_alpha2_coeff * k2 * visc @ mub
    )
    A[blk(1, 1)] = alpha_over_k * dUb - ik * visc @ (
        cross_grad_coeff * mub @ D1 + dmub
    )
    A[blk(1, 3)] = -visc @ (
        alpha_over_k * dmu_dT @ d2Ub
        + alpha_over_k * d2mu_dT2 @ dUb @ dTb_diag
        + alpha_over_k * dmu_dT @ dUb @ D1
    )
    A[blk(1, 4)] = ik * rhoI

    B[blk(1, 0)] = ia * I

    # Wall-normal momentum
    A[blk(2, 0)] = -ik * visc @ (
        cross_grad_coeff * mub @ D1 + y_u_algebraic_coeff * dmub
    )
    A[blk(2, 1)] = (
        ia * Ub
        - visc @ (
            y_laplacian_coeff * mub @ D2
            + y_laplacian_coeff * dmub @ D1
        )
        + k2 * visc @ mub
    )
    A[blk(2, 3)] = -ia * visc @ (dmu_dT @ dUb)
    A[blk(2, 4)] = rhoI @ D1

    B[blk(2, 1)] = ia * I

    # Momentum normal to the wave vector (s equation)
    A[blk(3, 1)] = -beta_over_k * dUb
    A[blk(3, 2)] = (
        ia * Ub
        - visc @ (mub @ D2 + dmub @ D1)
        - k2 * visc @ mub
    )
    A[blk(3, 3)] = visc @ (
        beta_over_k * dmu_dT @ d2Ub
        + beta_over_k * d2mu_dT2 @ dUb @ dTb_diag
        + beta_over_k * dmu_dT @ dUb @ D1
    )

    B[blk(3, 2)] = ia * I

    # Energy
    A[blk(4, 0)] = -diss @ (
        2.0 * alpha_over_k * mub @ np.diag(dU_v) @ D1
    )
    A[blk(4, 1)] = (
        np.diag(dT_v)
        - 2j * alpha * diss @ (mub @ np.diag(dU_v))
    )
    if include_spanwise_dissipation_coupling:
        A[blk(4, 2)] = diss @ (
            2.0 * beta_over_k * mub @ np.diag(dU_v) @ D1
        )
    A[blk(4, 3)] = (
        ia * Ub
        - cond @ (
            kappab @ D2 + 2.0 * dkappab @ D1
            + dkappa_dT @ d2Tb_diag
            + d2kappa_dT2 @ dTb_diag @ dTb_diag
        )
        - diss @ (dmu_dT @ np.diag(dU_v) @ np.diag(dU_v))
        + k2 * cond @ kappab
    )
    A[blk(4, 4)] = -ia * gm1 * Ma2 * Ub @ rhoI

    B[blk(4, 3)] = ia * I
    B[blk(4, 4)] = -ia * gm1 * Ma2 * rhoI

    return A, B, y, D1, n, alpha, beta, bf


def apply_wall_bc_3d(A, B, D1, n, wall_bc='isothermal'):
    """Apply no-slip plus thermal wall conditions to the 3D temporal EVP."""
    wall = n - 1
    for var in range(3):
        row = var * n + wall
        A[row, :] = 0
        B[row, :] = 0
        A[row, row] = 1.0

    temp_row = 3 * n + wall
    temp_slice = slice(3 * n, 4 * n)
    A[temp_row, :] = 0
    B[temp_row, :] = 0
    A[temp_row, temp_slice] = temperature_wall_operator(D1, n, wall_bc)


def apply_dirichlet_freestream_bc_3d(A, B, n):
    """Apply the crude finite-domain freestream truncation used historically."""
    free = 0
    for var in range(4):
        row = var * n + free
        A[row, :] = 0
        B[row, :] = 0
        A[row, row] = 1.0


def _apply_asymptotic_freestream_bc_3d(
    A,
    B,
    D1,
    n,
    alpha,
    beta,
    Re,
    Ma,
    Pr,
    gamma,
    c_ref,
    Pr_freestream=None,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Apply Mack Appendix-B boundedness conditions frozen at ``c_ref``."""
    sigma = Pr if Pr_freestream is None else Pr_freestream
    basis, _ = mack_freestream_decay_basis(
        alpha, beta, c_ref, Re, Ma, sigma, gamma,
        lambda_mu_ratio=lambda_mu_ratio,
    )
    left_null = linalg.null_space(basis.conj().T)
    if left_null.shape[1] != 4:
        raise ValueError(
            f'expected four freestream BC relations, found {left_null.shape[1]}'
        )

    k = float(np.hypot(alpha, beta))
    nn = 5 * n
    mapping = np.zeros((8, nn), dtype=complex)
    free = 0
    unit_top = np.zeros(n, dtype=complex)
    unit_top[free] = 1.0

    q_slice = slice(0, n)
    v_slice = slice(n, 2 * n)
    s_slice = slice(2 * n, 3 * n)
    T_slice = slice(3 * n, 4 * n)
    p_slice = slice(4 * n, 5 * n)

    mapping[0, q_slice] = k * unit_top
    mapping[1, q_slice] = k * D1[free, :]
    mapping[2, v_slice] = unit_top
    mapping[3, p_slice] = unit_top
    mapping[4, T_slice] = unit_top
    mapping[5, T_slice] = D1[free, :]
    mapping[6, s_slice] = k * unit_top
    mapping[7, s_slice] = k * D1[free, :]

    free_rows = [var * n + free for var in range(4)]
    for bc_index, row in enumerate(free_rows):
        bc_row = left_null[:, bc_index].conj().T @ mapping
        row_norm = linalg.norm(bc_row)
        if row_norm > 0.0:
            bc_row = bc_row / row_norm
        A[row, :] = 0
        B[row, :] = 0
        A[row, :] = bc_row


def _filter_temporal_modes_3d(
    eigenvalues,
    eigenvectors,
    D1,
    alpha,
    beta,
    Re,
    Ma,
    Pr,
    gamma,
    Pr_freestream=None,
    freestream_leakage_tol=None,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Apply basic physical filters and optional Appendix-B leakage filter."""
    valid = np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    phys = (
        (eigenvalues.real > -0.5) & (eigenvalues.real < 1.5)
        & (np.abs(eigenvalues.imag) < 0.5)
    )
    eigenvalues = eigenvalues[phys]
    eigenvectors = eigenvectors[:, phys]

    leakage = temporal_freestream_leakage_3d(
        eigenvalues, eigenvectors, D1, alpha, beta, Re, Ma, Pr, gamma,
        Pr_freestream=Pr_freestream,
        lambda_mu_ratio=lambda_mu_ratio,
    )
    if freestream_leakage_tol is not None:
        keep = leakage <= float(freestream_leakage_tol)
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]
        leakage = leakage[keep]

    idx = np.argsort(-eigenvalues.imag)
    return eigenvalues[idx], eigenvectors[:, idx], leakage[idx]


def solve_temporal_compressible_3d(baseflow, alpha, beta, Re, Ma, Pr, gamma,
                                   N=128, y_max=None, L=None,
                                   wall_bc='isothermal',
                                   include_spanwise_dissipation_coupling=True,
                                   freestream_leakage_tol=None,
                                   return_leakage=False,
                                   length_scale='delta_star',
                                   lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Solve the temporal compressible stability problem for oblique waves.

    State vector uses the wave-aligned variables

        q = (alpha*u + beta*w) / k
        s = (alpha*w - beta*u) / k
        phi = [q_hat, v_hat, s_hat, T_hat, p_hat]^T

    with ``k = sqrt(alpha^2 + beta^2)`` and eigenvalue ``c = omega / alpha``.
    For ``beta = 0``, this reduces to the 2D compressible solver plus a
    decoupled spanwise-velocity equation.

    If ``include_spanwise_dissipation_coupling`` is false, the solver drops the
    single 3D dissipation feedback term identified by Mack as the coupling from
    the last two first-order equations back into the primary six-equation
    subsystem. This corresponds to Mack's sixth-order approximation.

    If ``freestream_leakage_tol`` is provided, modes are post-filtered using
    the Appendix-B boundedness subspace at the finite upper boundary. The
    leakage score is a relative residual in Mack's first-order variables.
    """
    A, B, y, D1, n, alpha, beta, bf = assemble_temporal_compressible_3d_evp(
        baseflow,
        alpha,
        beta,
        Re,
        Ma,
        Pr,
        gamma,
        N=N,
        y_max=y_max,
        L=L,
        include_spanwise_dissipation_coupling=(
            include_spanwise_dissipation_coupling
        ),
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio,
    )
    Pr_freestream = bf['Pr_local'][0] if 'Pr_local' in bf else Pr

    apply_wall_bc_3d(A, B, D1, n, wall_bc=wall_bc)
    apply_dirichlet_freestream_bc_3d(A, B, n)

    eigenvalues, eigenvectors = linalg.eig(A, B)

    eigenvalues, eigenvectors, leakage = _filter_temporal_modes_3d(
        eigenvalues, eigenvectors, D1, alpha, beta, Re, Ma, Pr, gamma,
        Pr_freestream=Pr_freestream,
        freestream_leakage_tol=freestream_leakage_tol,
        lambda_mu_ratio=lambda_mu_ratio,
    )

    if return_leakage:
        return eigenvalues, eigenvectors, y, leakage
    return eigenvalues, eigenvectors, y


def temporal_candidate_spectrum_3d(
    baseflow,
    alpha,
    beta,
    Re,
    Ma,
    Pr,
    gamma,
    N=128,
    y_max=None,
    L=None,
    wall_bc='isothermal',
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    max_candidates=None,
    include_qr_residual=False,
    qr_n_steps=800,
):
    """Return the filtered 3D temporal spectrum plus selection diagnostics."""
    c_all, modes, y, leakage = solve_temporal_compressible_3d(
        baseflow,
        alpha,
        beta,
        Re,
        Ma,
        Pr,
        gamma,
        N=N,
        y_max=y_max,
        L=L,
        wall_bc=wall_bc,
        include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
        freestream_leakage_tol=freestream_leakage_tol,
        return_leakage=True,
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio,
    )

    if max_candidates is not None:
        keep = slice(0, min(int(max_candidates), len(c_all)))
        c_all = c_all[keep]
        modes = modes[:, keep]
        leakage = leakage[keep]

    omega_i = alpha * c_all.imag
    qr_residual = np.full(len(c_all), np.nan, dtype=float)

    if include_qr_residual and len(c_all) > 0:
        from .mack_shooting import temporal_shooting_residual_3d

        local_y_max = y_max
        if local_y_max is None:
            local_y_max = 6.0 if Ma > 2.0 else 12.0

        for j, c_val in enumerate(c_all):
            try:
                qr_residual[j] = abs(
                    temporal_shooting_residual_3d(
                        baseflow,
                        alpha,
                        beta,
                        c_val,
                        Re,
                        Ma,
                        Pr,
                        gamma,
                        local_y_max,
                        lambda_mu_ratio=lambda_mu_ratio,
                        length_scale=length_scale,
                        wall_bc=wall_bc,
                        method='qr',
                        n_steps=qr_n_steps,
                    )
                )
            except Exception:
                qr_residual[j] = np.inf

    return {
        'c': c_all,
        'modes': modes,
        'y': y,
        'omega_i': omega_i,
        'leakage': leakage,
        'qr_residual': qr_residual,
    }


def select_temporal_candidate_3d(
    spectrum,
    c_target=None,
    prefer_positive_growth=False,
    proximity_weight=1.0,
    leakage_weight=0.25,
    qr_weight=0.25,
    growth_weight=0.0,
    damping_penalty=1.0,
    c_real_bounds=None,
    c_imag_abs_max=None,
    out_of_bounds_penalty=10.0,
):
    """Choose one candidate mode from a diagnostic temporal spectrum."""
    c_all = np.asarray(spectrum['c'])
    if len(c_all) == 0:
        raise ValueError('cannot select from an empty temporal spectrum')

    omega_i = np.asarray(spectrum['omega_i'], dtype=float)
    leakage = np.asarray(spectrum['leakage'], dtype=float)
    qr_residual = np.asarray(spectrum['qr_residual'], dtype=float)

    scores = np.zeros(len(c_all), dtype=float)

    if c_target is not None:
        scale = max(abs(c_target), 1e-8)
        scores += proximity_weight * np.abs(c_all - c_target) / scale

    if leakage_weight:
        scores += leakage_weight * leakage

    if qr_weight:
        qr_safe = np.nan_to_num(qr_residual, nan=np.inf, posinf=np.inf)
        scores += qr_weight * qr_safe

    if c_real_bounds is not None:
        c_real_min, c_real_max = c_real_bounds
        out_of_bounds = (c_all.real < c_real_min) | (c_all.real > c_real_max)
        scores += out_of_bounds_penalty * out_of_bounds

    if c_imag_abs_max is not None:
        scores += out_of_bounds_penalty * (np.abs(c_all.imag) > c_imag_abs_max)

    if growth_weight:
        growth_scale = max(np.nanmax(np.abs(omega_i)), 1e-8)
        scores -= growth_weight * omega_i / growth_scale

    if prefer_positive_growth:
        scores += damping_penalty * (omega_i < 0.0)

    index = int(np.argmin(scores))
    return index, scores


def continue_temporal_mode_3d(
    case_sequence,
    baseflow_builder=None,
    *,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    L=None,
    wall_bc='isothermal',
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    initial_c=None,
    initial_selector='max_growth',
    prefer_positive_growth=True,
    proximity_weight=1.0,
    leakage_weight=0.25,
    qr_weight=0.25,
    growth_weight=0.0,
    damping_penalty=1.0,
    c_real_bounds=None,
    c_imag_abs_max=None,
    out_of_bounds_penalty=10.0,
    use_asymptotic_refinement=True,
    include_qr_residual=False,
    qr_n_steps=800,
):
    """Track one 3D temporal mode family across a sequence of parameter points.

    Each entry of ``case_sequence`` must provide ``alpha``, ``beta``, ``Re``,
    and ``Ma``. A per-case ``baseflow`` may be supplied directly; otherwise
    ``baseflow_builder(case_dict)`` is used.
    """
    tracked = []
    c_target = initial_c

    for case in case_sequence:
        case_data = dict(case)
        alpha = float(case_data['alpha'])
        beta = float(case_data['beta'])
        Re = float(case_data['Re'])
        Ma = float(case_data['Ma'])
        Pr_case = float(case_data.get('Pr', Pr))
        gamma_case = float(case_data.get('gamma', gamma))
        baseflow = case_data.get('baseflow')
        if baseflow is None:
            if baseflow_builder is None:
                raise ValueError('baseflow_builder is required when case has no baseflow')
            baseflow = baseflow_builder(case_data)

        spectrum = temporal_candidate_spectrum_3d(
            baseflow,
            alpha,
            beta,
            Re,
            Ma,
            Pr_case,
            gamma_case,
            N=N,
            y_max=y_max,
            L=L,
            wall_bc=wall_bc,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            freestream_leakage_tol=freestream_leakage_tol,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
            include_qr_residual=include_qr_residual,
            qr_n_steps=qr_n_steps,
        )

        if len(spectrum['c']) == 0:
            tracked.append({
                'case': case_data,
                'spectrum': spectrum,
                'selected_index': None,
                'selected_c': np.nan + 0j,
                'refined_c': np.nan + 0j,
                'refined_converged': False,
                'scores': np.array([], dtype=float),
            })
            c_target = None
            continue

        if c_target is None:
            if initial_selector == 'max_growth':
                idx = int(np.argmax(spectrum['omega_i']))
                scores = np.full(len(spectrum['c']), np.nan)
            elif initial_selector == 'min_leakage':
                idx = int(np.argmin(spectrum['leakage']))
                scores = np.full(len(spectrum['c']), np.nan)
            else:
                idx, scores = select_temporal_candidate_3d(
                    spectrum,
                    c_target=None,
                    prefer_positive_growth=prefer_positive_growth,
                    proximity_weight=proximity_weight,
                    leakage_weight=leakage_weight,
                    qr_weight=qr_weight,
                    growth_weight=growth_weight,
                    damping_penalty=damping_penalty,
                    c_real_bounds=c_real_bounds,
                    c_imag_abs_max=c_imag_abs_max,
                    out_of_bounds_penalty=out_of_bounds_penalty,
                )
        else:
            idx, scores = select_temporal_candidate_3d(
                spectrum,
                c_target=c_target,
                prefer_positive_growth=prefer_positive_growth,
                proximity_weight=proximity_weight,
                leakage_weight=leakage_weight,
                qr_weight=qr_weight,
                growth_weight=growth_weight,
                damping_penalty=damping_penalty,
                c_real_bounds=c_real_bounds,
                c_imag_abs_max=c_imag_abs_max,
                out_of_bounds_penalty=out_of_bounds_penalty,
            )

        selected_c = spectrum['c'][idx]
        refined_c = selected_c
        refined_converged = False
        refined_leakage = spectrum['leakage'][idx]

        if use_asymptotic_refinement:
            refined_c, _, _, refined_converged, refined_leakage = (
                refine_temporal_compressible_3d_asymptotic(
                    baseflow,
                    alpha,
                    beta,
                    Re,
                    Ma,
                    Pr_case,
                    gamma_case,
                    c_guess=selected_c,
                    N=N,
                    y_max=y_max,
                    L=L,
                    wall_bc=wall_bc,
                    include_spanwise_dissipation_coupling=(
                        include_spanwise_dissipation_coupling
                    ),
                    freestream_leakage_tol=freestream_leakage_tol,
                    length_scale=length_scale,
                    lambda_mu_ratio=lambda_mu_ratio,
                )
            )
            if np.isfinite(refined_c):
                c_target = refined_c
            else:
                c_target = selected_c
        else:
            c_target = selected_c

        tracked.append({
            'case': case_data,
            'spectrum': spectrum,
            'selected_index': idx,
            'selected_c': selected_c,
            'refined_c': refined_c,
            'refined_converged': refined_converged,
            'refined_leakage': refined_leakage,
            'scores': scores,
        })

    return tracked


def refine_temporal_compressible_3d_asymptotic(
    baseflow,
    alpha,
    beta,
    Re,
    Ma,
    Pr,
    gamma,
    c_guess=None,
    N=128,
    y_max=None,
    L=None,
    wall_bc='isothermal',
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    tol=1e-8,
    max_iter=8,
    return_history=False,
):
    """Refine a 3D temporal mode with Mack Appendix-B freestream BCs.

    The full boundedness condition is nonlinear because the freestream decay
    subspace depends on the eigenvalue. This routine freezes the Appendix-B
    boundary relations at a trial ``c``, solves the resulting linear EVP, and
    iterates until the selected mode is self-consistent.
    """
    if c_guess is None:
        c_all, _, _, leakage = solve_temporal_compressible_3d(
            baseflow,
            alpha,
            beta,
            Re,
            Ma,
            Pr,
            gamma,
            N=N,
            y_max=y_max,
            L=L,
            wall_bc=wall_bc,
            include_spanwise_dissipation_coupling=(
                include_spanwise_dissipation_coupling
            ),
            freestream_leakage_tol=freestream_leakage_tol,
            return_leakage=True,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
        )
        mask = (
            (c_all.real > 0.0) & (c_all.real < 1.2)
            & (np.abs(c_all.imag) < 0.3)
        )
        if freestream_leakage_tol is not None:
            mask &= leakage <= float(freestream_leakage_tol)
        c_all = c_all[mask]
        if len(c_all) == 0:
            if return_history:
                return np.nan + 0j, None, None, False, np.nan, []
            return np.nan + 0j, None, None, False, np.nan
        c_ref = c_all[np.argmax(alpha * c_all.imag)]
    else:
        c_ref = complex(c_guess)

    history = [c_ref]
    best_mode = None
    best_y = None
    best_leakage = np.nan
    converged = False

    for _ in range(max_iter):
        A, B, y, D1, n, alpha_eval, beta_eval, bf = assemble_temporal_compressible_3d_evp(
            baseflow,
            alpha,
            beta,
            Re,
            Ma,
            Pr,
            gamma,
            N=N,
            y_max=y_max,
            L=L,
            include_spanwise_dissipation_coupling=(
                include_spanwise_dissipation_coupling
            ),
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
        )
        Pr_freestream = bf['Pr_local'][0] if 'Pr_local' in bf else Pr
        apply_wall_bc_3d(A, B, D1, n, wall_bc=wall_bc)
        _apply_asymptotic_freestream_bc_3d(
            A, B, D1, n, alpha_eval, beta_eval, Re, Ma, Pr, gamma, c_ref,
            Pr_freestream=Pr_freestream,
            lambda_mu_ratio=lambda_mu_ratio,
        )

        eigenvalues, eigenvectors = linalg.eig(A, B)
        eigenvalues, eigenvectors, leakage = _filter_temporal_modes_3d(
            eigenvalues, eigenvectors, D1, alpha_eval, beta_eval, Re, Ma, Pr, gamma,
            Pr_freestream=Pr_freestream,
            freestream_leakage_tol=freestream_leakage_tol,
            lambda_mu_ratio=lambda_mu_ratio,
        )
        if len(eigenvalues) == 0:
            break

        pick = np.argmin(np.abs(eigenvalues - c_ref))
        c_new = eigenvalues[pick]
        best_mode = eigenvectors[:, pick]
        best_y = y
        best_leakage = leakage[pick]
        history.append(c_new)

        if abs(c_new - c_ref) < tol:
            c_ref = c_new
            converged = True
            break
        c_ref = c_new

    if return_history:
        return c_ref, best_mode, best_y, converged, best_leakage, history
    return c_ref, best_mode, best_y, converged, best_leakage


def track_mode(alphas, target, n_nearest=1):
    """Find eigenvalue(s) nearest to a target."""
    if len(alphas) == 0:
        return np.array([target])
    dist = np.abs(alphas - target)
    idx = np.argsort(dist)[:n_nearest]
    return alphas[idx]


# ========================================================================
# SPECTRAL RAYLEIGH SOLVER (inviscid, replaces broken shooting method)
# ========================================================================

def solve_rayleigh_spectral(baseflow, alpha, N=200, y_max=20.0, L=None,
                            contour_shift=0.0):
    """Solve the Rayleigh equation using Chebyshev spectral collocation.

    The Rayleigh equation:
        [(alpha*U - omega)(D^2 - alpha^2) - alpha*D^2U] v_hat = 0

    Rearranged as generalized EVP:  A*v = omega*B*v

    For profiles WITHOUT an inflection point (e.g. Blasius), damped
    eigenvalues exist only on a deformed contour (Lin 1945). Set
    contour_shift > 0 to deform the y-contour into the lower complex
    half-plane, enabling detection of these damped modes.

    The profile is analytically continued to complex y using Taylor:
        U(y + i*dy) = U(y) + i*dy*U'(y) - dy^2/2*U''(y)

    Parameters
    ----------
    baseflow : callable
        Profile object returning dict with U, dU, d2U.
    alpha : float
        Real wavenumber.
    N : int
        Number of Chebyshev intervals.
    y_max : float
        Domain height.
    contour_shift : float
        Magnitude of imaginary shift for contour deformation.
        0 = real axis (standard), >0 = deformed below (Lin's contour).

    Returns
    -------
    omega : array
        Complex frequencies, sorted by omega_i descending.
    modes : array
        Eigenvectors.
    y : array
        Real part of grid points.
    """
    D_eta = chebyshev_D(N)
    y_real, D1_real, D2_real = physical_derivatives(D_eta, y_max, N, L)

    bf = baseflow(y_real)
    n = len(y_real)
    I_mat = np.eye(n)

    if contour_shift > 0:
        # Deform contour: y_complex = y_real - i * shift * bump(y)
        # Bump is a smooth function zero at boundaries, max in interior
        bump = y_real * (y_max - y_real) / (y_max / 2)**2
        y_imag = -contour_shift * bump

        # Analytically continue U to complex y via Taylor expansion
        # U(y_r + i*y_i) ≈ U(y_r) + i*y_i*U'(y_r) - y_i^2/2*U''(y_r)
        U_vals = (bf['U'] + 1j * y_imag * bf['dU']
                  - 0.5 * y_imag**2 * bf['d2U'])
        d2U_vals = bf['d2U'] + 0j  # Leading order

        # Metric: dy_complex/dy_real = 1 + i * d(y_imag)/d(y_real)
        # d(y_imag)/d(y_real) = -shift * (y_max - 2*y_real) / (y_max/2)^2
        dy_imag_dy = -contour_shift * (y_max - 2 * y_real) / (y_max / 2)**2
        metric = 1.0 + 1j * dy_imag_dy

        # Transform derivatives: D_complex = (1/metric) * D_real
        metric_inv = 1.0 / metric
        D1 = np.diag(metric_inv) @ D1_real
        D2 = (np.diag(metric_inv**2) @ D2_real
              - np.diag(metric_inv**3 * 1j *
                        2 * contour_shift / (y_max / 2)**2) @ D1_real)
    else:
        U_vals = bf['U'] + 0j
        d2U_vals = bf['d2U'] + 0j
        D1 = D1_real
        D2 = D2_real

    U_diag = np.diag(U_vals)
    d2U_diag = np.diag(d2U_vals)

    L2 = D2 - alpha**2 * I_mat

    A = alpha * U_diag @ L2 - alpha * d2U_diag
    B = L2.copy().astype(complex)

    # BCs
    A[0, :] = 0;   A[0, 0] = 1;   B[0, :] = 0
    A[-1, :] = 0;  A[-1, -1] = 1; B[-1, :] = 0

    eigenvalues, eigenvectors = linalg.eig(A, B)

    valid = np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    c = eigenvalues / alpha
    phys = ((c.real > -0.1) & (c.real < 1.2) &
            (np.abs(c.imag) < 0.5))
    eigenvalues = eigenvalues[phys]
    eigenvectors = eigenvectors[:, phys]

    idx = np.argsort(-eigenvalues.imag)
    return eigenvalues[idx], eigenvectors[:, idx], y_real


# ========================================================================
# SPATIAL SOLVER VIA MULLER'S METHOD (replaces broken QEP companion)
# ========================================================================

def _spatial_determinant(alpha, C0, C1, C2):
    """Evaluate the eigenvalue of L(alpha) closest to zero.

    L(alpha) = C0 + alpha*C1 + alpha^2*C2
    Returns the complex eigenvalue nearest to zero — this is zero
    when alpha is a spatial eigenvalue of the stability problem.
    Using eigenvalue (complex) instead of SVD (real) gives Muller's
    method the phase information it needs for quadratic interpolation.
    """
    L = C0 + alpha * C1 + alpha**2 * C2
    try:
        eigs = linalg.eigvals(L)
        idx = np.argmin(np.abs(eigs))
        return eigs[idx]
    except Exception:
        return 1e30 + 0j


def _muller_step(z0, z1, z2, f0, f1, f2):
    """One step of Muller's method for complex root-finding.

    Given three points (z0,f0), (z1,f1), (z2,f2), fit a quadratic
    and return the root closest to z2.
    """
    h0 = z1 - z0
    h1 = z2 - z1
    d0 = (f1 - f0) / h0
    d1 = (f2 - f1) / h1
    a = (d1 - d0) / (h1 + h0)
    b = d1 + a * h1
    c_val = f2

    disc = b**2 - 4 * a * c_val
    sqrt_disc = np.sqrt(disc)

    # Choose denominator with larger magnitude (numerical stability)
    denom1 = b + sqrt_disc
    denom2 = b - sqrt_disc
    if abs(denom1) > abs(denom2):
        dz = -2 * c_val / denom1
    else:
        dz = -2 * c_val / denom2

    return z2 + dz


def solve_spatial_muller(baseflow, omega, Re, Ma, Pr, gamma, alpha_guess,
                         N=100, y_max=None, tol=1e-8, max_iter=30,
                         length_scale='delta_star'):
    """Find spatial eigenvalue alpha using Muller's method.

    Instead of solving the full QEP and filtering, this directly
    searches for the complex alpha that makes L(alpha) singular.
    Uses the smallest singular value of L(alpha) as the objective.

    The key insight: Muller's method needs only 3 initial points and
    converges quadratically for simple roots. No companion system needed.

    Parameters
    ----------
    baseflow : callable
        Profile object.
    omega : float
        Real frequency.
    Re, Ma, Pr, gamma : float
        Flow parameters.
    alpha_guess : complex
        Initial guess (e.g., from Gaster transform of temporal result).
    N : int
        Chebyshev intervals.
    y_max : float
        Domain height.
    tol : float
        Convergence tolerance on |sigma_min|.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    alpha : complex
        Spatial eigenvalue.
    converged : bool
        Whether the iteration converged.
    sigma_min : float
        Final smallest singular value (should be ~0 if converged).
    """
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N)
    bf, D1, D2 = _scaled_compressible_problem(baseflow, y, D1, D2, length_scale)

    C0, C1, C2 = assemble_compressible_matrices(
        D1, D2, y, bf, omega, Re, Ma, Pr, gamma)

    n = len(y)
    nn = 4 * n

    # Apply BCs
    wall = n - 1
    free = 0
    for var in range(3):
        for loc in [wall, free]:
            row = var * n + loc
            C0[row, :] = 0
            C1[row, :] = 0
            C2[row, :] = 0
            C0[row, row] = 1.0

    # Muller's method needs 3 initial points
    ag = complex(alpha_guess)
    z0 = ag - 0.01 - 0.005j
    z1 = ag + 0.01
    z2 = ag

    f0 = _spatial_determinant(z0, C0, C1, C2)
    f1 = _spatial_determinant(z1, C0, C1, C2)
    f2 = _spatial_determinant(z2, C0, C1, C2)

    for iteration in range(max_iter):
        if abs(f2) < tol:
            return z2, True, abs(f2)

        try:
            z_new = _muller_step(z0, z1, z2, f0, f1, f2)
        except (ZeroDivisionError, ValueError):
            # Muller step failed, try small perturbation
            z_new = z2 + 0.001 * (1 + 1j)

        f_new = _spatial_determinant(z_new, C0, C1, C2)

        # Shift: oldest point dropped
        z0, z1, z2 = z1, z2, z_new
        f0, f1, f2 = f1, f2, f_new

    return z2, abs(f2) < tol * 100, abs(f2)


def _temporal_at_complex_alpha(baseflow, alpha_complex, Re, Ma, Pr, gamma,
                                N, y_max, c_target,
                                length_scale='delta_star',
                                lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Evaluate temporal EVP at complex alpha; return c nearest to target.

    This is the key building block for spatial Newton iteration.
    The temporal solver works with complex alpha — the matrices just
    become fully complex.
    """
    c_all, _, _ = solve_temporal_compressible(
        baseflow, alpha_complex, Re, Ma, Pr, gamma, N=N, y_max=y_max,
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio)
    if len(c_all) == 0:
        return None
    # Find mode nearest to target
    dist = np.abs(c_all - c_target)
    return c_all[np.argmin(dist)]


def solve_spatial_newton(baseflow, omega, Re, Ma, Pr, gamma,
                          alpha_guess, c_target, N=100, y_max=None,
                          tol=1e-6, max_iter=20,
                          length_scale='delta_star',
                          lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Find spatial eigenvalue alpha via Newton on omega(alpha) = omega_target.

    Given the converged temporal mode c at real alpha, iterate on
    complex alpha until alpha*c(alpha) = omega.

    Parameters
    ----------
    alpha_guess : complex
        Initial guess from Gaster relation.
    c_target : complex
        Phase speed of the tracked mode (for nearest-mode selection).

    Returns
    -------
    alpha : complex
        Converged spatial eigenvalue.
    converged : bool
    """
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    alpha = complex(alpha_guess)
    da = 1e-5  # finite difference step

    for it in range(max_iter):
        c = _temporal_at_complex_alpha(
            baseflow, alpha, Re, Ma, Pr, gamma, N, y_max, c_target,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio)
        if c is None:
            return alpha, False

        # Residual: F(alpha) = alpha*c - omega
        F = alpha * c - omega
        if abs(F) < tol:
            return alpha, True

        # Jacobian by finite difference: dF/dalpha ≈ [F(alpha+da) - F(alpha)] / da
        c_p = _temporal_at_complex_alpha(
            baseflow, alpha + da, Re, Ma, Pr, gamma, N, y_max, c_target,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio)
        if c_p is None:
            return alpha, False

        F_p = (alpha + da) * c_p - omega
        dF_dalpha = (F_p - F) / da

        if abs(dF_dalpha) < 1e-30:
            return alpha, False

        # Newton step
        alpha_new = alpha - F / dF_dalpha

        # Update target for mode tracking
        c_target = c

        # Damped step if change is too large
        step = alpha_new - alpha
        if abs(step) > 0.5:
            step = 0.5 * step / abs(step)
        alpha = alpha + step

    return alpha, abs(alpha * c - omega) < tol * 100


def solve_spatial_from_temporal(baseflow, omega, Re, Ma, Pr, gamma,
                                 N=100, y_max=None,
                                 length_scale='delta_star',
                                 lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Solve spatial problem using temporal result + Gaster + Muller.

    Pipeline:
    1. Find converged temporal eigenvalue (two-resolution filtering)
    2. Use Gaster relation for spatial initial guess
    3. Refine with Muller's method on sigma_min[L(alpha)]

    Returns
    -------
    alpha : complex
        Spatial eigenvalue.
    converged : bool
    """
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    # Step 1: estimate alpha_r from phase speed
    if Ma > 3:
        c_est = 0.98  # discrete second mode
    else:
        c_est = 0.35

    alpha_r = omega / c_est

    # Step 2: convergence-filtered temporal eigenvalue
    N_lo, N_hi = max(N - 30, 60), N
    c_lo, _, _ = solve_temporal_compressible(
        baseflow, alpha_r, Re, Ma, Pr, gamma, N=N_lo, y_max=y_max,
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio)
    c_hi, _, _ = solve_temporal_compressible(
        baseflow, alpha_r, Re, Ma, Pr, gamma, N=N_hi, y_max=y_max,
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio)

    filt = lambda c: ((c.real > 0.3) & (c.real < 1.05) &
                       (np.abs(c.imag) < 0.3))
    cl, ch = c_lo[filt(c_lo)], c_hi[filt(c_hi)]

    best = None
    if len(cl) > 0 and len(ch) > 0:
        converged_modes = []
        for c in ch:
            if np.min(np.abs(cl - c)) < 0.008:
                converged_modes.append(c)
        if converged_modes:
            converged_modes = np.array(converged_modes)
            best = converged_modes[np.argmax(converged_modes.imag)]

    if best is None:
        # Fallback: use unfiltered most unstable
        if len(ch) > 0:
            best = ch[np.argmax(ch.imag)]
        else:
            return alpha_r + 0j, False

    # Step 3: Gaster relation
    # -alpha_i = alpha * c_i / c_r (spatial growth from temporal)
    c_r = best.real
    c_i = best.imag
    alpha_i_guess = -alpha_r * c_i / max(c_r, 0.1)

    alpha_guess = alpha_r + 1j * alpha_i_guess

    # Step 4: Newton refinement
    alpha, converged = solve_spatial_newton(
        baseflow, omega, Re, Ma, Pr, gamma,
        alpha_guess=alpha_guess, c_target=best,
        N=N, y_max=y_max, length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio)

    return alpha, converged


# --- Backwards-compatible aliases (private names promoted to public API) ---
_temperature_wall_operator = temperature_wall_operator
_assemble_temporal_compressible_3d_evp = assemble_temporal_compressible_3d_evp
_apply_wall_bc_3d = apply_wall_bc_3d
_apply_dirichlet_freestream_bc_3d = apply_dirichlet_freestream_bc_3d
