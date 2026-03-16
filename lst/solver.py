"""
Eigenvalue solver with mode filtering and tracking.

Temporal solver for Orr-Sommerfeld (incompressible validation)
and spatial solver for the compressible stability equations.
"""

import numpy as np
from scipy import linalg

from .spectral import chebyshev_points, chebyshev_D, physical_derivatives
from .equations import assemble_orr_sommerfeld, assemble_compressible_matrices


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
                  target_alpha=None, n_modes=20):
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
    if y_max is None:
        y_max = 6.0 if Ma > 2.0 else 12.0

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)
    bf = baseflow(y)

    C0, C1, C2 = assemble_compressible_matrices(
        D1, D2, y, bf, omega, Re, Ma, Pr, gamma)

    n = len(y)
    nn = 4 * n

    # Apply BCs: u=v=T=0 at wall and freestream
    wall = n - 1
    free = 0
    for var in range(3):
        for loc in [wall, free]:
            row = var * n + loc
            C0[row, :] = 0
            C1[row, :] = 0
            C2[row, :] = 0
            C0[row, row] = 1.0

    # Estimate target if not given
    if target_alpha is None:
        # Second mode phase speed: c ~ 1 - 1/Ma for Ma > 3
        # First mode: c ~ 0.4
        if Ma > 3:
            c_2nd = 1.0 - 1.0 / Ma
            target_alpha = omega / c_2nd + 0j
        else:
            target_alpha = omega / 0.4 + 0j

    # Use the determinant-minimization approach:
    # For each alpha, L(alpha) = C0 + alpha*C1 + alpha^2*C2
    # Eigenvalue alpha makes L(alpha) singular.
    # Use companion + shift-invert near target.
    LL = np.zeros((2*nn, 2*nn), dtype=complex)
    RR = np.zeros((2*nn, 2*nn), dtype=complex)

    LL[:nn, :nn] = -C1
    LL[:nn, nn:] = -C0
    LL[nn:, :nn] = np.eye(nn)

    RR[:nn, :nn] = C2
    RR[nn:, nn:] = np.eye(nn)

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
                                wall_bc='isothermal'):
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
    bf = baseflow(y)

    n = len(y)
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Mean flow
    Ub = np.diag(bf['U'])
    dUb = np.diag(bf['dU'])
    d2Ub = np.diag(bf['d2U'])
    Tb = np.diag(bf['T'])
    dTb_diag = np.diag(bf['dT'])
    mub = np.diag(bf['mu'])
    dmub = np.diag(bf['dmu'])

    T_v = bf['T']
    rho_v = bf['rho']
    mu_v = bf['mu']
    dT_v = bf['dT']
    dU_v = bf['dU']

    rhoI = np.diag(1.0 / rho_v)
    TI = np.diag(1.0 / T_v)

    # Viscosity derivatives
    log_T = np.log(np.maximum(T_v, 1e-30))
    log_mu = np.log(np.maximum(mu_v, 1e-30))
    omega_v = np.where(np.abs(log_T) > 1e-10, log_mu / log_T, 0.74)
    omega_v = np.clip(omega_v, 0.5, 1.0)
    dmu_dT = np.diag(omega_v * mu_v / T_v)
    d2mu_dT2 = np.diag(omega_v * (omega_v - 1) * mu_v / T_v**2)

    kappab = mub
    dkappab = dmub
    dkappa_dT = dmu_dT

    gm1 = gamma - 1.0
    Ma2 = Ma**2
    ia = 1j * alpha
    a2 = alpha**2

    visc = rhoI / Re
    cond = rhoI / (Pr * Re)
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
                     - visc @ (mub @ D2 + dmub @ D1 + dmu_dT @ dTb_diag @ D1)
                     + (4.0/3.0) * a2 * visc @ mub)
    A[blk(1, 1)] = dUb - (ia/3.0) * visc @ (mub @ D1 + dmub)
    A[blk(1, 2)] = -visc @ (dmu_dT @ d2Ub + d2mu_dT2 @ dUb @ dTb_diag
                             + dmu_dT @ dUb @ D1)
    A[blk(1, 3)] = ia * rhoI

    # y-mom: i*alpha*U*v + Dp/rho - visc*[viscous terms]
    A[blk(2, 0)] = -(ia/3.0) * visc @ (mub @ D1 + dmub)
    A[blk(2, 1)] = (ia * Ub
                     - visc @ (mub @ D2 + (4.0/3.0) * dmub @ D1
                               + dmu_dT @ dTb_diag @ D1)
                     + a2 * visc @ mub)
    A[blk(2, 3)] = rhoI @ D1

    # Energy (Form 1): i*alpha*U*T + DT*v - (gm1)*Ma^2*i*alpha*U*p/rho
    #   - cond*[kappa*(D^2 - alpha^2)T + ...] - diss*[2*mu*DU*(Du + i*alpha*v)]
    A[blk(3, 0)] = -diss @ (2.0 * mub @ np.diag(dU_v) @ D1)
    A[blk(3, 1)] = (np.diag(dT_v)
                     - 2j * alpha * diss @ (mub @ np.diag(dU_v)))
    A[blk(3, 2)] = (ia * Ub
                     - cond @ (kappab @ D2 + dkappab @ D1
                               + dkappa_dT @ dTb_diag @ D1)
                     + a2 * cond @ kappab)
    A[blk(3, 3)] = -ia * gm1 * Ma2 * Ub @ rhoI

    # ====== B matrix (coefficient of c, from -c*B part) ======

    # Continuity: c*[i*alpha*(gMa^2*p - T/T)] → B[cont,T] = i*alpha*TI, B[cont,p] = -i*alpha*gMa^2
    B[blk(0, 2)] = ia * TI
    B[blk(0, 3)] = -ia * gamma * Ma2 * I

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
    for var in range(3):  # u, v, T
        for loc in [wall, free]:
            row = var * n + loc
            A[row, :] = 0
            B[row, :] = 0
            A[row, row] = 1.0  # phi_i = 0 regardless of c

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


def track_mode(alphas, target, n_nearest=1):
    """Find eigenvalue(s) nearest to a target."""
    if len(alphas) == 0:
        return np.array([target])
    dist = np.abs(alphas - target)
    idx = np.argsort(dist)[:n_nearest]
    return alphas[idx]
