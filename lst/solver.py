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
                         N=100, y_max=None, tol=1e-8, max_iter=30):
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
    bf = baseflow(y)

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
                                N, y_max, c_target):
    """Evaluate temporal EVP at complex alpha; return c nearest to target.

    This is the key building block for spatial Newton iteration.
    The temporal solver works with complex alpha — the matrices just
    become fully complex.
    """
    c_all, _, _ = solve_temporal_compressible(
        baseflow, alpha_complex, Re, Ma, Pr, gamma, N=N, y_max=y_max)
    if len(c_all) == 0:
        return None
    # Find mode nearest to target
    dist = np.abs(c_all - c_target)
    return c_all[np.argmin(dist)]


def solve_spatial_newton(baseflow, omega, Re, Ma, Pr, gamma,
                          alpha_guess, c_target, N=100, y_max=None,
                          tol=1e-6, max_iter=20):
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
            baseflow, alpha, Re, Ma, Pr, gamma, N, y_max, c_target)
        if c is None:
            return alpha, False

        # Residual: F(alpha) = alpha*c - omega
        F = alpha * c - omega
        if abs(F) < tol:
            return alpha, True

        # Jacobian by finite difference: dF/dalpha ≈ [F(alpha+da) - F(alpha)] / da
        c_p = _temporal_at_complex_alpha(
            baseflow, alpha + da, Re, Ma, Pr, gamma, N, y_max, c_target)
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
                                 N=100, y_max=None):
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
        baseflow, alpha_r, Re, Ma, Pr, gamma, N=N_lo, y_max=y_max)
    c_hi, _, _ = solve_temporal_compressible(
        baseflow, alpha_r, Re, Ma, Pr, gamma, N=N_hi, y_max=y_max)

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
        N=N, y_max=y_max)

    return alpha, converged
