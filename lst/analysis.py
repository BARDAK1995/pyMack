"""
Parameter sweeps, neutral curves, and N-factor integration.
"""

import numpy as np
from .solver import solve_temporal_os, solve_spatial, track_mode


def temporal_spectrum(baseflow, alpha, Re, N=128, y_max=40.0):
    """Compute temporal eigenspectrum for given α and Re."""
    c, modes, y = solve_temporal_os(baseflow, alpha, Re, N, y_max)
    return c, modes, y


def temporal_growth_scan(baseflow, Re, alpha_range, N=128, y_max=40.0):
    """Scan growth rate vs α for temporal O-S.

    Returns
    -------
    alphas : array
    omega_i : array  (temporal growth rates: ω_i = α c_i)
    c_vals : array   (most unstable c for each α)
    """
    alphas = np.asarray(alpha_range)
    omega_i = np.zeros(len(alphas))
    c_vals = np.zeros(len(alphas), dtype=complex)

    for i, alpha in enumerate(alphas):
        c, _, _ = solve_temporal_os(baseflow, alpha, Re, N, y_max)
        if len(c) > 0:
            c_vals[i] = c[0]  # most unstable
            omega_i[i] = alpha * c[0].imag
        else:
            c_vals[i] = np.nan
            omega_i[i] = np.nan

    return alphas, omega_i, c_vals


def find_critical_Re(baseflow, alpha_range=(0.1, 0.5), Re_range=(300, 1200),
                     N=128, y_max=40.0, tol=1):
    """Find critical Reynolds number where max growth rate crosses zero.

    Uses bisection on Re, scanning α at each Re to find max ω_i.
    """
    from scipy.optimize import brentq

    alpha_arr = np.linspace(alpha_range[0], alpha_range[1], 30)

    def max_growth(Re):
        _, omega_i, _ = temporal_growth_scan(
            baseflow, Re, alpha_arr, N, y_max)
        return np.nanmax(omega_i)

    try:
        Re_crit = brentq(max_growth, Re_range[0], Re_range[1], xtol=tol)
    except ValueError:
        # Growth rate doesn't change sign — scan and interpolate
        Re_arr = np.linspace(Re_range[0], Re_range[1], 20)
        gi = [max_growth(R) for R in Re_arr]
        # Find zero crossing
        gi = np.array(gi)
        crossings = np.where(np.diff(np.sign(gi)))[0]
        if len(crossings) > 0:
            i = crossings[0]
            Re_crit = np.interp(0, gi[i:i+2], Re_arr[i:i+2])
        else:
            Re_crit = np.nan
    return Re_crit


def frequency_sweep(baseflow, Re, Ma, omega_range, Pr=0.72, gamma=1.4,
                    N=128, y_max=None, wall_bc='isothermal'):
    """Compute spatial growth rate σ = -α_i vs frequency.

    Parameters
    ----------
    baseflow : callable
        Mean flow profile.
    Re : float
        Reynolds number.
    Ma : float
        Mach number.
    omega_range : array-like
        Frequencies to scan.
    Pr, gamma : float
        Gas properties.
    N : int
        Chebyshev points.
    y_max : float, optional
        Domain height.
    wall_bc : str
        Wall thermal BC.

    Returns
    -------
    omegas : array
    sigma : array  (growth rates)
    alpha_r : array  (wavenumbers)
    """
    omegas = np.asarray(omega_range)
    sigma = np.zeros(len(omegas))
    alpha_r = np.zeros(len(omegas))
    alpha_tracked = None

    for i, om in enumerate(omegas):
        alphas, _, _ = solve_spatial(
            baseflow, om, Re, Ma, Pr, gamma, N, y_max, wall_bc=wall_bc)

        if len(alphas) == 0:
            sigma[i] = np.nan
            alpha_r[i] = np.nan
            continue

        if alpha_tracked is not None:
            nearest = track_mode(alphas, alpha_tracked)
            a = nearest[0]
        else:
            # Pick most unstable
            idx = np.argmin(alphas.imag)
            a = alphas[idx]

        alpha_tracked = a
        sigma[i] = -a.imag
        alpha_r[i] = a.real

    return omegas, sigma, alpha_r


def neutral_curve(baseflow_func, Ma, Re_range, omega_range,
                  Pr=0.72, gamma=1.4, N=128, y_max=None,
                  wall_bc='isothermal'):
    """Compute neutral stability curve in (Re, ω) space.

    baseflow_func(Re) should return a baseflow callable for that Re.

    Returns
    -------
    Re_arr : array
    omega_arr : array
    sigma_map : 2D array  (growth rates, Re × ω)
    """
    Re_arr = np.asarray(Re_range)
    omega_arr = np.asarray(omega_range)
    sigma_map = np.zeros((len(Re_arr), len(omega_arr)))

    for i, Re in enumerate(Re_arr):
        bf = baseflow_func(Re)
        _, sig, _ = frequency_sweep(
            bf, Re, Ma, omega_arr, Pr, gamma, N, y_max, wall_bc)
        sigma_map[i, :] = sig

    return Re_arr, omega_arr, sigma_map


def nfactor(baseflow_func, Ma, omega, Re_range, Pr=0.72, gamma=1.4,
            N=128, y_max=None, wall_bc='isothermal'):
    """Compute N-factor by integrating spatial growth rate.

    N(x) = -∫_{x_0}^{x} α_i dx

    where x_0 is the Branch I neutral point.

    Parameters
    ----------
    baseflow_func : callable
        baseflow_func(Re) returns profile for that Re.
    omega : float
        Fixed frequency.
    Re_range : array
        Re values (proxy for x stations).

    Returns
    -------
    Re_arr : array
    N_vals : array
    sigma : array
    """
    Re_arr = np.asarray(Re_range)
    sigma = np.zeros(len(Re_arr))
    alpha_tracked = None

    for i, Re in enumerate(Re_arr):
        bf = baseflow_func(Re)
        alphas, _, _ = solve_spatial(
            bf, omega, Re, Ma, Pr, gamma, N, y_max, wall_bc=wall_bc)

        if len(alphas) == 0:
            sigma[i] = np.nan
            continue

        if alpha_tracked is not None:
            nearest = track_mode(alphas, alpha_tracked)
            a = nearest[0]
        else:
            idx = np.argmin(alphas.imag)
            a = alphas[idx]

        alpha_tracked = a
        sigma[i] = -a.imag

    # Integrate: N = ∫ σ dx, only where σ > 0
    # Use Re as proxy for x (Re ∝ x)
    N_vals = np.zeros(len(Re_arr))
    sigma_pos = np.maximum(sigma, 0)

    for i in range(1, len(Re_arr)):
        dRe = Re_arr[i] - Re_arr[i-1]
        N_vals[i] = N_vals[i-1] + 0.5 * (sigma_pos[i] + sigma_pos[i-1]) * dRe

    return Re_arr, N_vals, sigma
