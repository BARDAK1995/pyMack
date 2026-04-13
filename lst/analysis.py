"""
Parameter sweeps, neutral curves, and N-factor integration.
"""

import numpy as np

from .equations import DEFAULT_LAMBDA_MU_RATIO
from .mack_shooting import (
    continue_temporal_mode_3d_shooting_sigma_min,
    solve_temporal_mode_3d_shooting,
    solve_temporal_mode_3d_shooting_sigma_min,
)
from .solver import (
    solve_spatial,
    solve_spatial_from_temporal,
    solve_temporal_compressible_3d,
    solve_temporal_os,
    track_mode,
)


def temporal_spectrum(baseflow, alpha, Re, N=128, y_max=40.0):
    """Compute the temporal eigenspectrum for a given alpha and Re."""
    c, modes, y = solve_temporal_os(baseflow, alpha, Re, N, y_max)
    return c, modes, y


def temporal_growth_scan(baseflow, Re, alpha_range, N=128, y_max=40.0):
    """Scan temporal growth rate versus alpha for Orr-Sommerfeld."""
    alphas = np.asarray(alpha_range)
    omega_i = np.zeros(len(alphas))
    c_vals = np.zeros(len(alphas), dtype=complex)

    for i, alpha in enumerate(alphas):
        c, _, _ = solve_temporal_os(baseflow, alpha, Re, N, y_max)
        if len(c) > 0:
            c_vals[i] = c[0]
            omega_i[i] = alpha * c[0].imag
        else:
            c_vals[i] = np.nan
            omega_i[i] = np.nan

    return alphas, omega_i, c_vals


def _broadcast_float_scan_parameter(value, size, name):
    """Broadcast a scalar or 1D array parameter over a scan."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(size, float(arr))
    if arr.shape != (size,):
        raise ValueError(f'{name} must be scalar or have the same length as alpha_range')
    return arr.astype(float)


def _broadcast_int_scan_parameter(value, size, name):
    """Broadcast an integer-valued scalar or 1D array parameter over a scan."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(size, int(arr), dtype=int)
    if arr.shape != (size,):
        raise ValueError(f'{name} must be scalar or have the same length as alpha_range')
    return arr.astype(int)


def _candidate_root_key(candidate):
    """Pick the best complex root representative for uniqueness checks."""
    for key in ('c_final', 'c_sigma_min'):
        value = candidate[key]
        if np.isfinite(value.real) and np.isfinite(value.imag):
            return value
    return candidate['c_final']


def search_temporal_roots_3d_shooting(
    baseflow,
    alpha,
    beta,
    Re,
    Ma,
    seed_list,
    *,
    Pr=0.72,
    gamma=1.4,
    y_max,
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    length_scale='delta_star',
    wall_bc='isothermal',
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
    root_atol=1e-6,
    polish_with_determinant=True,
):
    """Search unique exact first-order temporal roots from multiple complex seeds."""
    candidates = []

    for seed in seed_list:
        c_opt, sigma_min, sigma_converged, sigma_history = (
            solve_temporal_mode_3d_shooting_sigma_min(
                baseflow,
                alpha,
                beta,
                complex(seed),
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
                xatol=xatol,
                fatol=fatol,
                max_iter=max_iter,
            )
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
                Pr,
                gamma,
                y_max,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                wall_bc=wall_bc,
                method=method,
                n_steps=max(int(n_steps), 800),
            )

        candidates.append({
            'seed': complex(seed),
            'c_sigma_min': c_opt,
            'c_final': c_final,
            'omega_i': alpha * c_final.imag,
            'sigma_min': sigma_min,
            'sigma_min_converged': sigma_converged,
            'determinant_converged': det_converged,
            'sigma_min_history': sigma_history,
            'determinant_history': det_history,
        })

    candidates.sort(key=lambda item: (item['sigma_min'], -item['omega_i']))

    unique = []
    for candidate in candidates:
        candidate_root = _candidate_root_key(candidate)
        keep = True
        for existing in unique:
            existing_root = _candidate_root_key(existing)
            if abs(candidate_root - existing_root) < root_atol:
                keep = False
                break
        if keep:
            unique.append(candidate)

    return unique


def temporal_growth_scan_3d_shooting(
    baseflow,
    Re,
    Ma,
    alpha_range,
    *,
    initial_c,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    y_max,
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
    """Track one exact first-order temporal branch across an alpha scan.

    The root is continued from `initial_c` using the exact Appendix-A/B shooting
    system, so this scan can follow branch families missing from the reduced EVP.
    """
    alphas = np.asarray(alpha_range, dtype=float)
    if alphas.ndim != 1:
        raise ValueError('alpha_range must be one-dimensional')
    if beta_range is not None and psi_deg is not None:
        raise ValueError('Specify either beta_range or psi_deg, not both')

    if beta_range is None:
        if psi_deg is None:
            betas = np.zeros_like(alphas)
        else:
            betas = alphas * np.tan(np.deg2rad(float(psi_deg)))
    else:
        betas = _broadcast_float_scan_parameter(beta_range, len(alphas), 'beta_range')

    y_max_arr = _broadcast_float_scan_parameter(y_max, len(alphas), 'y_max')
    n_steps_arr = _broadcast_int_scan_parameter(n_steps, len(alphas), 'n_steps')

    case_sequence = []
    for alpha, beta, y_max_case, n_steps_case in zip(alphas, betas, y_max_arr, n_steps_arr):
        case_sequence.append({
            'alpha': alpha,
            'beta': beta,
            'Re': Re,
            'Ma': Ma,
            'Pr': Pr,
            'gamma': gamma,
            'y_max': y_max_case,
            'n_steps': n_steps_case,
            'baseflow': baseflow,
        })

    tracked = continue_temporal_mode_3d_shooting_sigma_min(
        case_sequence,
        initial_c=complex(initial_c),
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=int(np.max(n_steps_arr)),
        xatol=xatol,
        fatol=fatol,
        max_iter=max_iter,
        polish_with_determinant=polish_with_determinant,
    )

    omega_i = np.array([item['omega_i'] for item in tracked], dtype=float)
    c_vals = np.array([item['c_final'] for item in tracked], dtype=complex)
    sigma_min = np.array([item['sigma_min'] for item in tracked], dtype=float)

    return alphas, omega_i, c_vals, sigma_min, tracked


def temporal_growth_scan_3d_shooting_from_anchor(
    baseflow,
    Re,
    Ma,
    alpha_range,
    *,
    anchor_index,
    initial_c,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    y_max,
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
    """Track one exact first-order temporal branch away from an interior anchor."""
    alphas = np.asarray(alpha_range, dtype=float)
    if alphas.ndim != 1:
        raise ValueError('alpha_range must be one-dimensional')
    if not 0 <= int(anchor_index) < len(alphas):
        raise ValueError('anchor_index must lie inside alpha_range')

    anchor_index = int(anchor_index)

    if anchor_index > 0:
        lower_alphas = alphas[:anchor_index + 1][::-1]
        lower_betas = None
        if beta_range is not None:
            lower_betas = np.asarray(beta_range, dtype=float)[:anchor_index + 1][::-1]
        lower_scan = temporal_growth_scan_3d_shooting(
            baseflow,
            Re,
            Ma,
            lower_alphas,
            initial_c=initial_c,
            beta_range=lower_betas,
            psi_deg=psi_deg,
            Pr=Pr,
            gamma=gamma,
            y_max=_broadcast_float_scan_parameter(y_max, len(alphas), 'y_max')[:anchor_index + 1][::-1],
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            wall_bc=wall_bc,
            method=method,
            n_steps=_broadcast_int_scan_parameter(n_steps, len(alphas), 'n_steps')[:anchor_index + 1][::-1],
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
            polish_with_determinant=polish_with_determinant,
        )
        _, lower_omega_i, lower_c_vals, lower_sigma_min, lower_tracked = lower_scan
        lower_omega_i = lower_omega_i[::-1][:-1]
        lower_c_vals = lower_c_vals[::-1][:-1]
        lower_sigma_min = lower_sigma_min[::-1][:-1]
        lower_tracked = lower_tracked[::-1][:-1]
    else:
        lower_omega_i = np.array([], dtype=float)
        lower_c_vals = np.array([], dtype=complex)
        lower_sigma_min = np.array([], dtype=float)
        lower_tracked = []

    upper_scan = temporal_growth_scan_3d_shooting(
        baseflow,
        Re,
        Ma,
        alphas[anchor_index:],
        initial_c=initial_c,
        beta_range=None if beta_range is None else np.asarray(beta_range, dtype=float)[anchor_index:],
        psi_deg=psi_deg,
        Pr=Pr,
        gamma=gamma,
        y_max=_broadcast_float_scan_parameter(y_max, len(alphas), 'y_max')[anchor_index:],
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=_broadcast_int_scan_parameter(n_steps, len(alphas), 'n_steps')[anchor_index:],
        xatol=xatol,
        fatol=fatol,
        max_iter=max_iter,
        polish_with_determinant=polish_with_determinant,
    )
    _, upper_omega_i, upper_c_vals, upper_sigma_min, upper_tracked = upper_scan

    omega_i = np.concatenate([lower_omega_i, upper_omega_i])
    c_vals = np.concatenate([lower_c_vals, upper_c_vals])
    sigma_min = np.concatenate([lower_sigma_min, upper_sigma_min])
    tracked = lower_tracked + upper_tracked

    return alphas, omega_i, c_vals, sigma_min, tracked


def temporal_neutral_points_from_scan(alpha_values, omega_i_values, atol=1e-14):
    """Locate temporal neutral points by linear interpolation of a growth scan."""
    alphas = np.asarray(alpha_values, dtype=float)
    omega_i = np.asarray(omega_i_values, dtype=float)
    if alphas.ndim != 1 or omega_i.ndim != 1 or len(alphas) != len(omega_i):
        raise ValueError('alpha_values and omega_i_values must be 1D arrays of equal length')

    roots = []
    for i in range(len(alphas) - 1):
        a0 = alphas[i]
        a1 = alphas[i + 1]
        g0 = omega_i[i]
        g1 = omega_i[i + 1]
        if not np.isfinite(g0) or not np.isfinite(g1):
            continue

        if abs(g0) <= atol:
            roots.append(a0)
            continue

        if abs(g1) <= atol:
            roots.append(a1)
            continue

        if g0 * g1 > 0.0:
            continue

        root = a0 - g0 * (a1 - a0) / (g1 - g0)
        roots.append(root)

    if not roots:
        return np.array([], dtype=float)

    unique = [roots[0]]
    for root in roots[1:]:
        if abs(root - unique[-1]) > atol:
            unique.append(root)
    return np.asarray(unique, dtype=float)


def _resolve_beta_values(alpha_values, beta_range=None, psi_deg=None):
    """Resolve spanwise wavenumbers for a temporal scan."""
    alphas = np.asarray(alpha_values, dtype=float)
    if beta_range is not None and psi_deg is not None:
        raise ValueError('Specify either beta_range or psi_deg, not both')
    if beta_range is None:
        if psi_deg is None:
            return np.zeros_like(alphas)
        return alphas * np.tan(np.deg2rad(float(psi_deg)))
    return _broadcast_float_scan_parameter(beta_range, len(alphas), 'beta_range')


def _candidate_phase_speed(c_values, alpha, beta, metric):
    """Return the requested phase-speed metric for temporal candidates."""
    if metric == 'streamwise':
        return np.asarray(c_values).real
    if metric == 'wave':
        k = float(np.hypot(alpha, beta))
        if k <= 0.0:
            return np.asarray(c_values).real
        return np.asarray(c_values).real * (abs(float(alpha)) / k)
    raise ValueError("phase_speed_metric must be 'streamwise' or 'wave'")


def _physical_temporal_candidates(
    c_values,
    leakage,
    *,
    alpha=None,
    beta=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
):
    """Return the physical subset of a compressible temporal spectrum."""
    mask = (
        np.isfinite(c_values)
        & (c_values.real > 0.0)
        & (c_values.real < 1.2)
        & (np.abs(c_values.imag) < 0.3)
    )
    if phase_speed_bounds is not None:
        if alpha is None or beta is None:
            raise ValueError('alpha and beta are required when phase_speed_bounds is used')
        speed = _candidate_phase_speed(c_values, alpha, beta, phase_speed_metric)
        lo, hi = phase_speed_bounds
        mask &= (speed > float(lo)) & (speed < float(hi))
    return c_values[mask], leakage[mask]


def find_temporal_mode_anchor_3d_shooting(
    baseflow,
    Re,
    alpha,
    *,
    Ma,
    seed_list,
    beta=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    y_max,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    prefer_positive_growth=True,
    c_real_bounds=(0.0, 1.2),
    c_imag_abs_max=0.3,
    wall_bc='isothermal',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
    root_atol=1e-6,
    polish_with_determinant=True,
):
    """Search exact 3D temporal roots and select one anchor mode candidate."""
    alpha = float(alpha)
    beta_arr = _resolve_beta_values(
        np.array([alpha], dtype=float),
        beta_range=None if beta is None else np.array([beta], dtype=float),
        psi_deg=psi_deg,
    )
    beta_val = float(beta_arr[0])

    candidates = search_temporal_roots_3d_shooting(
        baseflow,
        alpha,
        beta_val,
        Re,
        Ma,
        seed_list,
        Pr=Pr,
        gamma=gamma,
        y_max=y_max,
        lambda_mu_ratio=lambda_mu_ratio,
        length_scale=length_scale,
        wall_bc=wall_bc,
        method=method,
        n_steps=n_steps,
        xatol=xatol,
        fatol=fatol,
        max_iter=max_iter,
        root_atol=root_atol,
        polish_with_determinant=polish_with_determinant,
    )

    if len(candidates) == 0:
        return {
            'alpha': alpha,
            'beta': beta_val,
            'c': np.array([], dtype=complex),
            'omega_i': np.array([], dtype=float),
            'sigma_min': np.array([], dtype=float),
            'phase_speed': np.array([], dtype=float),
            'selected_index': None,
            'selected_c': np.nan + 1j * np.nan,
            'selected_candidate': None,
            'candidates': [],
        }

    c_values = np.array([_candidate_root_key(item) for item in candidates], dtype=complex)
    sigma_min = np.array([item['sigma_min'] for item in candidates], dtype=float)
    omega_i = alpha * c_values.imag
    phase_speed = _candidate_phase_speed(c_values, alpha, beta_val, phase_speed_metric)

    mask_physical = (
        np.isfinite(c_values)
        & (c_values.real > float(c_real_bounds[0]))
        & (c_values.real < float(c_real_bounds[1]))
        & (np.abs(c_values.imag) < float(c_imag_abs_max))
    )
    mask_selected = mask_physical.copy()
    if phase_speed_bounds is not None:
        lo, hi = phase_speed_bounds
        mask_selected &= (phase_speed > float(lo)) & (phase_speed < float(hi))
        if not np.any(mask_selected):
            mask_selected = mask_physical

    if prefer_positive_growth and np.any(mask_selected & (omega_i > 0.0)):
        mask_selected &= omega_i > 0.0

    selected_index = None
    if np.any(mask_selected):
        idx_pool = np.flatnonzero(mask_selected)
        sort_idx = np.lexsort((sigma_min[idx_pool], -omega_i[idx_pool]))
        selected_index = int(idx_pool[sort_idx[0]])

    selected_candidate = None if selected_index is None else candidates[selected_index]
    selected_c = (
        np.nan + 1j * np.nan
        if selected_index is None
        else c_values[selected_index]
    )

    return {
        'alpha': alpha,
        'beta': beta_val,
        'c': c_values,
        'omega_i': omega_i,
        'sigma_min': sigma_min,
        'phase_speed': phase_speed,
        'selected_index': selected_index,
        'selected_c': selected_c,
        'selected_candidate': selected_candidate,
        'candidates': candidates,
    }


def temporal_growth_curve(
    baseflow,
    Re,
    alpha_range,
    *,
    Ma=None,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='auto',
    branch='most_unstable',
    initial_c=None,
    anchor_index=None,
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
):
    """Compute a temporal growth curve for incompressible or compressible flow."""
    alphas = np.asarray(alpha_range, dtype=float)
    if alphas.ndim != 1:
        raise ValueError('alpha_range must be one-dimensional')

    if Ma is None:
        alpha_out, omega_i, c_vals = temporal_growth_scan(
            baseflow, Re, alphas, N=N, y_max=40.0 if y_max is None else y_max,
        )
        return {
            'alpha': alpha_out,
            'beta': np.zeros_like(alpha_out),
            'omega_i': omega_i,
            'c': c_vals,
            'leakage': np.full_like(alpha_out, np.nan, dtype=float),
            'method': 'orr_sommerfeld',
            'tracked': None,
        }

    if method == 'auto':
        method = 'reduced' if initial_c is None else 'shooting'

    betas = _resolve_beta_values(alphas, beta_range=beta_range, psi_deg=psi_deg)

    if method == 'shooting':
        if initial_c is None:
            raise ValueError('initial_c is required for method="shooting"')
        alpha_out, omega_i, c_vals, sigma_min, tracked = temporal_growth_scan_3d_shooting(
            baseflow,
            Re,
            Ma,
            alphas,
            initial_c=initial_c,
            beta_range=betas,
            Pr=Pr,
            gamma=gamma,
            y_max=12.0 if y_max is None else y_max,
            lambda_mu_ratio=lambda_mu_ratio,
            length_scale=length_scale,
            wall_bc=wall_bc,
            method='qr',
            n_steps=n_steps,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )
        return {
            'alpha': alpha_out,
            'beta': betas,
            'omega_i': omega_i,
            'c': c_vals,
            'leakage': sigma_min,
            'method': 'shooting',
            'tracked': tracked,
        }

    if method == 'shooting_anchor':
        if initial_c is None or anchor_index is None:
            raise ValueError(
                'initial_c and anchor_index are required for method="shooting_anchor"'
            )
        alpha_out, omega_i, c_vals, sigma_min, tracked = (
            temporal_growth_scan_3d_shooting_from_anchor(
                baseflow,
                Re,
                Ma,
                alphas,
                anchor_index=anchor_index,
                initial_c=initial_c,
                beta_range=betas,
                Pr=Pr,
                gamma=gamma,
                y_max=12.0 if y_max is None else y_max,
                lambda_mu_ratio=lambda_mu_ratio,
                length_scale=length_scale,
                wall_bc=wall_bc,
                method='qr',
                n_steps=n_steps,
                xatol=xatol,
                fatol=fatol,
                max_iter=max_iter,
            )
        )
        return {
            'alpha': alpha_out,
            'beta': betas,
            'omega_i': omega_i,
            'c': c_vals,
            'leakage': sigma_min,
            'method': 'shooting_anchor',
            'tracked': tracked,
        }

    if method != 'reduced':
        raise ValueError(f'Unknown temporal growth method: {method}')

    omega_i = np.full(len(alphas), np.nan, dtype=float)
    c_vals = np.full(len(alphas), np.nan + 1j * np.nan, dtype=complex)
    leakage_vals = np.full(len(alphas), np.nan, dtype=float)
    c_seed = None if initial_c is None else complex(initial_c)

    for i, (alpha, beta) in enumerate(zip(alphas, betas)):
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
            wall_bc=wall_bc,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            freestream_leakage_tol=freestream_leakage_tol,
            return_leakage=True,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
        )
        c_phys, leakage_phys = _physical_temporal_candidates(
            c_all,
            leakage,
            alpha=alpha,
            beta=beta,
            phase_speed_bounds=phase_speed_bounds,
            phase_speed_metric=phase_speed_metric,
        )
        if len(c_phys) == 0:
            continue

        if branch == 'tracked' and c_seed is not None and np.isfinite(c_seed):
            idx = int(np.argmin(np.abs(c_phys - c_seed)))
        else:
            idx = int(np.argmax(alpha * c_phys.imag))

        c_sel = c_phys[idx]
        c_seed = c_sel
        c_vals[i] = c_sel
        omega_i[i] = alpha * c_sel.imag
        leakage_vals[i] = float(leakage_phys[idx])

    return {
        'alpha': alphas,
        'beta': betas,
        'omega_i': omega_i,
        'c': c_vals,
        'leakage': leakage_vals,
        'method': 'reduced',
        'tracked': None,
    }


def trace_temporal_neutral_curve(
    baseflow_builder,
    Ma,
    Re_range,
    alpha_range,
    *,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='auto',
    branch='most_unstable',
    initial_c=None,
    anchor_index=None,
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
):
    """Trace temporal lower and upper neutral branches across Reynolds number."""
    Re_arr = np.asarray(Re_range, dtype=float)
    lower_alpha = np.full(len(Re_arr), np.nan, dtype=float)
    upper_alpha = np.full(len(Re_arr), np.nan, dtype=float)
    scans = []
    c_seed = initial_c

    for i, Re in enumerate(Re_arr):
        baseflow = baseflow_builder(Re)
        scan = temporal_growth_curve(
            baseflow,
            Re,
            alpha_range,
            Ma=Ma,
            beta_range=beta_range,
            psi_deg=psi_deg,
            Pr=Pr,
            gamma=gamma,
            N=N,
            y_max=y_max,
            wall_bc=wall_bc,
            method=method,
            branch=branch,
            initial_c=c_seed,
            anchor_index=anchor_index,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            freestream_leakage_tol=freestream_leakage_tol,
            phase_speed_bounds=phase_speed_bounds,
            phase_speed_metric=phase_speed_metric,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
            n_steps=n_steps,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )
        neutrals = temporal_neutral_points_from_scan(scan['alpha'], scan['omega_i'])
        if len(neutrals) >= 1:
            lower_alpha[i] = neutrals[0]
        if len(neutrals) >= 2:
            upper_alpha[i] = neutrals[-1]

        finite_c = scan['c'][np.isfinite(scan['c'])]
        if anchor_index is not None and 0 <= int(anchor_index) < len(scan['c']):
            anchor_c = scan['c'][int(anchor_index)]
            if np.isfinite(anchor_c):
                c_seed = anchor_c
            elif len(finite_c) > 0:
                c_seed = finite_c[min(len(finite_c) - 1, int(anchor_index))]
        elif len(finite_c) > 0:
            c_seed = finite_c[np.nanargmax(scan['omega_i'][np.isfinite(scan['c'])])]

        scans.append({
            'Re': Re,
            'scan': scan,
            'neutral_points': neutrals,
        })

    return {
        'Re': Re_arr,
        'lower_alpha': lower_alpha,
        'upper_alpha': upper_alpha,
        'scans': scans,
    }


def trace_temporal_neutral_curve_shooting(
    baseflow_builder,
    Ma,
    Re_range,
    alpha_range,
    *,
    anchor_index,
    seed_list=None,
    initial_c=None,
    anchor_re_index=0,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    y_max=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    prefer_positive_growth=True,
    wall_bc='isothermal',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    method='qr',
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
    root_atol=1e-6,
    polish_with_determinant=True,
):
    """Trace temporal neutral branches using an exact-shooting anchor root."""
    Re_arr = np.asarray(Re_range, dtype=float)
    alpha_arr = np.asarray(alpha_range, dtype=float)
    if Re_arr.ndim != 1 or alpha_arr.ndim != 1:
        raise ValueError('Re_range and alpha_range must be one-dimensional')
    if not 0 <= int(anchor_index) < len(alpha_arr):
        raise ValueError('anchor_index must lie inside alpha_range')
    if not 0 <= int(anchor_re_index) < len(Re_arr):
        raise ValueError('anchor_re_index must lie inside Re_range')

    anchor_index = int(anchor_index)
    anchor_re_index = int(anchor_re_index)
    beta_arr = _resolve_beta_values(alpha_arr, beta_range=beta_range, psi_deg=psi_deg)

    anchor_search = None
    anchor_c = initial_c
    if anchor_c is None:
        if seed_list is None:
            raise ValueError('seed_list is required when initial_c is not provided')
        anchor_re = float(Re_arr[anchor_re_index])
        anchor_baseflow = baseflow_builder(anchor_re)
        anchor_search = find_temporal_mode_anchor_3d_shooting(
            anchor_baseflow,
            anchor_re,
            alpha_arr[anchor_index],
            Ma=Ma,
            seed_list=seed_list,
            beta=beta_arr[anchor_index],
            Pr=Pr,
            gamma=gamma,
            y_max=y_max,
            phase_speed_bounds=phase_speed_bounds,
            phase_speed_metric=phase_speed_metric,
            prefer_positive_growth=prefer_positive_growth,
            wall_bc=wall_bc,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
            method=method,
            n_steps=n_steps,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
            root_atol=root_atol,
            polish_with_determinant=polish_with_determinant,
        )
        anchor_c = anchor_search['selected_c']
        if not np.isfinite(anchor_c):
            raise ValueError('No physical shooting anchor root was found')

    common = dict(
        beta_range=beta_arr,
        Pr=Pr,
        gamma=gamma,
        y_max=y_max,
        wall_bc=wall_bc,
        method='shooting_anchor',
        anchor_index=anchor_index,
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio,
        n_steps=n_steps,
        xatol=xatol,
        fatol=fatol,
        max_iter=max_iter,
    )

    forward = trace_temporal_neutral_curve(
        baseflow_builder,
        Ma,
        Re_arr[anchor_re_index:],
        alpha_arr,
        initial_c=anchor_c,
        **common,
    )

    if anchor_re_index == 0:
        result = forward
    else:
        backward = trace_temporal_neutral_curve(
            baseflow_builder,
            Ma,
            Re_arr[:anchor_re_index + 1][::-1],
            alpha_arr,
            initial_c=anchor_c,
            **common,
        )
        result = {
            'Re': np.concatenate([backward['Re'][::-1][:-1], forward['Re']]),
            'lower_alpha': np.concatenate([
                backward['lower_alpha'][::-1][:-1],
                forward['lower_alpha'],
            ]),
            'upper_alpha': np.concatenate([
                backward['upper_alpha'][::-1][:-1],
                forward['upper_alpha'],
            ]),
            'scans': list(reversed(backward['scans']))[:-1] + forward['scans'],
        }

    result['anchor_index'] = anchor_index
    result['anchor_re_index'] = anchor_re_index
    result['anchor_c'] = anchor_c
    result['anchor_search'] = anchor_search
    result['beta'] = beta_arr
    return result


def temporal_growth_map(
    baseflow_builder,
    Ma,
    Re_range,
    alpha_range,
    *,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='auto',
    branch='most_unstable',
    initial_c=None,
    anchor_index=None,
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
):
    """Compute a temporal growth map over Reynolds number and wavenumber."""
    Re_arr = np.asarray(Re_range, dtype=float)
    alpha_arr = np.asarray(alpha_range, dtype=float)
    if Re_arr.ndim != 1 or alpha_arr.ndim != 1:
        raise ValueError('Re_range and alpha_range must be one-dimensional')

    beta_arr = _resolve_beta_values(alpha_arr, beta_range=beta_range, psi_deg=psi_deg)
    omega_i_map = np.full((len(Re_arr), len(alpha_arr)), np.nan, dtype=float)
    c_map = np.full((len(Re_arr), len(alpha_arr)), np.nan + 1j * np.nan, dtype=complex)
    leakage_map = np.full((len(Re_arr), len(alpha_arr)), np.nan, dtype=float)
    lower_alpha = np.full(len(Re_arr), np.nan, dtype=float)
    upper_alpha = np.full(len(Re_arr), np.nan, dtype=float)
    scans = []
    c_seed = initial_c

    for i, Re in enumerate(Re_arr):
        baseflow = baseflow_builder(Re)
        scan = temporal_growth_curve(
            baseflow,
            Re,
            alpha_arr,
            Ma=Ma,
            beta_range=beta_arr,
            Pr=Pr,
            gamma=gamma,
            N=N,
            y_max=y_max,
            wall_bc=wall_bc,
            method=method,
            branch=branch,
            initial_c=c_seed,
            anchor_index=anchor_index,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            freestream_leakage_tol=freestream_leakage_tol,
            phase_speed_bounds=phase_speed_bounds,
            phase_speed_metric=phase_speed_metric,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
            n_steps=n_steps,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )

        omega_i_map[i, :] = scan['omega_i']
        c_map[i, :] = scan['c']
        leakage_map[i, :] = scan['leakage']
        neutrals = temporal_neutral_points_from_scan(scan['alpha'], scan['omega_i'])
        if len(neutrals) >= 1:
            lower_alpha[i] = neutrals[0]
        if len(neutrals) >= 2:
            upper_alpha[i] = neutrals[-1]

        finite_mask = np.isfinite(scan['c'])
        if anchor_index is not None and 0 <= int(anchor_index) < len(scan['c']):
            anchor_c = scan['c'][int(anchor_index)]
            if np.isfinite(anchor_c):
                c_seed = anchor_c
            elif np.any(finite_mask):
                c_seed = scan['c'][np.where(finite_mask)[0][0]]
        elif np.any(finite_mask):
            finite_idx = np.where(finite_mask)[0]
            finite_growth = scan['omega_i'][finite_mask]
            if np.any(np.isfinite(finite_growth)):
                c_seed = scan['c'][finite_idx[int(np.nanargmax(finite_growth))]]
            else:
                c_seed = scan['c'][finite_idx[0]]

        scans.append({
            'Re': Re,
            'scan': scan,
            'neutral_points': neutrals,
        })

    return {
        'Re': Re_arr,
        'alpha': alpha_arr,
        'beta': beta_arr,
        'omega_i': omega_i_map,
        'c': c_map,
        'c_i': c_map.imag,
        'leakage': leakage_map,
        'lower_alpha': lower_alpha,
        'upper_alpha': upper_alpha,
        'scans': scans,
    }


def critical_reynolds_from_growth_series(Re_values, growth_values, atol=1e-14):
    """Estimate the critical Reynolds number from a 1D maximum-growth series."""
    Re_arr = np.asarray(Re_values, dtype=float)
    growth = np.asarray(growth_values, dtype=float)
    if Re_arr.ndim != 1 or growth.ndim != 1 or len(Re_arr) != len(growth):
        raise ValueError('Re_values and growth_values must be 1D arrays of equal length')

    finite = np.isfinite(growth)
    if not np.any(finite):
        return np.nan

    idx_finite = np.where(finite)[0]
    first_idx = int(idx_finite[0])
    if growth[first_idx] > atol:
        return float(Re_arr[first_idx])

    for i in range(len(Re_arr) - 1):
        g0 = growth[i]
        g1 = growth[i + 1]
        if not np.isfinite(g0) or not np.isfinite(g1):
            continue
        if abs(g0) <= atol:
            return float(Re_arr[i])
        if abs(g1) <= atol:
            return float(Re_arr[i + 1])
        if g0 <= 0.0 and g1 > 0.0:
            return float(
                Re_arr[i] - g0 * (Re_arr[i + 1] - Re_arr[i]) / (g1 - g0)
            )

    return np.nan


def spatial_growth_curve(
    baseflow,
    Re,
    Ma,
    omega_range,
    *,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='refined',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Compute a spatial growth curve sigma(omega)."""
    omega_out, sigma, alpha_r = frequency_sweep(
        baseflow,
        Re,
        Ma,
        omega_range,
        Pr=Pr,
        gamma=gamma,
        N=N,
        y_max=y_max,
        wall_bc=wall_bc,
        method=method,
        lambda_mu_ratio=lambda_mu_ratio,
    )
    return {
        'omega': omega_out,
        'sigma': sigma,
        'alpha_r': alpha_r,
        'alpha': alpha_r - 1j * sigma,
        'method': method,
    }


def spatial_growth_map(
    baseflow_builder,
    Ma,
    Re_range,
    omega_range,
    *,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='refined',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Compute a spatial growth map over Reynolds number and frequency."""
    Re_arr = np.asarray(Re_range, dtype=float)
    omega_arr = np.asarray(omega_range, dtype=float)
    if Re_arr.ndim != 1 or omega_arr.ndim != 1:
        raise ValueError('Re_range and omega_range must be one-dimensional')

    sigma_map = np.full((len(Re_arr), len(omega_arr)), np.nan, dtype=float)
    alpha_r_map = np.full((len(Re_arr), len(omega_arr)), np.nan, dtype=float)
    lower_omega = np.full(len(Re_arr), np.nan, dtype=float)
    upper_omega = np.full(len(Re_arr), np.nan, dtype=float)
    scans = []

    for i, Re in enumerate(Re_arr):
        baseflow = baseflow_builder(Re)
        scan = spatial_growth_curve(
            baseflow,
            Re,
            Ma,
            omega_arr,
            Pr=Pr,
            gamma=gamma,
            N=N,
            y_max=y_max,
            wall_bc=wall_bc,
            method=method,
            lambda_mu_ratio=lambda_mu_ratio,
        )

        sigma_map[i, :] = scan['sigma']
        alpha_r_map[i, :] = scan['alpha_r']

        neutrals = temporal_neutral_points_from_scan(scan['omega'], scan['sigma'])
        if len(neutrals) >= 1:
            lower_omega[i] = neutrals[0]
        if len(neutrals) >= 2:
            upper_omega[i] = neutrals[-1]

        scans.append({
            'Re': Re,
            'scan': scan,
            'neutral_points': neutrals,
        })

    return {
        'Re': Re_arr,
        'omega': omega_arr,
        'sigma': sigma_map,
        'alpha_r': alpha_r_map,
        'alpha': alpha_r_map - 1j * sigma_map,
        'lower_omega': lower_omega,
        'upper_omega': upper_omega,
        'scans': scans,
    }


def trace_spatial_neutral_curve(
    baseflow_builder,
    Ma,
    Re_range,
    omega_range,
    *,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='refined',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
):
    """Trace spatial neutral branches by locating sigma=0 across Reynolds number."""
    Re_arr, omega_arr, sigma_map = neutral_curve(
        baseflow_builder,
        Ma,
        Re_range,
        omega_range,
        Pr=Pr,
        gamma=gamma,
        N=N,
        y_max=y_max,
        wall_bc=wall_bc,
        method=method,
        lambda_mu_ratio=lambda_mu_ratio,
    )

    lower_omega = np.full(len(Re_arr), np.nan, dtype=float)
    upper_omega = np.full(len(Re_arr), np.nan, dtype=float)
    for i, sigma_row in enumerate(sigma_map):
        neutrals = temporal_neutral_points_from_scan(omega_arr, sigma_row)
        if len(neutrals) >= 1:
            lower_omega[i] = neutrals[0]
        if len(neutrals) >= 2:
            upper_omega[i] = neutrals[-1]

    return {
        'Re': Re_arr,
        'omega': omega_arr,
        'sigma_map': sigma_map,
        'lower_omega': lower_omega,
        'upper_omega': upper_omega,
    }


def critical_reynolds_curve(
    baseflow_builder,
    Ma,
    Re_range,
    alpha_range,
    *,
    beta_range=None,
    psi_deg=None,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='auto',
    branch='most_unstable',
    initial_c=None,
    anchor_index=None,
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
):
    """Return the temporal neutral curve plus its approximate critical point."""
    neutral = trace_temporal_neutral_curve(
        baseflow_builder,
        Ma,
        Re_range,
        alpha_range,
        beta_range=beta_range,
        psi_deg=psi_deg,
        Pr=Pr,
        gamma=gamma,
        N=N,
        y_max=y_max,
        wall_bc=wall_bc,
        method=method,
        branch=branch,
        initial_c=initial_c,
        anchor_index=anchor_index,
        include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
        freestream_leakage_tol=freestream_leakage_tol,
        phase_speed_bounds=phase_speed_bounds,
        phase_speed_metric=phase_speed_metric,
        length_scale=length_scale,
        lambda_mu_ratio=lambda_mu_ratio,
        n_steps=n_steps,
        xatol=xatol,
        fatol=fatol,
        max_iter=max_iter,
    )
    gap = neutral['upper_alpha'] - neutral['lower_alpha']
    valid = np.isfinite(gap)
    if np.any(valid):
        idx = np.nanargmin(np.where(valid, np.abs(gap), np.nan))
        Re_crit = float(neutral['Re'][idx])
        alpha_crit = 0.5 * (neutral['lower_alpha'][idx] + neutral['upper_alpha'][idx])
    else:
        Re_crit = np.nan
        alpha_crit = np.nan

    neutral['Re_crit'] = Re_crit
    neutral['alpha_crit'] = alpha_crit
    return neutral


def most_unstable_wave_angle(
    baseflow_builder,
    Ma,
    psi_range,
    Re_range,
    alpha_range,
    *,
    Pr=0.72,
    gamma=1.4,
    N=128,
    y_max=None,
    wall_bc='isothermal',
    method='auto',
    branch='most_unstable',
    initial_c=None,
    anchor_index=None,
    include_spanwise_dissipation_coupling=True,
    freestream_leakage_tol=None,
    phase_speed_bounds=None,
    phase_speed_metric='streamwise',
    length_scale='delta_star',
    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
    n_steps=600,
    xatol=1e-7,
    fatol=1e-9,
    max_iter=120,
):
    """Scan wave angle and return the smallest critical Reynolds number found."""
    psi_arr = np.asarray(psi_range, dtype=float)
    Re_crit = np.full(len(psi_arr), np.nan, dtype=float)
    alpha_crit = np.full(len(psi_arr), np.nan, dtype=float)
    curves = []

    for i, psi_deg in enumerate(psi_arr):
        curve = critical_reynolds_curve(
            baseflow_builder,
            Ma,
            Re_range,
            alpha_range,
            psi_deg=psi_deg,
            Pr=Pr,
            gamma=gamma,
            N=N,
            y_max=y_max,
            wall_bc=wall_bc,
            method=method,
            branch=branch,
            initial_c=initial_c,
            anchor_index=anchor_index,
            include_spanwise_dissipation_coupling=include_spanwise_dissipation_coupling,
            freestream_leakage_tol=freestream_leakage_tol,
            phase_speed_bounds=phase_speed_bounds,
            phase_speed_metric=phase_speed_metric,
            length_scale=length_scale,
            lambda_mu_ratio=lambda_mu_ratio,
            n_steps=n_steps,
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )
        Re_crit[i] = curve['Re_crit']
        alpha_crit[i] = curve['alpha_crit']
        curves.append(curve)

    if np.any(np.isfinite(Re_crit)):
        idx = int(np.nanargmin(Re_crit))
        psi_opt = float(psi_arr[idx])
    else:
        idx = None
        psi_opt = np.nan

    return {
        'psi_deg': psi_arr,
        'Re_crit': Re_crit,
        'alpha_crit': alpha_crit,
        'psi_opt': psi_opt,
        'curves': curves,
        'index_opt': idx,
    }


def find_critical_Re(baseflow, alpha_range=(0.1, 0.5), Re_range=(300, 1200),
                     N=128, y_max=40.0, tol=1):
    """Find the Reynolds number where the maximum temporal growth crosses zero."""
    from scipy.optimize import brentq

    alpha_arr = np.linspace(alpha_range[0], alpha_range[1], 30)

    def max_growth(Re):
        _, omega_i, _ = temporal_growth_scan(baseflow, Re, alpha_arr, N, y_max)
        return np.nanmax(omega_i)

    try:
        Re_crit = brentq(max_growth, Re_range[0], Re_range[1], xtol=tol)
    except ValueError:
        Re_arr = np.linspace(Re_range[0], Re_range[1], 20)
        gi = np.array([max_growth(R) for R in Re_arr])
        crossings = np.where(np.diff(np.sign(gi)))[0]
        if len(crossings) > 0:
            i = crossings[0]
            Re_crit = np.interp(0, gi[i:i + 2], Re_arr[i:i + 2])
        else:
            Re_crit = np.nan
    return Re_crit


def _solve_spatial_mode(baseflow, omega, Re, Ma, Pr, gamma, N, y_max,
                        wall_bc, method,
                        lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Return one spatial eigenvalue using the requested solution path."""
    if method == 'refined':
        alpha, converged = solve_spatial_from_temporal(
            baseflow, omega, Re, Ma, Pr, gamma, N=N, y_max=y_max,
            lambda_mu_ratio=lambda_mu_ratio)
        if converged and np.isfinite(alpha):
            return alpha
        return np.nan + 1j * np.nan

    if method == 'qep':
        alphas, _, _ = solve_spatial(
            baseflow, omega, Re, Ma, Pr, gamma, N, y_max, wall_bc=wall_bc,
            lambda_mu_ratio=lambda_mu_ratio)
        if len(alphas) == 0:
            return np.array([], dtype=complex)
        return alphas

    raise ValueError(f"Unknown spatial solve method: {method}")


def frequency_sweep(baseflow, Re, Ma, omega_range, Pr=0.72, gamma=1.4,
                    N=128, y_max=None, wall_bc='isothermal',
                    method='refined',
                    lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Compute spatial growth rate sigma = -alpha_i versus frequency.

    Parameters
    ----------
    method : {"refined", "qep"}
        `refined` uses a temporal solve, a Gaster-based initial guess, and
        a complex-alpha refinement. `qep` uses the older companion-form
        quadratic eigenvalue problem directly.
    """
    omegas = np.asarray(omega_range)
    sigma = np.zeros(len(omegas))
    alpha_r = np.zeros(len(omegas))
    alpha_tracked = None

    for i, om in enumerate(omegas):
        result = _solve_spatial_mode(
            baseflow, om, Re, Ma, Pr, gamma, N, y_max, wall_bc, method,
            lambda_mu_ratio=lambda_mu_ratio)

        if method == 'qep':
            alphas = result
            if len(alphas) == 0:
                a = np.nan + 1j * np.nan
            elif alpha_tracked is not None:
                a = track_mode(alphas, alpha_tracked)[0]
            else:
                a = alphas[np.argmin(alphas.imag)]
        else:
            a = result

        if not np.isfinite(a):
            sigma[i] = np.nan
            alpha_r[i] = np.nan
            continue

        alpha_tracked = a
        sigma[i] = -a.imag
        alpha_r[i] = a.real

    return omegas, sigma, alpha_r


def neutral_curve(baseflow_func, Ma, Re_range, omega_range,
                  Pr=0.72, gamma=1.4, N=128, y_max=None,
                  wall_bc='isothermal', method='refined',
                  lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Compute a neutral map in (Re, omega) space."""
    Re_arr = np.asarray(Re_range)
    omega_arr = np.asarray(omega_range)
    sigma_map = np.zeros((len(Re_arr), len(omega_arr)))

    for i, Re in enumerate(Re_arr):
        bf = baseflow_func(Re)
        _, sig, _ = frequency_sweep(
            bf, Re, Ma, omega_arr, Pr, gamma, N, y_max, wall_bc, method,
            lambda_mu_ratio=lambda_mu_ratio)
        sigma_map[i, :] = sig

    return Re_arr, omega_arr, sigma_map


def nfactor(baseflow_func, Ma, omega, Re_range, Pr=0.72, gamma=1.4,
            N=128, y_max=None, wall_bc='isothermal', method='refined',
            lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Compute an N-factor curve by integrating spatial growth."""
    Re_arr = np.asarray(Re_range)
    sigma = np.zeros(len(Re_arr))
    alpha_tracked = None

    for i, Re in enumerate(Re_arr):
        bf = baseflow_func(Re)
        result = _solve_spatial_mode(
            bf, omega, Re, Ma, Pr, gamma, N, y_max, wall_bc, method,
            lambda_mu_ratio=lambda_mu_ratio)

        if method == 'qep':
            alphas = result
            if len(alphas) == 0:
                a = np.nan + 1j * np.nan
            elif alpha_tracked is not None:
                a = track_mode(alphas, alpha_tracked)[0]
            else:
                a = alphas[np.argmin(alphas.imag)]
        else:
            a = result

        if not np.isfinite(a):
            sigma[i] = np.nan
            continue

        alpha_tracked = a
        sigma[i] = -a.imag

    N_vals = np.zeros(len(Re_arr))
    sigma_pos = np.maximum(sigma, 0)

    for i in range(1, len(Re_arr)):
        dRe = Re_arr[i] - Re_arr[i - 1]
        N_vals[i] = N_vals[i - 1] + 0.5 * (sigma_pos[i] + sigma_pos[i - 1]) * dRe

    return Re_arr, N_vals, sigma
