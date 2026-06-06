"""Mack Table 10.1 exact-shooting reproduction helpers."""

from __future__ import annotations

import numpy as np

from .analysis import search_temporal_roots_3d_shooting, search_temporal_roots_6_shooting
from .mack_conditions import make_mack_profile
from .mack_shooting import (
    continue_temporal_mode_3d_shooting_sigma_min,
    continue_temporal_mode_6_shooting_sigma_min,
)
from .reference_data import select_mack_table_10_1_cases
from .solver import solve_temporal_compressible_3d


DEFAULT_TABLE_10_1_WALL_BC = 'isothermal'
DEFAULT_TABLE_10_1_CONDITION = 'table_11_1'


TABLE_10_1_FAMILY_SETTINGS = {
    1.3: {
        'Ma': 1.3,
        'psi_deg': 45.0,
        'y_max': 26.0,
        'n_steps': 1500,
        'seed_list': [
            0.6100864888364268 + 0.12417131483672231j,
            0.55 + 0.008j,
            0.50 + 0.004j,
            0.70 + 0.020j,
        ],
        'fallback_initial_c': 0.4483542634413574 + 0.009881298294162798j,
    },
    1.6: {
        'Ma': 1.6,
        'psi_deg': 55.0,
        'y_max': 24.0,
        'n_steps': 1300,
        'seed_list': [
            0.62 + 0.12j,
            0.54 + 0.0001j,
            0.50 + 0.004j,
            0.70 + 0.020j,
        ],
        'fallback_initial_c': 0.49545888112115044 + 0.010025718938172355j,
    },
    2.2: {
        'Ma': 2.2,
        'psi_deg': 60.0,
        'y_max': 30.0,
        'n_steps': 1500,
        'seed_list': [
            0.60 + 0.015j,
            0.56 + 0.010j,
            0.70 + 0.020j,
            0.45 + 0.030j,
        ],
        'fallback_initial_c': 0.5597129082856475 + 0.013806412477744867j,
    },
}


def json_scalar(value):
    """Return a strict-JSON numeric scalar, or None for non-finite values."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def leading_reduced_growth(profile, alpha_l, beta_l, re_l, ma, *, wall_bc):
    """Return the leading positive reduced-EVP temporal growth at y_max=12."""
    c_all, _, _, leakage = solve_temporal_compressible_3d(
        profile,
        alpha_l,
        beta_l,
        re_l,
        ma,
        0.72,
        1.4,
        N=90,
        y_max=12.0,
        length_scale='L_star',
        wall_bc=wall_bc,
        return_leakage=True,
    )
    mask = (c_all.real > 0.0) & (c_all.real < 1.2) & (np.abs(c_all.imag) < 0.3)
    c_phys = c_all[mask]
    leakage = leakage[mask]
    if len(c_phys) == 0:
        return np.nan, np.nan, np.nan

    omega_i = alpha_l * c_phys.imag
    idx = np.argmax(omega_i)
    return omega_i[idx], c_phys[idx], float(leakage[idx])


def choose_initial_root(candidates):
    """Choose the amplified exact-shooting candidate for continuation."""
    if not candidates:
        return None

    converged = [
        item for item in candidates
        if item['sigma_min_converged'] and item['determinant_converged']
    ]
    if converged:
        candidates = converged

    positive = [item for item in candidates if item['omega_i'] > 0.0]
    if positive:
        candidates = positive

    return max(candidates, key=lambda item: item['omega_i'])


def family_case_sequence(family):
    """Build the continuation cases for one Table 10.1 family."""
    ma = family['Ma']
    psi = np.deg2rad(family['psi_deg'])
    case_sequence = []
    for case in family['cases']:
        alpha_l = case.alpha_L
        case_sequence.append({
            'alpha': alpha_l,
            'beta': alpha_l * np.tan(psi),
            'Re': case.Re_L,
            'Ma': ma,
            'y_max': family['y_max'],
            'n_steps': family['n_steps'],
        })
    return case_sequence


def load_low_mid_table_10_1_families(
    mach_values=None,
    condition=DEFAULT_TABLE_10_1_CONDITION,
):
    """Load low/mid Mach Table 10.1 families and attach shooting settings."""
    if mach_values is None:
        mach_values = sorted(TABLE_10_1_FAMILY_SETTINGS)
    families = []
    for ma in mach_values:
        if float(ma) not in TABLE_10_1_FAMILY_SETTINGS:
            raise ValueError(f'No curated low/mid Table 10.1 shooting settings for Ma={ma}')
        settings = dict(TABLE_10_1_FAMILY_SETTINGS[float(ma)])
        cases = select_mack_table_10_1_cases(Ma=float(ma))
        settings['cases'] = cases
        settings['condition'] = condition
        families.append(settings)
    return families


def order_specs_from_label(order):
    """Return diagnostic order labels and full-system coupling flags."""
    if order == 'both':
        return [('sixth', False), ('eighth', True)]
    if order == 'sixth':
        return [('sixth', False)]
    if order == 'eighth':
        return [('eighth', True)]
    raise ValueError("order must be 'sixth', 'eighth', or 'both'")


def search_family_initial_roots(
    family,
    order_label,
    include_coupling,
    *,
    condition,
    wall_bc,
    a68_scale=1.0,
):
    """Search initial exact-shooting roots for the first row of one family."""
    first_case = family['cases'][0]
    alpha0 = first_case.alpha_L
    beta0 = alpha0 * np.tan(np.deg2rad(family['psi_deg']))
    ma = family['Ma']
    profile = make_mack_profile(ma, condition=condition)

    if order_label == 'sixth':
        return search_temporal_roots_6_shooting(
            profile,
            alpha0,
            beta0,
            first_case.Re_L,
            ma,
            family['seed_list'],
            y_max=family['y_max'],
            length_scale='L_star',
            method='qr',
            n_steps=family['n_steps'],
            wall_bc=wall_bc,
        )

    return search_temporal_roots_3d_shooting(
        profile,
        alpha0,
        beta0,
        first_case.Re_L,
        ma,
        family['seed_list'],
        y_max=family['y_max'],
        length_scale='L_star',
        method='qr',
        n_steps=family['n_steps'],
        include_spanwise_dissipation_coupling=include_coupling,
        spanwise_dissipation_coupling_scale=a68_scale,
        wall_bc=wall_bc,
    )


def continue_family_order(
    family,
    order_label,
    include_coupling,
    *,
    condition,
    wall_bc,
    initial_c=None,
    a68_scale=1.0,
):
    """Continue one Table 10.1 family/order with the exact shooting solver."""
    if initial_c is None:
        initial_c = family['fallback_initial_c']
    common_kwargs = dict(
        case_sequence=family_case_sequence(family),
        baseflow_builder=lambda data, condition=condition: make_mack_profile(
            data['Ma'],
            condition=condition,
        ),
        initial_c=initial_c,
        length_scale='L_star',
        wall_bc=wall_bc,
        method='qr',
    )
    if order_label == 'sixth':
        return continue_temporal_mode_6_shooting_sigma_min(**common_kwargs)
    if order_label == 'eighth':
        return continue_temporal_mode_3d_shooting_sigma_min(
            **common_kwargs,
            include_spanwise_dissipation_coupling=include_coupling,
            spanwise_dissipation_coupling_scale=a68_scale,
        )
    raise ValueError("order_label must be 'sixth' or 'eighth'")


def evaluate_table_10_1_exact_shooting(
    mach_values=None,
    *,
    condition=DEFAULT_TABLE_10_1_CONDITION,
    wall_bc=DEFAULT_TABLE_10_1_WALL_BC,
    n_steps=None,
    order='both',
    limit=None,
    skip_reduced=True,
    a68_scale=1.0,
):
    """Return machine-readable exact-shooting errors for Table 10.1 rows."""
    output_rows = []
    families = load_low_mid_table_10_1_families(mach_values, condition=condition)

    for family in families:
        if limit is not None:
            family['cases'] = family['cases'][:max(0, int(limit))]
        if n_steps is not None:
            family['n_steps'] = int(n_steps)

        ma = family['Ma']
        psi_deg = family['psi_deg']
        profile = make_mack_profile(ma, condition=family['condition'])

        for order_label, include_coupling in order_specs_from_label(order):
            tracked = continue_family_order(
                family,
                order_label,
                include_coupling,
                condition=family['condition'],
                wall_bc=wall_bc,
                a68_scale=a68_scale,
            )

            for case, item in zip(family['cases'], tracked):
                alpha_l = case.alpha_L
                beta_l = alpha_l * np.tan(np.deg2rad(psi_deg))
                omega_table = case.omega_i_6th if order_label == 'sixth' else case.omega_i_8th
                shooting_omega_i = item['omega_i']
                rel_err = (shooting_omega_i - omega_table) / omega_table
                if skip_reduced:
                    reduced_omega_i = np.nan
                    reduced_leak = np.nan
                else:
                    reduced_omega_i, _, reduced_leak = leading_reduced_growth(
                        profile,
                        alpha_l,
                        beta_l,
                        case.Re_L,
                        ma,
                        wall_bc=wall_bc,
                    )

                output_rows.append({
                    'Ma': float(ma),
                    'Re_L': float(case.Re_L),
                    'alpha_L': float(alpha_l),
                    'psi_deg': float(psi_deg),
                    'condition': family['condition'],
                    'wall_bc': wall_bc,
                    'system_order': order_label,
                    'formulation': (
                        'primary_6x6_appendix_a' if order_label == 'sixth'
                        else 'full_8x8_appendix_a'
                    ),
                    'include_spanwise_dissipation_coupling': include_coupling,
                    'spanwise_dissipation_coupling_scale': (
                        json_scalar(a68_scale) if order_label == 'eighth'
                        else None
                    ),
                    'shooting_omega_i': json_scalar(shooting_omega_i),
                    'omega_i_table': json_scalar(omega_table),
                    'shooting_rel_error': json_scalar(rel_err),
                    'reduced_omega_i': json_scalar(reduced_omega_i),
                    'reduced_leakage': json_scalar(reduced_leak),
                    'sigma_min': json_scalar(item['sigma_min']),
                    'c_final_real': json_scalar(item['c_final'].real),
                    'c_final_imag': json_scalar(item['c_final'].imag),
                })

    return output_rows
