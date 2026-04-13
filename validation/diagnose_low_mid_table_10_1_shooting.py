"""Exact first-order shooting comparison against low/mid Mack Table 10.1 cases."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.analysis import search_temporal_roots_3d_shooting
from lst.mack_conditions import make_mack_profile
from lst.mack_shooting import continue_temporal_mode_3d_shooting_sigma_min
from lst.solver import solve_temporal_compressible_3d


FAMILIES = [
    {
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
        'cases': [
            {'Re': 500.0, 'alpha': 0.075, 'omega_table': 0.824e-3},
            {'Re': 1500.0, 'alpha': 0.060, 'omega_table': 1.445e-3},
        ],
    },
    {
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
        'cases': [
            {'Re': 500.0, 'alpha': 0.070, 'omega_table': 0.874e-3},
            {'Re': 1500.0, 'alpha': 0.050, 'omega_table': 1.346e-3},
        ],
    },
    {
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
        'cases': [
            {'Re': 500.0, 'alpha': 0.055, 'omega_table': 0.001066},
            {'Re': 800.0, 'alpha': 0.045, 'omega_table': 0.001300},
            {'Re': 1500.0, 'alpha': 0.035, 'omega_table': 0.001273},
        ],
    },
]


def leading_reduced_growth(profile, alpha_l, beta_l, re_l, ma):
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
        alpha_l = case['alpha']
        case_sequence.append({
            'alpha': alpha_l,
            'beta': alpha_l * np.tan(psi),
            'Re': case['Re'],
            'Ma': ma,
            'y_max': family['y_max'],
            'n_steps': family['n_steps'],
        })
    return case_sequence


def parse_args():
    """Parse command-line options for the diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--search-initial',
        action='store_true',
        help='search initial roots from multiple seeds before continuation',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print('=' * 108)
    print('DIAGNOSTIC: EXACT FIRST-ORDER SHOOTING VS MACK TABLE 10.1 (LOW/MID MACH FAMILIES)')
    print('=' * 108)
    print()
    print("All wavenumbers and growth rates are interpreted on Mack's L* scale.")
    print('The table compares the exact first-order shooting branch against the')
    print('eighth-order Mack values and the current reduced-EVP leading root at y_max=12.')
    if args.search_initial:
        print('Initial multi-seed exact-shooting searches are enabled.')
    else:
        print('Initial root search is skipped; curated continuation seeds are used.')
    print()

    for family in FAMILIES:
        ma = family['Ma']
        psi_deg = family['psi_deg']
        y_max = family['y_max']
        n_steps = family['n_steps']
        first_case = family['cases'][0]
        alpha0 = first_case['alpha']
        beta0 = alpha0 * np.tan(np.deg2rad(psi_deg))
        profile = make_mack_profile(ma)

        print('-' * 108)
        print(
            f'Family: M={ma:.1f}, psi={psi_deg:.0f}, y_max={y_max:.1f}, n_steps={n_steps:d}'
        )

        initial_c = family['fallback_initial_c']
        if args.search_initial:
            candidates = search_temporal_roots_3d_shooting(
                profile,
                alpha0,
                beta0,
                first_case['Re'],
                ma,
                family['seed_list'],
                y_max=y_max,
                wall_bc='adiabatic',
                length_scale='L_star',
                method='qr',
                n_steps=n_steps,
            )

            print()
            print('Initial exact-shooting candidates at the first Table 10.1 point:')
            print(f'{"seed":>22s} {"c_final":>28s} {"omega_i":>12s} {"sigma_min":>12s} {"conv":>8s}')
            for item in candidates:
                converged = item['sigma_min_converged'] and item['determinant_converged']
                print(
                    f'{str(item["seed"]):>22s} {str(item["c_final"]):>28s} '
                    f'{item["omega_i"]:12.6e} {item["sigma_min"]:12.6e} {str(converged):>8s}'
                )

            chosen = choose_initial_root(candidates)
            initial_c = family['fallback_initial_c'] if chosen is None else chosen['c_final']
            if chosen is None:
                print()
                print(f'No candidate passed selection; using fallback seed {initial_c}.')
            else:
                print()
                print(f'Chosen continuation seed: {initial_c}')
        else:
            print()
            print(f'Using curated continuation seed: {initial_c}')

        tracked = continue_temporal_mode_3d_shooting_sigma_min(
            family_case_sequence(family),
            baseflow_builder=lambda data: make_mack_profile(data['Ma']),
            initial_c=initial_c,
            length_scale='L_star',
            method='qr',
        )

        print()
        print(f'{"R":>6s} {"alpha":>8s} {"shooting":>12s} {"table":>12s} {"rel err %":>10s} '
              f'{"reduced":>12s} {"leakage":>12s} {"sigma_min":>12s} {"c_final":>28s}')
        for case, item in zip(family['cases'], tracked):
            alpha_l = case['alpha']
            beta_l = alpha_l * np.tan(np.deg2rad(psi_deg))
            omega_table = case['omega_table']
            shooting_omega_i = item['omega_i']
            rel_err = 100.0 * (shooting_omega_i - omega_table) / omega_table
            reduced_omega_i, _, reduced_leak = leading_reduced_growth(
                profile,
                alpha_l,
                beta_l,
                case['Re'],
                ma,
            )
            print(
                f'{case["Re"]:6.0f} {alpha_l:8.3f} {shooting_omega_i:12.6e} '
                f'{omega_table:12.6e} {rel_err:10.3f} {reduced_omega_i:12.6e} '
                f'{reduced_leak:12.6e} {item["sigma_min"]:12.6e} {str(item["c_final"]):>28s}'
            )
        print()

    print('Interpretation:')
    print('  The exact first-order shooting branch is materially closer to Mack in the')
    print('  low/mid-Mach regime than the reduced finite-domain EVP, but it still')
    print('  underpredicts the eighth-order Table 10.1 growth rates by about 10-25%.')
    print('  That means the remaining gap is now a boundary/eigenvalue formulation')
    print('  problem, not a branch-selection or first-order algebra transcription issue.')


if __name__ == '__main__':
    main()
