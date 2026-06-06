"""Exact first-order shooting comparison against low/mid Mack Table 10.1 cases."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.mack_table_10_1 import (
    DEFAULT_TABLE_10_1_CONDITION,
    DEFAULT_TABLE_10_1_WALL_BC,
    choose_initial_root,
    continue_family_order,
    family_case_sequence,
    json_scalar,
    leading_reduced_growth,
    load_low_mid_table_10_1_families,
    order_specs_from_label,
    search_family_initial_roots,
)
from pymack.mack_conditions import make_mack_profile


def parse_args():
    """Parse command-line options for the diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--search-initial',
        action='store_true',
        help='search initial roots from multiple seeds before continuation',
    )
    parser.add_argument('--Ma', dest='mach', action='append', type=float)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument(
        '--wall-bc',
        choices=('adiabatic', 'isothermal'),
        default=DEFAULT_TABLE_10_1_WALL_BC,
        help='thermal perturbation wall boundary condition',
    )
    parser.add_argument(
        '--condition',
        default=DEFAULT_TABLE_10_1_CONDITION,
        choices=('table_10_1', 'table_11_1', 'wind_tunnel', 'figure'),
        help='Mack mean-flow temperature condition set',
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=None,
        help='override the curated QR integration step count for faster diagnostics',
    )
    parser.add_argument(
        '--order',
        choices=('eighth', 'sixth', 'both'),
        default='eighth',
        help='which Mack system to evaluate with exact shooting',
    )
    parser.add_argument(
        '--a68-scale',
        type=float,
        default=1.0,
        help='diagnostic multiplier for Appendix-A a68 in the eighth-order path',
    )
    parser.add_argument(
        '--skip-reduced',
        action='store_true',
        help='skip reduced-EVP comparison columns for faster exact-shooting checks',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='emit machine-readable JSON after the diagnostic table',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    families = load_low_mid_table_10_1_families(args.mach, condition=args.condition)
    output_rows = []

    print('=' * 108)
    print('DIAGNOSTIC: EXACT FIRST-ORDER SHOOTING VS MACK TABLE 10.1 (LOW/MID MACH FAMILIES)')
    print('=' * 108)
    print()
    print("All wavenumbers and growth rates are interpreted on Mack's L* scale.")
    print('The table compares the exact first-order shooting branch against the selected')
    print('Mack Table 10.1 order and the current reduced-EVP leading root at y_max=12.')
    print(f'Mack mean-flow condition set: {args.condition}.')
    print(f'Thermal perturbation wall condition: {args.wall_bc}.')
    print(f'Exact-shooting system order: {args.order}.')
    if args.a68_scale != 1.0:
        print(f'Diagnostic Appendix-A a68 multiplier: {args.a68_scale:g}.')
    if args.search_initial:
        print('Initial multi-seed exact-shooting searches are enabled.')
    else:
        print('Initial root search is skipped; curated continuation seeds are used.')
    if args.skip_reduced:
        print('Reduced-EVP comparison is skipped for faster targeted execution.')
    print()

    for family in families:
        if args.limit is not None:
            family['cases'] = family['cases'][:max(0, int(args.limit))]
        if args.n_steps is not None:
            family['n_steps'] = int(args.n_steps)

        ma = family['Ma']
        condition = family['condition']
        psi_deg = family['psi_deg']
        y_max = family['y_max']
        n_steps = family['n_steps']
        profile = make_mack_profile(ma, condition=condition)

        print('-' * 108)
        print(
            f'Family: M={ma:.1f}, psi={psi_deg:.0f}, y_max={y_max:.1f}, n_steps={n_steps:d}'
        )

        fallback_initial_c = family['fallback_initial_c']
        if not args.search_initial:
            print()
            print(f'Using curated continuation seed: {fallback_initial_c}')

        for order_label, include_coupling in order_specs_from_label(args.order):
            initial_c = fallback_initial_c
            if args.search_initial:
                candidates = search_family_initial_roots(
                    family,
                    order_label,
                    include_coupling,
                    condition=condition,
                    wall_bc=args.wall_bc,
                    a68_scale=args.a68_scale,
                )

                print()
                print(f'Initial exact-shooting candidates for {order_label}-order system:')
                print(f'{"seed":>22s} {"c_final":>28s} {"omega_i":>12s} {"sigma_min":>12s} {"conv":>8s}')
                for item in candidates:
                    converged = item['sigma_min_converged'] and item['determinant_converged']
                    print(
                        f'{str(item["seed"]):>22s} {str(item["c_final"]):>28s} '
                        f'{item["omega_i"]:12.6e} {item["sigma_min"]:12.6e} {str(converged):>8s}'
                    )

                chosen = choose_initial_root(candidates)
                initial_c = fallback_initial_c if chosen is None else chosen['c_final']
                if chosen is None:
                    print()
                    print(f'No candidate passed selection; using fallback seed {initial_c}.')
                else:
                    print()
                    print(f'Chosen continuation seed: {initial_c}')

            tracked = continue_family_order(
                family,
                order_label,
                include_coupling,
                condition=condition,
                wall_bc=args.wall_bc,
                initial_c=initial_c,
                a68_scale=args.a68_scale,
            )

            print()
            print(f'Exact-shooting {order_label}-order system')
            print(f'{"R":>6s} {"alpha":>8s} {"shooting":>12s} {"table":>12s} {"rel err %":>10s} '
                  f'{"reduced":>12s} {"leakage":>12s} {"sigma_min":>12s} {"c_final":>28s}')
            for case, item in zip(family['cases'], tracked):
                alpha_l = case.alpha_L
                beta_l = alpha_l * np.tan(np.deg2rad(psi_deg))
                omega_table = case.omega_i_6th if order_label == 'sixth' else case.omega_i_8th
                shooting_omega_i = item['omega_i']
                rel_err = 100.0 * (shooting_omega_i - omega_table) / omega_table
                if args.skip_reduced:
                    reduced_omega_i = np.nan
                    reduced_leak = np.nan
                else:
                    reduced_omega_i, _, reduced_leak = leading_reduced_growth(
                        profile,
                        alpha_l,
                        beta_l,
                        case.Re_L,
                        ma,
                        wall_bc=args.wall_bc,
                    )
                print(
                    f'{case.Re_L:6.0f} {alpha_l:8.3f} {shooting_omega_i:12.6e} '
                    f'{omega_table:12.6e} {rel_err:10.3f} {reduced_omega_i:12.6e} '
                    f'{reduced_leak:12.6e} {item["sigma_min"]:12.6e} {str(item["c_final"]):>28s}'
                )
                output_rows.append({
                    'Ma': float(ma),
                    'Re_L': float(case.Re_L),
                    'alpha_L': float(alpha_l),
                    'psi_deg': float(psi_deg),
                    'condition': condition,
                    'wall_bc': args.wall_bc,
                    'system_order': order_label,
                    'formulation': (
                        'primary_6x6_appendix_a' if order_label == 'sixth'
                        else 'full_8x8_appendix_a'
                    ),
                    'include_spanwise_dissipation_coupling': include_coupling,
                    'spanwise_dissipation_coupling_scale': (
                        json_scalar(args.a68_scale) if order_label == 'eighth'
                        else None
                    ),
                    'shooting_omega_i': json_scalar(shooting_omega_i),
                    'omega_i_table': json_scalar(omega_table),
                    'shooting_rel_error': json_scalar(rel_err / 100.0),
                    'reduced_omega_i': json_scalar(reduced_omega_i),
                    'reduced_leakage': json_scalar(reduced_leak),
                    'sigma_min': json_scalar(item['sigma_min']),
                    'c_final_real': json_scalar(item['c_final'].real),
                    'c_final_imag': json_scalar(item['c_final'].imag),
                })
        print()

    print('Interpretation:')
    print('  The exact first-order shooting branch is materially closer to Mack in the')
    print('  low/mid-Mach regime than the reduced finite-domain EVP. The current')
    print('  best Table 10.1 match uses the Table 11.1 mean-flow temperature schedule')
    print('  with an isothermal thermal-disturbance wall condition; the colder')
    print('  table_10_1 schedule is retained only as a sensitivity check.')
    if args.json:
        print()
        print(json.dumps(output_rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
