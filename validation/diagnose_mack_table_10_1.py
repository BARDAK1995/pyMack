"""
Diagnostic comparison against Mack Table 10.1.

This is not a pass/fail validation. It reports the current sixth- and
eighth-order oblique-wave temporal amplification rates at Mack's tabulated
points so solver regressions and gaps are visible in one place.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.mack_conditions import make_mack_profile
from pymack.reference_data import select_mack_table_10_1_cases
from pymack.solver import solve_temporal_compressible_3d


DEFAULT_TABLE_10_1_CONDITION = 'table_10_1'
DEFAULT_TABLE_10_1_WALL_BC = 'adiabatic'


def json_scalar(value):
    """Return a strict-JSON numeric scalar, or None for non-finite values."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def leading_growth_rate(
    profile,
    alpha_l,
    beta_l,
    re_l,
    ma,
    *,
    include_coupling,
    wall_bc=DEFAULT_TABLE_10_1_WALL_BC,
    N=90,
    y_max=12.0,
):
    """Return the largest positive omega_i found at a Mack Table 10.1 point."""
    c_all, _, _ = solve_temporal_compressible_3d(
        profile,
        alpha_l,
        beta_l,
        re_l,
        ma,
        0.72,
        1.4,
        N=N,
        y_max=y_max,
        wall_bc=wall_bc,
        length_scale='L_star',
        include_spanwise_dissipation_coupling=include_coupling,
    )
    mask = (c_all.real > 0.0) & (c_all.real < 1.2) & (np.abs(c_all.imag) < 0.3)
    c_all = c_all[mask]
    if len(c_all) == 0:
        return np.nan, np.nan

    omega_i = alpha_l * c_all.imag
    idx = np.argmax(omega_i)
    return omega_i[idx], c_all[idx].real


def evaluate_reduced_table_10_1_cases(
    cases,
    *,
    N=90,
    y_max=12.0,
    condition=DEFAULT_TABLE_10_1_CONDITION,
    wall_bc=DEFAULT_TABLE_10_1_WALL_BC,
):
    """Evaluate reduced-EVP growth at selected Mack Table 10.1 cases."""
    rows = []
    for case in cases:
        profile = make_mack_profile(case.Ma, condition=condition)

        sixth_calc, _ = leading_growth_rate(
            profile,
            case.alpha_L,
            case.beta_L,
            case.Re_L,
            case.Ma,
            include_coupling=False,
            wall_bc=wall_bc,
            N=N,
            y_max=y_max,
        )
        eighth_calc, _ = leading_growth_rate(
            profile,
            case.alpha_L,
            case.beta_L,
            case.Re_L,
            case.Ma,
            include_coupling=True,
            wall_bc=wall_bc,
            N=N,
            y_max=y_max,
        )
        rows.append({
            'Ma': json_scalar(case.Ma),
            'Re_L': json_scalar(case.Re_L),
            'alpha_L': json_scalar(case.alpha_L),
            'psi_deg': json_scalar(case.psi_deg),
            'condition': condition,
            'wall_bc': wall_bc,
            'sixth_calc': json_scalar(sixth_calc),
            'sixth_tab': json_scalar(case.omega_i_6th),
            'sixth_rel_error': json_scalar(
                (sixth_calc - case.omega_i_6th) / case.omega_i_6th
            ),
            'eighth_calc': json_scalar(eighth_calc),
            'eighth_tab': json_scalar(case.omega_i_8th),
            'eighth_rel_error': json_scalar(
                (eighth_calc - case.omega_i_8th) / case.omega_i_8th
            ),
        })
    return rows


def parse_args():
    """Parse command-line options for the diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ma', dest='mach', action='append', type=float)
    parser.add_argument('--Re', dest='re_l', action='append', type=float)
    parser.add_argument('--psi', dest='psi_deg', action='append', type=float)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--N', type=int, default=90)
    parser.add_argument('--y-max', type=float, default=12.0)
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
        '--json',
        action='store_true',
        help='emit machine-readable JSON instead of the formatted table',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = select_mack_table_10_1_cases(
        Ma=args.mach,
        Re_L=args.re_l,
        psi_deg=args.psi_deg,
    )
    if args.limit is not None:
        cases = cases[:max(0, int(args.limit))]

    rows = evaluate_reduced_table_10_1_cases(
        cases,
        N=args.N,
        y_max=args.y_max,
        condition=args.condition,
        wall_bc=args.wall_bc,
    )

    if args.json:
        print(json.dumps(rows, indent=2, allow_nan=False))
        return

    print('=' * 104)
    print('DIAGNOSTIC: MACK TABLE 10.1 OBLIQUE-WAVE AMPLIFICATION RATES')
    print('=' * 104)
    print()
    print("All inputs are interpreted on Mack's L* scale.")
    print(f'Mack mean-flow condition set: {args.condition}.')
    print(f'Thermal perturbation wall condition: {args.wall_bc}.')
    print('The reported values are the largest positive omega_i found in the reduced spectrum.')
    print('Reference values are loaded from reference_data/mack/table_10_1_oblique_growth.csv.')
    print()
    print(f'{"M1":>4s} {"R":>6s} {"a":>6s} {"psi":>6s} '
          f'{"6th calc":>12s} {"6th tab":>12s} {"err %":>9s} '
          f'{"8th calc":>12s} {"8th tab":>12s} {"err %":>9s}')
    print('-' * 104)

    for row in rows:
        print(
            f'{row["Ma"]:4.1f} {row["Re_L"]:6.0f} {row["alpha_L"]:6.3f} '
            f'{row["psi_deg"]:6.0f} {row["sixth_calc"]:12.6e} '
            f'{row["sixth_tab"]:12.6e} {100.0 * row["sixth_rel_error"]:9.2f} '
            f'{row["eighth_calc"]:12.6e} {row["eighth_tab"]:12.6e} '
            f'{100.0 * row["eighth_rel_error"]:9.2f}'
        )

    print()
    print('Interpretation: this table is a diagnostic target for the current')
    print('oblique-wave solver, not a pass/fail benchmark yet.')


if __name__ == '__main__':
    main()
