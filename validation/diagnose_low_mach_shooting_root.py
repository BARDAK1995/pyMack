"""Search low-Mach Chapter 10 roots on the exact first-order shooting system."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.mack_conditions import make_mack_profile
from lst.mack_shooting import (
    continue_temporal_mode_3d_shooting_sigma_min,
    solve_temporal_mode_3d_shooting,
    solve_temporal_mode_3d_shooting_sigma_min,
)
from lst.solver import (
    refine_temporal_compressible_3d_asymptotic,
    solve_temporal_compressible_3d,
)


CASES = [
    {
        'Ma': 1.3,
        'Re': 500.0,
        'alpha': 0.075,
        'psi_deg': 45.0,
        'omega_table': 0.824e-3,
        'initial_c': 0.50 + 0.004j,
        'y_max_values': [12.0, 18.0, 22.0, 26.0],
        'n_steps_values': [500, 900, 1100, 1500],
    },
    {
        'Ma': 1.6,
        'Re': 500.0,
        'alpha': 0.070,
        'psi_deg': 55.0,
        'omega_table': 0.874e-3,
        'initial_c': 0.54 + 0.0001j,
        'y_max_values': [12.0, 18.0, 24.0],
        'n_steps_values': [500, 900, 1300],
    },
]

SEEDS = [
    0.6100864888364268 + 0.12417131483672231j,  # reduced-EVP leading amplified root
    0.55 + 0.008j,
    0.50 + 0.004j,
    0.70 + 0.020j,
]


def unique_roots(results, atol=1e-6):
    roots = []
    for c_val, sigma_min, converged in results:
        keep = True
        for existing, _, _ in roots:
            if abs(c_val - existing) < atol:
                keep = False
                break
        if keep:
            roots.append((c_val, sigma_min, converged))
    roots.sort(key=lambda item: (item[1], abs(item[0].imag)))
    return roots


def main():
    print('=' * 92)
    print('DIAGNOSTIC: LOW-MACH SHOOTING ROOT SEARCH')
    print('=' * 92)
    print()

    for case in CASES:
        ma = case['Ma']
        re_l = case['Re']
        alpha_l = case['alpha']
        psi_deg = case['psi_deg']
        beta_l = alpha_l * np.tan(np.deg2rad(psi_deg))
        omega_table = case['omega_table']
        profile = make_mack_profile(ma)

        print(
            f'Case: M={ma:.1f}, R={re_l:.0f}, alpha={alpha_l:.3f}, psi={psi_deg:.0f}, '
            f'Mack table omega_i={omega_table:.6e}'
        )

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
        if len(c_phys) > 0:
            print('Reduced-EVP physical candidates at y_max=12:')
            print(f'{"c":>28s} {"omega_i":>12s} {"leakage":>12s}')
            for c_val, leak in zip(c_phys[:6], leakage[:6]):
                print(
                    f'{str(c_val):>28s} {alpha_l * c_val.imag:12.6e} {float(leak):12.6e}'
                )
            print()

        if ma == 1.3:
            results = []
            for seed in SEEDS:
                c_opt, sigma_min, converged, _ = solve_temporal_mode_3d_shooting_sigma_min(
                    profile,
                    alpha_l,
                    beta_l,
                    seed,
                    re_l,
                    ma,
                    0.72,
                    1.4,
                    12.0,
                    length_scale='L_star',
                    method='qr',
                    n_steps=500,
                )
                c_det, det_converged, _ = solve_temporal_mode_3d_shooting(
                    profile,
                    alpha_l,
                    beta_l,
                    c_opt,
                    re_l,
                    ma,
                    0.72,
                    1.4,
                    12.0,
                    length_scale='L_star',
                    method='qr',
                    n_steps=1000,
                )
                results.append((c_det, sigma_min, converged and det_converged))

            roots = unique_roots(results)

            print('Exact first-order shooting roots found from singular-value search at y_max=12:')
            print(f'{"root c":>28s} {"omega_i":>12s} {"sigma_min":>12s} {"conv":>8s}')
            for c_val, sigma_min, converged in roots:
                print(
                    f'{str(c_val):>28s} {alpha_l * c_val.imag:12.6e} '
                    f'{sigma_min:12.6e} {str(converged):>8s}'
                )
            print()

        continuation_cases = []
        for y_max, n_steps in zip(case['y_max_values'], case['n_steps_values']):
            continuation_cases.append({
                'alpha': alpha_l,
                'beta': beta_l,
                'Re': re_l,
                'Ma': ma,
                'y_max': y_max,
                'n_steps': n_steps,
            })
        tracked = continue_temporal_mode_3d_shooting_sigma_min(
            continuation_cases,
            baseflow_builder=lambda data: make_mack_profile(data['Ma']),
            initial_c=case['initial_c'],
            length_scale='L_star',
            method='qr',
        )
        print('Exact first-order shooting root continued in y_max:')
        print(f'{"y_max":>8s} {"n_steps":>8s} {"root c":>28s} {"omega_i":>12s} {"sigma_min":>12s}')
        for item in tracked:
            case_data = item['case']
            print(
                f'{case_data["y_max"]:8.1f} {case_data["n_steps"]:8d} '
                f'{str(item["c_final"]):>28s} {item["omega_i"]:12.6e} '
                f'{item["sigma_min"]:12.6e}'
            )
        last = tracked[-1]
        last_case = last['case']
        c_ref, _, _, converged, leak_ref = refine_temporal_compressible_3d_asymptotic(
            profile,
            alpha_l,
            beta_l,
            re_l,
            ma,
            0.72,
            1.4,
            c_guess=last['c_final'],
            N=max(120, int(round(6 * last_case['y_max']))),
            y_max=last_case['y_max'],
            wall_bc='adiabatic',
            length_scale='L_star',
        )
        print()
        print(
            'Reduced asymptotic EVP seeded with the last shooting root: '
            f'c={c_ref}, omega_i={alpha_l * c_ref.imag:.6e}, '
            f'converged={converged}, leakage={leak_ref:.6e}'
        )
        print()

    print('Interpretation:')
    print('  If the exact shooting root moves toward Mack as y_max increases while the reduced')
    print('  EVP spectrum does not acquire the same branch, the finite-domain reduced EVP is')
    print('  still missing a physically relevant low-Mach mode family.')


if __name__ == '__main__':
    main()
