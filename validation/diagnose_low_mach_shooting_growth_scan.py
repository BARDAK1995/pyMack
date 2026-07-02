"""Bidirectional exact-shooting growth scan and neutral-point extraction."""


import numpy as np


from pymack.analysis import (
    temporal_growth_scan_3d_shooting_from_anchor,
    temporal_neutral_points_from_scan,
)
from pymack.mack_conditions import make_mack_profile


CASE = {
    'Ma': 1.3,
    'Re': 1500.0,
    'psi_deg': 45.0,
    'alpha_values': np.array([0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090]),
    'anchor_index': 5,
    'initial_c': 0.38889286865032374 + 0.02050770393584937j,
    'y_max': 26.0,
    'n_steps': 1500,
}


def main():
    ma = CASE['Ma']
    re_l = CASE['Re']
    psi_deg = CASE['psi_deg']
    alpha_values = CASE['alpha_values']
    anchor_index = CASE['anchor_index']
    initial_c = CASE['initial_c']

    print('=' * 96)
    print('DIAGNOSTIC: LOW-MACH EXACT-SHOOTING GROWTH SCAN')
    print('=' * 96)
    print()
    print(
        f'Case: M={ma:.1f}, R={re_l:.0f}, psi={psi_deg:.0f}, '
        f'anchor alpha={alpha_values[anchor_index]:.3f}'
    )
    print("All quantities are interpreted on Mack's L* scale.")
    print()

    profile = make_mack_profile(ma)
    alphas, omega_i, c_vals, sigma_min, _ = temporal_growth_scan_3d_shooting_from_anchor(
        profile,
        re_l,
        ma,
        alpha_values,
        anchor_index=anchor_index,
        initial_c=initial_c,
        psi_deg=psi_deg,
        y_max=CASE['y_max'],
        n_steps=CASE['n_steps'],
        length_scale='L_star',
        method='qr',
    )

    neutrals = temporal_neutral_points_from_scan(alphas, omega_i)

    print(f'{"alpha":>8s} {"omega_i":>12s} {"sigma_min":>12s} {"c_final":>28s}')
    for alpha_l, omega_i_l, sigma_min_l, c_val in zip(alphas, omega_i, sigma_min, c_vals):
        print(
            f'{alpha_l:8.3f} {omega_i_l:12.6e} {sigma_min_l:12.6e} {str(c_val):>28s}'
        )

    print()
    if len(neutrals) == 0:
        print('No temporal neutral points were found on this scan window.')
    else:
        print('Interpolated temporal neutral points:')
        for alpha_n in neutrals:
            print(f'  alpha_n = {alpha_n:.6f}')


if __name__ == '__main__':
    main()
