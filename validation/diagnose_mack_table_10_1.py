"""
Diagnostic comparison against Mack Table 10.1.

This is not a pass/fail validation. It reports the current sixth- and
eighth-order oblique-wave temporal amplification rates at Mack's tabulated
points so solver regressions and gaps are visible in one place.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.mack_conditions import make_mack_profile
from lst.solver import solve_temporal_compressible_3d


TABLE_10_1_CASES = [
    (1.3, 500, 0.075, 45, 0.883e-3, 0.824e-3),
    (1.3, 1500, 0.060, 45, 1.467e-3, 1.445e-3),
    (1.6, 500, 0.070, 55, 0.974e-3, 0.874e-3),
    (1.6, 1500, 0.050, 55, 1.384e-3, 1.346e-3),
    (2.2, 500, 0.055, 60, 1.198e-3, 1.066e-3),
    (2.2, 800, 0.045, 60, 1.391e-3, 1.300e-3),
    (2.2, 1500, 0.035, 60, 1.325e-3, 1.273e-3),
    (4.5, 500, 0.045, 60, 1.117e-3, 1.039e-3),
    (4.5, 1500, 0.050, 60, 1.641e-3, 1.613e-3),
    (5.8, 500, 0.050, 55, 0.790e-3, 0.736e-3),
    (5.8, 1500, 0.060, 55, 1.403e-3, 1.384e-3),
    (10.0, 1500, 0.040, 55, 0.444e-3, 0.434e-3),
]


def leading_growth_rate(profile, alpha_l, beta_l, re_l, ma, *, include_coupling):
    """Return the largest positive omega_i found at a Mack Table 10.1 point."""
    c_all, _, _ = solve_temporal_compressible_3d(
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
        include_spanwise_dissipation_coupling=include_coupling,
    )
    mask = (c_all.real > 0.0) & (c_all.real < 1.2) & (np.abs(c_all.imag) < 0.3)
    c_all = c_all[mask]
    if len(c_all) == 0:
        return np.nan, np.nan

    omega_i = alpha_l * c_all.imag
    idx = np.argmax(omega_i)
    return omega_i[idx], c_all[idx].real


def main():
    print('=' * 84)
    print('DIAGNOSTIC: MACK TABLE 10.1 OBLIQUE-WAVE AMPLIFICATION RATES')
    print('=' * 84)
    print()
    print("All inputs are interpreted on Mack's L* scale.")
    print('The reported values are the largest positive omega_i found in the spectrum.')
    print()
    print(f'{"M1":>4s} {"R":>6s} {"a":>6s} {"psi":>6s} '
          f'{"6th calc":>12s} {"6th tab":>12s} {"8th calc":>12s} {"8th tab":>12s}')
    print('-' * 84)

    for ma, re_l, alpha_l, psi_deg, sixth_tab, eighth_tab in TABLE_10_1_CASES:
        psi = np.deg2rad(psi_deg)
        beta_l = alpha_l * np.tan(psi)
        profile = make_mack_profile(ma)

        sixth_calc, _ = leading_growth_rate(
            profile, alpha_l, beta_l, re_l, ma, include_coupling=False)
        eighth_calc, _ = leading_growth_rate(
            profile, alpha_l, beta_l, re_l, ma, include_coupling=True)

        print(f'{ma:4.1f} {re_l:6.0f} {alpha_l:6.3f} {psi_deg:6.0f} '
              f'{sixth_calc:12.6e} {sixth_tab:12.6e} '
              f'{eighth_calc:12.6e} {eighth_tab:12.6e}')

    print()
    print('Interpretation: this table is a diagnostic target for the current')
    print('oblique-wave solver, not a pass/fail benchmark yet.')


if __name__ == '__main__':
    main()
