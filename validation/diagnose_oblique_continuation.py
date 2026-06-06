"""Track representative Mack Table 10.1 branches by alpha/beta continuation.

This diagnostic answers a specific question: if the oblique-wave spectrum is
seeded on a physically plausible first-mode family and then continued in
alpha and obliqueness, does it approach Mack's tabulated amplification rate or
does it remain on the same oversized branch found by direct max-growth picks?
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.mack_conditions import make_mack_profile
from pymack.solver import (
    continue_temporal_mode_3d,
    refine_temporal_compressible_3d_asymptotic,
    temporal_candidate_spectrum_3d,
)


REPRESENTATIVE_CASES = [
    {
        'Ma': 1.3,
        'Re': 500.0,
        'alpha': 0.075,
        'psi_deg': 45.0,
        'omega_table': 0.824e-3,
        'alpha_start': 0.010,
        'c_seed': 1.0 + 0.0j,
    },
    {
        'Ma': 4.5,
        'Re': 1500.0,
        'alpha': 0.050,
        'psi_deg': 60.0,
        'omega_table': 1.613e-3,
        'alpha_start': 0.010,
        'c_seed': 1.0 + 0.0j,
    },
    {
        'Ma': 10.0,
        'Re': 1500.0,
        'alpha': 0.040,
        'psi_deg': 55.0,
        'omega_table': 0.434e-3,
        'alpha_start': 0.010,
        'c_seed': 0.20 + 0.0j,
    },
]


def alpha_path(alpha_start, alpha_stop, n_steps=8):
    values = np.linspace(alpha_start, alpha_stop, n_steps)
    return [float(v) for v in values]


def beta_path(beta_stop, n_steps=6):
    values = np.linspace(0.0, beta_stop, n_steps)
    return [float(v) for v in values]


def tracked_growth(case):
    ma = case['Ma']
    re_l = case['Re']
    alpha_target = case['alpha']
    beta_target = alpha_target * math.tan(math.radians(case['psi_deg']))
    profile_builder = lambda data: make_mack_profile(data['Ma'])

    alpha_cases = [
        {'alpha': alpha, 'beta': 0.0, 'Re': re_l, 'Ma': ma}
        for alpha in alpha_path(case['alpha_start'], alpha_target)
    ]
    alpha_track = continue_temporal_mode_3d(
        alpha_cases,
        baseflow_builder=profile_builder,
        Pr=0.72,
        gamma=1.4,
        N=90,
        y_max=12.0,
        wall_bc='adiabatic',
        length_scale='L_star',
        initial_c=case['c_seed'],
        prefer_positive_growth=True,
        proximity_weight=2.0,
        leakage_weight=0.25,
        qr_weight=0.0,
        growth_weight=0.0,
        damping_penalty=1.0,
        c_real_bounds=(0.0, 1.2),
        c_imag_abs_max=0.30,
        out_of_bounds_penalty=10.0,
        use_asymptotic_refinement=False,
        include_qr_residual=False,
    )
    c_2d = alpha_track[-1]['selected_c']
    omega_2d = alpha_target * c_2d.imag

    beta_cases = [
        {'alpha': alpha_target, 'beta': beta, 'Re': re_l, 'Ma': ma}
        for beta in beta_path(beta_target)
    ]
    beta_track = continue_temporal_mode_3d(
        beta_cases,
        baseflow_builder=profile_builder,
        Pr=0.72,
        gamma=1.4,
        N=90,
        y_max=12.0,
        wall_bc='adiabatic',
        length_scale='L_star',
        initial_c=c_2d,
        prefer_positive_growth=True,
        proximity_weight=2.0,
        leakage_weight=0.25,
        qr_weight=0.0,
        growth_weight=0.0,
        damping_penalty=1.0,
        c_real_bounds=(0.0, 1.2),
        c_imag_abs_max=0.30,
        out_of_bounds_penalty=10.0,
        use_asymptotic_refinement=False,
        include_qr_residual=False,
    )
    c_oblique = beta_track[-1]['selected_c']
    omega_oblique = alpha_target * c_oblique.imag

    profile = make_mack_profile(ma)
    spectrum = temporal_candidate_spectrum_3d(
        profile,
        alpha_target,
        beta_target,
        re_l,
        ma,
        0.72,
        1.4,
        N=90,
        y_max=12.0,
        wall_bc='adiabatic',
        length_scale='L_star',
        include_qr_residual=False,
    )
    physical = (
        (spectrum['c'].real > 0.0)
        & (spectrum['c'].real < 1.2)
        & (np.abs(spectrum['c'].imag) < 0.30)
    )
    if not np.any(physical):
        direct_idx = int(np.argmax(spectrum['omega_i']))
    else:
        direct_idx = int(np.argmax(spectrum['omega_i'][physical]))
        direct_idx = int(np.flatnonzero(physical)[direct_idx])
    direct_c = spectrum['c'][direct_idx]
    direct_omega = spectrum['omega_i'][direct_idx]

    refined_c, _, _, refined_converged, refined_leakage = (
        refine_temporal_compressible_3d_asymptotic(
            profile,
            alpha_target,
            beta_target,
            re_l,
            ma,
            0.72,
            1.4,
            c_guess=c_oblique,
            N=90,
            y_max=12.0,
            wall_bc='adiabatic',
            length_scale='L_star',
        )
    )
    refined_omega = alpha_target * refined_c.imag

    return {
        'case': case,
        'c_2d': c_2d,
        'omega_2d': omega_2d,
        'c_oblique': c_oblique,
        'omega_oblique': omega_oblique,
        'direct_c': direct_c,
        'direct_omega': direct_omega,
        'refined_c': refined_c,
        'refined_omega': refined_omega,
        'refined_converged': refined_converged,
        'refined_leakage': refined_leakage,
    }


def main():
    print('=' * 92)
    print('DIAGNOSTIC: ALPHA/BETA CONTINUATION OF REPRESENTATIVE MACK TABLE 10.1 CASES')
    print('=' * 92)
    print()
    print('All quantities are reported on Mack L* scaling.')
    print('The tracked branch is seeded on a physically plausible c_r > 0 family,')
    print('first continued in 2D (beta = 0) and then in obliqueness to the target psi.')
    print()
    print(
        f'{"M":>4s} {"R":>6s} {"a":>6s} {"psi":>6s} '
        f'{"2D track":>12s} {"oblique track":>14s} '
        f'{"direct phys":>12s} {"refined":>12s} {"table":>12s}'
    )
    print('-' * 92)

    for case in REPRESENTATIVE_CASES:
        result = tracked_growth(case)
        ma = case['Ma']
        re_l = case['Re']
        alpha_target = case['alpha']
        psi_deg = case['psi_deg']
        omega_table = case['omega_table']
        print(
            f'{ma:4.1f} {re_l:6.0f} {alpha_target:6.3f} {psi_deg:6.0f} '
            f'{result["omega_2d"]:12.6e} {result["omega_oblique"]:14.6e} '
            f'{result["direct_omega"]:12.6e} {result["refined_omega"]:12.6e} '
            f'{omega_table:12.6e}'
        )

    print()
    print('Interpretation:')
    print('  If the oblique-track value remains near the direct max-growth value,')
    print('  the remaining error is not just a mode-selection failure.')
    print('  If the 2D-track value is already oversized at beta = 0,')
    print('  the defect is already present in the 2D temporal operator family.')


if __name__ == '__main__':
    main()
