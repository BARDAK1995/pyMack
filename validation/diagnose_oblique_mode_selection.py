"""Diagnose branch-selection ambiguity for Mack Chapter 10 oblique cases."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.mack_conditions import make_mack_profile
from lst.mack_shooting import temporal_shooting_residual_3d
from lst.solver import (
    solve_temporal_compressible_3d,
    refine_temporal_compressible_3d_asymptotic,
)


CASES = [
    (1.3, 500, 0.075, 45, 0.824e-3),
    (4.5, 1500, 0.050, 60, 1.613e-3),
    (10.0, 1500, 0.040, 55, 0.434e-3),
]


def candidate_rows(ma, re_l, alpha_l, psi_deg):
    beta_l = alpha_l * np.tan(np.deg2rad(psi_deg))
    profile = make_mack_profile(ma)

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
    c_all = c_all[mask]
    leakage = leakage[mask]

    rows = []
    for c_val, leak in zip(c_all[:6], leakage[:6]):
        qr_res = abs(
            temporal_shooting_residual_3d(
                profile,
                alpha_l,
                beta_l,
                c_val,
                re_l,
                ma,
                0.72,
                1.4,
                12.0,
                length_scale='L_star',
                method='qr',
                n_steps=800,
            )
        )
        rows.append((c_val, alpha_l * c_val.imag, float(leak), qr_res))

    refined = None
    if len(c_all) > 0:
        c_ref, _, _, converged, leak_ref = refine_temporal_compressible_3d_asymptotic(
            profile,
            alpha_l,
            beta_l,
            re_l,
            ma,
            0.72,
            1.4,
            c_guess=c_all[0],
            N=90,
            y_max=12.0,
            wall_bc='adiabatic',
            length_scale='L_star',
        )
        refined = (c_ref, alpha_l * c_ref.imag, converged, leak_ref)

    return rows, refined


def main():
    print('=' * 92)
    print('DIAGNOSTIC: OBLIQUE MODE-SELECTION AMBIGUITY')
    print('=' * 92)
    print()
    print('Columns:')
    print('  omega_i  = temporal amplification rate on Mack L* scale')
    print('  leakage  = distance from Appendix-B decay subspace at the top boundary')
    print('  qr_res   = bounded-shooting wall residual after QR-stabilized subspace marching')
    print()

    for ma, re_l, alpha_l, psi_deg, table_oi in CASES:
        rows, refined = candidate_rows(ma, re_l, alpha_l, psi_deg)
        print(f'M={ma:.1f}, R={re_l:.0f}, alpha={alpha_l:.3f}, psi={psi_deg:.0f}, table omega_i={table_oi:.6e}')
        print(f'{"candidate c":>28s} {"omega_i":>12s} {"leakage":>12s} {"qr_res":>12s}')
        for c_val, omega_i, leak, qr_res in rows:
            print(f'{str(c_val):>28s} {omega_i:12.6e} {leak:12.6e} {qr_res:12.6e}')
        if refined is not None:
            c_ref, omega_i_ref, converged, leak_ref = refined
            print(
                f'  asymptotic-refined: c={c_ref}, omega_i={omega_i_ref:.6e}, '
                f'converged={converged}, leakage={leak_ref:.6e}'
            )
        print()


if __name__ == '__main__':
    main()
