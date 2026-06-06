"""
Validation: compressible stability solver.

Tests:
1. Compressible base-flow profiles are physical.
2. Both spatial solver paths return usable eigenvalues.
3. The refined frequency sweep produces a growing branch at Ma=2.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.analysis import frequency_sweep
from pymack.baseflow import CompressibleBlasiusProfile
from pymack.equations import momentum_viscous_coefficients
from pymack.scales import delta_star_over_lstar
from pymack.solver import (
    solve_spatial,
    solve_spatial_from_temporal,
    solve_temporal_compressible,
    solve_temporal_compressible_3d,
)


def test_compressible_baseflow():
    """Verify compressible base-flow profiles are physical."""
    print('Test 1: Compressible base flow profiles')

    bf = CompressibleBlasiusProfile(
        Ma=5.35, T_wall=370.0, T_edge=56.0,
        gamma=1.4, Pr=0.72, omega=0.74, R_gas=296.8)

    y_test = np.linspace(0, 3, 50)
    prof = bf(y_test)

    assert abs(prof['U'][0]) < 0.01, f"U at wall should be ~0, got {prof['U'][0]}"
    assert prof['U'][-1] > 0.9, f"U at edge should be ~1, got {prof['U'][-1]}"

    T_wall_ratio = bf.T_wall / bf.T_edge
    assert prof['T'][0] > 3, f"T/Te at wall should be large, got {prof['T'][0]}"
    assert abs(prof['T'][-1] - 1.0) < 0.3, f"T/Te at edge should be ~1, got {prof['T'][-1]}"
    assert prof['rho'][0] < 0.5, f"rho/rho_e at wall should be small, got {prof['rho'][0]}"

    print(f'  U(0)={prof["U"][0]:.4f}, U(3)={prof["U"][-1]:.4f}')
    print(f'  T/Te(0)={prof["T"][0]:.2f}, T/Te(3)={prof["T"][-1]:.2f}')
    print(f'  rho/rho_e(0)={prof["rho"][0]:.3f}, rho/rho_e(3)={prof["rho"][-1]:.3f}')
    print(f'  T_wall/T_edge = {T_wall_ratio:.2f}')
    print('  PASSED\n')


def test_spatial_solver_basic():
    """Basic check: both spatial solver paths return physical eigenvalues."""
    print('Test 2: Spatial solver basic operation')

    bf = CompressibleBlasiusProfile(
        Ma=2.0, T_wall=300.0, T_edge=200.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    omega = 0.05
    Re = 1000
    Ma = 2.0

    alphas, _, _ = solve_spatial(
        bf, omega, Re, Ma, Pr=0.72, gamma=1.4, N=80, y_max=15.0)
    alpha_refined, converged = solve_spatial_from_temporal(
        bf, omega, Re, Ma, Pr=0.72, gamma=1.4, N=80, y_max=15.0)

    print(f'  Found {len(alphas)} physical eigenvalues from companion QEP')
    if len(alphas) > 0:
        print(f'  alpha_r range: [{alphas.real.min():.3f}, {alphas.real.max():.3f}]')
        print(f'  alpha_i range: [{alphas.imag.min():.3f}, {alphas.imag.max():.3f}]')
        idx = np.argmin(alphas.imag)
        print(f'  Most amplified (QEP): alpha = {alphas[idx].real:.4f} + {alphas[idx].imag:.4f}i')
        print(f'  QEP growth rate sigma = {-alphas[idx].imag:.4f}')

    if converged:
        print(f'  Refined solver: alpha = {alpha_refined.real:.4f} + {alpha_refined.imag:.4f}i')
        print(f'  Refined sigma = {-alpha_refined.imag:.4f}')

    assert len(alphas) > 0, 'Companion-form solver returned no eigenvalues'
    assert converged, 'Refined spatial solver did not converge'
    print('  PASSED\n')


def test_frequency_sweep_ma2():
    """Refined frequency sweep at Ma=2 should recover a growing branch."""
    print('Test 3: Frequency sweep at Ma=2.0, Re=1500')

    bf = CompressibleBlasiusProfile(
        Ma=2.0, T_wall=300.0, T_edge=150.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    omegas = np.linspace(0.02, 0.15, 20)
    omegas_out, sigma, _ = frequency_sweep(
        bf, Re=1500, Ma=2.0, omega_range=omegas,
        Pr=0.72, gamma=1.4, N=80, y_max=15.0, method='refined')

    valid = ~np.isnan(sigma)
    if not valid.any():
        raise AssertionError('Refined frequency sweep returned no valid growth rates')

    max_sigma = np.nanmax(sigma)
    idx_max = np.nanargmax(sigma)
    print(f'  Max growth rate: sigma = {max_sigma:.5f} at omega = {omegas_out[idx_max]:.4f}')
    print(f'  Valid points: {valid.sum()}/{len(omegas)}')
    assert max_sigma > 1e-4, f'Expected a growing branch, got sigma_max={max_sigma:.5e}'
    print('  PASSED\n')


def test_mack_viscous_coefficients():
    """The default 2D momentum coefficients should match Mack's d=lambda/mu=1.2."""
    print("Test 4: Mack momentum coefficients")

    x_alpha2, cross_grad, y_laplacian, y_u_alg = momentum_viscous_coefficients()
    print(f'  x alpha^2 coefficient = {x_alpha2:.6f}')
    print(f'  cross-gradient coefficient = {cross_grad:.6f}')
    print(f'  y Laplacian coefficient = {y_laplacian:.6f}')
    print(f'  y algebraic-u coefficient = {y_u_alg:.6f}')

    assert np.isclose(x_alpha2, 2.0 * (2.0 + 1.2) / 3.0)
    assert np.isclose(cross_grad, (1.0 + 2.0 * 1.2) / 3.0)
    assert np.isclose(y_laplacian, 2.0 * (2.0 + 1.2) / 3.0)
    assert np.isclose(y_u_alg, 2.0 * (1.2 - 1.0) / 3.0)
    print('  PASSED\n')


def test_oblique_solver_beta_zero_consistency():
    """The oblique temporal solver should reduce to the 2D solver at beta=0."""
    print('Test 5: Oblique solver beta=0 consistency')

    bf = CompressibleBlasiusProfile(
        Ma=2.2, T_wall=300.0, T_edge=150.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    alpha = 0.08
    c_2d, _, _ = solve_temporal_compressible(
        bf, alpha, Re=800, Ma=2.2, Pr=0.72, gamma=1.4, N=80, y_max=12.0)
    c_3d, _, _ = solve_temporal_compressible_3d(
        bf, alpha, beta=0.0, Re=800, Ma=2.2, Pr=0.72, gamma=1.4,
        N=80, y_max=12.0)

    if len(c_2d) == 0 or len(c_3d) == 0:
        raise AssertionError('Expected non-empty spectra from both solvers')

    lead_2d = c_2d[0]
    idx = np.argmin(np.abs(c_3d - lead_2d))
    lead_3d = c_3d[idx]
    diff = abs(lead_3d - lead_2d)

    print(f'  2D lead mode: {lead_2d.real:.6f} + {lead_2d.imag:.6f}i')
    print(f'  3D lead mode: {lead_3d.real:.6f} + {lead_3d.imag:.6f}i')
    print(f'  |delta c| = {diff:.3e}')
    assert diff < 5e-3, f'Oblique solver beta=0 mismatch too large: {diff:.3e}'
    print('  PASSED\n')


def test_lstar_wrapper_matches_manual_delta_conversion():
    """The explicit L* path should match manual delta* input conversion."""
    print('Test 6: L* wrapper matches manual delta* conversion')

    bf = CompressibleBlasiusProfile(
        Ma=4.5, T_wall=388.0, T_edge=311.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    delta_over_l = delta_star_over_lstar(bf)
    alpha_l = 0.05
    re_l = 500.0

    c_l, _, _ = solve_temporal_compressible(
        bf, alpha_l, Re=re_l, Ma=4.5, Pr=0.72, gamma=1.4,
        N=80, y_max=8.0, length_scale='L_star')
    c_d, _, _ = solve_temporal_compressible(
        bf, alpha_l * delta_over_l, Re=re_l * delta_over_l,
        Ma=4.5, Pr=0.72, gamma=1.4,
        N=80, y_max=8.0 / delta_over_l, length_scale='delta_star')

    lead_l = c_l[0]
    lead_d = c_d[np.argmin(np.abs(c_d - lead_l))]
    diff = abs(lead_l - lead_d)

    print(f'  delta*/L* = {delta_over_l:.6f}')
    print(f'  L* lead mode: {lead_l.real:.6f} + {lead_l.imag:.6f}i')
    print(f'  converted delta* lead: {lead_d.real:.6f} + {lead_d.imag:.6f}i')
    print(f'  |delta c| = {diff:.3e}')
    assert diff < 5e-3, f'L* wrapper mismatch too large: {diff:.3e}'
    print('  PASSED\n')


def test_sixth_order_switch_is_inactive_at_beta_zero():
    """The 6th-/8th-order oblique switch should be identical when beta=0."""
    print('Test 7: 6th vs 8th order switch at beta=0')

    bf = CompressibleBlasiusProfile(
        Ma=4.5, T_wall=388.0, T_edge=311.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    alpha = 0.06
    c_8, _, _ = solve_temporal_compressible_3d(
        bf, alpha, beta=0.0, Re=1000, Ma=4.5, Pr=0.72, gamma=1.4,
        N=80, y_max=8.0, include_spanwise_dissipation_coupling=True)
    c_6, _, _ = solve_temporal_compressible_3d(
        bf, alpha, beta=0.0, Re=1000, Ma=4.5, Pr=0.72, gamma=1.4,
        N=80, y_max=8.0, include_spanwise_dissipation_coupling=False)

    lead_8 = c_8[0]
    lead_6 = c_6[np.argmin(np.abs(c_6 - lead_8))]
    diff = abs(lead_8 - lead_6)

    print(f'  8th-order lead mode: {lead_8.real:.6f} + {lead_8.imag:.6f}i')
    print(f'  6th-order lead mode: {lead_6.real:.6f} + {lead_6.imag:.6f}i')
    print(f'  |delta c| = {diff:.3e}')
    assert diff < 5e-8, f'6th/8th order beta=0 mismatch too large: {diff:.3e}'
    print('  PASSED\n')


if __name__ == '__main__':
    print('=' * 60)
    print('COMPRESSIBLE STABILITY VALIDATION')
    print('=' * 60 + '\n')

    test_compressible_baseflow()
    test_spatial_solver_basic()
    test_frequency_sweep_ma2()
    test_mack_viscous_coefficients()
    test_oblique_solver_beta_zero_consistency()
    test_lstar_wrapper_matches_manual_delta_conversion()
    test_sixth_order_switch_is_inactive_at_beta_zero()

    print('=' * 60)
    print('COMPRESSIBLE TESTS COMPLETE')
    print('=' * 60)
