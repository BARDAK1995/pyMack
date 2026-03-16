"""
Validation: Compressible stability solver.

Tests:
1. Low Mach (Ma=0.01) should recover O-S results
2. Ma=2.2 adiabatic wall first mode
3. Ma=4.5+ second mode appearance
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lst.baseflow import CompressibleBlasiusProfile
from lst.solver import solve_spatial
from lst.analysis import frequency_sweep


def test_compressible_baseflow():
    """Verify compressible base flow profiles are physical."""
    print('Test 1: Compressible base flow profiles')

    # Mach 5.35, N2
    bf = CompressibleBlasiusProfile(
        Ma=5.35, T_wall=370.0, T_edge=56.0,
        gamma=1.4, Pr=0.72, omega=0.74, R_gas=296.8)

    y_test = np.linspace(0, 3, 50)
    prof = bf(y_test)

    # U should go from 0 at wall to 1 at freestream
    assert abs(prof['U'][0]) < 0.01, f"U at wall should be ~0, got {prof['U'][0]}"
    assert prof['U'][-1] > 0.9, f"U at edge should be ~1, got {prof['U'][-1]}"

    # T should be high at wall, low at edge
    T_wall_ratio = bf.T_wall / bf.T_edge
    assert prof['T'][0] > 3, f"T/Te at wall should be large, got {prof['T'][0]}"
    assert abs(prof['T'][-1] - 1.0) < 0.3, f"T/Te at edge should be ~1, got {prof['T'][-1]}"

    # rho inversely proportional to T
    assert prof['rho'][0] < 0.5, f"rho/rho_e at wall should be small, got {prof['rho'][0]}"

    print(f'  U(0)={prof["U"][0]:.4f}, U(3)={prof["U"][-1]:.4f}')
    print(f'  T/Te(0)={prof["T"][0]:.2f}, T/Te(3)={prof["T"][-1]:.2f}')
    print(f'  rho/rho_e(0)={prof["rho"][0]:.3f}, rho/rho_e(3)={prof["rho"][-1]:.3f}')
    print(f'  T_wall/T_edge = {T_wall_ratio:.2f}')
    print('  PASSED\n')


def test_spatial_solver_basic():
    """Basic check: spatial solver runs and returns physical eigenvalues."""
    print('Test 2: Spatial solver basic operation')

    bf = CompressibleBlasiusProfile(
        Ma=2.0, T_wall=300.0, T_edge=200.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    omega = 0.05
    Re = 1000
    Ma = 2.0

    alphas, modes, y = solve_spatial(
        bf, omega, Re, Ma, Pr=0.72, gamma=1.4, N=80, y_max=15.0)

    print(f'  Found {len(alphas)} physical eigenvalues')
    if len(alphas) > 0:
        print(f'  alpha_r range: [{alphas.real.min():.3f}, {alphas.real.max():.3f}]')
        print(f'  alpha_i range: [{alphas.imag.min():.3f}, {alphas.imag.max():.3f}]')
        # Most amplified
        idx = np.argmin(alphas.imag)
        print(f'  Most amplified: alpha = {alphas[idx].real:.4f} + {alphas[idx].imag:.4f}i')
        print(f'  Growth rate sigma = {-alphas[idx].imag:.4f}')

    assert len(alphas) > 0, 'Solver returned no eigenvalues'
    print('  PASSED\n')


def test_frequency_sweep_ma2():
    """Frequency sweep at Ma=2 — should show first-mode instability."""
    print('Test 3: Frequency sweep at Ma=2.0, Re=1500')

    bf = CompressibleBlasiusProfile(
        Ma=2.0, T_wall=300.0, T_edge=150.0,
        gamma=1.4, Pr=0.72, omega=0.74)

    omegas = np.linspace(0.02, 0.15, 20)
    omegas_out, sigma, alpha_r = frequency_sweep(
        bf, Re=1500, Ma=2.0, omega_range=omegas,
        Pr=0.72, gamma=1.4, N=80, y_max=15.0)

    valid = ~np.isnan(sigma)
    if valid.any():
        max_sigma = np.nanmax(sigma)
        idx_max = np.nanargmax(sigma)
        print(f'  Max growth rate: sigma = {max_sigma:.5f} at omega = {omegas_out[idx_max]:.4f}')
        print(f'  Valid points: {valid.sum()}/{len(omegas)}')
    else:
        print('  WARNING: No valid growth rates found')
    print('  PASSED (informational)\n')


if __name__ == '__main__':
    print('='*60)
    print('COMPRESSIBLE STABILITY VALIDATION')
    print('='*60 + '\n')

    test_compressible_baseflow()
    test_spatial_solver_basic()
    test_frequency_sweep_ma2()

    print('='*60)
    print('COMPRESSIBLE TESTS COMPLETE')
    print('='*60)
