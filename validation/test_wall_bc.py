"""Regression tests for thermal wall-boundary-condition handling."""


import numpy as np


from pymack.mack_shooting import wall_condition_rows_3d
from pymack.solver import apply_wall_bc_3d


def test_apply_wall_bc_3d_adiabatic_uses_temperature_derivative():
    """The 3D EVP must impose DT=0 at an adiabatic wall, not T=0."""
    print('Test 1: 3D adiabatic wall BC uses temperature-derivative row')

    n = 5
    nn = 5 * n
    A = np.zeros((nn, nn), dtype=complex)
    B = np.ones((nn, nn), dtype=complex)
    D1 = np.arange(n * n, dtype=float).reshape(n, n)

    apply_wall_bc_3d(A, B, D1, n, wall_bc='adiabatic')

    wall = n - 1
    temp_row = 3 * n + wall
    temp_slice = slice(3 * n, 4 * n)

    assert np.allclose(B[temp_row, :], 0.0)
    assert np.allclose(A[temp_row, :temp_slice.start], 0.0)
    assert np.allclose(A[temp_row, temp_slice.stop:], 0.0)
    assert np.allclose(A[temp_row, temp_slice], D1[wall, :])
    print('  PASSED\n')


def test_wall_condition_rows_3d_switch_temperature_index():
    """The exact-shooting wall matrix must switch from T to DT for adiabatic walls."""
    print('Test 2: Shooting wall rows switch temperature state for adiabatic walls')

    assert wall_condition_rows_3d('isothermal') == (0, 2, 4, 6)
    assert wall_condition_rows_3d('adiabatic') == (0, 2, 5, 6)
    print('  PASSED\n')


if __name__ == '__main__':
    print('=' * 60)
    print('THERMAL WALL-BC VALIDATION')
    print('=' * 60)
    print()
    test_apply_wall_bc_3d_adiabatic_uses_temperature_derivative()
    test_wall_condition_rows_3d_switch_temperature_index()
    print('=' * 60)
    print('THERMAL WALL-BC TESTS COMPLETE')
    print('=' * 60)
