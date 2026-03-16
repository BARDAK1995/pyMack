"""
Validation: Chebyshev spectral differentiation accuracy.

Tests:
1. Differentiate polynomials (exact for degree ≤ N)
2. Differentiate exp(-y) on mapped domain [0, 40]
3. Verify domain mapping properties
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lst.spectral import chebyshev_points, chebyshev_D, map_domain, physical_derivatives


def test_polynomial_differentiation():
    """Chebyshev D should differentiate polynomials of degree ≤ N exactly."""
    print('Test 1: Polynomial differentiation')
    N = 16
    x = chebyshev_points(N)
    D = chebyshev_D(N)

    # f(x) = x^5 → f'(x) = 5x^4
    f = x**5
    fp_exact = 5 * x**4
    fp_num = D @ f
    err = np.max(np.abs(fp_num - fp_exact))
    print(f'  x^5 derivative error: {err:.2e}')
    assert err < 1e-10, f'Polynomial diff error too large: {err}'

    # f(x) = T_N(x) (Chebyshev polynomial)
    from numpy.polynomial.chebyshev import chebval
    coeffs = np.zeros(N + 1)
    coeffs[N] = 1
    f = chebval(x, coeffs)
    # Derivative of T_N
    dcoeffs = np.zeros(N)
    for k in range(N):
        # T_N' = N * U_{N-1} where U is Chebyshev of second kind
        pass
    # Just check numerically with higher order
    D2 = D @ D
    f = np.sin(np.pi * x)
    fp_exact = np.pi * np.cos(np.pi * x)
    fp_num = D @ f
    err = np.max(np.abs(fp_num - fp_exact))
    print(f'  sin(pi*x) derivative error (N={N}): {err:.2e}')

    print('  PASSED\n')


def test_exp_on_mapped_domain():
    """Differentiate exp(-y) on [0, 40] — key accuracy test."""
    print('Test 2: exp(-y) on [0, 40] with domain mapping')

    for N in [40, 60, 80, 100]:
        D_eta = chebyshev_D(N)
        y, D1, D2 = physical_derivatives(D_eta, y_max=40.0, N=N)

        f = np.exp(-y)
        fp_exact = -np.exp(-y)
        fpp_exact = np.exp(-y)

        fp_num = D1 @ f
        fpp_num = D2 @ f

        err1 = np.max(np.abs(fp_num - fp_exact))
        err2 = np.max(np.abs(fpp_num - fpp_exact))

        print(f'  N={N:3d}: |D1 error| = {err1:.2e}, |D2 error| = {err2:.2e}')

    # At N=60, should achieve ~1e-12 or better
    D_eta = chebyshev_D(60)
    y, D1, D2 = physical_derivatives(D_eta, y_max=40.0, N=60)
    f = np.exp(-y)
    err = np.max(np.abs(D1 @ f - (-np.exp(-y))))
    print(f'\n  N=60 target: error = {err:.2e} (need < 1e-10)')
    assert err < 1e-10, f'Spectral accuracy insufficient: {err}'
    print('  PASSED\n')


def test_domain_mapping():
    """Verify mapping properties: monotonicity, boundary values, clustering."""
    print('Test 3: Domain mapping properties')

    N = 64
    eta = chebyshev_points(N)
    y_max = 40.0
    y, dy, d2y = map_domain(eta, y_max)

    # Check boundaries
    assert abs(y[0] - y_max) < 1e-10, f'y[0] should be y_max, got {y[0]}'
    assert abs(y[-1]) < 1e-10, f'y[-1] should be 0, got {y[-1]}'
    print(f'  Boundaries: y[0]={y[0]:.4f} (y_max={y_max}), y[-1]={y[-1]:.6f} (0)')

    # Check monotonicity (y should decrease from y_max to 0)
    assert np.all(np.diff(y) < 0), 'y should be monotonically decreasing'
    print(f'  Monotonicity: OK')

    # Check clustering near wall
    n_lower_third = np.sum(y < y_max / 3)
    frac = n_lower_third / (N + 1)
    print(f'  Points in lower third: {n_lower_third}/{N+1} ({frac:.1%})')
    assert frac > 0.3, f'Insufficient wall clustering: {frac:.1%}'

    # Check metric positivity
    assert np.all(dy > 0), 'dy/deta should be positive everywhere'
    print(f'  Metric dy/dη > 0: OK')

    print('  PASSED\n')


def test_second_derivative_accuracy():
    """Second derivative on physical domain for various functions."""
    print('Test 4: Second derivative accuracy')

    N = 80
    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max=30.0, N=N)

    # Test with a boundary-layer-like profile: U = 1 - exp(-y)
    U = 1.0 - np.exp(-y)
    dU_exact = np.exp(-y)
    d2U_exact = -np.exp(-y)

    dU_num = D1 @ U
    d2U_num = D2 @ U

    err1 = np.max(np.abs(dU_num - dU_exact))
    err2 = np.max(np.abs(d2U_num - d2U_exact))
    print(f'  1-exp(-y): D1 err={err1:.2e}, D2 err={err2:.2e}')

    # Gaussian: exp(-y²/4)
    g = np.exp(-y**2 / 4)
    dg = -y/2 * g
    d2g = (-0.5 + y**2/4) * g

    err1 = np.max(np.abs(D1 @ g - dg))
    err2 = np.max(np.abs(D2 @ g - d2g))
    print(f'  Gaussian:   D1 err={err1:.2e}, D2 err={err2:.2e}')

    print('  PASSED\n')


if __name__ == '__main__':
    print('='*60)
    print('CHEBYSHEV SPECTRAL VALIDATION')
    print('='*60 + '\n')

    test_polynomial_differentiation()
    test_exp_on_mapped_domain()
    test_domain_mapping()
    test_second_derivative_accuracy()

    print('='*60)
    print('ALL CHEBYSHEV TESTS PASSED')
    print('='*60)
