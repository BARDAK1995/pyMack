"""
Chebyshev spectral collocation infrastructure.

Provides differentiation matrices on Gauss-Lobatto points and
algebraic domain mapping from [-1,1] to [0, y_max].
"""

import numpy as np


def chebyshev_points(N):
    """Chebyshev-Gauss-Lobatto points on [-1, 1].

    Returns N+1 points: x_j = cos(π j / N), j = 0, ..., N.
    Points are ordered from +1 to -1 (x[0]=1, x[N]=-1).
    """
    j = np.arange(N + 1)
    return np.cos(np.pi * j / N)


def chebyshev_D(N):
    """First-derivative Chebyshev differentiation matrix (Don & Solomonoff).

    Uses the explicit formula with barycentric weights to build the
    (N+1) x (N+1) matrix D such that (Df)_j ≈ f'(x_j).
    """
    x = chebyshev_points(N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c[1::2] *= -1  # alternating signs

    X = np.outer(x, np.ones(N + 1))
    dX = X - X.T + np.eye(N + 1)  # avoid division by zero on diagonal

    D = np.outer(c, 1.0 / c) / dX
    D -= np.diag(D.sum(axis=1))  # diagonal: negative row sum
    return D


def map_domain(eta, y_max, L=None):
    """Algebraic mapping from Chebyshev domain [-1,1] to physical [0, y_max].

    Uses the mapping (Malik 1990):
        y = L * (1 + η) / (1 - η + 2L/y_max)

    where L controls clustering near the wall. Default L = y_max / 6
    places ~50% of points in the lower third of the domain.

    Parameters
    ----------
    eta : array
        Chebyshev points on [-1, 1], ordered from +1 to -1.
    y_max : float
        Physical domain height.
    L : float, optional
        Stretching parameter. Default: y_max / 6.

    Returns
    -------
    y : array
        Physical coordinates.
    dy_deta : array
        First metric term dy/dη.
    d2y_deta2 : array
        Second metric term d²y/dη².
    """
    if L is None:
        L = y_max / 6.0

    denom = 1.0 - eta + 2.0 * L / y_max
    y = L * (1.0 + eta) / denom

    # dy/dη
    dy_deta = L * (2.0 + 2.0 * L / y_max) / denom**2

    # d²y/dη²
    d2y_deta2 = 2.0 * L * (2.0 + 2.0 * L / y_max) / denom**3

    return y, dy_deta, d2y_deta2


def physical_derivatives(D_eta, y_max, N, L=None):
    """Build physical-space derivative matrices D1, D2 on [0, y_max].

    Parameters
    ----------
    D_eta : (N+1, N+1) array
        Chebyshev differentiation matrix on [-1,1].
    y_max : float
        Physical domain height.
    N : int
        Number of Chebyshev intervals (matrix is (N+1)x(N+1)).
    L : float, optional
        Stretching parameter.

    Returns
    -------
    y : array
        Physical grid points.
    D1 : array
        First derivative matrix in physical space.
    D2 : array
        Second derivative matrix in physical space.
    """
    eta = chebyshev_points(N)
    y, dy_deta, d2y_deta2 = map_domain(eta, y_max, L)

    # dη/dy = 1 / (dy/dη)
    deta_dy = 1.0 / dy_deta

    # d²η/dy² = -d²y/dη² / (dy/dη)³
    d2eta_dy2 = -d2y_deta2 / dy_deta**3

    D2_eta = D_eta @ D_eta

    # D1_y = dη/dy · D_η
    D1 = np.diag(deta_dy) @ D_eta

    # D2_y = (dη/dy)² · D²_η + d²η/dy² · D_η
    D2 = np.diag(deta_dy**2) @ D2_eta + np.diag(d2eta_dy2) @ D_eta

    return y, D1, D2
