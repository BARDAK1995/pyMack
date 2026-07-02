"""Cross-check the Ozgen 2-D shooting solver against the spectral solver."""


import numpy as np


from pymack.baseflow import make_flatplate_profile
from pymack.scales import delta_star_over_lstar
from pymack.temporal_solver import solve_temporal_2d
from pymack.temporal_shooting import ozgen_temporal_sigma_min_2d


def test_ozgen_shooting_wall_matrix_singular_at_spectral_eigenvalue():
    Ma, Re, alpha = 6.0, 5500.0, 0.174
    profile = make_flatplate_profile(Ma)
    y_max = 10.0 * float(delta_star_over_lstar(profile))
    eigenvalues, _, _ = solve_temporal_2d(
        profile, alpha, Re, Ma, N=140, y_max=y_max,
        wall_bc='isothermal', length_scale='L_star')
    c_spectral = eigenvalues[int(np.argmin(np.abs(eigenvalues - (0.93 + 0.02j))))]
    assert abs(c_spectral - (0.9301 + 0.0200j)) < 5e-3

    def sigma_min(c):
        return ozgen_temporal_sigma_min_2d(
            profile, alpha, c, Re, Ma,
            y_max=y_max, length_scale='L_star', n_steps=600)

    at_mode = sigma_min(c_spectral)
    off_mode = sigma_min(c_spectral + 0.02)
    assert at_mode < 5e-4, f'{at_mode:.3e}'
    assert off_mode > 20.0 * at_mode, f'{at_mode:.3e} vs {off_mode:.3e}'
