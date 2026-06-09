"""Regression tests for dimensional Mack flat-plate scaling."""

from __future__ import annotations

import math

import pytest

from pymack.scales import (
    DimensionalEdgeState,
    F_to_frequency_khz,
    R_L_to_x_mm,
    alpha_L_to_per_m,
    frequency_khz_to_F,
    lstar_m_from_R_L,
    sigma_L_to_per_m,
    sigma_L_to_per_mm,
    wavelength_L_to_mm,
    x_mm_to_R_L,
)


def _aps_state():
    return DimensionalEdgeState(
        U_e=858.0,
        nu_e=7.313e-5,
        T_e=52.0,
        M_e=5.85,
        gamma=1.4,
        gas="nitrogen",
        unit_reynolds_per_m=1.176e7,
    )


def test_aps_frequency_to_mack_F_anchors():
    state = _aps_state()

    assert 1.0e4 * frequency_khz_to_F(247.0, state) == pytest.approx(1.5417, rel=3.0e-5)
    assert 1.0e4 * frequency_khz_to_F(280.0, state) == pytest.approx(1.7477, rel=3.0e-5)
    assert 1.0e4 * frequency_khz_to_F(325.0, state) == pytest.approx(2.0285, rel=3.0e-5)

    F = frequency_khz_to_F(280.0, state)
    assert F_to_frequency_khz(F, state) == pytest.approx(280.0)


def test_aps_R_to_x_mm_anchors():
    state = _aps_state()

    assert R_L_to_x_mm(744.0, state) == pytest.approx(47.18, rel=2.0e-4)
    assert R_L_to_x_mm(849.0, state) == pytest.approx(61.44, rel=2.0e-4)
    assert R_L_to_x_mm(1038.0, state) == pytest.approx(91.83, rel=2.0e-4)
    assert R_L_to_x_mm(1183.0, state) == pytest.approx(119.28, rel=2.0e-4)

    R = x_mm_to_R_L(91.834, state)
    assert R == pytest.approx(1038.0, rel=5.0e-5)


def test_growth_wavenumber_and_wavelength_units_are_consistent():
    state = _aps_state()
    R = 1000.0
    sigma_L = 2.0e-3
    alpha_L = 1.2
    wavelength_L = 2.0 * math.pi / alpha_L
    L_star = lstar_m_from_R_L(R, state)

    assert sigma_L_to_per_m(sigma_L, R, state) == pytest.approx(sigma_L / L_star)
    assert sigma_L_to_per_mm(sigma_L, R, state) == pytest.approx(sigma_L / L_star / 1000.0)
    assert alpha_L_to_per_m(alpha_L, R, state) == pytest.approx(alpha_L / L_star)
    assert wavelength_L_to_mm(wavelength_L, R, state) == pytest.approx(wavelength_L * L_star * 1000.0)
