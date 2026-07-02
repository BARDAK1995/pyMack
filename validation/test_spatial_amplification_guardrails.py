"""Regression tests for spatial fixed-frequency amplification guardrails."""

import argparse
import math

import numpy as np
import pytest


from scripts import compute_spatial_fixed_frequency_curves as growth_curves
from scripts import postprocess_spatial_amplification as amplification


def test_second_mode_defaults_apply_alpha_family_filter():
    args = argparse.Namespace(
        mode_family="second_mode",
        alpha_min_l=None,
        alpha_max_l=None,
    )

    growth_curves._apply_mode_family_defaults(args)

    assert args.alpha_min_l == pytest.approx(growth_curves.SECOND_MODE_ALPHA_MIN_L)
    assert args.alpha_max_l is None


def test_filtered_alpha_selection_does_not_fallback_to_low_alpha_branch():
    low_alpha_roots = np.array([
        0.12 - 0.020j,
        0.18 - 0.030j,
    ])

    selected = growth_curves._select_alpha(
        low_alpha_roots,
        target_alpha=0.12,
        omega_L=0.12,
        delta_over_l=1.0,
        selection="max_sigma",
        phase_min=0.80,
        phase_max=1.05,
        alpha_min_L=0.6,
        alpha_max_L=None,
    )

    assert not np.isfinite(selected)


def test_amplification_postprocess_rejects_low_alpha_second_mode_rows():
    rows = [
        {
            "freq_parameter": 1.7e-4,
            "R_L": R,
            "omega_L": 1.7e-4 * R,
            "sigma_L": sigma,
            "wavelength_L": 2.0 * math.pi / 0.1,
            "alpha_r_L": 0.1,
            "alpha_i_L": -sigma,
        }
        for R, sigma in [(200.0, -1.0e-3), (400.0, 2.0e-2), (600.0, 1.0e-2)]
    ]

    out_rows, summary = amplification._process_frequency(
        rows,
        1.7e-4,
        view_min=200.0,
        view_max=600.0,
        alpha_min_l=0.6,
        alpha_max_l=None,
        dx_over_lstar_per_dR=1.0,
    )

    assert out_rows == []
    assert summary["status"] == "no_finite_rows"


def test_amplification_integration_uses_explicit_streamwise_multiplier():
    rows = [
        {
            "freq_parameter": 2.0e-3,
            "R_L": R,
            "omega_L": 2.0e-3 * R,
            "sigma_L": sigma,
            "wavelength_L": 2.0 * math.pi,
            "alpha_r_L": 1.0,
            "alpha_i_L": -sigma,
        }
        for R, sigma in [
            (0.0, -1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, -1.0),
        ]
    ]

    _rows_one, summary_one = amplification._process_frequency(
        rows,
        2.0e-3,
        view_min=0.0,
        view_max=3.0,
        alpha_min_l=0.6,
        alpha_max_l=None,
        dx_over_lstar_per_dR=1.0,
    )
    _rows_two, summary_two = amplification._process_frequency(
        rows,
        2.0e-3,
        view_min=0.0,
        view_max=3.0,
        alpha_min_l=0.6,
        alpha_max_l=None,
        dx_over_lstar_per_dR=2.0,
    )

    assert summary_one["N_signed_peak_from_start"] == pytest.approx(1.0)
    assert summary_two["N_signed_peak_from_start"] == pytest.approx(2.0)
    assert summary_two["amplification_signed_peak_from_start"] == pytest.approx(
        summary_one["amplification_signed_peak_from_start"] ** 2
    )


def test_open_downstream_lobe_integrates_from_lower_neutral():
    rows = [
        {
            "freq_parameter": 1.7e-4,
            "R_L": R,
            "omega_L": 1.7e-4 * R,
            "sigma_L": sigma,
            "wavelength_L": 2.0 * math.pi,
            "alpha_r_L": 1.0,
            "alpha_i_L": -sigma,
        }
        for R, sigma in [
            (0.0, -1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 1.0),
        ]
    ]

    out_rows, summary = amplification._process_frequency(
        rows,
        1.7e-4,
        view_min=0.0,
        view_max=3.0,
        alpha_min_l=0.6,
        alpha_max_l=None,
        dx_over_lstar_per_dR=1.0,
    )

    assert summary["status"] == "open_downstream_lobe"
    assert summary["lower_neutral_R_L"] == pytest.approx(0.5)
    assert math.isnan(summary["upper_neutral_R_L"])
    assert summary["N_signed_at_end_from_lower"] == pytest.approx(2.25)
    assert out_rows[-1]["amplification_region"] == "open_downstream_lobe"
    assert out_rows[-1]["amplification_signed_from_lower"] == pytest.approx(math.exp(2.25))
