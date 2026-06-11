"""Validation for sharp-cone (Mangler) support.

Covers, with no eigenvalue solves:
  (a) exact sqrt(3)/factor-3 station-mapping consistency against pymack.scales,
  (b) the cone N-factor path integral (factor 3 vs the plate over the same
      R_eq window, cross-checked against physical arc-length integration),
  (c) the runner's --geometry cone dry-run contract and metadata, with the
      default flat-plate behavior protected as unchanged.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pymack.scales import (
    DimensionalEdgeState,
    alpha_L_to_per_m,
    alpha_L_to_per_mm,
    frequency_khz_to_F,
    lstar_m_from_R_L,
    sigma_L_to_per_m,
    sigma_L_to_per_mm,
    wavelength_L_to_mm,
    x_mm_to_R_L,
)
from pymack.cone import (
    CONE_FREQUENCY_RATIO_AT_SAME_S,
    MANGLER_FACTOR,
    ConeGeometry,
    R_eq_from_R_s,
    R_s_from_R_eq,
    cone_F_to_frequency_khz,
    cone_R_eq_to_s_m,
    cone_R_eq_to_s_mm,
    cone_alpha_L_to_per_m,
    cone_alpha_L_to_per_mm,
    cone_frequency_khz_to_F,
    cone_lstar_m_from_R_eq,
    cone_n_factor,
    cone_n_factor_multiplier,
    cone_s_mm_to_R_eq,
    cone_sigma_L_to_per_m,
    cone_sigma_L_to_per_mm,
    cone_wavelength_L_to_mm,
)

from scripts.postprocess_spatial_amplification import _integrate_full_path
from scripts.run_mach6_spatial_neutral_case import (
    QUALITY,
    _add_dimensional_columns,
    _artifact_paths,
    _resolve_case,
    _write_manifest,
    main,
    parse_args,
)

# scipy's trapezoid is stable across numpy 1.x/2.x (np.trapz was removed in
# numpy 2.3, and getattr(np, "trapezoid", np.trapz) evaluates the fallback
# eagerly — AttributeError on import).
from scipy.integrate import trapezoid as _TRAPZ  # noqa: E402


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


# ---------------------------------------------------------------------------
# (a) Pure-math sqrt(3) consistency against pymack.scales
# ---------------------------------------------------------------------------

def test_station_mapping_sqrt3_anchors():
    state = _aps_state()
    R_s = float(x_mm_to_R_L(91.83, state))
    assert R_s == pytest.approx(1037.978001567471, rel=1.0e-12)

    R_eq = float(cone_s_mm_to_R_eq(91.83, state))
    assert R_eq == pytest.approx(599.2768786178892, rel=1.0e-12)
    assert R_eq == pytest.approx(float(x_mm_to_R_L(91.83 / 3.0, state)), rel=1.0e-12)
    assert R_eq == pytest.approx(R_s / math.sqrt(3.0), rel=1.0e-12)

    # Round trips
    assert float(cone_R_eq_to_s_mm(R_eq, state)) == pytest.approx(91.83, rel=1.0e-12)
    assert float(cone_R_eq_to_s_m(R_eq, state)) == pytest.approx(0.09183, rel=1.0e-12)
    assert R_eq_from_R_s(R_s_from_R_eq(R_eq)) == pytest.approx(R_eq, rel=1.0e-15)
    assert R_s_from_R_eq(1.0) == pytest.approx(math.sqrt(3.0), rel=1.0e-15)
    assert MANGLER_FACTOR == 3.0


def test_lstar_and_frequency_sqrt3_relations_at_same_physical_station():
    state = _aps_state()
    s_mm = 91.83
    R_eq = float(cone_s_mm_to_R_eq(s_mm, state))
    R_s = float(x_mm_to_R_L(s_mm, state))

    L_cone = float(cone_lstar_m_from_R_eq(R_eq, state))
    L_plate = float(lstar_m_from_R_L(R_s, state))
    assert L_cone == pytest.approx(L_plate / math.sqrt(3.0), rel=1.0e-12)
    assert L_cone == pytest.approx(7.313e-5 * R_eq / 858.0, rel=1.0e-12)

    # Fixed omega_L: dimensional second-mode frequency at the same s is
    # sqrt(3) times higher on the cone than on the plate.
    omega_L = 0.1
    f_cone = omega_L * state.U_e / (2.0 * math.pi * L_cone)
    f_plate = omega_L * state.U_e / (2.0 * math.pi * L_plate)
    assert f_cone / f_plate == pytest.approx(math.sqrt(3.0), rel=1.0e-12)
    assert f_cone / f_plate == pytest.approx(CONE_FREQUENCY_RATIO_AT_SAME_S, rel=1.0e-12)


def test_frequency_converters_are_geometry_independent():
    state = _aps_state()
    assert 1.0e4 * cone_frequency_khz_to_F(280.0, state) == pytest.approx(1.7477, rel=3.0e-5)
    assert cone_frequency_khz_to_F(280.0, state) == frequency_khz_to_F(280.0, state)
    F = cone_frequency_khz_to_F(280.0, state)
    assert cone_F_to_frequency_khz(F, state) == pytest.approx(280.0)


def test_per_lstar_converters_delegate_verbatim_at_R_eq():
    state = _aps_state()
    R_eq = 599.2768786178892
    assert cone_sigma_L_to_per_m(2.0e-3, R_eq, state) == sigma_L_to_per_m(2.0e-3, R_eq, state)
    assert cone_sigma_L_to_per_mm(2.0e-3, R_eq, state) == sigma_L_to_per_mm(2.0e-3, R_eq, state)
    assert cone_alpha_L_to_per_m(1.25, R_eq, state) == alpha_L_to_per_m(1.25, R_eq, state)
    assert cone_alpha_L_to_per_mm(1.25, R_eq, state) == alpha_L_to_per_mm(1.25, R_eq, state)
    assert cone_wavelength_L_to_mm(5.0, R_eq, state) == wavelength_L_to_mm(5.0, R_eq, state)


def test_station_converters_preserve_scalar_and_array_contracts():
    state = _aps_state()
    scalar = cone_s_mm_to_R_eq(91.83, state)
    assert isinstance(scalar, float)

    s_arr = np.array([10.0, 91.83, 120.0])
    R_arr = cone_s_mm_to_R_eq(s_arr, state)
    assert isinstance(R_arr, np.ndarray)
    assert R_arr.shape == s_arr.shape
    back = cone_R_eq_to_s_mm(R_arr, state)
    assert np.allclose(back, s_arr, rtol=1.0e-12)
    assert isinstance(R_s_from_R_eq(R_arr), np.ndarray)
    assert isinstance(R_eq_from_R_s(599.0), float)


def test_cone_geometry_dataclass_metadata():
    geom = ConeGeometry(half_angle_deg=7.0)
    block = geom.to_dict()
    assert block["type"] == "sharp_cone_mangler"
    assert block["mangler_factor"] == 3.0
    assert block["half_angle_deg"] == 7.0
    assert block["transverse_curvature_terms"] == "omitted"
    assert ConeGeometry().to_dict()["half_angle_deg"] is None
    with pytest.raises(ValueError):
        ConeGeometry(half_angle_deg=95.0)


# ---------------------------------------------------------------------------
# (b) N-factor: cone path integral, factor 3 over the same R_eq window
# ---------------------------------------------------------------------------

def test_cone_n_factor_multiplier_is_three_times_plate_convention():
    assert cone_n_factor_multiplier() == pytest.approx(6.0, rel=1.0e-15)
    assert cone_n_factor_multiplier(2.0) == pytest.approx(6.0, rel=1.0e-15)
    # R = sqrt(2 Re_x) convention has plate multiplier 1.0 -> cone 3.0
    assert cone_n_factor_multiplier(1.0) == pytest.approx(3.0, rel=1.0e-15)
    with pytest.raises(ValueError):
        cone_n_factor_multiplier(0.0)


def test_n_factor_factor_three_constant_sigma():
    sigma0 = 2.0e-3
    R = np.linspace(700.0, 1000.0, 21)
    rows = [{"R_L": float(r), "sigma_L": sigma0} for r in R]

    plate = _integrate_full_path(rows, dx_over_lstar_per_dR=2.0)
    cone = _integrate_full_path(rows, dx_over_lstar_per_dR=cone_n_factor_multiplier())
    assert plate["signed_N"][-1] == pytest.approx(1.2, rel=1.0e-12)
    assert cone["signed_N"][-1] == pytest.approx(3.6, rel=1.0e-12)
    assert cone["signed_N"][-1] / plate["signed_N"][-1] == pytest.approx(3.0, rel=1.0e-12)

    # Equivalent dR_s parameterization: N = integral 2*sqrt(3)*sigma_L dR_s
    n_via_R_s = 2.0 * math.sqrt(3.0) * sigma0 * (R_s_from_R_eq(1000.0) - R_s_from_R_eq(700.0))
    assert n_via_R_s == pytest.approx(3.6, rel=1.0e-12)


def test_n_factor_factor_three_linear_sigma():
    a = 1.0e-6
    R = np.linspace(700.0, 1000.0, 31)
    rows = [{"R_L": float(r), "sigma_L": a * float(r)} for r in R]
    # Trapezoid is exact for the linear integrand:
    # N_plate = 2*a*(1000^2 - 700^2)/2 = a*510000 = 0.51
    plate = _integrate_full_path(rows, dx_over_lstar_per_dR=2.0)["signed_N"][-1]
    cone = _integrate_full_path(rows, dx_over_lstar_per_dR=6.0)["signed_N"][-1]
    assert plate == pytest.approx(0.51, rel=1.0e-12)
    assert cone == pytest.approx(1.53, rel=1.0e-12)


def test_cone_n_matches_physical_arclength_integration():
    state = _aps_state()
    sigma0 = 2.0e-3
    R = np.linspace(700.0, 1000.0, 4001)
    s_m = cone_R_eq_to_s_m(R, state)
    sigma_per_m = sigma_L_to_per_m(sigma0 * np.ones_like(R), R, state)
    N_phys = float(_TRAPZ(sigma_per_m, s_m))
    assert N_phys == pytest.approx(3.6, rel=1.0e-8)


def test_cone_n_factor_helper():
    R = np.linspace(700.0, 1000.0, 21)
    sigma = 2.0e-3 * np.ones_like(R)
    result = cone_n_factor(sigma, R, clip_negative=False)
    assert result["N"][0] == 0.0
    assert result["N"][-1] == pytest.approx(3.6, rel=1.0e-12)
    assert result["multiplier"] == pytest.approx(6.0)

    clipped = cone_n_factor(-sigma, R)
    assert clipped["N"][-1] == 0.0

    with pytest.raises(ValueError):
        cone_n_factor(sigma, R[::-1])


# ---------------------------------------------------------------------------
# (c) Runner integration: dry-run contract, columns, manifest
# ---------------------------------------------------------------------------

def test_add_dimensional_columns_cone_maps_station_to_surface_distance():
    state = _aps_state()
    R_eq = float(cone_s_mm_to_R_eq(91.83, state))
    F_280 = float(frequency_khz_to_F(280.0, state))
    row = {
        "R_L": R_eq,
        "freq_parameter": F_280,
        "sigma_L": 2.0e-3,
        "alpha_r_L": 1.25,
        "alpha_i_L": -2.0e-3,
        "wavelength_L": 5.0,
        "phase_speed_L": 0.92,
        "lower_neutral_R_L": R_eq,
    }
    plate_out = _add_dimensional_columns(row, state)
    cone_out = _add_dimensional_columns(row, state, "cone")

    assert cone_out["x_mm"] == pytest.approx(91.83, rel=2.0e-4)
    assert plate_out["x_mm"] == pytest.approx(91.83 / 3.0, rel=2.0e-4)
    assert cone_out["x_eq_mm"] == pytest.approx(plate_out["x_mm"], rel=1.0e-12)
    assert "x_eq_mm" not in plate_out

    # Per-L* quantities are identical at the same R_eq: only the station
    # label changes between cone and plate.
    assert cone_out["L_star_m"] == pytest.approx(plate_out["L_star_m"], rel=1.0e-12)
    assert cone_out["sigma_per_m"] == pytest.approx(plate_out["sigma_per_m"], rel=1.0e-12)
    assert cone_out["sigma_per_m"] == pytest.approx(2.0e-3 * 858.0 / (7.313e-5 * R_eq), rel=1.0e-12)
    assert cone_out["frequency_khz"] == pytest.approx(280.0)
    assert cone_out["lower_neutral_x_mm"] == pytest.approx(91.83, rel=2.0e-4)
    assert plate_out["lower_neutral_x_mm"] == pytest.approx(91.83 / 3.0, rel=2.0e-4)


def test_cone_dry_run_emits_mangler_postprocess_and_R_eq_grid(tmp_path, capsys):
    main([
        "--preset", "aps-paper-baseline",
        "--geometry", "cone",
        "--cone-half-angle-deg", "7",
        "--quality", "smoke",
        "--output-dir", str(tmp_path),
        "--dry-run",
    ])
    captured = capsys.readouterr().out

    # Postprocess inherits the cone multiplier exactly as the custom pair.
    assert "--r-convention custom" in captured
    assert "--dx-over-lstar-per-dr 6" in captured

    # x bounds are surface distance s; the solver grid is the R_eq grid.
    state = _aps_state()
    r_min = float(cone_s_mm_to_R_eq(10.0, state))
    r_max = float(cone_s_mm_to_R_eq(120.0, state))
    assert r_min == pytest.approx(float(x_mm_to_R_L(10.0 / 3.0, state)), rel=1.0e-12)
    assert (
        f"dimensional_range=s_mm=[10, 120] -> R_eq=[{r_min:.6g}, {r_max:.6g}]"
        in captured
    )
    assert f"--r-min {r_min}" in captured
    assert f"--r-max {r_max}" in captured
    # Frequency conversion is geometry-independent and unchanged.
    assert "F*1e4=1.5417" in captured


def test_plate_default_dry_run_is_unchanged(tmp_path, capsys):
    main([
        "--preset", "aps-paper-baseline",
        "--quality", "smoke",
        "--output-dir", str(tmp_path),
        "--dry-run",
    ])
    captured = capsys.readouterr().out
    assert "dimensional_range=x_mm=[10, 120]" in captured
    assert "--r-convention" not in captured
    assert "--dx-over-lstar-per-dr" not in captured
    assert "Mangler" not in captured
    assert "CONE" not in captured
    assert "s_mm=" not in captured


def test_half_angle_flag_requires_cone_geometry(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([
            "--quality", "smoke",
            "--output-dir", str(tmp_path),
            "--cone-half-angle-deg", "7",
            "--dry-run",
        ])


def test_manifest_records_cone_geometry_block(tmp_path):
    args = _resolve_case(parse_args([
        "--preset", "aps-paper-baseline",
        "--geometry", "cone",
        "--cone-half-angle-deg", "7",
        "--quality", "smoke",
        "--output-dir", str(tmp_path),
    ]))
    artifacts = _artifact_paths(tmp_path)
    _write_manifest(artifacts["manifest"], args, QUALITY["smoke"], artifacts)
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))

    geometry = manifest["geometry"]
    assert geometry["type"] == "sharp_cone_mangler"
    assert geometry["mangler_factor"] == 3.0
    assert geometry["half_angle_deg"] == 7.0
    assert geometry["ds_over_lstar_per_dR_eq"] == 6.0
    assert geometry["transverse_curvature_terms"] == "omitted"
    assert "post-shock" in geometry["edge_state_meaning"]


def test_manifest_for_plate_has_no_geometry_block(tmp_path):
    args = _resolve_case(parse_args([
        "--quality", "smoke",
        "--output-dir", str(tmp_path),
    ]))
    artifacts = _artifact_paths(tmp_path)
    _write_manifest(artifacts["manifest"], args, QUALITY["smoke"], artifacts)
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert "geometry" not in manifest
