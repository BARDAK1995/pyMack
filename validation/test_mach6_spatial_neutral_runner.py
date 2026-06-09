"""Validation for the canonical Mach-6 spatial-neutral runner."""

from __future__ import annotations

import argparse
import csv
import json

from pymack.scales import DimensionalEdgeState, frequency_khz_to_F

from scripts.run_mach6_spatial_neutral_case import (
    _write_dimensional_outputs,
    main,
)


def test_runner_dry_run_emits_single_sweep_commands(tmp_path, capsys):
    out_dir = tmp_path / "mach6_case"
    main([
        "--quality",
        "smoke",
        "--output-dir",
        str(out_dir),
        "--dry-run",
    ])

    captured = capsys.readouterr().out
    assert "compute_spatial_fixed_frequency_curves.py" in captured
    assert "postprocess_spatial_amplification.py" in captured
    assert "plot_spatial_neutral_envelope.py" in captured
    assert "--backend pymack_dense" in captured
    assert "--selection pymack_continuation" in captured
    assert "--sigma-display-limit 0.00325" in captured
    assert "stitched" not in captured.lower()
    assert "smooth" not in captured.lower()


def test_dimensional_runner_dry_run_emits_paper_conversions(tmp_path, capsys):
    out_dir = tmp_path / "aps_case"
    main([
        "--preset",
        "aps-paper-baseline",
        "--quality",
        "smoke",
        "--output-dir",
        str(out_dir),
        "--dry-run",
    ])

    captured = capsys.readouterr().out
    assert "dimensional_range=x_mm=[10, 120]" in captured
    assert "selected_frequency=247 kHz ->" in captured
    assert "F*1e4=1.5417" in captured
    assert "--ma 5.85" in captured
    assert "--gas nitrogen" in captured
    assert "--profile-family power_law" in captured
    assert "--viscosity-exponent 0.74" in captured
    assert "postprocess_spatial_amplification.py" in captured


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_dimensional_output_metadata_records_units_and_policy(tmp_path):
    state = DimensionalEdgeState(
        U_e=858.0,
        nu_e=7.313e-5,
        T_e=52.0,
        M_e=5.85,
        gamma=1.4,
        gas="nitrogen",
        unit_reynolds_per_m=1.176e7,
    )
    F_247 = frequency_khz_to_F(247.0, state)
    F_280 = frequency_khz_to_F(280.0, state)
    artifacts = {
        "growth_csv": tmp_path / "growth.csv",
        "amplification_curves_csv": tmp_path / "amplification.csv",
        "neutral_csv": tmp_path / "neutral.csv",
        "dimensional_growth_csv": tmp_path / "spatial_fixed_frequency_growth_curves_dimensional.csv",
        "dimensional_amplification_csv": tmp_path / "spatial_fixed_frequency_amplification_curves_dimensional.csv",
        "dimensional_neutral_csv": tmp_path / "spatial_fixed_frequency_neutral_envelope_dimensional.csv",
        "dimensional_neutral_png": tmp_path / "spatial_neutral_envelope_dimensional.png",
        "dimensional_metadata": tmp_path / "spatial_dimensional_metadata.json",
        "selected_growth_png": tmp_path / "selected_frequency_growth_dimensional.png",
        "selected_amplification_png": tmp_path / "selected_frequency_amplification_dimensional.png",
        "selected_phase_wavelength_png": tmp_path / "selected_frequency_phase_speed_wavelength_dimensional.png",
    }
    growth_rows = []
    amp_rows = []
    for F in (F_247, F_280):
        for R, sigma in ((700.0, -1.0e-3), (850.0, 2.0e-3), (1030.0, -8.0e-4)):
            base = {
                "freq_parameter": F,
                "R_L": R,
                "omega_L": F * R,
                "sigma_L": sigma,
                "wavelength_L": 5.0,
                "alpha_r_L": 1.25,
                "alpha_i_L": -sigma,
                "phase_speed_L": 0.92,
            }
            growth_rows.append(base)
            amp = dict(base)
            amp.update({
                "lower_neutral_R_L": 744.0,
                "upper_neutral_R_L": 1038.0,
                "N_signed_from_lower": max(0.0, 2.0e-3 * (R - 744.0)),
                "amplification_signed_from_lower": 1.1,
            })
            amp_rows.append(amp)
    _write_csv(artifacts["growth_csv"], growth_rows)
    _write_csv(artifacts["amplification_curves_csv"], amp_rows)
    _write_csv(
        artifacts["neutral_csv"],
        [
            {
                "freq_parameter": F_280,
                "lower_neutral_R_L": 744.0,
                "upper_neutral_R_L": 1038.0,
                "peak_growth_R_L": 850.0,
                "peak_wavelength_L": 5.0,
            }
        ],
    )
    args = argparse.Namespace(
        edge_state=state,
        ma=5.85,
        gas="nitrogen",
        t_wall=300.0,
        selected_frequency_khz_values=[280.0],
    )

    metadata = _write_dimensional_outputs(artifacts, args)

    assert artifacts["dimensional_growth_csv"].exists()
    assert artifacts["dimensional_neutral_png"].exists()
    assert artifacts["selected_amplification_png"].exists()
    assert metadata["units"]["x_units"] == "mm"
    assert metadata["units"]["frequency_units"] == "kHz"
    assert metadata["units"]["growth_units"] == "1/mm"
    assert metadata["formulas"]["N"] == "N = integral sigma_phys dx = integral 2*sigma_L dR_L"
    assert metadata["plot_policy"]["stitching"] == "none"
    assert metadata["plot_policy"]["smoothing"] == "none"

    saved = json.loads(artifacts["dimensional_metadata"].read_text(encoding="utf-8"))
    assert saved["neutral_windows_for_selected"]["280.0"]["lower_neutral_x_mm"] > 0.0
