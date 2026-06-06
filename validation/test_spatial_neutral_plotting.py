"""Validation tests for spatial-neutral plotting policy."""

from __future__ import annotations

import csv
import json

from scripts.plot_spatial_neutral_envelope import plot_neutral_envelope


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_spatial_neutral_plot_records_single_sweep_policy(tmp_path):
    growth_csv = tmp_path / "growth.csv"
    envelope_csv = tmp_path / "neutral.csv"
    output_png = tmp_path / "neutral.png"
    metadata_path = tmp_path / "neutral_metadata.json"

    _write_csv(
        growth_csv,
        [
            {"R_L": 100.0, "freq_parameter": 1.0e-4, "sigma_L": -1.0e-3},
            {"R_L": 200.0, "freq_parameter": 1.0e-4, "sigma_L": 2.0e-3},
            {"R_L": 300.0, "freq_parameter": 1.0e-4, "sigma_L": -1.5e-3},
            {"R_L": 100.0, "freq_parameter": 2.0e-4, "sigma_L": -2.0e-3},
            {"R_L": 200.0, "freq_parameter": 2.0e-4, "sigma_L": 1.5e-3},
            {"R_L": 300.0, "freq_parameter": 2.0e-4, "sigma_L": -1.0e-3},
        ],
    )
    _write_csv(
        envelope_csv,
        [
            {
                "freq_parameter": 1.0e-4,
                "lower_neutral_R_L": 150.0,
                "upper_neutral_R_L": 250.0,
            },
            {
                "freq_parameter": 2.0e-4,
                "lower_neutral_R_L": 140.0,
                "upper_neutral_R_L": 260.0,
            },
        ],
    )

    metadata = plot_neutral_envelope(
        growth_csv=growth_csv,
        envelope_csv=envelope_csv,
        output_png=output_png,
        output_metadata=metadata_path,
        title="test neutral curve",
        frequency_scale="raw",
        x_min=None,
        x_max=None,
        f_min=None,
        f_max=None,
        n_levels=9,
        sigma_display_limit=0.0025,
        zero_contour=False,
    )

    assert output_png.exists()
    assert metadata["status"] == "computed_single_sweep"
    assert metadata["stitching"] == "none"
    assert metadata["smoothing"] == "none"
    assert metadata["sigma_data_range"] == [-0.002, 0.002]
    assert metadata["sigma_display_limits"] == [-0.0025, 0.0025]

    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved["n_finite_growth_rows"] == 6
    assert saved["n_neutral_rows"] == 2
