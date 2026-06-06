"""Plot a spatial neutral envelope from one fixed-frequency growth sweep.

This script is intentionally conservative: it does not stitch runs, smooth the
growth field, or alter the neutral branches.  The contour is drawn from the
sampled growth CSV and the branch overlay is drawn from the neutral-envelope
CSV produced by ``postprocess_spatial_amplification.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_float_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def _finite_growth_rows(rows):
    finite = []
    for row in rows:
        R = float(row.get("R_L", math.nan))
        freq = float(row.get("freq_parameter", math.nan))
        sigma = float(row.get("sigma_L", math.nan))
        if math.isfinite(R) and math.isfinite(freq) and math.isfinite(sigma):
            finite.append((R, freq, sigma))
    return finite


def _build_rectangular_grid(finite_rows):
    R_values = np.array(sorted({row[0] for row in finite_rows}), dtype=float)
    F_values = np.array(sorted({row[1] for row in finite_rows}), dtype=float)
    r_index = {value: idx for idx, value in enumerate(R_values)}
    f_index = {value: idx for idx, value in enumerate(F_values)}
    Z = np.full((len(F_values), len(R_values)), np.nan, dtype=float)
    for R, freq, sigma in finite_rows:
        Z[f_index[freq], r_index[R]] = sigma
    return R_values, F_values, Z


def _display_frequency(freq_values, scale):
    if scale == "raw":
        return freq_values
    if scale == "times1e4":
        return 1.0e4 * freq_values
    raise ValueError(f"unknown frequency scale: {scale}")


def _frequency_label(scale):
    if scale == "raw":
        return r"$F=\omega_L/R_L$"
    if scale == "times1e4":
        return r"$F \times 10^4$"
    raise ValueError(f"unknown frequency scale: {scale}")


def plot_neutral_envelope(
    *,
    growth_csv: Path,
    envelope_csv: Path,
    output_png: Path,
    output_metadata: Path | None,
    title: str,
    frequency_scale: str,
    x_min: float | None,
    x_max: float | None,
    f_min: float | None,
    f_max: float | None,
    n_levels: int,
    sigma_display_limit: float | None,
    zero_contour: bool,
):
    growth_rows = _read_float_rows(growth_csv)
    envelope_rows = _read_float_rows(envelope_csv)
    finite_rows = _finite_growth_rows(growth_rows)
    if not finite_rows:
        raise ValueError(f"no finite growth rows found in {growth_csv}")

    R_values, F_values, sigma_grid = _build_rectangular_grid(finite_rows)
    F_display = _display_frequency(F_values, frequency_scale)

    sigma_masked = np.ma.masked_invalid(sigma_grid)
    max_abs = float(np.nanmax(np.abs(sigma_grid)))
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0
    display_abs = max_abs
    if sigma_display_limit is not None:
        if sigma_display_limit <= 0.0:
            raise ValueError("--sigma-display-limit must be positive")
        display_abs = float(sigma_display_limit)
    levels = np.linspace(-display_abs, display_abs, int(n_levels))

    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    contour = ax.contourf(
        R_values,
        F_display,
        sigma_masked,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    if zero_contour:
        try:
            ax.contour(
                R_values,
                F_display,
                sigma_masked,
                levels=[0.0],
                colors="0.15",
                linewidths=1.1,
            )
        except ValueError:
            pass

    valid_envelope = [
        row for row in envelope_rows
        if math.isfinite(float(row.get("freq_parameter", math.nan)))
        and math.isfinite(float(row.get("lower_neutral_R_L", math.nan)))
        and math.isfinite(float(row.get("upper_neutral_R_L", math.nan)))
    ]
    valid_envelope.sort(key=lambda row: float(row["freq_parameter"]))
    if valid_envelope:
        freq = np.array([float(row["freq_parameter"]) for row in valid_envelope])
        F_env = _display_frequency(freq, frequency_scale)
        lower = np.array([float(row["lower_neutral_R_L"]) for row in valid_envelope])
        upper = np.array([float(row["upper_neutral_R_L"]) for row in valid_envelope])
        ax.plot(lower, F_env, "o-", color="#3210a8", lw=2.4, ms=4.5, label="lower neutral")
        ax.plot(upper, F_env, "o-", color="#ff7433", lw=2.4, ms=4.5, label="upper neutral")
        ax.legend(loc="upper right", frameon=True, fontsize=12)

    ax.set_xlabel(r"$R_L=\sqrt{Re_x}=U_eL^*/\nu_e$", fontsize=14)
    ax.set_ylabel(_frequency_label(frequency_scale), fontsize=14)
    ax.set_title(title, fontsize=20, pad=12)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if f_min is not None or f_max is not None:
        ax.set_ylim(
            bottom=None if f_min is None else _display_frequency(np.array([f_min]), frequency_scale)[0],
            top=None if f_max is None else _display_frequency(np.array([f_max]), frequency_scale)[0],
        )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.36, linestyle="--")
    colorbar = fig.colorbar(contour, ax=ax, pad=0.035)
    colorbar.set_label(r"$\sigma_L=-\mathrm{Im}(\alpha_L)$", fontsize=13)
    colorbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    metadata = {
        "status": "computed_single_sweep",
        "growth_csv": str(growth_csv),
        "neutral_envelope_csv": str(envelope_csv),
        "output_png": str(output_png),
        "frequency_scale": frequency_scale,
        "zero_contour": bool(zero_contour),
        "smoothing": "none",
        "stitching": "none",
        "sigma_data_range": [
            float(np.nanmin(sigma_grid)),
            float(np.nanmax(sigma_grid)),
        ],
        "sigma_display_limits": [-display_abs, display_abs],
        "n_growth_rows": len(growth_rows),
        "n_finite_growth_rows": len(finite_rows),
        "n_neutral_rows": len(valid_envelope),
        "R_range": [float(np.nanmin(R_values)), float(np.nanmax(R_values))],
        "F_range": [float(np.nanmin(F_values)), float(np.nanmax(F_values))],
    }
    if output_metadata is not None:
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        with output_metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--growth-csv", required=True)
    parser.add_argument("--neutral-envelope-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-metadata", default=None)
    parser.add_argument(
        "--title",
        default="Spatial neutral curve",
    )
    parser.add_argument(
        "--frequency-scale",
        choices=["raw", "times1e4"],
        default="raw",
    )
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--f-min", type=float, default=None)
    parser.add_argument("--f-max", type=float, default=None)
    parser.add_argument("--n-levels", type=int, default=33)
    parser.add_argument(
        "--sigma-display-limit",
        type=float,
        default=None,
        help=(
            "Optional symmetric color scale for sigma_L. This changes only the "
            "display levels; the plotted neutral branches and source CSV are "
            "not modified."
        ),
    )
    parser.add_argument("--zero-contour", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = plot_neutral_envelope(
        growth_csv=Path(args.growth_csv),
        envelope_csv=Path(args.neutral_envelope_csv),
        output_png=Path(args.output_png),
        output_metadata=Path(args.output_metadata) if args.output_metadata else None,
        title=args.title,
        frequency_scale=args.frequency_scale,
        x_min=args.x_min,
        x_max=args.x_max,
        f_min=args.f_min,
        f_max=args.f_max,
        n_levels=args.n_levels,
        sigma_display_limit=args.sigma_display_limit,
        zero_contour=args.zero_contour,
    )
    print(f"output_png={metadata['output_png']}")
    if args.output_metadata:
        print(f"metadata={args.output_metadata}")
    print(
        "plot_policy="
        f"stitching={metadata['stitching']}, smoothing={metadata['smoothing']}, "
        f"status={metadata['status']}"
    )


if __name__ == "__main__":
    main()
