"""Extract neutral branches from fixed-frequency spatial growth curves."""

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


def _read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            for key in (
                "freq_parameter",
                "R_L",
                "omega_L",
                "sigma_L",
                "alpha_r_L",
                "alpha_i_L",
                "wavelength_L",
            ):
                if key in out:
                    try:
                        out[key] = float(out[key])
                    except ValueError:
                        out[key] = math.nan
            rows.append(out)
    return rows


def _interpolate_crossing(left, right):
    r0 = float(left["R_L"])
    r1 = float(right["R_L"])
    s0 = float(left["sigma_L"])
    s1 = float(right["sigma_L"])
    if s1 == s0:
        weight = 0.0
    else:
        weight = (0.0 - s0) / (s1 - s0)
    out = {
        "freq_parameter": float(left["freq_parameter"]),
        "R_L": float(r0 + weight * (r1 - r0)),
        "sigma_L": 0.0,
        "crossing_direction_R": (
            "stable_to_unstable" if s0 < 0.0 and s1 > 0.0 else
            "unstable_to_stable" if s0 > 0.0 and s1 < 0.0 else
            "touch_or_exact"
        ),
    }
    for key in ("omega_L", "alpha_r_L", "alpha_i_L", "wavelength_L"):
        if key not in left or key not in right:
            continue
        v0 = float(left[key])
        v1 = float(right[key])
        if math.isfinite(v0) and math.isfinite(v1):
            out[key] = float(v0 + weight * (v1 - v0))
        else:
            out[key] = math.nan
    return out


def _extract_neutrals(rows):
    records = []
    summaries = []
    freqs = sorted({float(row["freq_parameter"]) for row in rows})
    for freq in freqs:
        freq_rows = sorted([
            row for row in rows
            if math.isclose(float(row["freq_parameter"]), freq, rel_tol=0.0, abs_tol=1.0e-14)
            and math.isfinite(float(row["R_L"]))
            and math.isfinite(float(row["sigma_L"]))
        ], key=lambda row: float(row["R_L"]))
        crossings = []
        for left, right in zip(freq_rows[:-1], freq_rows[1:]):
            s0 = float(left["sigma_L"])
            s1 = float(right["sigma_L"])
            if s0 == 0.0:
                item = dict(left)
                item["crossing_direction_R"] = "touch_or_exact"
                item["sigma_L"] = 0.0
                crossings.append(item)
            elif s0 * s1 < 0.0:
                crossings.append(_interpolate_crossing(left, right))
        stable_to_unstable = [
            item for item in crossings
            if item.get("crossing_direction_R") == "stable_to_unstable"
        ]
        unstable_to_stable = [
            item for item in crossings
            if item.get("crossing_direction_R") == "unstable_to_stable"
        ]
        if stable_to_unstable:
            lower = dict(stable_to_unstable[0])
            lower["branch_label"] = "lower"
            records.append(lower)
        else:
            lower = None
        if unstable_to_stable:
            upper = dict(unstable_to_stable[-1])
            upper["branch_label"] = "upper"
            records.append(upper)
        else:
            upper = None
        summaries.append({
            "freq_parameter": freq,
            "n_crossings": len(crossings),
            "lower_R_L": math.nan if lower is None else float(lower["R_L"]),
            "upper_R_L": math.nan if upper is None else float(upper["R_L"]),
            "status": "ok" if lower is not None and upper is not None else "missing_branch",
        })
    records.sort(key=lambda row: (row["branch_label"], float(row["R_L"])))
    return records, summaries


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, records, summaries, title):
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    colors = {"lower": "#3423a6", "upper": "#e56b2f"}
    for label in ("lower", "upper"):
        rows = [
            row for row in records
            if row.get("branch_label") == label and math.isfinite(float(row["R_L"]))
        ]
        rows.sort(key=lambda row: float(row["R_L"]))
        if not rows:
            continue
        ax.plot(
            [float(row["R_L"]) for row in rows],
            [float(row["freq_parameter"]) for row in rows],
            "o-",
            lw=2.0,
            ms=4,
            color=colors[label],
            label=f"{label} neutral",
        )
    ok = [
        row for row in summaries
        if row.get("status") == "ok"
        and math.isfinite(float(row["lower_R_L"]))
        and math.isfinite(float(row["upper_R_L"]))
    ]
    for row in ok:
        f = float(row["freq_parameter"])
        ax.plot(
            [float(row["lower_R_L"]), float(row["upper_R_L"])],
            [f, f],
            color="0.65",
            lw=0.8,
            alpha=0.45,
            zorder=0,
        )
    ax.set_xlabel(r"$R_L=\sqrt{Re_x}=U_eL^*/\nu_e$")
    ax.set_ylabel(r"$F=\omega_L/R_L$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Fixed-frequency spatial neutral curve")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    records, summaries = _extract_neutrals(rows)

    neutral_csv = output_dir / "fixed_frequency_spatial_neutral_branches.csv"
    summary_csv = output_dir / "fixed_frequency_spatial_neutral_summary.csv"
    png_path = output_dir / "fixed_frequency_spatial_neutral_curve.png"
    metadata_path = output_dir / "fixed_frequency_spatial_neutral_metadata.json"

    _write_csv(neutral_csv, records)
    _write_csv(summary_csv, summaries)
    _plot(png_path, records, summaries, args.title)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "status": "diagnostic_not_paper_certified",
            "source_csv": str(input_csv).replace("\\", "/"),
            "quantity": "spatial neutral branches from fixed-frequency sigma_L(R_L)=0 crossings",
            "branch_convention": (
                "lower is the first stable_to_unstable crossing as R_L increases; "
                "upper is the last unstable_to_stable crossing as R_L increases."
            ),
            "note": (
                "This postprocess follows each fixed-frequency tracked branch. "
                "It is more appropriate for propagated waves than a pointwise "
                "maximum-growth contour, but final production still needs direct "
                "predictor-corrector neutral continuation."
            ),
        }, handle, indent=2)

    print(f"neutral_csv={neutral_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"png={png_path}")
    print(f"metadata={metadata_path}")
    for row in summaries:
        print(
            f"F={float(row['freq_parameter']):.6g}: "
            f"status={row['status']}, "
            f"lower={float(row['lower_R_L']):.3f}, "
            f"upper={float(row['upper_R_L']):.3f}"
        )


if __name__ == "__main__":
    main()
