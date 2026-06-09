"""Compare a pyMack Mach-5.35 neutral curve with Sean's benchmark curve."""

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


def _read_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = math.nan
            rows.append(out)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_branch(rows: list[dict[str, float]], x_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (row["frequency_khz"], row[x_key])
        for row in rows
        if math.isfinite(row.get("frequency_khz", math.nan))
        and math.isfinite(row.get(x_key, math.nan))
    ]
    pairs.sort()
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _to_reynolds_f(
    frequency_khz: np.ndarray,
    x_mm: np.ndarray,
    *,
    unit_reynolds_per_m: float,
    edge_velocity_m_per_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_m = x_mm * 1.0e-3
    frequency_hz = frequency_khz * 1000.0
    nu = edge_velocity_m_per_s / unit_reynolds_per_m
    R = np.sqrt(np.maximum(x_m, 0.0) * unit_reynolds_per_m)
    F = 2.0 * math.pi * frequency_hz * nu / edge_velocity_m_per_s**2
    return R, F


def _branch_errors(py_rows, benchmark_rows, branch: str, min_frequency_khz: float):
    py_key = "lower_neutral_x_mm" if branch == "left" else "upper_neutral_x_mm"
    bench_key = "x_left_mm" if branch == "left" else "x_right_mm"
    bench_f, bench_x = _finite_branch(benchmark_rows, bench_key)
    rows = []
    errors = []
    for row in py_rows:
        f = row.get("frequency_khz", math.nan)
        x = row.get(py_key, math.nan)
        if not (math.isfinite(f) and math.isfinite(x)):
            continue
        if f < min_frequency_khz:
            continue
        benchmark_x = float(np.interp(f, bench_f, bench_x))
        error = x - benchmark_x
        rows.append(
            {
                "branch": branch,
                "frequency_khz": f,
                "pymack_x_mm": x,
                "benchmark_x_mm": benchmark_x,
                "error_mm": error,
                "abs_error_mm": abs(error),
            }
        )
        errors.append(error)
    errors_array = np.array(errors, dtype=float)
    if errors_array.size:
        summary = {
            "branch": branch,
            "min_frequency_khz": min_frequency_khz,
            "n": int(errors_array.size),
            "bias_mm": float(np.mean(errors_array)),
            "mae_mm": float(np.mean(np.abs(errors_array))),
            "rmse_mm": float(np.sqrt(np.mean(errors_array**2))),
            "max_abs_error_mm": float(np.max(np.abs(errors_array))),
        }
    else:
        summary = {
            "branch": branch,
            "min_frequency_khz": min_frequency_khz,
            "n": 0,
            "bias_mm": math.nan,
            "mae_mm": math.nan,
            "rmse_mm": math.nan,
            "max_abs_error_mm": math.nan,
        }
    return rows, summary


def _plot(benchmark_rows, py_rows, output_png: Path, *, unit_reynolds_per_m: float, edge_velocity_m_per_s: float) -> None:
    bench_f, bench_left = _finite_branch(benchmark_rows, "x_left_mm")
    _, bench_right = _finite_branch(benchmark_rows, "x_right_mm")
    py_f_left, py_left = _finite_branch(py_rows, "lower_neutral_x_mm")
    py_f_right, py_right = _finite_branch(py_rows, "upper_neutral_x_mm")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

    ax = axes[0]
    ax.plot(bench_left, bench_f, color="0.05", lw=2.3, label="Sean benchmark")
    ax.plot(bench_right, bench_f, color="0.05", lw=2.3)
    ax.plot(py_left, py_f_left, "o--", color="#1f77b4", lw=2.0, ms=4.0, label="pyMack lower")
    ax.plot(py_right, py_f_right, "s--", color="#d95f02", lw=2.0, ms=4.0, label="pyMack upper")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title("Dimensional neutral envelope")
    ax.set_xlim(0.0, 1050.0)
    ax.set_ylim(90.0, 610.0)
    ax.grid(True, alpha=0.30, linestyle="--")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    bR_l, bF_l = _to_reynolds_f(
        bench_f,
        bench_left,
        unit_reynolds_per_m=unit_reynolds_per_m,
        edge_velocity_m_per_s=edge_velocity_m_per_s,
    )
    bR_r, bF_r = _to_reynolds_f(
        bench_f,
        bench_right,
        unit_reynolds_per_m=unit_reynolds_per_m,
        edge_velocity_m_per_s=edge_velocity_m_per_s,
    )
    pR_l, pF_l = _to_reynolds_f(
        py_f_left,
        py_left,
        unit_reynolds_per_m=unit_reynolds_per_m,
        edge_velocity_m_per_s=edge_velocity_m_per_s,
    )
    pR_r, pF_r = _to_reynolds_f(
        py_f_right,
        py_right,
        unit_reynolds_per_m=unit_reynolds_per_m,
        edge_velocity_m_per_s=edge_velocity_m_per_s,
    )
    ax.plot(bR_l, bF_l * 1.0e4, color="0.05", lw=2.3, label="Sean benchmark")
    ax.plot(bR_r, bF_r * 1.0e4, color="0.05", lw=2.3)
    ax.plot(pR_l, pF_l * 1.0e4, "o--", color="#1f77b4", lw=2.0, ms=4.0, label="pyMack lower")
    ax.plot(pR_r, pF_r * 1.0e4, "s--", color="#d95f02", lw=2.0, ms=4.0, label="pyMack upper")
    ax.set_xlabel(r"$R_L=\sqrt{Re_x}$")
    ax.set_ylabel(r"$10^4 F$")
    ax.set_title("Mack Reynolds-frequency plane")
    ax.grid(True, alpha=0.30, linestyle="--")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Mach 5.35 nitrogen neutral-curve benchmark comparison", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--pymack-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit-reynolds-per-m", type=float, default=1.176e7)
    parser.add_argument("--edge-velocity", type=float, default=857.0)
    parser.add_argument("--summary-min-frequency-khz", type=float, default=200.0)
    args = parser.parse_args(argv)

    benchmark_rows = _read_csv(Path(args.benchmark_csv))
    py_rows = _read_csv(Path(args.pymack_csv))
    output_dir = Path(args.output_dir)
    output_png = output_dir / "pymack_vs_sean_mach5p35_neutral_comparison.png"
    _plot(
        benchmark_rows,
        py_rows,
        output_png,
        unit_reynolds_per_m=args.unit_reynolds_per_m,
        edge_velocity_m_per_s=args.edge_velocity,
    )

    error_rows = []
    summaries = []
    for branch in ("left", "right"):
        rows, summary = _branch_errors(
            py_rows,
            benchmark_rows,
            branch,
            min_frequency_khz=args.summary_min_frequency_khz,
        )
        error_rows.extend(rows)
        summaries.append(summary)
    errors_csv = output_dir / "pymack_vs_sean_mach5p35_neutral_errors.csv"
    _write_csv(errors_csv, error_rows)
    summary_json = output_dir / "pymack_vs_sean_mach5p35_neutral_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "comparison": "pyMack phase-targeted Mach 5.35 neutral curve versus Sean benchmark",
                "summary_min_frequency_khz": args.summary_min_frequency_khz,
                "unit_reynolds_per_m": args.unit_reynolds_per_m,
                "edge_velocity_m_per_s": args.edge_velocity,
                "branches": summaries,
                "artifacts": {
                    "comparison_png": str(output_png),
                    "errors_csv": str(errors_csv),
                },
            },
            handle,
            indent=2,
        )

    print(f"comparison_png={output_png}")
    print(f"errors_csv={errors_csv}")
    print(f"summary_json={summary_json}")
    for summary in summaries:
        print(
            f"{summary['branch']}: n={summary['n']} "
            f"MAE={summary['mae_mm']:.3f} mm RMSE={summary['rmse_mm']:.3f} mm "
            f"bias={summary['bias_mm']:.3f} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
