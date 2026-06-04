"""Plot fixed-frequency growth curves from a Mach-6 diagnostic map CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_map(path):
    rows_by_R = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            R = float(row["R_L"])
            alpha = float(row["alpha_L"])
            omega_r = float(row["omega_r_L"])
            omega_i = float(row["omega_i_L"])
            c_r = float(row["c_r"])
            c_i = float(row["c_i"])
            if np.isfinite([R, alpha, omega_r, omega_i, c_r, c_i]).all():
                rows_by_R[R].append({
                    "alpha": alpha,
                    "omega_r": omega_r,
                    "omega_i": omega_i,
                    "c_r": c_r,
                    "c_i": c_i,
                })
    return rows_by_R


def _group_velocity(alpha, omega_r):
    order = np.argsort(alpha)
    alpha = alpha[order]
    omega_r = omega_r[order]
    return np.gradient(omega_r, alpha, edge_order=1)


def _interp_unique(x, y, x0):
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, first = np.unique(np.round(x, 12), return_index=True)
    y_unique = y[first]
    if len(x_unique) < 2 or x0 < x_unique[0] or x0 > x_unique[-1]:
        return np.nan
    return float(np.interp(x0, x_unique, y_unique))


def fixed_frequency_curves(rows_by_R, frequencies, *, r_min=None, r_max=None,
                           frequency_mode="fixed_omega"):
    R_values = np.array(sorted(rows_by_R), dtype=float)
    if r_min is not None:
        R_values = R_values[R_values >= float(r_min)]
    if r_max is not None:
        R_values = R_values[R_values <= float(r_max)]

    curves = {
        float(freq): {
            "R_L": [],
            "omega_i_L": [],
            "sigma_gaster_R": [],
            "alpha_interp": [],
        }
        for freq in frequencies
    }
    for R in R_values:
        row = rows_by_R[float(R)]
        alpha = np.array([item["alpha"] for item in row], dtype=float)
        omega_r = np.array([item["omega_r"] for item in row], dtype=float)
        omega_i = np.array([item["omega_i"] for item in row], dtype=float)
        vg = _group_velocity(alpha, omega_r)
        sigma = np.divide(
            omega_i,
            vg,
            out=np.full_like(omega_i, np.nan),
            where=np.isfinite(vg) & (np.abs(vg) > 1.0e-12),
        )

        for freq in frequencies:
            freq = float(freq)
            omega_target = freq * R if frequency_mode == "fixed_physical" else freq
            curves[freq]["R_L"].append(float(R))
            curves[freq]["omega_i_L"].append(_interp_unique(omega_r, omega_i, omega_target))
            curves[freq]["sigma_gaster_R"].append(_interp_unique(omega_r, sigma, omega_target))
            curves[freq]["alpha_interp"].append(_interp_unique(omega_r, alpha, omega_target))

    for data in curves.values():
        for key, values in data.items():
            data[key] = np.asarray(values, dtype=float)
    return curves


def _write_csv(path, curves):
    freqs = list(curves)
    R = curves[freqs[0]]["R_L"]
    fieldnames = ["R_L"]
    for freq in freqs:
        tag = f"{freq:.5f}".replace(".", "p")
        fieldnames.extend([
            f"omega_i_L__omega_{tag}",
            f"sigma_gaster_R__omega_{tag}",
            f"alpha_L__omega_{tag}",
        ])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, R_val in enumerate(R):
            row = {"R_L": float(R_val)}
            for freq in freqs:
                tag = f"{freq:.5f}".replace(".", "p")
                row[f"omega_i_L__omega_{tag}"] = float(curves[freq]["omega_i_L"][i])
                row[f"sigma_gaster_R__omega_{tag}"] = float(curves[freq]["sigma_gaster_R"][i])
                row[f"alpha_L__omega_{tag}"] = float(curves[freq]["alpha_interp"][i])
            writer.writerow(row)


def _plot(path, curves, title, frequency_mode):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(curves)))
    for color, (freq, data) in zip(colors, curves.items()):
        if frequency_mode == "fixed_physical":
            label = rf"$\omega_{{r,L}}/R_L={freq:.1e}$"
        else:
            label = rf"$\omega_{{r,L}}={freq:.2f}$"
        axes[0].plot(data["R_L"], data["omega_i_L"], "o-", color=color, label=label)
        axes[1].plot(data["R_L"], data["sigma_gaster_R"], "o-", color=color, label=label)

    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel(r"temporal growth $\omega_{i,L}$")
    axes[0].set_title(title)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].axhline(0.0, color="0.2", lw=0.8)
    axes[1].set_xlabel(r"$R_L = \sqrt{Re_x} = U_e L^*/\nu_e$")
    axes[1].set_ylabel(r"Gaster spatial diagnostic $\sigma_R$")

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frequencies", default="0.11,0.15,0.19,0.23,0.27")
    parser.add_argument(
        "--frequency-mode",
        choices=["fixed_omega", "fixed_physical"],
        default="fixed_omega",
        help=(
            "fixed_omega holds nondimensional omega_r,L constant; "
            "fixed_physical holds omega_r,L/R_L constant so omega_r,L grows with R_L."
        ),
    )
    parser.add_argument("--r-min", type=float, default=300.0)
    parser.add_argument("--r-max", type=float, default=1200.0)
    parser.add_argument(
        "--title",
        default="Mach 6 nitrogen, Tw/Te=5.55: fixed-frequency growth curves",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    freqs = [float(item) for item in args.frequencies.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_R = _read_map(args.map_csv)
    curves = fixed_frequency_curves(
        rows_by_R,
        freqs,
        r_min=args.r_min,
        r_max=args.r_max,
        frequency_mode=args.frequency_mode,
    )
    csv_path = output_dir / "fixed_frequency_growth_curves.csv"
    png_path = output_dir / "fixed_frequency_growth_curves.png"
    _write_csv(csv_path, curves)
    _plot(png_path, curves, args.title, args.frequency_mode)
    print(f"csv={csv_path}")
    print(f"png={png_path}")
    for freq, data in curves.items():
        finite = np.isfinite(data["omega_i_L"])
        positive = finite & (data["omega_i_L"] > 0.0)
        print(
            f"omega={freq:.5f}: finite={np.count_nonzero(finite)}, "
            f"positive={np.count_nonzero(positive)}, "
            f"min_omega_i={np.nanmin(data['omega_i_L']):.6e}, "
            f"max_omega_i={np.nanmax(data['omega_i_L']):.6e}"
        )


if __name__ == "__main__":
    main()
