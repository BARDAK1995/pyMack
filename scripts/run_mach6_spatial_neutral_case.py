"""Run the canonical Mach-6 spatial growth/neutral-envelope case.

This is the reproducible entry point for the Mach 6 air, Tw/Te=5.88
second-mode spatial calculation used during development.  It intentionally
uses one fixed-frequency sweep and then postprocesses that same sweep; it does
not stitch frequency bands or smooth the computed growth field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYMACK_NO_BANNER", "1")

from pymack.scales import (  # noqa: E402
    DimensionalEdgeState,
    F_to_frequency_khz,
    R_L_to_x_mm,
    alpha_L_to_per_m,
    alpha_L_to_per_mm,
    frequency_khz_to_F,
    lstar_m_from_R_L,
    sigma_L_to_per_m,
    sigma_L_to_per_mm,
    wavelength_L_to_mm,
    x_mm_to_R_L,
)


@dataclass(frozen=True)
class SweepQuality:
    n_frequencies: int
    r_points: int
    workers: int


QUALITY = {
    "smoke": SweepQuality(n_frequencies=5, r_points=89, workers=8),
    "standard": SweepQuality(n_frequencies=41, r_points=89, workers=24),
    "production": SweepQuality(n_frequencies=81, r_points=177, workers=32),
}


APS_PAPER_PRESET = {
    "mach": 5.85,
    "gas": "nitrogen",
    "T_edge_K": 52.0,
    "T_wall_K": 300.0,
    "U_e_m_per_s": 858.0,
    "nu_e_m2_per_s": 7.313e-5,
    "unit_reynolds_per_m": 1.176e7,
    "Pr": 0.72,
    "gamma": 1.4,
    "profile_family": "power_law",
    "viscosity_exponent": 0.74,
    "wall_bc": "isothermal",
}


def _linspace_csv(start: float, stop: float, count: int) -> str:
    if count <= 1:
        return f"{start:.10f}"
    values = [
        start + i * (stop - start) / (count - 1)
        for i in range(count)
    ]
    return ",".join(f"{value:.10f}" for value in values)


def _linspace_values(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    return [
        float(start + i * (stop - start) / (count - 1))
        for i in range(count)
    ]


def _merge_values(base_values, include_values, *, lo: float, hi: float, tol: float = 1.0e-12):
    values = [float(value) for value in base_values]
    for value in include_values or []:
        value = float(value)
        if lo - tol <= value <= hi + tol:
            values.append(value)
    values.sort()
    merged = []
    for value in values:
        if not merged or abs(value - merged[-1]) > tol:
            merged.append(value)
    return merged


def _values_csv(values) -> str:
    return ",".join(f"{float(value):.10f}" for value in values)


def _parse_float_csv(text: str | None) -> list[float]:
    if text is None or str(text).strip() == "":
        return []
    return [
        float(part.strip())
        for part in str(text).split(",")
        if part.strip()
    ]


def _float_or_default(value, default):
    return float(default if value is None else value)


def _str_or_default(value, default):
    return str(default if value is None else value)


def _resolve_case(args):
    """Resolve defaults and dimensional conversions before command creation."""
    if args.preset == "aps-paper-baseline":
        args.dimensional = True

    use_aps_defaults = bool(args.dimensional)
    preset = APS_PAPER_PRESET if use_aps_defaults else {}

    args.ma = _float_or_default(args.ma, preset.get("mach", 6.0))
    args.gas = _str_or_default(args.gas, preset.get("gas", "air"))
    args.t_edge = _float_or_default(args.t_edge, preset.get("T_edge_K", 300.0))
    args.pr = _float_or_default(args.pr, preset.get("Pr", 0.72))
    args.gamma = _float_or_default(args.gamma, preset.get("gamma", 1.4))
    args.wall_bc = _str_or_default(args.wall_bc, preset.get("wall_bc", "isothermal"))
    args.profile_family = _str_or_default(
        args.profile_family,
        preset.get("profile_family", "sutherland_blasius"),
    )
    args.viscosity_exponent = _float_or_default(
        args.viscosity_exponent,
        preset.get("viscosity_exponent", 0.74),
    )
    args.sutherland_s = (
        None
        if args.sutherland_s is None and args.profile_family == "power_law"
        else _float_or_default(args.sutherland_s, 110.4)
    )

    if args.tw_over_te is None:
        if args.t_wall is not None:
            args.tw_over_te = float(args.t_wall) / float(args.t_edge)
        elif use_aps_defaults:
            args.tw_over_te = float(preset["T_wall_K"]) / float(args.t_edge)
        else:
            args.tw_over_te = 5.88
    else:
        args.tw_over_te = float(args.tw_over_te)

    args.t_wall = (
        float(args.t_wall)
        if args.t_wall is not None
        else float(args.tw_over_te) * float(args.t_edge)
    )

    edge_state = None
    selected_frequency_khz = _parse_float_csv(args.selected_frequencies_khz)
    selected_F = []
    if args.dimensional:
        U_e = _float_or_default(args.edge_velocity, preset.get("U_e_m_per_s"))
        nu_e = _float_or_default(args.nu_edge, preset.get("nu_e_m2_per_s"))
        unit_re = _float_or_default(
            args.unit_reynolds_per_m,
            preset.get("unit_reynolds_per_m", U_e / nu_e),
        )
        edge_state = DimensionalEdgeState(
            U_e=U_e,
            nu_e=nu_e,
            T_e=args.t_edge,
            M_e=args.ma,
            gamma=args.gamma,
            gas=args.gas,
            unit_reynolds_per_m=unit_re,
        )
        args.x_min_mm = _float_or_default(args.x_min_mm, 10.0)
        args.x_max_mm = _float_or_default(args.x_max_mm, 120.0)
        args.frequency_min_khz = _float_or_default(args.frequency_min_khz, 150.0)
        args.frequency_max_khz = _float_or_default(args.frequency_max_khz, 500.0)
        if not selected_frequency_khz:
            selected_frequency_khz = [247.0, 280.0, 325.0]

        args.r_min = float(x_mm_to_R_L(args.x_min_mm, edge_state))
        args.r_max = float(x_mm_to_R_L(args.x_max_mm, edge_state))
        args.f_min = float(frequency_khz_to_F(args.frequency_min_khz, edge_state))
        args.f_max = float(frequency_khz_to_F(args.frequency_max_khz, edge_state))
        selected_F = [
            float(frequency_khz_to_F(freq_khz, edge_state))
            for freq_khz in selected_frequency_khz
        ]
    else:
        args.r_min = _float_or_default(args.r_min, 200.0)
        args.r_max = _float_or_default(args.r_max, 3300.0)
        args.f_min = _float_or_default(args.f_min, 6.0e-5)
        args.f_max = _float_or_default(args.f_max, 2.30e-4)
        args.x_min_mm = None
        args.x_max_mm = None
        args.frequency_min_khz = None
        args.frequency_max_khz = None

    args.selected_frequency_khz_values = selected_frequency_khz
    args.selected_F_values = selected_F
    args.edge_state = edge_state
    return args


def _case_slug(args) -> str:
    if args.dimensional:
        return (
            f"APS_M{args.ma:g}_{args.gas}_Tw{args.t_wall:g}K_"
            f"x{args.x_min_mm:g}_{args.x_max_mm:g}mm_"
            f"f{args.frequency_min_khz:g}_{args.frequency_max_khz:g}kHz_"
            f"c{args.phase_min:.2f}_{args.phase_max:.2f}_"
            f"{args.quality}_single_sweep"
        ).replace(".", "p")
    return (
        f"M6_air_TwTe{args.tw_over_te:g}_"
        f"F{args.f_min * 1e6:03.0f}_{args.f_max * 1e6:03.0f}_"
        f"R{args.r_min:g}_{args.r_max:g}_"
        f"c{args.phase_min:.2f}_{args.phase_max:.2f}_"
        f"{args.quality}_single_sweep"
    ).replace(".", "p")


def _default_output_dir(args) -> Path:
    return (
        REPO_ROOT
        / "chapters"
        / "ozgen_kircali_2008"
        / "results"
        / "mach6_spatial_neutral"
        / _case_slug(args)
    )


def _run(cmd, *, cwd: Path, env: dict[str, str], dry_run: bool):
    printable = " ".join(str(part) for part in cmd)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _artifact_paths(output_dir: Path):
    growth_csv = output_dir / "spatial_fixed_frequency_growth_curves.csv"
    amplification_dir = output_dir / "amplification"
    amplification_curves_csv = amplification_dir / "spatial_fixed_frequency_amplification_curves.csv"
    amplification_summary_csv = amplification_dir / "spatial_fixed_frequency_amplification_summary.csv"
    neutral_csv = amplification_dir / "spatial_fixed_frequency_neutral_envelope.csv"
    neutral_png = output_dir / "spatial_neutral_envelope.png"
    neutral_metadata = output_dir / "spatial_neutral_envelope_metadata.json"
    dimensional_growth_csv = output_dir / "spatial_fixed_frequency_growth_curves_dimensional.csv"
    dimensional_amplification_csv = output_dir / "spatial_fixed_frequency_amplification_curves_dimensional.csv"
    dimensional_neutral_csv = output_dir / "spatial_fixed_frequency_neutral_envelope_dimensional.csv"
    dimensional_neutral_png = output_dir / "spatial_neutral_envelope_dimensional.png"
    dimensional_metadata = output_dir / "spatial_dimensional_metadata.json"
    selected_growth_png = output_dir / "selected_frequency_growth_dimensional.png"
    selected_amplification_png = output_dir / "selected_frequency_amplification_dimensional.png"
    selected_phase_wavelength_png = output_dir / "selected_frequency_phase_speed_wavelength_dimensional.png"
    manifest = output_dir / "mach6_spatial_neutral_case_manifest.json"
    return {
        "growth_csv": growth_csv,
        "amplification_dir": amplification_dir,
        "amplification_curves_csv": amplification_curves_csv,
        "amplification_summary_csv": amplification_summary_csv,
        "neutral_csv": neutral_csv,
        "neutral_png": neutral_png,
        "neutral_metadata": neutral_metadata,
        "dimensional_growth_csv": dimensional_growth_csv,
        "dimensional_amplification_csv": dimensional_amplification_csv,
        "dimensional_neutral_csv": dimensional_neutral_csv,
        "dimensional_neutral_png": dimensional_neutral_png,
        "dimensional_metadata": dimensional_metadata,
        "selected_growth_png": selected_growth_png,
        "selected_amplification_png": selected_amplification_png,
        "selected_phase_wavelength_png": selected_phase_wavelength_png,
        "manifest": manifest,
    }


def _read_float_rows(path: Path) -> list[dict]:
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


def _write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _neutral_x_column(row, source_key, output_key, edge_state):
    if _is_finite(row.get(source_key)):
        row[output_key] = float(R_L_to_x_mm(float(row[source_key]), edge_state))
    else:
        row[output_key] = math.nan


def _add_dimensional_columns(row: dict, edge_state: DimensionalEdgeState) -> dict:
    out = dict(row)
    R = float(out.get("R_L", math.nan))
    F = float(out.get("freq_parameter", math.nan))

    if math.isfinite(F):
        out["frequency_khz"] = float(F_to_frequency_khz(F, edge_state))
        out["F_times_1e4"] = 1.0e4 * F
    else:
        out["frequency_khz"] = math.nan
        out["F_times_1e4"] = math.nan

    if math.isfinite(R) and R > 0.0:
        out["x_mm"] = float(R_L_to_x_mm(R, edge_state))
        out["L_star_m"] = float(lstar_m_from_R_L(R, edge_state))
        out["L_star_mm"] = 1000.0 * out["L_star_m"]
    else:
        out["x_mm"] = math.nan
        out["L_star_m"] = math.nan
        out["L_star_mm"] = math.nan

    for source, per_m_key, per_mm_key in (
        ("sigma_L", "sigma_per_m", "sigma_per_mm"),
        ("alpha_r_L", "alpha_r_per_m", "alpha_r_per_mm"),
        ("alpha_i_L", "alpha_i_per_m", "alpha_i_per_mm"),
    ):
        if math.isfinite(R) and R > 0.0 and _is_finite(out.get(source)):
            value = float(out[source])
            if source == "sigma_L":
                out[per_m_key] = float(sigma_L_to_per_m(value, R, edge_state))
                out[per_mm_key] = float(sigma_L_to_per_mm(value, R, edge_state))
            else:
                out[per_m_key] = float(alpha_L_to_per_m(value, R, edge_state))
                out[per_mm_key] = float(alpha_L_to_per_mm(value, R, edge_state))
        else:
            out[per_m_key] = math.nan
            out[per_mm_key] = math.nan

    if math.isfinite(R) and R > 0.0 and _is_finite(out.get("wavelength_L")):
        out["wavelength_mm"] = float(wavelength_L_to_mm(float(out["wavelength_L"]), R, edge_state))
    else:
        out["wavelength_mm"] = math.nan

    if _is_finite(out.get("phase_speed_L")):
        out["phase_speed_m_per_s"] = float(out["phase_speed_L"]) * edge_state.U_e
    else:
        out["phase_speed_m_per_s"] = math.nan

    for source, dest in (
        ("lower_neutral_R_L", "lower_neutral_x_mm"),
        ("upper_neutral_R_L", "upper_neutral_x_mm"),
        ("peak_growth_R_L", "peak_growth_x_mm"),
        ("peak_N_R_L", "peak_N_x_mm"),
        ("peak_N_R_L_from_start", "peak_N_x_mm_from_start"),
        ("peak_N_R_L_from_lower_full_path", "peak_N_x_mm_from_lower_full_path"),
    ):
        if source in out:
            _neutral_x_column(out, source, dest, edge_state)

    if _is_finite(out.get("peak_wavelength_L")) and _is_finite(out.get("peak_growth_R_L")):
        out["peak_wavelength_mm"] = float(
            wavelength_L_to_mm(float(out["peak_wavelength_L"]), float(out["peak_growth_R_L"]), edge_state)
        )
    elif "peak_wavelength_L" in out:
        out["peak_wavelength_mm"] = math.nan

    return out


def _convert_rows_to_dimensional(rows: list[dict], edge_state: DimensionalEdgeState) -> list[dict]:
    return [_add_dimensional_columns(row, edge_state) for row in rows]


def _build_grid(rows, x_key, y_key, z_key):
    finite = [
        (float(row[x_key]), float(row[y_key]), float(row[z_key]))
        for row in rows
        if _is_finite(row.get(x_key)) and _is_finite(row.get(y_key)) and _is_finite(row.get(z_key))
    ]
    if not finite:
        raise ValueError(f"no finite rows for grid keys {x_key}, {y_key}, {z_key}")
    x_values = np.array(sorted({item[0] for item in finite}), dtype=float)
    y_values = np.array(sorted({item[1] for item in finite}), dtype=float)
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    Z = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for x, y, z in finite:
        Z[y_index[y], x_index[x]] = z
    return x_values, y_values, Z


def _plot_dimensional_neutral(
    *,
    growth_rows: list[dict],
    neutral_rows: list[dict],
    output_png: Path,
    title: str,
):
    x_values, freq_values, sigma_grid = _build_grid(
        growth_rows,
        "x_mm",
        "frequency_khz",
        "sigma_per_mm",
    )
    sigma_masked = np.ma.masked_invalid(sigma_grid)
    max_abs = float(np.nanmax(np.abs(sigma_grid)))
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0
    levels = np.linspace(-max_abs, max_abs, 33)

    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    contour = ax.contourf(
        x_values,
        freq_values,
        sigma_masked,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    try:
        ax.contour(x_values, freq_values, sigma_masked, levels=[0.0], colors="0.15", linewidths=1.1)
    except ValueError:
        pass

    lower_neutral = [
        row for row in neutral_rows
        if _is_finite(row.get("frequency_khz"))
        and _is_finite(row.get("lower_neutral_x_mm"))
    ]
    upper_neutral = [
        row for row in neutral_rows
        if _is_finite(row.get("frequency_khz"))
        and _is_finite(row.get("upper_neutral_x_mm"))
    ]
    lower_neutral.sort(key=lambda row: float(row["frequency_khz"]))
    upper_neutral.sort(key=lambda row: float(row["frequency_khz"]))
    if lower_neutral or upper_neutral:
        if lower_neutral:
            freq = np.array([float(row["frequency_khz"]) for row in lower_neutral], dtype=float)
            lower = np.array([float(row["lower_neutral_x_mm"]) for row in lower_neutral], dtype=float)
            ax.plot(lower, freq, "o-", color="#3210a8", lw=2.4, ms=4.5, label="lower neutral")
        if upper_neutral:
            freq = np.array([float(row["frequency_khz"]) for row in upper_neutral], dtype=float)
            upper = np.array([float(row["upper_neutral_x_mm"]) for row in upper_neutral], dtype=float)
            ax.plot(upper, freq, "o-", color="#ff7433", lw=2.4, ms=4.5, label="upper neutral")
        ax.legend(loc="upper right", frameon=True, fontsize=12)

    ax.set_xlabel("x (mm)", fontsize=14)
    ax.set_ylabel("frequency (kHz)", fontsize=14)
    ax.set_title(title, fontsize=19, pad=12)
    ax.grid(True, alpha=0.36, linestyle="--")
    ax.tick_params(labelsize=12)
    colorbar = fig.colorbar(contour, ax=ax, pad=0.035)
    colorbar.set_label(r"$\sigma=-\mathrm{Im}(\alpha)$ [1/mm]", fontsize=13)
    colorbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {
        "sigma_per_mm_range": [
            float(np.nanmin(sigma_grid)),
            float(np.nanmax(sigma_grid)),
        ],
        "x_mm_range": [float(np.nanmin(x_values)), float(np.nanmax(x_values))],
        "frequency_khz_range": [
            float(np.nanmin(freq_values)),
            float(np.nanmax(freq_values)),
        ],
        "n_lower_neutral_rows": int(len(lower_neutral)),
        "n_upper_neutral_rows": int(len(upper_neutral)),
    }


def _selected_rows(rows: list[dict], selected_frequency_khz: list[float]):
    selected = {}
    for target in selected_frequency_khz:
        target = float(target)
        freq_rows = [
            row for row in rows
            if _is_finite(row.get("frequency_khz"))
            and abs(float(row["frequency_khz"]) - target) <= 5.0e-2
        ]
        freq_rows.sort(key=lambda row: float(row.get("x_mm", math.nan)))
        selected[target] = freq_rows
    return selected


def _plot_selected_growth(rows: list[dict], selected_frequency_khz, output_png: Path, title: str):
    selected = _selected_rows(rows, selected_frequency_khz)
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    for target, freq_rows in selected.items():
        finite = [
            row for row in freq_rows
            if _is_finite(row.get("x_mm")) and _is_finite(row.get("sigma_per_mm"))
        ]
        if not finite:
            continue
        ax.plot(
            [float(row["x_mm"]) for row in finite],
            [float(row["sigma_per_mm"]) for row in finite],
            "o-",
            ms=3.5,
            lw=1.8,
            label=f"{target:g} kHz",
        )
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel(r"$\sigma=-\mathrm{Im}(\alpha)$ [1/mm]")
    ax.set_title(f"{title}: selected-wave spatial growth")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def _plot_selected_amplification(rows: list[dict], selected_frequency_khz, output_png: Path, title: str):
    selected = _selected_rows(rows, selected_frequency_khz)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    for target, freq_rows in selected.items():
        finite = [
            row for row in freq_rows
            if _is_finite(row.get("x_mm"))
            and _is_finite(row.get("N_signed_from_lower"))
            and _is_finite(row.get("amplification_signed_from_lower"))
        ]
        if not finite:
            continue
        x = [float(row["x_mm"]) for row in finite]
        axes[0].plot(x, [float(row["N_signed_from_lower"]) for row in finite], "o-", ms=3.5, lw=1.8, label=f"{target:g} kHz")
        axes[1].plot(x, [float(row["amplification_signed_from_lower"]) for row in finite], "o-", ms=3.5, lw=1.8, label=f"{target:g} kHz")
    axes[0].set_ylabel("N from lower neutral")
    axes[1].set_ylabel(r"$A/A_{\mathrm{lower}}$")
    axes[1].set_xlabel("x (mm)")
    axes[0].set_title(f"{title}: signed amplification from lower neutral")
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def _plot_selected_phase_wavelength(rows: list[dict], selected_frequency_khz, output_png: Path, title: str):
    selected = _selected_rows(rows, selected_frequency_khz)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    for target, freq_rows in selected.items():
        finite_phase = [
            row for row in freq_rows
            if _is_finite(row.get("x_mm")) and _is_finite(row.get("phase_speed_m_per_s"))
        ]
        finite_wave = [
            row for row in freq_rows
            if _is_finite(row.get("x_mm")) and _is_finite(row.get("wavelength_mm"))
        ]
        if finite_phase:
            axes[0].plot(
                [float(row["x_mm"]) for row in finite_phase],
                [float(row["phase_speed_m_per_s"]) for row in finite_phase],
                "o-",
                ms=3.5,
                lw=1.8,
                label=f"{target:g} kHz",
            )
        if finite_wave:
            axes[1].plot(
                [float(row["x_mm"]) for row in finite_wave],
                [float(row["wavelength_mm"]) for row in finite_wave],
                "o-",
                ms=3.5,
                lw=1.8,
                label=f"{target:g} kHz",
            )
    axes[0].set_ylabel("phase speed (m/s)")
    axes[1].set_ylabel("wavelength (mm)")
    axes[1].set_xlabel("x (mm)")
    axes[0].set_title(f"{title}: phase speed and wavelength")
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def _neutral_windows_for_selected(neutral_rows, selected_frequency_khz):
    windows = {}
    for target in selected_frequency_khz:
        candidates = [
            row for row in neutral_rows
            if _is_finite(row.get("frequency_khz"))
            and abs(float(row["frequency_khz"]) - float(target)) <= 5.0e-2
        ]
        if not candidates:
            windows[str(target)] = {"status": "not_found"}
            continue
        row = candidates[0]
        lower_finite = _is_finite(row.get("lower_neutral_R_L"))
        upper_finite = _is_finite(row.get("upper_neutral_R_L"))
        if lower_finite and upper_finite:
            status = "ok"
        elif lower_finite:
            status = "open_downstream_lobe"
        else:
            status = "missing_two_crossings"
        windows[str(target)] = {
            "status": status,
            "frequency_khz": float(row.get("frequency_khz", math.nan)),
            "F": float(row.get("freq_parameter", math.nan)),
            "F_times_1e4": float(row.get("F_times_1e4", math.nan)),
            "lower_neutral_R_L": float(row.get("lower_neutral_R_L", math.nan)),
            "upper_neutral_R_L": float(row.get("upper_neutral_R_L", math.nan)),
            "lower_neutral_x_mm": float(row.get("lower_neutral_x_mm", math.nan)),
            "upper_neutral_x_mm": float(row.get("upper_neutral_x_mm", math.nan)),
        }
    return windows


def _write_dimensional_outputs(artifacts, args):
    edge_state = args.edge_state
    growth_rows = _read_float_rows(artifacts["growth_csv"])
    amp_rows = _read_float_rows(artifacts["amplification_curves_csv"])
    neutral_rows = _read_float_rows(artifacts["neutral_csv"])

    growth_dim = _convert_rows_to_dimensional(growth_rows, edge_state)
    amp_dim = _convert_rows_to_dimensional(amp_rows, edge_state)
    neutral_dim = _convert_rows_to_dimensional(neutral_rows, edge_state)

    _write_rows(artifacts["dimensional_growth_csv"], growth_dim)
    _write_rows(artifacts["dimensional_amplification_csv"], amp_dim)
    _write_rows(artifacts["dimensional_neutral_csv"], neutral_dim)

    title = (
        f"Mach {args.ma:g} {args.gas}, Tw={args.t_wall:g} K: "
        "spatial neutral curve"
    )
    neutral_plot_metadata = _plot_dimensional_neutral(
        growth_rows=growth_dim,
        neutral_rows=neutral_dim,
        output_png=artifacts["dimensional_neutral_png"],
        title=title,
    )
    _plot_selected_growth(
        growth_dim,
        args.selected_frequency_khz_values,
        artifacts["selected_growth_png"],
        f"Mach {args.ma:g} {args.gas}, Tw={args.t_wall:g} K",
    )
    _plot_selected_amplification(
        amp_dim,
        args.selected_frequency_khz_values,
        artifacts["selected_amplification_png"],
        f"Mach {args.ma:g} {args.gas}, Tw={args.t_wall:g} K",
    )
    _plot_selected_phase_wavelength(
        growth_dim,
        args.selected_frequency_khz_values,
        artifacts["selected_phase_wavelength_png"],
        f"Mach {args.ma:g} {args.gas}, Tw={args.t_wall:g} K",
    )

    metadata = {
        "status": "computed_single_sweep_dimensional",
        "source_growth_csv": str(artifacts["growth_csv"]),
        "source_amplification_csv": str(artifacts["amplification_curves_csv"]),
        "source_neutral_envelope_csv": str(artifacts["neutral_csv"]),
        "edge_state": edge_state.to_dict(),
        "units": {
            "x_units": "mm",
            "frequency_units": "kHz",
            "growth_units": "1/mm",
            "growth_alternate_units": "1/m",
            "wavelength_units": "mm",
            "phase_speed_units": "m/s",
        },
        "formulas": {
            "x": "x = nu_e * R_L^2 / U_e",
            "L_star": "L* = nu_e * R_L / U_e",
            "F": "F = 2*pi*f_Hz*nu_e/U_e^2",
            "sigma": "sigma_phys = sigma_L/L*",
            "N": "N = integral sigma_phys dx = integral 2*sigma_L dR_L",
            "amplification": "A/A_lower = exp(N_signed_from_lower)",
        },
        "plot_policy": {
            "single_sweep": True,
            "stitching": "none",
            "smoothing": "none",
            "frequency_conversion": "direct F=2*pi*f_Hz*nu_e/U_e^2; no phase-speed proxy",
        },
        "selected_frequencies_khz": [float(value) for value in args.selected_frequency_khz_values],
        "selected_frequency_parameters": [
            {
                "frequency_khz": float(freq_khz),
                "F": float(frequency_khz_to_F(freq_khz, edge_state)),
                "F_times_1e4": 1.0e4 * float(frequency_khz_to_F(freq_khz, edge_state)),
            }
            for freq_khz in args.selected_frequency_khz_values
        ],
        "neutral_windows_for_selected": _neutral_windows_for_selected(
            neutral_dim,
            args.selected_frequency_khz_values,
        ),
        "neutral_plot": neutral_plot_metadata,
        "artifacts": {
            "dimensional_growth_csv": str(artifacts["dimensional_growth_csv"]),
            "dimensional_amplification_csv": str(artifacts["dimensional_amplification_csv"]),
            "dimensional_neutral_csv": str(artifacts["dimensional_neutral_csv"]),
            "dimensional_neutral_png": str(artifacts["dimensional_neutral_png"]),
            "selected_growth_png": str(artifacts["selected_growth_png"]),
            "selected_amplification_png": str(artifacts["selected_amplification_png"]),
            "selected_phase_wavelength_png": str(artifacts["selected_phase_wavelength_png"]),
        },
    }
    artifacts["dimensional_metadata"].parent.mkdir(parents=True, exist_ok=True)
    with artifacts["dimensional_metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def _write_manifest(path: Path, args, quality: SweepQuality, artifacts, dimensional_metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "complete",
        "case": (
            "APS paper baseline Mach-6 dimensional spatial second-mode neutral envelope"
            if args.dimensional
            else "Mach 6 air Tw/Te=5.88 spatial second-mode neutral envelope"
        ),
        "policy": {
            "single_sweep": True,
            "stitching": "none",
            "smoothing": "none",
            "branch_family": "high-phase-speed second mode",
            "dimensional_frequency_conversion": (
                "direct F=2*pi*f_Hz*nu_e/U_e^2"
                if args.dimensional
                else None
            ),
        },
        "parameters": {
            "mach": float(args.ma),
            "gas": args.gas,
            "T_edge_K": float(args.t_edge),
            "T_wall_K": float(args.t_wall),
            "Tw_over_Te": float(args.tw_over_te),
            "Pr": float(args.pr),
            "gamma": float(args.gamma),
            "profile_family": args.profile_family,
            "viscosity_exponent": float(args.viscosity_exponent),
            "sutherland_S_K": None if args.sutherland_s is None else float(args.sutherland_s),
            "R_range": [float(args.r_min), float(args.r_max)],
            "F_range": [float(args.f_min), float(args.f_max)],
            "phase_speed_filter": [float(args.phase_min), float(args.phase_max)],
            "quality": args.quality,
            "base_n_frequencies": int(quality.n_frequencies),
            "n_frequencies": int(getattr(args, "n_frequency_values", quality.n_frequencies)),
            "r_points": int(quality.r_points),
            "workers": int(args.workers or quality.workers),
            "sigma_display_limit": float(args.sigma_display_limit),
        },
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    if args.dimensional:
        manifest["dimensional"] = {
            "preset": args.preset or "custom-dimensional-defaults",
            "edge_state": args.edge_state.to_dict(),
            "x_range_mm": [float(args.x_min_mm), float(args.x_max_mm)],
            "frequency_range_khz": [
                float(args.frequency_min_khz),
                float(args.frequency_max_khz),
            ],
            "selected_frequencies_khz": [
                float(value) for value in args.selected_frequency_khz_values
            ],
            "selected_F_values": [float(value) for value in args.selected_F_values],
        }
        if dimensional_metadata is not None:
            manifest["dimensional"]["neutral_windows_for_selected"] = dimensional_metadata.get(
                "neutral_windows_for_selected",
                {},
            )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY),
        default="production",
        help="Grid density. production reproduces the polished curve.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--preset", choices=["aps-paper-baseline"], default=None)
    parser.add_argument("--dimensional", action="store_true")
    parser.add_argument("--r-min", type=float, default=None)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--f-min", type=float, default=None)
    parser.add_argument("--f-max", type=float, default=None)
    parser.add_argument("--x-min-mm", type=float, default=None)
    parser.add_argument("--x-max-mm", type=float, default=None)
    parser.add_argument("--frequency-min-khz", type=float, default=None)
    parser.add_argument("--frequency-max-khz", type=float, default=None)
    parser.add_argument(
        "--selected-frequencies-khz",
        default=None,
        help="Comma-separated dimensional frequencies to force into the solve grid.",
    )
    parser.add_argument("--edge-velocity", type=float, default=None)
    parser.add_argument("--nu-edge", type=float, default=None)
    parser.add_argument("--unit-reynolds-per-m", type=float, default=None)
    parser.add_argument("--ma", type=float, default=None)
    parser.add_argument("--gas", default=None)
    parser.add_argument("--t-edge", type=float, default=None)
    parser.add_argument("--t-wall", type=float, default=None)
    parser.add_argument("--tw-over-te", type=float, default=None)
    parser.add_argument("--profile-family", choices=["sutherland_blasius", "power_law"], default=None)
    parser.add_argument("--viscosity-exponent", type=float, default=None)
    parser.add_argument("--sutherland-s", type=float, default=None)
    parser.add_argument("--pr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--wall-bc", choices=["isothermal"], default=None)
    parser.add_argument("--phase-min", type=float, default=0.90)
    parser.add_argument("--phase-max", type=float, default=0.97)
    parser.add_argument("--sigma-display-limit", type=float, default=0.00325)
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Reuse an existing growth CSV in output-dir and rerun postprocessing/plotting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and manifest location without executing them.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _resolve_case(parse_args(argv))
    quality = QUALITY[args.quality]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args)
    artifacts = _artifact_paths(output_dir)
    workers = int(args.workers or quality.workers)
    frequency_values = _merge_values(
        _linspace_values(args.f_min, args.f_max, quality.n_frequencies),
        args.selected_F_values,
        lo=float(args.f_min),
        hi=float(args.f_max),
    )
    args.n_frequency_values = len(frequency_values)
    frequencies = _values_csv(frequency_values)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYMACK_NO_BANNER", "1")

    if args.dimensional:
        print(
            "dimensional_range="
            f"x_mm=[{args.x_min_mm:.6g}, {args.x_max_mm:.6g}] -> "
            f"R_L=[{args.r_min:.6g}, {args.r_max:.6g}], "
            f"frequency_khz=[{args.frequency_min_khz:.6g}, {args.frequency_max_khz:.6g}] -> "
            f"F=[{args.f_min:.8g}, {args.f_max:.8g}]",
            flush=True,
        )
        for freq_khz, F_value in zip(args.selected_frequency_khz_values, args.selected_F_values):
            print(
                f"selected_frequency={freq_khz:.6g} kHz -> F={F_value:.8g}, F*1e4={F_value * 1.0e4:.5g}",
                flush=True,
            )

    if not args.skip_compute:
        compute_cmd = [
            sys.executable,
            "scripts/compute_spatial_fixed_frequency_curves.py",
            "--backend",
            "pymack_dense",
            "--workers",
            str(workers),
            "--r-min",
            str(args.r_min),
            "--r-max",
            str(args.r_max),
            "--r-points",
            str(quality.r_points),
            "--frequencies",
            frequencies,
            "--ma",
            str(args.ma),
            "--gas",
            str(args.gas),
            "--t-edge",
            str(args.t_edge),
            "--tw-over-te",
            str(args.tw_over_te),
            "--profile-family",
            str(args.profile_family),
            "--pr",
            str(args.pr),
            "--gamma",
            str(args.gamma),
            "--wall-bc",
            str(args.wall_bc),
            "--profile-points",
            "3000",
            "--eta-max",
            "16",
            "--N",
            "31",
            "--y-max-lstar",
            "30",
            "--dense-eta-nodes",
            "80",
            "--dense-bvp-tol",
            "1e-4",
            "--phase-min",
            str(args.phase_min),
            "--phase-max",
            str(args.phase_max),
            "--selection",
            "pymack_continuation",
            "--output-dir",
            str(output_dir),
        ]
        if args.sutherland_s is not None:
            compute_cmd.extend(["--sutherland-s", str(args.sutherland_s)])
        if args.profile_family == "power_law":
            compute_cmd.extend(["--viscosity-exponent", str(args.viscosity_exponent)])
        _run(compute_cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)

    postprocess_cmd = [
        sys.executable,
        "scripts/postprocess_spatial_amplification.py",
        str(artifacts["growth_csv"]),
        "--output-dir",
        str(artifacts["amplification_dir"]),
        "--view-min",
        str(args.r_min),
        "--view-max",
        str(args.r_max),
        "--plot-x-min",
        str(args.r_min),
        "--plot-x-max",
        str(args.r_max),
        "--title",
        f"Mach {args.ma:g} {args.gas}, Tw/Te={args.tw_over_te:g}",
    ]
    _run(postprocess_cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)

    plot_cmd = [
        sys.executable,
        "scripts/plot_spatial_neutral_envelope.py",
        "--growth-csv",
        str(artifacts["growth_csv"]),
        "--neutral-envelope-csv",
        str(artifacts["neutral_csv"]),
        "--output-png",
        str(artifacts["neutral_png"]),
        "--output-metadata",
        str(artifacts["neutral_metadata"]),
        "--title",
        f"Mach {args.ma:g} {args.gas}, Tw/Te={args.tw_over_te:g}: spatial neutral curve",
        "--frequency-scale",
        "raw",
        "--x-min",
        str(args.r_min),
        "--x-max",
        str(args.r_max),
        "--f-min",
        str(args.f_min),
        "--f-max",
        str(args.f_max),
        "--sigma-display-limit",
        str(args.sigma_display_limit),
    ]
    _run(plot_cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)

    if not args.dry_run:
        dimensional_metadata = None
        if args.dimensional:
            dimensional_metadata = _write_dimensional_outputs(artifacts, args)
            print(f"dimensional_neutral_png={artifacts['dimensional_neutral_png']}")
            print(f"selected_growth_png={artifacts['selected_growth_png']}")
            print(f"selected_amplification_png={artifacts['selected_amplification_png']}")
            print(f"selected_phase_wavelength_png={artifacts['selected_phase_wavelength_png']}")
        _write_manifest(artifacts["manifest"], args, quality, artifacts, dimensional_metadata)
        print(f"manifest={artifacts['manifest']}")
        print(f"neutral_png={artifacts['neutral_png']}")


if __name__ == "__main__":
    main()
