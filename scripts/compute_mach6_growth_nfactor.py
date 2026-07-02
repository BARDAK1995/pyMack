"""Mach 6 flat-plate growth-rate and N-factor diagnostic.

This script computes a 2D second-mode temporal growth map for an Ozgen
flat-plate mean flow and integrates a Gaster-style spatial-growth diagnostic
along the supplied R = sqrt(Re_x) coordinate.

The N-factor written here is an N(R) diagnostic.  A dimensional transition
N-factor requires a physical streamwise coordinate and spatial growth rate in
reciprocal physical length units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymack import integrate_n_factor, make_flatplate_profile  # noqa: E402
from pymack.temporal_solver import solve_temporal_2d  # noqa: E402
from pymack.scales import delta_star_over_lstar  # noqa: E402


_WORKER_PROFILE = None
_WORKER_DELTA_OVER_L = None
_WORKER_ARGS = None


def _worker_init(config):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    global _WORKER_PROFILE, _WORKER_DELTA_OVER_L, _WORKER_ARGS
    _WORKER_ARGS = config
    t_wall = (
        None
        if config["tw_over_te"] is None
        else config["tw_over_te"] * config["t_edge"]
    )
    _WORKER_PROFILE = make_flatplate_profile(
        config["ma"],
        T_edge=config["t_edge"],
        T_wall=t_wall,
        gamma=config["gamma"],
        Pr=config["pr"],
        S=config["sutherland_s"],
        n_points=config["profile_points"],
        eta_max=config["eta_max"],
    )
    _WORKER_DELTA_OVER_L = float(delta_star_over_lstar(_WORKER_PROFILE))


def _mode_candidates(c_all, phase_bounds, c_imag_abs_max):
    c_all = np.asarray(c_all)
    mask = (
        np.isfinite(c_all)
        & (c_all.real >= phase_bounds[0])
        & (c_all.real <= phase_bounds[1])
        & (np.abs(c_all.imag) <= c_imag_abs_max)
    )
    return c_all[mask]


def _solve_point(task):
    i, j, R_L, alpha_L = task
    cfg = _WORKER_ARGS
    delta_over_l = _WORKER_DELTA_OVER_L
    Re_delta = float(R_L) * delta_over_l
    alpha_delta = float(alpha_L) * delta_over_l
    try:
        c_all, _, _ = solve_temporal_2d(
            _WORKER_PROFILE,
            alpha_delta,
            Re_delta,
            cfg["ma"],
            Pr=cfg["pr"],
            gamma=cfg["gamma"],
            N=cfg["N"],
            y_max=cfg["y_max_delta"],
            wall_bc=cfg["wall_bc"],
        )
        candidates = _mode_candidates(
            c_all,
            cfg["phase_bounds"],
            cfg["c_imag_abs_max"],
        )
        if cfg["convergence_delta"] > 0 and len(candidates) > 0:
            c_low, _, _ = solve_temporal_2d(
                _WORKER_PROFILE,
                alpha_delta,
                Re_delta,
                cfg["ma"],
                Pr=cfg["pr"],
                gamma=cfg["gamma"],
                N=max(40, cfg["N"] - cfg["convergence_delta"]),
                y_max=cfg["y_max_delta"],
                wall_bc=cfg["wall_bc"],
            )
            candidates_low = _mode_candidates(
                c_low,
                cfg["phase_bounds"],
                cfg["c_imag_abs_max"],
            )
            if len(candidates_low) == 0:
                candidates = np.array([], dtype=complex)
            else:
                candidates = np.array([
                    c for c in candidates
                    if np.min(np.abs(candidates_low - c)) <= cfg["convergence_tol"]
                ], dtype=complex)
        if len(candidates) == 0:
            c = np.nan + 1j * np.nan
            n_candidates = 0
        else:
            c = complex(candidates[int(np.argmax(candidates.imag))])
            n_candidates = int(len(candidates))
        status = "ok" if np.isfinite(c) else "no_candidate"
    except Exception as exc:  # pragma: no cover - diagnostic provenance path
        c = np.nan + 1j * np.nan
        n_candidates = 0
        status = f"error:{type(exc).__name__}"
    return {
        "i": int(i),
        "j": int(j),
        "R_L": float(R_L),
        "alpha_L": float(alpha_L),
        "c_r": float(np.real(c)),
        "c_i": float(np.imag(c)),
        "omega_r_L": float(alpha_L * np.real(c)),
        "omega_i_L": float(alpha_L * np.imag(c)),
        "n_candidates": int(n_candidates),
        "status": status,
    }


def _local_derivative(x, y, index):
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return np.nan
    x_f = x[finite]
    y_f = y[finite]
    idx_map = np.where(finite)[0]
    pos_candidates = np.where(idx_map == index)[0]
    if len(pos_candidates) == 0:
        return np.nan
    pos = int(pos_candidates[0])
    if pos == 0:
        left, right = 0, 1
    elif pos == len(x_f) - 1:
        left, right = len(x_f) - 2, len(x_f) - 1
    else:
        left, right = pos - 1, pos + 1
    dx = x_f[right] - x_f[left]
    if dx == 0.0:
        return np.nan
    return float((y_f[right] - y_f[left]) / dx)


def _build_summary(R_values, alpha_values, c_r, c_i, n_candidates):
    omega_r = c_r * alpha_values[None, :]
    omega_i = c_i * alpha_values[None, :]
    rows = []
    sigma_gaster = np.full(len(R_values), np.nan)

    for i, R_L in enumerate(R_values):
        row_growth = omega_i[i, :]
        finite = np.isfinite(row_growth)
        if not np.any(finite):
            rows.append({
                "R_L": float(R_L),
                "alpha_L_peak": np.nan,
                "c_r_peak": np.nan,
                "c_i_peak": np.nan,
                "omega_r_L_peak": np.nan,
                "omega_i_L_peak": np.nan,
                "group_velocity_L": np.nan,
                "sigma_gaster_R": np.nan,
                "n_finite_alpha": 0,
                "n_positive_alpha": 0,
                "max_candidates_at_peak": 0,
            })
            continue

        j = int(np.nanargmax(row_growth))
        group_velocity = _local_derivative(alpha_values, omega_r[i, :], j)
        if np.isfinite(group_velocity) and abs(group_velocity) > 1.0e-12:
            sigma_gaster[i] = float(omega_i[i, j] / group_velocity)

        rows.append({
            "R_L": float(R_L),
            "alpha_L_peak": float(alpha_values[j]),
            "c_r_peak": float(c_r[i, j]),
            "c_i_peak": float(c_i[i, j]),
            "omega_r_L_peak": float(omega_r[i, j]),
            "omega_i_L_peak": float(omega_i[i, j]),
            "group_velocity_L": float(group_velocity),
            "sigma_gaster_R": float(sigma_gaster[i]),
            "n_finite_alpha": int(np.count_nonzero(finite)),
            "n_positive_alpha": int(np.count_nonzero(row_growth[finite] > 0.0)),
            "max_candidates_at_peak": int(n_candidates[i, j]),
        })

    n_result = integrate_n_factor(
        {"sigma": sigma_gaster, "Re": R_values},
        clip_negative=True,
    )
    for row, N_val in zip(rows, n_result["N"]):
        row["N_R_diagnostic"] = float(N_val)
    return rows, omega_r, omega_i, sigma_gaster


def _write_summary_csv(path, rows):
    fieldnames = [
        "R_L",
        "alpha_L_peak",
        "c_r_peak",
        "c_i_peak",
        "omega_r_L_peak",
        "omega_i_L_peak",
        "group_velocity_L",
        "sigma_gaster_R",
        "N_R_diagnostic",
        "n_finite_alpha",
        "n_positive_alpha",
        "max_candidates_at_peak",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_map_csv(path, records):
    fieldnames = [
        "R_L",
        "alpha_L",
        "c_r",
        "c_i",
        "omega_r_L",
        "omega_i_L",
        "n_candidates",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _plot_outputs(out_dir, R_values, alpha_values, omega_i, summary_rows, case_title):
    summary_R = np.array([row["R_L"] for row in summary_rows], dtype=float)
    summary_growth = np.array([row["omega_i_L_peak"] for row in summary_rows], dtype=float)
    summary_alpha = np.array([row["alpha_L_peak"] for row in summary_rows], dtype=float)
    summary_sigma = np.array([row["sigma_gaster_R"] for row in summary_rows], dtype=float)
    summary_N = np.array([row["N_R_diagnostic"] for row in summary_rows], dtype=float)

    R_grid, alpha_grid = np.meshgrid(R_values, alpha_values, indexing="ij")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    levels = np.r_[-0.01, -0.005, 0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
    cs = ax.contour(R_grid, alpha_grid, omega_i, levels=levels, colors="0.2")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
    ax.plot(summary_R, summary_alpha, "o-", color="crimson", ms=3, label="max omega_i")
    ax.set_xlabel("R = sqrt(Re_x) = Ue L*/nu_e")
    ax.set_ylabel("alpha_L")
    ax.set_title(f"{case_title}, 2D second-mode temporal growth")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mach6_growth_map.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].plot(summary_R, summary_growth, "o-", color="crimson")
    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel("max omega_i,L")
    axes[0].set_title(f"{case_title}: growth envelope and N(R) diagnostic")

    axes[1].plot(summary_R, summary_sigma, "o-", color="navy")
    axes[1].axhline(0.0, color="0.2", lw=0.8)
    axes[1].set_ylabel("sigma_Gaster")

    axes[2].plot(summary_R, summary_N, "o-", color="black")
    axes[2].set_xlabel("R = sqrt(Re_x) = Ue L*/nu_e")
    axes[2].set_ylabel("N_R")

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "mach6_growth_nfactor_summary.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-min", type=float, default=100.0)
    parser.add_argument("--r-max", type=float, default=1200.0)
    parser.add_argument("--r-points", type=int, default=12)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.40)
    parser.add_argument("--alpha-points", type=int, default=25)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ma", type=float, default=6.0)
    parser.add_argument("--t-edge", type=float, default=288.0)
    parser.add_argument("--tw-over-te", type=float, default=None)
    parser.add_argument("--gas", default="nitrogen")
    parser.add_argument("--sutherland-s", type=float, default=None)
    parser.add_argument("--pr", type=float, default=0.72)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--wall-bc", choices=["adiabatic", "isothermal"], default="adiabatic")
    parser.add_argument("--profile-points", type=int, default=3000)
    parser.add_argument("--eta-max", type=float, default=40.0)
    parser.add_argument("--y-max-delta", type=float, default=10.0)
    parser.add_argument("--c-imag-abs-max", type=float, default=0.12)
    parser.add_argument("--convergence-delta", type=int, default=0)
    parser.add_argument("--convergence-tol", type=float, default=0.035)
    parser.add_argument("--phase-min", type=float, default=0.80)
    parser.add_argument("--phase-max", type=float, default=1.05)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "chapters"
            / "ozgen_kircali_2008"
            / "diagnostics"
            / "mach6_growth_nfactor"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gas = str(args.gas)
    gas_key = gas.strip().lower()
    sutherland_s = float(
        args.sutherland_s
        if args.sutherland_s is not None
        else (111.0 if gas_key in {"nitrogen", "n2"} else 110.0)
    )
    wall_label = (
        "adiabatic wall"
        if args.tw_over_te is None
        else f"T_w/T_e={args.tw_over_te:g}"
    )
    wall_slug = (
        "adiabatic"
        if args.tw_over_te is None
        else f"TwTe{args.tw_over_te:.3f}".replace(".", "p")
    )
    gas_slug = "".join(ch if ch.isalnum() else "_" for ch in gas_key).strip("_") or "gas"
    case_title = f"Mach {args.ma:g} {gas}, {wall_label}"

    R_values = np.linspace(args.r_min, args.r_max, args.r_points)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    config = {
        "ma": float(args.ma),
        "t_edge": float(args.t_edge),
        "tw_over_te": None if args.tw_over_te is None else float(args.tw_over_te),
        "gas": gas,
        "sutherland_s": sutherland_s,
        "pr": float(args.pr),
        "gamma": float(args.gamma),
        "wall_bc": args.wall_bc,
        "profile_points": int(args.profile_points),
        "eta_max": float(args.eta_max),
        "y_max_delta": float(args.y_max_delta),
        "N": int(args.N),
        "phase_bounds": (float(args.phase_min), float(args.phase_max)),
        "c_imag_abs_max": float(args.c_imag_abs_max),
        "convergence_delta": int(args.convergence_delta),
        "convergence_tol": float(args.convergence_tol),
    }

    tasks = [
        (i, j, float(R_L), float(alpha_L))
        for i, R_L in enumerate(R_values)
        for j, alpha_L in enumerate(alpha_values)
    ]

    records = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_worker_init,
        initargs=(config,),
    ) as pool:
        futures = [pool.submit(_solve_point, task) for task in tasks]
        for k, fut in enumerate(as_completed(futures), start=1):
            records.append(fut.result())
            if k % max(1, math.ceil(len(futures) / 10)) == 0:
                print(f"completed {k}/{len(futures)} points", flush=True)

    records.sort(key=lambda row: (row["R_L"], row["alpha_L"]))
    c_r = np.full((len(R_values), len(alpha_values)), np.nan)
    c_i = np.full_like(c_r, np.nan)
    n_candidates = np.zeros_like(c_r, dtype=int)
    for row in records:
        i = int(np.argmin(np.abs(R_values - row["R_L"])))
        j = int(np.argmin(np.abs(alpha_values - row["alpha_L"])))
        c_r[i, j] = row["c_r"]
        c_i[i, j] = row["c_i"]
        n_candidates[i, j] = row["n_candidates"]

    summary_rows, _omega_r, omega_i, sigma_gaster = _build_summary(
        R_values,
        alpha_values,
        c_r,
        c_i,
        n_candidates,
    )

    prefix = f"mach{args.ma:g}_{gas_slug}_{wall_slug}".replace(".", "p")
    summary_path = out_dir / f"{prefix}_growth_nfactor_summary.csv"
    map_path = out_dir / f"{prefix}_growth_map.csv"
    metadata_path = out_dir / f"{prefix}_growth_nfactor_metadata.json"
    _write_summary_csv(summary_path, summary_rows)
    _write_map_csv(map_path, records)
    _plot_outputs(out_dir, R_values, alpha_values, omega_i, summary_rows, case_title)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "diagnostic_not_paper_certified",
                "ma": float(args.ma),
                "gas": gas,
                "gas_model": "calorically perfect gas with Ozgen variable transport",
                "sutherland_s_K": sutherland_s,
                "mean_flow": f"Ozgen flat plate, {wall_label}, Sutherland/variable-property profile",
                "T_edge_K": float(args.t_edge),
                "T_wall_K": (
                    None
                    if args.tw_over_te is None
                    else float(args.tw_over_te * args.t_edge)
                ),
                "T_wall_over_T_edge": (
                    None if args.tw_over_te is None else float(args.tw_over_te)
                ),
                "thermal_disturbance_wall_bc": args.wall_bc,
                "mode": "2D second-mode candidates",
                "phase_speed_bounds": [float(args.phase_min), float(args.phase_max)],
                "R_L": [float(v) for v in R_values],
                "alpha_L": [float(v) for v in alpha_values],
                "N_collocation": int(args.N),
                "convergence_delta": int(args.convergence_delta),
                "convergence_tol": float(args.convergence_tol),
                "profile_points": int(args.profile_points),
                "eta_max": float(args.eta_max),
                "y_max_delta": float(args.y_max_delta),
                "n_total_points": int(len(records)),
                "n_finite_points": int(np.count_nonzero(np.isfinite(c_i))),
                "n_positive_points": int(np.count_nonzero(omega_i[np.isfinite(omega_i)] > 0.0)),
                "max_omega_i_L": float(np.nanmax(omega_i)),
                "max_sigma_gaster_R": float(np.nanmax(sigma_gaster)),
                "final_N_R_diagnostic": float(summary_rows[-1]["N_R_diagnostic"]),
                "n_factor_note": (
                    "N_R_diagnostic integrates Gaster sigma over R_L. It is useful "
                    "for branch/path diagnostics, but it is not a dimensional "
                    "transition N-factor without a physical x path."
                ),
            },
            handle,
            indent=2,
        )

    print(f"summary_csv={summary_path}")
    print(f"map_csv={map_path}")
    print(f"metadata_json={metadata_path}")
    print(f"growth_map_png={out_dir / 'mach6_growth_map.png'}")
    print(f"summary_png={out_dir / 'mach6_growth_nfactor_summary.png'}")
    print(f"max_omega_i_L={np.nanmax(omega_i):.8e}")
    print(f"final_N_R_diagnostic={summary_rows[-1]['N_R_diagnostic']:.8e}")


if __name__ == "__main__":
    main()
