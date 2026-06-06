"""Compute a diagnostic spatial neutral curve from sigma_L = 0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set BLAS/OpenMP limits before NumPy/SciPy are imported in spawned workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymack import CompressibleBlasiusProfile, make_ozgen_profile  # noqa: E402
from pymack.scales import delta_star_over_lstar  # noqa: E402
from pymack.solver import solve_spatial  # noqa: E402


_WORKER_PROFILE = None
_WORKER_DELTA_OVER_L = None
_WORKER_ARGS = None


def _default_sutherland_s_from_config(config):
    gas_key = str(config["gas"]).strip().lower()
    return float(
        config["sutherland_s"]
        if config["sutherland_s"] is not None
        else (111.0 if gas_key in {"nitrogen", "n2"} else 110.0)
    )


def _make_power_law_profile_from_config(config):
    if config["wall_bc"] != "isothermal":
        raise ValueError("power_law diagnostic path currently requires an isothermal wall")
    if config["tw_over_te"] is None:
        raise ValueError("tw_over_te is required for an isothermal power_law profile")

    target_ratio = float(config["tw_over_te"])
    target_wall = target_ratio * float(config["t_edge"])
    kwargs = dict(
        Ma=config["ma"],
        T_edge=config["t_edge"],
        gamma=config["gamma"],
        Pr=config["pr"],
        omega=config["viscosity_exponent"],
        wall_bc=config["wall_bc"],
        viscosity_model="power_law",
        n_points=config["profile_points"],
        eta_max=config["eta_max"],
    )
    try:
        return CompressibleBlasiusProfile(T_wall=target_wall, **kwargs)
    except RuntimeError:
        # Continue wall temperature from the edge state.  This makes the hot
        # wall baseline robust without changing the final similarity equations.
        profile = CompressibleBlasiusProfile(T_wall=config["t_edge"], **kwargs)
        n_steps = max(6, int(np.ceil(abs(target_ratio - 1.0) / 0.20)))
        for ratio in np.linspace(1.0, target_ratio, n_steps + 1)[1:]:
            profile = CompressibleBlasiusProfile(
                T_wall=float(ratio * config["t_edge"]),
                initial_guess_profile=profile,
                **kwargs,
            )
        return profile


def _make_sutherland_blasius_profile_from_config(config):
    if config["wall_bc"] != "isothermal":
        raise ValueError("sutherland_blasius diagnostic path currently requires an isothermal wall")
    if config["tw_over_te"] is None:
        raise ValueError("tw_over_te is required for an isothermal sutherland_blasius profile")

    target_ratio = float(config["tw_over_te"])
    target_wall = target_ratio * float(config["t_edge"])
    kwargs = dict(
        Ma=config["ma"],
        T_edge=config["t_edge"],
        gamma=config["gamma"],
        Pr=config["pr"],
        wall_bc=config["wall_bc"],
        viscosity_model="sutherland",
        sutherland_S=_default_sutherland_s_from_config(config),
        n_points=config["profile_points"],
        eta_max=config["eta_max"],
    )
    try:
        return CompressibleBlasiusProfile(T_wall=target_wall, **kwargs)
    except RuntimeError:
        profile = CompressibleBlasiusProfile(T_wall=config["t_edge"], **kwargs)
        n_steps = max(6, int(np.ceil(abs(target_ratio - 1.0) / 0.20)))
        for ratio in np.linspace(1.0, target_ratio, n_steps + 1)[1:]:
            profile = CompressibleBlasiusProfile(
                T_wall=float(ratio * config["t_edge"]),
                initial_guess_profile=profile,
                **kwargs,
            )
        return profile


def _make_profile_from_config(config):
    sutherland_s = _default_sutherland_s_from_config(config)
    t_wall = (
        None
        if config["tw_over_te"] is None
        else config["tw_over_te"] * config["t_edge"]
    )
    if config["profile_family"] == "power_law":
        return _make_power_law_profile_from_config(config)
    if config["profile_family"] == "sutherland_blasius":
        return _make_sutherland_blasius_profile_from_config(config)
    return make_ozgen_profile(
        config["ma"],
        T_edge=config["t_edge"],
        T_wall=t_wall,
        gamma=config["gamma"],
        Pr=config["pr"],
        S=sutherland_s,
        n_points=config["profile_points"],
        eta_max=config["eta_max"],
    )


def _worker_init(config):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    global _WORKER_PROFILE, _WORKER_DELTA_OVER_L, _WORKER_ARGS
    _WORKER_ARGS = config
    _WORKER_PROFILE = _make_profile_from_config(config)
    _WORKER_DELTA_OVER_L = float(delta_star_over_lstar(_WORKER_PROFILE))


def _select_spatial_candidate(alphas_delta, *, omega_L, delta_over_l, phase_min, phase_max):
    alphas_delta = np.asarray(alphas_delta, dtype=complex)
    alphas_delta = alphas_delta[np.isfinite(alphas_delta)]
    if len(alphas_delta) == 0:
        return np.nan + 1j * np.nan, 0

    alpha_L = alphas_delta / delta_over_l
    phase_speed = np.divide(
        omega_L,
        alpha_L.real,
        out=np.full_like(alpha_L.real, np.nan, dtype=float),
        where=alpha_L.real != 0.0,
    )
    mask = (
        np.isfinite(phase_speed)
        & (phase_speed >= phase_min)
        & (phase_speed <= phase_max)
        & (alpha_L.real > 0.0)
    )
    candidates = alphas_delta[mask]
    if len(candidates) == 0:
        return np.nan + 1j * np.nan, 0
    sigma = -candidates.imag / delta_over_l
    return complex(candidates[int(np.nanargmax(sigma))]), int(len(candidates))


def _omega_from_frequency_parameter(freq_parameter, R_L, frequency_mode):
    if frequency_mode == "fixed_omega":
        return float(freq_parameter)
    return float(freq_parameter * R_L)


def _frequency_axis_label(frequency_mode):
    if frequency_mode == "fixed_omega":
        return r"$\omega_L$"
    return r"$F=\omega_L/R_L$"


def _frequency_parameter_description(frequency_mode):
    if frequency_mode == "fixed_omega":
        return "omega_L, fixed dimensionless angular frequency"
    return "F = omega_L/R_L"


def _solve_point(task):
    i, j, R_L, freq_parameter = task
    cfg = _WORKER_ARGS
    delta_over_l = _WORKER_DELTA_OVER_L
    omega_L = _omega_from_frequency_parameter(
        freq_parameter,
        R_L,
        cfg["frequency_mode"],
    )
    omega_delta = omega_L * delta_over_l
    Re_delta = float(R_L * delta_over_l)
    target_alpha_L = omega_L / cfg["target_phase_speed"]
    target_alpha_delta = target_alpha_L * delta_over_l
    try:
        alphas_delta, _modes, _y = solve_spatial(
            _WORKER_PROFILE,
            omega_delta,
            Re_delta,
            cfg["ma"],
            cfg["pr"],
            cfg["gamma"],
            N=cfg["N"],
            y_max=cfg["y_max_delta"],
            wall_bc=cfg["wall_bc"],
            target_alpha=target_alpha_delta,
            n_modes=cfg["n_modes"],
            length_scale="delta_star",
            lambda_mu_ratio=cfg["lambda_mu_ratio"],
        )
        alpha_delta, n_candidates = _select_spatial_candidate(
            alphas_delta,
            omega_L=omega_L,
            delta_over_l=delta_over_l,
            phase_min=cfg["phase_min"],
            phase_max=cfg["phase_max"],
        )
        status = "ok" if np.isfinite(alpha_delta) else "no_phase_candidate"
    except Exception as exc:  # pragma: no cover - diagnostic provenance
        alpha_delta = np.nan + 1j * np.nan
        n_candidates = 0
        status = f"error:{type(exc).__name__}"

    alpha_L = (
        alpha_delta / delta_over_l
        if np.isfinite(alpha_delta)
        else np.nan + 1j * np.nan
    )
    alpha_r = float(np.real(alpha_L))
    sigma = float(-np.imag(alpha_L))
    phase_speed = (
        float(omega_L / alpha_r)
        if math.isfinite(alpha_r) and alpha_r != 0.0
        else math.nan
    )
    wavelength = (
        float(2.0 * math.pi / alpha_r)
        if math.isfinite(alpha_r) and alpha_r > 0.0
        else math.nan
    )
    return {
        "i": int(i),
        "j": int(j),
        "R_L": float(R_L),
        "freq_parameter": float(freq_parameter),
        "omega_L": float(omega_L),
        "omega_delta": float(omega_delta),
        "Re_delta": float(Re_delta),
        "alpha_r_L": alpha_r,
        "alpha_i_L": float(np.imag(alpha_L)),
        "sigma_L": sigma,
        "wavelength_L": wavelength,
        "phase_speed_L": phase_speed,
        "n_phase_candidates": int(n_candidates),
        "status": status,
    }


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _interp_record(left, right, freq_cross, frequency_mode):
    f0 = float(left["freq_parameter"])
    f1 = float(right["freq_parameter"])
    if f1 == f0:
        weight = 0.0
    else:
        weight = (freq_cross - f0) / (f1 - f0)

    out = {
        "R_L": float(left["R_L"]),
        "freq_parameter": float(freq_cross),
        "omega_L": _omega_from_frequency_parameter(
            freq_cross,
            float(left["R_L"]),
            frequency_mode,
        ),
        "sigma_L": 0.0,
        "branch_index": None,
    }
    for key in ("alpha_r_L", "alpha_i_L", "wavelength_L", "phase_speed_L"):
        v0 = float(left[key])
        v1 = float(right[key])
        if math.isfinite(v0) and math.isfinite(v1):
            out[key] = float(v0 + weight * (v1 - v0))
        else:
            out[key] = math.nan
    out["status"] = "interpolated_neutral"
    s0 = float(left["sigma_L"])
    s1 = float(right["sigma_L"])
    if s0 < 0.0 and s1 > 0.0:
        out["crossing_direction"] = "stable_to_unstable"
    elif s0 > 0.0 and s1 < 0.0:
        out["crossing_direction"] = "unstable_to_stable"
    else:
        out["crossing_direction"] = "touch_or_exact"
    return out


def _extract_neutrals(records, R_values, freq_values, frequency_mode):
    by_key = {(float(row["R_L"]), float(row["freq_parameter"])): row for row in records}
    neutral_rows = []
    branch_rows = []
    counts = {}
    for R in R_values:
        row = []
        for freq in freq_values:
            item = by_key[(float(R), float(freq))]
            if math.isfinite(float(item["sigma_L"])):
                row.append(item)
        row.sort(key=lambda item: float(item["freq_parameter"]))

        crossings = []
        for left, right in zip(row[:-1], row[1:]):
            s0 = float(left["sigma_L"])
            s1 = float(right["sigma_L"])
            if not (math.isfinite(s0) and math.isfinite(s1)):
                continue
            if s0 == 0.0:
                exact = dict(left)
                exact["crossing_direction"] = "touch_or_exact"
                crossings.append(exact)
            elif s0 * s1 < 0.0:
                f0 = float(left["freq_parameter"])
                f1 = float(right["freq_parameter"])
                f_cross = f0 + (0.0 - s0) * (f1 - f0) / (s1 - s0)
                crossings.append(_interp_record(left, right, f_cross, frequency_mode))
        counts[float(R)] = len(crossings)
        for branch_index, item in enumerate(crossings):
            item["branch_index"] = int(branch_index)
            neutral_rows.append(item)
        if len(crossings) == 1:
            only = dict(crossings[0])
            only["branch_label"] = (
                "upper" if only.get("crossing_direction") == "unstable_to_stable" else "lower"
            )
            branch_rows.append(only)
        elif crossings:
            lower = dict(crossings[0])
            lower["branch_label"] = "lower"
            branch_rows.append(lower)
            upper = dict(crossings[-1])
            upper["branch_label"] = "upper"
            branch_rows.append(upper)
    return neutral_rows, branch_rows, counts


def _plot_map(path, records, branch_rows, R_values, freq_values, title, frequency_mode):
    sigma = np.full((len(R_values), len(freq_values)), np.nan)
    for row in records:
        i = int(row["i"])
        j = int(row["j"])
        sigma[i, j] = float(row["sigma_L"])
    R_grid, F_grid = np.meshgrid(R_values, freq_values, indexing="ij")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    finite = sigma[np.isfinite(sigma)]
    vmax = max(1.0e-6, float(np.nanpercentile(np.abs(finite), 95))) if finite.size else 1.0
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(R_grid, F_grid, sigma, levels=levels, cmap="RdBu_r", extend="both")
    fig.colorbar(cf, ax=ax, label=r"$\sigma_L=-\mathrm{Im}(\alpha_L)$")
    try:
        ax.contour(R_grid, F_grid, sigma, levels=[0.0], colors="black", linewidths=2.0)
    except ValueError:
        pass
    if branch_rows:
        colors = {"lower": "#3b0a9f", "upper": "#e8743b"}
        for branch in ("lower", "upper"):
            rows = [row for row in branch_rows if row.get("branch_label") == branch]
            if not rows:
                continue
            rows.sort(key=lambda item: float(item["R_L"]))
            ax.plot(
                [float(row["R_L"]) for row in rows],
                [float(row["freq_parameter"]) for row in rows],
                "o-",
                color=colors[branch],
                ms=3,
                lw=1.5,
                label=f"{branch} neutral",
            )
    ax.set_xlabel(r"$R_L=\sqrt{Re_x}=U_eL^*/\nu_e$")
    ax.set_ylabel(_frequency_axis_label(frequency_mode))
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--")
    if branch_rows:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_branches(path, branch_rows, title, frequency_mode):
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.4), sharex=True)
    colors = {"lower": "#3b0a9f", "upper": "#e8743b"}
    for label in ("lower", "upper"):
        rows = [row for row in branch_rows if row.get("branch_label") == label]
        if not rows:
            continue
        rows.sort(key=lambda item: float(item["R_L"]))
        R = [float(row["R_L"]) for row in rows]
        axes[0].plot(R, [float(row["freq_parameter"]) for row in rows], "o-", color=colors[label], label=label)
        axes[1].plot(R, [float(row["alpha_r_L"]) for row in rows], "o-", color=colors[label], label=label)
        axes[2].plot(R, [float(row["wavelength_L"]) for row in rows], "o-", color=colors[label], label=label)
    axes[0].set_ylabel(_frequency_axis_label(frequency_mode))
    axes[1].set_ylabel(r"neutral $\alpha_{r,L}$")
    axes[2].set_ylabel(r"neutral $\lambda_L$")
    axes[2].set_xlabel(r"$R_L=\sqrt{Re_x}=U_eL^*/\nu_e$")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-min", type=float, default=400.0)
    parser.add_argument("--r-max", type=float, default=1200.0)
    parser.add_argument("--r-points", type=int, default=33)
    parser.add_argument("--f-min", type=float, default=0.0005)
    parser.add_argument("--f-max", type=float, default=0.0016)
    parser.add_argument("--f-points", type=int, default=45)
    parser.add_argument(
        "--frequency-mode",
        choices=["omega_over_R", "fixed_omega"],
        default="omega_over_R",
        help=(
            "Interpret the scanned frequency parameter as either F=omega_L/R_L "
            "or as the fixed omega_L used by fixed-frequency growth curves."
        ),
    )
    parser.add_argument("--ma", type=float, default=6.0)
    parser.add_argument("--gas", default="nitrogen")
    parser.add_argument("--t-edge", type=float, default=288.0)
    parser.add_argument("--tw-over-te", type=float, default=5.55)
    parser.add_argument("--sutherland-s", type=float, default=None)
    parser.add_argument(
        "--profile-family",
        choices=["sutherland_blasius", "ozgen", "power_law"],
        default="sutherland_blasius",
    )
    parser.add_argument("--viscosity-exponent", type=float, default=0.74)
    parser.add_argument("--pr", type=float, default=0.72)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--wall-bc", choices=["isothermal", "adiabatic"], default="isothermal")
    parser.add_argument("--profile-points", type=int, default=3000)
    parser.add_argument("--eta-max", type=float, default=40.0)
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--y-max-delta", type=float, default=10.0)
    parser.add_argument("--n-modes", type=int, default=28)
    parser.add_argument(
        "--lambda-mu-ratio",
        type=float,
        default=0.0,
        help=(
            "Bulk-viscosity parameter for the compressible operator. 0.0 "
            "uses the Stokes-hypothesis coefficients used by the pyMack "
            "spatial reference path."
        ),
    )
    parser.add_argument("--target-phase-speed", type=float, default=0.86)
    parser.add_argument("--phase-min", type=float, default=0.75)
    parser.add_argument("--phase-max", type=float, default=1.15)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "chapters"
            / "ozgen_kircali_2008"
            / "diagnostics"
            / "mach6_growth_nfactor"
            / "R400_1200_N2_TwTe5p55_spatial_neutral_curve"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gas_key = args.gas.strip().lower()
    sutherland_s = float(
        args.sutherland_s
        if args.sutherland_s is not None
        else (111.0 if gas_key in {"nitrogen", "n2"} else 110.0)
    )
    R_values = np.linspace(args.r_min, args.r_max, args.r_points)
    freq_values = np.linspace(args.f_min, args.f_max, args.f_points)
    config = {
        "ma": float(args.ma),
        "gas": args.gas,
        "t_edge": float(args.t_edge),
        "tw_over_te": None if args.tw_over_te is None else float(args.tw_over_te),
        "sutherland_s": sutherland_s,
        "profile_family": args.profile_family,
        "viscosity_exponent": float(args.viscosity_exponent),
        "pr": float(args.pr),
        "gamma": float(args.gamma),
        "wall_bc": args.wall_bc,
        "frequency_mode": args.frequency_mode,
        "profile_points": int(args.profile_points),
        "eta_max": float(args.eta_max),
        "N": int(args.N),
        "y_max_delta": float(args.y_max_delta),
        "n_modes": int(args.n_modes),
        "lambda_mu_ratio": float(args.lambda_mu_ratio),
        "target_phase_speed": float(args.target_phase_speed),
        "phase_min": float(args.phase_min),
        "phase_max": float(args.phase_max),
    }

    tasks = [
        (i, j, float(R), float(freq))
        for i, R in enumerate(R_values)
        for j, freq in enumerate(freq_values)
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

    records.sort(key=lambda row: (row["R_L"], row["freq_parameter"]))
    neutral_rows, branch_rows, counts = _extract_neutrals(
        records,
        R_values,
        freq_values,
        args.frequency_mode,
    )

    grid_csv = output_dir / "spatial_neutral_growth_grid.csv"
    neutral_csv = output_dir / "spatial_neutral_points.csv"
    branches_csv = output_dir / "spatial_neutral_branches.csv"
    map_png = output_dir / "spatial_neutral_curve_map.png"
    branches_png = output_dir / "spatial_neutral_branches.png"
    metadata_json = output_dir / "spatial_neutral_metadata.json"

    _write_csv(grid_csv, records)
    _write_csv(neutral_csv, neutral_rows)
    _write_csv(branches_csv, branch_rows)
    title = f"Mach {args.ma:g} {args.gas}, Tw/Te={args.tw_over_te:g}: spatial neutral curve"
    _plot_map(map_png, records, branch_rows, R_values, freq_values, title, args.frequency_mode)
    _plot_branches(branches_png, branch_rows, title, args.frequency_mode)

    finite_sigma = np.array([float(row["sigma_L"]) for row in records], dtype=float)
    two_crossing_count = int(sum(1 for value in counts.values() if value >= 2))
    with open(metadata_json, "w", encoding="utf-8") as handle:
        json.dump({
            "status": "diagnostic_not_paper_certified",
            "quantity": "spatial neutral curve from sigma_L = -Im(alpha_L) = 0",
            "frequency_mode": args.frequency_mode,
            "frequency_parameter": _frequency_parameter_description(args.frequency_mode),
            "ma": float(args.ma),
            "gas": args.gas,
            "T_edge_K": float(args.t_edge),
            "T_wall_over_T_edge": None if args.tw_over_te is None else float(args.tw_over_te),
            "T_wall_K": None if args.tw_over_te is None else float(args.tw_over_te * args.t_edge),
            "profile_family": args.profile_family,
            "viscosity_model": (
                "power_law"
                if args.profile_family == "power_law"
                else (
                    "sutherland"
                    if args.profile_family == "sutherland_blasius"
                    else "ozgen_sutherland"
                )
            ),
            "viscosity_exponent": float(args.viscosity_exponent) if args.profile_family == "power_law" else None,
            "sutherland_s_K": None if args.profile_family == "power_law" else float(sutherland_s),
            "wall_bc": args.wall_bc,
            "R_L": [float(v) for v in R_values],
            "freq_parameter": [float(v) for v in freq_values],
            "N": int(args.N),
            "n_modes": int(args.n_modes),
            "phase_speed_filter": [float(args.phase_min), float(args.phase_max)],
            "target_phase_speed": float(args.target_phase_speed),
            "solver": "lst.solver.solve_spatial companion QEP",
            "lambda_mu_ratio": float(args.lambda_mu_ratio),
            "selection": "phase-filtered maximum sigma envelope",
            "n_grid_points": int(len(records)),
            "n_finite_sigma": int(np.count_nonzero(np.isfinite(finite_sigma))),
            "n_positive_sigma": int(np.count_nonzero(finite_sigma[np.isfinite(finite_sigma)] > 0.0)),
            "n_neutral_points": int(len(neutral_rows)),
            "n_branch_points": int(len(branch_rows)),
            "two_or_more_crossing_R_count": two_crossing_count,
            "neutral_counts_by_R_L": {f"{key:.8g}": int(value) for key, value in counts.items()},
            "note": (
                "Diagnostic neutral envelope. It identifies sigma=0 crossings "
                "on a finite frequency grid and linearly interpolates; production "
                "certification still requires direct root continuation and "
                "residual checks."
            ),
        }, handle, indent=2)

    print(f"grid_csv={grid_csv}")
    print(f"neutral_csv={neutral_csv}")
    print(f"branches_csv={branches_csv}")
    print(f"map_png={map_png}")
    print(f"branches_png={branches_png}")
    print(f"metadata={metadata_json}")
    print(f"neutral_points={len(neutral_rows)}")
    print(f"R_with_two_or_more_crossings={two_crossing_count}/{len(R_values)}")
    print_label = "omega" if args.frequency_mode == "fixed_omega" else "F"
    for R in R_values:
        rows = [row for row in branch_rows if math.isclose(float(row["R_L"]), float(R), abs_tol=1.0e-9)]
        if rows:
            summary = ", ".join(
                f"{row['branch_label']} {print_label}={float(row['freq_parameter']):.6g}"
                for row in rows
            )
            print(f"R={R:.1f}: {summary}")


if __name__ == "__main__":
    main()
