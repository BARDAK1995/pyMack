"""Compute spatial growth curves for fixed physical-frequency parameters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lst import CompressibleBlasiusProfile, make_ozgen_profile  # noqa: E402
from lst.scales import delta_star_over_lstar  # noqa: E402
from lst.solver import solve_spatial  # noqa: E402


SECOND_MODE_ALPHA_MIN_L = 0.6


def _select_alpha(alphas, target_alpha, previous_alpha=None, *, omega_L=None,
                  delta_over_l=1.0, selection="tracked", phase_min=0.75,
                  phase_max=1.15, alpha_min_L=None, alpha_max_L=None):
    alphas = np.asarray(alphas, dtype=complex)
    alphas = alphas[np.isfinite(alphas)]
    if len(alphas) == 0:
        return np.nan + 1j * np.nan
    if omega_L is not None:
        alphas = _phase_filtered_candidates(
            alphas,
            omega_L=omega_L,
            delta_over_l=delta_over_l,
            phase_min=phase_min,
            phase_max=phase_max,
            alpha_min_L=alpha_min_L,
            alpha_max_L=alpha_max_L,
        )
        if len(alphas) == 0:
            return np.nan + 1j * np.nan
    if selection == "max_sigma":
        alpha_L = alphas / delta_over_l
        sigma = -alpha_L.imag
        return alphas[int(np.nanargmax(sigma))]
    if previous_alpha is not None and np.isfinite(previous_alpha):
        scale = max(abs(previous_alpha), 1.0)
        candidates = alphas[np.abs(alphas - previous_alpha) < 0.35 * scale]
        if len(candidates) > 0:
            return candidates[int(np.argmin(np.abs(candidates - previous_alpha)))]
    return alphas[int(np.argmin(np.abs(alphas - target_alpha)))]


def _phase_filtered_candidates(alphas, *, omega_L, delta_over_l, phase_min=0.75,
                               phase_max=1.15, alpha_min_L=None,
                               alpha_max_L=None):
    """Return second-mode-like spatial candidates in L* units."""
    alphas = np.asarray(alphas, dtype=complex)
    alphas = alphas[np.isfinite(alphas)]
    if len(alphas) == 0:
        return alphas
    alpha_L = alphas / delta_over_l
    c_phase = np.divide(
        omega_L,
        alpha_L.real,
        out=np.full_like(alpha_L.real, np.nan, dtype=float),
        where=alpha_L.real != 0.0,
    )
    mask = np.isfinite(c_phase) & (c_phase >= phase_min) & (c_phase <= phase_max)
    if alpha_min_L is not None:
        mask &= alpha_L.real >= alpha_min_L
    if alpha_max_L is not None:
        mask &= alpha_L.real <= alpha_max_L
    return alphas[mask]


def _solve_spatial_candidates(profile, args, *, omega_L, R, delta_over_l,
                              target_alpha_delta):
    omega_delta = omega_L * delta_over_l
    Re_delta = float(R * delta_over_l)
    alphas_delta, _modes, _y = solve_spatial(
        profile,
        omega_delta,
        Re_delta,
        args.ma,
        args.pr,
        args.gamma,
        N=args.N,
        y_max=args.y_max_delta,
        wall_bc=args.wall_bc,
        target_alpha=target_alpha_delta,
        n_modes=args.n_modes,
        length_scale="delta_star",
    )
    return np.asarray(alphas_delta, dtype=complex), omega_delta, Re_delta


def _track_anchored_branch(candidate_rows, anchor_i, anchor_alpha):
    tracked = [np.nan + 1j * np.nan for _ in candidate_rows]
    tracked[anchor_i] = complex(anchor_alpha)

    prev = complex(anchor_alpha)
    for i in range(anchor_i + 1, len(candidate_rows)):
        candidates = candidate_rows[i]["candidates"]
        if len(candidates) == 0 or not np.isfinite(prev):
            continue
        scale = max(abs(prev), 1.0)
        distances = np.abs(candidates - prev)
        idx = int(np.argmin(distances))
        if distances[idx] <= 0.45 * scale:
            tracked[i] = complex(candidates[idx])
            prev = tracked[i]

    prev = complex(anchor_alpha)
    for i in range(anchor_i - 1, -1, -1):
        candidates = candidate_rows[i]["candidates"]
        if len(candidates) == 0 or not np.isfinite(prev):
            continue
        scale = max(abs(prev), 1.0)
        distances = np.abs(candidates - prev)
        idx = int(np.argmin(distances))
        if distances[idx] <= 0.45 * scale:
            tracked[i] = complex(candidates[idx])
            prev = tracked[i]

    return tracked


def _default_sutherland_s(args):
    gas_key = args.gas.strip().lower()
    return float(
        args.sutherland_s
        if args.sutherland_s is not None
        else (111.0 if gas_key in {"nitrogen", "n2"} else 110.0)
    )


def _make_power_law_profile(args):
    if args.wall_bc != "isothermal":
        raise ValueError("power_law diagnostic path currently requires an isothermal wall")
    if args.tw_over_te is None:
        raise ValueError("--tw-over-te is required for an isothermal power_law profile")

    target_ratio = float(args.tw_over_te)
    target_wall = target_ratio * float(args.t_edge)

    kwargs = dict(
        Ma=args.ma,
        T_edge=args.t_edge,
        gamma=args.gamma,
        Pr=args.pr,
        omega=args.viscosity_exponent,
        wall_bc=args.wall_bc,
        viscosity_model="power_law",
        n_points=args.profile_points,
        eta_max=args.eta_max,
    )
    try:
        return CompressibleBlasiusProfile(T_wall=target_wall, **kwargs)
    except RuntimeError:
        # Very hot isothermal walls can be harder to converge directly.  Use
        # wall-temperature continuation so each BVP starts from a nearby state.
        profile = CompressibleBlasiusProfile(T_wall=args.t_edge, **kwargs)
        n_steps = max(6, int(np.ceil(abs(target_ratio - 1.0) / 0.20)))
        for ratio in np.linspace(1.0, target_ratio, n_steps + 1)[1:]:
            profile = CompressibleBlasiusProfile(
                T_wall=float(ratio * args.t_edge),
                initial_guess_profile=profile,
                **kwargs,
            )
        return profile


def _make_profile(args):
    sutherland_s = _default_sutherland_s(args)
    t_wall = None if args.tw_over_te is None else args.tw_over_te * args.t_edge
    if args.profile_family == "power_law":
        profile = _make_power_law_profile(args)
        transport = {
            "profile_family": "power_law",
            "viscosity_model": "power_law",
            "viscosity_exponent": float(args.viscosity_exponent),
            "sutherland_s_K": None,
        }
    else:
        profile = make_ozgen_profile(
            args.ma,
            T_edge=args.t_edge,
            T_wall=t_wall,
            gamma=args.gamma,
            Pr=args.pr,
            S=sutherland_s,
            n_points=args.profile_points,
            eta_max=args.eta_max,
        )
        transport = {
            "profile_family": "ozgen",
            "viscosity_model": "ozgen_sutherland",
            "viscosity_exponent": None,
            "sutherland_s_K": float(sutherland_s),
        }
    return profile, sutherland_s, transport


def _apply_mode_family_defaults(args):
    """Apply explicit mode-family filters before any roots are selected.

    For the Mach-6 hot-wall work in this repo, the intended branch is the
    trapped/acoustic second-mode family.  A phase-speed window alone also admits
    long-wave low-alpha candidates; those produced the absurd amplification
    ratios.  The alpha-family filter makes that branch identity explicit.
    """
    if args.mode_family == "second_mode" and args.alpha_min_l is None:
        args.alpha_min_l = SECOND_MODE_ALPHA_MIN_L
    return args


def compute_curves(args):
    profile, sutherland_s, transport = _make_profile(args)
    delta_over_l = float(delta_star_over_lstar(profile))
    R_L = np.linspace(args.r_min, args.r_max, args.r_points)
    freqs = np.array([float(item) for item in args.frequencies.split(",") if item.strip()])

    if args.selection in {"anchored_max_sigma", "anchored_resolve"}:
        return _compute_curves_anchored(args, profile, delta_over_l, freqs, R_L, sutherland_s, transport)

    records = []
    for freq in freqs:
        previous = None
        for R in R_L:
            omega_L = float(freq * R if args.frequency_mode == "fixed_physical" else freq)
            omega_delta = omega_L * delta_over_l
            Re_delta = float(R * delta_over_l)
            target_alpha_L = omega_L / args.c_phase
            target_alpha_delta = target_alpha_L * delta_over_l
            target_alpha = target_alpha_delta if previous is None else previous
            try:
                alphas_delta, _modes, _y = solve_spatial(
                    profile,
                    omega_delta,
                    Re_delta,
                    args.ma,
                    args.pr,
                    args.gamma,
                    N=args.N,
                    y_max=args.y_max_delta,
                    wall_bc=args.wall_bc,
                    target_alpha=target_alpha,
                    n_modes=args.n_modes,
                    length_scale="delta_star",
                )
                alpha_delta = _select_alpha(
                    alphas_delta,
                    target_alpha_delta,
                    previous,
                    omega_L=omega_L,
                    delta_over_l=delta_over_l,
                    selection=args.selection,
                    phase_min=args.phase_min,
                    phase_max=args.phase_max,
                    alpha_min_L=args.alpha_min_l,
                    alpha_max_L=args.alpha_max_l,
                )
                filtered = _phase_filtered_candidates(
                    alphas_delta,
                    omega_L=omega_L,
                    delta_over_l=delta_over_l,
                    phase_min=args.phase_min,
                    phase_max=args.phase_max,
                    alpha_min_L=args.alpha_min_l,
                    alpha_max_L=args.alpha_max_l,
                )
                status = "ok" if np.isfinite(alpha_delta) else "no_candidate"
            except Exception as exc:  # pragma: no cover - diagnostic provenance
                alpha_delta = np.nan + 1j * np.nan
                alphas_delta = np.array([], dtype=complex)
                filtered = np.array([], dtype=complex)
                status = f"error:{type(exc).__name__}"

            previous = alpha_delta if np.isfinite(alpha_delta) else previous
            alpha_L = alpha_delta / delta_over_l if np.isfinite(alpha_delta) else np.nan + 1j * np.nan
            records.append({
                "freq_parameter": float(freq),
                "R_L": float(R),
                "omega_L": float(omega_L),
                "omega_delta": float(omega_delta),
                "Re_delta": float(Re_delta),
                "alpha_r_L": float(np.real(alpha_L)),
                "alpha_i_L": float(np.imag(alpha_L)),
                "wavelength_L": float(
                    2.0 * np.pi / np.real(alpha_L)
                    if np.isfinite(alpha_L) and np.real(alpha_L) > 0.0
                    else np.nan
                ),
                "sigma_L": float(-np.imag(alpha_L)),
                "alpha_r_delta": float(np.real(alpha_delta)),
                "alpha_i_delta": float(np.imag(alpha_delta)),
                "sigma_delta": float(-np.imag(alpha_delta)),
                "n_candidates": int(len(alphas_delta)),
                "n_filtered_candidates": int(len(filtered)),
                "target_alpha_L": float(target_alpha_L),
                "status": status,
            })
    return records, delta_over_l, freqs, R_L, sutherland_s, transport


def _compute_curves_anchored(args, profile, delta_over_l, freqs, R_L, sutherland_s, transport):
    records = []
    for freq in freqs:
        candidate_rows = []
        for R in R_L:
            omega_L = float(freq * R if args.frequency_mode == "fixed_physical" else freq)
            target_alpha_L = omega_L / args.c_phase
            target_alpha_delta = target_alpha_L * delta_over_l
            try:
                alphas_delta, omega_delta, Re_delta = _solve_spatial_candidates(
                    profile,
                    args,
                    omega_L=omega_L,
                    R=float(R),
                    delta_over_l=delta_over_l,
                    target_alpha_delta=target_alpha_delta,
                )
                candidates = _phase_filtered_candidates(
                    alphas_delta,
                    omega_L=omega_L,
                    delta_over_l=delta_over_l,
                    phase_min=args.phase_min,
                    phase_max=args.phase_max,
                    alpha_min_L=args.alpha_min_l,
                    alpha_max_L=args.alpha_max_l,
                )
                status = "ok" if len(candidates) > 0 else "no_filtered_candidate"
            except Exception as exc:  # pragma: no cover - diagnostic provenance
                alphas_delta = np.array([], dtype=complex)
                candidates = np.array([], dtype=complex)
                omega_delta = omega_L * delta_over_l
                Re_delta = float(R * delta_over_l)
                status = f"error:{type(exc).__name__}"
            candidate_rows.append({
                "R_L": float(R),
                "omega_L": omega_L,
                "omega_delta": float(omega_delta),
                "Re_delta": float(Re_delta),
                "target_alpha_L": float(target_alpha_L),
                "target_alpha_delta": complex(target_alpha_delta),
                "candidates": candidates,
                "n_candidates": int(len(alphas_delta)),
                "n_filtered_candidates": int(len(candidates)),
                "status": status,
            })

        anchor = None
        for i, row in enumerate(candidate_rows):
            candidates = row["candidates"]
            if len(candidates) == 0:
                continue
            sigma_L = -candidates.imag / delta_over_l
            j = int(np.nanargmax(sigma_L))
            value = float(sigma_L[j])
            if anchor is None or value > anchor[0]:
                anchor = (value, i, complex(candidates[j]))

        tracked = (
            [np.nan + 1j * np.nan for _ in candidate_rows]
            if anchor is None
            else (
                _resolve_anchored_branch(args, profile, delta_over_l, candidate_rows, anchor[1], anchor[2])
                if args.selection == "anchored_resolve"
                else _track_anchored_branch(candidate_rows, anchor[1], anchor[2])
            )
        )

        for row, alpha_delta in zip(candidate_rows, tracked):
            alpha_L = (
                alpha_delta / delta_over_l
                if np.isfinite(alpha_delta)
                else np.nan + 1j * np.nan
            )
            row_status = row["status"]
            if row_status == "ok" and not np.isfinite(alpha_delta):
                row_status = "not_tracked"
            records.append({
                "freq_parameter": float(freq),
                "R_L": float(row["R_L"]),
                "omega_L": float(row["omega_L"]),
                "omega_delta": float(row["omega_delta"]),
                "Re_delta": float(row["Re_delta"]),
                "alpha_r_L": float(np.real(alpha_L)),
                "alpha_i_L": float(np.imag(alpha_L)),
                "wavelength_L": float(
                    2.0 * np.pi / np.real(alpha_L)
                    if np.isfinite(alpha_L) and np.real(alpha_L) > 0.0
                    else np.nan
                ),
                "sigma_L": float(-np.imag(alpha_L)),
                "alpha_r_delta": float(np.real(alpha_delta)),
                "alpha_i_delta": float(np.imag(alpha_delta)),
                "sigma_delta": float(-np.imag(alpha_delta)),
                "n_candidates": int(row["n_candidates"]),
                "n_filtered_candidates": int(row["n_filtered_candidates"]),
                "target_alpha_L": float(row["target_alpha_L"]),
                "status": row_status,
            })

    return records, delta_over_l, freqs, R_L, sutherland_s, transport


def _resolve_anchored_branch(args, profile, delta_over_l, candidate_rows, anchor_i, anchor_alpha):
    """Track an anchored spatial branch by re-solving near a predictor.

    The simpler ``anchored_max_sigma`` path precomputes each Reynolds station
    near a nominal phase speed, then connects nearest candidates.  That can
    miss the same root after it drifts away from the nominal target.  Here the
    anchor is still selected from the strongest second-mode-like candidate, but
    each neighboring station is re-solved around a predictor based on the
    previous accepted root and the nominal alpha increment.
    """
    tracked = [np.nan + 1j * np.nan for _ in candidate_rows]
    tracked[anchor_i] = complex(anchor_alpha)

    def choose(row, previous, previous_target):
        target_delta = row["target_alpha_delta"] - previous_target
        predictor = previous + target_delta
        try:
            alphas_delta, _omega_delta, _re_delta = _solve_spatial_candidates(
                profile,
                args,
                omega_L=float(row["omega_L"]),
                R=float(row["R_L"]),
                delta_over_l=delta_over_l,
                target_alpha_delta=predictor,
            )
            candidates = _phase_filtered_candidates(
                alphas_delta,
                omega_L=float(row["omega_L"]),
                delta_over_l=delta_over_l,
                phase_min=args.phase_min,
                phase_max=args.phase_max,
                alpha_min_L=args.alpha_min_l,
                alpha_max_L=args.alpha_max_l,
            )
        except Exception:
            candidates = np.asarray([], dtype=complex)
        if len(candidates) == 0:
            candidates = row["candidates"]
        if len(candidates) == 0:
            return np.nan + 1j * np.nan

        distances = np.abs(candidates - predictor)
        idx = int(np.argmin(distances))
        # Permit the real part to advance with frequency/Reynolds number, but
        # reject very large complex jumps that are usually root-family swaps.
        allowed = max(1.5, 0.20 * max(abs(predictor), abs(previous), 1.0))
        if distances[idx] > allowed:
            return np.nan + 1j * np.nan
        return complex(candidates[idx])

    prev = complex(anchor_alpha)
    prev_target = candidate_rows[anchor_i]["target_alpha_delta"]
    for i in range(anchor_i + 1, len(candidate_rows)):
        if not np.isfinite(prev):
            break
        alpha = choose(candidate_rows[i], prev, prev_target)
        tracked[i] = alpha
        if np.isfinite(alpha):
            prev = alpha
            prev_target = candidate_rows[i]["target_alpha_delta"]

    prev = complex(anchor_alpha)
    prev_target = candidate_rows[anchor_i]["target_alpha_delta"]
    for i in range(anchor_i - 1, -1, -1):
        if not np.isfinite(prev):
            break
        alpha = choose(candidate_rows[i], prev, prev_target)
        tracked[i] = alpha
        if np.isfinite(alpha):
            prev = alpha
            prev_target = candidate_rows[i]["target_alpha_delta"]

    return tracked


def _write_csv(path, records):
    fieldnames = [
        "freq_parameter",
        "R_L",
        "omega_L",
        "omega_delta",
        "Re_delta",
        "alpha_r_L",
        "alpha_i_L",
        "wavelength_L",
        "sigma_L",
        "alpha_r_delta",
        "alpha_i_delta",
        "sigma_delta",
        "n_candidates",
        "n_filtered_candidates",
        "target_alpha_L",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot(path, records, freqs, title):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
    for color, freq in zip(colors, freqs):
        rows = [row for row in records if np.isclose(row["freq_parameter"], freq)]
        rows.sort(key=lambda row: row["R_L"])
        R = np.array([row["R_L"] for row in rows])
        sigma = np.array([row["sigma_L"] for row in rows])
        ax.plot(R, sigma, "o-", color=color, label=rf"$\omega_L/R_L={freq:.3e}$")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel(r"$R_L=\sqrt{Re_x}=U_e L^*/\nu_e$")
    ax.set_ylabel(r"spatial growth $\sigma_L=-\mathrm{Im}(\alpha_L)$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-min", type=float, default=300.0)
    parser.add_argument("--r-max", type=float, default=3000.0)
    parser.add_argument("--r-points", type=int, default=19)
    parser.add_argument("--frequencies", default="0.00008,0.00010,0.00012,0.00014,0.00016")
    parser.add_argument("--frequency-mode", choices=["fixed_physical", "fixed_omega"], default="fixed_physical")
    parser.add_argument("--ma", type=float, default=6.0)
    parser.add_argument("--gas", default="nitrogen")
    parser.add_argument("--t-edge", type=float, default=288.0)
    parser.add_argument("--tw-over-te", type=float, default=5.55)
    parser.add_argument("--sutherland-s", type=float, default=None)
    parser.add_argument("--profile-family", choices=["ozgen", "power_law"], default="ozgen")
    parser.add_argument("--viscosity-exponent", type=float, default=0.74)
    parser.add_argument("--pr", type=float, default=0.72)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--wall-bc", choices=["isothermal", "adiabatic"], default="isothermal")
    parser.add_argument("--profile-points", type=int, default=3000)
    parser.add_argument("--eta-max", type=float, default=40.0)
    parser.add_argument("--N", type=int, default=56)
    parser.add_argument("--y-max-delta", type=float, default=10.0)
    parser.add_argument("--n-modes", type=int, default=10)
    parser.add_argument("--c-phase", type=float, default=0.86)
    parser.add_argument("--phase-min", type=float, default=0.75)
    parser.add_argument("--phase-max", type=float, default=1.15)
    parser.add_argument(
        "--mode-family",
        choices=["second_mode", "unconstrained"],
        default="second_mode",
        help=(
            "Branch-family contract. second_mode applies alpha_L>=0.6 by "
            "default so low-alpha long-wave branches cannot be reported as "
            "second-mode amplification. Use unconstrained for exploratory or "
            "first-mode work."
        ),
    )
    parser.add_argument("--alpha-min-l", type=float, default=None)
    parser.add_argument("--alpha-max-l", type=float, default=None)
    parser.add_argument(
        "--selection",
        choices=["tracked", "max_sigma", "anchored_max_sigma", "anchored_resolve"],
        default="max_sigma",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "chapters"
            / "ozgen_kircali_2008"
            / "diagnostics"
            / "mach6_growth_nfactor"
            / "R300_3000_N2_TwTe5p55_spatial"
        ),
    )
    return parser.parse_args()


def main():
    args = _apply_mode_family_defaults(parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, delta_over_l, freqs, R_L, sutherland_s, transport = compute_curves(args)
    csv_path = output_dir / "spatial_fixed_frequency_growth_curves.csv"
    png_path = output_dir / "spatial_fixed_frequency_growth_curves.png"
    metadata_path = output_dir / "spatial_fixed_frequency_growth_metadata.json"
    _write_csv(csv_path, records)
    _plot(
        png_path,
        records,
        freqs,
        f"Mach {args.ma:g} {args.gas}, Tw/Te={args.tw_over_te:g}: spatial growth",
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "status": "diagnostic_not_paper_certified",
            "quantity": "spatial growth sigma_L = -Im(alpha_L)",
            "frequency_mode": args.frequency_mode,
            "frequencies": [float(f) for f in freqs],
            "R_L": [float(r) for r in R_L],
            "ma": float(args.ma),
            "gas": args.gas,
            "T_edge_K": float(args.t_edge),
            "T_wall_over_T_edge": None if args.tw_over_te is None else float(args.tw_over_te),
            "T_wall_K": None if args.tw_over_te is None else float(args.tw_over_te * args.t_edge),
            "profile_family": transport["profile_family"],
            "viscosity_model": transport["viscosity_model"],
            "viscosity_exponent": transport["viscosity_exponent"],
            "sutherland_s_K": transport["sutherland_s_K"],
            "delta_star_over_L_star": float(delta_over_l),
            "wall_bc": args.wall_bc,
            "N": int(args.N),
            "solver": "lst.solver.solve_spatial companion QEP",
            "selection": args.selection,
            "mode_family": args.mode_family,
            "phase_speed_filter": [float(args.phase_min), float(args.phase_max)],
            "alpha_L_filter": [
                None if args.alpha_min_l is None else float(args.alpha_min_l),
                None if args.alpha_max_l is None else float(args.alpha_max_l),
            ],
            "note": "This uses the shared Mack-style compressible spatial solver, not the Ozgen temporal temperature-EVP.",
        }, handle, indent=2)
    print(f"csv={csv_path}")
    print(f"png={png_path}")
    print(f"metadata={metadata_path}")
    for freq in freqs:
        rows = [row for row in records if np.isclose(row["freq_parameter"], freq)]
        sig = np.array([row["sigma_L"] for row in rows], dtype=float)
        finite = np.isfinite(sig)
        positive = finite & (sig > 0.0)
        if np.any(finite):
            sigma_summary = (
                f"min_sigma_L={np.nanmin(sig):.6e}, "
                f"max_sigma_L={np.nanmax(sig):.6e}"
            )
        else:
            sigma_summary = "min_sigma_L=nan, max_sigma_L=nan"
        print(
            f"freq={freq:.6g}: finite={np.count_nonzero(finite)}, "
            f"positive={np.count_nonzero(positive)}, {sigma_summary}"
        )


if __name__ == "__main__":
    main()
