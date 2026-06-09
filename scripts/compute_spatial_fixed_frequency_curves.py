"""Compute spatial growth curves for fixed physical-frequency parameters."""

from __future__ import annotations

import argparse
import csv
import json
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

# Process-pool workers re-import this module on Windows.  Set this before the
# package imports so parallel sweeps do not print one banner per worker.
os.environ.setdefault("PYMACK_NO_BANNER", "1")

from pymack import CompressibleBlasiusProfile, make_ozgen_profile  # noqa: E402
from pymack.pymack_dense import (  # noqa: E402
    DenseBaseFlowConfig,
    DenseGasModel,
    DenseLSTConfig,
    prepare_dense_case,
    solve_mack_branch,
)
from pymack.scales import delta_star_over_lstar  # noqa: E402
from pymack.solver import solve_spatial, solve_spatial_full_spectrum  # noqa: E402


SECOND_MODE_ALPHA_MIN_L = None


def _configure_worker_threads():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYMACK_NO_BANNER", "1")


def _chunk_array(values, n_chunks):
    values = list(values)
    if n_chunks <= 1 or len(values) <= 1:
        return [values]
    n_chunks = min(int(n_chunks), len(values))
    return [list(chunk) for chunk in np.array_split(np.asarray(values, dtype=float), n_chunks) if len(chunk) > 0]


def _dense_records_for_frequency_chunk(payload):
    _configure_worker_threads()
    gas = DenseGasModel(**payload["gas"])
    base_cfg = DenseBaseFlowConfig(**payload["base_cfg"])
    lst_cfg = DenseLSTConfig(**payload["lst_cfg"])
    _base, y, D, base_grid = prepare_dense_case(gas, base_cfg, lst_cfg)

    records = []
    R_L = np.asarray(payload["R_L"], dtype=float)
    delta_over_l = float(payload["delta_over_l"])
    c_phase = float(payload["c_phase"])
    for freq in payload["freqs"]:
        rows = solve_mack_branch(
            float(freq),
            R_L,
            y,
            D,
            base_grid,
            gas,
            lst_cfg,
            convention="mack",
        )
        for row in rows:
            alpha_L = complex(row["alpha_real"], row["alpha_imag"])
            alpha_delta = alpha_L * delta_over_l
            omega_L = float(row["omega"])
            R = float(row["R"])
            records.append({
                "freq_parameter": float(freq),
                "R_L": R,
                "omega_L": omega_L,
                "omega_delta": float(omega_L * delta_over_l),
                "Re_delta": float(R * delta_over_l),
                "alpha_r_L": float(alpha_L.real),
                "alpha_i_L": float(alpha_L.imag),
                "phase_speed_L": float(row["phase_speed"]),
                "wavelength_L": float(
                    2.0 * np.pi / alpha_L.real
                    if np.isfinite(alpha_L.real) and alpha_L.real > 0.0
                    else np.nan
                ),
                "sigma_L": float(row["growth"]),
                "alpha_r_delta": float(alpha_delta.real),
                "alpha_i_delta": float(alpha_delta.imag),
                "sigma_delta": float(row["growth"] * delta_over_l),
                "n_candidates": int(row["n_candidates"]),
                "n_filtered_candidates": int(row["n_candidates"]),
                "target_alpha_L": float(omega_L / c_phase),
                "status": "ok" if row["selected"] else "not_tracked",
            })
    return records


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


def _phase_speed_from_alpha(omega_L, alpha_L):
    alpha_r = float(np.real(alpha_L))
    if not np.isfinite(alpha_r) or alpha_r <= 0.0:
        return np.nan
    return float(omega_L / alpha_r)


def _solve_spatial_candidates(profile, args, *, omega_L, R, delta_over_l,
                              target_alpha_delta):
    omega_delta = omega_L * delta_over_l
    Re_delta = float(R * delta_over_l)
    if args.solver_length_scale == "L_star":
        solver_omega = float(omega_L)
        solver_Re = float(R)
        solver_y_max = (
            float(args.y_max_lstar)
            if args.y_max_lstar is not None
            else float(args.y_max_delta * delta_over_l)
        )
        solver_target_alpha = target_alpha_delta / delta_over_l
        output_alpha_scale = float(delta_over_l)
    else:
        solver_omega = float(omega_delta)
        solver_Re = float(Re_delta)
        solver_y_max = float(args.y_max_delta)
        solver_target_alpha = target_alpha_delta
        output_alpha_scale = 1.0

    if args.candidate_source == "full_spectrum":
        alphas_solver, _modes, _y = solve_spatial_full_spectrum(
            profile,
            solver_omega,
            solver_Re,
            args.ma,
            args.pr,
            args.gamma,
            N=args.N,
            y_max=solver_y_max,
            wall_bc=args.wall_bc,
            length_scale=args.solver_length_scale,
            lambda_mu_ratio=args.lambda_mu_ratio,
            max_abs_alpha=args.full_spectrum_max_abs_alpha,
            max_abs_alpha_i=args.full_spectrum_max_abs_alpha_i,
            residual_tol=args.full_spectrum_residual_tol,
        )
    else:
        alphas_solver, _modes, _y = solve_spatial(
            profile,
            solver_omega,
            solver_Re,
            args.ma,
            args.pr,
            args.gamma,
            N=args.N,
            y_max=solver_y_max,
            wall_bc=args.wall_bc,
            target_alpha=solver_target_alpha,
            n_modes=args.n_modes,
            length_scale=args.solver_length_scale,
            lambda_mu_ratio=args.lambda_mu_ratio,
        )
    alphas_delta = np.asarray(alphas_solver, dtype=complex) * output_alpha_scale
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


def _track_pymack_continuation(candidate_rows, *, delta_over_l, c_phase):
    """Track the Mack/S fixed-F branch with pyMack-style alpha~R prediction."""
    tracked = [np.nan + 1j * np.nan for _ in candidate_rows]
    c_target = float(c_phase)
    row_best = []

    for i, row in enumerate(candidate_rows):
        candidates = row["candidates"]
        if len(candidates) == 0:
            continue
        alpha_L = candidates / delta_over_l
        phase = np.divide(
            float(row["omega_L"]),
            alpha_L.real,
            out=np.full_like(alpha_L.real, np.nan, dtype=float),
            where=alpha_L.real != 0.0,
        )
        sigma_L = -alpha_L.imag
        score = (
            sigma_L
            - 0.12 * np.abs(phase - c_target)
            - 0.35 * np.maximum(phase - 0.955, 0.0)
        )
        finite = np.isfinite(score) & np.isfinite(sigma_L)
        if not np.any(finite):
            continue
        safe_score = np.where(finite, score, -np.inf)
        j = int(np.nanargmax(safe_score))
        # pyMack's robust behavior comes from seeding on the strongest valid
        # lobe point, not from the first Reynolds station that has a candidate.
        row_best.append((float(sigma_L[j]), float(safe_score[j]), i, complex(candidates[j])))

    if not row_best:
        return tracked

    _seed_growth, _seed_score, seed_i, seed_alpha = max(
        row_best,
        key=lambda item: (item[0], item[1]),
    )
    tracked[seed_i] = seed_alpha

    def choose(row, previous, previous_R):
        candidates = row["candidates"]
        if len(candidates) == 0 or not np.isfinite(previous):
            return np.nan + 1j * np.nan
        alpha_L = candidates / delta_over_l
        phase = np.divide(
            float(row["omega_L"]),
            alpha_L.real,
            out=np.full_like(alpha_L.real, np.nan, dtype=float),
            where=alpha_L.real != 0.0,
        )
        sigma_L = -alpha_L.imag
        predictor = previous * (float(row["R_L"]) / float(previous_R))
        score = (
            np.abs(candidates - predictor) / max(abs(predictor), 1.0e-12)
            + 0.08 * np.abs(phase - c_target)
            - 0.005 * sigma_L
        )
        finite = np.isfinite(score)
        if not np.any(finite):
            return np.nan + 1j * np.nan
        safe_score = np.where(finite, score, np.inf)
        return complex(candidates[int(np.nanargmin(safe_score))])

    prev = seed_alpha
    prev_R = candidate_rows[seed_i]["R_L"]
    for i in range(seed_i + 1, len(candidate_rows)):
        alpha = choose(candidate_rows[i], prev, prev_R)
        tracked[i] = alpha
        if np.isfinite(alpha):
            prev = alpha
            prev_R = candidate_rows[i]["R_L"]

    prev = seed_alpha
    prev_R = candidate_rows[seed_i]["R_L"]
    for i in range(seed_i - 1, -1, -1):
        alpha = choose(candidate_rows[i], prev, prev_R)
        tracked[i] = alpha
        if np.isfinite(alpha):
            prev = alpha
            prev_R = candidate_rows[i]["R_L"]

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


def _make_sutherland_blasius_profile(args):
    if args.wall_bc != "isothermal":
        raise ValueError("sutherland_blasius diagnostic path currently requires an isothermal wall")
    if args.tw_over_te is None:
        raise ValueError("--tw-over-te is required for an isothermal sutherland_blasius profile")

    target_ratio = float(args.tw_over_te)
    target_wall = target_ratio * float(args.t_edge)
    kwargs = dict(
        Ma=args.ma,
        T_edge=args.t_edge,
        gamma=args.gamma,
        Pr=args.pr,
        wall_bc=args.wall_bc,
        viscosity_model="sutherland",
        sutherland_S=_default_sutherland_s(args),
        n_points=args.profile_points,
        eta_max=args.eta_max,
    )
    try:
        return CompressibleBlasiusProfile(T_wall=target_wall, **kwargs)
    except RuntimeError:
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
    elif args.profile_family == "sutherland_blasius":
        profile = _make_sutherland_blasius_profile(args)
        transport = {
            "profile_family": "sutherland_blasius",
            "viscosity_model": "sutherland",
            "viscosity_exponent": None,
            "sutherland_s_K": float(sutherland_s),
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

    pyMack-equivalent Mach-6 hot-wall cases show that the Mack/S branch for
    F=O(1e-4) has alpha_L=O(0.1), so an unconditional high-alpha floor rejects
    the validated branch.  The family contract is therefore phase-speed and
    continuation based by default; users can still provide --alpha-min-l for
    deliberately high-alpha diagnostics.
    """
    if (
        args.mode_family == "second_mode"
        and args.alpha_min_l is None
        and SECOND_MODE_ALPHA_MIN_L is not None
    ):
        args.alpha_min_l = SECOND_MODE_ALPHA_MIN_L
    return args


def compute_curves(args):
    profile, sutherland_s, transport = _make_profile(args)
    delta_over_l = float(delta_star_over_lstar(profile))
    R_L = np.linspace(args.r_min, args.r_max, args.r_points)
    freqs = np.array([float(item) for item in args.frequencies.split(",") if item.strip()])

    if args.backend == "pymack_dense":
        return _compute_curves_pymack_dense(
            args,
            delta_over_l,
            freqs,
            R_L,
            sutherland_s,
            transport,
        )

    if args.selection in {"anchored_max_sigma", "anchored_resolve", "pymack_continuation"}:
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
                    lambda_mu_ratio=args.lambda_mu_ratio,
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
            phase_speed_L = _phase_speed_from_alpha(omega_L, alpha_L)
            records.append({
                "freq_parameter": float(freq),
                "R_L": float(R),
                "omega_L": float(omega_L),
                "omega_delta": float(omega_delta),
                "Re_delta": float(Re_delta),
                "alpha_r_L": float(np.real(alpha_L)),
                "alpha_i_L": float(np.imag(alpha_L)),
                "phase_speed_L": phase_speed_L,
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
                _track_pymack_continuation(
                    candidate_rows,
                    delta_over_l=delta_over_l,
                    c_phase=args.c_phase,
                )
                if args.selection == "pymack_continuation"
                else
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
            phase_speed_L = _phase_speed_from_alpha(row["omega_L"], alpha_L)
            records.append({
                "freq_parameter": float(freq),
                "R_L": float(row["R_L"]),
                "omega_L": float(row["omega_L"]),
                "omega_delta": float(row["omega_delta"]),
                "Re_delta": float(row["Re_delta"]),
                "alpha_r_L": float(np.real(alpha_L)),
                "alpha_i_L": float(np.imag(alpha_L)),
                "phase_speed_L": phase_speed_L,
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


def _compute_curves_pymack_dense(args, delta_over_l, freqs, R_L, sutherland_s, transport):
    if args.frequency_mode != "fixed_physical":
        raise ValueError("pymack_dense backend currently requires --frequency-mode fixed_physical")
    if args.wall_bc != "isothermal" and not args.dense_adiabatic_wall:
        raise ValueError("pymack_dense non-adiabatic path currently supports isothermal-wall runs")
    if args.profile_family not in {"sutherland_blasius", "power_law"}:
        raise ValueError("pymack_dense backend supports sutherland_blasius or power_law profiles")

    y_max_lstar = (
        float(args.y_max_lstar)
        if args.y_max_lstar is not None
        else float(args.dense_y_max_lstar)
    )
    payload_base = {
        "gas": {
            "gamma": float(args.gamma),
            "prandtl": float(args.pr),
            "viscosity_law": "power" if args.profile_family == "power_law" else "sutherland",
            "mu_power": float(args.viscosity_exponent),
            "sutherland_S_K": float(sutherland_s),
            "T_edge_K": float(args.t_edge),
        },
        "base_cfg": {
            "mach_edge": float(args.ma),
            "Tw_Te": float(args.tw_over_te),
            "eta_max": float(args.eta_max),
            "eta_nodes": int(args.dense_eta_nodes),
            "bvp_tol": float(args.dense_bvp_tol),
            "adiabatic": bool(args.dense_adiabatic_wall),
        },
        "lst_cfg": {
            "ny": int(args.N),
            "y_max": y_max_lstar,
            "c_min": float(args.phase_min),
            "c_max": float(args.phase_max),
            "c_target": float(args.c_phase),
            "max_abs_alpha": float(args.dense_max_abs_alpha),
            "max_abs_ai": float(args.dense_max_abs_ai),
            "max_ai_over_ar": float(args.dense_max_ai_over_ar),
        },
        "R_L": [float(r) for r in R_L],
        "delta_over_l": float(delta_over_l),
        "c_phase": float(args.c_phase),
    }
    records = []
    workers = max(1, int(args.workers))
    chunks = _chunk_array(freqs, workers)
    if workers == 1 or len(chunks) == 1:
        for chunk in chunks:
            payload = dict(payload_base)
            payload["freqs"] = [float(freq) for freq in chunk]
            records.extend(_dense_records_for_frequency_chunk(payload))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_configure_worker_threads,
        ) as executor:
            futures = []
            for chunk in chunks:
                payload = dict(payload_base)
                payload["freqs"] = [float(freq) for freq in chunk]
                futures.append(executor.submit(_dense_records_for_frequency_chunk, payload))
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda row: (float(row["freq_parameter"]), float(row["R_L"])))

    transport = dict(transport)
    transport["backend"] = "pymack_dense"
    transport["dense_y_max_lstar"] = y_max_lstar
    transport["workers"] = workers
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
        "phase_speed_L",
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


def _plot_phase_speed(path, records, freqs, title, *, ma=None):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
    for color, freq in zip(colors, freqs):
        rows = [row for row in records if np.isclose(row["freq_parameter"], freq)]
        rows.sort(key=lambda row: row["R_L"])
        R = np.array([row["R_L"] for row in rows], dtype=float)
        c = np.array([row.get("phase_speed_L", np.nan) for row in rows], dtype=float)
        ax.plot(R, c, "o-", color=color, label=rf"$\omega_L/R_L={freq:.3e}$")

    if ma is not None and np.isfinite(ma) and ma > 0.0:
        slow = 1.0 - 1.0 / float(ma)
        fast = 1.0 + 1.0 / float(ma)
        ax.axhline(slow, color="0.35", lw=0.9, ls=":", label=r"$1-1/M_e$")
        ax.axhline(fast, color="0.35", lw=0.9, ls="--", label=r"$1+1/M_e$")
    ax.set_xlabel(r"$R_L=\sqrt{Re_x}=U_e L^*/\nu_e$")
    ax.set_ylabel(r"phase speed $c_L=\omega_L/\alpha_{r,L}$")
    ax.set_title(f"{title}: phase speed")
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
    parser.add_argument("--N", type=int, default=56)
    parser.add_argument("--y-max-delta", type=float, default=10.0)
    parser.add_argument(
        "--y-max-lstar",
        type=float,
        default=None,
        help=(
            "Domain height in L* units when --solver-length-scale L_star is "
            "used. If omitted, y_max_lstar = y_max_delta * delta*/L*."
        ),
    )
    parser.add_argument("--n-modes", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of process workers for independent dense-backend "
            "frequency branches. The ms_lst backend remains serial here."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["ms_lst", "pymack_dense"],
        default="ms_lst",
        help=(
            "ms_lst uses the general shared spatial solver. pymack_dense uses "
            "the independent pyMack-style dense QEP backend for 2-D Mack/S "
            "branch validation and production once accepted."
        ),
    )
    parser.add_argument(
        "--solver-length-scale",
        choices=["delta_star", "L_star"],
        default="delta_star",
        help="Length scale used inside the spatial solver.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=["shift_invert", "full_spectrum"],
        default="shift_invert",
        help=(
            "Use fast shift-invert candidates near the target alpha, or solve "
            "the full companion spectrum for more robust branch discovery."
        ),
    )
    parser.add_argument("--full-spectrum-max-abs-alpha", type=float, default=8.0)
    parser.add_argument("--full-spectrum-max-abs-alpha-i", type=float, default=0.4)
    parser.add_argument("--full-spectrum-residual-tol", type=float, default=None)
    parser.add_argument("--dense-eta-nodes", type=int, default=80)
    parser.add_argument("--dense-bvp-tol", type=float, default=1.0e-4)
    parser.add_argument("--dense-y-max-lstar", type=float, default=30.0)
    parser.add_argument("--dense-max-abs-alpha", type=float, default=8.0)
    parser.add_argument("--dense-max-abs-ai", type=float, default=0.4)
    parser.add_argument("--dense-max-ai-over-ar", type=float, default=1.0)
    parser.add_argument(
        "--dense-adiabatic-wall",
        action="store_true",
        help="Use the dense backend adiabatic-wall base-flow condition.",
    )
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
    parser.add_argument("--c-phase", type=float, default=0.86)
    parser.add_argument("--phase-min", type=float, default=0.75)
    parser.add_argument("--phase-max", type=float, default=1.15)
    parser.add_argument(
        "--mode-family",
        choices=["second_mode", "unconstrained"],
        default="second_mode",
        help=(
            "Branch-family contract. second_mode uses phase-speed and "
            "continuation by default; provide --alpha-min-l only for "
            "deliberately high-alpha diagnostics."
        ),
    )
    parser.add_argument("--alpha-min-l", type=float, default=None)
    parser.add_argument("--alpha-max-l", type=float, default=None)
    parser.add_argument(
        "--selection",
        choices=[
            "tracked",
            "max_sigma",
            "anchored_max_sigma",
            "anchored_resolve",
            "pymack_continuation",
        ],
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
    phase_png_path = output_dir / "spatial_fixed_frequency_phase_speed.png"
    metadata_path = output_dir / "spatial_fixed_frequency_growth_metadata.json"
    _write_csv(csv_path, records)
    _plot(
        png_path,
        records,
        freqs,
        f"Mach {args.ma:g} {args.gas}, Tw/Te={args.tw_over_te:g}: spatial growth",
    )
    _plot_phase_speed(
        phase_png_path,
        records,
        freqs,
        f"Mach {args.ma:g} {args.gas}, Tw/Te={args.tw_over_te:g}",
        ma=args.ma,
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
            "backend": args.backend,
            "workers": int(args.workers),
            "solver_length_scale": args.solver_length_scale,
            "y_max_delta": float(args.y_max_delta),
            "y_max_lstar": (
                None
                if args.y_max_lstar is None
                else float(args.y_max_lstar)
            ),
            "solver": (
                "pymack.pymack_dense dense full-spectrum QEP"
                if args.backend == "pymack_dense"
                else "pymack.solver.solve_spatial companion QEP"
            ),
            "dense_backend_config": {
                "eta_nodes": int(args.dense_eta_nodes),
                "bvp_tol": float(args.dense_bvp_tol),
                "y_max_lstar": (
                    float(args.y_max_lstar)
                    if args.y_max_lstar is not None
                    else float(args.dense_y_max_lstar)
                ),
                "max_abs_alpha": float(args.dense_max_abs_alpha),
                "max_abs_ai": float(args.dense_max_abs_ai),
                "max_ai_over_ar": float(args.dense_max_ai_over_ar),
                "adiabatic_wall": bool(args.dense_adiabatic_wall),
            },
            "candidate_source": args.candidate_source,
            "full_spectrum_filter": {
                "max_abs_alpha_delta": float(args.full_spectrum_max_abs_alpha),
                "max_abs_alpha_i_delta": float(args.full_spectrum_max_abs_alpha_i),
                "residual_tol": args.full_spectrum_residual_tol,
            },
            "lambda_mu_ratio": float(args.lambda_mu_ratio),
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
    print(f"phase_png={phase_png_path}")
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
