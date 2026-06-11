"""Qualitative overlay of pyMack maximum temporal growth vs Mack (1984) Fig. 10.3.

Layer-6 demonstration figure (see ``docs/VALIDATION_STRATEGY.md``): pyMack's
maximum-over-alpha temporal growth rate versus Reynolds number for the
M = 1.3, psi = 45 deg oblique first mode is overlaid on the digitized paper
curve ``reference_data/digitized/mack_ch10_fig10_3_M13_paper_psi45.csv``.

The computation rides on the Layer-3 table-validated exact-shooting machinery
(full 8x8 Appendix-A first-order system, sigma-min continuation), anchored at
the Mack Table 10.1 point (R = 500, alpha_L = 0.075) and continued in both
Reynolds number and wavenumber. The two Mack Table 10.1 rows for this family
(R = 500 and R = 1500) are drawn as distinct anchor markers: those points are
*quantitative* (Layer 3, 0.07-0.91 % exact-shooting reproduction); the curve
overlay itself is a *qualitative* demonstration, not a validation gate.

Conditions note
---------------
The default mean-flow temperature schedule is ``table_11_1`` (constant total
temperature ~305 K, i.e. T_1* ~= 228 K at M = 1.3). This is the condition set
under which pyMack's Layer-3 Table 10.1 reproduction holds (see
``pymack.mack_table_10_1.DEFAULT_TABLE_10_1_CONDITION`` and the docstring of
``pymack.mack_conditions.mack_table_10_1_edge_temperature``). The
figure-caption wind-tunnel schedule (T_1* = 311 K, ``--condition
wind_tunnel``) shifts the computed growth rates roughly 10 % below the table
values at R = 500, so the table cross-check gate is only meaningful with the
default schedule.

Axis convention (same as the digitized paper data)
--------------------------------------------------
    x = R x 1e-2          (so x = 15 means R = 1500)
    y = omega_i x 1e3     (temporal growth on Mack's L* scale)

Usage
-----
    python scripts/make_mack_fig10_3_overlay.py --quality production
    python scripts/make_mack_fig10_3_overlay.py --quality smoke \
        --output-png /tmp/overlay.png --output-json /tmp/overlay.json

Outputs: PNG figure (default ``docs/figures/mack_fig10_3_overlay.png``),
JSON metadata, and a CSV of computed (R, alpha_opt, omega_i_max) rows next to
the PNG.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymack import (  # noqa: E402
    find_temporal_mode_anchor_3d_shooting,
    load_reference_csv,
    make_mack_profile,
    select_mack_table_10_1_cases,
    temporal_growth_scan_3d_shooting_from_anchor,
)
from pymack.mack_table_10_1 import TABLE_10_1_FAMILY_SETTINGS  # noqa: E402


# --- Fixed physical setup (Mack 1984, Fig. 10.3 / Table 10.1, M=1.3 family) ---
MA = 1.3
PSI_DEG = 45.0
WALL_BC = "isothermal"  # disturbance thermal wall condition (Layer-3 setting)
LENGTH_SCALE = "L_star"
Y_MAX = 26.0
N_STEPS = 1500
ANCHOR_R = 500.0
ANCHOR_ALPHA = 0.075
DIGITIZED_REFERENCE = "digitized/mack_ch10_fig10_3_M13_paper_psi45.csv"

# Known near-optimal wavenumbers from Mack Table 10.1 (M=1.3, psi=45):
# alpha_opt drifts from ~0.075 at R=500 to ~0.060 at R=1500.
ALPHA_CENTER_KNOTS = ((500.0, 0.075), (1500.0, 0.060))

# Physicality screens applied on top of the library results.
# The first-mode branch lives at c ~ 0.45 + 0.01j; anything outside the box
# below, or with a wall-matrix sigma_min that is not tiny, is not a converged
# shooting root (e.g. a sigma-min penalty-boundary artifact at c ~ 1.2 + 0.3j
# with sigma_min ~ 0.1, or a diverged Muller polish with |c_i| ~ 1e13).
C_REAL_BOUNDS = (0.0, 1.2)
C_IMAG_ABS_MAX = 0.3
ROOT_SIGMA_TOL = 1e-3  # converged roots sit at sigma_min ~ 1e-9..1e-6
ANCHOR_OMEGA_RTOL = 0.2  # anchor must land within 20 % of the table value

QUALITY_PRESETS = {
    "smoke": {
        "r_list": [500.0, 1500.0],
        "n_alpha": 4,
        "alpha_span": None,  # fixed grids around the table alphas (see below)
        "anchor_seed_count": 1,
    },
    "production": {
        "r_list": [200.0, 300.0, 500.0, 700.0, 1000.0, 1300.0, 1600.0, 2000.0],
        "n_alpha": 8,
        "alpha_span": (0.70, 1.30),
        "anchor_seed_count": None,  # all curated seeds
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PRESETS),
        default="production",
        help=(
            "smoke: R in {500, 1500} with 4 alpha points each, cross-checked "
            "against Mack Table 10.1; production: full 8-point Reynolds sweep."
        ),
    )
    parser.add_argument(
        "--output-png",
        default=str(REPO_ROOT / "docs" / "figures" / "mack_fig10_3_overlay.png"),
        help="output figure path (default: docs/figures/mack_fig10_3_overlay.png)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="output metadata path (default: PNG path with .json extension)",
    )
    parser.add_argument(
        "--condition",
        choices=("table_11_1", "wind_tunnel", "table_10_1"),
        default="table_11_1",
        help=(
            "Mack mean-flow temperature schedule. Default 'table_11_1' is the "
            "condition under which the Layer-3 Table 10.1 reproduction (and "
            "therefore the smoke gate) holds; 'wind_tunnel' is the literal "
            "figure-caption T_1*=311 K schedule, which sits ~10%% below the "
            "table values and fails the table cross-check."
        ),
    )
    parser.add_argument(
        "--table-rtol",
        type=float,
        default=0.02,
        help=(
            "relative tolerance for the Mack Table 10.1 cross-check at "
            "(R=500, alpha=0.075) and (R=1500, alpha=0.060); compared against "
            "the 8th-order table column (default 0.02)"
        ),
    )
    return parser.parse_args(argv)


def alpha_center(R):
    """Log-linear interpolation of the drifting optimum wavenumber in R."""
    (r0, a0), (r1, a1) = ALPHA_CENTER_KNOTS
    value = a0 + (a1 - a0) * math.log(R / r0) / math.log(r1 / r0)
    return float(np.clip(value, 0.02, 0.15))


def table_alpha_map():
    """Return {R: MackTable101Case} for the M=1.3, psi=45 family."""
    cases = select_mack_table_10_1_cases(Ma=MA, psi_deg=PSI_DEG)
    return {float(case.Re_L): case for case in cases}


def build_alpha_grid(R, quality, table_cases):
    """Build the alpha scan grid for one Reynolds number."""
    preset = QUALITY_PRESETS[quality]
    table_case = table_cases.get(float(R))

    if quality == "smoke":
        if table_case is None:
            raise ValueError(f"smoke mode expects table-anchored R, got R={R}")
        ta = table_case.alpha_L
        return np.array([ta - 0.010, ta - 0.005, ta, ta + 0.005], dtype=float)

    lo, hi = preset["alpha_span"]
    center = alpha_center(R)
    grid = list(np.linspace(lo * center, hi * center, preset["n_alpha"]))
    if table_case is not None:
        # Force the exact Table 10.1 wavenumber onto the grid so the
        # quantitative cross-check also runs at production quality.
        grid.append(float(table_case.alpha_L))
    grid = sorted(grid)
    deduped = [grid[0]]
    for value in grid[1:]:
        if value - deduped[-1] > 1e-9:
            deduped.append(value)
    return np.array(deduped, dtype=float)


def find_anchor_root(profile, quality, table_cases):
    """Locate the first-mode anchor root at (R=500, alpha=0.075, psi=45)."""
    family = TABLE_10_1_FAMILY_SETTINGS[MA]
    seeds = [family["fallback_initial_c"]]
    count = QUALITY_PRESETS[quality]["anchor_seed_count"]
    extra = list(family["seed_list"])
    if count is None:
        seeds += extra
    else:
        seeds += extra[: max(0, count - 1)]

    anchor = find_temporal_mode_anchor_3d_shooting(
        profile,
        ANCHOR_R,
        ANCHOR_ALPHA,
        Ma=MA,
        seed_list=seeds,
        psi_deg=PSI_DEG,
        y_max=Y_MAX,
        n_steps=N_STEPS,
        wall_bc=WALL_BC,
        length_scale=LENGTH_SCALE,
        method="qr",
    )

    # Do NOT trust anchor["selected_c"]: the library helper picks the
    # largest-omega_i candidate inside the (open) search box without
    # requiring that it actually is a converged shooting root. With the
    # full production seed list, one seed can wander to the corner of the
    # sigma-min penalty box (c ~ 1.19 + 0.295j, sigma_min ~ 0.16, i.e. not
    # a root at all) and win that selection; continuing the sweep from a
    # non-root then diverges (omega_i ~ 1e12) and poisons every R. Select
    # here instead: genuine roots only (tiny sigma_min, inside the physical
    # box), closest to the validated Mack Table 10.1 growth at this point.
    table_case = table_cases.get(float(ANCHOR_R))
    if table_case is None:
        raise RuntimeError(f"no Mack Table 10.1 case at the anchor R={ANCHOR_R:g}")
    omega_expected = float(table_case.omega_i_8th)

    best_c, best_err = None, None
    for cand in anchor["candidates"]:
        c = cand["c_final"]
        if not (np.isfinite(c.real) and np.isfinite(c.imag)):
            c = cand["c_sigma_min"]
        if not (np.isfinite(c.real) and np.isfinite(c.imag)):
            continue
        if not (C_REAL_BOUNDS[0] < c.real < C_REAL_BOUNDS[1]):
            continue
        if abs(c.imag) >= C_IMAG_ABS_MAX:
            continue
        sigma = cand["sigma_min"]
        if not (np.isfinite(sigma) and sigma <= ROOT_SIGMA_TOL):
            continue
        err = abs(ANCHOR_ALPHA * c.imag - omega_expected)
        if best_err is None or err < best_err:
            best_c, best_err = complex(c), err

    if best_c is None or best_err > ANCHOR_OMEGA_RTOL * abs(omega_expected):
        raise RuntimeError(
            "anchor root search failed at (R=500, alpha=0.075): no converged "
            f"physical shooting root within {100 * ANCHOR_OMEGA_RTOL:.0f} % of "
            f"the Mack Table 10.1 value {omega_expected:.3e} "
            f"(candidates: {[str(c['c_final']) for c in anchor['candidates']]})"
        )
    return best_c, [complex(seed) for seed in seeds]


def refine_optimum(alphas, omega_i):
    """Discrete argmax plus parabolic refinement through the peak triple."""
    finite = np.isfinite(omega_i)
    if not np.any(finite):
        return None
    idx_pool = np.flatnonzero(finite)
    idx = int(idx_pool[np.argmax(omega_i[idx_pool])])
    result = {
        "index": idx,
        "alpha_opt_discrete": float(alphas[idx]),
        "omega_i_max_discrete": float(omega_i[idx]),
        "bracket_edge": bool(idx == 0 or idx == len(alphas) - 1),
        "alpha_opt_refined": None,
        "omega_i_max_refined": None,
    }
    if 0 < idx < len(alphas) - 1 and finite[idx - 1] and finite[idx + 1]:
        x = alphas[idx - 1 : idx + 2]
        y = omega_i[idx - 1 : idx + 2]
        coeffs = np.polyfit(x, y, 2)
        if coeffs[0] < 0.0:
            alpha_star = float(-coeffs[1] / (2.0 * coeffs[0]))
            if x[0] <= alpha_star <= x[2]:
                result["alpha_opt_refined"] = alpha_star
                result["omega_i_max_refined"] = float(np.polyval(coeffs, alpha_star))
    result["alpha_opt"] = (
        result["alpha_opt_refined"]
        if result["alpha_opt_refined"] is not None
        else result["alpha_opt_discrete"]
    )
    result["omega_i_max"] = (
        result["omega_i_max_refined"]
        if result["omega_i_max_refined"] is not None
        else result["omega_i_max_discrete"]
    )
    return result


def scan_one_reynolds(profile, R, alphas, anchor_alpha, initial_c):
    """Scan omega_i(alpha) at one Reynolds number from an interior anchor."""
    anchor_index = int(np.argmin(np.abs(alphas - anchor_alpha)))
    t0 = time.time()
    _, omega_i, c_vals, sigma_min, _ = temporal_growth_scan_3d_shooting_from_anchor(
        profile,
        float(R),
        MA,
        alphas,
        anchor_index=anchor_index,
        initial_c=initial_c,
        psi_deg=PSI_DEG,
        y_max=Y_MAX,
        n_steps=N_STEPS,
        wall_bc=WALL_BC,
        length_scale=LENGTH_SCALE,
        method="qr",
    )
    elapsed = time.time() - t0

    # Physicality screen: only converged physical roots may become the
    # optimum (and therefore the continuation seed for the next R). A point
    # where the sigma-min search or the Muller determinant polish ran away
    # (sigma_min not tiny, or c outside the first-mode box) is masked out so
    # one bad point cannot poison the whole sweep.
    omega_arr = np.asarray(omega_i, dtype=float)
    sigma_arr = np.asarray(sigma_min, dtype=float)
    c_arr = np.asarray(c_vals, dtype=complex)
    physical = (
        np.isfinite(omega_arr)
        & np.isfinite(sigma_arr)
        & np.isfinite(c_arr.real)
        & np.isfinite(c_arr.imag)
        & (c_arr.real > C_REAL_BOUNDS[0])
        & (c_arr.real < C_REAL_BOUNDS[1])
        & (np.abs(c_arr.imag) < C_IMAG_ABS_MAX)
        & (sigma_arr <= ROOT_SIGMA_TOL)
    )

    optimum = refine_optimum(alphas, np.where(physical, omega_arr, np.nan))
    if optimum is None:
        raise RuntimeError(
            f"alpha scan at R={R:g} produced no physical converged roots"
        )

    return {
        "R": float(R),
        "alpha_grid": [float(a) for a in alphas],
        "omega_i": [float(w) if np.isfinite(w) else None for w in omega_i],
        "sigma_min": [float(s) if np.isfinite(s) else None for s in sigma_min],
        "c": [[float(c.real), float(c.imag)] for c in c_vals],
        "physical": [bool(p) for p in physical],
        "n_unphysical": int(np.sum(~physical)),
        "anchor_index": anchor_index,
        "elapsed_s": round(elapsed, 1),
        **optimum,
    }


def run_sweep(profile, quality, table_cases):
    """Anchor once, then continue the branch over the Reynolds list."""
    r_list = QUALITY_PRESETS[quality]["r_list"]
    print(f"[anchor] searching root at R={ANCHOR_R:g}, alpha={ANCHOR_ALPHA:g} ...", flush=True)
    t0 = time.time()
    anchor_c, anchor_seeds = find_anchor_root(profile, quality, table_cases)
    print(
        f"[anchor] selected c = {anchor_c.real:.6f}{anchor_c.imag:+.6f}j "
        f"(omega_i = {ANCHOR_ALPHA * anchor_c.imag:.4e}) in {time.time() - t0:.0f} s",
        flush=True,
    )

    results = {}

    def sweep(r_values, start_alpha, start_c):
        prev_alpha, prev_c = start_alpha, start_c
        for R in r_values:
            alphas = build_alpha_grid(R, quality, table_cases)
            result = scan_one_reynolds(profile, R, alphas, prev_alpha, prev_c)
            results[float(R)] = result
            idx = result["index"]
            prev_alpha = result["alpha_opt_discrete"]
            prev_c = complex(result["c"][idx][0], result["c"][idx][1])
            print(
                f"[scan] R={R:6g}: alpha_opt={result['alpha_opt']:.4f}, "
                f"omega_i_max={result['omega_i_max']:.4e} "
                f"({len(alphas)} alphas, {result['elapsed_s']:.0f} s"
                + (", bracket edge!" if result["bracket_edge"] else "")
                + (
                    f", {result['n_unphysical']} unphysical pts dropped"
                    if result["n_unphysical"]
                    else ""
                )
                + ")",
                flush=True,
            )

    upward = sorted(r for r in r_list if r >= ANCHOR_R)
    downward = sorted((r for r in r_list if r < ANCHOR_R), reverse=True)
    sweep(upward, ANCHOR_ALPHA, anchor_c)

    if downward:
        if float(ANCHOR_R) in results:
            seed_result = results[float(ANCHOR_R)]
            idx = seed_result["index"]
            start_alpha = seed_result["alpha_opt_discrete"]
            start_c = complex(seed_result["c"][idx][0], seed_result["c"][idx][1])
        else:
            start_alpha, start_c = ANCHOR_ALPHA, anchor_c
        sweep(downward, start_alpha, start_c)

    ordered = [results[float(R)] for R in sorted(r_list)]
    anchor_info = {
        "R": ANCHOR_R,
        "alpha": ANCHOR_ALPHA,
        "seeds": [[s.real, s.imag] for s in anchor_seeds],
        "selected_c": [anchor_c.real, anchor_c.imag],
        "omega_i": ANCHOR_ALPHA * anchor_c.imag,
    }
    return ordered, anchor_info


def cross_check_against_table(results, table_cases, rtol):
    """Compare scan values at the exact Table 10.1 wavenumbers."""
    checks = []
    for result in results:
        case = table_cases.get(result["R"])
        if case is None:
            continue
        alphas = np.asarray(result["alpha_grid"], dtype=float)
        hits = np.flatnonzero(np.abs(alphas - case.alpha_L) < 1e-9)
        if len(hits) == 0:
            continue
        omega = result["omega_i"][int(hits[0])]
        if omega is None:
            checks.append({
                "R": result["R"],
                "alpha_L": case.alpha_L,
                "omega_i_scan": None,
                "passed": False,
                "note": "scan did not converge at the table wavenumber",
            })
            continue
        rel_8th = (omega - case.omega_i_8th) / case.omega_i_8th
        rel_6th = (omega - case.omega_i_6th) / case.omega_i_6th
        checks.append({
            "R": result["R"],
            "alpha_L": case.alpha_L,
            "omega_i_scan": omega,
            "omega_i_table_8th": case.omega_i_8th,
            "omega_i_table_6th": case.omega_i_6th,
            "rel_err_vs_8th": rel_8th,
            "rel_err_vs_6th": rel_6th,
            "rtol": rtol,
            "passed": bool(abs(rel_8th) <= rtol),
        })
    return checks


def load_digitized_curve():
    """Load the digitized paper curve (x = R x 1e-2, y = omega_i x 1e3)."""
    rows = load_reference_csv(DIGITIZED_REFERENCE)
    x = np.array([float(row["x"]) for row in rows], dtype=float)
    y = np.array([float(row["y"]) for row in rows], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def write_csv(path, results):
    fields = [
        "R", "x_paper", "alpha_opt", "omega_i_max", "y_paper",
        "alpha_opt_discrete", "omega_i_max_discrete", "refined",
        "bracket_edge", "c_r_at_opt", "c_i_at_opt", "sigma_min_at_opt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            idx = result["index"]
            writer.writerow({
                "R": result["R"],
                "x_paper": result["R"] * 1e-2,
                "alpha_opt": result["alpha_opt"],
                "omega_i_max": result["omega_i_max"],
                "y_paper": result["omega_i_max"] * 1e3,
                "alpha_opt_discrete": result["alpha_opt_discrete"],
                "omega_i_max_discrete": result["omega_i_max_discrete"],
                "refined": result["alpha_opt_refined"] is not None,
                "bracket_edge": result["bracket_edge"],
                "c_r_at_opt": result["c"][idx][0],
                "c_i_at_opt": result["c"][idx][1],
                "sigma_min_at_opt": result["sigma_min"][idx],
            })


def make_figure(path, results, table_cases, condition, t_edge, quality):
    paper_x, paper_y = load_digitized_curve()
    comp_x = np.array([r["R"] * 1e-2 for r in results], dtype=float)
    comp_y = np.array([r["omega_i_max"] * 1e3 for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.plot(
        paper_x, paper_y, "o--", color="0.35", lw=1.8, ms=5,
        label=r"Mack (1984) Fig. 10.3, $\psi=45^\circ$ (digitized)",
    )
    ax.plot(
        comp_x, comp_y, "o-", color="#3210a8", lw=2.4, ms=6.5,
        label=r"pyMack $\max_\alpha\,\omega_i$ (exact shooting, full 8$\times$8)",
    )

    anchor_x = [case.Re_L * 1e-2 for case in table_cases.values()]
    anchor_y = [case.omega_i_6th * 1e3 for case in table_cases.values()]
    ax.scatter(
        anchor_x, anchor_y, marker="*", s=260, color="#ff7433",
        edgecolor="black", linewidth=0.8, zorder=5,
        label="Mack Table 10.1 (6th order) — Layer-3 validated anchors",
    )

    ax.set_xlabel(r"$R \times 10^{-2}$", fontsize=14)
    ax.set_ylabel(r"$\omega_i \times 10^{3}$  (Mack $L^*$ scale)", fontsize=14)
    ax.set_title(
        r"Mack (1984) Fig. 10.3 overlay — $M=1.3$, $\psi=45^\circ$ oblique first mode",
        fontsize=16, pad=12,
    )
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.36, linestyle="--")
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    ax.text(
        0.025, 0.965,
        "Qualitative comparison (Layer 6) — demonstration, not a validation gate\n"
        f"condition: {condition} ($T_1^*$ = {t_edge:.0f} K), quality: {quality}",
        transform=ax.transAxes, ha="left", va="top", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                  edgecolor="0.6"),
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()

    output_png = Path(args.output_png)
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else output_png.with_suffix(".json")
    )
    output_csv = output_png.with_suffix(".csv")

    profile = make_mack_profile(MA, condition=args.condition)
    t_edge = float(profile.T_edge)
    print(
        f"[setup] M={MA}, psi={PSI_DEG:g} deg, condition={args.condition} "
        f"(T_1*={t_edge:.1f} K), wall_bc={WALL_BC}, y_max={Y_MAX:g}, "
        f"n_steps={N_STEPS}, quality={args.quality}",
        flush=True,
    )

    table_cases = table_alpha_map()
    results, anchor_info = run_sweep(profile, args.quality, table_cases)
    checks = cross_check_against_table(results, table_cases, args.table_rtol)

    all_passed = all(check["passed"] for check in checks) if checks else False
    print()
    print("Mack Table 10.1 cross-check (quantitative, 8th-order column):")
    for check in checks:
        if check.get("omega_i_scan") is None:
            print(f"  R={check['R']:6g}: FAILED ({check.get('note', 'no value')})")
            continue
        print(
            f"  R={check['R']:6g}, alpha={check['alpha_L']:.3f}: "
            f"scan={check['omega_i_scan']:.4e}, table_8th={check['omega_i_table_8th']:.4e} "
            f"({100 * check['rel_err_vs_8th']:+.2f} %), "
            f"table_6th={check['omega_i_table_6th']:.4e} "
            f"({100 * check['rel_err_vs_6th']:+.2f} %) -> "
            + ("PASS" if check["passed"] else "FAIL")
        )

    write_csv(output_csv, results)
    make_figure(output_png, results, table_cases, args.condition, t_edge, args.quality)

    metadata = {
        "script": "scripts/make_mack_fig10_3_overlay.py",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_s": round(time.time() - t_start, 1),
        "comparison_type": "qualitative_overlay_layer6",
        "validation_strategy": "docs/VALIDATION_STRATEGY.md (Layer 6 demonstration)",
        "quality": args.quality,
        "parameters": {
            "Ma": MA,
            "psi_deg": PSI_DEG,
            "condition": args.condition,
            "T_edge_K": t_edge,
            "mean_flow_wall": "adiabatic (insulated)",
            "disturbance_wall_bc": WALL_BC,
            "length_scale": LENGTH_SCALE,
            "y_max": Y_MAX,
            "n_steps": N_STEPS,
            "system": (
                "exact first-order shooting, full 8x8 Appendix-A system "
                "(include_spanwise_dissipation_coupling=True); compare with "
                "Table 10.1 8th-order column"
            ),
        },
        "anchor": anchor_info,
        "axis_convention": {
            "x": "R x 1e-2 (x=15 means R=1500)",
            "y": "omega_i x 1e3 on Mack's L* scale",
        },
        "digitized_reference": f"reference_data/{DIGITIZED_REFERENCE}",
        "table_anchor_markers": [
            {
                "R": case.Re_L,
                "alpha_L": case.alpha_L,
                "omega_i_6th": case.omega_i_6th,
                "omega_i_8th": case.omega_i_8th,
                "marker_value": "omega_i_6th (Mack's primary 6th-order system)",
            }
            for case in table_cases.values()
        ],
        "table_cross_checks": checks,
        "table_cross_checks_passed": all_passed,
        "results": results,
        "outputs": {
            "png": str(output_png),
            "json": str(output_json),
            "csv": str(output_csv),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print(f"output_png={output_png}")
    print(f"output_json={output_json}")
    print(f"output_csv={output_csv}")
    print(f"runtime_s={metadata['runtime_s']}")

    if checks and not all_passed:
        print(
            "ERROR: Mack Table 10.1 cross-check failed beyond "
            f"{100 * args.table_rtol:.1f} % — something is wrong "
            "(see table_cross_checks in the JSON metadata).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
