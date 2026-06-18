#!/usr/bin/env python3
"""Mack (1984) Fig. 10.1 first-mode NEUTRAL-frequency verification engine.

Category: ``neutral_curve``. One verdict per Mach panel (M=1.6, M=2.2).

Fig. 10.1 plots a single-valued, monotonically decreasing neutral-stability
*frequency* F x 1e4 versus Reynolds number R = sqrt(Re_x), for the 2-D first
mode of an adiabatic flat plate at low supersonic Mach. (F = omega_dim nu_e/U_e**2
= omega_L/R, with omega_L = alpha_L c_r the L*-scaled circular frequency.)

This engine measures pyMack's agreement HONESTLY:

  * compute_mack_fig10_1 supplies, per R, the first-mode omega_i(alpha) lobe on
    the L* scale (temporal: real alpha -> complex c; omega_i = alpha c_i).
  * For each R the c_i = 0 (omega_i = 0) boundary has up to two crossings in
    alpha: a LOWER (onset, - -> +) and an UPPER (cutoff, + -> -). Each maps to a
    neutral frequency F = alpha_neutral c_r / R * 1e4. We also record the
    most-amplified frequency F_peak at the alpha of maximum omega_i.
  * The digitized Mack 'complete' curve F_dig(R) is the comparison target. We
    interpolate pyMack's lower-branch, upper-branch and peak-frequency curves
    onto the digitized R's (over the overlapping R range where pyMack is
    actually unstable) and report the median relative error of each against
    F_dig, picking the BEST-matching branch as the headline (the single-valued
    digitized curve does not say which neutral branch it is, so we let pyMack's
    closest neutral locus carry the score -- analogous to the Ozgen Fig.3
    nearest-crossing rule).
  * Topology is part of the verdict: Mack's curve exists (is unstable) down to
    R~80-90; if pyMack's first mode is stable there (no neutral band), that
    censored low-R region is a topology/critical-R discrepancy, reported as a
    coverage caveat and folded into the verdict honestly.

Classification uses the shared 3-tier thresholds in _compare_lib. The verdict is
written into verification/neutralCurve_verification/mack_fig10_1_m{tok}/verdict.json.

The grid itself is produced by compute_mack_fig10_1.compute_grid_parallel (single
-thread BLAS, parallel across cores). This script can recompute it or load a
cached grid CSV.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compare_lib import classify_relative, write_verdict  # noqa: E402
from compute_mack_fig10_1 import (  # noqa: E402
    compute_grid_parallel,
    neutral_frequencies,
)

REPO = Path(__file__).resolve().parent.parent
DIG = REPO / "reference_data" / "digitized"
OUT_ROOT = REPO / "verification" / "neutralCurve_verification"
SOURCE = "Mack (1984) Fig 10.1 (AGARD R-709)"

# R sweep for the first-mode neutral extraction.
R_LIST = [200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600]


def alpha_list_for(mach):
    """Per-Mach first-mode alpha window (true mode c_r ~ 0.5).

    The unstable lobe's upper cutoff drifts up with Mach: at M1.6 it sits below
    alpha~0.10, at M2.2 it reaches ~0.12-0.13, so M>=2.0 needs the ceiling
    extended to ~0.135 to bracket the upper neutral crossing (otherwise the
    cutoff is censored and the upper branch is spuriously reported absent --
    a scan-window artifact). CR_LO=0.40 in the engine rejects the spurious
    c_r~0.31 lobe that appears at still-higher alpha.
    """
    a_hi = 0.1375 if mach >= 2.0 else 0.1225
    return list(np.round(np.arange(0.015, a_hi, 0.0025), 5))

CASES = {
    1.6: {"tok": "16", "case_id": "mack_fig10_1_m1p6"},
    2.2: {"tok": "22", "case_id": "mack_fig10_1_m2p2"},
}


def load_complete(tok):
    p = DIG / f"mack_ch10_fig10_1_M{tok}_paper_complete.csv"
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["x"]), float(r["y"])))
    a = np.array(rows)
    return a[:, 0], a[:, 1], p


def extract_branches(grid):
    """grid: dict[R]->scan -> dict of arrays R, F_lower, F_upper, F_peak, peak_oi."""
    Rs = sorted(grid)
    R = np.array(Rs, float)
    Flo = np.full(R.size, np.nan)
    Fup = np.full(R.size, np.nan)
    Fpk = np.full(R.size, np.nan)
    oi = np.full(R.size, np.nan)
    for i, r in enumerate(Rs):
        nf = neutral_frequencies(grid[r])
        if nf["lower"] is not None:
            Flo[i] = nf["lower"]
        if nf["upper"] is not None:
            Fup[i] = nf["upper"]
        if nf["peak_F"] is not None:
            Fpk[i] = nf["peak_F"]
        if nf["peak_oi"] is not None:
            oi[i] = nf["peak_oi"]
    return {"R": R, "F_lower": Flo, "F_upper": Fup, "F_peak": Fpk, "peak_oi": oi}


def branch_median_relerr(Rb, Fb, Rd, Fd):
    """Median rel-err of branch curve Fb(Rb) vs digitized Fd(Rd) on overlap.

    Interpolates the digitized curve onto the branch R's where Fb is finite and
    inside the digitized R span. Returns (median_rel, n, R_overlap_lo, R_overlap_hi).
    """
    finite = np.isfinite(Fb)
    if not finite.any():
        return None, 0, None, None
    Rmask = (Rb >= Rd.min()) & (Rb <= Rd.max()) & finite
    if not Rmask.any():
        return None, 0, None, None
    rr = Rb[Rmask]
    fb = Fb[Rmask]
    fd = np.interp(rr, Rd, Fd)
    rel = np.abs(fb - fd) / np.maximum(np.abs(fd), 1e-9)
    return float(np.median(rel)), int(rr.size), float(rr.min()), float(rr.max())


def compare_one(mach, grid):
    tok = CASES[mach]["tok"]
    Rd, Fd, ref_path = load_complete(tok)
    br = extract_branches(grid)

    # critical-R / topology: lowest R at which pyMack's first mode is unstable
    unstable = np.isfinite(br["peak_oi"]) & (br["peak_oi"] > 0)
    R_crit_pymack = float(br["R"][unstable].min()) if unstable.any() else None
    R_dig_lo = float(Rd.min())

    results = {}
    for name, key in (("lower", "F_lower"), ("upper", "F_upper"),
                      ("peak", "F_peak")):
        med, n, rlo, rhi = branch_median_relerr(br["R"], br[key], Rd, Fd)
        results[name] = {"median_rel_err": med, "n": n,
                         "R_overlap": [rlo, rhi]}

    # Headline = best-matching branch with at least 4 overlap points.
    candidates = [(v["median_rel_err"], k, v) for k, v in results.items()
                  if v["median_rel_err"] is not None and v["n"] >= 4]
    if candidates:
        candidates.sort(key=lambda t: t[0])
        best_rel, best_branch, best = candidates[0]
    else:
        best_rel, best_branch, best = None, None, None

    # Topology: Mack's neutral frequency exists (unstable) down to R~80-90.
    # pyMack is stable below R_crit_pymack. If R_crit_pymack >> R_dig_lo, the
    # low-R portion of the digitized curve has NO pyMack counterpart -> a real
    # critical-Reynolds / under-amplification topology gap (the documented
    # first-mode weakness). We require the unstable overlap to cover most of the
    # digitized R range for topology_ok.
    if R_crit_pymack is None:
        topology_ok = False
        covered_frac = 0.0
    else:
        n_dig_covered = int(np.sum(Rd >= R_crit_pymack))
        covered_frac = n_dig_covered / Rd.size
        topology_ok = covered_frac >= 0.85

    if best_rel is None:
        verdict = "disagrees"
        reason = (
            f"pyMack's first mode has no neutral band overlapping the digitized "
            f"R range [{R_dig_lo:.0f},{Rd.max():.0f}] for M={mach}: a complete "
            f"topology gap (R_crit_pymack={R_crit_pymack})."
        )
    else:
        verdict = classify_relative(best_rel, topology_ok)
        parts = [
            f"pyMack first-mode neutral frequency vs Mack's digitized 'complete' "
            f"curve, M={mach}: best-matching branch = '{best_branch}', median "
            f"|dF|/F = {best_rel*100:.1f}% over {best['n']} R points in "
            f"[{best['R_overlap'][0]:.0f},{best['R_overlap'][1]:.0f}]."
        ]
        for nm in ("lower", "upper", "peak"):
            r = results[nm]
            if r["median_rel_err"] is not None:
                parts.append(f"{nm} branch median {r['median_rel_err']*100:.0f}% "
                             f"({r['n']} pts).")
        if not topology_ok:
            parts.append(
                f"Topology/critical-R gap: Mack's curve is unstable down to "
                f"R={R_dig_lo:.0f}, but pyMack's first mode is STABLE below "
                f"R~{R_crit_pymack:.0f} (only {covered_frac*100:.0f}% of the "
                f"digitized R range is covered by an unstable pyMack band). This "
                f"is the repo's documented low-Mach first-mode under-amplification "
                f"/ too-high critical Reynolds number -- a real physics "
                f"discrepancy, not digitization noise -- so the verdict is "
                f"'{verdict}'."
            )
        reason = " ".join(parts)

    metrics = {
        "headline_branch": best_branch,
        "headline_median_rel_err": best_rel,
        "median_rel_err_lower": results["lower"]["median_rel_err"],
        "median_rel_err_upper": results["upper"]["median_rel_err"],
        "median_rel_err_peak": results["peak"]["median_rel_err"],
        "n_overlap_lower": results["lower"]["n"],
        "n_overlap_upper": results["upper"]["n"],
        "n_overlap_peak": results["peak"]["n"],
        "R_crit_pymack": R_crit_pymack,
        "R_dig_lo": R_dig_lo,
        "digitized_R_coverage_fraction": round(covered_frac, 3),
        "topology_ok": bool(topology_ok),
        "y_axis": "F x 1e4, F = omega_L / R, omega_L = alpha_L * c_r",
    }
    return {"metrics": metrics, "verdict": verdict, "reason": reason,
            "ref_path": ref_path, "branches": br, "Rd": Rd, "Fd": Fd}


def write_pymack_csv(out_dir, mach, br):
    path = out_dir / f"pymack_mack_fig10_1_M{CASES[mach]['tok']}_neutral.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R", "F_lower_x1e4", "F_upper_x1e4", "F_peak_x1e4", "peak_omega_i"])
        for i, r in enumerate(br["R"]):
            def g(a): return "" if not np.isfinite(a[i]) else f"{a[i]:.6f}"
            w.writerow([f"{r:.0f}", g(br["F_lower"]), g(br["F_upper"]),
                        g(br["F_peak"]), g(br["peak_oi"])])
    return path


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mack Fig 10.1 first-mode neutral-frequency verification.")
    ap.add_argument("--machs", default="1.6,2.2")
    # y_max=45 (N=140): the first-mode neutral frequencies shift ~11-13% from
    # y_max=30->45 (the marginal mode under-resolves on a too-short box), so the
    # better-converged domain is used for the recorded verdict. delta*/L* ~ 2.8
    # (M1.6) / 3.7 (M2.2) so y_max=45 is ~12-16x delta* -- amply converged.
    ap.add_argument("--N", type=int, default=140)
    ap.add_argument("--y-max", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    machs = [float(x) for x in args.machs.split(",") if x.strip()]
    for mach in machs:
        print(f"\n########## M={mach} ##########", flush=True)
        grid = compute_grid_parallel(mach, R_LIST, alpha_list_for(mach), N=args.N,
                                     y_max=args.y_max, max_workers=args.workers)
        res = compare_one(mach, grid)
        out_dir = OUT_ROOT / CASES[mach]["case_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        pm_csv = write_pymack_csv(out_dir, mach, res["branches"])
        # self-contained reference copy
        ref_dst = out_dir / f"reference_mack_fig10_1_M{CASES[mach]['tok']}_complete.csv"
        import shutil
        shutil.copyfile(res["ref_path"], ref_dst)

        verdict = {
            "case_id": CASES[mach]["case_id"],
            "category": "neutral_curve",
            "source": SOURCE,
            "conditions": {
                "Ma": float(mach),
                "gas": "air",
                "wall": "adiabatic",
                "psi_deg": 0,
                "formulation": "temporal 2D first-mode neutral frequency",
                "transport": "Mack (viscosity_model='mack')",
                "condition": "table_11_1 (Mack cold-edge total-temperature schedule)",
                "length_scale": "L_star = sqrt(nu_e x / U_e); R = sqrt(Re_x)",
                "N": args.N,
                "y_max": args.y_max,
                "lambda_mu_ratio": 0.0,
            },
            "quantity": "first-mode neutral-stability frequency F x 1e4 vs R (temporal, c_i=0)",
            "metrics": res["metrics"],
            "verdict": res["verdict"],
            "verdict_reason": res["reason"],
            "generated": "new",
            "artifacts": {
                "pymack": _rel(pm_csv),
                "reference": _rel(ref_dst),
                "overlay": None,
            },
            "pymack_provenance": (
                f"verification/compute_mack_fig10_1.py + compare_mack_fig10_1.py "
                f"(this session). make_mack_profile(M={mach}, condition='table_11_1') "
                f"adiabatic Mack flat plate (viscosity_model='mack'); "
                f"solve_temporal_compressible(length_scale='L_star', "
                f"lambda_mu_ratio=0.0, N={args.N}, y_max={args.y_max}); first mode "
                f"selected by c_r in [0.40,0.85], |c_i|<0.10, max-c_i in band. "
                f"omega_i(alpha) lobe per R over alpha in "
                f"[0.015,{alpha_list_for(mach)[-1]:.4f}]; neutral "
                f"frequency F = alpha_neutral*c_r/R*1e4 from the c_i=0 crossings "
                f"(lower/onset and upper/cutoff), plus F at peak omega_i. Grid "
                f"R={R_LIST}. Parallel across cores (single-thread BLAS)."
            ),
        }
        write_verdict(out_dir, verdict)
        print(f"=== M={mach} -> {res['verdict']} ===")
        print(f"  {res['reason']}")
        print(f"  wrote {out_dir/'verdict.json'}")
    print("ALLDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
