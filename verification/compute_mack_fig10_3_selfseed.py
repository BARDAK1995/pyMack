"""Compute pyMack's maximum FIRST-mode TEMPORAL growth omega_i,max(R) for the
two Mack (1984) Fig. 10.3 families that have a digitized curve but NO Table 10.1
anchor: M=2.2/psi=45 and M=3.0/psi=60.

WHY a new engine (the blocker): scripts/make_mack_fig10_3_overlay.py anchors the
exact-shooting continuation on a Mack Table 10.1 row for the requested (M, psi).
Table 10.1 has no (2.2,45) or (3.0,60) row, so that script errors out rather than
fabricate an anchor. Here we SELF-SEED: at one (R, alpha) per Reynolds number we
run the same validated exact first-order shooting search
(find_temporal_mode_anchor_3d_shooting, full 8x8 Appendix-A system) over a fan of
first-mode seeds spanning the oblique first-mode phase-speed band (c_r ~ 0.3-0.7
for these psi), then CONTINUE that branch across a wavenumber scan
(temporal_growth_scan_3d_shooting_from_anchor) and maximize omega_i over alpha.

We deliberately do NOT use the reduced dense EVP (solve_temporal_compressible_3d):
verified at M2.2/R600/alpha0.07 it returns ONLY decaying modes (most-unstable
filtered root c=0.637-0.018j, omega_i<0) while exact shooting finds the GROWING
first mode c=0.581+0.0083j. The reduced EVP misses this first-mode branch -- using
it would spuriously kill the mode (a tool artifact). The validated M=1.3 Fig 10.3
path uses exact shooting for exactly this reason.

Physics / conventions (identical to the validated M=1.3 Fig 10.3 path):
  - Mean flow : make_mack_profile(M, condition='table_11_1'), adiabatic wall,
                Mack viscosity (cold-edge schedule; same as the validated path).
  - Solver    : exact first-order shooting, full 8x8 Appendix-A system,
                psi_deg=psi, length_scale='L_star', wall_bc='isothermal',
                method='qr'. omega_i = alpha * c_i (Mack's L* scale).
  - Domain    : y_max ~ 4x delta*/L* (high-Mach domain lesson). delta*/L* ~ 3.7
                (M2.2) / ~5.0 (M3.0). y_max convergence checked at 3x/4x/4.5x.
  - Mode pick : converged physical first-mode root -- c_r in [0.2,0.95],
                |c_i|<0.3, sigma_min<=1e-3.

Axis convention (same as the digitized paper CSV):
    x = R x 1e-2          (x=15 -> R=1500)
    y = omega_i x 1e3

PARALLEL: each Reynolds number is one independent process-pool work unit (anchor
+ alpha-scan in that one process). Process isolation contains the intermittent
native MKL/OMP abort seen on this Windows box -- a crashed worker loses one R, not
the sweep. Single-thread BLAS forced per process; KMP_DUPLICATE_LIB_OK set.

CLI
---
  python verification/compute_mack_fig10_3_selfseed.py --probe
  python verification/compute_mack_fig10_3_selfseed.py --mach 2.2 --psi 45 \
      --workers 11 --out /path/result.json
  python verification/compute_mack_fig10_3_selfseed.py --mach 2.2 --psi 45 \
      --ymax-factor 4.5 --n-steps 1200   # convergence variant
"""
from __future__ import annotations

# --- Force single-thread BLAS + Windows MKL/OMP dup-runtime fix BEFORE numpy --
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pymack import make_mack_profile  # noqa: E402
from pymack.analysis import (  # noqa: E402
    find_temporal_mode_anchor_3d_shooting,
    temporal_growth_scan_3d_shooting_from_anchor,
)
from pymack.scales import delta_star_over_lstar  # noqa: E402

WALL_BC = "isothermal"
LENGTH_SCALE = "L_star"

# First-mode anchor seeds spanning the oblique first-mode phase-speed band.
# Verified: seeds 0.40-0.62 all converge to the same first-mode root (e.g.
# c~0.547+0.0198j at M2.2/R600/alpha0.05). One robust seed suffices; a short
# fan guards off-peak alphas and the higher-Mach band.
SEED_FAN = [0.50 + 0.010j, 0.45 + 0.008j, 0.58 + 0.013j]

# Converged-physical first-mode screen.
C_REAL_BOUNDS = (0.20, 0.95)
C_IMAG_ABS_MAX = 0.30
ROOT_SIGMA_TOL = 1e-3

# Per-family R sweep (x = R*1e-2 spans 2..20 in the digitized CSV).
R_SWEEP = [200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
           1200.0, 1500.0, 1800.0, 2000.0]

# Per-family wavenumber scan grid (L* scale). The first-mode peak alpha sits low
# and drifts slightly with R; verified peak at M2.2/R600 is alpha~0.05. The grid
# brackets the peak on BOTH sides (down to 0.02) at every R.
# NOTE (2026-07-02): (1.6, 45.0) added here for the Fig 10.3 M1.6 cross-check
# task, following the exact same pattern as the two existing families. As of
# this date the module-level import of find_temporal_mode_anchor_3d_shooting /
# temporal_growth_scan_3d_shooting_from_anchor at the top of this file FAILS:
# pymack 0.1.0's "curated public API" repackaging (commit d0ddcd3) removed or
# renamed these internal shooting helpers (scripts/make_mack_fig10_3_overlay.py
# has the identical broken import). This is a pymack-side API break unrelated
# to the (1.6, 45.0) parameters added here -- fixing it means adapting this
# whole self-seed/continuation pipeline to pymack's new public solver API
# (solve_temporal_mode_3d_shooting_sigma_min / continue_temporal_mode_3d_shooting_sigma_min
# have different signatures and return shapes), which is real restructuring,
# not a param tweak. Left unresolved per task scope; the M1.6 Fig 10.3 curves
# were finalized on tracing/visual confidence alone, no pyMack cross-check.
ALPHA_GRID = {
    (1.6, 45.0): [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.10, 0.12],
    (2.2, 45.0): [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.10, 0.12],
    (3.0, 60.0): [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.10, 0.12],
}
# Anchor wavenumber per family (interior of the grid, near the expected peak).
ANCHOR_ALPHA = {(1.6, 45.0): 0.05, (2.2, 45.0): 0.05, (3.0, 60.0): 0.05}

N_STEPS_DEFAULT = 800  # verified identical root to n=1500 at 6x lower cost


def make_profile(mach):
    return make_mack_profile(float(mach), condition="table_11_1")


def _select_anchor_c(res):
    """Pick the converged physical first-mode root with the largest omega_i."""
    best_c, best_oi = None, -np.inf
    alpha = res["alpha"]
    for cd in res["candidates"]:
        c = cd.get("c_final")
        if c is None or not (np.isfinite(c.real) and np.isfinite(c.imag)):
            c = cd.get("c_sigma_min")
        if c is None or not (np.isfinite(c.real) and np.isfinite(c.imag)):
            continue
        if not (C_REAL_BOUNDS[0] < c.real < C_REAL_BOUNDS[1]):
            continue
        if abs(c.imag) >= C_IMAG_ABS_MAX:
            continue
        sig = cd["sigma_min"]
        if not (np.isfinite(sig) and sig <= ROOT_SIGMA_TOL):
            continue
        oi = float(alpha) * c.imag
        if oi > best_oi:
            best_oi, best_c = oi, complex(c)
    return best_c


def first_mode_root_at(profile, R, alpha, mach, psi_deg, *, y_max, n_steps):
    """Self-seed: find one converged first-mode root at (R, alpha)."""
    res = find_temporal_mode_anchor_3d_shooting(
        profile, float(R), float(alpha), Ma=float(mach), seed_list=SEED_FAN,
        psi_deg=float(psi_deg), y_max=float(y_max), n_steps=int(n_steps),
        wall_bc=WALL_BC, length_scale=LENGTH_SCALE, method="qr",
    )
    return _select_anchor_c(res)


def scan_one_reynolds(mach, psi_deg, R, alpha_grid, anchor_alpha, *,
                      y_max, n_steps):
    """Anchor the first mode at (R, anchor_alpha), continue across alpha_grid.

    Returns the full scan (alpha, omega_i, c, sigma_min) plus the converged
    interior optimum (parabolic-refined). All physics screening is applied so a
    diverged point cannot win the optimum.
    """
    profile = make_profile(mach)
    alphas = np.asarray(alpha_grid, dtype=float)
    anchor_index = int(np.argmin(np.abs(alphas - anchor_alpha)))

    anchor_c = first_mode_root_at(profile, R, alphas[anchor_index], mach,
                                  psi_deg, y_max=y_max, n_steps=n_steps)
    if anchor_c is None:
        return None

    a, oi, cv, sg, _ = temporal_growth_scan_3d_shooting_from_anchor(
        profile, float(R), float(mach), alphas, anchor_index=anchor_index,
        initial_c=anchor_c, psi_deg=float(psi_deg), y_max=float(y_max),
        n_steps=int(n_steps), wall_bc=WALL_BC, length_scale=LENGTH_SCALE,
        method="qr",
    )
    oi = np.asarray(oi, float)
    cv = np.asarray(cv, complex)
    sg = np.asarray(sg, float)
    physical = (
        np.isfinite(oi) & np.isfinite(sg) & np.isfinite(cv.real)
        & np.isfinite(cv.imag)
        & (cv.real > C_REAL_BOUNDS[0]) & (cv.real < C_REAL_BOUNDS[1])
        & (np.abs(cv.imag) < C_IMAG_ABS_MAX) & (sg <= ROOT_SIGMA_TOL)
    )
    oi_screen = np.where(physical, oi, -np.inf)
    if not np.any(np.isfinite(oi_screen) & (oi_screen > -np.inf)):
        return None
    idx = int(np.argmax(oi_screen))
    alpha_peak, oi_peak = float(a[idx]), float(oi[idx])
    refined = False
    if 0 < idx < len(a) - 1 and physical[idx - 1] and physical[idx + 1]:
        xx, yy = a[idx - 1:idx + 2], oi[idx - 1:idx + 2]
        co = np.polyfit(xx, yy, 2)
        if co[0] < 0:
            astar = -co[1] / (2 * co[0])
            if xx[0] <= astar <= xx[2]:
                alpha_peak, oi_peak = float(astar), float(np.polyval(co, astar))
                refined = True
    return {
        "R": float(R),
        "alpha_grid": [float(x) for x in a],
        "omega_i": [float(v) if np.isfinite(v) else None for v in oi],
        "sigma_min": [float(s) if np.isfinite(s) else None for s in sg],
        "c_r": [float(z.real) for z in cv],
        "c_i": [float(z.imag) for z in cv],
        "physical": [bool(b) for b in physical],
        "anchor_index": anchor_index,
        "anchor_c": [anchor_c.real, anchor_c.imag],
        "idx_opt": idx,
        "alpha_peak": alpha_peak,
        "omega_i_max": oi_peak,
        "alpha_peak_discrete": float(a[idx]),
        "omega_i_max_discrete": float(oi[idx]),
        "c_r_at_opt": float(cv[idx].real),
        "c_i_at_opt": float(cv[idx].imag),
        "refined": refined,
        "bracket_edge": bool(idx == 0 or idx == len(a) - 1),
        "n_physical": int(np.sum(physical)),
    }


# --- Parallel: one independent Reynolds number per work unit -----------------

def work_unit(args):
    mach, psi_deg, R, alpha_grid, anchor_alpha, y_max, n_steps = args
    try:
        row = scan_one_reynolds(mach, psi_deg, R, alpha_grid, anchor_alpha,
                                y_max=y_max, n_steps=n_steps)
    except Exception as exc:  # noqa: BLE001
        return {"R": float(R), "omega_i_max": None,
                "error": f"{type(exc).__name__}: {exc}"}
    if row is None:
        return {"R": float(R), "omega_i_max": None,
                "error": "no converged first-mode root"}
    row["error"] = None
    return row


def compute(mach, psi_deg, *, ymax_factor=4.0, y_max=None,
            n_steps=N_STEPS_DEFAULT, workers=11, r_sweep=None, alpha_grid=None):
    key = (round(mach, 1), round(psi_deg, 1))
    if alpha_grid is None:
        alpha_grid = ALPHA_GRID[key]
    anchor_alpha = ANCHOR_ALPHA[key]
    if r_sweep is None:
        r_sweep = R_SWEEP

    profile = make_profile(mach)
    dstar = delta_star_over_lstar(profile)
    if y_max is None:
        y_max = round(ymax_factor * dstar, 1)

    print(f"=== M={mach}, psi={psi_deg} deg, T_edge={profile.T_edge:.1f} K, "
          f"delta*/L*={dstar:.3f}, y_max={y_max} ({y_max/dstar:.2f}x), "
          f"n_steps={n_steps}, n_alpha={len(alpha_grid)}, n_R={len(r_sweep)} ===",
          flush=True)

    units = [(float(mach), float(psi_deg), float(R), list(alpha_grid),
              float(anchor_alpha), float(y_max), int(n_steps)) for R in r_sweep]
    n = max(1, min(workers, len(units)))
    print(f"parallel: {len(units)} Reynolds units across {n} workers "
          f"(of {os.cpu_count()} cores)", flush=True)

    t0 = time.time()
    rows = []
    with cf.ProcessPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(work_unit, u): u[2] for u in units}
        for fut in cf.as_completed(futs):
            R = futs[fut]
            res = fut.result()
            rows.append(res)
            if res.get("omega_i_max") is not None:
                print(f"  R={R:6.0f}: omega_i_max={res['omega_i_max']:+.4e} "
                      f"alpha*={res['alpha_peak']:.4f} "
                      f"c={res['c_r_at_opt']:.4f}{res['c_i_at_opt']:+.5f}j"
                      + ("  [EDGE]" if res['bracket_edge'] else ""), flush=True)
            else:
                print(f"  R={R:6.0f}: -- ({res.get('error')})", flush=True)
    rows.sort(key=lambda r: r["R"])
    elapsed = time.time() - t0
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"  done in {elapsed:.0f}s, {n_err} R with no mode", flush=True)

    return {
        "mach": mach, "psi_deg": psi_deg, "T_edge": float(profile.T_edge),
        "delta_star_over_L": float(dstar), "y_max": float(y_max),
        "ymax_over_dstar": float(y_max / dstar), "n_steps": int(n_steps),
        "alpha_grid": list(alpha_grid), "anchor_alpha": float(anchor_alpha),
        "rows": rows, "n_errors": n_err, "elapsed_s": round(elapsed, 1),
    }


def print_curve(res):
    print(f"\nM={res['mach']} psi={res['psi_deg']} y_max={res['y_max']} "
          f"({res['ymax_over_dstar']:.2f}x dstar) n_steps={res['n_steps']}")
    print("    R      x     omega_i_max  y_paper  alpha*   c_r     c_i      edge")
    for row in res["rows"]:
        if row.get("omega_i_max") is None:
            print(f"  {row['R']:6.0f}  {row['R']*1e-2:5.1f}  -- {row.get('error','')}")
            continue
        print(f"  {row['R']:6.0f}  {row['R']*1e-2:5.1f}  {row['omega_i_max']:+.4e}  "
              f"{row['omega_i_max']*1e3:6.3f}  {row['alpha_peak']:.4f}  "
              f"{row['c_r_at_opt']:.4f}  {row['c_i_at_opt']:+.5f}  "
              f"{'EDGE' if row['bracket_edge'] else ''}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--mach", type=float, default=None)
    p.add_argument("--psi", type=float, default=None)
    p.add_argument("--ymax-factor", type=float, default=4.0)
    p.add_argument("--y-max", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT)
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--r-list", type=float, nargs="+", default=None,
                   help="explicit R subset (default: full sweep)")
    args = p.parse_args(argv)

    if args.probe:
        for mach, psi in [(1.6, 45.0), (2.2, 45.0), (3.0, 60.0)]:
            prof = make_profile(mach)
            d = delta_star_over_lstar(prof)
            ym = round(4 * d, 1)
            c = first_mode_root_at(prof, 600.0, 0.05, mach, psi,
                                   y_max=ym, n_steps=N_STEPS_DEFAULT)
            oi = None if c is None else 0.05 * c.imag
            print(f"M={mach} psi={psi}: T_edge={prof.T_edge:.1f}K "
                  f"dstar/L={d:.3f} y_max(4x)={ym} root@R600,a0.05: "
                  f"c={c} omega_i={oi}")
        return 0

    if args.mach is None or args.psi is None:
        p.error("pass --probe, or --mach and --psi")

    res = compute(args.mach, args.psi, ymax_factor=args.ymax_factor,
                  y_max=args.y_max, n_steps=args.n_steps, workers=args.workers,
                  r_sweep=args.r_list)
    print_curve(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
