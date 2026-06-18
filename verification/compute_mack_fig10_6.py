"""Compute pyMack's maximum second-mode TEMPORAL growth rate omega_i,max(R)
for Mack (1984) AGARD R-709 Fig. 10.6 (adiabatic flat plate, M=4.5/5.8/7/10).

This is the reusable engine behind the Fig. 10.6 verification. It implements
the *validated* recipe (the diagnostic showed the earlier ~6x gap was an
edge-TEMPERATURE error, not a length-scale mapping issue):

  - Mean flow : pymack.make_mack_profile(M, condition='table_11_1')
                (Mack's COLD hypersonic-tunnel edge: ~60 K at M4.5, 50 K at
                M>=5), adiabatic wall, viscosity_model='mack'.
                NOT 'wind_tunnel'.
  - Eigenvalue: pymack.solver.solve_temporal_compressible(profile, alpha, R, M,
                Pr, gamma, N=110, y_max=30, length_scale='L_star',
                lambda_mu_ratio=0.0). Temporal: fixed real alpha -> complex c;
                omega_i = alpha * c_i.
  - Mode      : the discrete SECOND mode -- the high phase-speed acoustic mode
                with c_r ~ 0.88-0.97. Selected from the spectrum by a phase-
                speed band (c_r in [0.78, 0.99]) AND a tight c_i cap that
                rejects the spurious numerical root that appears around the
                peak with c_i ~ 0.5 (orders of magnitude too large for a
                boundary-layer mode). Of the survivors the largest c_i is taken.
  - omega_i,max(R): maximize omega_i over real alpha per R, via a coarse alpha
                scan then a fine local refine (the temporal peak is SHARP).

SELF-CHECK (the diagnostic's anchor): M=4.5, R=1500 -> omega_i,max ~ 3.36e-3,
matching Mack's digitized 3.36e-3. Run with --self-check.

Single-thread BLAS is forced in-process (os.environ set BEFORE numpy/scipy
import) so the dense EVP sweeps do not oversubscribe cores.

CLI
---
  python verification/compute_mack_fig10_6.py --self-check
  python verification/compute_mack_fig10_6.py --mach 4.5
  python verification/compute_mack_fig10_6.py --mach 7.0 --r-list 300 500 ...
"""

from __future__ import annotations

# --- Force single-thread BLAS BEFORE importing numpy/scipy -----------------
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")

import argparse
import sys

import numpy as np

import pymack
from pymack.solver import solve_temporal_compressible


# --- Constants --------------------------------------------------------------
PR = 0.72
GAMMA = 1.4
N_DEFAULT = 110
Y_MAX_DEFAULT = 30.0

# Phase-speed band for the discrete SECOND mode (high-c_r acoustic mode).
# At M4.5-7 the second-mode c_r sits ~0.88-0.96; a slightly wider window
# tolerates the M-dependence while still excluding the low-c_r vorticity /
# first-mode and the c_r->1 continuous spectrum.
CR_LO, CR_HI = 0.78, 0.99
# Tight imaginary cap: physical boundary-layer growth is |c_i| ~ O(1e-2). The
# generalized EVP occasionally returns a spurious root with c_i ~ 0.5 near the
# peak; this cap rejects it. (The solver's own physical filter only bounds
# |c_i| < 0.5, which is too loose to exclude that root.)
CI_CAP = 0.05

# Default R sweeps bracketing each panel's digitized R range (180-2000).
DEFAULT_R_SWEEPS = {
    4.5: [300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000],
    5.8: [240, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000],
    7.0: [240, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000],
    10.0: [300, 400, 520, 640, 800, 1000, 1200, 1400, 1600, 1800, 2000],
}

# Per-Mach alpha scan window for the second-mode peak. The peak wavenumber on
# the L* scale DECREASES with M (the boundary layer thickens — delta*/L* grows
# from ~8 at M4.5 to ~37 at M10 — so the second-mode wavelength ~2*delta grows
# and alpha ~ pi/delta shrinks). Verified peaks: M4.5 ~0.22, M5.8 ~0.14. Windows
# bracket those generously on both sides; the M7/M10 peaks are estimated lower.
# (An earlier version had these increasing with M, which made the scan MISS the
# true peak at M>=5.8 and report a spurious low-growth mode — a tool artifact.)
ALPHA_SCAN = {
    4.5: (0.08, 0.40, 0.005),
    5.8: (0.05, 0.30, 0.005),
    7.0: (0.03, 0.26, 0.005),
    10.0: (0.02, 0.22, 0.005),
}


def make_profile(mach: float):
    """Mack cold-tunnel (table_11_1) adiabatic flat-plate profile."""
    return pymack.make_mack_profile(float(mach), condition="table_11_1")


def second_mode_growth(profile, alpha, R, mach, *, N=N_DEFAULT,
                       y_max=Y_MAX_DEFAULT, cr_lo=CR_LO, cr_hi=CR_HI,
                       ci_cap=CI_CAP):
    """Return (c, omega_i) for the discrete second mode at this (alpha, R).

    omega_i = alpha * c_i. Returns (None, None) if no eigenvalue falls in the
    second-mode band.
    """
    c, _modes, _y = solve_temporal_compressible(
        profile, float(alpha), float(R), float(mach), PR, GAMMA,
        N=N, y_max=y_max, length_scale="L_star", lambda_mu_ratio=0.0,
    )
    cr = c.real
    ci = c.imag
    band = (cr > cr_lo) & (cr < cr_hi) & (ci < ci_cap)
    if not np.any(band):
        return None, None
    idx = np.where(band)[0]
    best = idx[int(np.argmax(ci[idx]))]
    return c[best], float(alpha) * float(ci[best])


def maximize_growth(profile, R, mach, *, N=N_DEFAULT, y_max=Y_MAX_DEFAULT,
                    alpha_scan=None, verbose=False):
    """Maximize the second-mode omega_i over real alpha at this R.

    Coarse scan then a fine local refine around the coarse peak (the peak is
    sharp). Returns (omega_i_max, alpha_peak, c_peak) or (None, None, None) if
    the second mode could not be isolated anywhere in the scan.
    """
    if alpha_scan is None:
        alpha_scan = ALPHA_SCAN.get(round(mach, 1), (0.03, 0.45, 0.005))
    a0, a1, da = alpha_scan

    best_oi, best_a, best_c = -np.inf, None, None
    for a in np.arange(a0, a1 + 0.5 * da, da):
        c, oi = second_mode_growth(profile, a, R, mach, N=N, y_max=y_max)
        if oi is not None and oi > best_oi:
            best_oi, best_a, best_c = oi, float(a), c
    if best_a is None:
        return None, None, None

    # Fine local refine (+/- one coarse step, 1/4-step resolution).
    for a in np.arange(best_a - da, best_a + da + 1e-9, da / 4.0):
        if a <= 0:
            continue
        c, oi = second_mode_growth(profile, a, R, mach, N=N, y_max=y_max)
        if oi is not None and oi > best_oi:
            best_oi, best_a, best_c = oi, float(a), c
    if verbose:
        print(f"   R={R:6.0f}  alpha*={best_a:.4f}  c_r={best_c.real:.4f} "
              f"c_i={best_c.imag:+.5f}  omega_i,max={best_oi:+.5e}")
    return best_oi, best_a, best_c


# --- Parallel path: one independent (mach, R) work unit per core -----------
# The dense temporal EVPs are single-threaded (BLAS pinned to 1 above), so we
# get throughput by running MANY (mach, R) units concurrently across cores.
import concurrent.futures as _cf  # noqa: E402

_PROFILE_CACHE = {}


def _get_profile(mach):
    key = round(float(mach), 4)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = make_profile(key)
    return _PROFILE_CACHE[key]


def work_unit(args):
    """Module-level worker (picklable for ProcessPoolExecutor spawn).

    args = (mach, R, N, y_max). Returns a result dict for this (mach, R).
    """
    mach, R, N, y_max = args
    profile = _get_profile(mach)
    oi, a, c = maximize_growth(profile, R, mach, N=N, y_max=y_max,
                               alpha_scan=ALPHA_SCAN.get(round(mach, 1)),
                               verbose=False)
    if oi is None:
        return {"mach": float(mach), "R": float(R), "omega_i_max": None,
                "alpha_peak": None, "c_r": None, "c_i": None}
    return {"mach": float(mach), "R": float(R), "omega_i_max": float(oi),
            "alpha_peak": float(a), "c_r": float(c.real), "c_i": float(c.imag)}


def compute_curves_parallel(machs, *, N=N_DEFAULT, y_max=Y_MAX_DEFAULT,
                            max_workers=48):
    """Compute omega_i,max(R) for several Mach numbers in parallel.

    Builds the full (mach, R) work list across all requested Mach numbers and
    runs them in one process pool (each unit single-threaded). Returns
    {mach: [row, ...]} with rows sorted by R.
    """
    units = []
    for mach in machs:
        rl = DEFAULT_R_SWEEPS.get(round(float(mach), 1))
        if rl is None:
            raise ValueError(f"no default R sweep for M={mach}")
        for R in rl:
            units.append((float(mach), float(R), N, y_max))

    n_workers = max(1, min(max_workers, len(units)))
    print(f"parallel: {len(units)} (mach,R) units across {n_workers} workers "
          f"(of {os.cpu_count()} cores), N={N}", flush=True)

    by_mach = {}
    with _cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(work_unit, units):
            by_mach.setdefault(round(res["mach"], 1), []).append(res)
            tag = ("ok" if res["omega_i_max"] else "--")
            print(f"  done M={res['mach']:.1f} R={res['R']:.0f} [{tag}]", flush=True)
    for m in by_mach:
        by_mach[m].sort(key=lambda r: r["R"])
    return by_mach


def compute_curve(mach, r_list=None, *, N=N_DEFAULT, y_max=Y_MAX_DEFAULT,
                  verbose=True):
    """Compute omega_i,max(R) over an R sweep for one Mach number.

    Returns a list of dicts: {R, omega_i_max, alpha_peak, c_r, c_i}.
    """
    mach = float(mach)
    if r_list is None:
        r_list = DEFAULT_R_SWEEPS.get(round(mach, 1))
        if r_list is None:
            raise ValueError(f"no default R sweep for M={mach}; pass --r-list")
    profile = make_profile(mach)
    alpha_scan = ALPHA_SCAN.get(round(mach, 1))
    out = []
    if verbose:
        print(f"=== M={mach}  N={N}  y_max={y_max}  condition=table_11_1 ===")
    for R in r_list:
        oi, a, c = maximize_growth(profile, R, mach, N=N, y_max=y_max,
                                   alpha_scan=alpha_scan, verbose=verbose)
        if oi is None:
            out.append({"R": float(R), "omega_i_max": None, "alpha_peak": None,
                        "c_r": None, "c_i": None})
        else:
            out.append({"R": float(R), "omega_i_max": float(oi),
                        "alpha_peak": float(a), "c_r": float(c.real),
                        "c_i": float(c.imag)})
    return out


def run_self_check():
    """M=4.5, R=1500 must give omega_i,max ~ 3.36e-3 (Mack + diagnostic)."""
    print("SELF-CHECK: M=4.5, R=1500 (Mack digitized = 3.36e-3)")
    profile = make_profile(4.5)
    oi, a, c = maximize_growth(profile, 1500.0, 4.5, verbose=True)
    ratio = oi / 3.36e-3
    print(f"  omega_i,max = {oi:.6e}  at alpha={a:.4f}  "
          f"(c_r={c.real:.4f}, c_i={c.imag:+.5f})")
    print(f"  ratio vs Mack 3.36e-3 = {ratio:.4f}")
    ok = 0.90 <= ratio <= 1.10
    print(f"  SELF-CHECK {'PASS' if ok else 'FAIL'} "
          f"(accept band 0.90-1.10)")
    return ok, oi, a, c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-check", action="store_true",
                   help="run the M4.5/R1500 anchor self-check and exit")
    p.add_argument("--mach", type=float, default=None,
                   help="Mach number (4.5, 5.8, 7.0, 10.0)")
    p.add_argument("--r-list", type=float, nargs="+", default=None,
                   help="explicit R sweep; default brackets the digitized panel")
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--y-max", type=float, default=Y_MAX_DEFAULT)
    args = p.parse_args(argv)

    if args.self_check:
        ok, *_ = run_self_check()
        return 0 if ok else 1

    if args.mach is None:
        p.error("pass --self-check or --mach")

    rows = compute_curve(args.mach, r_list=args.r_list, N=args.N,
                         y_max=args.y_max, verbose=True)
    print("\nR, omega_i_max, alpha_peak, c_r, c_i")
    for r in rows:
        if r["omega_i_max"] is None:
            print(f"{r['R']:.0f}, NONE")
        else:
            print(f"{r['R']:.0f}, {r['omega_i_max']:.6e}, "
                  f"{r['alpha_peak']:.4f}, {r['c_r']:.4f}, {r['c_i']:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
