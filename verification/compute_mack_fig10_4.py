"""Compute pyMack's maximum FIRST-mode TEMPORAL growth omega_i,max(R) for
Mack (1984) AGARD R-709 Fig. 10.4 (adiabatic flat plate, M=4.5/5.8/7/10).

This is the FIRST-mode analogue of compute_mack_fig10_6.py. The high-Mach first
mode is OBLIQUE-dominated: its maximum temporal growth is over BOTH the
streamwise wavenumber alpha AND the wave angle psi (beta = alpha*tan(psi)).
Unlike the 2D second mode of Fig 10.6, a 2D (psi=0) search badly under-predicts
the first mode -- the oblique 3D wave is far more amplified (a textbook Mack
first-mode result).

Recipe (mirrors the validated Fig 10.6 recipe, swapped to the 3D solver + first
mode):
  - Mean flow : pymack.make_mack_profile(M, condition='table_11_1') (Mack's COLD
                hypersonic-tunnel edge), adiabatic wall, viscosity 'mack'.
  - Eigenvalue: pymack.solver.solve_temporal_compressible_3d(profile, alpha, beta,
                R, M, Pr, gamma, N, y_max, length_scale='L_star',
                lambda_mu_ratio=0.0). Returns complex c = omega/alpha;
                omega_i = alpha * c_i.
  - Mode      : the discrete FIRST mode -- the oblique-amplified mode with phase
                speed BELOW the 2nd-mode acoustic mode. Selected by a c_r band
                (CR_LO..CR_HI ~ 0.40..0.95) AND a c_i cap that rejects the
                spurious large-c_i numerical root. Of the survivors the largest
                c_i is taken.
  - omega_i,max(R): maximize omega_i over (alpha, psi). The first-mode growth
                peaks at high psi (~50-65 deg) and LOW alpha and is broad in psi
                but sharp in alpha; a coarse (alpha,psi) grid then a local alpha
                refine at the best psi.
  - Domain    : y_max ~4x delta*/L* per Mach (delta*/L* = 10.3/15.7/21.1/37.1 for
                M=4.5/5.8/7/10) with N scaled to resolve the wider box. A fixed
                short box starves the thick high-Mach BL and spuriously kills the
                mode -- the SAME domain lesson as Fig 10.6.

Digitized axes (reference_data/digitized/mack_ch10_fig10_4_M*_paper.csv):
  x = raw R, y = omega_i * 1e3.

Single-thread BLAS is forced in-process; sweeps run in a ProcessPool, one
independent (mach, R) unit per core.

CLI
---
  python verification/compute_mack_fig10_4.py --mach 4.5 --r-list 1500
  python verification/compute_mack_fig10_4.py --probe 4.5 1500
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
import concurrent.futures as _cf
import sys

import numpy as np
from scipy import linalg

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.solver import (
    apply_dirichlet_freestream_bc_3d,
    apply_wall_bc_3d,
    assemble_temporal_compressible_3d_evp,
    solve_temporal_compressible_3d,  # kept for reference / cross-checks
)


# --- Constants --------------------------------------------------------------
PR = 0.72
GAMMA = 1.4

# First-mode phase-speed band: the oblique first mode sits BELOW the 2nd-mode
# acoustic mode (c_r ~ 0.9-1.0) and above the slow vorticity / continuous
# modes. Across M4.5-10 the most-amplified oblique first mode has c_r ~ 0.6-0.92.
CR_LO, CR_HI = 0.40, 0.95
# Imaginary cap: physical first-mode c_i is O(1e-2). The dense EVP can return a
# spurious root with c_i ~ 0.3-0.5; this cap (looser than the 2nd-mode 0.05 cap
# because the first-mode c_i is larger and psi-amplified) rejects it while
# keeping the real mode. The solver's own filter only bounds |c_i| < 0.5.
CI_CAP = 0.12

# Default R sweeps bracketing each panel's digitized R range. M10 uses a leaner
# set (each M10 station is a ~119-solve dense-EVP grid at dim~1005, ~8 s/solve
# single-thread) so the parallel sweep stays tractable without oversubscribing.
DEFAULT_R_SWEEPS = {
    4.5: [200, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000],
    5.8: [220, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000],
    7.0: [240, 360, 520, 720, 920, 1120, 1320, 1520, 1720, 1920, 2000],
    10.0: [450, 650, 850, 1050, 1250, 1450, 1650, 1850, 2000],
}

# (alpha, psi) search grid per Mach. The first-mode peak alpha on the L* scale
# DECREASES with M (the BL thickens; delta*/L* grows 10->37), so the peak
# wavelength grows and alpha shrinks. psi is the wave angle in degrees;
# beta = alpha*tan(psi). The first-mode growth peaks at psi ~ 50-60 deg.
#
# IMPORTANT (alpha-window lesson): the grid must bracket the WHOLE first-mode
# ridge on BOTH sides. Broad diagnostic scans (verification/_probe_m10_wide.py)
# showed the M4.5 peak at alpha~0.055 and the M10 first-mode-band peak at
# alpha~0.035 -- the peak alpha decreases with M. The grids below bracket those
# generously. (Above the first-mode band the dominant root is the 2nd mode
# c_r->0.96-0.97; the first-mode band cap CR_HI keeps the scan on the 1st mode.)
ALPHA_GRID = {
    4.5: np.round(np.arange(0.020, 0.100 + 1e-9, 0.005), 5),
    5.8: np.round(np.arange(0.015, 0.080 + 1e-9, 0.004), 5),
    7.0: np.round(np.arange(0.012, 0.065 + 1e-9, 0.004), 5),
    # M10: the first-mode band peak is at alpha~0.030 (c_r~0.945); by alpha~0.040
    # the discrete band-mode has collapsed (the dominant root leaves the c_r band
    # and only the decaying continuous spectrum remains). Capping at 0.050 covers
    # the whole first-mode ridge without wasting solves on the dead high-alpha
    # region. (Every M10 solve is a uniform ~7.5 s; no per-solve QZ pathology.)
    10.0: np.round(np.arange(0.010, 0.050 + 1e-9, 0.0025), 5),
}
PSI_GRID = {
    4.5: np.arange(45.0, 66.0 + 1e-9, 3.0),
    5.8: np.arange(45.0, 66.0 + 1e-9, 3.0),
    7.0: np.arange(45.0, 66.0 + 1e-9, 3.0),
    10.0: np.arange(42.0, 63.0 + 1e-9, 3.0),
}

# Per-Mach wall-normal domain (y_max in L* units, ~4x delta*/L*) + resolution.
# delta*/L* = 10.34 / 15.70 / 21.15 / 37.05 at M = 4.5 / 5.8 / 7 / 10.
# M10 uses N=200, y_max=150: a dedicated convergence sweep (N 180/200/220/260 x
# y_max 120/150) showed omega_i,max stable to <4% -- N=200 is converged and far
# cheaper than N=220+ (the dense EVP cost grows steeply with N at this size).
Y_MAX_BY_MACH = {4.5: 42.0, 5.8: 64.0, 7.0: 86.0, 10.0: 150.0}
N_BY_MACH = {4.5: 130, 5.8: 150, 7.0: 170, 10.0: 200}


def _ymax_for(mach):
    return Y_MAX_BY_MACH[round(float(mach), 1)]


def _N_for(mach):
    return N_BY_MACH[round(float(mach), 1)]


def make_profile(mach: float):
    return pymack.make_mack_profile(float(mach), condition="table_11_1")


def first_mode_growth(profile, alpha, beta, R, mach, *, N, y_max,
                      cr_lo=CR_LO, cr_hi=CR_HI, ci_cap=CI_CAP):
    """Return (omega_i, c) for the discrete first mode at (alpha, beta, R).

    omega_i = alpha * c_i. Returns (None, None) if no eigenvalue falls in the
    first-mode band.

    FAST PATH: assembles the 3D temporal generalized EVP and computes the
    EIGENVALUES ONLY (scipy.linalg.eig right=False). The full
    solve_temporal_compressible_3d additionally computes the eigenVECTORS and
    the Appendix-B freestream-leakage score for every mode -- an O(n^2) loop
    over O(n) modes that, at the M10 domain (n~1100), costs ~190 s/solve vs
    ~8 s for values-only (measured). Growth-rate band selection needs only the
    eigenvalues c = omega/alpha, so the vectors + leakage filter are skipped.
    The basic physical filter (c_r in [-0.5,1.5], |c_i|<0.5) is replicated here,
    then the first-mode band (cr_lo..cr_hi, c_i<ci_cap) selects the mode with
    the largest c_i. This reproduces the leakage-filtered selection for the
    discrete first mode (verified against the full solver at anchor points).
    """
    A, B, y, D1, n, al, be, bf = assemble_temporal_compressible_3d_evp(
        profile, float(alpha), float(beta), float(R), float(mach), PR, GAMMA,
        N=N, y_max=y_max, length_scale="L_star", lambda_mu_ratio=0.0,
    )
    apply_wall_bc_3d(A, B, D1, n)
    apply_dirichlet_freestream_bc_3d(A, B, n)
    c = linalg.eig(A, B, right=False, check_finite=False)
    c = c[np.isfinite(c)]
    # basic physical filter (mirror _filter_temporal_modes_3d)
    phys = (c.real > -0.5) & (c.real < 1.5) & (np.abs(c.imag) < 0.5)
    c = c[phys]
    if c.size == 0:
        return None, None
    cr, ci = c.real, c.imag
    band = (cr > cr_lo) & (cr < cr_hi) & (ci < ci_cap)
    if not np.any(band):
        return None, None
    idx = np.where(band)[0]
    best = idx[int(np.argmax(ci[idx]))]
    return float(alpha) * float(ci[best]), c[best]


def first_mode_growth_full(profile, alpha, beta, R, mach, *, N, y_max,
                           cr_lo=CR_LO, cr_hi=CR_HI, ci_cap=CI_CAP):
    """Slow reference: full solver (eigenvectors + leakage filter) + band
    selection. Used only to cross-check the fast path at anchor points."""
    c, _modes, _y = solve_temporal_compressible_3d(
        profile, float(alpha), float(beta), float(R), float(mach), PR, GAMMA,
        N=N, y_max=y_max, length_scale="L_star", lambda_mu_ratio=0.0,
    )
    cr, ci = c.real, c.imag
    band = (cr > cr_lo) & (cr < cr_hi) & (ci < ci_cap)
    if not np.any(band):
        return None, None
    idx = np.where(band)[0]
    best = idx[int(np.argmax(ci[idx]))]
    return float(alpha) * float(ci[best]), c[best]


def maximize_growth(profile, R, mach, *, N, y_max, alpha_grid=None,
                    psi_grid=None, verbose=False):
    """Maximize first-mode omega_i over (alpha, psi) at this R.

    Coarse (alpha, psi) grid, then a fine alpha refine at the best psi (growth
    is broad in psi but sharp in alpha). Returns
    (omega_i_max, alpha_peak, psi_peak, c_peak) or (None,...) if the first mode
    could not be isolated anywhere on the grid.
    """
    if alpha_grid is None:
        alpha_grid = ALPHA_GRID[round(mach, 1)]
    if psi_grid is None:
        psi_grid = PSI_GRID[round(mach, 1)]

    best_oi, best_a, best_psi, best_c = -np.inf, None, None, None
    for psi in psi_grid:
        tan_psi = np.tan(np.radians(psi))
        for a in alpha_grid:
            beta = a * tan_psi
            oi, c = first_mode_growth(profile, a, beta, R, mach, N=N, y_max=y_max)
            if oi is not None and oi > best_oi:
                best_oi, best_a, best_psi, best_c = oi, float(a), float(psi), c
    if best_a is None:
        return None, None, None, None

    # Fine alpha refine at the best psi (+/- one coarse alpha step, 1/4 step).
    da = float(alpha_grid[1] - alpha_grid[0])
    tan_psi = np.tan(np.radians(best_psi))
    for a in np.arange(best_a - da, best_a + da + 1e-12, da / 4.0):
        if a <= 0:
            continue
        beta = a * tan_psi
        oi, c = first_mode_growth(profile, a, beta, R, mach, N=N, y_max=y_max)
        if oi is not None and oi > best_oi:
            best_oi, best_a, best_c = oi, float(a), c

    if verbose:
        print(f"   R={R:6.0f} alpha*={best_a:.4f} psi*={best_psi:.0f}deg "
              f"c_r={best_c.real:.4f} c_i={best_c.imag:+.5f} "
              f"omega_i,max={best_oi:+.5e}")
    return best_oi, best_a, best_psi, best_c


# --- Parallel path: one independent (mach, R) work unit per core -----------
_PROFILE_CACHE = {}


def _get_profile(mach):
    key = round(float(mach), 4)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = make_profile(key)
    return _PROFILE_CACHE[key]


def work_unit(args):
    """Module-level worker (picklable for ProcessPoolExecutor)."""
    mach, R, N, y_max = args
    profile = _get_profile(mach)
    oi, a, psi, c = maximize_growth(profile, R, mach, N=N, y_max=y_max)
    if oi is None:
        return {"mach": float(mach), "R": float(R), "omega_i_max": None,
                "alpha_peak": None, "psi_peak": None, "c_r": None, "c_i": None}
    return {"mach": float(mach), "R": float(R), "omega_i_max": float(oi),
            "alpha_peak": float(a), "psi_peak": float(psi),
            "c_r": float(c.real), "c_i": float(c.imag)}


def compute_curves_parallel(machs, *, max_workers=48, r_sweeps=None):
    """Compute omega_i,max(R) for several Mach numbers in parallel.

    One process pool, one (mach, R) unit per task (each unit single-threaded).
    Returns {round(mach,1): [row, ...]} sorted by R.
    """
    units = []
    for mach in machs:
        rl = (r_sweeps or DEFAULT_R_SWEEPS).get(round(float(mach), 1))
        if rl is None:
            raise ValueError(f"no default R sweep for M={mach}")
        mN, mY = _N_for(mach), _ymax_for(mach)
        for R in rl:
            units.append((float(mach), float(R), mN, mY))

    n_workers = max(1, min(max_workers, len(units)))
    print(f"parallel: {len(units)} (mach,R) units across {n_workers} workers "
          f"(of {os.cpu_count()} cores)", flush=True)

    by_mach = {}
    with _cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(work_unit, units):
            by_mach.setdefault(round(res["mach"], 1), []).append(res)
            tag = "ok" if res["omega_i_max"] else "--"
            print(f"  done M={res['mach']:.1f} R={res['R']:.0f} [{tag}]"
                  + (f" oi={res['omega_i_max']:.3e} psi={res['psi_peak']:.0f}"
                     if res["omega_i_max"] else ""), flush=True)
    for m in by_mach:
        by_mach[m].sort(key=lambda r: r["R"])
    return by_mach


def compute_curve(mach, r_list=None, *, N=None, y_max=None, verbose=True):
    """Sequential single-Mach curve (used for spot checks / convergence)."""
    mach = float(mach)
    if r_list is None:
        r_list = DEFAULT_R_SWEEPS[round(mach, 1)]
    if N is None:
        N = _N_for(mach)
    if y_max is None:
        y_max = _ymax_for(mach)
    profile = make_profile(mach)
    out = []
    if verbose:
        print(f"=== M={mach} N={N} y_max={y_max} condition=table_11_1 (3D first mode) ===")
    for R in r_list:
        oi, a, psi, c = maximize_growth(profile, R, mach, N=N, y_max=y_max,
                                        verbose=verbose)
        if oi is None:
            out.append({"R": float(R), "omega_i_max": None, "alpha_peak": None,
                        "psi_peak": None, "c_r": None, "c_i": None})
        else:
            out.append({"R": float(R), "omega_i_max": float(oi),
                        "alpha_peak": float(a), "psi_peak": float(psi),
                        "c_r": float(c.real), "c_i": float(c.imag)})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mach", type=float, default=None)
    p.add_argument("--r-list", type=float, nargs="+", default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--y-max", type=float, default=None)
    p.add_argument("--probe", type=float, nargs=2, default=None,
                   metavar=("MACH", "R"),
                   help="print omega_i,max at one (mach,R)")
    args = p.parse_args(argv)

    if args.probe is not None:
        m, R = args.probe
        prof = make_profile(m)
        ds = delta_star_over_lstar(prof)
        oi, a, psi, c = maximize_growth(prof, R, m, N=_N_for(m),
                                        y_max=_ymax_for(m), verbose=True)
        print(f"M={m} R={R}: delta*/L*={ds:.2f} omega_i,max*1e3={1e3*oi:.4f} "
              f"alpha={a:.4f} psi={psi:.0f} c_r={c.real:.4f}")
        return 0

    if args.mach is None:
        p.error("pass --mach or --probe")
    rows = compute_curve(args.mach, r_list=args.r_list, N=args.N,
                         y_max=args.y_max, verbose=True)
    print("\nR, omega_i_max, alpha_peak, psi_peak, c_r, c_i")
    for r in rows:
        if r["omega_i_max"] is None:
            print(f"{r['R']:.0f}, NONE")
        else:
            print(f"{r['R']:.0f}, {r['omega_i_max']:.6e}, {r['alpha_peak']:.4f}, "
                  f"{r['psi_peak']:.0f}, {r['c_r']:.4f}, {r['c_i']:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
