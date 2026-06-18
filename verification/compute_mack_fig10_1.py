"""Compute pyMack's first-mode TEMPORAL neutral-stability frequency F(R) for
Mack (1984) AGARD R-709 Fig. 10.1 (2-D adiabatic flat plate, M=1.6 and M=2.2).

Fig. 10.1 plots the neutral-stability *frequency* of the 2-D first mode against
Reynolds number R = sqrt(Re_x) = U_e L*/nu_e, with L* = sqrt(nu_e x / U_e). The
y-axis is the dimensionless frequency F x 1e4, where

    F = omega_dim * nu_e / U_e**2 = omega_L / R,

and omega_L = alpha_L * c_r is the L*-scaled circular frequency of the wave.

Recipe (mirrors the validated compute_mack_fig10_6 engine):
  - Mean flow : pymack.make_mack_profile(M, condition='table_11_1') -- Mack's
                cold-edge adiabatic flat plate, viscosity_model='mack'.
  - Eigenvalue: solver.solve_temporal_compressible(profile, alpha, R, M, Pr,
                gamma, length_scale='L_star', lambda_mu_ratio=0.0). Temporal:
                fixed real alpha -> complex c; omega_i = alpha * c_i.
  - Mode      : the discrete FIRST mode (the more TS-like vorticity mode at low
                supersonic M). Selected by a phase-speed band c_r in
                [CR_LO, CR_HI] ~ [0.30, 0.80] and a tight |c_i| cap. Of the
                survivors the largest c_i is taken (most-unstable first-mode
                root at this alpha).
  - For each R, omega_i(alpha) traces an unstable LOBE (c_i>0 for an alpha
                band). The neutral boundary has two crossings in alpha: a lower
                (onset, - -> +) and an upper (cutoff, + -> -). Each crossing
                maps to a neutral frequency F = alpha_neutral * c_r / R. We
                also record the most-amplified frequency F at peak omega_i.

The wall-normal domain is generous (delta*/L* ~ 2.8 at M1.6, 3.7 at M2.2, so a
fixed y_max well above ~4x delta* is comfortable); y_max and N are swept for
convergence by the verification driver, not hard-wired here.

Single-thread BLAS is forced in-process (os.environ set BEFORE numpy import) so
the dense EVP sweeps do not oversubscribe cores; many (R, alpha) units run
concurrently in a process pool.
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

import concurrent.futures as _cf
import numpy as np

import pymack
from pymack.solver import solve_temporal_compressible

PR = 0.72
GAMMA = 1.4

# First-mode phase-speed band at low supersonic M. The 2-D first mode is the
# vorticity (TS-like) mode; at M=1.6/2.2 its c_r sits ~0.45-0.60 across the
# unstable band (probed: 0.49-0.56 at M1.6, 0.55-0.62 at M2.2). The window
# [0.40, 0.85] brackets that with margin while excluding BOTH the c_r->1
# acoustic/continuous spectrum AND a spurious low-c_r (~0.30) root that appears
# at high alpha / high R with a small positive c_i — that root corrupted an
# earlier 'max-c_i-in-band' selection at R>=1200 (the scan jumped to c_r~0.30).
CR_LO, CR_HI = 0.40, 0.85
# |c_i| cap: physical first-mode growth is |c_i| ~ O(1e-2). Reject spurious
# large-|c_i| roots.
CI_CAP = 0.10


def make_profile(mach: float):
    return pymack.make_mack_profile(float(mach), condition="table_11_1")


def first_mode_c(profile, alpha, R, mach, *, N, y_max,
                 cr_lo=CR_LO, cr_hi=CR_HI, ci_cap=CI_CAP):
    """Return the discrete first-mode complex c at (alpha, R), or None.

    Picks the largest-c_i eigenvalue inside the first-mode phase-speed band.
    """
    c, _modes, _y = solve_temporal_compressible(
        profile, float(alpha), float(R), float(mach), PR, GAMMA,
        N=N, y_max=y_max, length_scale="L_star", lambda_mu_ratio=0.0,
    )
    cr = c.real
    ci = c.imag
    band = (cr > cr_lo) & (cr < cr_hi) & (np.abs(ci) < ci_cap)
    if not np.any(band):
        return None
    idx = np.where(band)[0]
    best = idx[int(np.argmax(ci[idx]))]
    return c[best]


def scan_alpha(profile, R, mach, alphas, *, N, y_max):
    """Trace the first-mode (alpha, c_r, c_i, omega_i, F) over an alpha grid.

    Returns a structured array of finite rows (alpha, c_r, c_i, omega_i, F)
    where omega_i = alpha*c_i and F = alpha*c_r/R*1e4 (the y-axis of Fig 10.1).
    """
    rows = []
    for a in alphas:
        c = first_mode_c(profile, a, R, mach, N=N, y_max=y_max)
        if c is None:
            continue
        oi = float(a) * c.imag
        F = float(a) * c.real / R * 1e4
        rows.append((float(a), c.real, c.imag, oi, F))
    if not rows:
        return np.empty((0, 5))
    return np.array(rows)


def neutral_frequencies(scan):
    """From an alpha-scan array, locate the c_i=0 (omega_i=0) crossings.

    Returns dict with:
      'lower'  : F at the onset crossing (omega_i - -> + as alpha rises),
      'upper'  : F at the cutoff crossing (omega_i + -> - as alpha rises),
      'peak_F' : F at maximum omega_i,
      'peak_oi': maximum omega_i,
      'peak_alpha', 'peak_cr'.
    Branch entries are None when that crossing is not bracketed by the scan.
    """
    if scan.shape[0] < 2:
        return {"lower": None, "upper": None, "peak_F": None, "peak_oi": None,
                "peak_alpha": None, "peak_cr": None}
    a = scan[:, 0]
    oi = scan[:, 3]
    F = scan[:, 4]
    lower = upper = None
    for i in range(len(a) - 1):
        if oi[i] == 0.0:
            continue
        if oi[i] * oi[i + 1] < 0.0:
            t = (0.0 - oi[i]) / (oi[i + 1] - oi[i])
            Fc = F[i] + t * (F[i + 1] - F[i])
            if oi[i + 1] > oi[i]:      # - -> + : onset (lower-alpha) branch
                if lower is None:
                    lower = float(Fc)
            else:                       # + -> - : cutoff (upper-alpha) branch
                upper = float(Fc)
    imax = int(np.argmax(oi))
    return {
        "lower": lower,
        "upper": upper,
        "peak_F": float(F[imax]),
        "peak_oi": float(oi[imax]),
        "peak_alpha": float(a[imax]),
        "peak_cr": float(scan[imax, 1]),
    }


# --- Parallel work units ---------------------------------------------------
_PROFILE_CACHE = {}


def _get_profile(mach):
    key = round(float(mach), 4)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = make_profile(key)
    return _PROFILE_CACHE[key]


def work_unit(args):
    """args = (mach, R, alpha, N, y_max) -> (mach,R,alpha,c_r,c_i,oi,F) or None c."""
    mach, R, alpha, N, y_max = args
    profile = _get_profile(mach)
    c = first_mode_c(profile, alpha, R, mach, N=N, y_max=y_max)
    if c is None:
        return (float(mach), float(R), float(alpha), None, None, None, None)
    oi = float(alpha) * c.imag
    F = float(alpha) * c.real / R * 1e4
    return (float(mach), float(R), float(alpha), float(c.real), float(c.imag),
            float(oi), float(F))


def compute_grid_parallel(mach, r_list, alpha_list, *, N, y_max, max_workers=48):
    """Compute the (R, alpha) first-mode grid for one Mach in parallel.

    Returns dict[R] -> scan array (rows sorted by alpha), only finite-c rows.
    """
    units = [(float(mach), float(R), float(a), int(N), float(y_max))
             for R in r_list for a in alpha_list]
    n_workers = max(1, min(max_workers, len(units)))
    print(f"M={mach}: {len(units)} (R,alpha) units across {n_workers} workers "
          f"(of {os.cpu_count()} cores), N={N}, y_max={y_max}", flush=True)
    by_R = {float(R): [] for R in r_list}
    with _cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for (m, R, a, cr, ci, oi, F) in ex.map(work_unit, units):
            if cr is not None:
                by_R[R].append((a, cr, ci, oi, F))
    out = {}
    for R, rows in by_R.items():
        if rows:
            arr = np.array(rows)
            out[R] = arr[np.argsort(arr[:, 0])]
        else:
            out[R] = np.empty((0, 5))
    return out
