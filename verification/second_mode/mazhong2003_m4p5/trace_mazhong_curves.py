"""Trace pyMack's FULL first- and second-mode neutral curves (omega vs R) for the
Ma & Zhong (2003) Mach 4.5 flat plate, to overlay on Fig. 15.

Grid-scan spatial growth -alpha_i over (R, omega) for each mode (selected by
phase-speed band), write the grid; a separate step extracts the -alpha_i = 0
neutral contour.  Reuses the verified base flow + solve setup from
compute_mazhong_m4p5.py.  Resumable (appends per node, skips done).
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compute_mazhong_m4p5 import (build_profile, MA, PR, GAMMA, N, Y_MAX,
                                  LAMBDA_MU, WALL_BC)  # noqa: E402
from pymack.solver import solve_spatial  # noqa: E402

OUT = HERE / "mazhong_curve_grid.csv"

# (R grid, omega grid, c-band) per mode.  2nd mode: c~0.9; 1st mode: c~0.4-0.7.
MODES = {
    "second": {"R": np.linspace(240, 2000, 16), "omega": np.linspace(0.155, 0.235, 22),
               "c_lo": 0.80, "c_hi": 0.995},
    "first":  {"R": np.linspace(450, 2000, 16), "omega": np.linspace(0.010, 0.130, 22),
               "c_lo": 0.30, "c_hi": 0.76},
}


def growth(profile, R, omega, c_lo, c_hi):
    """-alpha_i of the most-unstable spatial mode with phase speed c in (c_lo,c_hi)."""
    c_guess = 0.90 if c_hi > 0.8 else 0.55
    try:
        alphas, _m, _y = solve_spatial(
            profile, float(omega), float(R), MA, PR, GAMMA,
            N=N, y_max=Y_MAX, wall_bc=WALL_BC,
            target_alpha=omega / c_guess + 0j, n_modes=25,
            length_scale="L_star", lambda_mu_ratio=LAMBDA_MU,
        )
    except Exception:
        return np.nan
    if alphas.size == 0:
        return np.nan
    c = omega / alphas.real
    m = (c > c_lo) & (c < c_hi) & (np.abs(alphas.imag) < 0.06) & (alphas.real > 0)
    cand = alphas[m]
    if cand.size == 0:
        return np.nan
    return float(-cand[np.argmin(cand.imag)].imag)


def done_keys():
    if not OUT.exists():
        return set()
    s = set()
    with open(OUT) as f:
        for row in csv.DictReader(f):
            s.add((row["mode"], f"{float(row['R']):.1f}", f"{float(row['omega']):.5f}"))
    return s


def main(which):
    prof = build_profile()
    seen = done_keys()
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["mode", "R", "omega", "neg_alpha_i"])
            f.flush()
        for mode in which:
            g = MODES[mode]
            t0 = time.perf_counter(); nd = 0
            for R in g["R"]:
                for om in g["omega"]:
                    key = (mode, f"{R:.1f}", f"{om:.5f}")
                    if key in seen:
                        continue
                    val = growth(prof, R, om, g["c_lo"], g["c_hi"])
                    w.writerow([mode, f"{R:.1f}", f"{om:.5f}",
                                "" if not np.isfinite(val) else f"{val:.6e}"])
                    f.flush(); nd += 1
                print(f"[{mode}] R={R:.0f} done ({nd} new, {time.perf_counter()-t0:.0f}s)", flush=True)
            print(f"[{mode}] complete: {nd} nodes in {time.perf_counter()-t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["second", "first"]
    main(which)
