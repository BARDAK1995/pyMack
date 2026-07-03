"""Re-trace the M4.5 FIRST-mode neutral curve with the DISCRETE-mode extractor
(y_max-stationarity + freestream decay), replacing the phase-speed-band classifier
that mixed in continuous-spectrum modes (the old `trace_mazhong_curves.py`).

Result (writes pymack_firstmode_cutoff_discrete.csv): the discrete first mode gives
a CLEAN cutoff (upper) neutral branch that matches Ma & Zhong's Fig.15 upper branch
to ~3.5% median. The ONSET (lower) branch is NOT recovered: below omega ~ 0.03 the
2-D first mode delocalises into the slow-acoustic continuous spectrum (c_r -> below
~0.45, y_max-drifting), so no clean discrete eigenmode exists there in global
collocation. The same truncation occurs in the temporal solver and in QR shooting;
an oblique (psi) sweep confirms the 2-D mode is the MOST unstable, so this is a 2-D
first mode, not an oblique one. Recovering the onset branch would need a
compound-matrix / Riccati shooting solver robust to the poorly-separated near-CS
decay rates (see docs/work_in_progress.tex).

Method: at each R, scan omega; at each omega isolate the least-stable DISCRETE mode
(alpha domain-height-invariant between two y_max), record spatial growth -alpha_i,
locate the neutral crossing. Run: PYMACK_NO_BANNER=1 python trace_firstmode_discrete.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from compute_mazhong_m4p5 import build_profile, MA, PR, GAMMA, N, LAMBDA_MU, WALL_BC  # noqa: E402
from pymack.solver import solve_spatial  # noqa: E402

prof = build_profile()


def solve(omega, R, ymax, cg=0.6):
    return solve_spatial(prof, float(omega), float(R), MA, PR, GAMMA, N=N, y_max=ymax,
                         wall_bc=WALL_BC, target_alpha=omega/cg + 0j, n_modes=30,
                         length_scale="L_star", lambda_mu_ratio=LAMBDA_MU)


def fs_decay(vec_col, y):
    n = len(y); amp = np.zeros(n)
    for b in range(min(4, len(vec_col)//n)):
        amp += np.abs(vec_col[b*n:(b+1)*n])
    mx = amp.max(); ntop = max(2, n//10)
    return (amp[:ntop].max()/mx) if mx > 0 else np.nan


def discrete_first(omega, R, ym1=40.0, ym2=60.0, cband=(0.40, 0.78),
                   amax=0.03, match_tol=4e-3):
    """Least-stable DISCRETE (y_max-invariant) first mode at (R, omega), or None."""
    a1, m1, y1 = solve(omega, R, ym1)
    a2, _m2, _y2 = solve(omega, R, ym2)
    if a1.size == 0:
        return None
    c1 = omega / a1.real
    cand = np.where((c1 > cband[0]) & (c1 < cband[1]) & (a1.real > 0)
                    & (np.abs(a1.imag) < amax))[0]
    disc = [a1[k] for k in cand if np.min(np.abs(a1[k] - a2)) < match_tol]
    if not disc:
        return None
    return complex(min(disc, key=lambda a: a.imag))   # most negative alpha_i (least stable)


def main():
    Rs = [560, 700, 850, 1000, 1150, 1300, 1500]
    OMS = np.linspace(0.004, 0.130, 18)
    rows = []
    for R in Rs:
        ai = []
        for om in OMS:
            r = discrete_first(om, R)
            ai.append(r.imag if r is not None else np.nan)
        ai = np.array(ai)
        for i in range(len(OMS)-1):
            f0, f1 = ai[i], ai[i+1]
            if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 < 0:
                om = OMS[i] + (f0/(f0-f1))*(OMS[i+1]-OMS[i])
                rows.append((R, "lower" if f0 > 0 else "upper", om))
        print(f"  R={R}: {[(b, round(o,4)) for r_,b,o in rows if r_ == R]}", flush=True)
    with open(HERE / "pymack_firstmode_cutoff_discrete.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["R", "branch", "omega"])
        for R, b, om in rows:
            w.writerow([R, b, f"{om:.6f}"])
    print(f"wrote pymack_firstmode_cutoff_discrete.csv ({len(rows)} crossings)")


if __name__ == "__main__":
    main()
