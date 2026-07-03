"""Eigenvalue-continuation tracer for the Özgen temporal first mode, to push the
neutral curve past the continuous-spectrum wall where the band-filter fails (M2,
M3). Lock onto the discrete mode's complex phase speed c at a clean seed, then
march in R, at each step following the eigenvalue NEAREST the tracked c (not a
phase-speed band) and root-finding the neutral alpha (c_i=0). Stops honestly when
the discrete mode can no longer be tracked (merged into the CS).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]; sys.path.insert(0, str(REPO))
from pymack import make_flatplate_profile  # noqa: E402
from pymack.temporal_solver import solve_temporal_2d  # noqa: E402
from pymack.scales import delta_star_over_lstar  # noqa: E402

_P = {}
def prof(Ma):
    if Ma not in _P:
        _P[Ma] = make_flatplate_profile(float(Ma))
    return _P[Ma]


def eigs_c(Ma, R, alpha, ymf, N):
    p = prof(Ma); d = delta_star_over_lstar(p)
    ev, _v, _y = solve_temporal_2d(p, float(alpha), float(R), float(Ma),
                                         N=N, y_max=ymf * d, length_scale="L_star")
    return ev


def tracked(Ma, R, alpha, c_prev, ymf, N, tol):
    """c of the eigenvalue nearest c_prev (the tracked discrete mode), or None."""
    ev = eigs_c(Ma, R, alpha, ymf, N)
    if ev.size == 0:
        return None
    j = int(np.argmin(np.abs(ev - c_prev)))
    return ev[j] if abs(ev[j] - c_prev) < tol else None


def neutral_alpha(Ma, R, a_guess, c_prev, ymf, N, tol=0.06, da=0.002, itmax=18):
    """alpha where tracked-mode c_i=0 near a_guess (secant), tracking c. Returns
    (alpha, c) or (None, None)."""
    c0 = tracked(Ma, R, a_guess, c_prev, ymf, N, tol)
    if c0 is None:
        return None, None
    a0, f0, cc = a_guess, c0.imag, c0
    a1 = a_guess + (da if f0 > 0 else -da)   # step toward neutral
    c1 = tracked(Ma, R, a1, cc, ymf, N, tol)
    if c1 is None:
        return None, None
    f1 = c1.imag
    for _ in range(itmax):
        if abs(f1) < 2e-5:
            return a1, c1
        if f1 == f0:
            break
        a2 = a1 - f1 * (a1 - a0) / (f1 - f0)
        a2 = float(np.clip(a2, 0.001, 0.30))
        c2 = tracked(Ma, R, a2, c1, ymf, N, tol)
        if c2 is None:
            return None, None
        a0, f0 = a1, f1
        a1, f1, cc = a2, c2.imag, c2
    return (a1, cc) if abs(f1) < 2e-4 else (None, None)


def trace(Ma, R_seed, a_seed, c_seed, ymf, N, R_min, R_max, dR=80.0):
    """March R in both directions from the seed, tracing the neutral alpha."""
    out = {}
    for direction in (+1, -1):
        R, a, c = R_seed, a_seed, c_seed
        steps = int((R_max - R_seed) / dR) if direction > 0 else int((R_seed - R_min) / dR)
        for _ in range(steps):
            Rn = R + direction * dR
            if not (R_min <= Rn <= R_max):
                break
            an, cn = neutral_alpha(Ma, Rn, a, c, ymf, N)
            if an is None:
                break
            out[round(Rn, 1)] = an
            R, a, c = Rn, an, cn
    out[round(R_seed, 1)] = a_seed
    return dict(sorted(out.items()))


def seed_branches(Ma, R_seed, c_hint, ymf, N, a_scan):
    """At R_seed, march alpha tracking the discrete mode (c~c_hint); return the
    neutral crossings (alpha, c) found = branch seeds."""
    c_prev = None
    prev = None
    crossings = []
    for a in a_scan:
        ev = eigs_c(Ma, R_seed, a, ymf, N)
        if ev.size == 0:
            prev = None; continue
        if c_prev is None:
            # lock onto the discrete mode near c_hint, decaying
            cand = ev[np.abs(ev - c_hint) < 0.12]
            if cand.size == 0:
                prev = None; continue
            c = cand[np.argmax(cand.imag)]
        else:
            j = int(np.argmin(np.abs(ev - c_prev)))
            if abs(ev[j] - c_prev) > 0.06:
                prev = None; c_prev = None; continue
            c = ev[j]
        if prev is not None and prev[1] * c.imag < 0:   # sign change in c_i
            frac = prev[1] / (prev[1] - c.imag)
            a_cross = prev[0] + frac * (a - prev[0])
            crossings.append((a_cross, c))
        prev = (a, c.imag); c_prev = c
    return crossings


import csv
# per-Mach seed: (R_seed, c_hint, alpha_scan_lo, alpha_scan_hi)
SEED = {2: (520.0, 0.56, 0.02, 0.085), 3: (820.0, 0.55, 0.004, 0.032)}

if __name__ == "__main__":
    Ma = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    YMF, N = 40.0, 220   # single tall-domain multiple (continuation tracks by eigenvalue proximity)
    Rs0, chint, alo, ahi = SEED[int(Ma)]
    d = delta_star_over_lstar(prof(Ma))
    print(f"M{Ma:g}: delta*/L*={d:.2f}, seeding at R={Rs0:.0f}", flush=True)
    cr = seed_branches(Ma, Rs0, chint, YMF, N, np.linspace(alo, ahi, 28))
    print(f"  seed crossings @R={Rs0:.0f}: {[(round(a,4), round(c.real,3)) for a,c in cr]}", flush=True)
    rows = []
    for k, (a0, c0) in enumerate(cr):
        br = trace(Ma, Rs0, a0, c0, YMF, N, R_min=300, R_max=5000, dR=120)
        Rlist = sorted(br)
        label = "lower" if (cr and a0 == min(c[0] for c in cr)) else "upper"
        for R in Rlist:
            rows.append((label, R, br[R]))
        if Rlist:
            print(f"  branch {label} from ({Rs0:.0f},{a0:.4f}) c_r={c0.real:.3f}: traced R[{Rlist[0]:.0f},{Rlist[-1]:.0f}], "
                  f"{len(Rlist)} pts, alpha[{min(br.values()):.4f},{max(br.values()):.4f}]", flush=True)
    out = HERE / f"continuation_M{int(Ma)}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["branch", "R", "alpha"])
        for lab, R, a in sorted(rows, key=lambda r: (r[0], r[1])):
            w.writerow([lab, f"{R:.1f}", f"{a:.6f}"])
    print(f"wrote {out.name} ({len(rows)} pts)", flush=True)
    print("CONT DONE", flush=True)
