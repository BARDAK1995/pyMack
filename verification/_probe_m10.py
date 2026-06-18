"""Focused probe: can pyMack isolate a GROWING M=10 second mode in Mack Fig 10.6
if we widen the wall-normal domain? At M10 delta*/L* ~ 37, so the default
y_max=30 (L* units) does not even reach the displacement thickness -- the mode
cannot form. Try larger y_max (+ matching N) at a representative R.
"""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_mack_fig10_6 as e

MACH = 10.0
R = 1500.0
profile = e.make_profile(MACH)
print(f"M={MACH} profile delta*/L* = {getattr(profile, '_delta_star', float('nan')):.2f}")

# Wide alpha scan (M10 peak alpha is low); test several domains/resolutions.
for y_max, N in [(40, 140), (80, 160), (120, 200), (160, 220)]:
    best_oi, best_a, best_c = -np.inf, None, None
    for a in np.arange(0.02, 0.22, 0.01):
        c, oi = e.second_mode_growth(profile, a, R, MACH, N=N, y_max=y_max,
                                     cr_lo=0.80, cr_hi=0.995, ci_cap=0.05)
        if oi is not None and oi > best_oi:
            best_oi, best_a, best_c = oi, float(a), c
    if best_a is None:
        print(f"  y_max={y_max:3.0f} N={N}: no mode in band")
    else:
        sign = "GROWING" if best_oi > 0 else "decaying"
        print(f"  y_max={y_max:3.0f} N={N}: omega_i,max={best_oi:+.4e} ({sign}) "
              f"at alpha={best_a:.3f}, c={best_c.real:.4f}{best_c.imag:+.5f}j")
