"""Compute pyMack's FIRST-MODE c_i field on the L* (Re_L, alpha_L) plane using
the eigenfunction-decay discrete-mode extractor (discrete_mode.py), for the
Mach numbers where the first mode is cleanly resolvable (M4, M6).  The c_i=0
contour of this field is pyMack's first-mode neutral curve, to be overlaid on
Ozgen's digitized lower lobe.

Resumable: appends one row per (Ma, Re, alpha) node to the output CSV and skips
nodes already present.  Run detached; safe to re-launch.

    python build_firstmode_grid.py 4 6
"""
from __future__ import annotations
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYMACK_NO_BANNER", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from discrete_mode import discrete_mode  # noqa: E402

OUT = HERE / "firstmode_grid.csv"

# Per-Mach grid covering Ozgen's first-mode lobe (low-alpha) + a margin.
GRID = {
    4: {"re": np.logspace(np.log10(1200), np.log10(5500), 13),
        "alpha": np.linspace(0.006, 0.11, 15)},
    6: {"re": np.logspace(np.log10(900), np.log10(5500), 13),
        "alpha": np.linspace(0.012, 0.13, 15)},
    2: {"re": np.logspace(np.log10(350), np.log10(5500), 13),
        "alpha": np.linspace(0.010, 0.075, 15)},
    3: {"re": np.logspace(np.log10(650), np.log10(5500), 13),
        "alpha": np.linspace(0.006, 0.055, 15)},
    7: {"re": np.logspace(np.log10(600), np.log10(5500), 13),
        "alpha": np.linspace(0.005, 0.12, 15)},
    8: {"re": np.logspace(np.log10(600), np.log10(5500), 13),
        "alpha": np.linspace(0.005, 0.12, 15)},
    10: {"re": np.logspace(np.log10(600), np.log10(5500), 13),
         "alpha": np.linspace(0.005, 0.12, 15)},
}

N = 200
# y_max as (short, tall) multiples of delta*/L*, set so the ABSOLUTE domain is
# ~450-600 L* (delta*/L* grows with Mach, so the multiple shrinks).
YMF_BY_MACH = {2: (35.0, 45.0), 3: (35.0, 45.0), 4: (35.0, 45.0), 6: (35.0, 45.0),
               7: (28.0, 37.0), 8: (23.0, 31.0), 10: (17.0, 22.0)}
YMF = (35.0, 45.0)  # fallback


def done_nodes():
    if not OUT.exists():
        return set()
    seen = set()
    with open(OUT) as f:
        for row in csv.DictReader(f):
            seen.add((row["Ma"], f"{float(row['Re']):.1f}", f"{float(row['alpha']):.5f}"))
    return seen


def main(machs):
    seen = done_nodes()
    new_file = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["Ma", "Re", "alpha", "c_r", "c_i", "fs", "resolved"])
            f.flush()
        for Ma in machs:
            g = GRID[Ma]
            t0 = time.perf_counter()
            ndone = 0
            for Re in g["re"]:
                for al in g["alpha"]:
                    key = (f"{Ma:g}", f"{Re:.1f}", f"{al:.5f}")
                    if key in seen:
                        continue
                    m = discrete_mode(float(Ma), float(Re), float(al), N=N,
                                      ymf_pair=YMF_BY_MACH.get(Ma, YMF))
                    if m is None:
                        w.writerow([f"{Ma:g}", f"{Re:.1f}", f"{al:.5f}", "", "", "", 0])
                    else:
                        w.writerow([f"{Ma:g}", f"{Re:.1f}", f"{al:.5f}",
                                    f"{m['c_r']:.6f}", f"{m['c_i']:.6e}", f"{m['fs']:.4f}", 1])
                    f.flush()
                    ndone += 1
                print(f"[M={Ma:g}] Re={Re:.0f} done ({ndone} new nodes, "
                      f"{time.perf_counter()-t0:.0f}s)", flush=True)
            print(f"[M={Ma:g}] complete: {ndone} new nodes in "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    machs = [int(x) for x in (sys.argv[1:] or ["4", "6"])]
    main(machs)
