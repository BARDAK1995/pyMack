"""Compute pyMack's SECOND-mode (Mack) c_i field for M4 and M6 over the high-alpha
range Ozgen's Fig-3 upper lobe occupies (M4 ~0.33, M6 ~0.18).  The second mode is
short-wavelength and wall-trapped, so it needs only a SHORT domain (y_max~8-12 d*)
and is cleanly discrete (eigenfunction decays).  Uses the same discrete-mode
extractor.  Resumable; writes secondmode_grid.csv.

    python build_secondmode_grid.py 4 6
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

OUT = HERE / "secondmode_grid.csv"

GRID = {
    4: {"re": np.logspace(np.log10(1130), np.log10(5500), 13),
        "alpha": np.linspace(0.27, 0.40, 14)},
    6: {"re": np.logspace(np.log10(600), np.log10(5500), 13),
        "alpha": np.linspace(0.12, 0.22, 14)},
    7: {"re": np.logspace(np.log10(600), np.log10(5500), 13),
        "alpha": np.linspace(0.09, 0.21, 14)},
    8: {"re": np.logspace(np.log10(600), np.log10(5500), 13),
        "alpha": np.linspace(0.09, 0.21, 14)},
    10: {"re": np.logspace(np.log10(500), np.log10(5500), 13),
         "alpha": np.linspace(0.08, 0.22, 14)},
}
N = 180
YMF = (8.0, 12.0)            # short domain: 2nd mode is wall-trapped
CR_BAND = (0.4, 0.99)        # Mack-mode phase speed band (excludes free-stream c~1)


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
            nd = 0
            for Re in g["re"]:
                for al in g["alpha"]:
                    key = (f"{Ma:g}", f"{Re:.1f}", f"{al:.5f}")
                    if key in seen:
                        continue
                    m = discrete_mode(float(Ma), float(Re), float(al),
                                      N=N, ymf_pair=YMF, cr_band=CR_BAND)
                    if m is None:
                        w.writerow([f"{Ma:g}", f"{Re:.1f}", f"{al:.5f}", "", "", "", 0])
                    else:
                        w.writerow([f"{Ma:g}", f"{Re:.1f}", f"{al:.5f}",
                                    f"{m['c_r']:.6f}", f"{m['c_i']:.6e}", f"{m['fs']:.4f}", 1])
                    f.flush()
                    nd += 1
                print(f"[M={Ma:g} 2nd] Re={Re:.0f} ({nd} new, {time.perf_counter()-t0:.0f}s)", flush=True)
            print(f"[M={Ma:g} 2nd] complete: {nd} new in {time.perf_counter()-t0:.0f}s", flush=True)
    print("ALL DONE 2ND", flush=True)


if __name__ == "__main__":
    machs = [int(x) for x in (sys.argv[1:] or ["4", "6"])]
    main(machs)
