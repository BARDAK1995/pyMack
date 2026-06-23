"""Extend the M7/M8/M10 grids to the low-R nose (R ~150-650) that Özgen's curves
reach but our grids (R-floor 600) missed. Appends to firstmode_grid.csv (tall
domain, 1st mode) and secondmode_grid.csv (short domain, 2nd mode). Resumable."""
from __future__ import annotations
import csv, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("PYMACK_NO_BANNER", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from discrete_mode import discrete_mode  # noqa: E402
from build_firstmode_grid import YMF_BY_MACH  # noqa: E402

FIRST = HERE / "firstmode_grid.csv"; SECOND = HERE / "secondmode_grid.csv"
RE_LOW = np.logspace(np.log10(150), np.log10(640), 9)
FIRST_A = {7: np.linspace(0.005, 0.12, 15), 8: np.linspace(0.005, 0.12, 15), 10: np.linspace(0.005, 0.12, 15)}
SECOND_A = {7: np.linspace(0.09, 0.21, 14), 8: np.linspace(0.09, 0.21, 14), 10: np.linspace(0.08, 0.22, 14)}


def done(path):
    if not path.exists():
        return set()
    s = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            s.add((r["Ma"], f"{float(r['Re']):.1f}", f"{float(r['alpha']):.5f}"))
    return s


def run(path, alphas_by_m, ymf_fn, cr_band, machs):
    seen = done(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for Ma in machs:
            ymf = ymf_fn(Ma); t0 = time.perf_counter(); nd = 0
            for R in RE_LOW:
                for al in alphas_by_m[Ma]:
                    key = (f"{Ma:g}", f"{R:.1f}", f"{al:.5f}")
                    if key in seen:
                        continue
                    kw = {"ymf_pair": ymf, "N": 200}
                    if cr_band:
                        kw["cr_band"] = cr_band; kw["N"] = 180
                    m = discrete_mode(float(Ma), float(R), float(al), **kw)
                    if m is None:
                        w.writerow([f"{Ma:g}", f"{R:.1f}", f"{al:.5f}", "", "", "", 0])
                    else:
                        w.writerow([f"{Ma:g}", f"{R:.1f}", f"{al:.5f}",
                                    f"{m['c_r']:.6f}", f"{m['c_i']:.6e}", f"{m['fs']:.4f}", 1])
                    f.flush(); nd += 1
                print(f"[{path.stem} M{Ma:g}] R={R:.0f} ({nd} new, {time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"DONE {path.stem}", flush=True)


if __name__ == "__main__":
    machs = [int(x) for x in (sys.argv[1:] or [7, 8, 10])]
    run(SECOND, SECOND_A, lambda m: (8.0, 12.0), (0.4, 0.99), machs)  # 2nd mode first (fast)
    run(FIRST, FIRST_A, lambda m: YMF_BY_MACH.get(m, (35., 45.)), None, machs)
    print("ALL DONE LOWR", flush=True)
