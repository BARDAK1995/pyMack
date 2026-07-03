"""Phase 2: extend the first-mode grids BELOW their alpha-floor (into the low-alpha
onset region Özgen covers, down to ~0.003), with tall domains + the discrete-mode
filter, to push onset coverage. Appends to firstmode_grid.csv (resumable). Where the
discrete mode no longer resolves (continuous-spectrum-limited), the node is recorded
empty -> honest coverage cap.

    python build_onset_extension.py 2 3 4 6 7 8 10
"""
from __future__ import annotations
import csv, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("PYMACK_NO_BANNER", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from discrete_mode import discrete_mode  # noqa: E402
from build_firstmode_grid import YMF_BY_MACH, OUT, GRID  # noqa: E402

N = 200
# low-alpha extension band per Mach (below the existing floor, covering Özgen's onset)
EXT = {
    2:  np.linspace(0.003, 0.010, 8),
    3:  np.linspace(0.003, 0.006, 6),
    4:  np.linspace(0.003, 0.006, 6),
    6:  np.linspace(0.004, 0.012, 9),
    7:  np.linspace(0.003, 0.005, 4),
    8:  np.linspace(0.003, 0.005, 4),
    10: np.linspace(0.003, 0.005, 4),
}


def done_keys():
    if not OUT.exists():
        return set()
    s = set()
    with open(OUT) as f:
        for r in csv.DictReader(f):
            s.add((r["Ma"], f"{float(r['Re']):.1f}", f"{float(r['alpha']):.5f}"))
    return s


def main(machs):
    seen = done_keys()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        for Ma in machs:
            res = GRID[Ma]["re"]; alphas = EXT[Ma]; ymf = YMF_BY_MACH.get(Ma, (35., 45.))
            t0 = time.perf_counter(); nd = 0
            for R in res:
                for al in alphas:
                    key = (f"{Ma:g}", f"{R:.1f}", f"{al:.5f}")
                    if key in seen:
                        continue
                    m = discrete_mode(float(Ma), float(R), float(al), N=N, ymf_pair=ymf)
                    if m is None:
                        w.writerow([f"{Ma:g}", f"{R:.1f}", f"{al:.5f}", "", "", "", 0])
                    else:
                        w.writerow([f"{Ma:g}", f"{R:.1f}", f"{al:.5f}",
                                    f"{m['c_r']:.6f}", f"{m['c_i']:.6e}", f"{m['fs']:.4f}", 1])
                    f.flush(); nd += 1
                print(f"[M{Ma:g} onset] R={R:.0f} ({nd} new, {time.perf_counter()-t0:.0f}s)", flush=True)
            print(f"[M{Ma:g} onset] done {nd} nodes {time.perf_counter()-t0:.0f}s", flush=True)
    print("ALL DONE ONSET", flush=True)


if __name__ == "__main__":
    machs = [int(x) for x in (sys.argv[1:] or list(EXT))]
    main(machs)
