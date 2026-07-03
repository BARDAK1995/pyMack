"""Phase 1: extract the COMPLETE c_i=0 neutral contour from the existing grids
(marching squares via matplotlib), instead of the crude min/max-alpha crossing.
Gives every branch of every lobe over the full grid -- free, no new compute.

Overlays pyMack's full c_i=0 contour vs the Ozgen v2 reference and reports the
(Re, alpha) extent covered vs Ozgen's, so we can see exactly what (if anything)
still needs Phase-2 compute.

    python extract_contours.py 6        # one Mach
    python extract_contours.py 2 3 4 6 7 8 10
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VER = HERE.parents[1]
REPO = HERE.parents[2]
FIRST = HERE / "firstmode_grid.csv"
SECOND = HERE / "secondmode_grid.csv"


def load_grid(path, Ma):
    if not path.exists():
        return None
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if abs(float(r["Ma"]) - Ma) > 1e-6:
                continue
            ci = r["c_i"]
            d[(float(r["Re"]), float(r["alpha"]))] = (float(ci) if ci not in ("", "nan") else np.nan)
    if not d:
        return None
    res = np.array(sorted({k[0] for k in d}))
    als = np.array(sorted({k[1] for k in d}))
    Z = np.full((len(als), len(res)), np.nan)
    for (Re, al), ci in d.items():
        Z[list(als).index(al), list(res).index(Re)] = ci
    return res, als, Z


def contour_segments(res, als, Z):
    """All c_i=0 polyline segments via matplotlib contour (marching squares)."""
    Zm = np.where(np.isnan(Z), -9.99, Z)   # treat unresolved as 'stable' sentinel
    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(res, als, Zm, levels=[0.0])
    segs = []
    for col in cs.allsegs:
        for s in col:
            if len(s) >= 2:
                segs.append(s)
    plt.close(fig)
    return segs


def ozgen_branches(Ma):
    ref = REPO / f"reference_data/digitized/ozgen_fig3_M{Ma}_neutral_v2.csv"
    out = {}
    with open(ref) as f:
        for r in csv.DictReader(f):
            out.setdefault((r.get("mode", "first"), r["lobe"]), []).append((float(r["Re"]), float(r["alpha"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def main(machs):
    for Ma in machs:
        oz = ozgen_branches(Ma)
        fig, ax = plt.subplots(figsize=(9, 6))
        # pyMack contours (both grids)
        cov = []
        for path, col, lab in ((FIRST, "#0072b2", "pyMack 1st-mode $c_i=0$"),
                               (SECOND, "#d55e00", "pyMack 2nd-mode $c_i=0$")):
            g = load_grid(path, Ma)
            if g is None:
                continue
            res, als, Z = g
            for s in contour_segments(res, als, Z):
                ax.plot(s[:, 0], s[:, 1], "-", color=col, lw=2.4)
                cov.append((col, s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max()))
            ax.plot([], [], "-", color=col, lw=2.4, label=lab)
        # Ozgen points
        for (mode, lobe), arr in oz.items():
            mk = "s" if mode == "second" else "o"
            cc = "#d55e00" if mode == "second" else "#009e73"
            ax.plot(arr[:, 0], arr[:, 1], mk, mfc="none", mec=cc, mew=1.6, ms=6)
        ax.plot([], [], "o", mfc="none", mec="#009e73", label="Özgen v2 (digitized)")
        ax.set_xlabel(r"$R_L=\sqrt{Re_x}$", fontsize=14); ax.set_ylabel(r"$\alpha_{L^*}$", fontsize=14)
        ax.set_title(f"M={Ma}: pyMack full c_i=0 contour (marching squares) vs Özgen", fontsize=13)
        ax.tick_params(labelsize=11); ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
        out = HERE / f"_contour_M{Ma}.png"
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
        # coverage report vs Ozgen extent
        oz_re = np.concatenate([a[:, 0] for a in oz.values()]) if oz else np.array([0])
        oz_al = np.concatenate([a[:, 1] for a in oz.values()]) if oz else np.array([0])
        print(f"M{Ma}: Özgen Re[{oz_re.min():.0f},{oz_re.max():.0f}] alpha[{oz_al.min():.3f},{oz_al.max():.3f}]"
              f"  | wrote {out.name}")


if __name__ == "__main__":
    machs = [int(x) for x in (sys.argv[1:] or ["6"])]
    main(machs)
