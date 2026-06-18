"""Overlay pyMack's max first-mode omega_i(R) on the digitized Mack Fig 10.4
panels for M=4.5/5.8/7/10. Reads each case's pymack_curve.json and the digitized
CSV, writes one overlay PNG per Mach into the case folder.

Run AFTER verify_mack_fig10_4.py has produced pymack_curve.json files.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GROWTH = HERE / "growthRate_verification"
PANELS = {4.5: "45", 5.8: "58", 7.0: "70", 10.0: "100"}

plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "axes.titlesize": 16,
                     "legend.fontsize": 12})


def load_digitized(suffix):
    p = REPO / "reference_data" / "digitized" / f"mack_ch10_fig10_4_M{suffix}_paper.csv"
    R, y = [], []
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            R.append(float(row["x"])); y.append(float(row["y"]))  # y = oi*1e3
    o = np.argsort(R)
    return np.array(R)[o], np.array(y)[o]


def overlay(mach):
    suffix = PANELS[mach]
    case = GROWTH / f"mack_fig10_4_M{suffix}"
    cj = case / "pymack_curve.json"
    if not cj.exists():
        print(f"[M{suffix}] no pymack_curve.json; skip"); return None
    rows = json.loads(cj.read_text(encoding="utf-8"))
    tR = np.array([r["R"] for r in rows], float)
    tY = np.array([1e3*r["omega_i_max"] if r["omega_i_max"] else np.nan for r in rows], float)
    dR, dY = load_digitized(suffix)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dR, dY, "k-o", ms=4, lw=1.6, label="Mack (1984) Fig 10.4 (digitized)")
    m = np.isfinite(tY)
    ax.plot(tR[m], tY[m], "r--s", ms=5, lw=1.8, label="pyMack (3D first mode)")
    ax.set_xlabel("R")
    ax.set_ylabel(r"$\omega_i \times 10^3$ (max over $\alpha,\psi$)")
    ax.set_title(f"Mack Fig 10.4  M={mach}  (adiabatic, first mode)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = case / "overlay.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[M{suffix}] wrote {out}")
    return str(out.relative_to(REPO)).replace("\\", "/")


def main(argv=None):
    machs = [float(a) for a in (argv or sys.argv[1:])] or list(PANELS)
    for m in machs:
        overlay(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
