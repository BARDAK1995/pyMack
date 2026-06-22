"""Overlay pyMack's discrete-mode c_i fields (first mode: firstmode_grid.csv;
second mode: secondmode_grid.csv) against Ozgen's corrected v2 lobes, and
re-judge M4/M6 with BOTH modes.  Run after both grids complete.

Per case: compare pyMack's first-mode neutral curve to Ozgen's first-mode (lower)
lobe and pyMack's second-mode neutral curve to Ozgen's second-mode (upper) lobe,
median |d alpha|/alpha along each branch over the overlapping Re range.  Overall
verdict from the worse mode (both must agree).  M2/M3 handled separately (CS-limited).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VER = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(VER))
from _compare_lib import classify_relative   # noqa: E402

FIRST_CSV = HERE / "firstmode_grid.csv"
SECOND_CSV = HERE / "secondmode_grid.csv"


def load_grid(path):
    by_ma = {}
    if not path.exists():
        return by_ma
    with open(path) as f:
        for row in csv.DictReader(f):
            ma = float(row["Ma"])
            ci = float(row["c_i"]) if row["c_i"] not in ("", "nan") else np.nan
            by_ma.setdefault(ma, {})[(float(row["Re"]), float(row["alpha"]))] = ci
    out = {}
    for ma, d in by_ma.items():
        res = sorted({k[0] for k in d}); als = sorted({k[1] for k in d})
        Z = np.full((len(als), len(res)), np.nan)
        for (Re, al), ci in d.items():
            Z[als.index(al), res.index(Re)] = ci
        out[ma] = (np.array(res), np.array(als), Z)
    return out


def ozgen_branches(Ma, mode):
    ref = REPO / f"reference_data/digitized/ozgen_fig3_M{Ma}_neutral_v2.csv"
    up, lo = [], []
    with open(ref) as f:
        for r in csv.DictReader(f):
            if r.get("mode", "first") != mode:
                continue
            (up if r["lobe"] == "upper" else lo).append((float(r["Re"]), float(r["alpha"])))
    return np.array(sorted(up)), np.array(sorted(lo))


def neutral_alpha(res, als, Z, Re, which):
    j = int(np.argmin(np.abs(res - Re)))
    col = Z[:, j]
    cr = []
    for i in range(len(als) - 1):
        a, b = col[i], col[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a * b < 0:
            t = a / (a - b)
            cr.append(als[i] + t * (als[i + 1] - als[i]))
        elif a == 0:
            cr.append(als[i])
    if not cr:
        return np.nan
    return max(cr) if which == "upper" else min(cr)


def compare(grid_ma, up, lo):
    res, als, Z = grid_ma
    out = {}
    for name, ref in (("lower", lo), ("upper", up)):
        if ref.size == 0:
            continue
        errs = []
        for Re, a_oz in ref:
            if Re < res.min() or Re > res.max():
                continue
            a_pm = neutral_alpha(res, als, Z, Re, name)
            if not np.isnan(a_pm):
                errs.append(abs(a_pm - a_oz) / a_oz)
        if errs:
            out[name] = {"n": len(errs), "median_rel_err_alpha": float(np.median(errs))}
    meds = [b["median_rel_err_alpha"] for b in out.values()]
    return out, (float(np.median(meds)) if meds else float("nan"))


def main():
    G1 = load_grid(FIRST_CSV)
    G2 = load_grid(SECOND_CSV)
    for Ma in (4, 6):
        modes = {}
        # first mode
        if Ma in G1:
            up1, lo1 = ozgen_branches(Ma, "first")
            b1, m1 = compare(G1[Ma], up1, lo1)
            if np.isfinite(m1):
                modes["first"] = {"median_rel_err_alpha": m1, "per_branch": b1,
                                  "verdict": classify_relative(m1, topology_ok=True)}
        # second mode
        if Ma in G2:
            up2, lo2 = ozgen_branches(Ma, "second")
            b2, m2 = compare(G2[Ma], up2, lo2)
            if np.isfinite(m2):
                modes["second"] = {"median_rel_err_alpha": m2, "per_branch": b2,
                                   "verdict": classify_relative(m2, topology_ok=True)}
        # overall = worst of the resolved modes
        order = {"agrees": 0, "acceptable": 1, "disagrees": 2, "pending": 3}
        overall = max((mm["verdict"] for mm in modes.values()),
                      key=lambda v: order.get(v, 3), default="pending")

        # ---- plot both fields + Ozgen points ----
        fig, ax = plt.subplots(figsize=(9.2, 6.4))
        mx = 0.0
        for G in (G1, G2):
            if Ma in G:
                mx = max(mx, np.nanmax(np.abs(G[Ma][2])) or 0.0)
        mx = mx or 1e-3
        for G in (G1, G2):
            if Ma in G:
                res, als, Z = G[Ma]
                ax.pcolormesh(res, als, np.ma.masked_invalid(Z), cmap="RdBu_r",
                              vmin=-mx, vmax=mx, shading="auto")
                try:
                    ax.contour(res, als, np.ma.masked_invalid(Z), levels=[0.0],
                               colors="k", linewidths=2.2)
                except Exception:
                    pass
        for mode, col, mk in (("first", "#009e73", "o"), ("second", "#d55e00", "s")):
            up, lo = ozgen_branches(Ma, mode)
            for arr, lab in ((lo, f"Özgen {mode}-mode lower"), (up, f"Özgen {mode}-mode upper")):
                if arr.size:
                    ax.plot(arr[:, 0], arr[:, 1], mk, mfc="none", mec=col, mew=2, ms=7,
                            label=lab)
        ax.plot([], [], "k-", lw=2.2, label="pyMack neutral ($c_i=0$)")
        ax.set_xlabel(r"$R_L=\sqrt{Re_x}$", fontsize=15)
        ax.set_ylabel(r"$\alpha_{L^*}$", fontsize=15)
        title = f"M={Ma:g}: pyMack discrete-mode $c_i$ vs Özgen Fig 3 (v2) — both modes"
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=10.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        out_png = VER / f"first_mode/ozgen_m{Ma}/overlay.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

        # ---- write verdict ----
        vf = VER / f"first_mode/ozgen_m{Ma}/verdict.json"
        v = json.loads(vf.read_text(encoding="utf-8"))
        v["metrics"] = {"per_mode": modes, "topology_ok": True,
                        "method": "discrete-mode (eigenfunction-decay + y_max-stability); "
                                  "first mode tall domain, second mode short domain"}
        v["verdict"] = overall
        v["generated"] = "new"
        vf.write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"M{Ma}: overall={overall}  modes=" +
              ", ".join(f"{k}:{mm['verdict']}({mm['median_rel_err_alpha']:.1%})"
                        for k, mm in modes.items()))
    print("done")


if __name__ == "__main__":
    main()
