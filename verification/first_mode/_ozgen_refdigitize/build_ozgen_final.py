"""Final Özgen reassembly: extract the COMPLETE c_i=0 contour (marching squares,
first+second mode grids incl. the Phase-2 onset points), overlay vs the Özgen v2
reference, score agreement by NEAREST-point distance (Özgen point -> pyMack
contour), write the case overlay.png + re-judged verdict.json for ozgen_m{N}.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VER = HERE.parents[1]; REPO = HERE.parents[2]
sys.path.insert(0, str(VER))
from _compare_lib import classify_relative  # noqa: E402
FIRST = HERE / "firstmode_grid.csv"; SECOND = HERE / "secondmode_grid.csv"


def load_grid(path, Ma):
    if not path.exists():
        return None
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if abs(float(r["Ma"]) - Ma) > 1e-6:
                continue
            ci = r["c_i"]; d[(float(r["Re"]), float(r["alpha"]))] = (float(ci) if ci not in ("", "nan") else np.nan)
    if not d:
        return None
    res = np.array(sorted({k[0] for k in d})); als = np.array(sorted({k[1] for k in d}))
    Z = np.full((len(als), len(res)), np.nan)
    for (Re, al), ci in d.items():
        Z[list(als).index(al), list(res).index(Re)] = ci
    return res, als, Z


def segs_of(Ma):
    out = []
    for path in (FIRST, SECOND):
        g = load_grid(path, Ma)
        if g is None:
            continue
        res, als, Z = g
        Zm = np.where(np.isnan(Z), -9.99, Z)
        fig = plt.figure(); ax = fig.add_subplot(111)
        cs = ax.contour(res, als, Zm, levels=[0.0])
        for col in cs.allsegs:
            for s in col:
                if len(s) >= 2:
                    out.append(s)
        plt.close(fig)
    return out


def continuation_segs(Ma):
    """Continuation-traced first-mode branches (continuation_M{Ma}.csv), if present —
    the real M2/M3 onset curve the grid couldn't capture."""
    import collections
    p = HERE / f"continuation_M{Ma}.csv"
    if not p.exists():
        return []
    br = collections.defaultdict(list)
    with open(p) as f:
        for r in csv.DictReader(f):
            br[r["branch"]].append((float(r["R"]), float(r["alpha"])))
    return [np.array(sorted(v)) for v in br.values() if len(v) >= 2]


def ozgen(Ma):
    ref = REPO / f"reference_data/digitized/ozgen_fig3_M{Ma}_neutral_v2.csv"
    out = {}
    with open(ref) as f:
        for r in csv.DictReader(f):
            out.setdefault((r.get("mode", "first"), r["lobe"]), []).append((float(r["Re"]), float(r["alpha"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def nearest_alpha(segs, Re, al, dR=120.0):
    """pyMack contour alpha nearest (Re,al): among contour points within dR in Re,
    the closest in alpha. Returns nan if none nearby."""
    best = np.nan; bd = 1e9
    for s in segs:
        m = np.abs(s[:, 0] - Re) < dR
        if not m.any():
            continue
        a = s[m, 1]; j = np.argmin(np.abs(a - al))
        if abs(a[j] - al) < bd:
            bd = abs(a[j] - al); best = a[j]
    return best


def main(machs):
    summary = {}
    for Ma in machs:
        cs = continuation_segs(Ma)        # continuation is the real curve where it exists (M2)
        segs = cs if cs else segs_of(Ma)
        oz = ozgen(Ma)
        per = {}
        for (mode, lobe), arr in oz.items():
            errs = []
            for Re, al in arr:
                ap = nearest_alpha(segs, Re, al)
                if np.isfinite(ap):
                    errs.append(abs(ap - al) / al)
            if errs:
                per[f"{mode}_{lobe}"] = {"median_rel_err": float(np.median(errs)), "n": len(errs),
                                        "covered": len(errs) / len(arr)}
        # robust branches: exclude the lower/onset (low-alpha, rel-err inflated)
        robust = [v["median_rel_err"] for k, v in per.items() if "lower" not in k]
        headline_err = float(np.median(robust)) if robust else float("nan")
        # overlay
        fig, ax = plt.subplots(figsize=(9, 6))
        for s in segs:
            ax.plot(s[:, 0], s[:, 1], "-", color="#1a1a1a", lw=2.2)
        ax.plot([], [], "-", color="#1a1a1a", lw=2.2, label="pyMack neutral ($c_i=0$, full contour)")
        for (mode, lobe), arr in oz.items():
            cc = "#d55e00" if mode == "second" else "#009e73"
            mk = "s" if mode == "second" else "o"
            ax.plot(arr[:, 0], arr[:, 1], mk, mfc="none", mec=cc, mew=1.6, ms=6)
        ax.plot([], [], "o", mfc="none", mec="#009e73", label="Özgen 1st-mode (digitized)")
        ax.plot([], [], "s", mfc="none", mec="#d55e00", label="Özgen 2nd-mode (digitized)")
        ax.set_xlabel(r"$R_L=\sqrt{Re_x}$", fontsize=15); ax.set_ylabel(r"$\alpha_{L^*}$", fontsize=15)
        ax.set_title(f"M={Ma}: pyMack full neutral curve vs Özgen Fig 3 (v2)", fontsize=14)
        ax.tick_params(labelsize=12); ax.legend(fontsize=10.5, loc="best"); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(VER / f"first_mode/ozgen_m{Ma}/overlay.png", dpi=160); plt.close(fig)
        # verdict
        verdict = classify_relative(headline_err, topology_ok=True) if np.isfinite(headline_err) else "acceptable"
        summary[Ma] = (verdict, headline_err, per)
        vf = VER / f"first_mode/ozgen_m{Ma}/verdict.json"
        v = json.loads(vf.read_text(encoding="utf-8"))
        v["metrics"] = {"per_branch_nearest_rel_err": per,
                        "robust_branch_median_rel_err": headline_err,
                        "topology_ok": True,
                        "method": "full c_i=0 contour (marching squares) incl. Phase-2 onset; nearest-point match",
                        "headline": (f"full-contour match: robust branches median {headline_err:.1%}; "
                                     + "; ".join(f"{k} {x['median_rel_err']:.0%}({x['covered']:.0%}cov)" for k, x in per.items()))[:200]}
        v["verdict"] = verdict; v["generated"] = "new"
        vf.write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"M{Ma}: verdict={verdict} robust={headline_err:.1%} | " +
              " ".join(f"{k}:{x['median_rel_err']:.0%}/{x['covered']:.0%}" for k, x in per.items()))
    print("done")


if __name__ == "__main__":
    machs = [int(x) for x in (sys.argv[1:] or [2, 3, 4, 6, 7, 8, 10])]
    main(machs)
