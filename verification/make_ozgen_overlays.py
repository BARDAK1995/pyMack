#!/usr/bin/env python
"""Canonical Özgen & Kırcalı (2008) Fig. 3 neutral-curve overlay generator.

This is the ONE producer of the per-case overlay.png for every Özgen Fig. 3
neutral-curve case:

    verification/mixed_mode/ozgen_fig3/M{2,3,4,6,7,8,10}/overlay.png

For each Mach it plots:
  * Our LST code's OWN computed neutral locus -- the full c_i = 0 contour extracted (by
    marching squares) from the committed first-/second-mode c_i grids
    (`_refdigitize/firstmode_grid.csv`, `secondmode_grid.csv`); where an
    eigenvalue-continuation trace exists (`continuation_M{N}.csv`, M2/M3) that
    is used for the first-mode branches instead, exactly as the finalize pipeline
    does. NO physics is recomputed here -- only committed grids are read.
  * The CORRECTED multi-branch digitized reference POINTS from
    `reference_data/digitized/ozgen_fig3_M{N}_neutral_v2.csv` (columns
    lobe,Re,alpha,mode), styled distinctly per (mode, lobe): first vs second
    mode, upper vs lower lobe -- mirroring the branch structure of the
    `_refdigitize/_verify2_M*.png` proof figures but WITHOUT the scanned paper
    panel (points + curve only -> copyright-clean).

The title reads the verdict word + headline metric LIVE from that case's
verdict.json (never hardcoded); verdict.json numbers are NOT modified (only the
artifacts.overlay path is refreshed to the case's real overlay.png).

Usage:
    python verification/make_ozgen_overlays.py            # all Machs
    python verification/make_ozgen_overlays.py 4 6        # a subset
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- HARD style rule (workspace): labels >=14, ticks >=12, title >=16 --------
# These are embedded in the LaTeX writeup at width=0.75\linewidth, so the saved
# image is downscaled ~0.55x on the page. Fonts are therefore set well ABOVE the
# minimums so the rendered text stays publication-legible after that downscale.
plt.rcParams.update({
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.titlesize": 19,
    "legend.fontsize": 16,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.1,
})

HERE = Path(__file__).resolve().parent            # verification/
REPO = HERE.parent
OZ = HERE / "mixed_mode" / "ozgen_fig3"
RD = OZ / "_refdigitize"
FIRST = RD / "firstmode_grid.csv"
SECOND = RD / "secondmode_grid.csv"
DIG = REPO / "reference_data" / "digitized"

PYMACK_BLUE = "#000000"      # Our LST code curve: black, dashed, thick
PUB_RED = "#d62728"          # single-series colour for the simplified writeup figure
# (mode, lobe) -> (marker, edgecolor, label)  -- mirrors the proof figures.
# Labels kept short so the (below-axes) legend stays compact and readable.
STYLES = {
    ("first", "lower"):  ("o", "#0072b2", "Özgen 1st lower (onset)"),
    ("first", "upper"):  ("o", "#009e73", "Özgen 1st upper (cutoff)"),
    ("second", "lower"): ("s", "#cc79a7", "Özgen 2nd lower (onset)"),
    ("second", "upper"): ("s", "#d55e00", "Özgen 2nd upper (cutoff)"),
}
ORDER = [("first", "lower"), ("first", "upper"),
         ("second", "lower"), ("second", "upper")]


# ---------------------------------------------------------------------------
# Our LST code neutral locus: full c_i=0 contour of the committed grids (or the
# continuation trace where it exists). This mirrors build_ozgen_final.segs_of /
# continuation_segs exactly, but only READS committed data (no recompute).
# ---------------------------------------------------------------------------
def _load_grid(path: Path, Ma: float):
    if not path.exists():
        return None
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if abs(float(r["Ma"]) - Ma) > 1e-6:
                continue
            ci = r["c_i"]
            d[(float(r["Re"]), float(r["alpha"]))] = (
                float(ci) if ci not in ("", "nan") else np.nan)
    if not d:
        return None
    res = np.array(sorted({k[0] for k in d}))
    als = np.array(sorted({k[1] for k in d}))
    Z = np.full((len(als), len(res)), np.nan)
    for (Re, al), ci in d.items():
        Z[list(als).index(al), list(res).index(Re)] = ci
    return res, als, Z


def _grid_segs(Ma: float):
    """c_i=0 contour segments from the first- and second-mode grids."""
    out = []
    for path in (FIRST, SECOND):
        g = _load_grid(path, Ma)
        if g is None:
            continue
        res, als, Z = g
        Zm = np.where(np.isnan(Z), -9.99, Z)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contour(res, als, Zm, levels=[0.0])
        for col in cs.allsegs:
            for s in col:
                if len(s) >= 2:
                    out.append(s)
        plt.close(fig)
    return out


def _continuation_segs(Ma: int):
    """Continuation-traced first-mode branches (the real M2/M3 onset curve)."""
    p = RD / f"continuation_M{Ma}.csv"
    if not p.exists():
        return []
    br = collections.defaultdict(list)
    with open(p) as f:
        for r in csv.DictReader(f):
            br[r["branch"]].append((float(r["R"]), float(r["alpha"])))
    return [np.array(sorted(v)) for v in br.values() if len(v) >= 2]


def pymack_segs(Ma: int):
    cs = _continuation_segs(Ma)          # continuation is the real curve (M2/M3)
    return cs if cs else _grid_segs(Ma)


# ---------------------------------------------------------------------------
# Corrected multi-branch digitized reference (v2)
# ---------------------------------------------------------------------------
def ozgen_v2(Ma: int):
    ref = DIG / f"ozgen_fig3_M{Ma}_neutral_v2.csv"
    out = collections.defaultdict(list)
    with open(ref) as f:
        for r in csv.DictReader(f):
            out[(r.get("mode", "first"), r["lobe"])].append(
                (float(r["Re"]), float(r["alpha"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Title from verdict.json (live; numbers untouched)
# ---------------------------------------------------------------------------
def title_for(case_dir: Path, Ma: int):
    """Concise, case-identifying title only. No verdict word and no metric
    headline on the plot -- those live in the LaTeX caption and the success
    matrix, so the figure carries just the case identity."""
    v = json.loads((case_dir / "verdict.json").read_text(encoding="utf-8"))
    return (f"Özgen & Kırcalı (2008) Fig. 3:  $M={Ma}$  neutral curve"), v


def set_overlay_path(case_dir: Path, rel: str):
    """Refresh ONLY artifacts.overlay -- verdict / metric numbers untouched."""
    vf = case_dir / "verdict.json"
    v = json.loads(vf.read_text(encoding="utf-8"))
    v.setdefault("artifacts", {})
    v["artifacts"]["overlay"] = rel
    vf.write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")


def _common_axes(ax, title):
    ax.set_xlabel(r"Reynolds number  $R_L=\sqrt{Re_x}$")
    ax.set_ylabel(r"Wavenumber  $\alpha_{L}$")
    ax.set_title(title, color="#222222", pad=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(width=1.1, length=5.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.06)


def _render(Ma: int, segs, ref, title, simple: bool):
    """Build one overlay figure. ``simple=False`` -> the DETAILED development
    figure (per-branch markers: 1st/2nd x lower/upper). ``simple=True`` -> the
    PUBLICATION figure for the writeup: Our LST code's neutral curve (black line) plus
    ALL Ozgen reference points collapsed into a SINGLE red series with one label.
    Returns (fig, legend)."""
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    if simple:
        # All digitised reference points as ONE red series (no per-branch split).
        arrs = [ref[k] for k in ORDER if k in ref]
        pts = np.vstack(arrs) if arrs else np.empty((0, 2))
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], "o", color=PUB_RED, ms=4.0, mew=0,
                    linestyle="none", alpha=0.9, zorder=2,
                    label=f"Özgen & Kırcalı (2008),  $M={Ma}$")
        # Our LST code curve on TOP so the blue locus reads through the red band.
        for s in segs:
            ax.plot(s[:, 0], s[:, 1], "--", color=PYMACK_BLUE, lw=3.8, zorder=4)
        ax.plot([], [], "--", color=PYMACK_BLUE, lw=3.8,
                label="Our LST code")
        ncol, anchor = 2, -0.185
    else:
        for s in segs:
            ax.plot(s[:, 0], s[:, 1], "--", color=PYMACK_BLUE, lw=3.8)
        ax.plot([], [], "--", color=PYMACK_BLUE, lw=3.8,
                label="Our LST code neutral ($c_i=0$)")
        for key in ORDER:
            if key not in ref:
                continue
            arr = ref[key]
            mk, col, lab = STYLES[key]
            ax.plot(arr[:, 0], arr[:, 1], mk, mfc="none", mec=col, mew=1.4, ms=6.0,
                    linestyle="none", label=lab)
        n_entries = 1 + sum(1 for k in ORDER if k in ref)
        ncol, anchor = min(3, n_entries), -0.26
    _common_axes(ax, title)
    # Legend BELOW the axes (was outside-right, which stole ~30% of the width and
    # shrank everything on the page). Below -> the plot keeps the full image width.
    # Anchor well below the (large) x-axis label so the two never overlap.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, anchor), ncol=ncol,
                    framealpha=0.95, borderaxespad=0.0, handletextpad=0.5,
                    columnspacing=1.4, fontsize=15)
    return fig, leg


def make_overlay(Ma: int):
    case_dir = OZ / f"M{Ma}"
    if not (case_dir / "verdict.json").exists():
        print(f"  [skip] M={Ma}: no verdict.json at {case_dir}")
        return None
    segs = pymack_segs(Ma)
    ref = ozgen_v2(Ma)
    title, _v = title_for(case_dir, Ma)

    # (1) DETAILED development overlay -- kept for the success matrix / galleries
    #     (the per-branch breakdown is the working-diagnostic artifact).
    fig, leg = _render(Ma, segs, ref, title, simple=False)
    fig.savefig(case_dir / "overlay.png", dpi=170, bbox_inches="tight",
                bbox_extra_artists=[leg])
    plt.close(fig)

    # (2) SIMPLIFIED publication overlay -- single red reference series, one label;
    #     this is the figure embedded in docs/validation.tex.
    figp, legp = _render(Ma, segs, ref, title, simple=True)
    figp.savefig(case_dir / "overlay_pub.png", dpi=170, bbox_inches="tight",
                 bbox_extra_artists=[legp])
    plt.close(figp)

    rel = f"verification/mixed_mode/ozgen_fig3/M{Ma}/overlay.png"
    set_overlay_path(case_dir, rel)              # matrix/gallery -> detailed one
    n_ref = sum(len(ref[k]) for k in ref)
    print(f"  M{Ma}: {len(segs)} Our LST code contour segs, {n_ref} ref pts "
          f"({sorted('/'.join(k) for k in ref)}) -> overlay.png + overlay_pub.png")
    return rel


def main(machs):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print("Özgen Fig. 3 neutral-curve overlays:")
    written = []
    for Ma in machs:
        rel = make_overlay(Ma)
        if rel:
            written.append(rel)
    print(f"WROTE {len(written)} overlays.")
    return 0


if __name__ == "__main__":
    ms = [int(x) for x in (sys.argv[1:] or [2, 3, 4, 6, 7, 8, 10])]
    raise SystemExit(main(ms))
