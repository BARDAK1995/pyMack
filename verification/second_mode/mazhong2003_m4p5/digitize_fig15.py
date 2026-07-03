# -*- coding: utf-8 -*-
"""Pixel-traced digitization of Ma & Zhong (2003) Fig. 15 neutral curves
(omega vs R), M=4.5.  UPGRADED from hand-eyeballed points to a real continuity
trace off the 400-DPI figure crop.

Source crop (2048 x 1502 px, RGB):
    refPapers/NewPapers/figures/mazhong2003_fig15_neutral_curves_2D.png

=== PIXEL -> DATA CALIBRATION (reproducible) ===
Established from the plot axis frame + tick marks (ticks point INWARD):
  * R-axis (R=0 vertical line)     : pixel column  ~173-176
  * omega=0 horizontal line        : pixel row     ~1330-1332
  * X major ticks (inward, above the x-axis) detected at columns
        174(=R0), 348, 520, 693, 866, 1036, 1210, 1382  -> R = 0,200,...,1400
    linear fit:  R = 1.159656 * col - 202.9371   (tick residuals < 1.5 in R)
  * Y major ticks (inward, right of the y-axis) detected at rows
        1146, 962, 779, 594, 412, 230, 46         -> omega = 0.04,...,0.28
    linear fit:  omega = -0.00021828 * row + 0.290001  (residuals < 3e-4)
Interior-tick verification (independent of the endpoints used in the fit):
    R=1000  -> predicted col 1037.3  (tick detected at 1036)  OK
    omega=0.12 -> predicted row 778.8 (tick detected at 779)  OK
Anchor pixels used (from the fits):
    R=0    <-> col 175.00      R=2000 <-> col 1899.65
    omega=0 <-> row 1328.57    omega=0.28 <-> row 45.82

=== TRACE METHOD ===
For each of the 4 branches, march in R across the figure, and in each pixel
column take the dark-pixel runs (gray < 110) inside that branch's omega window.
Runs whose omega falls on either dotted reference ray (F=0.6e-4 or F=2.2e-4,
omega=F*R, tol 0.0035) are rejected, so the sparse dotted F-rays are never
traced into the neutral-curve data.  Among the remaining runs the one closest in
omega to the previous point is taken (continuity); a per-step jump gate rejects
strays/labels.  Seeds are clean segments (2nd mode from the R=2000 right edge;
1st-mode lower from a mid-curve seed marched both ways).

CROSS-CHECK (independent textual anchor, Ma & Zhong sec. 6): the F=2.2e-4 line
crosses the 2nd-mode lower branch (I) at R=806 and the upper branch (II) at
R=999.6.  The traced branches give ~778.8 (I) and ~1002.5 (II) -- see the
printout / _pixel_calibration.md.

Writes reference_mazhong_fig15.csv (schema mode,branch,R,omega -- unchanged) and
_verify_fig15_pixel_digitized.png (traced points over the source crop).
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = (HERE.parent.parent.parent / "refPapers" / "NewPapers" / "figures"
       / "mazhong2003_fig15_neutral_curves_2D.png")

# --- calibration constants (see module docstring) -------------------------
MX, BX = 1.159656, -202.9371        # R     = MX*col + BX
MY, BY = -0.00021828, 0.290001      # omega = MY*row + BY
DARK_THRESH = 110                   # gray level below which a pixel is "ink"
F1, F2 = 0.6e-4, 2.2e-4             # dotted reference rays (must NOT be traced)
RAY_TOL = 0.0035                    # omega tolerance for rejecting F-ray pixels

_IM = None
_H = _W = None


def _load():
    global _IM, _H, _W
    if _IM is None:
        g = np.asarray(Image.open(FIG).convert("L")).astype(np.float64)
        _IM = g < DARK_THRESH
        _H, _W = g.shape
    return _IM


def R2col(R):
    return (R - BX) / MX


def om2row(o):
    return (o - BY) / MY


def row2om(r):
    return MY * r + BY


def _dark_runs(col, r0, r1):
    """(row_center, length) of contiguous dark runs in `col`, rows r0..r1."""
    dark = _load()
    col = int(round(col)); r0 = max(0, int(r0)); r1 = min(_H, int(r1))
    seg = dark[r0:r1, col]
    runs = []; i = 0; n = len(seg)
    while i < n:
        if seg[i]:
            j = i
            while j < n and seg[j]:
                j += 1
            runs.append(((r0 + (i + j - 1) / 2.0), j - i))
            i = j
        else:
            i += 1
    return runs


def _is_ray(omega, R):
    return (abs(omega - F1 * R) < RAY_TOL) or (abs(omega - F2 * R) < RAY_TOL)


def _march(R_targets, om_seed, om_band, jump=0.012, max_miss=3):
    om_lo, om_hi = om_band
    out = []; prev = om_seed; miss = 0
    for R in R_targets:
        runs = _dark_runs(R2col(R), om2row(om_hi), om2row(om_lo))
        cands = [(row2om(rc), ln) for rc, ln in runs
                 if om_lo <= row2om(rc) <= om_hi and not _is_ray(row2om(rc), R)]
        if not cands:
            miss += 1
            if miss > max_miss:
                break
            continue
        miss = 0
        cands.sort(key=lambda t: abs(t[0] - prev))
        if abs(cands[0][0] - prev) > jump and len(cands) > 1:
            continue
        out.append((float(R), cands[0][0])); prev = cands[0][0]
    return out


def _bidir(R_seed, om_seed, om_band, Rmin, Rmax, step=20, jump=0.012):
    left = _march(np.arange(R_seed, Rmin - 1, -step), om_seed, om_band, jump)
    right = _march(np.arange(R_seed, Rmax + 1, step), om_seed, om_band, jump)
    d = {}
    for R, om in left + right:
        d[round(R)] = om
    return sorted(d.items())


def trace_all():
    _load()
    sec_up = _march(np.arange(2000, 246, -25), 0.2244, (0.184, 0.232), jump=0.010)
    sec_lo = _march(np.arange(2000, 246, -25), 0.1766, (0.166, 0.185), jump=0.008)
    fir_up = _march(np.arange(2000, 558, -20), 0.1212, (0.033, 0.128), jump=0.013)
    fir_lo = _bidir(1000, 0.0091, (0.0055, 0.045), 560, 2000, step=20, jump=0.010)
    for lst in (sec_up, sec_lo, fir_up, fir_lo):
        lst.sort()
    return {
        ("second", "upper"): sec_up,
        ("second", "lower"): sec_lo,
        ("first", "upper"): fir_up,
        ("first", "lower"): fir_lo,
    }


def cross_F2(pts):
    """R at which omega = F2*R crosses a branch (linear interp)."""
    arr = np.array(sorted(pts)); Rr = arr[:, 0]; om = arr[:, 1]
    g = om - F2 * Rr
    for i in range(len(Rr) - 1):
        if g[i] * g[i + 1] < 0:
            t = g[i] / (g[i] - g[i + 1])
            return Rr[i] + t * (Rr[i + 1] - Rr[i])
    return None


def write_csv(curves):
    p = HERE / "reference_mazhong_fig15.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "branch", "R", "omega"])
        for (mode, br), pts in curves.items():
            for R, om in pts:
                w.writerow([mode, br, round(R, 1), round(om, 5)])
    return p


def verify_plot(curves):
    img = np.asarray(Image.open(FIG).convert("RGB"))
    xL = BX; xR = MX * (_W - 1) + BX
    yT = BY; yB = MY * (_H - 1) + BY
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(img, extent=[xL, xR, yB, yT], aspect="auto", origin="upper")
    colmap = {("second", "upper"): "red", ("second", "lower"): "lime",
              ("first", "upper"): "cyan", ("first", "lower"): "magenta"}
    for key, pts in curves.items():
        arr = np.array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=24, c=colmap[key],
                   edgecolors="k", linewidths=0.4, zorder=5,
                   label=f"{key[0]} {key[1]}")
    ax.set_xlim(0, 2050); ax.set_ylim(0, 0.29)
    ax.set_xlabel("R", fontsize=15); ax.set_ylabel(r"$\omega$", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="center right")
    ax.set_title("Ma & Zhong (2003) Fig. 15 -- pixel-traced points over source crop",
                 fontsize=15)
    fig.tight_layout()
    out = HERE / "_verify_fig15_pixel_digitized.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    curves = trace_all()
    print("Traced points per branch:")
    for k, v in curves.items():
        arr = np.array(v)
        print(f"  {k[0]:6s} {k[1]:5s}: {len(v):3d} pts, "
              f"R {arr[:,0].min():.0f}..{arr[:,0].max():.0f}, "
              f"omega {arr[:,1].min():.4f}..{arr[:,1].max():.4f}")
    bI = cross_F2(curves[("second", "lower")])
    bII = cross_F2(curves[("second", "upper")])
    print("\nCROSS-CHECK vs Ma & Zhong sec.6 (F=2.2e-4 branch points):")
    print(f"  Branch I  (2nd lower): R = {bI:.1f}   vs paper 806    "
          f"({abs(bI-806)/806*100:.1f}%)")
    print(f"  Branch II (2nd upper): R = {bII:.1f}  vs paper 999.6  "
          f"({abs(bII-999.6)/999.6*100:.1f}%)")
    print("\nwrote", write_csv(curves))
    print("wrote", verify_plot(curves))
