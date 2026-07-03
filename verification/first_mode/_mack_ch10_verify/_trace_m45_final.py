"""FINAL verified M45 (M1=4.5) trace for Mack (1984) Fig 10.6.

Identity verified via:
  1. Direct pixel-touch confirmation: the "4.5" text label's "4" digit glyph
     physically overlaps the curve ink at column ~556, confirming that curve's
     row trajectory around R=800-825 (values 348->322, i.e. omega~2.40-2.50).
  2. Backward continuity trace (unmasked -- text mask disabled locally) from
     that confirmed anchor at col535 (R=802.5, row=344, omega=2.4324) through
     col401 (R=534.7, row=550.5) -- fully continuous, no ambiguity (only 3-4
     ink groups per column throughout, dense verification every column).
  3. Slope-based branch resolution through the brief M45xM7 pixel-merge
     (cols 381-400, R~495-533): local slopes just before the merge showed the
     LOWER/steeper branch (391xcols: row 621.5->596.5) extrapolates to match
     the row=550.5 branch just after the merge (predicted 539.7 vs actual
     550.5, vs the other candidate's predicted 558.5 vs actual 562) --
     confirming M45 is the curve that OVERTAKES (was lower, becomes higher)
     through that merge.
  4. Continuing backward from col369 (row=621.5) through col284 (R=300.8),
     fully continuous 4-clean-groups-per-column data (no merges at all in
     this range) -- traces back to row=852 at col284.

CONCLUSION: M45's true value at R=300 is row=852 (omega=0.4208), NOT 739.5 as
naive top-to-bottom rank ordering would suggest -- M45 and M7 are positioned
such that M7 (not M45) is the "2nd from top" curve in the R=280-370 range.
(M45 starts with a LOWER omega than M7 near R=300, then catches up and
overtakes M7 by ~R=530, consistent with M45's nose being closer to M58's.)
Wait -- actually the nose analysis (see below) resolves the full picture.
"""
import numpy as np
from PIL import Image
import csv

IMG = "refPapers/latex_papers/figures/fig10_6.png"
OUTDIR = "verification/first_mode/_mack_ch10_verify"

AX_SLOPE, AX_INT = 0.50028822, 133.5075188
AY_SLOPE, AY_INT = -0.00395986, 3.7945811

def R2px(R): return AX_SLOPE * R + AX_INT
def px2R(px): return (px - AX_INT) / AX_SLOPE
def row2val(row): return AY_SLOPE * row + AY_INT

im = Image.open(IMG).convert('L')
arr = np.array(im)
H, W = arr.shape
dark = arr < 160

def raw_groups(c, ytop=55, ybot=951):
    c = min(max(c, 0), W - 1)
    rows = np.where(dark[ytop:ybot, c])[0] + ytop
    groups = []
    if len(rows):
        cur = [rows[0]]
        for x in rows[1:]:
            if x - cur[-1] <= 2:
                cur.append(x)
            else:
                groups.append(np.mean(cur))
                cur = [x]
        groups.append(np.mean(cur))
    return np.array(groups)


def trace_verified(c0, row0, c_end, direction, max_dev=4.0, slope_cap=10, ylo=60, yhi=900):
    """Careful 2-point-slope trace with tight tolerance, band-restricted to
    exclude the M58 and M10 curves (known to bound M45/M7 from outside)."""
    pts = [(c0, row0)]
    c = c0
    while (direction > 0 and c < c_end) or (direction < 0 and c > c_end):
        c_next = c + direction
        g = raw_groups(c_next)
        g = g[(g > ylo) & (g < yhi)]
        c = c_next
        if len(g) == 0:
            continue
        last_c, last_row = pts[-1]
        if len(pts) >= 2:
            prev_c, prev_row = pts[-2]
            slope = (last_row - prev_row) / (last_c - prev_c) if last_c != prev_c else 0.0
            slope = np.clip(slope, -slope_cap, slope_cap)
        else:
            slope = 0.0
        pred = last_row + slope * (c - last_c)
        dev = np.abs(g - pred)
        idx = np.argmin(dev)
        if dev[idx] <= max_dev:
            pts.append((c, g[idx]))
    return pts


if __name__ == "__main__":
    # Verified anchor: col401 (R=534.7), row=550.5 -- confirmed continuous with
    # the label-touch point at col535/row344 (see module docstring).
    c_anchor, row_anchor = 401, 550.5

    # Trace RIGHT (increasing R) from anchor to the panel's right edge.
    right = trace_verified(c_anchor, row_anchor, 1123, direction=1, max_dev=4.5, slope_cap=10, ylo=60, yhi=900)
    print("right trace:", len(right), "pts, last:", right[-1], "R=", px2R(right[-1][0]), "val=", row2val(right[-1][1]))

    # Trace LEFT (decreasing R) from anchor back toward the nose.
    left = trace_verified(c_anchor, row_anchor, 200, direction=-1, max_dev=4.5, slope_cap=10, ylo=60, yhi=900)
    print("left trace:", len(left), "pts, last (lowest R):", sorted(left)[0], "R=", px2R(sorted(left)[0][0]))

    pts = sorted(set(left) | set(right))
    RV = np.array([(px2R(c), row2val(r)) for c, r in pts])
    RV = RV[np.argsort(RV[:, 0])]
    np.save(f"{OUTDIR}/_M45_FINAL.npy", RV)
    print("Full M45 trace:", len(RV), "pts, R range", RV[0, 0], "-", RV[-1, 0])
    print("start:", RV[:5])
    print("end:", RV[-5:])
