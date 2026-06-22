"""Digitize Özgen M6's two INNER branches (the ones facing the stable gap that the
v2 reference missed): the 1st-mode UPPER/cutoff (~0.11) and the 2nd-mode
LOWER/onset (~0.15).  Overlay all four branches on the panel to verify, then
append the two new branches to the M6 v2 reference.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PANEL = HERE / "panels_v2/ozgen_M6_panele.png"   # axis: Re 0..5000, alpha 0..0.4
REF = REPO / "reference_data/digitized/ozgen_fig3_M6_neutral_v2.csv"

# --- my reads of the two inner branches (Re, alpha) from the panel ---
second_lower = [   # 2nd-mode onset (bottom edge of upper lobe), ~flat near 0.15
    (220, 0.146), (300, 0.150), (500, 0.151), (800, 0.150), (1200, 0.150),
    (1800, 0.149), (2600, 0.149), (3500, 0.150), (4500, 0.151), (5000, 0.151),
]
first_upper = [    # 1st-mode cutoff (top edge of lower lobe), rises with Re
    (880, 0.052), (1000, 0.070), (1300, 0.085), (1800, 0.097), (2500, 0.104),
    (3300, 0.109), (4200, 0.113), (5000, 0.116),
]

# --- existing v2 branches (for the verification plot) ---
ex_up, ex_lo = [], []
with open(REF) as f:
    for r in csv.DictReader(f):
        (ex_up if r["lobe"] == "upper" else ex_lo).append(
            (float(r["Re"]), float(r["alpha"]), r["mode"]))

# --- calibrate the panel image to data coords (axis box detected by eye-fit) ---
# panel render: find axis box. We know axis spans Re 0..5000 (x), alpha 0..0.4 (y).
img = mpimg.imread(str(PANEL))
H, W = img.shape[:2]
# Axis-box fractions for this 6x render (tuned to the panels_v2 M6 crop):
# left/right/top/bottom as fraction of width/height where the data box sits.
xl, xr = 0.135, 0.985      # Re=0 .. Re=5000
yt, yb = 0.115, 0.760      # alpha=0.4 (top) .. alpha=0 (bottom)


def to_px(Re, al):
    x = xl + (xr - xl) * (Re / 5000.0)
    y = yt + (yb - yt) * (1 - al / 0.4)
    return x * W, y * H


fig, ax = plt.subplots(figsize=(11, 7.5))
ax.imshow(img)
for arr, c, lab in ((second_lower, "#d55e00", "NEW 2nd-mode lower (onset)"),
                    (first_upper, "#0072b2", "NEW 1st-mode upper (cutoff)")):
    px = [to_px(Re, al) for Re, al in arr]
    ax.plot([p[0] for p in px], [p[1] for p in px], "o-", color=c, ms=7, lw=2, label=lab)
for arr, c, lab in ((ex_up, "#cc79a7", "v2 existing upper"), (ex_lo, "#009e73", "v2 existing lower")):
    px = [to_px(Re, al) for Re, al, _ in arr]
    ax.plot([p[0] for p in px], [p[1] for p in px], "s", mfc="none", mec=c, mew=2, ms=6, label=lab)
ax.set_title("M6: digitized 4 branches over panel — verify inner branches on c_i=0", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.axis("off")
fig.tight_layout()
fig.savefig(HERE / "_verify_M6_innerbranches.png", dpi=130)
plt.close(fig)
print("wrote _verify_M6_innerbranches.png — READ IT to confirm inner branches sit on c_i=0")
print(f"2nd-mode lower: {len(second_lower)} pts, alpha {min(a for _,a in second_lower):.3f}-{max(a for _,a in second_lower):.3f}")
print(f"1st-mode upper: {len(first_upper)} pts, alpha {min(a for _,a in first_upper):.3f}-{max(a for _,a in first_upper):.3f}")
