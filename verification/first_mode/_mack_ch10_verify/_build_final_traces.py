"""Build final M45 / M58 traces for Mack (1984) Fig 10.6 using verified curve
identification logic derived from careful manual+numeric inspection:

  - M58 (M1=5.8): topmost of the 4 curves from its nose (R~140) all the way out
    to R~1700, where it merges (pixel-touches) with M45; after the merge
    (R~1860 on) M58 becomes the SLIGHTLY LOWER of the pair (M45 has a steeper
    local slope there and overtakes it) out to the panel's right edge (R=2000).
  - M45 (M1=4.5): second curve down to R~490, where it crosses BELOW the M7
    curve (3rd curve) -- so between R~490 and ~540 M45 is temporarily the
    3rd-ranked curve -- then re-emerges above M7 again by R~540 and continues
    as the 2nd curve up to R~1700, where it merges with M58 and becomes the
    TOPMOST of the pair from R~1860 to the right edge (R=2000).

Because the M45/M7 crossing (R~490-540) and the M45/M58 merge (R~1700-1860)
both involve near-tangent ink with sub-pixel separation, we trace using
strict rank-continuity (nearest ink group to the predicted extrapolation of
the physical curve's own recent history) rather than naive nearest-pixel,
and we bridge the two ambiguous zones using the pre/post slope match verified
manually above.
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

LABEL_BOXES = [
    (438, 668, 184, 232),
    (533, 607, 326, 370),
    (583, 627, 483, 532),
    (513, 577, 693, 742),
]
masked = dark.copy()
for (x0, x1, y0, y1) in LABEL_BOXES:
    masked[y0:y1, x0:x1] = False

YTOP, YBOT = 55, 951

def col_groups(c):
    c = min(max(c, 0), W - 1)
    col = masked[YTOP:YBOT, c]
    rows = np.where(col)[0] + YTOP
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


def rank_trace(c0, c1, row0, rank_selector, max_dev=3.5, slope_cap=12):
    """Trace using 2-point slope extrapolation + a rank_selector fallback.

    rank_selector(groups, pred) -> candidate row, used when nothing is within
    max_dev of the prediction (lets us jump past brief occlusion/merges using
    domain knowledge, e.g. "topmost" or "2nd topmost").
    """
    pts = [(c0, row0)]
    for c in range(c0 + 1, c1 + 1):
        last_c, last_row = pts[-1]
        if len(pts) >= 2:
            prev_c, prev_row = pts[-2]
            slope = (last_row - prev_row) / (last_c - prev_c) if last_c != prev_c else 0.0
            slope = np.clip(slope, -slope_cap, slope_cap)
        else:
            slope = 0.0
        pred = last_row + slope * (c - last_c)
        g = col_groups(c)
        if len(g) == 0:
            continue
        dev = np.abs(g - pred)
        idx = np.argmin(dev)
        if dev[idx] <= max_dev:
            pts.append((c, g[idx]))
        else:
            cand = rank_selector(g, pred)
            if cand is not None:
                pts.append((c, cand))
    return pts


def pts_to_RV(pts):
    return [(px2R(c), row2val(r)) for c, r in pts]

# ---------------------------------------------------------------------
# M58: seed at R=300 (col 284), row=599 (topmost). Trace forward with strict
# small max_dev; use topmost-group fallback during any brief gap.
# ---------------------------------------------------------------------
def topmost(g, pred):
    return g.min() if len(g) else None

c_seed = int(round(R2px(300)))
fwd58 = rank_trace(c_seed, 1123, 599.0, topmost, max_dev=3.5, slope_cap=12)
bwd58 = list(reversed(rank_trace(1123 - (c_seed - 134) if False else c_seed, 134, 599.0, topmost, max_dev=3.5, slope_cap=12)))
# backward trace (direction -1) implemented via a small local routine:

def rank_trace_bwd(c0, c_min, row0, rank_selector, max_dev=3.5, slope_cap=12):
    pts = [(c0, row0)]
    c = c0
    while c > c_min:
        c_next = c - 1
        last_c, last_row = pts[-1]
        if len(pts) >= 2:
            prev_c, prev_row = pts[-2]
            slope = (last_row - prev_row) / (last_c - prev_c) if last_c != prev_c else 0.0
            slope = np.clip(slope, -slope_cap, slope_cap)
        else:
            slope = 0.0
        pred = last_row + slope * (c_next - last_c)
        g = col_groups(c_next)
        c = c_next
        if len(g) == 0:
            continue
        dev = np.abs(g - pred)
        idx = np.argmin(dev)
        if dev[idx] <= max_dev:
            pts.append((c, g[idx]))
        else:
            cand = rank_selector(g, pred)
            if cand is not None:
                pts.append((c, cand))
    return pts

bwd58 = rank_trace_bwd(c_seed, 134, 599.0, topmost, max_dev=3.5, slope_cap=12)
pts58 = sorted(set(fwd58) | set(bwd58))
print("M58 raw trace:", len(pts58), "cols", pts58[0][0], "-", pts58[-1][0])

RV58 = np.array(pts_to_RV(pts58))

# Drop known bad-fallback segments identified by manual inspection:
#  - cols very near the left frame (R<130) where "topmost" fallback grabbed
#    axis/tick-label ink instead of the (not-yet-started) M58 curve
#  - R in [180,189] where fallback grabbed the "3.6" y-axis tick-label text
bad = (RV58[:, 0] < 130) | ((RV58[:, 0] > 179) & (RV58[:, 0] < 190))
RV58 = RV58[~bad]
RV58 = RV58[np.argsort(RV58[:, 0])]

np.save(f"{OUTDIR}/_RV58_v2.npy", RV58)
print("M58 cleaned:", len(RV58), "pts, R range", RV58[0, 0], "-", RV58[-1, 0])
print("M58 sample start:", RV58[:3])
print("M58 sample end:", RV58[-5:])

# ---------------------------------------------------------------------
# M45: seed at R=300 (col 284), row=739.5 (2nd from topmost). This curve:
#   - is the 2nd-highest from its nose (R~163) out to R~490
#   - dips to 3rd-highest (crosses below M7) from R~490 to R~540
#   - returns to 2nd-highest from R~540 to R~1700
#   - merges with M58 (pixel-touch) R~1700-1852, re-emerging as the
#     TOPMOST of the pair from R~1856 to the right edge (R=2000), since its
#     local slope there is steeper than M58's (verified above).
# We trace using strict small-max_dev continuity throughout; through the two
# ambiguous zones (merges) simple 2-point extrapolation naturally threads
# the correct branch because the two curves have distinguishably different
# local slopes at the point where ink separates again.
# ---------------------------------------------------------------------

def second_rank(g, pred):
    """Fallback: pick the 2nd-smallest (2nd highest omega) group, used only
    when nothing is close to the predicted row (brief occlusion)."""
    if len(g) >= 2:
        s = np.sort(g)
        return s[1]
    elif len(g) == 1:
        return g[0]
    return None

fwd45 = rank_trace(c_seed, 1123, 739.5, second_rank, max_dev=3.5, slope_cap=12)
bwd45 = rank_trace_bwd(c_seed, 204, 739.5, second_rank, max_dev=3.5, slope_cap=12)
pts45 = sorted(set(fwd45) | set(bwd45))
print("\nM45 raw trace:", len(pts45), "cols", pts45[0][0], "-", pts45[-1][0])

RV45 = np.array(pts_to_RV(pts45))
RV45 = RV45[np.argsort(RV45[:, 0])]
np.save(f"{OUTDIR}/_RV45_v2.npy", RV45)
print("M45 sample start:", RV45[:5])
print("M45 sample end:", RV45[-5:])

# Quick sanity: print M45 through the two tricky zones
for lo, hi, label in [(470, 560, "M45xM7 crossing"), (1680, 1880, "M45xM58 merge")]:
    zone = RV45[(RV45[:, 0] >= lo) & (RV45[:, 0] <= hi)]
    print(f"\n--- {label} ---")
    for r, v in zone[::4]:
        print(round(r, 1), round(v, 4))
