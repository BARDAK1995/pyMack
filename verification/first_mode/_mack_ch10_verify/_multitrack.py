"""Multi-object tracking approach: at every column, we have up to 4 ink groups
(M58, M45, M7, M10 -- top to bottom in general, but M45/M7 and M45/M58 nearly
touch at a few R). We track ALL curves simultaneously frame-by-frame using
greedy nearest-neighbor assignment based on each track's own 2-point slope
extrapolation, processing columns in order and always assigning the best
overall pairing (track x candidate) via a simple greedy min-cost matching.
This is much more robust than single-curve tracing because at each column we
use ALL tracks' relative order/spacing as a consistency check.
"""
import numpy as np
from PIL import Image

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


N_TRACKS = 4  # M58, M45, M7, M10 (top to bottom at seed column)
SEED_COL = int(round(R2px(300)))
SEED_ROWS = None  # filled below


def init_tracks():
    g = col_groups(SEED_COL)
    g = np.sort(g)
    assert len(g) == N_TRACKS, g
    return list(g)  # ordered top(M58) to bottom(M10)


from scipy.optimize import linear_sum_assignment

BIG = 1e6


def _predict(history, i, c_target, slope_cap):
    h = history[i]
    if len(h) >= 2:
        (c2, r2), (c1p, r1p) = h[-1], h[-2]
        slope = (r2 - r1p) / (c2 - c1p) if c2 != c1p else 0.0
        slope = np.clip(slope, -slope_cap, slope_cap)
        return r2 + slope * (c_target - c2)
    elif len(h) == 1:
        return h[-1][1]
    return None


def _step(history, c_target, g, max_dev, slope_cap):
    """One optimal-assignment step: match all tracks to available groups (or
    to a 'no match' dummy) minimizing total absolute deviation."""
    n = N_TRACKS
    m = len(g)
    preds = [_predict(history, i, c_target, slope_cap) for i in range(n)]
    # cost matrix: rows=tracks, cols = groups + n dummy "no-match" columns (cost=max_dev, so
    # matching is only preferred over no-match when strictly better)
    cost = np.full((n, m + n), BIG)
    for i in range(n):
        if preds[i] is None:
            cost[i, m:] = 0  # any dummy is fine (no cost)
            continue
        for j in range(m):
            d = abs(g[j] - preds[i])
            cost[i, j] = d if d <= max_dev else BIG
        cost[i, m:] = max_dev  # cost of not matching = threshold (so real matches within
                                 # max_dev are always preferred over skipping)
    row_ind, col_ind = linear_sum_assignment(cost)
    for i, j in zip(row_ind, col_ind):
        if j < m and cost[i, j] < BIG:
            history[i].append((c_target, g[j]))


def track_forward(c0, c1, seed_rows, max_dev=5.0, slope_cap=12):
    history = [[(c0, seed_rows[i])] for i in range(N_TRACKS)]
    for c in range(c0 + 1, c1 + 1):
        g = col_groups(c)
        if len(g) == 0:
            continue
        _step(history, c, g, max_dev, slope_cap)
    return history


def track_backward(c0, c_min, seed_rows, max_dev=5.0, slope_cap=12):
    history = [[(c0, seed_rows[i])] for i in range(N_TRACKS)]
    c = c0
    while c > c_min:
        c_next = c - 1
        g = col_groups(c_next)
        c = c_next
        if len(g) == 0:
            continue
        _step(history, c_next, g, max_dev, slope_cap)
    return history


if __name__ == "__main__":
    seed_rows = init_tracks()
    print("seed rows (M58,M45,M7,M10):", seed_rows, [round(row2val(r), 4) for r in seed_rows])

    hist_fwd = track_forward(SEED_COL, 1123, seed_rows, max_dev=5.0, slope_cap=12)
    hist_bwd = track_backward(SEED_COL, 134, seed_rows, max_dev=5.0, slope_cap=12)

    names = ["M58", "M45", "M7", "M10"]
    for i, name in enumerate(names):
        allpts = sorted(set(hist_fwd[i]) | set(hist_bwd[i]))
        RV = np.array([(px2R(c), row2val(r)) for c, r in allpts])
        np.save(f"{OUTDIR}/_MT_{name}.npy", RV)
        print(name, "npts", len(RV), "R range", RV[:,0].min(), "-", RV[:,0].max())
