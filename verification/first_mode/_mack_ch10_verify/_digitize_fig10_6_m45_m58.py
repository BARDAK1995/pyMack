"""Re-digitize Mack (1984) Fig 10.6 M45 and M58 curves via pixel tracing.

Calibration (from axis tick pixel positions, sub-pixel linear fit residuals < 3px in x, < 0.01 in y):
  X: R = (px_col - 133.5075188) / 0.50028822
  Y: omega_i*1e3 = 3.7945811 - 0.00395986 * px_row

Method: slope-extrapolation tracer. At each column, predict the expected row from
a local linear fit of the last few accepted points, then pick the ink group closest
to that prediction (within a max deviation). This correctly threads curve crossings
(where nearest-neighbor-to-previous-point fails because two curves interleave, since
at a crossing the two curves have different, well-defined slopes that disambiguate them).

Curve identification (verified via careful numeric tracing across the whole panel):
  - M58 (M1=5.8): topmost of the 4 curves essentially everywhere (fastest nose rise,
    highest asymptote); very slightly overtaken by M45 only in the last ~20 R units
    near R=2000 (curves visually touch at the right edge).
  - M45 (M1=4.5): second curve. CROSSES BELOW M7 (the 3rd curve) once, near R~490-510
    (omega_i*1e3 ~ 1.48), then stays below M7 out to at least R=1000+ before M7
    asymptotes lower and M45 continues rising to rejoin M58 near R=2000.
"""
import numpy as np
from PIL import Image
import csv

IMG = "refPapers/latex_papers/figures/fig10_6.png"
OUTDIR = "verification/first_mode/_mack_ch10_verify"

AX_SLOPE, AX_INT = 0.50028822, 133.5075188   # px = AX_SLOPE*R + AX_INT
AY_SLOPE, AY_INT = -0.00395986, 3.7945811     # val = AY_SLOPE*row + AY_INT

def R2px(R):
    return AX_SLOPE * R + AX_INT

def px2R(px):
    return (px - AX_INT) / AX_SLOPE

def row2val(row):
    return AY_SLOPE * row + AY_INT

im = Image.open(IMG).convert('L')
arr = np.array(im)
H, W = arr.shape
dark = arr < 160

# Label mask boxes (x0,x1,y0,y1) to exclude from ink search - in-plot text
LABEL_BOXES = [
    (438, 668, 184, 232),   # "M1 = 5.8" label
    (533, 607, 326, 370),   # "4.5" label
    (583, 627, 483, 532),   # "7" label  (not in scope, but avoid confusion)
    (513, 577, 693, 742),   # "10" label (not in scope, but avoid confusion)
]

masked = dark.copy()
for (x0, x1, y0, y1) in LABEL_BOXES:
    masked[y0:y1, x0:x1] = False

# Frame bounds (axis box lines are at col~131-133 (left), ~1129-1130 (right), row~44-46 (top), row~953-957 (bottom))
YTOP, YBOT = 55, 951   # exclude bottom frame line (953-957) AND top frame/tick contamination (<55) from search


def col_groups(c):
    """Return sorted row-centers of dark pixel groups in column c within [YTOP,YBOT]."""
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


def trace_slope(c0, c1, row0, init_slope, max_dev=4.0, hist_n=6, slope_cap=None):
    """Trace left(c0)->right(c1) using linear extrapolation from recent history."""
    pts = [(c0, row0)]
    slope = init_slope
    stale = 0
    for c in range(c0 + 1, c1 + 1):
        last_c, last_row = pts[-1]
        pred = last_row + slope * (c - last_c)
        g = col_groups(c)
        if len(g) == 0:
            stale += 1
            continue
        dev = np.abs(g - pred)
        idx = np.argmin(dev)
        allowed = max_dev * (1 + 0.5 * stale)  # relax tolerance a bit after gaps
        if dev[idx] > allowed:
            stale += 1
            continue
        stale = 0
        cand = g[idx]
        pts.append((c, cand))
        if len(pts) >= 3:
            recent = pts[-hist_n:]
            cs = np.array([p[0] for p in recent])
            rs = np.array([p[1] for p in recent])
            if (cs.max() - cs.min()) > 0:
                A = np.polyfit(cs, rs, 1)
                new_slope = A[0]
            else:
                new_slope = (cand - last_row) / (c - last_c)
        else:
            new_slope = (cand - last_row) / (c - last_c)
        if slope_cap is not None:
            new_slope = np.clip(new_slope, -slope_cap, slope_cap)
        slope = new_slope
    return pts


def trace_bidir(c_seed, row_seed, c_min, c_max, init_slope, max_dev=4.0, hist_n=6,
                 slope_cap=None, direction=1, max_stale=30):
    """Trace starting at c_seed going in `direction` (+1 right, -1 left) to c_min/c_max.

    Uses local linear-slope extrapolation (robust through crossings where two curves
    have different slopes). Tolerance relaxes slightly during brief gaps (missing ink,
    anti-aliasing) but the trace aborts if it goes stale for too long (max_stale cols).
    """
    pts = [(c_seed, row_seed)]
    slope = init_slope
    stale = 0
    c = c_seed
    end = c_max if direction > 0 else c_min
    while True:
        c_next = c + direction
        if (direction > 0 and c_next > end) or (direction < 0 and c_next < end):
            break
        last_c, last_row = pts[-1]
        pred = last_row + slope * (c_next - last_c)
        g = col_groups(c_next)
        if len(g) == 0:
            stale += 1
            c = c_next
            if stale > max_stale:
                break
            continue
        dev = np.abs(g - pred)
        idx = np.argmin(dev)
        allowed = max_dev * (1 + 0.3 * stale)
        if dev[idx] > allowed:
            stale += 1
            c = c_next
            if stale > max_stale:
                break
            continue
        stale = 0
        cand = g[idx]
        pts.append((c_next, cand))
        recent = pts[-hist_n:]
        cs = np.array([p[0] for p in recent])
        rs = np.array([p[1] for p in recent])
        if (cs.max() - cs.min()) > 0:
            slope = np.polyfit(cs, rs, 1)[0]
        if slope_cap is not None:
            slope = np.clip(slope, -slope_cap, slope_cap)
        c = c_next
    return pts


def pts_to_RV(pts):
    return [(px2R(c), row2val(r)) for c, r in pts]


if __name__ == "__main__":
    # Seed both curves at R=300 (col 284) where all 4 curves are cleanly separated:
    # groups = [599 (M58), 739.5 (M45), 852 (M7), 890.5 (M10)]
    c_seed = int(round(R2px(300)))
    RIGHT_EDGE = 1123  # last column before right-edge tick/frame contamination
    LEFT_EDGE = 134    # just right of left frame (R=0)

    # --- M58 ---
    # Forward (R>300): shallow decelerating slope, always the topmost curve, no crossings
    # until it nearly touches M45 right at R~2000 (edge) -- trace stops naturally there.
    fwd58 = trace_bidir(c_seed, 599.0, LEFT_EDGE, RIGHT_EDGE, init_slope=-1.0,
                         max_dev=3.0, hist_n=6, slope_cap=10, direction=1)
    # Backward (R<300) down to the nose: slope steepens a lot near onset.
    bwd58 = trace_bidir(c_seed, 599.0, LEFT_EDGE, RIGHT_EDGE, init_slope=1.0,
                         max_dev=3.0, hist_n=4, slope_cap=30, direction=-1)
    pts58 = sorted(set(bwd58 + fwd58))

    # --- M45 ---
    # Forward (R>300): crosses BELOW M7 near R~490-510, then continues rising to
    # rejoin M58 near R~2000.
    fwd45 = trace_bidir(c_seed, 739.5, LEFT_EDGE, RIGHT_EDGE, init_slope=-1.5,
                         max_dev=3.0, hist_n=6, slope_cap=10, direction=1)
    # Backward (R<300) down to the nose (starts a touch later / less steep than M58).
    bwd45 = trace_bidir(c_seed, 739.5, LEFT_EDGE, RIGHT_EDGE, init_slope=1.5,
                         max_dev=3.0, hist_n=4, slope_cap=30, direction=-1)
    pts45 = sorted(set(bwd45 + fwd45))

    print("M58: traced", len(pts58), "pts, col range", pts58[0][0], pts58[-1][0])
    print("M45: traced", len(pts45), "pts, col range", pts45[0][0], pts45[-1][0])

    RV58 = pts_to_RV(pts58)
    RV45 = pts_to_RV(pts45)

    np.save(f"{OUTDIR}/_RV58.npy", np.array(RV58))
    np.save(f"{OUTDIR}/_RV45.npy", np.array(RV45))

    print("M58 sample:", RV58[:3], "...", RV58[-3:])
    print("M45 sample:", RV45[:3], "...", RV45[-3:])
