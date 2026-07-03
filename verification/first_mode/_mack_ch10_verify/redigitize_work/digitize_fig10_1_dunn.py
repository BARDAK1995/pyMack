"""Re-digitize Mack (1984) Fig 10.1 dunn_asymptotic / dunn_numerical / complete_equations
neutral loops for M1.6 (panel a) and M2.2 (panel b).

Approach:
  1. Threshold the source PNG for ink (<150 gray).
  2. Restrict to each panel's plot interior (exclude frame lines) and keep only the
     single largest connected component -> this strips all text/label glyphs while
     retaining the full curve network (3 nested loops + small leader-line arrow stubs
     that touch the curves near each nose).
  3. Pixel<->data calibration from axis tick pixel positions (regression, see below).
  4. Trace each of the 3 loops' upper and lower branches by column-wise ink-group
     continuity tracing seeded at the loop's nose (leftmost/topmost point), enforcing
     a max jump per column step so the trace cannot hop onto a neighboring loop or
     onto an arrow stub.
  5. Sample traced pixel paths onto output R grids and write CSVs + overlay plot.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "refPapers/latex_papers/figures/fig10_1.png"
OUTDIR = "verification/first_mode/_mack_ch10_verify"

# ---------------------------------------------------------------------------
# Calibration (from axis tick pixel regression, see session notes):
#   x (R):  px = 0.8577198*R + 146.4835   (shared between panels a and b)
#   Panel a F: px = -216.6477*F + 877.6136   (F=0 -> row 877.6, F=4.0 -> row 12.0)
#   Panel b F: px = -215.5000*F + 1814.9545  (F=0 -> row 1815.0, F=4.0 -> row 953.5)
# ---------------------------------------------------------------------------
RX_M, RX_B = 0.8577197802197801, 146.48351648351644


def px2R(px):
    return (px - RX_B) / RX_M


def R2px(R):
    return RX_M * R + RX_B


PANELS = {
    "a": dict(
        rows=(16, 875), cols=(148, 1174),
        fy_m=-216.64772727272745, fy_b=877.6136363636367,
        frame_rows=(13, 14, 876, 877, 878, 879, 880),
    ),
    "b": dict(
        rows=(954, 1815), cols=(148, 1174),
        fy_m=-215.50000000000023, fy_b=1814.9545454545462,
        frame_rows=(951, 952, 953, 954, 955, 1816, 1817, 1818),
    ),
}


def px2F(px, panel):
    p = PANELS[panel]
    return (px - p["fy_b"]) / p["fy_m"]


def load_curve_mask(panel):
    """Threshold, restrict to plot interior, keep largest connected component.

    Also strips major-tick stubs near the top/bottom frame: these ticks are
    long enough to poke into the "interior" row range, and if a traced curve
    passes close to F~0 or F~4 near a tick's R position the tick can fuse
    with the curve into the same connected component and hijack a trace onto
    the frame line itself. Ticks sit at R = 0,100,...,1200 (px2R calibration).
    """
    im = Image.open(SRC).convert("L")
    a = np.array(im)
    dark = a < 150
    p = PANELS[panel]
    r0, r1 = p["rows"]
    c0, c1 = p["cols"]
    interior = np.zeros_like(dark)
    interior[r0:r1, c0:c1] = dark[r0:r1, c0:c1]
    # Blank out major-tick stubs near the top/bottom frame at each tick's
    # column. A tick is a short vertical run of ink whose column span is
    # narrower than any real curve crossing it at a shallow angle; rather
    # than blanking a fixed rectangle (which can also delete genuine curve
    # ink that happens to pass near F~0/F~4 at a tick's R), we only remove
    # rows, at the tick column, where the dark run is confined to a single
    # tick-width column band across the whole row (i.e. no wider curve ink
    # is also present in that row nearby) -- this targets the pure vertical
    # tick shaft while leaving curve+tick fusion rows untouched.
    tick_R = range(0, 1201, 100)
    for R in tick_R:
        c = int(round(R2px(R)))
        clo, chi = max(c0, c - 2), min(c1, c + 3)
        for band_r0, band_r1 in ((r0, r0 + 22), (r1 - 22, r1)):
            for rr in range(band_r0, band_r1):
                row_seg = dark[rr, clo - 6:chi + 6]
                # count contiguous dark runs touching the tick column window
                if row_seg.sum() <= (chi - clo) + 2:
                    interior[rr, clo:chi] = False
    lbl, n = ndimage.label(interior, structure=np.ones((3, 3)))
    sizes = ndimage.sum(interior, lbl, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    mask = lbl == biggest
    return mask


def groups_in_col(mask, c, r0, r1):
    col = mask[r0:r1, c]
    rows = np.where(col)[0] + r0
    if len(rows) == 0:
        return []
    g = []
    cur = [rows[0]]
    for x in rows[1:]:
        if x - cur[-1] <= 2:
            cur.append(x)
        else:
            g.append((cur[0] + cur[-1]) / 2.0)
            cur = [x]
    g.append((cur[0] + cur[-1]) / 2.0)
    return g


def trace_branch(mask, r0, r1, c_start, c_end, row_start, max_jump, direction=1):
    """Trace one branch column-by-column with nearest-neighbor continuity.
    direction=1: c increases; direction=-1: c decreases.
    Returns list of (col, row)."""
    res = []
    prev = row_start
    c = c_start
    while (direction == 1 and c <= c_end) or (direction == -1 and c >= c_end):
        g = groups_in_col(mask, c, r0, r1)
        if len(g) == 0:
            c += direction
            continue
        g = np.array(g)
        near = g[np.abs(g - prev) <= max_jump]
        if len(near) == 0:
            c += direction
            continue
        cand = near[np.argmin(np.abs(near - prev))]
        res.append((c, cand))
        prev = cand
        c += direction
    return res


def find_nose(mask, r0, r1, c0, c1):
    """Nose = column of the loop's topmost (min-row) ink pixel within [c0,c1)."""
    best_row = None
    best_col = None
    for c in range(c0, c1):
        rows = np.where(mask[r0:r1, c])[0]
        if len(rows) == 0:
            continue
        top = rows.min() + r0
        if best_row is None or top < best_row:
            best_row = top
            best_col = c
    return best_col, best_row
