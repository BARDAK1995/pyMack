"""Re-digitize Mack (1984) Fig 10.3 curves via programmatic pixel tracing.

Pattern follows verification/first_mode/_ozgen_refdigitize/_digitize_all_v2.py:
threshold the image for ink, then trace each curve column-by-column enforcing
continuity (nearest ink cluster to the previous column's traced row, within a
max-jump bound) to avoid hopping onto neighboring curves or label text.

Calibration reused (and independently re-verified against axis tick pixel
positions in fig10_3.png) from verify_fig10_3.py:
    panel a (M1.3, top-left):    col0=200,col1=1287 (x:0-20); row0=477(y=0),row1=49(y=1.6)
    panel b (M1.6, bottom-left): col0=200,col1=1287; row0=962(y=0),row1=527(y=1.6)
    panel c (M2.2, top-right):   col0=1340.5,col1=2429; row0=477(y=0),row1=49(y=1.6)
    panel d (M3.0, bottom-right):col0=1340.5,col1=2429; row0=962(y=0),row1=527(y=1.6)
"""
import numpy as np
from PIL import Image
import csv
import os

BASE = r"C:\Users\merts\OneDrive\Masaüstü\MS_LST"
IMG_PATH = os.path.join(BASE, "refPapers", "latex_papers", "figures", "fig10_3.png")
OUTDIR = os.path.join(BASE, "verification", "first_mode", "_mack_ch10_verify", "_redigitize_2026")

im = Image.open(IMG_PATH).convert("L")
ARR = np.array(im)
DARK = ARR < 128
H, W = ARR.shape

CAL = {
    "a": {"x0": 0, "col0": 200.0, "x1": 20, "col1": 1287.0,
          "y0": 0, "row0": 477.0, "y1": 1.6, "row1": 49.0},
    "b": {"x0": 0, "col0": 200.0, "x1": 20, "col1": 1287.0,
          "y0": 0, "row0": 962.0, "y1": 1.6, "row1": 527.0},
    "c": {"x0": 0, "col0": 1340.5, "x1": 20, "col1": 2429.0,
          "y0": 0, "row0": 477.0, "y1": 1.6, "row1": 49.0},
    "d": {"x0": 0, "col0": 1340.5, "x1": 20, "col1": 2429.0,
          "y0": 0, "row0": 962.0, "y1": 1.6, "row1": 527.0},
}


def x2col(panel, x):
    c = CAL[panel]
    return c["col0"] + (x - c["x0"]) * (c["col1"] - c["col0"]) / (c["x1"] - c["x0"])


def col2x(panel, col):
    c = CAL[panel]
    return c["x0"] + (col - c["col0"]) * (c["x1"] - c["x0"]) / (c["col1"] - c["col0"])


def row2y(panel, row):
    c = CAL[panel]
    return c["y0"] + (row - c["row0"]) * (c["y1"] - c["y0"]) / (c["row1"] - c["row0"])


def y2row(panel, y):
    c = CAL[panel]
    return c["row0"] + (y - c["y0"]) * (c["row1"] - c["row0"]) / (c["y1"] - c["y0"])


def panel_row_bounds(panel):
    """Interior plot-box row bounds (top,bottom), excluding frame lines."""
    c = CAL[panel]
    top = min(c["row0"], c["row1"])
    bot = max(c["row0"], c["row1"])
    return top, bot


def col_groups(col, row_lo, row_hi, min_gap=3):
    """Dark-pixel clusters in column `col` between rows [row_lo,row_hi]."""
    rows = np.where(DARK[row_lo:row_hi, col])[0] + row_lo
    if len(rows) == 0:
        return []
    groups = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= min_gap:
            cur.append(r)
        else:
            groups.append((np.mean(cur), cur[0], cur[-1]))
            cur = [r]
    groups.append((np.mean(cur), cur[0], cur[-1]))
    return groups


def trace_curve(panel, col_start, col_end, row_seed, *, mask_boxes=(),
                 max_jump=10, row_pad_top=4, row_pad_bot=4, step=1,
                 pick="nearest", search_row_lo=None, search_row_hi=None):
    """Trace a single curve across columns [col_start,col_end] (inclusive),
    starting near row_seed, following the nearest ink cluster within max_jump
    of the previous column's traced row. mask_boxes excludes label text:
    list of (col_lo,col_hi,row_lo,row_hi) to skip when picking clusters.

    pick: 'nearest' (default, follow closest to prev) is used throughout;
    kept as a hook for future extension.
    """
    top, bot = panel_row_bounds(panel)
    if search_row_lo is None:
        search_row_lo = int(top) + row_pad_top
    if search_row_hi is None:
        search_row_hi = int(bot) - row_pad_bot

    def in_mask(col, row):
        for (clo, chi, rlo, rhi) in mask_boxes:
            if clo <= col <= chi and rlo <= row <= rhi:
                return True
        return False

    prev = row_seed
    result = []
    cols = range(col_start, col_end + 1, step) if col_end >= col_start else range(col_start, col_end - 1, -step)
    for col in cols:
        if col < 0 or col >= W:
            continue
        groups = col_groups(col, search_row_lo, search_row_hi)
        # filter out clusters fully inside a mask box
        groups = [g for g in groups if not in_mask(col, g[0])]
        if not groups:
            continue
        if prev is None:
            cand = min(groups, key=lambda g: g[0])
        else:
            near = [g for g in groups if abs(g[0] - prev) <= max_jump]
            if not near:
                continue
            cand = min(near, key=lambda g: abs(g[0] - prev))
        prev = cand[0]
        result.append((col, cand[0]))
    return result


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print("wrote", path, len(rows), "rows")
