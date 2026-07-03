"""Panel (d) M3.0: trace 2D (lower) and psi=60deg envelope (upper) curves.
Prior check flagged the OLD 2D digitization as ~4-5x too high -- a serious
error, so this curve needs especially careful re-tracing. The 2D curve stays
extremely flat/near-zero for x~2-6 (barely distinguishable from the bottom
frame line) before gently rising through the rest of the range to x=20.

The "INVISCID psi=65deg" dashed asymptote + its diagonal leader line (joining
the solid psi=60 curve's plateau to the dashed line further right) must be
excluded -- it is NOT one of our target curves.
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "d"
SEARCH_LO = 530
SEARCH_HI = 964  # extended to capture the 2D curve where it nearly coincides
                 # with the bottom-frame ink (row 960-964) for x~0-6

MASKS = [
    (1300, 1420, 530, 610),   # "(d) M1=3.0" title
    (1360, 1425, 892, 924),   # "50deg" label text at toe
    (1420, 1500, 640, 700),   # "psi = 60 deg" label
    (1740, 1820, 858, 944),   # "2D" leader-line arrow (curve stays >=945 here, verified)
    (1822, 1858, 855, 882),   # "2D" label text
]


def masked(col, row):
    for (clo, chi, rlo, rhi) in MASKS:
        if clo <= col <= chi and rlo <= row <= rhi:
            return True
    return False


def col_groups_masked(col, row_lo, row_hi):
    g = col_groups(col, row_lo, row_hi)
    return [x for x in g if not masked(col, x[0])]


def trace_dir(col_start, col_end, row_seed, max_jump, direction=1):
    prev = row_seed
    out = []
    col = col_start
    while (direction == 1 and col <= col_end) or (direction == -1 and col >= col_end):
        g = col_groups_masked(col, SEARCH_LO, SEARCH_HI)
        if g:
            near = [z for z in g if abs(z[0] - prev) <= max_jump]
            if near:
                cand = min(near, key=lambda z: abs(z[0] - prev))
                prev = cand[0]
                out.append((col, cand[0]))
        col += direction
    return out


if __name__ == "__main__":
    # 2D curve stays flat at row~960.5 (y~0, indistinguishable from the bottom
    # frame + "50deg" leader-line clutter) from x~0 to x~2.1 (col 1342-1455).
    # Take that flat stretch directly (verified by direct pixel inspection --
    # the only ink there besides the frame is the "50deg" text/leader, which
    # sits well above row 900-924, clear of the row~960.5 baseline).
    flat = [(c, 960.5) for c in range(1342, 1456)]

    # From col 1456 on the curve is cleanly resolved and rises gradually.
    lower_fwd = trace_dir(1456, 2429, 960.5, max_jump=10, direction=1)
    lower = sorted(flat + lower_fwd)
    print("lower", lower[0], lower[-1], "n=", len(lower))
    np.save("lower_d.npy", np.array(lower))
