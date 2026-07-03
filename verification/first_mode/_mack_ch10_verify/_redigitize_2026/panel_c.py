"""Panel (c) M2.2: trace ONLY the 2D (lower) curve. The psi=45 upper-envelope
curve (mack_ch10_fig10_3_M22_paper_psi45.csv) was independently confirmed
accurate by the prior recheck and must NOT be touched/retraced.

The 2D curve shares its toe with the upper envelope (both start at the same
point, x~2.5) but separates almost immediately since 2D rises much more
gently. It extends the full visible range to x~20 (right frame), peaking
gently around x~7-9 then declining slowly.
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "c"
SEARCH_LO = 53
SEARCH_HI = 473

MASKS = [
    (1340, 1460, 60, 140),    # "(c) M1=2.2" title
    (1385, 1470, 340, 400),   # "45deg" label text at toe (leader tick only, small)
    (1785, 1875, 340, 430),   # "2D" leader-line arrow diagonal (curve peak itself sits at row~427-439)
    (1875, 1910, 335, 365),   # "2D" label text
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
    # seed near col 1730 (clean single cluster ~434) and trace both directions
    fwd = trace_dir(1730, 2429, 434.0, max_jump=11, direction=1)
    back = trace_dir(1729, 1478, 434.0, max_jump=11, direction=-1)
    full = sorted(back + fwd)
    print("2D:", full[0], full[-1], "n=", len(full))
    np.save("lower_c.npy", np.array(full))
