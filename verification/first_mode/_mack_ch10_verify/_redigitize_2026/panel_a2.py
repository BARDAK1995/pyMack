"""Panel (a) M1.3: trace 2D (lower) and psi-envelope (upper) curves.

Strategy: continuity-based nearest-cluster tracking, seeded near the toe
(col~349, merged single curve), moving rightward. Once the ink splits into two
clusters with a visible gap we track upper/lower independently. Label-text
mask boxes (measured from zoomed crops) exclude text so it cannot hijack the
continuity tracker. The X/+ point markers and 2D leader-line arrowhead sit ON
or immediately touching the curves so they are NOT masked (harmless, they are
single-column blips within max_jump).
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "a"
SEARCH_LO = 53   # just below top frame ink (49-51)
SEARCH_HI = 473  # just above bottom frame ink (475-479)

# Label mask boxes (col_lo,col_hi,row_lo,row_hi) -- measured off zoomed crops,
# padded generously; all sit clearly above/away from the curve ink rows.
MASKS = [
    (215, 480, 75, 143),     # "(a) M1=1.3" title
    (215, 476, 165, 209),    # "psi = 30 deg" label (curves cross around col 478-483 -- keep clear)
    (895, 1000, 105, 155),   # "40deg" label text
    (985, 1075, 155, 210),   # "45deg" label text
    (940, 1030, 240, 300),   # "2D" label text
    (782, 1010, 205, 473),   # "2D" leader-line arrow diagonal + text (below the real 2D curve, which stays <=~200 here)
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
    # Merged toe region: single curve from col 349 to ~419 (last col before
    # split becomes clearly resolved i.e. two groups >=8px apart consistently)
    toe = trace_dir(349, 419, 465.0, max_jump=12, direction=1)
    print("toe n=", len(toe), toe[:5], toe[-5:])

    # Upper branch: seed at col 424 near row ~92-96 (top cluster observed)
    upper_fwd = trace_dir(424, 1021, 94.0, max_jump=8, direction=1)
    print("upper_fwd n=", len(upper_fwd), "range", upper_fwd[0], upper_fwd[-1])

    # Lower (2D) branch: seed at col 424 near row ~120
    lower_fwd = trace_dir(424, 1020, 120.0, max_jump=8, direction=1)
    print("lower_fwd n=", len(lower_fwd), "range", lower_fwd[0], lower_fwd[-1])

    np.save("toe_a.npy", np.array(toe))
    np.save("upper_a.npy", np.array(upper_fwd))
    np.save("lower_a.npy", np.array(lower_fwd))
