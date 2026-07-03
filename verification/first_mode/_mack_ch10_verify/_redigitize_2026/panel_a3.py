"""Panel (a) M1.3: final tracer for 2D (lower) and psi-envelope (upper, merged
30/40/45deg) curves.

Regions (columns, established by manual pixel inspection):
  349-421   merged single line (toe -> pre-crossing), psi30 momentarily rides
            ABOVE the 2D curve here (see crossing at ~col 480, x~5.2)
  421-477   two distinct curves resolved, cluster A (smaller row=higher y) is
            psi-envelope (upper), cluster B is 2D (lower) -- wait, verified
            below: actually need per-region seed since the two curves swap
            which one is "first found" depending on local band ordering.
  477-483   crossing zone (extra ink cluster from the plotted X marker)
  483-1020  two curves cleanly separated the whole way to their endpoints
            (psi-envelope ~x=15 T-bar endpoint; 2D also ends ~x=15)

Both curves terminate at x~15 (verified: no ink beyond col~1021), so per the
task caveat about fabricated tails, we STOP at the true visible endpoint and
do not extrapolate to x=20.
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "a"
SEARCH_LO = 53
SEARCH_HI = 473

MASKS = [
    (215, 480, 75, 143),     # "(a) M1=1.3" title
    (215, 476, 165, 209),    # "psi = 30 deg" label (curves separate again by col~483)
    (725, 790, 116, 159),    # "40deg" label text
    (988, 1054, 105, 149),   # "45deg" label text
    (876, 916, 243, 280),    # "2D" label text
    (782, 852, 191, 268),    # "2D" leader-line arrow diagonal (curve itself stays <=190 here)
    (848, 878, 255, 270),    # "2D" leader-line short horizontal dash
]


def masked(col, row):
    for (clo, chi, rlo, rhi) in MASKS:
        if clo <= col <= chi and rlo <= row <= rhi:
            return True
    return False


def col_groups_masked(col, row_lo, row_hi):
    g = col_groups(col, row_lo, row_hi)
    return [x for x in g if not masked(col, x[0])]


def trace_dir(col_start, col_end, row_seed, max_jump, direction=1,
              pick="nearest"):
    prev = row_seed
    out = []
    col = col_start
    while (direction == 1 and col <= col_end) or (direction == -1 and col >= col_end):
        g = col_groups_masked(col, SEARCH_LO, SEARCH_HI)
        if g:
            near = [z for z in g if abs(z[0] - prev) <= max_jump]
            if near:
                if pick == "nearest":
                    cand = min(near, key=lambda z: abs(z[0] - prev))
                elif pick == "topmost":
                    cand = min(near, key=lambda z: z[0])
                elif pick == "botmost":
                    cand = max(near, key=lambda z: z[0])
                prev = cand[0]
                out.append((col, cand[0]))
        col += direction
    return out


if __name__ == "__main__":
    # --- merged toe: single line, col 349 -> 421 ---
    toe = trace_dir(349, 421, 469.5, max_jump=25, direction=1)
    print("toe:", toe[0], toe[-1], "n=", len(toe))

    # --- upper (psi-envelope) curve ---
    # Seed just past the crossing zone at col 497 (clean single cluster ~192.5)
    # and trace forward to endpoint (~1021) and backward through the crossing
    # to the toe-merge boundary (~col 421).
    upper_fwd = trace_dir(497, 1021, 192.5, max_jump=8, direction=1)
    upper_back = trace_dir(496, 422, 192.5, max_jump=10, direction=-1)
    upper = sorted(upper_back + upper_fwd)
    print("upper:", upper[0], upper[-1], "n=", len(upper))

    # --- lower (2D) curve ---
    lower_fwd = trace_dir(497, 1020, 227.5, max_jump=8, direction=1)
    lower_back = trace_dir(496, 422, 227.5, max_jump=10, direction=-1)
    lower = sorted(lower_back + lower_fwd)
    print("lower:", lower[0], lower[-1], "n=", len(lower))

    np.save("toe_a.npy", np.array(toe))
    np.save("upper_a.npy", np.array(upper))
    np.save("lower_a.npy", np.array(lower))
