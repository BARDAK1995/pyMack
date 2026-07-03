"""Trace panel (a) M1.3 curves: 2D (lower) and psi envelope (upper, merged
30/40/45deg line). Both curves visibly terminate near x~15 (T-bar endpoint
markers), confirmed by direct pixel inspection -- do NOT extrapolate to x=20.
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "a"
top, bot = panel_row_bounds(PANEL)  # 49, 477
SEARCH_HI = 473  # exclude bottom-frame ink (475-479)
SEARCH_LO = int(top) + 3  # exclude top-frame ink

# Label / marker mask boxes (col_lo,col_hi,row_lo,row_hi), from visual inspection:
# "(a) M1=1.3" title text, "psi=30deg" label, "40deg"/"45deg" labels, "2D" label + arrow, X marker
MASKS = [
    (220, 330, 60, 140),    # "(a) M1=1.3" title
    (300, 400, 85, 145),    # "psi = 30 deg" label text
    (990, 1060, 55, 100),   # "45deg" label text (above upper curve, near its endpoint)
    (920, 995, 55, 110),    # "40deg" label text
    (900, 1010, 230, 320),  # "2D" label text + leader arrow (approx region below lower curve)
]

def masked(col, row):
    for (clo, chi, rlo, rhi) in MASKS:
        if clo <= col <= chi and rlo <= row <= rhi:
            return True
    return False


def col_groups_masked(col, row_lo, row_hi):
    g = col_groups(col, row_lo, row_hi)
    return [x for x in g if not masked(col, x[0])]


def trace(col_start, col_end, row_seed, max_jump=9):
    prev = row_seed
    out = []
    for col in range(col_start, col_end + 1):
        g = col_groups_masked(col, SEARCH_LO, SEARCH_HI)
        if not g:
            continue
        if prev is None:
            cand = min(g, key=lambda z: z[0])
        else:
            near = [z for z in g if abs(z[0] - prev) <= max_jump]
            if not near:
                continue
            cand = min(near, key=lambda z: abs(z[0] - prev))
        prev = cand[0]
        out.append((col, cand[0]))
    return out


if __name__ == "__main__":
    # --- Upper envelope curve (psi 30/40/45 merged) ---
    # Seed near the toe where it first clearly separates upward; work backward
    # to find true liftoff, and forward to endpoint ~col 1020.
    # From scan: at col=320, top cluster row~107 already the upper curve rising.
    # Trace forward from col 305 (just past liftoff) to col 1021 (endpoint).
    seed_col = 330
    g0 = col_groups_masked(seed_col, SEARCH_LO, SEARCH_HI)
    print("seed clusters at col", seed_col, g0)

    upper_fwd = trace(seed_col, 1021, row_seed=110.0, max_jump=9)
    upper_back = trace(seed_col, 296, row_seed=110.0, max_jump=9)
    upper = sorted(upper_back + upper_fwd)
    print("upper curve n=", len(upper), "col range", upper[0][0], upper[-1][0])

    # --- Lower 2D curve ---
    lower_fwd = trace(seed_col, 1020, row_seed=190.0, max_jump=9)
    lower_back = trace(seed_col, 296, row_seed=190.0, max_jump=9)
    lower = sorted(lower_back + lower_fwd)
    print("lower curve n=", len(lower), "col range", lower[0][0], lower[-1][0])

    np.save("upper_a.npy", np.array(upper))
    np.save("lower_a.npy", np.array(lower))
