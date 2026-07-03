"""Panel (b) M1.6: trace 2D (lower) and psi-envelope (upper, merged 45/50/55)
curves. Both curves extend all the way to the right frame (x=20) -- confirmed
by direct pixel inspection (curve ink present at col 1284-1286, right at the
frame). The "INVISCID psi=60deg" dashed line sits well below the 2D curve
near the bottom of the panel and is explicitly excluded (not one of our
target curves).
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "b"
SEARCH_LO = 530  # just below top frame (527-529)
SEARCH_HI = 958  # just above bottom frame (960-964)

# Label masks (col_lo,col_hi,row_lo,row_hi), located via zoomed crops.
MASKS = [
    (178, 425, 558, 628),    # "(b) M1=1.6" title
    (340, 465, 632, 682),    # "psi = 45 deg" label
    (150, 330, 880, 945),    # "30deg" label text + leader line near toe
    (695, 765, 600, 655),    # "50deg" label text
    (925, 995, 600, 655),    # "55deg" label text (interior, near x~14)
    (1195, 1255, 522, 570),  # "55deg" label text at far right (leader to upper curve endpoint)
    (825, 955, 700, 800),    # "2D" leader-line arrow + text (curve itself stays row>=780 here)
    (1090, 1290, 858, 950),  # "INVISCID psi=60deg" dashed line + text (NOT a target curve)
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
    # merged toe: single line col ~338 -> ~352
    toe = trace_dir(334, 352, 956.0, max_jump=15, direction=1)
    print("toe", toe[0] if toe else None, toe[-1] if toe else None, "n=", len(toe))

    # upper envelope: seed near col 400 (clean single cluster ~807.5 pre-mask check)
    upper_fwd = trace_dir(400, 1286, 807.5, max_jump=17, direction=1)
    upper_back = trace_dir(399, 353, 807.5, max_jump=12, direction=-1)
    upper = sorted(upper_back + upper_fwd)
    print("upper", upper[0], upper[-1], "n=", len(upper))

    # lower 2D: seed near col 400 cluster ~864.5
    lower_fwd = trace_dir(400, 1286, 864.5, max_jump=17, direction=1)
    lower_back = trace_dir(399, 353, 864.5, max_jump=12, direction=-1)
    lower = sorted(lower_back + lower_fwd)
    print("lower", lower[0], lower[-1], "n=", len(lower))

    np.save("toe_b.npy", np.array(toe))
    np.save("upper_b.npy", np.array(upper))
    np.save("lower_b.npy", np.array(lower))
