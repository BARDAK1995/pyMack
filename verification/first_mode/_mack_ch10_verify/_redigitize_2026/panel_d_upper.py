"""Panel (d) M3.0: trace psi=60deg envelope (upper) curve only (2D already
handled in panel_d.py). Curve rises from toe near x~0 to plateau, continuing
through "65deg" labels (x3) then joining the INVISCID psi=65deg dashed
asymptote via a short diagonal connector near the right edge -- we trace the
solid curve only, stopping where it visibly becomes the dashed line /
connector (do not trace the dashed inviscid asymptote itself).
"""
import sys
sys.path.insert(0, r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_redigitize_2026")
from trace_fig10_3 import *
import numpy as np

PANEL = "d"
SEARCH_LO = 530
SEARCH_HI = 962

MASKS = [
    (1355, 1605, 545, 610),   # "(d) M1=3.0" title
    (1590, 1660, 662, 715),   # "psi = 60 deg" label (curve dips to row~645-659 here, stays clear above 662)
    (1360, 1425, 892, 924),   # "50deg" label at toe
    (1855, 1935, 613, 660),   # "65deg" label (interior, first; curve itself sits ~592-596, above this)
    (2085, 2170, 613, 660),   # "65deg" label (interior, second; curve itself sits ~598-604, above this)
    (2360, 2427, 616, 700),   # "65deg" label (far right) + its own tiny leader arrow (curve stays ~608, above this)
    (1900, 2440, 855, 962),   # "INVISCID psi=65deg" text + dashed line (well below the psi60 curve here)
    (1650, 2440, 565, 580),   # "INVISCID psi=65deg" dashed asymptote line itself (row~572-576, NOT a target curve)
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
    fwd = trace_dir(1550, 2319, 691.5, max_jump=10, direction=1)
    back = trace_dir(1549, 1341, 691.5, max_jump=10, direction=-1)

    # col 2320-2426: the solid curve plateaus flat at row~607-609 while a
    # diagonal connector (annotation showing convergence to the dashed
    # INVISCID psi=65deg asymptote) crosses through this zone above it and
    # transiently merges into the same ink blob. Take the BOTTOM-most
    # (largest-row) cluster edge here to stay on the flat plateau, not the
    # diagonal connector.
    tail = []
    for col in range(2320, 2427):
        g = col_groups_masked(col, 530, 962)
        if not g:
            continue
        cand = max(g, key=lambda z: z[0])  # bottommost = flat plateau
        if 595 <= cand[0] <= 615:
            tail.append((col, cand[0]))

    full = sorted(back + fwd + tail)
    print("upper:", full[0], full[-1], "n=", len(full))
    np.save("upper_d.npy", np.array(full))
