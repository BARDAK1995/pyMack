"""Full re-digitization of Mack (1984) Fig 10.1 (a) M1.6 and (b) M2.2 nested
neutral loops: dunn_asymptotic (outer), dunn_numerical (middle), complete_equations
(inner, panel a only -- panel b complete_equations is confirmed-good, untouched).

Produces per-branch pixel traces -> data-coordinate CSV candidates + an overlay
plot for visual QA.
"""
import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
from digitize_fig10_1_dunn import *  # noqa
from tracer import trace_adaptive
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C0, C1 = 148, 1174


def simple_fill(mask, c0, c1, seed_row, search_r=6):
    """Simple per-column nearest-neighbor fill for short, well-separated,
    near-horizontal-in-row-space segments (no adaptive row/col switching
    needed)."""
    def groups_in_col(c, rlo, rhi):
        col = mask[rlo:rhi, c]
        rows = np.where(col)[0] + rlo
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

    out = []
    prev = seed_row
    for c in range(c0, c1 + 1):
        g = np.array(groups_in_col(c, int(prev - search_r), int(prev + search_r + 1)))
        if len(g) == 0:
            continue
        cand = g[np.argmin(np.abs(g - prev))]
        out.append((cand, float(c)))
        prev = cand
    return out


def trace_panel(panel):
    mask = load_curve_mask(panel)
    r0, r1 = PANELS[panel]['rows']

    # ---- Loop 1 (dunn_asymptotic): nose = topmost ink pixel overall ----
    nose1_c, nose1_r = find_nose(mask, r0, r1, 200, 320)

    # Seeds found by inspecting row-groups just past the nose cusp (method
    # validated interactively; column offsets are small and robust across
    # both panels since the nose geometry is qualitatively identical).
    if panel == 'a':
        seed1A = [(36.0, 274.0), (40.0, 275.5)]
        seed1B = [(36.0, 284.0), (40.0, 287.0)]
    else:
        # panel b: locate analogous seeds dynamically (see main() calibration probe)
        seed1A, seed1B = PANEL_B_SEEDS['loop1']

    path1A = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=8, seed_path=seed1A)
    path1B = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=8, seed_path=seed1B)

    result = dict(loop1=dict(A=path1A, B=path1B), nose1=(nose1_r, nose1_c))

    # ---- Loop 2 (dunn_numerical): isolate by removing loop1 ----
    remove = np.zeros_like(mask)
    for path in (path1A, path1B):
        for (rr, cc) in path:
            rr = int(round(rr)); cc = int(round(cc))
            remove[max(0, rr - 3):rr + 4, max(0, cc - 3):cc + 4] = True
    mask_wo1 = mask & (~remove)
    lbl, n = ndimage.label(mask_wo1, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask_wo1, lbl, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    mask_loop2plus = lbl == biggest

    if panel == 'a':
        nose2_c, nose2_r = find_nose(mask_loop2plus, r0, r1, 300, 420)
        seed2A = [(353.0, 339.0), (358.0, 339.0)]
        seed2B_fwd = [(397.0, 374.0), (400.0, 377.0)]
        seed2B_back_start = [(400.0, 377.0), (397.0, 374.0)]
    else:
        nose2_c, nose2_r = find_nose(mask_loop2plus, r0, r1, *PANEL_B_SEEDS['loop2_nose_search'])
        seed2A, seed2B_fwd, seed2B_back_start = PANEL_B_SEEDS['loop2']

    path2A = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=6, seed_path=seed2A)
    path2B_fwd = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=6, seed_path=seed2B_fwd)
    path2B_back = trace_adaptive(mask, direction_sign=-1, row_dir=-1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=6, seed_path=seed2B_back_start)
    path2B = list(reversed(path2B_back)) + path2B_fwd[2:]

    result['loop2'] = dict(A=path2A, B=path2B)
    result['nose2'] = (nose2_r, nose2_c)

    if panel == 'a':
        # ---- Loop 3 (complete_equations, panel a only) ----
        remove2 = np.zeros_like(mask)
        for path in (path1A, path1B, path2A, path2B):
            for (rr, cc) in path:
                rr = int(round(rr)); cc = int(round(cc))
                remove2[max(0, rr - 3):rr + 4, max(0, cc - 3):cc + 4] = True
        mask_wo12 = mask & (~remove2)
        lbl3, n3 = ndimage.label(mask_wo12, structure=np.ones((3, 3)))
        sizes3 = ndimage.sum(mask_wo12, lbl3, range(1, n3 + 1))
        biggest3 = np.argmax(sizes3) + 1
        mask_loop3 = lbl3 == biggest3
        nose3_c, nose3_r = find_nose(mask_loop3, r0, r1, 340, 460)

        seed3A = [(488.5, 427.0), (488.5, 429.0)]
        seed3B = [(528.5, 427.0), (528.5, 429.0)]
        path3A_far = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=6, seed_path=seed3A)
        path3B_far = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(C0, C1), search_radius=6, seed_path=seed3B)
        near3A = simple_fill(mask, 407, 427, 483.5, search_r=5)
        near3B = simple_fill(mask, 407, 427, 503.5, search_r=5)
        # simple_fill returns (row,col); adaptive returns (row,col) too -> concat
        path3A = near3A + path3A_far[2:]
        path3B = near3B + path3B_far[2:]

        result['loop3'] = dict(A=path3A, B=path3B)
        result['nose3'] = (nose3_r, nose3_c)

    return result


# Panel B seed bank -- filled in interactively below and hardcoded once found.
PANEL_B_SEEDS = {}


def path_to_RF(path, panel):
    rows = np.array([p[0] for p in path])
    cols = np.array([p[1] for p in path])
    R = px2R(cols)
    F = px2F(rows, panel)
    order = np.argsort(R)
    return R[order], F[order]


if __name__ == '__main__':
    pass
