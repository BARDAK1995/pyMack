"""Final assembly: trace every branch for panels a (M1.6) and b (M2.2),
convert to data coordinates, write candidate CSVs, and render the QA overlay.
"""
import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
from digitize_fig10_1_dunn import *  # noqa
from tracer import trace_adaptive
from tracer2 import trace_arclength
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C0, C1 = 148, 1174
OUTDIR = 'verification/first_mode/_mack_ch10_verify/redigitize_work'


def simple_fill(mask, c0, c1, seed_row, search_r=5):
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


def path_to_RF(path, panel):
    rows = np.array([p[0] for p in path], dtype=float)
    cols = np.array([p[1] for p in path], dtype=float)
    R = px2R(cols)
    F = px2F(rows, panel)
    order = np.argsort(R)
    R, F = R[order], F[order]
    # de-duplicate near-identical R (keep first)
    keep = np.ones(len(R), dtype=bool)
    last = -1e9
    for i in range(len(R)):
        if R[i] - last < 0.5:
            keep[i] = False
        else:
            last = R[i]
    return R[keep], F[keep]


# =============================================================================
# PANEL A (M1.6)
# =============================================================================
maskA = load_curve_mask('a')
r0a, r1a = PANELS['a']['rows']

# Loop1 (dunn_asymptotic)
seed1A = [(36.0, 274.0), (40.0, 275.5)]
path1A = trace_adaptive(maskA, direction_sign=+1, row_dir=+1, r_bounds=(r0a, r1a), c_bounds=(C0, C1), search_radius=8, seed_path=seed1A)
seed1B = [(36.0, 284.0), (40.0, 287.0)]
path1B = trace_adaptive(maskA, direction_sign=+1, row_dir=+1, r_bounds=(r0a, r1a), c_bounds=(C0, C1), search_radius=8, seed_path=seed1B)

# Loop2 (dunn_numerical)
seed2A = [(353.0, 339.0), (358.0, 339.0)]
path2A = trace_adaptive(maskA, direction_sign=+1, row_dir=+1, r_bounds=(r0a, r1a), c_bounds=(C0, C1), search_radius=6, seed_path=seed2A)
seed2B_fwd = [(397.0, 374.0), (400.0, 377.0)]
path2B_fwd = trace_adaptive(maskA, direction_sign=+1, row_dir=+1, r_bounds=(r0a, r1a), c_bounds=(C0, C1), search_radius=6, seed_path=seed2B_fwd)
path2B_back = trace_adaptive(maskA, direction_sign=-1, row_dir=-1, r_bounds=(r0a, r1a), c_bounds=(C0, C1), search_radius=6, seed_path=[(400.0, 377.0), (397.0, 374.0)])
path2B = list(reversed(path2B_back)) + path2B_fwd[2:]

# Loop3 (complete_equations, panel a only -- re-digitize per task).
# NOTE: loop3's two branches sit extremely close to loop2's cutoff branch
# just past the nose (R~350-450, all three ink lines within ~20px), which
# caused earlier attempts to hijack onto loop2 immediately after the nose.
# These seeds (found via careful manual column/row-group inspection past
# the nose, ~col 384/408 at row 480) and trace_arclength's cosine-similarity
# continuity check (no column-monotonicity constraint needed this close to
# the nose, since the true curve briefly has near-zero/slightly-negative
# d(col)/d(row)) reliably stay on loop3 for its full visible extent.
seed3A = [(480.0, 384.0), (488.0, 380.5), (494.0, 380.5)]
path3A = trace_arclength(maskA, (r0a, r1a), (C0, C1), seed3A, step=1.0, perp_radius=4, baseline=8, col_monotonic=None)
seed3B_near = [(480.0, 408.0), (488.0, 403.0), (494.0, 400.5)]
path3B_near = trace_arclength(maskA, (r0a, r1a), (C0, C1), seed3B_near, step=1.0, perp_radius=4, baseline=8, col_monotonic=None)
seed3B_far = [(715.0, 681.0), (717.0, 685.0), (717.0, 689.0)]
path3B_far = trace_arclength(maskA, (r0a, r1a), (C0, C1), seed3B_far, step=1.0, perp_radius=4, baseline=8, col_monotonic='increasing', col_backlash=2)
path3B = path3B_near + path3B_far

RF = {}
RF['M16_dunn_asymptotic_A'] = path_to_RF(path1A, 'a')
RF['M16_dunn_asymptotic_B'] = path_to_RF(path1B, 'a')
RF['M16_dunn_numerical_A'] = path_to_RF(path2A, 'a')
RF['M16_dunn_numerical_B'] = path_to_RF(path2B, 'a')
RF['M16_complete_equations_A'] = path_to_RF(path3A, 'a')
RF['M16_complete_equations_B'] = path_to_RF(path3B, 'a')

# =============================================================================
# PANEL B (M2.2)
# =============================================================================
maskB = load_curve_mask('b')
r0b, r1b = PANELS['b']['rows']

# Loop1 (dunn_asymptotic): cutoff (upper) branch
seedU_fwd = [(1614.0, 598.0), (1615.0, 600.0), (1618.0, 604.0)]
seedU_back = [(1618.0, 604.0), (1615.0, 600.0), (1614.0, 598.0)]
pU_fwd = trace_arclength(maskB, (r0b, r1b), (C0, C1), seedU_fwd, step=1.5, perp_radius=6, baseline=6, col_monotonic='increasing', col_backlash=3)
pU_back = trace_arclength(maskB, (r0b, r1b), (C0, C1), seedU_back, step=1.5, perp_radius=6, baseline=6, col_monotonic='decreasing', col_backlash=3)
path1_cutoff = list(reversed(pU_back)) + pU_fwd[3:]

# Loop1 onset (lower) branch
seedL_back = [(1638.0, 604.0), (1638.0, 600.0), (1636.5, 598.0)]
pL_back = trace_arclength(maskB, (r0b, r1b), (C0, C1), seedL_back, step=1.5, perp_radius=6, baseline=6, col_monotonic='decreasing', col_backlash=3)
seedL2 = [(1673.0, 720.0), (1675.0, 726.0), (1676.5, 730.0), (1678.0, 734.0)]
pL2 = trace_arclength(maskB, (r0b, r1b), (C0, C1), seedL2, step=1.5, perp_radius=6, baseline=6, col_monotonic='increasing', col_backlash=3)
path1_onset = list(reversed(pL_back)) + pL2

# Loop2 (dunn_numerical) onset branch
seed2onset = [(1316.0, 327.0), (1318.0, 325.5), (1322.0, 324.0)]
p2onset = trace_arclength(maskB, (r0b, r1b), (C0, C1), seed2onset, step=1.5, perp_radius=6, baseline=6, col_monotonic='increasing', col_backlash=3)

# Loop2 cutoff branch: nose segment + two continuations past crossing zones
seed2cutoff_near = [(1316.0, 334.0), (1318.0, 337.0), (1322.0, 339.0)]
p2cutoff_near = trace_arclength(maskB, (r0b, r1b), (C0, C1), seed2cutoff_near, step=1.5, perp_radius=6, baseline=6, col_monotonic='increasing', col_backlash=3)
seed2cutoff_mid = [(1438.0, 399.0), (1444.0, 402.0), (1450.0, 404.0)]
p2cutoff_mid = trace_arclength(maskB, (r0b, r1b), (C0, C1), seed2cutoff_mid, step=1.5, perp_radius=6, baseline=6, col_monotonic='increasing', col_backlash=3)
seed2cutoff_far = [(1534.0, 419.5), (1540.0, 420.5), (1546.0, 433.0)]
p2cutoff_far = trace_arclength(maskB, (r0b, r1b), (C0, C1), seed2cutoff_far, step=1.5, perp_radius=6, baseline=6, col_monotonic='increasing', col_backlash=3)
path2_cutoff = p2cutoff_near + p2cutoff_mid[3:] + p2cutoff_far[3:]

RF['M22_dunn_asymptotic_A'] = path_to_RF(path1_onset, 'b')
RF['M22_dunn_asymptotic_B'] = path_to_RF(path1_cutoff, 'b')
RF['M22_dunn_numerical_A'] = path_to_RF(p2onset, 'b')
RF['M22_dunn_numerical_B'] = path_to_RF(path2_cutoff, 'b')

for k, (R, F) in RF.items():
    print(k, 'n=', len(R), 'R range', R.min(), R.max())

import pickle
with open(f'{OUTDIR}/RF_all.pkl', 'wb') as f:
    pickle.dump(RF, f)
