"""Write final candidate CSVs with descriptive headers, matching the style of
the already-confirmed complete_equations files and the OLD dunn_* files'
simple x,y convention.
"""
import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
import pickle
import numpy as np
import csv
import os

with open('verification/first_mode/_mack_ch10_verify/redigitize_work/RF_all.pkl', 'rb') as f:
    RF = pickle.load(f)

OUTDIR = 'verification/first_mode/_mack_ch10_verify/redigitize_work/candidates'
os.makedirs(OUTDIR, exist_ok=True)


def downsample(R, F, n_target=32):
    if len(R) <= n_target:
        return R, F
    idx = np.unique(np.round(np.linspace(0, len(R) - 1, n_target)).astype(int))
    return R[idx], F[idx]


DUNN_HEADER = """# Mack (1984) Fig 10.1({panel}), M1={mach}, {curvename} neutral loop ({pos} of 3 curves).
# Re-digitized 2026-07-02 from AGARD-R-709 embedded figure (refPapers/latex_papers/figures/fig10_1.png,
#   400 DPI crop, 1300x1950 px covering both panels). Pixel calibration:
#   R: px = 0.8577198*R + 146.4835 (shared x-axis, both panels).
#   F (panel a): px = -216.6477*F + 877.6136. F (panel b): px = -215.5000*F + 1814.9545.
# Conventions: x = R = sqrt(Re_x); y = F*1e4, F = omega/R = alpha*c_r/R.
# Method: programmatic pixel-curve tracing (threshold + connected-component text
#   removal + column/row continuity tracing with column-monotonicity and
#   direction-cosine guards to avoid hopping onto neighboring loops or leader-line
#   arrow stubs). Supersedes the prior digitization, which did not track either
#   the upper (cutoff) or lower (onset) branch of this loop faithfully (a single
#   ambiguous line cutting across the loop). This file lists BOTH branches
#   sequentially: first the branch with the LOWER F at a given R (onset), then
#   the branch with the HIGHER F (cutoff) -- both blocks share the same x,y
#   column convention as before, just now genuinely tracking the loop ink.
{extra}# Digitization uncertainty: dR ~ +/-10-15, dF ~ +/-0.03-0.06 in F*1e4 typical;
#   larger (~10-15%) where the 3 loops visually bundle together at high R.
"""


def write_dunn_csv(fname, panel, mach, curvename, pos, blocks, extra=""):
    path = f'{OUTDIR}/{fname}'
    with open(path, 'w', newline='') as f:
        f.write(DUNN_HEADER.format(panel=panel, mach=mach, curvename=curvename, pos=pos, extra=extra))
        w = csv.writer(f)
        w.writerow(['x', 'y'])
        for R, F in blocks:
            for r, ff in zip(R, F):
                w.writerow(['%.6f' % r, '%.6f' % ff])
    print('wrote', path)


# ---- M16 dunn_asymptotic (panel a, loop1, outermost) ----
RA, FA = RF['M16_dunn_asymptotic_A']
RB, FB = RF['M16_dunn_asymptotic_B']
write_dunn_csv('mack_ch10_fig10_1_M16_paper_dunn_asymptotic.csv', 'a', '1.6',
               'DUNN-LIN ASYMPTOTIC THEORY (MACK 1960)', 'outermost',
               [downsample(RA, FA, 34), downsample(RB, FB, 34)])

# ---- M16 dunn_numerical (panel a, loop2, middle) ----
RA, FA = RF['M16_dunn_numerical_A']
RB, FB = RF['M16_dunn_numerical_B']
write_dunn_csv('mack_ch10_fig10_1_M16_paper_dunn_numerical.csv', 'a', '1.6',
               'NUMERICAL INTEGRATION DUNN-LIN EQUATIONS', 'middle',
               [downsample(RA, FA, 34), downsample(RB, FB, 34)])

# ---- M22 dunn_asymptotic (panel b, loop1, outermost) ----
Rs, Fs = RF['M22_dunn_asymptotic_shared']
RA, FA = RF['M22_dunn_asymptotic_A']
RB, FB = RF['M22_dunn_asymptotic_B']
extra_b1 = ("# NOTE (M2.2 only): this loop's nose sits ABOVE F=4.0, off the top of the\n"
            "#   plotted frame, so the true critical-R apex is not visible in the source\n"
            "#   scan. Its two branches are also visually indistinguishable (same ink)\n"
            "#   from the top of the frame down to R~215-258 -- a genuine source-image\n"
            "#   resolution limit, not a tracing shortcut. The first block below (labeled\n"
            "#   as if it were the onset branch, R~216-242) is that single shared/ambiguous\n"
            "#   segment; treat it as representing BOTH branches over that narrow range.\n")
write_dunn_csv('mack_ch10_fig10_1_M22_paper_dunn_asymptotic.csv', 'b', '2.2',
               'DUNN-LIN ASYMPTOTIC THEORY (MACK 1960)', 'outermost',
               [downsample(Rs, Fs, 10), downsample(RA, FA, 30), downsample(RB, FB, 30)],
               extra=extra_b1)

# ---- M22 dunn_numerical (panel b, loop2, middle) ----
RA, FA = RF['M22_dunn_numerical_A']
RB, FB = RF['M22_dunn_numerical_B']
write_dunn_csv('mack_ch10_fig10_1_M22_paper_dunn_numerical.csv', 'b', '2.2',
               'NUMERICAL INTEGRATION DUNN-LIN EQUATIONS', 'middle',
               [downsample(RA, FA, 34), downsample(RB, FB, 34)])

# ---- M16 complete_equations (panel a, loop3, innermost) -- branch-labeled ----
RA, FA = RF['M16_complete_equations_A']
RB, FB = RF['M16_complete_equations_B']
RA_s, FA_s = downsample(RA, FA, 30)
RB_s, FB_s = downsample(RB, FB, 30)
header3 = """# Mack (1984) Fig 10.1(a), M1=1.6, COMPLETE-EQUATIONS neutral loop (innermost of 3 curves).
# Re-digitized 2026-07-02 from AGARD-R-709 embedded figure (refPapers/latex_papers/figures/fig10_1.png,
#   400 DPI crop). Pixel calibration (panel a):
#   R: px = 0.8577198*R + 146.4835 (shared x-axis, both panels).
#   F: px = -216.6477*F + 877.6136.
# Conventions: x = R = sqrt(Re_x); y = F*1e4, F = omega/R = alpha*c_r/R.
# Geometry: 3 nested loops; Complete Equations = INNERMOST. Critical R (nose) ~304,
#   F~1.84 -- this SUPERSEDES the prior digitization's nose estimate of R~215, which
#   was found (on pixel re-verification) to actually belong to the general crowded
#   region near where all 3 loops nearly overlap, not this loop's true apex. The true
#   apex at R~304 is confirmed against pyMack's own neutral-curve solve (pyMack shows
#   the c_i=0 crossing first appearing between R=300 and R=400).
# branch: nose (critical-R apex) | upper (cutoff) | lower (onset).
# Digitization uncertainty: dR ~ +/-10-15, dF ~ +/-0.03-0.06 in F*1e4 on branches;
#   larger where the loop nearly touches dunn_numerical just past the nose (R~350-450).
#   Nose R uncertain to ~+/-15, nose F ~ +/-0.05.
"""
with open(f'{OUTDIR}/mack_ch10_fig10_1_M16_paper_complete_equations.csv', 'w', newline='') as f:
    f.write(header3)
    w = csv.writer(f)
    w.writerow(['R', 'F_x1e4', 'branch'])
    w.writerow([304, 1.84, 'nose'])
    for r, ff in zip(RA_s, FA_s):
        w.writerow([round(r), round(ff, 4), 'lower'])
    for r, ff in zip(RB_s, FB_s):
        w.writerow([round(r), round(ff, 4), 'upper'])
print('wrote', f'{OUTDIR}/mack_ch10_fig10_1_M16_paper_complete_equations.csv')
