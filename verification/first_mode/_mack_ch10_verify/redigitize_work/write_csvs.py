"""Write final candidate CSVs (x,y format matching the OLD dunn_* files) for
all re-digitized curves, downsampled to a reasonable point density.
"""
import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
import pickle
import numpy as np
import csv

with open('verification/first_mode/_mack_ch10_verify/redigitize_work/RF_all.pkl', 'rb') as f:
    RF = pickle.load(f)

OUTDIR = 'verification/first_mode/_mack_ch10_verify/redigitize_work/candidates'
import os
os.makedirs(OUTDIR, exist_ok=True)


def downsample(R, F, n_target=30):
    """Pick n_target points roughly evenly spaced in R (log-ish near the
    nose where curvature is high, linear further out) -- simplest robust
    approach: evenly spaced in sqrt(R) to give more density at low R where
    the curve changes fastest, matching how the OLD files were denser at
    low R (e.g. steps of 25 below R=200, growing to 100+ above R=1000)."""
    if len(R) <= n_target:
        return R, F
    # dedupe / sort already done; sample by index evenly
    idx = np.unique(np.round(np.linspace(0, len(R) - 1, n_target)).astype(int))
    return R[idx], F[idx]


def write_xy_csv(path, blocks, fmt='%.6f'):
    """blocks: list of (R,F) arrays; concatenated in order, each internally
    sorted by increasing R (matches how complete_equations lists nose then
    lower then upper as sequential blocks)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y'])
        for R, F in blocks:
            for r, ff in zip(R, F):
                w.writerow([fmt % r, fmt % ff])


# ---- M16 dunn_asymptotic (panel a, loop1) ----
RA, FA = RF['M16_dunn_asymptotic_A']
RB, FB = RF['M16_dunn_asymptotic_B']
RA_s, FA_s = downsample(RA, FA, 35)
RB_s, FB_s = downsample(RB, FB, 35)
write_xy_csv(f'{OUTDIR}/mack_ch10_fig10_1_M16_paper_dunn_asymptotic_candidate.csv',
             [(RA_s, FA_s), (RB_s, FB_s)])

# ---- M16 dunn_numerical (panel a, loop2) ----
RA, FA = RF['M16_dunn_numerical_A']
RB, FB = RF['M16_dunn_numerical_B']
RA_s, FA_s = downsample(RA, FA, 35)
RB_s, FB_s = downsample(RB, FB, 35)
write_xy_csv(f'{OUTDIR}/mack_ch10_fig10_1_M16_paper_dunn_numerical_candidate.csv',
             [(RA_s, FA_s), (RB_s, FB_s)])

# ---- M16 complete_equations (panel a, loop3) -- branch-labeled like OLD file ----
RA, FA = RF['M16_complete_equations_A']
RB, FB = RF['M16_complete_equations_B']
RA_s, FA_s = downsample(RA, FA, 30)
RB_s, FB_s = downsample(RB, FB, 30)
with open(f'{OUTDIR}/mack_ch10_fig10_1_M16_paper_complete_equations_candidate.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['R', 'F_x1e4', 'branch'])
    w.writerow([304, 1.84, 'nose'])
    for r, ff in zip(RA_s, FA_s):
        w.writerow([round(r), round(ff, 4), 'lower'])
    for r, ff in zip(RB_s, FB_s):
        w.writerow([round(r), round(ff, 4), 'upper'])

# ---- M22 dunn_asymptotic (panel b, loop1) -- includes shared low-R prefix ----
Rs, Fs = RF['M22_dunn_asymptotic_shared']
RA, FA = RF['M22_dunn_asymptotic_A']
RB, FB = RF['M22_dunn_asymptotic_B']
Rs_s, Fs_s = downsample(Rs, Fs, 10)
RA_s, FA_s = downsample(RA, FA, 30)
RB_s, FB_s = downsample(RB, FB, 30)
write_xy_csv(f'{OUTDIR}/mack_ch10_fig10_1_M22_paper_dunn_asymptotic_candidate.csv',
             [(Rs_s, Fs_s), (RA_s, FA_s), (RB_s, FB_s)])

# ---- M22 dunn_numerical (panel b, loop2) ----
RA, FA = RF['M22_dunn_numerical_A']
RB, FB = RF['M22_dunn_numerical_B']
RA_s, FA_s = downsample(RA, FA, 30)
RB_s, FB_s = downsample(RB, FB, 30)
write_xy_csv(f'{OUTDIR}/mack_ch10_fig10_1_M22_paper_dunn_numerical_candidate.csv',
             [(RA_s, FA_s), (RB_s, FB_s)])

print('wrote candidates to', OUTDIR)
import subprocess
print(subprocess.run(['ls', '-la', OUTDIR], capture_output=True, text=True).stdout)
