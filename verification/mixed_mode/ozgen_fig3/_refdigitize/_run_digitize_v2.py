"""Produce all four ozgen_fig3_M{N}_neutral_v2.csv files."""
import sys; sys.path.insert(0, 'verification/first_mode/_ozgen_refdigitize')
import numpy as np
from _digitize_all_v2 import (load_dark, all_cross, trace_top, trace_bot,
                              re2px, px2al, write_csv)

def col_alphas(dark, C, Re):
    c = int(round(re2px(Re, C)))
    g = all_cross(dark, c, C)
    return np.sort(np.array([px2al(x, C) for x in g])), c

# ---------------- M2 : single first-mode band ----------------
def do_M2():
    dark, C, shp = load_dark('M2')
    # nose peak
    peakRe, peakrow = None, None
    for Re in range(300, 520, 5):
        c = int(round(re2px(Re, C))); g = all_cross(dark, c, C)
        if len(g) and (peakrow is None or g.min() < peakrow):
            peakrow = g.min(); peakRe = Re
    up = trace_top(dark, C, shp, peakRe, 4980, 50, start_row=peakrow, max_jump=14)
    lo = []
    for Re in range(peakRe, 4981, 50):
        a, c = col_alphas(dark, C, Re)
        if len(a): lo.append((Re, a.min()))
    rows = [['upper', r, round(al, 4), 'first'] for r, al in up] + \
           [['lower', r, round(al, 4), 'first'] for r, al in lo]
    write_csv('M2', rows)

# ---------------- M3 : left closed lobe + right open band w/ notch ~Re2400 --
def do_M3():
    dark, C, shp = load_dark('M3')
    # The c_i=0 outermost contour has two parts:
    #  (1) a left closed lobe (Re~700-2000), and
    #  (2) a right open band with a notch near Re~2400 opening rightward.
    # Leader lines for the "c_i=0" / "0.00015" text intrude as topmost spikes,
    # so the upper branch is traced with continuity (rejects spikes).
    #
    # RIGHT-BAND upper branch: seed at clean right edge, trace leftward to notch.
    cR = int(round(re2px(4900, C))); gR = all_cross(dark, cR, C)
    up_right = trace_top(dark, C, shp, 4900, 2250, -50, start_row=gR.min(),
                         max_jump=8)
    # LEFT-LOBE top edge: the lobe's own upper boundary. Track topmost with
    # continuity from the lobe nose (~Re700) but reject leader-line spikes by
    # capping at alpha<=0.03 (the lobe never exceeds ~0.027).
    up_left = []
    prev = None
    for Re in range(700, 2050, 50):
        a, c = col_alphas(dark, C, Re)
        s = a[a <= 0.03]                      # exclude leader lines well above lobe
        if len(s) == 0:
            continue
        if prev is None:
            cand = s.max()
        else:
            # topmost lobe-edge crossing within continuity of the running value
            # (rejects "0.00015" leader-line strays that spike up near Re~1700-1800)
            near = s[np.abs(s - prev) <= 0.004]
            cand = near.max() if len(near) else s[np.argmin(np.abs(s - prev))]
        prev = cand
        up_left.append((Re, cand))
    up = sorted(up_left) + sorted(up_right)
    # LOWER branch (bottom-most envelope): left-lobe bottom + lower open branch.
    lo = []
    for Re in range(300, 4981, 50):
        a, c = col_alphas(dark, C, Re)
        if len(a): lo.append((Re, a.min()))
    rows = [['upper', r, round(al, 4), 'first'] for r, al in up] + \
           [['lower', r, round(al, 4), 'first'] for r, al in lo]
    write_csv('M3', rows)

# ---------------- M4 : two open lobes ----------------
def do_M4():
    dark, C, shp = load_dark('M4')
    # Two lobes. lobe column = branch (upper/lower). mode column = which lobe:
    #   mode='first'  -> lower (first-mode) lobe
    #   mode='second' -> upper (second-mode) lobe
    rows = []
    # LOWER (first-mode) lobe. The "c_i=0" leader line descends through the lobe
    # near the nose (Re~1400-1550, alpha~0.073-0.092). The lobe LOWER branch is
    # the bottom-most crossing (clean). The lobe UPPER branch is traced with
    # continuity from the clean right edge leftward to reject the leader line.
    lower_pts = {}
    for Re in range(300, 4981, 50):
        a, c = col_alphas(dark, C, Re)
        s = a[(a >= 0.0) & (a <= 0.085)]
        if len(s):
            lower_pts[Re] = s
    res = sorted(lower_pts)
    # lower branch = bottom-most
    for Re in res:
        rows.append(['lower', Re, round(lower_pts[Re][0], 4), 'first'])
    # upper branch = continuity trace from right
    prev = lower_pts[res[-1]][-1]
    upbr = {}
    for Re in reversed(res):
        s = lower_pts[Re]
        near = s[np.abs(s - prev) <= 0.006]
        cand = near.max() if len(near) else s[np.argmin(np.abs(s - prev))]
        prev = cand
        upbr[Re] = cand
    for Re in res:
        rows.append(['upper', Re, round(upbr[Re], 4), 'first'])
    # UPPER (second-mode) lobe: alpha in [0.28, 0.40]
    for Re in range(300, 4981, 50):
        a, c = col_alphas(dark, C, Re)
        s = a[(a >= 0.28) & (a <= 0.40)]
        if len(s):
            rows.append(['upper', Re, round(s[-1], 4), 'second'])
            rows.append(['lower', Re, round(s[0], 4), 'second'])
    write_csv('M4', rows)

# ---------------- M6 : connected band, notch ~Re850 ----------------
def do_M6():
    dark, C, shp = load_dark('M6')
    # UPPER branch (second mode): topmost crossing, reject text spikes via
    # continuity. Seed from clean Re=700.
    cS = int(round(re2px(700, C))); gS = all_cross(dark, cS, C)
    up_r = trace_top(dark, C, shp, 700, 4980, 50, start_row=gS.min(), max_jump=8)
    # Nose (Re 180-650): the "c_i=0" leader line intrudes above the band as a
    # spike up to alpha~0.23. The genuine band-top here is the topmost crossing
    # with alpha <= 0.21 (the band never exceeds ~0.205).
    up_l = []
    for Re in range(180, 700, 20):
        a, c = col_alphas(dark, C, Re)
        s = a[a <= 0.21]
        if len(s):
            up_l.append((Re, s.max()))
    up = sorted(set(up_l + up_r))
    # LOWER branch (first mode): bottom-most crossing below 0.11, from notch
    lo = []
    for Re in range(150, 4981, 50):
        a, c = col_alphas(dark, C, Re)
        s = a[a < 0.11]
        if len(s): lo.append((Re, s.min()))
    rows = [['upper', r, round(al, 4), 'second'] for r, al in up] + \
           [['lower', r, round(al, 4), 'first'] for r, al in lo]
    write_csv('M6', rows)

if __name__ == '__main__':
    do_M2(); do_M3(); do_M4(); do_M6()
