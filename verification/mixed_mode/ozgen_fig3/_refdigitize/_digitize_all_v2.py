"""Digitize Ozgen & Kircali (2008) Fig 3 c_i=0 NEUTRAL curves for M=2,3,4,6.
Verified panel clips/axis ranges (per task spec). Per-panel calibration via
auto-detected axis-frame pixels. Outputs ozgen_fig3_M{N}_neutral_v2.csv.
"""
import numpy as np
from PIL import Image
import csv, os

OUT = "verification/first_mode/_ozgen_refdigitize/_myrender"
DIG = "reference_data/digitized"
os.makedirs(DIG, exist_ok=True)

# Axis-box pixel bounds auto-detected from the rendered panels (frame lines).
CAL = {
 'M2': dict(file='panel_M2_b.png', left=180.5, right=1254.5, top=140.5, bottom=757.5, amax=0.08),
 'M3': dict(file='panel_M3_c.png', left=280.5, right=1353.5, top=128.5, bottom=744.0, amax=0.08),
 'M4': dict(file='panel_M4_d.png', left=179.5, right=1255.5, top=128.5, bottom=744.0, amax=0.4),
 'M6': dict(file='panel_M6_e.png', left=278.0, right=1355.5, top=120.5, bottom=737.5, amax=0.4),
}

def re2px(re, C): return C['left'] + re/5000.0*(C['right']-C['left'])
def px2al(r, C):  return (C['bottom']-r)/(C['bottom']-C['top'])*C['amax']

def load_dark(M):
    C = CAL[M]
    im = np.array(Image.open(f"{OUT}/{C['file']}").convert('L'))
    return im < 110, C, im.shape

def all_cross(dark, c, C, padtop=6, padbot=12):
    """Return row positions (top->bottom) of dark-pixel groups in column c,
    strictly inside the axis box (frame lines excluded)."""
    col = dark[:, c]
    rows = np.where(col)[0]
    rows = rows[(rows > C['top']+padtop) & (rows < C['bottom']-padbot)]
    groups = []
    if len(rows):
        cur = [rows[0]]
        for x in rows[1:]:
            if x-cur[-1] <= 3: cur.append(x)
            else: groups.append(np.mean(cur)); cur = [x]
        groups.append(np.mean(cur))
    return np.array(groups)

def trace_top(dark, C, shp, Re0, Re1, step, start_row, max_jump=14, padtop=6, padbot=12):
    """Follow the top-most (highest-alpha) crossing with continuity to avoid
    text/leader-line artifacts that sit inside the band."""
    res = []; prev = start_row
    for Re in range(Re0, Re1+(1 if step > 0 else -1), step):
        c = int(round(re2px(Re, C))); c = min(max(c, 0), shp[1]-1)
        g = all_cross(dark, c, C, padtop, padbot)
        if len(g) == 0: continue
        if prev is None:
            cand = g.min()
        else:
            near = g[np.abs(g-prev) <= max_jump]
            if len(near) == 0: continue
            cand = near.min()
        prev = cand
        res.append((Re, round(px2al(cand, C), 4)))
    return res

def trace_bot(dark, C, shp, Re0, Re1, step, start_row, max_jump=20, padtop=6, padbot=12):
    """Follow the bottom-most (lowest-alpha) crossing with continuity."""
    res = []; prev = start_row
    for Re in range(Re0, Re1+(1 if step > 0 else -1), step):
        c = int(round(re2px(Re, C))); c = min(max(c, 0), shp[1]-1)
        g = all_cross(dark, c, C, padtop, padbot)
        if len(g) == 0: continue
        if prev is None:
            cand = g.max()
        else:
            near = g[np.abs(g-prev) <= max_jump]
            if len(near) == 0: continue
            cand = near.max()
        prev = cand
        res.append((Re, round(px2al(cand, C), 4)))
    return res

def write_csv(M, rows):
    fn = f"{DIG}/ozgen_fig3_M{M[1:]}_neutral_v2.csv"
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['lobe', 'Re', 'alpha', 'mode'])
        for r in rows: w.writerow(r)
    print("wrote", fn, len(rows), "rows")
    return fn
