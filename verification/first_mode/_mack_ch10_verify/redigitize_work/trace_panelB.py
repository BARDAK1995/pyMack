import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
from digitize_fig10_1_dunn import *
from tracer2 import trace_arclength
from scipy import ndimage
import pickle

mask = load_curve_mask('b')
r0, r1 = PANELS['b']['rows']
c0, c1 = 148, 1174

# ---- Loop1 (dunn_asymptotic) cutoff (upper) branch ----
seedU_fwd = [(1614.0, 598.0), (1615.0, 600.0), (1618.0, 604.0)]
seedU_back = [(1618.0, 604.0), (1615.0, 600.0), (1614.0, 598.0)]
pU_fwd = trace_arclength(mask, (r0, r1), (c0, c1), seedU_fwd, step=1.5, perp_radius=6, baseline=6)
pU_back = trace_arclength(mask, (r0, r1), (c0, c1), seedU_back, step=1.5, perp_radius=6, baseline=6)
path1_cutoff = list(reversed(pU_back)) + pU_fwd[3:]

# ---- Loop1 onset (lower) branch ----
seedL_back = [(1638.0, 604.0), (1638.0, 600.0), (1636.5, 598.0)]
pL_back = trace_arclength(mask, (r0, r1), (c0, c1), seedL_back, step=1.5, perp_radius=6, baseline=6)
seedL2 = [(1673.0, 720.0), (1675.0, 726.0), (1676.5, 730.0), (1678.0, 734.0)]
pL2 = trace_arclength(mask, (r0, r1), (c0, c1), seedL2, step=1.5, perp_radius=6, baseline=6)
path1_onset = list(reversed(pL_back)) + pL2

print('loop1 cutoff', len(path1_cutoff), 'R range',
      px2R(min(p[1] for p in path1_cutoff)), px2R(max(p[1] for p in path1_cutoff)))
print('loop1 onset', len(path1_onset), 'R range',
      px2R(min(p[1] for p in path1_onset)), px2R(max(p[1] for p in path1_onset)))

# ---- Isolate loop2+loop3 by removing loop1's dilated pixels ----
remove = np.zeros_like(mask)
for path in (path1_cutoff, path1_onset):
    for (rr, cc) in path:
        rr = int(round(rr)); cc = int(round(cc))
        remove[max(0, rr - 3):rr + 4, max(0, cc - 3):cc + 4] = True
mask_wo1 = mask & (~remove)
lbl, n = ndimage.label(mask_wo1, structure=np.ones((3, 3)))
sizes = ndimage.sum(mask_wo1, lbl, range(1, n + 1))
biggest = np.argmax(sizes) + 1
mask_loop23 = lbl == biggest
Image.fromarray((~mask_loop23 * 255).astype('uint8')).save(
    'verification/first_mode/_mack_ch10_verify/redigitize_work/panelB_loop23_isolated.png')

nose2_c, nose2_r = find_nose(mask_loop23, r0, r1, 300, 400)
print('loop2 nose', nose2_c, nose2_r, 'R=', px2R(nose2_c), 'F=', px2F(nose2_r, 'b'))

with open('verification/first_mode/_mack_ch10_verify/redigitize_work/panelB_state.pkl', 'wb') as f:
    pickle.dump(dict(path1_cutoff=path1_cutoff, path1_onset=path1_onset), f)
