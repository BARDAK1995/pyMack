import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
from digitize_fig10_1_dunn import *
from tracer import trace_adaptive
from scipy import ndimage
import pickle

mask = load_curve_mask('a')
r0, r1 = PANELS['a']['rows']
c0, c1 = 148, 1174

# Loop1 (dunn_asymptotic): branch A = onset (ends low F), branch B = cutoff (ends high F)
seed1A = [(36.0, 274.0), (40.0, 275.5)]
path1A = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(c0, c1), search_radius=8, seed_path=seed1A)
seed1B = [(36.0, 284.0), (40.0, 287.0)]
path1B = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(c0, c1), search_radius=8, seed_path=seed1B)

# Loop2 (dunn_numerical): branch A = onset, branch B = cutoff
seed2A = [(353.0, 339.0), (358.0, 339.0)]
path2A = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(c0, c1), search_radius=6, seed_path=seed2A)
seed2B_fwd = [(397.0, 374.0), (400.0, 377.0)]
path2B_fwd = trace_adaptive(mask, direction_sign=+1, row_dir=+1, r_bounds=(r0, r1), c_bounds=(c0, c1), search_radius=6, seed_path=seed2B_fwd)
path2B_back = trace_adaptive(mask, direction_sign=-1, row_dir=-1, r_bounds=(r0, r1), c_bounds=(c0, c1), search_radius=6, seed_path=[(400.0, 377.0), (397.0, 374.0)])
path2B = list(reversed(path2B_back)) + path2B_fwd[2:]

# Remove loop1+loop2 from mask (dilate traced pixels) to isolate loop3 + label arrows.
remove = np.zeros_like(mask)
for path in (path1A, path1B, path2A, path2B):
    for (rr, cc) in path:
        rr = int(round(rr)); cc = int(round(cc))
        remove[max(0, rr - 3):rr + 4, max(0, cc - 3):cc + 4] = True
mask_wo12 = mask & (~remove)
lbl, n = ndimage.label(mask_wo12, structure=np.ones((3, 3)))
sizes = ndimage.sum(mask_wo12, lbl, range(1, n + 1))
biggest = np.argmax(sizes) + 1
mask3 = lbl == biggest
print('loop3-isolated component size', sizes[biggest - 1], 'of', n, 'components')

c, r = find_nose(mask3, r0, r1, 340, 460)
print('loop3 nose px', c, r, 'R=', px2R(c), 'F=', px2F(r, 'a'))

with open('verification/first_mode/_mack_ch10_verify/redigitize_work/panelA_state.pkl', 'wb') as f:
    pickle.dump(dict(path1A=path1A, path1B=path1B, path2A=path2A, path2B=path2B), f)

Image.fromarray((~mask3 * 255).astype('uint8')).save(
    'verification/first_mode/_mack_ch10_verify/redigitize_work/panelA_loop3_isolated.png')
