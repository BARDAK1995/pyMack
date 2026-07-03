"""TASK B: overlay digitized points on the rendered panel, calibrated to the
axis box via detected tick/frame pixels. Save _verify2_M{N}.png."""
import sys; sys.path.insert(0, 'verification/first_mode/_ozgen_refdigitize')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from _digitize_all_v2 import CAL, OUT

REFDIR = "verification/first_mode/_ozgen_refdigitize"

def load_csv(M):
    fn = f"reference_data/digitized/ozgen_fig3_M{M}_neutral_v2.csv"
    rows = list(csv.DictReader(open(fn)))
    return [(r['lobe'], float(r['Re']), float(r['alpha']), r['mode']) for r in rows]

def verify(Mn):
    M = f'M{Mn}'
    C = CAL[M]
    im = np.array(Image.open(f"{OUT}/{C['file']}"))
    H, W = im.shape[:2]
    # extent: map full image pixels to data coords using axis-box calibration.
    # x: data Re = (px - left)/(right-left)*5000  ->  left edge px0 maps to Re_at_0
    # Use imshow with extent so the axis box aligns to data.
    re_at_x0 = (0 - C['left']) / (C['right'] - C['left']) * 5000.0
    re_at_x1 = (W - C['left']) / (C['right'] - C['left']) * 5000.0
    # y: alpha = (bottom - px)/(bottom-top)*amax ; row 0 (top of image) and row H
    al_at_y0 = (C['bottom'] - 0) / (C['bottom'] - C['top']) * C['amax']      # top of image
    al_at_yH = (C['bottom'] - H) / (C['bottom'] - C['top']) * C['amax']      # bottom of image
    extent = [re_at_x0, re_at_x1, al_at_yH, al_at_y0]  # left,right,bottom,top

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(im, extent=extent, aspect='auto', origin='upper')

    rows = load_csv(Mn)
    # color by (lobe,mode)
    styles = {
        ('upper', 'first'):  ('o', 'red',     'upper (1st)'),
        ('lower', 'first'):  ('o', 'blue',    'lower (1st)'),
        ('upper', 'second'): ('^', 'magenta', 'upper (2nd)'),
        ('lower', 'second'): ('^', 'cyan',    'lower (2nd)'),
    }
    seen = set()
    for lobe, Re, al, mode in rows:
        mk, col, lab = styles[(lobe, mode)]
        ax.scatter(Re, al, marker=mk, s=22, facecolors='none',
                   edgecolors=col, linewidths=1.1,
                   label=(lab if (lobe, mode) not in seen else None))
        seen.add((lobe, mode))

    ax.set_xlim(0, 5000); ax.set_ylim(0, C['amax'])
    ax.set_xlabel('Re', fontsize=14); ax.set_ylabel('alpha', fontsize=14)
    ax.set_title(f'Ozgen Fig3 M={Mn}: digitized c_i=0 over panel', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    out = f"{REFDIR}/_verify2_M{Mn}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)

if __name__ == '__main__':
    for n in [2, 3, 4, 6]:
        verify(n)
