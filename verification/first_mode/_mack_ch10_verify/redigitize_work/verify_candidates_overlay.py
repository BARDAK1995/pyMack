import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
from digitize_fig10_1_dunn import *
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAND = 'verification/first_mode/_mack_ch10_verify/redigitize_work/candidates'


def load_xy(path):
    R, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('x,y') or not line.strip():
                continue
            a, b = line.strip().split(',')
            R.append(float(a)); F.append(float(b))
    return R, F


def load_branch_csv(path):
    R, F = [], []
    with open(path) as f:
        rdr = csv.reader(l for l in f if not l.startswith('#'))
        header = next(rdr)
        for row in rdr:
            R.append(float(row[0])); F.append(float(row[1]))
    return R, F


im = Image.open(SRC)
fig, axes = plt.subplots(2, 1, figsize=(15, 22))

ax = axes[0]
ax.imshow(im.crop((0, 0, 1300, 900)), cmap='gray', extent=[0, 1300, 900, 0])
for fname, color, label in [
    ('mack_ch10_fig10_1_M16_paper_dunn_asymptotic.csv', 'red', 'dunn_asymptotic'),
    ('mack_ch10_fig10_1_M16_paper_dunn_numerical.csv', 'blue', 'dunn_numerical'),
]:
    R, F = load_xy(f'{CAND}/{fname}')
    px = R2px(np.array(R)); py = PANELS['a']['fy_m'] * np.array(F) + PANELS['a']['fy_b']
    ax.scatter(px, py, s=14, c=color, label=label, zorder=5)
R, F = load_branch_csv(f'{CAND}/mack_ch10_fig10_1_M16_paper_complete_equations.csv')
px = R2px(np.array(R)); py = PANELS['a']['fy_m'] * np.array(F) + PANELS['a']['fy_b']
ax.scatter(px, py, s=14, c='green', label='complete_equations (NEW)', zorder=5)
ax.set_title('Panel (a) M1.6 -- FINAL CANDIDATE CSVs (downsampled)')
ax.legend(loc='upper right')

ax = axes[1]
ax.imshow(im.crop((0, 950, 1300, 1820)), cmap='gray', extent=[0, 1300, 1820 - 950, 0])
for fname, color, label in [
    ('mack_ch10_fig10_1_M22_paper_dunn_asymptotic.csv', 'red', 'dunn_asymptotic'),
    ('mack_ch10_fig10_1_M22_paper_dunn_numerical.csv', 'blue', 'dunn_numerical'),
]:
    R, F = load_xy(f'{CAND}/{fname}')
    px = R2px(np.array(R)); py = PANELS['b']['fy_m'] * np.array(F) + PANELS['b']['fy_b'] - 950
    ax.scatter(px, py, s=14, c=color, label=label, zorder=5)
ax.set_title('Panel (b) M2.2 -- FINAL CANDIDATE CSVs (downsampled; complete_equations CONFIRMED, not re-plotted)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('verification/first_mode/_mack_ch10_verify/redigitize_work/final_candidates_overlay.png', dpi=110)
print('saved')
