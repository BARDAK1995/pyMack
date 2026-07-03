import sys
sys.path.insert(0, 'verification/first_mode/_mack_ch10_verify/redigitize_work')
from digitize_fig10_1_dunn import *
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open('verification/first_mode/_mack_ch10_verify/redigitize_work/RF_all.pkl', 'rb') as f:
    RF = pickle.load(f)

im = Image.open(SRC)

fig, axes = plt.subplots(2, 1, figsize=(15, 22))

# Panel a
ax = axes[0]
imcropA = im.crop((0, 0, 1300, 900))
ax.imshow(imcropA, cmap='gray', extent=[0, 1300, 900, 0])
for key, color, marker, label in [
    ('M16_dunn_asymptotic_A', 'red', 'o', 'dunn_asymptotic onset'),
    ('M16_dunn_asymptotic_B', 'darkred', 's', 'dunn_asymptotic cutoff'),
    ('M16_dunn_numerical_A', 'blue', 'o', 'dunn_numerical onset'),
    ('M16_dunn_numerical_B', 'darkblue', 's', 'dunn_numerical cutoff'),
    ('M16_complete_equations_A', 'green', 'o', 'complete_eq onset'),
    ('M16_complete_equations_B', 'darkgreen', 's', 'complete_eq cutoff'),
]:
    R, F = RF[key]
    px = R2px(R)
    py = PANELS['a']['fy_m'] * F + PANELS['a']['fy_b']
    ax.scatter(px, py, s=4, c=color, marker=marker, label=label)
ax.set_title('Panel (a) M1.6 -- new digitization overlay')
ax.legend(loc='upper right', fontsize=8)

# Panel b
ax = axes[1]
imcropB = im.crop((0, 950, 1300, 1820))
ax.imshow(imcropB, cmap='gray', extent=[0, 1300, 1820 - 950, 0])
for key, color, marker, label in [
    ('M22_dunn_asymptotic_shared', 'magenta', '^', 'dunn_asymptotic shared(low-R)'),
    ('M22_dunn_asymptotic_A', 'red', 'o', 'dunn_asymptotic onset'),
    ('M22_dunn_asymptotic_B', 'darkred', 's', 'dunn_asymptotic cutoff'),
    ('M22_dunn_numerical_A', 'blue', 'o', 'dunn_numerical onset'),
    ('M22_dunn_numerical_B', 'darkblue', 's', 'dunn_numerical cutoff'),
]:
    R, F = RF[key]
    px = R2px(R)
    py = PANELS['b']['fy_m'] * F + PANELS['b']['fy_b'] - 950
    ax.scatter(px, py, s=4, c=color, marker=marker, label=label)
ax.set_title('Panel (b) M2.2 -- new digitization overlay (complete_equations CONFIRMED, not shown)')
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
outpath = 'verification/first_mode/_mack_ch10_verify/_redigitized_fig10_1.png'
plt.savefig(outpath, dpi=110)
print('saved', outpath)
