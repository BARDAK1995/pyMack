"""
Verify digitized fig10_6 CSVs against the source Mack (1984) figure crop.

Calibration derived from pixel analysis of refPapers/latex_papers/figures/fig10_6.png
(1190x1090 px), using inward tick marks along the frame:

X-axis (R):  px_col = 0.499630 * R + 134.534
   R=0    -> col 134.53
   R=2000 -> col 1133.79
   (fit over ticks at R=400,800,1000,1200,1400,1600,1800,2000; residuals < 2.3 px)

Y-axis (omega_i * 1e3):  px_row = -252.8106 * y + 958.809
   y=0.0 -> row 958.81
   y=3.6 -> row 48.69
   (fit over ticks at y=3.6,3.2,2.8,2.4,2.0,1.6,1.2,0.8,0.4,0.0; residuals < 2.5 px)
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os

BASE = r"C:\Users\merts\OneDrive\Masaüstü\MS_LST"
IMG_PATH = os.path.join(BASE, "refPapers", "latex_papers", "figures", "fig10_6.png")
OUT_PATH = os.path.join(BASE, "verification", "first_mode", "_mack_ch10_verify", "_verify_fig10_6.png")

# Calibration constants
X_SLOPE, X_INT = 0.499630325814536, 134.53383458646624   # col = X_SLOPE*R + X_INT
Y_SLOPE, Y_INT = -252.81060606060592, 958.8090909090904  # row = Y_SLOPE*y + Y_INT

def data_to_px(x, y):
    col = X_SLOPE * np.asarray(x, dtype=float) + X_INT
    row = Y_SLOPE * np.asarray(y, dtype=float) + Y_INT
    return col, row

def load_csv(path):
    xs, ys = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.array(xs), np.array(ys)

datasets = [
    ("M1=4.5",  "reference_data/digitized/mack_ch10_fig10_6_M45_paper.csv",  "o", "red"),
    ("M1=5.8",  "reference_data/digitized/mack_ch10_fig10_6_M58_paper.csv",  "s", "blue"),
    ("M1=7.0",  "reference_data/digitized/mack_ch10_fig10_6_M70_paper.csv",  "^", "lime"),
    ("M1=10.0", "reference_data/digitized/mack_ch10_fig10_6_M100_paper.csv", "D", "magenta"),
]

img = plt.imread(IMG_PATH)

fig, ax = plt.subplots(figsize=(14, 18))
ax.imshow(img)

for label, relpath, marker, color in datasets:
    path = os.path.join(BASE, relpath)
    x, y = load_csv(path)
    px, py = data_to_px(x, y)
    ax.scatter(px, py, s=40, marker=marker, edgecolors=color, facecolors='none',
               linewidths=1.6, label=label)

ax.set_title("Verification overlay: Mack (1984) Fig 10.6 -- digitized points vs source curves", fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print("saved", OUT_PATH)
