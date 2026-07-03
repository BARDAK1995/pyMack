"""
Verify digitized CSVs for Mack (1984) Fig. 10.4 against the source figure crop.

Calibration (derived from pixel analysis of fig10_4.png, 1160x1230 px):
  Frame box: left border col=143 (R=0), right border col=1147 (R=2000)
             top border row=16 (omega_i*1e3=2.2), bottom border row=1116 (omega_i*1e3=0.0)
  Confirmed via tick-mark detection on both axes (see analysis below).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

IMG_PATH = r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\refPapers\latex_papers\figures\fig10_4.png"
OUT_PATH = r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\verification\first_mode\_mack_ch10_verify\_verify_fig10_4.png"

# Calibration anchors (pixel <-> data)
X0_PIX, X0_VAL = 143.0, 0.0
X1_PIX, X1_VAL = 1147.0, 2000.0
Y0_PIX, Y0_VAL = 1116.0, 0.0   # bottom
Y1_PIX, Y1_VAL = 16.0, 2.2     # top

def data_to_pixel(x, y):
    px = X0_PIX + (x - X0_VAL) * (X1_PIX - X0_PIX) / (X1_VAL - X0_VAL)
    py = Y0_PIX + (y - Y0_VAL) * (Y1_PIX - Y0_PIX) / (Y1_VAL - Y0_VAL)
    return px, py

datasets = [
    ("M1=4.5", r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\reference_data\digitized\mack_ch10_fig10_4_M45_paper.csv", "o", "red"),
    ("M1=5.8", r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\reference_data\digitized\mack_ch10_fig10_4_M58_paper.csv", "s", "blue"),
    ("M1=7.0", r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\reference_data\digitized\mack_ch10_fig10_4_M70_paper.csv", "^", "lime"),
    ("M1=10.0", r"C:\Users\merts\OneDrive\Masaüstü\MS_LST\reference_data\digitized\mack_ch10_fig10_4_M100_paper.csv", "D", "magenta"),
]

img = plt.imread(IMG_PATH)

fig, ax = plt.subplots(figsize=(14, 18))
ax.imshow(img)

for label, path, marker, color in datasets:
    df = pd.read_csv(path)
    px, py = data_to_pixel(df["x"].values, df["y"].values)
    ax.scatter(px, py, s=18, marker=marker, edgecolors=color, facecolors="none",
               linewidths=1.2, label=label)

ax.legend(loc="upper left", fontsize=11)
ax.set_title("Mack 1984 Fig 10.4 verification overlay: digitized points vs source curves")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print("saved", OUT_PATH)
