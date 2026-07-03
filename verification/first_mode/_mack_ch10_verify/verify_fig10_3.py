import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

BASE = r"C:\Users\merts\OneDrive\Masaüstü\MS_LST"
IMG_PATH = os.path.join(BASE, "refPapers", "latex_papers", "figures", "fig10_3.png")
OUT_PATH = os.path.join(BASE, "verification", "first_mode", "_mack_ch10_verify", "_verify_fig10_3.png")
CSV_DIR = os.path.join(BASE, "reference_data", "digitized")

img = plt.imread(IMG_PATH)
print("image shape:", img.shape)

# ---------------------------------------------------------------
# Calibration (pixel coords measured directly off fig10_3.png, 2490x1140)
# Each panel: linear map data(x,y) -> pixel(col,row)
#   col = col0 + (x - x0) * (col1-col0)/(x1-x0)
#   row = row0 + (y - y0) * (row1-row0)/(y1-y0)
# ---------------------------------------------------------------
calib = {
    "a": {  # M1=1.3, top-left
        "x0": 0, "col0": 200.0, "x1": 20, "col1": 1287.0,
        "y0": 0, "row0": 477.0, "y1": 1.6, "row1": 49.0,
    },
    "b": {  # M1=1.6, bottom-left
        "x0": 0, "col0": 200.0, "x1": 20, "col1": 1287.0,
        "y0": 0, "row0": 962.0, "y1": 1.6, "row1": 527.0,
    },
    "c": {  # M1=2.2, top-right
        "x0": 0, "col0": 1340.5, "x1": 20, "col1": 2429.0,
        "y0": 0, "row0": 477.0, "y1": 1.6, "row1": 49.0,
    },
    "d": {  # M1=3.0, bottom-right
        "x0": 0, "col0": 1340.5, "x1": 20, "col1": 2429.0,
        "y0": 0, "row0": 962.0, "y1": 1.6, "row1": 527.0,
    },
}


def to_pixel(panel, x, y):
    c = calib[panel]
    col = c["col0"] + (x - c["x0"]) * (c["col1"] - c["col0"]) / (c["x1"] - c["x0"])
    row = c["row0"] + (y - c["y0"]) * (c["row1"] - c["row0"]) / (c["y1"] - c["y0"])
    return col, row


def load_csv(name):
    path = os.path.join(CSV_DIR, name)
    xs, ys = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.array(xs), np.array(ys)


datasets = [
    # (filename, panel, label, marker, color)
    ("mack_ch10_fig10_3_M13_paper_2D.csv", "a", "(a) M1.3 2D", "o", "red"),
    ("mack_ch10_fig10_3_M13_paper_psi30.csv", "a", "(a) M1.3 psi=30", "s", "blue"),
    ("mack_ch10_fig10_3_M13_paper_psi40.csv", "a", "(a) M1.3 psi=40", "^", "green"),
    ("mack_ch10_fig10_3_M13_paper_psi45.csv", "a", "(a) M1.3 psi=45", "D", "magenta"),
    ("mack_ch10_fig10_3_M16_paper_2d.csv", "b", "(b) M1.6 2D", "o", "red"),
    ("mack_ch10_fig10_3_M16_paper_psi45.csv", "b", "(b) M1.6 psi=45", "s", "blue"),
    ("mack_ch10_fig10_3_M22_paper_2d.csv", "c", "(c) M2.2 2D", "o", "red"),
    ("mack_ch10_fig10_3_M22_paper_psi45.csv", "c", "(c) M2.2 psi=45(env)", "s", "blue"),
    ("mack_ch10_fig10_3_M30_paper_2d.csv", "d", "(d) M3.0 2D", "o", "red"),
    ("mack_ch10_fig10_3_M30_paper_psi60.csv", "d", "(d) M3.0 psi=60(env)", "s", "blue"),
]

fig, ax = plt.subplots(figsize=(14, 18))
ax.imshow(img)

for fname, panel, label, marker, color in datasets:
    xs, ys = load_csv(fname)
    px, py = to_pixel(panel, xs, ys)
    ax.scatter(px, py, s=40, marker=marker, edgecolors=color, facecolors="none",
               linewidths=1.6, label=label)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)
ax.set_title("Mack (1984) Fig 10.3 -- digitized CSV overlay verification", fontsize=14)
# NOTE: do not use bbox_inches='tight' -- it breaks the linear mapping between
# saved-PNG pixel coordinates and the imshow data (=original image pixel) coordinates,
# which is needed for independent pixel-based verification of this overlay.
plt.savefig(OUT_PATH, dpi=150)
print("saved:", OUT_PATH)
print("image data shape:", img.shape)
