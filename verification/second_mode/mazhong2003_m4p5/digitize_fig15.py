"""Digitized Ma & Zhong (2003) Fig. 15 neutral curves (omega vs R), M=4.5.

Read from the figure; the SECOND-mode branches are anchored to the two values
quoted in §6 — the F = 2.2e-4 line crosses the lower branch at (R=806, w=0.177)
and the upper branch at (R=999.6, w=0.220). A verification plot reproduces the
figure layout (axes, both modes, the F = 2.2e-4 and 0.6e-4 lines) for visual
comparison against the published Fig. 15.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# (R, omega) read from Fig. 15.  Second mode anchored to §6 crossings.
CURVES = {
    ("second", "upper"): [(245, 0.190), (300, 0.203), (380, 0.211), (500, 0.216),
                          (700, 0.219), (1000, 0.220), (1300, 0.222), (1600, 0.224), (2000, 0.225)],
    ("second", "lower"): [(245, 0.190), (320, 0.183), (450, 0.180), (600, 0.178),
                          (806, 0.177), (1100, 0.175), (1500, 0.173), (2000, 0.170)],
    ("first", "upper"):  [(565, 0.038), (650, 0.063), (780, 0.082), (950, 0.095),
                          (1150, 0.104), (1400, 0.112), (1700, 0.118), (2000, 0.122)],
    ("first", "lower"):  [(565, 0.030), (640, 0.020), (780, 0.015), (1000, 0.013),
                          (1300, 0.012), (1700, 0.0115), (2000, 0.011)],
}


def write_csv():
    p = HERE / "reference_mazhong_fig15.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["mode", "branch", "R", "omega"])
        for (mode, br), pts in CURVES.items():
            for R, om in pts:
                w.writerow([mode, br, R, om])
    return p


def verify_plot():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    styles = {"second": dict(color="#d55e00", ls="-.", lw=2.2, label="2nd mode (digitized)"),
              "first": dict(color="#0072b2", ls="-", lw=2.2, label="1st mode (digitized)")}
    seen = set()
    for (mode, br), pts in CURVES.items():
        a = np.array(pts); st = dict(styles[mode])
        if mode in seen:
            st.pop("label", None)
        seen.add(mode)
        ax.plot(a[:, 0], a[:, 1], **st)
    # F lines
    R = np.linspace(0, 2000, 50)
    ax.plot(R, R * 2.2e-4, ":", color="0.3", lw=1.5)
    ax.plot(R, R * 0.6e-4, ":", color="0.3", lw=1.5)
    ax.text(1350, 0.255, r"$F=2.2\times10^{-4}$", fontsize=12)
    ax.text(1500, 0.105, r"$F=0.6\times10^{-4}$", fontsize=12)
    # anchor crossings
    ax.plot([806, 999.6], [0.177, 0.220], "k*", ms=13, label="§6 anchors (806, 999.6)")
    ax.set_xlim(0, 2000); ax.set_ylim(0, 0.28)
    ax.set_xlabel("R", fontsize=15); ax.set_ylabel(r"$\omega$", fontsize=15)
    ax.set_title("Digitized Ma & Zhong (2003) Fig. 15  (verify vs published)", fontsize=14)
    ax.tick_params(labelsize=12); ax.legend(fontsize=11, loc="center right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(HERE / "_verify_fig15_digitized.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    print("wrote", write_csv())
    verify_plot()
    print("wrote _verify_fig15_digitized.png")
