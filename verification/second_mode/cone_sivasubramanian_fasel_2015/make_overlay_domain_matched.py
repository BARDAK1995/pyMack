#!/usr/bin/env python
"""Regenerate overlay.png for the Sivasubramanian-Fasel Mach-6 sharp cone case,
anchored to the OG thesis Fig. 5.1 (Sivasubramanian 2012 PhD thesis).

Two clean data panels (NO copyrighted figure image embedded):
  (left)  pyMack domain-matched peak N-factor vs frequency, with the corrected
          thesis benchmark: peak N ~= 9.5 at f ~= 210 kHz (F = 1.071e-5).
  (right) pyMack domain-matched N(x*) at 210 kHz vs the thesis Fig. 5.1 bold
          (kc=0, F=1.071e-5) curve, hand-read at a few anchor points.

Fonts obey the repo HARD rule: labels>=14, ticks>=12, titles>=16, legend>=11.
Reads only committed data (domain_matched_result.json); recomputes nothing.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.titlesize": 17, "legend.fontsize": 12.5, "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0, "lines.linewidth": 3.4,
})
PYMACK_BLUE = "#000000"; REF_ORANGE = "#D55E00"; REF_VERMIL = "#E69F00"; BAND_GREEN = "#009E73"

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "domain_matched_result.json")))

freqs = sorted(float(k) for k in d["results"])
Npk = [d["results"][f"{f}"]["N_peak"] if f"{f}" in d["results"] else d["results"][str(f)]["N_peak"] for f in freqs]
Npk = [d["results"][str(f)]["N_peak"] for f in freqs]
f_peak = d["f_peak_khz"]; N_peak = d["N_peak_domain_matched"]; N_210 = d["N_at_210khz"]

# thesis Fig 5.1 bold (kc=0, F=1.071e-5) curve, hand-read anchor points (axial x* in m)
thesis_x = [0.31, 0.35, 0.40, 0.45, 0.47, 0.50, 0.53, 0.55, 0.57, 0.59]
thesis_N = [1.0,  1.4,  2.1,  4.0,  5.2,  6.6,  7.8,  8.5,  9.1,  9.5]

# pyMack domain-matched N(x*) at 210 kHz
r210 = d["results"]["210.0"]
R = np.array(r210["R_curve"]); N = np.array(r210["N_curve"]); sig = np.array(r210["sigma_L"])
s_mm = np.array(d["s_mm"]); cos_tc = d["cos_theta_c"]
x_axial = s_mm * cos_tc / 1000.0

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 6.4))

# ---- LEFT: peak-N vs frequency ----
axL.axvline(210.0, color=BAND_GREEN, lw=2.4, ls="--", zorder=1,
            label="thesis most-amplified f = 210 kHz (F=1.071e-5)")
axL.axhline(9.5, color=REF_VERMIL, lw=2.4, ls="--", zorder=1,
            label="thesis peak N ≈ 9.5 (Fig. 5.1)")
axL.plot(freqs, Npk, "-", color=PYMACK_BLUE, zorder=4, label="pyMack domain-matched peak N")
axL.plot(freqs, Npk, "o", color=PYMACK_BLUE, ms=7, zorder=5)
axL.plot([f_peak], [N_peak], "*", color=REF_ORANGE, ms=22, mec="k", mew=0.8, zorder=6,
         label=f"pyMack peak: N={N_peak:.2f} @ {f_peak:.0f} kHz")
axL.set_xlabel("frequency  (kHz)")
axL.set_ylabel("domain-matched peak N-factor")
axL.set_title("(a)  peak N vs frequency")
axL.set_ylim(0, 10.4)
axL.margins(x=0.03)
axL.grid(True, alpha=0.3)
axL.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=1,
           framealpha=0.95, borderaxespad=0.0)

# ---- RIGHT: N(x*) at 210 kHz ----
axR.plot(thesis_x, thesis_N, "s--", color=REF_ORANGE, lw=2.0, ms=8, mec="k", mew=0.6,
         zorder=4, label="thesis Fig. 5.1  (kc=0, F=1.071e-5)")
axR.plot(x_axial, N, "-", color=PYMACK_BLUE, zorder=5,
         label="pyMack domain-matched  (210 kHz)")
axR.axhline(9.5, color=REF_VERMIL, lw=1.6, ls=":", alpha=0.8, zorder=1)
axR.set_xlabel("axial distance  x*  (m)")
axR.set_ylabel("N-factor  (from lower neutral point)")
axR.set_title("(b)  N(x*) at the most-amplified frequency")
axR.set_xlim(0.29, 0.605)
axR.set_ylim(0, 10.4)
axR.grid(True, alpha=0.3)
axR.legend(loc="upper left", framealpha=0.95)

fig.suptitle("Sivasubramanian (2012) PhD thesis Fig. 5.1  —  Mach-6 sharp 7° cone (BAM6QT)\n"
             f"verdict: acceptable  |  f: 210 kHz exact match  |  N: pyMack {N_210:.1f} vs thesis 9.5 "
             f"({100*(9.5-N_210)/9.5:.0f}% short, Mangler/self-similar)",
             fontsize=16, y=1.02)
fig.subplots_adjust(bottom=0.30, top=0.84, left=0.07, right=0.985, wspace=0.22)
out = os.path.join(HERE, "overlay.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
