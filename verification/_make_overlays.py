#!/usr/bin/env python
"""Generate publication-quality overlay.png plots for the pyMack verification audit.
Works ONLY from committed per-case data files (no recomputation).
Saves overlay.png in each case folder and updates verdict.json artifacts.overlay.
"""
import json
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- HARD style rule ----
plt.rcParams.update({
    "axes.labelsize": 15,   # >= 14
    "xtick.labelsize": 13,  # >= 12
    "ytick.labelsize": 13,  # >= 12
    "axes.titlesize": 17,   # >= 16
    "legend.fontsize": 12,  # >= 11
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
})

# Colorblind-friendly (Okabe-Ito)
PYMACK_BLUE = "#0072B2"   # pyMack
REF_ORANGE = "#D55E00"    # reference / paper
REF_VERMIL = "#E69F00"
BAND_GREEN = "#009E73"

ROOT = r"C:/Users/merts/OneDrive/Masaüstü/MS_LST"
VER = os.path.join(ROOT, "verification")
DIG = os.path.join(ROOT, "reference_data", "digitized")


def read_csv_xy(path):
    xs, ys = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if not row or row[0].strip() == "":
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return xs, ys


def set_overlay(verdict_path, rel):
    with open(verdict_path, "r", encoding="utf-8") as f:
        v = json.load(f)
    v["artifacts"]["overlay"] = rel
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


written = []

# =====================================================================
# 1. Mack Fig 10.6 second mode: M4.5, M5.8, M7.0, M10.0
# =====================================================================
mack_cases = [
    ("mack_fig10_6_M45",  "M45",  4.5,  "agrees",     "1.0%"),
    ("mack_fig10_6_M58",  "M58",  5.8,  "acceptable", "6.7%"),
    ("mack_fig10_6_M70",  "M70",  7.0,  "agrees",     "2.8%"),
    ("mack_fig10_6_M100", "M100", 10.0, "agrees",     "4.0%"),
]
for case, tag, M, verdict, med in mack_cases:
    cdir = os.path.join(VER, "second_mode", case)
    with open(os.path.join(cdir, "pymack_curve.json")) as f:
        curve = json.load(f)
    pm_R = [d["R"] for d in curve if d.get("omega_i_max") is not None]
    pm_y = [d["omega_i_max"] * 1e3 for d in curve if d.get("omega_i_max") is not None]

    ref_path = os.path.join(DIG, f"mack_ch10_fig10_6_{tag}_paper.csv")
    rx, ry = read_csv_xy(ref_path)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.plot(rx, ry, "--", color=REF_ORANGE, linewidth=2.2, zorder=2,
            label="Mack (1984) Fig 10.6 (digitized)")
    ax.plot(rx, ry, "s", mfc="none", mec=REF_ORANGE, mew=1.8, ms=8, zorder=3)
    ax.plot(pm_R, pm_y, "-", color=PYMACK_BLUE, zorder=4, label="pyMack")
    ax.plot(pm_R, pm_y, "o", color=PYMACK_BLUE, ms=7, zorder=5)

    ax.set_xlabel(r"Reynolds number  $R$")
    ax.set_ylabel(r"max second-mode growth  $\omega_i \times 10^{3}$")
    ax.set_title(f"Mack Fig 10.6  second mode  M={M}\n"
                 f"verdict: {verdict}  (median rel. err. {med})")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)
    fig.tight_layout()
    out = os.path.join(cdir, "overlay.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    set_overlay(os.path.join(cdir, "verdict.json"),
                f"verification/second_mode/{case}/overlay.png")
    written.append((out, f"M={M}: pyMack omega_i*1e3 vs R (solid blue) over digitized "
                         f"Mack Fig 10.6 (dashed orange); {verdict}, median {med}."))

# =====================================================================
# 2. Cone Sivasubramanian & Fasel 2015: N-factor vs frequency
# =====================================================================
cdir = os.path.join(VER, "second_mode", "cone_sivasubramanian_fasel_2015")
with open(os.path.join(cdir, "pymack_cone_curve.json")) as f:
    cone = json.load(f)
res = cone["results"]
freqs = sorted(float(k) for k in res.keys())
Npk = [res[f"{f:.1f}" if f"{f:.1f}" in res else str(f)]["N_peak"] for f in freqs]
# robust key lookup
def cone_key(f):
    for k in res:
        if abs(float(k) - f) < 1e-6:
            return k
    raise KeyError(f)
Npk = [res[cone_key(f)]["N_peak"] for f in freqs]

peak_f = cone["f_peak_khz"]
peak_N = cone["N_peak"]

fig, ax = plt.subplots(figsize=(8.2, 6.0))
# benchmark most-amplified band 220-280 kHz
ax.axvspan(220, 280, color=BAND_GREEN, alpha=0.18, zorder=0,
           label="benchmark most-amplified band\n(220-280 kHz)")
# benchmark peak-N band N~7-8
ax.axhspan(7, 8, color=REF_VERMIL, alpha=0.16, zorder=0,
           label="benchmark peak N (7-8)")
ax.plot(freqs, Npk, "-", color=PYMACK_BLUE, zorder=4, label="pyMack peak N-factor")
ax.plot(freqs, Npk, "o", color=PYMACK_BLUE, ms=8, zorder=5)
ax.plot([peak_f], [peak_N], "*", color=REF_ORANGE, ms=22, mec="k", mew=0.8,
        zorder=6, label=f"pyMack peak: N={peak_N:.2f} @ {peak_f:.0f} kHz")

ax.set_xlabel("frequency  (kHz)")
ax.set_ylabel("peak N-factor over cone  (s = 120-520 mm)")
ax.set_title("Sivasubramanian & Fasel (2015)  Mach-6 sharp cone\n"
             "verdict: acceptable  (peak N=7.06 vs N~7-8; f at band lower edge)")
ax.legend(loc="upper right", framealpha=0.95)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(cdir, "overlay.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
set_overlay(os.path.join(cdir, "verdict.json"),
            "verification/second_mode/cone_sivasubramanian_fasel_2015/overlay.png")
written.append((out, "Cone: pyMack peak N-factor vs frequency (blue) with benchmark "
                     "N~7-8 band (orange) and 220-280 kHz most-amplified band (green); "
                     "peak N=7.06 @ 210 kHz; acceptable."))

# =====================================================================
# 3. Egorov 2006 M6: spatial growth vs omega
# =====================================================================
cdir = os.path.join(VER, "second_mode", "egorov2006_m6")
with open(os.path.join(cdir, "pymack_band.json")) as f:
    eg = json.load(f)
rows = eg["rows"]
om = [r["omega_E"] for r in rows]
g = [r["growth"] * 1e3 for r in rows]   # -alpha_i scaled to ~milli units for readability
peak = eg["peak"]
band = eg["band"]

fig, ax = plt.subplots(figsize=(8.2, 6.0))
ax.axhline(0.0, color="0.4", lw=1.0, ls=":", zorder=1)
# unstable band shading
ax.axvspan(band[0], band[1], color=BAND_GREEN, alpha=0.15, zorder=0,
           label=f"pyMack unstable band\n(omega {band[0]:.0f}-{band[1]:.0f})")
ax.plot(om, g, "-", color=PYMACK_BLUE, zorder=4, label="pyMack spatial growth")
ax.plot(om, g, "o", color=PYMACK_BLUE, ms=6, zorder=5)
# Egorov forced omega = 200
ax.axvline(200.0, color=REF_ORANGE, ls="--", lw=2.2, zorder=3,
           label="Egorov (2006) forced  omega=200")
# pyMack peak
ax.plot([peak["omega_E"]], [peak["growth"] * 1e3], "*", color=REF_VERMIL, ms=22,
        mec="k", mew=0.8, zorder=6,
        label=f"pyMack peak  omega={peak['omega_E']:.0f}")

ax.set_xlabel(r"dimensionless frequency  $\omega$  (plate-length units)")
ax.set_ylabel(r"spatial growth  $-\alpha_i \times 10^{3}$")
ax.set_title("Egorov, Fedorov & Soudakov (2006)  Mach-6 flat plate\n"
             "verdict: acceptable  (peak omega=215 vs forced 200; +7.5% offset)")
ax.legend(loc="upper left", framealpha=0.95)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(cdir, "overlay.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
set_overlay(os.path.join(cdir, "verdict.json"),
            "verification/second_mode/egorov2006_m6/overlay.png")
written.append((out, "Egorov M6: pyMack spatial second-mode growth vs omega (blue) with "
                     "unstable band (green) and Egorov forced omega=200 (dashed orange); "
                     "peak at omega=215; acceptable (7.5% offset)."))

# =====================================================================
# 4. Ozgen Fig 3 lobes: M2 and M4, c_i=0.004 contour (two-panel)
# =====================================================================
cdir = os.path.join(VER, "first_mode", "ozgen_fig3_lobes")

def read_ozgen_pymack(path):
    Re, al = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = row["alpha_pymack_ci_level"].strip()
            if v == "":
                continue
            Re.append(float(row["Re_L"]))
            al.append(float(v))
    return Re, al

panels = [
    ("M2", 2, "pymack_ozgen_M2_ci004.csv", "reference_ozgen_fig3_M2_004.csv"),
    ("M4", 4, "pymack_ozgen_M4_ci004.csv", "reference_ozgen_fig3_M4_004.csv"),
]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=False)
for ax, (tag, M, pmf, reff) in zip(axes, panels):
    pRe, pAl = read_ozgen_pymack(os.path.join(cdir, pmf))
    rRe, rAl = read_csv_xy(os.path.join(cdir, reff))
    ax.plot(rRe, rAl, "--", color=REF_ORANGE, lw=2.2, zorder=2,
            label="Ozgen Fig 3 (digitized)")
    ax.plot(rRe, rAl, "s", mfc="none", mec=REF_ORANGE, mew=1.8, ms=8, zorder=3)
    ax.plot(pRe, pAl, "-", color=PYMACK_BLUE, zorder=4, label="pyMack")
    ax.plot(pRe, pAl, "o", color=PYMACK_BLUE, ms=6, zorder=5)
    ax.set_xlabel(r"Reynolds number  $Re_L$")
    if tag == "M2":
        ax.set_ylabel(r"wavenumber  $\alpha_L$")
    ax.set_title(f"M={M},  $c_i=0.004$ contour")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
fig.suptitle("Ozgen & Kircali (2008) Fig 3  first-mode growth lobes\n"
             "verdict: disagrees  (overall median rel. err. ~18.8%, open lobe)",
             fontsize=17)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(cdir, "overlay.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
set_overlay(os.path.join(cdir, "verdict.json"),
            "verification/first_mode/ozgen_fig3_lobes/overlay.png")
written.append((out, "Ozgen lobes (M2|M4): pyMack c_i=0.004 alpha-vs-Re contour (blue) vs "
                     "digitized 0.004 lobe (dashed orange); pyMack stays flat / open while "
                     "paper arches down; disagrees (~18.8%)."))

# ---- report ----
print("WROTE_OVERLAYS")
for p, desc in written:
    print(p)
    print("  " + desc)
