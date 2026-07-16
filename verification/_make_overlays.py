#!/usr/bin/env python
"""Generate publication-quality overlay.png plots for the pyMack verification audit.

Works ONLY from committed per-case data files (no recomputation). For each case it
re-plots pyMack (solid/filled) against the digitized reference (dashed/hollow) and
writes overlay.png in place, then updates verdict.json artifacts.overlay.

Visual-QA rules enforced here (see verification audit task):
  * No legend / annotation / verdict text / inset may overlap or obscure plotted data.
    Legend placement is chosen PER FIGURE for a genuinely empty region (loc='best'
    is not trusted); where no interior space is clear the legend goes OUTSIDE the axes.
  * Fonts (hard minimums): axis labels >= 14 pt, ticks >= 12 pt, titles >= 16 pt,
    legend >= 11 pt.
  * Verdict + headline metric stay visible in the title (never over data).
  * Generous axis margins so curves/markers are not jammed against the frame and
    nothing is clipped.
"""
import json
import os
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- HARD style rule ----
plt.rcParams.update({
    "axes.labelsize": 17,   # >= 14
    "xtick.labelsize": 14,  # >= 12
    "ytick.labelsize": 14,  # >= 12
    "axes.titlesize": 17,   # >= 16
    "legend.fontsize": 12.5,  # >= 11
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
    "lines.linewidth": 3.4,   # thick, very visible pyMack curve
})

# Colorblind-friendly (Okabe-Ito)
PYMACK_BLUE = "#000000"   # pyMack -> thick black
REF_ORANGE = "#D55E00"    # reference / paper
REF_VERMIL = "#E69F00"
BAND_GREEN = "#009E73"

ROOT = Path(__file__).resolve().parents[1]
VER = os.path.join(ROOT, "verification")
DIG = os.path.join(ROOT, "reference_data", "digitized")

DPI = 170


def read_csv_xy(path):
    xs, ys = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
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


def save(fig, out):
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


written = []

# =====================================================================
# 1. Mack Fig 10.6 second mode: M4.5, M5.8, M7.0, M10.0
#    Monotonic rising curves. Data climbs to the upper-right; the
#    lower-right quadrant is empty for every Mach number, so the legend
#    sits there clear of all data. A small top y-margin keeps the
#    high-R markers off the frame and clear of the two-line title.
# =====================================================================
mack_cases = [
    ("mack_fig10_6_M45",  "M45",  4.5,  "acceptable", "6.0%"),
    ("mack_fig10_6_M58",  "M58",  5.8,  "acceptable", "8.9%"),
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

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    ax.plot(rx, ry, "--", color=REF_ORANGE, linewidth=2.2, zorder=2,
            label="Mack (1984) Fig 10.6 (digitized)")
    ax.plot(rx, ry, "s", mfc="none", mec=REF_ORANGE, mew=1.8, ms=8, zorder=3)
    ax.plot(pm_R, pm_y, "-", color=PYMACK_BLUE, zorder=4, label="pyMack")
    ax.plot(pm_R, pm_y, "o", color=PYMACK_BLUE, ms=7, zorder=5)

    # top headroom so high-R markers clear the frame and the two-line title
    ymax = max(max(ry), max(pm_y))
    ax.set_ylim(top=ymax * 1.10)
    ax.margins(x=0.03)

    ax.set_xlabel(r"Reynolds number  $R$")
    ax.set_ylabel(r"max second-mode growth  $\omega_i \times 10^{3}$")
    ax.set_title(f"Mack Fig 10.6  second mode  M={M}\n"
                 f"verdict: {verdict}  (median rel. err. {med})")
    # lower-right is the empty quadrant for a monotonic rising curve
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    out = os.path.join(cdir, "overlay.png")
    save(fig, out)
    set_overlay(os.path.join(cdir, "verdict.json"),
                f"verification/second_mode/{case}/overlay.png")
    written.append((out, f"M={M}: pyMack omega_i*1e3 vs R (solid blue) over digitized "
                         f"Mack Fig 10.6 (dashed orange); {verdict}, median {med}. "
                         f"Legend lower-right (empty)."))

# =====================================================================
# 2. Cone Sivasubramanian & Fasel 2015: N-factor vs frequency
#    Peaked curve: rises to N~7 at 210 kHz then descends to the right.
#    The peak/star sit upper-left; the orange peak-N band (N=7-8) runs
#    full width across the TOP, and the green most-amplified band is a
#    vertical strip 220-280 kHz. The previous upper-right legend covered
#    the orange band, the star and the curve near the peak.
#    Fix: legend goes BELOW the axes (the curve + both bands fill the
#    whole interior), and extra top headroom keeps the orange band and
#    star clear of the frame / title.
# =====================================================================
cdir = os.path.join(VER, "second_mode", "cone_sivasubramanian_fasel_2015")
with open(os.path.join(cdir, "pymack_cone_curve.json")) as f:
    cone = json.load(f)
res = cone["results"]
freqs = sorted(float(k) for k in res.keys())


def cone_key(f):
    for k in res:
        if abs(float(k) - f) < 1e-6:
            return k
    raise KeyError(f)


Npk = [res[cone_key(f)]["N_peak"] for f in freqs]
peak_f = cone["f_peak_khz"]
peak_N = cone["N_peak"]

fig, ax = plt.subplots(figsize=(8.6, 6.6))
# benchmark most-amplified band 220-280 kHz (vertical strip)
ax.axvspan(220, 280, color=BAND_GREEN, alpha=0.18, zorder=0,
           label="benchmark most-amplified band (220-280 kHz)")
# benchmark peak-N band N~7-8 (horizontal strip across top)
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
ax.set_ylim(top=9.0)          # headroom above the N=8 band edge + star
ax.margins(x=0.03)
ax.grid(True, alpha=0.3)
# four entries (two bands + curve + star) fill the interior; put the legend
# below the axes in two columns so it never touches data
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
          framealpha=0.95, borderaxespad=0.0)
fig.subplots_adjust(bottom=0.26, top=0.88, left=0.12, right=0.97)
out = os.path.join(cdir, "overlay.png")
save(fig, out)
set_overlay(os.path.join(cdir, "verdict.json"),
            "verification/second_mode/cone_sivasubramanian_fasel_2015/overlay.png")
written.append((out, "Cone: pyMack peak N-factor vs frequency (blue) with benchmark "
                     "N~7-8 band (orange) and 220-280 kHz most-amplified band (green); "
                     "peak N=7.06 @ 210 kHz; acceptable. Legend below axes (clear of "
                     "bands/curve/star)."))

# =====================================================================
# 3. Egorov 2006 M6: spatial growth vs omega
#    Single peaked bump: near zero at the left, rises to a peak at
#    omega=215 (top-centre), falls below zero on the far right. The
#    green unstable band shades the FULL x-range, so any in-axes legend
#    sits on the band, and the previous upper-left legend also covered
#    the rising flank + peak star.
#    Fix: legend goes BELOW the axes; symmetric x/y margins keep the
#    peak star and the negative tail off the frame.
# =====================================================================
cdir = os.path.join(VER, "second_mode", "egorov2006_m6")
with open(os.path.join(cdir, "pymack_band.json")) as f:
    eg = json.load(f)
rows = eg["rows"]
om = [r["omega_E"] for r in rows]
g = [r["growth"] * 1e3 for r in rows]   # -alpha_i scaled for readability
peak = eg["peak"]
band = eg["band"]

fig, ax = plt.subplots(figsize=(8.6, 6.6))
ax.axhline(0.0, color="0.4", lw=1.0, ls=":", zorder=1)
ax.axvspan(band[0], band[1], color=BAND_GREEN, alpha=0.15, zorder=0,
           label=f"pyMack unstable band (omega {band[0]:.0f}-{band[1]:.0f})")
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
gmax = max(g)
gmin = min(g)
ax.set_ylim(gmin - 0.25, gmax + 0.35)   # clear the star at top, tail at bottom
ax.margins(x=0.03)
ax.grid(True, alpha=0.3)
# band shading covers the full interior, so put the legend below the axes
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
          framealpha=0.95, borderaxespad=0.0)
fig.subplots_adjust(bottom=0.26, top=0.88, left=0.12, right=0.97)
out = os.path.join(cdir, "overlay.png")
save(fig, out)
set_overlay(os.path.join(cdir, "verdict.json"),
            "verification/second_mode/egorov2006_m6/overlay.png")
written.append((out, "Egorov M6: pyMack spatial second-mode growth vs omega (blue) with "
                     "unstable band (green) and Egorov forced omega=200 (dashed orange); "
                     "peak at omega=215; acceptable (7.5% offset). Legend below axes "
                     "(off the band and peak star)."))

# =====================================================================
# 4. Ozgen Fig 3 lobes: M2 and M4, c_i=0.004 contour (two-panel)
#    Both panels are arch-shaped lobes peaking upper-centre-left and
#    descending to the right, so an upper-right per-panel legend lands
#    on the descending reference markers. The two panels share the same
#    two series, so a SINGLE shared legend below both panels removes all
#    overlap and frees the panel interiors.
# =====================================================================
cdir = os.path.join(VER, "mixed_mode", "ozgen_fig3", "lobes")


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
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=False)
handles = labels = None
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
    ax.margins(x=0.04, y=0.12)   # keep arches off the frame
    ax.grid(True, alpha=0.3)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()

# single shared legend below both panels (series are identical per panel)
fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.95,
           bbox_to_anchor=(0.5, 0.0))
# Title reads the live verdict word + overall metric from verdict.json (no
# hardcoded numbers) so it can never drift from the recorded judgement.
_lobes_v = json.load(open(os.path.join(cdir, "verdict.json"), encoding="utf-8"))
_lobes_verd = _lobes_v.get("verdict", "?")
_lobes_pct = _lobes_v.get("metrics", {}).get("overall_median_rel_err_alpha")
_lobes_pct_s = f"{100 * _lobes_pct:.1f}%" if _lobes_pct is not None else "n/a"
fig.suptitle("Ozgen & Kircali (2008) Fig 3  first-mode growth lobes\n"
             f"verdict: {_lobes_verd}  (overall median rel. err. ~{_lobes_pct_s}, open lobe)",
             fontsize=17)
# leave room: top for two-line suptitle, bottom for shared legend
fig.tight_layout(rect=[0, 0.09, 1, 0.90])
out = os.path.join(cdir, "overlay.png")
save(fig, out)
set_overlay(os.path.join(cdir, "verdict.json"),
            "verification/mixed_mode/ozgen_fig3/lobes/overlay.png")
written.append((out, "Ozgen lobes (M2|M4): pyMack c_i=0.004 alpha-vs-Re contour (blue) vs "
                     "digitized 0.004 lobe (dashed orange); pyMack stays flat / open while "
                     f"paper arches down; {_lobes_verd} (~{_lobes_pct_s}). Single shared "
                     "legend below both panels."))

# ---- report ----
print("WROTE_OVERLAYS")
for p, desc in written:
    print(p)
    print("  " + desc)
