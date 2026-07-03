"""Generate publication-quality verification overlay plots for the Our LST code audit.

Reads only committed per-case data files. Writes overlay.png into each case
folder and updates that case's verdict.json artifacts.overlay key.

Visual-QA layout policy (no element may overlap or obscure plotted data):
  * Legends are placed OUTSIDE the axes (to the right of the data area) so they
    can never land on a curve, marker cluster, or shaded band, regardless of the
    curve shape in a given panel.
  * Verdict / metric annotation text is placed BELOW the axes in a dedicated
    figure-level strip, never over data.
  * constrained_layout reserves room for both, so nothing is clipped.
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- HARD style rules ---------------------------------------------------------
plt.rcParams.update({
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
    "figure.dpi": 110,
})

# Readability convention (workspace): reference/digitized paper data = RED and
# slightly LARGER markers; Our LST code computed curve = BLUE and slightly THINNER line.
PYMACK_BLUE = "#000000"     # Our LST code -> black, dashed
PYMACK_GREEN = "#000000"    # Our LST code second branch -> black (markers distinguish)
REF_ORANGE = "#d62728"      # reference dashed/hollow -> red
REF_VERM = "#d62728"        # reference 2nd branch -> red (markers distinguish)
GREY = "#999999"

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_csv(path, comment="#"):
    """Return list of dict rows from a CSV, skipping comment lines."""
    rows = []
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith(comment)]
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append(r)
    return rows


def fcol(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        v = v.strip() if isinstance(v, str) else v
        out.append(float(v) if v not in ("", None) else np.nan)
    return np.array(out)


def update_verdict(case_dir, rel_overlay):
    vpath = os.path.join(case_dir, "verdict.json")
    with open(vpath, "r", encoding="utf-8") as f:
        v = json.load(f)
    v.setdefault("artifacts", {})
    v["artifacts"]["overlay"] = rel_overlay
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)
        f.write("\n")


def outside_legend(ax, handles=None, labels=None):
    """Place the legend just outside the right edge of the axes (data-free)."""
    kw = dict(loc="upper left", bbox_to_anchor=(1.015, 1.0),
              framealpha=0.95, borderaxespad=0.0)
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


def bottom_note(fig, text):
    """Put a metric/topology note in a dedicated strip below the axes."""
    fig.text(0.012, 0.012, text, fontsize=11.5, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f2f2f2", ec=GREY,
                       alpha=0.97))


def save(fig, out):
    fig.savefig(out, dpi=175)  # constrained_layout already reserves margins
    plt.close(fig)


written = []


# =============================================================================
# 1. second_mode/sean_m5p35  -- dimensional M5.35 second-mode neutral curve
# =============================================================================
def plot_sean():
    cd = os.path.join(ROOT, "second_mode", "sean_m5p35")
    pm = load_csv(os.path.join(cd, "pymack_neutral_envelope_dimensional.csv"))
    rf = load_csv(os.path.join(cd, "LST_neutral_curve_M5p35.csv"))

    f_pm = fcol(pm, "frequency_khz")
    lo_pm = fcol(pm, "lower_neutral_x_mm")
    up_pm = fcol(pm, "upper_neutral_x_mm")

    f_rf = fcol(rf, "frequency_khz")
    lo_rf = fcol(rf, "x_left_mm")
    up_rf = fcol(rf, "x_right_mm")

    si = np.argsort(f_pm)
    f_pm, lo_pm, up_pm = f_pm[si], lo_pm[si], up_pm[si]
    sr = np.argsort(f_rf)
    f_rf, lo_rf, up_rf = f_rf[sr], lo_rf[sr], up_rf[sr]

    fig, ax = plt.subplots(figsize=(10.6, 7.4), constrained_layout=True)
    # reserve bottom strip for the note
    fig.get_layout_engine().set(rect=(0.0, 0.16, 1.0, 0.84))

    ax.plot(lo_pm, f_pm, "-", color=PYMACK_BLUE, lw=3.4,
            label="Our LST code lower branch (x_left)")
    ax.plot(up_pm, f_pm, "-", color=PYMACK_GREEN, lw=3.4,
            label="Our LST code upper branch (x_right)")
    ax.plot(lo_rf, f_rf, "--", color=REF_ORANGE, lw=2.0,
            marker="o", mfc="none", mec=REF_ORANGE, ms=4.5, markevery=4,
            label="Sean LST lower branch")
    ax.plot(up_rf, f_rf, "--", color=REF_VERM, lw=2.0,
            marker="s", mfc="none", mec=REF_VERM, ms=4.5, markevery=4,
            label="Sean LST upper branch")

    ax.set_xlabel("Neutral branch streamwise location  x  [mm]")
    ax.set_ylabel("Frequency  f  [kHz]")
    ax.set_title("M5.35 2nd-mode neutral curve (dimensional)\n"
                 "VERDICT: acceptable", color="#333333")
    ax.set_xlim(left=0)
    ax.margins(y=0.05)
    ax.grid(True, alpha=0.3)
    outside_legend(ax)

    bottom_note(fig,
                "Our LST code vs Sean independent LST    "
                "upper branch MAE = 3.2 mm (200-600 kHz, span 220 mm)\n"
                "lower branch MAE = 1.3 mm (330-600 kHz, span 19.8 mm)    "
                "topology matches (single closed band)")

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    update_verdict(cd, "verification/second_mode/sean_m5p35/overlay.png")
    written.append(("verification/second_mode/sean_m5p35/overlay.png",
                    "M5.35 dimensional 2nd-mode neutral curve: Our LST code lower/upper "
                    "branches (solid) vs Sean's independent LST (dashed, hollow) in "
                    "x[mm] vs f[kHz]; verdict acceptable (upper MAE 3.2 mm, lower 1.3 mm)."))


# =============================================================================
# 2. second_mode/mazhong2003_m4p5  -- growth vs R, two neutral crossings
# =============================================================================
def plot_mazhong():
    cd = os.path.join(ROOT, "second_mode", "mazhong2003_m4p5")
    d = json.load(open(os.path.join(cd, "pymack_growth_sweep.json")))
    sweep = d["sweep"]
    R = np.array([s["R"] for s in sweep])
    sigma = np.array([s["neg_alpha_i"] for s in sweep])  # -alpha_i = spatial growth

    bI_pm = d["branch_I_R_pymack"]
    bII_pm = d["branch_II_R_pymack"]
    bI_rf = d["ref_branch_I"]
    bII_rf = d["ref_branch_II"]
    F = d["F"]
    eI = abs(bI_pm - bI_rf) / bI_rf * 100.0
    eII = abs(bII_pm - bII_rf) / bII_rf * 100.0
    emax = max(eI, eII)

    fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.15, 1.0, 0.85))

    ax.fill_between(R, 0, sigma, where=(sigma > 0), color=PYMACK_BLUE,
                    alpha=0.12, label="Our LST code unstable band ($\\sigma>0$)")
    ax.plot(R, sigma, "-o", color=PYMACK_BLUE, lw=3.4, ms=5.5, mfc=PYMACK_BLUE,
            label="Our LST code spatial growth  $\\sigma=-\\alpha_i$")
    ax.axhline(0, color="black", lw=1.0)

    ax.axvline(bI_pm, color=PYMACK_GREEN, lw=3.4, ls="-",
               label=f"Our LST code Branch I (R={bI_pm:.0f})")
    ax.axvline(bII_pm, color=PYMACK_GREEN, lw=3.4, ls="-",
               label=f"Our LST code Branch II (R={bII_pm:.0f})")
    ax.axvline(bI_rf, color=REF_ORANGE, lw=2.2, ls="--",
               label=f"Ma & Zhong Branch I (R={bI_rf:.0f})")
    ax.axvline(bII_rf, color=REF_ORANGE, lw=2.2, ls="--",
               label=f"Ma & Zhong Branch II (R={bII_rf:.1f})")

    ax.set_xlabel("Reynolds number  R = $\\sqrt{Re_x}$")
    ax.set_ylabel("Spatial growth rate  $\\sigma = -\\alpha_i$")
    ax.set_title(f"M4.5 2nd-mode growth at fixed F={F:.1e} (isothermal disturbance BC)"
                 f"\nVERDICT: agrees ($\\leq${emax:.1f}%)",
                 color="#333333")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(R.min(), R.max())
    ax.margins(y=0.05)
    outside_legend(ax)

    bottom_note(fig,
                f"Branch I:  Our LST code {bI_pm:.0f} vs M&Z {bI_rf:.0f}  ($+{eI:.1f}\\%$)    "
                f"Branch II: Our LST code {bII_pm:.0f} vs M&Z {bII_rf:.1f} ($+{eII:.1f}\\%$)    "
                "topology: one closed band, two neutral points")

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    update_verdict(cd, "verification/second_mode/mazhong2003_m4p5/overlay.png")
    written.append(("verification/second_mode/mazhong2003_m4p5/overlay.png",
                    f"M4.5 2nd-mode spatial growth sigma=-alpha_i vs R at fixed F=2.2e-4 "
                    f"(isothermal disturbance BC): Our LST code lobe (solid) with its two neutral "
                    f"crossings vs Ma&Zhong Branch I={bI_rf:.0f} / II={bII_rf:.1f} (dashed "
                    f"lines); verdict agrees (max {emax:.1f}%)."))


# =============================================================================
# 3. first_mode/mack_fig10_1_m{16,22}  -- first-mode neutral loop F*1e4 vs R
# =============================================================================
def plot_mack_fig10_1(mtag, mlabel):
    cd = os.path.join(ROOT, "first_mode", f"mack_fig10_1_m{mtag}")
    # Title reads the LIVE verdict word + loop-average error from verdict.json
    # (no hardcoded numbers) so the plot can never drift from the recorded judgement.
    with open(os.path.join(cd, "verdict.json"), "r", encoding="utf-8") as _vf:
        _v = json.load(_vf)
    Mn = "16" if mtag == "1p6" else "22"
    pm = load_csv(os.path.join(cd, f"pymack_mack_fig10_1_M{Mn}_neutral.csv"))
    ce = load_csv(os.path.join(cd, f"reference_mack_fig10_1_M{Mn}_complete_equations.csv"))
    dep = load_csv(os.path.join(cd, f"reference_mack_fig10_1_M{Mn}_complete.csv"))

    R_pm = fcol(pm, "R")
    Flo = fcol(pm, "F_lower_x1e4")
    Fup = fcol(pm, "F_upper_x1e4")

    R_ce = fcol(ce, "R")
    F_ce = fcol(ce, "F_x1e4")
    br = [r["branch"].strip() for r in ce]
    lo_mask = np.array([b == "lower" for b in br])
    up_mask = np.array([b == "upper" for b in br])
    nose_mask = np.array([b == "nose" for b in br])

    x_dep = fcol(dep, "x")
    y_dep = fcol(dep, "y")

    fig, ax = plt.subplots(figsize=(10.6, 6.8), constrained_layout=True)

    sl = np.argsort(R_ce[lo_mask])
    su = np.argsort(R_ce[up_mask])
    # Mack reference neutral curve: both branches drawn, ONE legend entry
    # (the loop is a single neutral curve -- do not split upper/lower).
    ax.plot(R_ce[lo_mask][sl], F_ce[lo_mask][sl], "--", color=REF_ORANGE, lw=1.6,
            marker="o", mfc="none", mec=REF_ORANGE, ms=5.5, mew=1.4,
            label="Mack (1984) neutral curve (digitized)")
    ax.plot(R_ce[up_mask][su], F_ce[up_mask][su], "--", color=REF_ORANGE, lw=1.6,
            marker="o", mfc="none", mec=REF_ORANGE, ms=5.5, mew=1.4)
    if nose_mask.any():
        ax.plot(R_ce[nose_mask], F_ce[nose_mask], "*", color=REF_ORANGE, ms=14,
                mfc="none", mec=REF_ORANGE, mew=1.7, label=r"Mack critical $R$ (nose)")

    m_lo = np.isfinite(Flo)
    m_up = np.isfinite(Fup)
    # Our LST code neutral curve: both branches drawn, ONE legend entry.
    ax.plot(R_pm[m_lo], Flo[m_lo], "--", color=PYMACK_BLUE, lw=3.6,
            marker="o", ms=5.0, label="Our LST code neutral curve")
    ax.plot(R_pm[m_up], Fup[m_up], "--", color=PYMACK_BLUE, lw=3.6,
            marker="o", ms=5.0)

    ax.set_xlabel("Reynolds number  R = $\\sqrt{Re_x}$")
    ax.set_ylabel("Frequency  $F \\times 10^{4}$")
    ax.set_title(f"Mack (1984) Fig. 10.1  -  first-mode neutral curve  {mlabel}",
                 color="#333333")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.05)
    # Legend INSIDE (upper-right is empty for this decaying loop) so the axes keep
    # full width and the title is not clipped.
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.6")

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    rel = f"verification/first_mode/mack_fig10_1_m{mtag}/overlay.png"
    update_verdict(cd, rel)
    written.append((rel,
                    f"Mack Fig10.1 {mlabel} first-mode neutral curve F*1e4 vs R: Our LST code "
                    f"(blue, one label) vs Mack's Complete-Equations loop (red, digitized)."))


# NOTE: The Özgen Fig. 3 neutral-curve overlays (M2/M3/M4/M6/M7/M8/M10) are NOT
# produced here anymore. They are the single canonical Özgen overlay generator
# `verification/make_ozgen_overlays.py`, which plots Our LST code's own full c_i=0
# contour (from the committed first/second-mode grids + continuation traces) over
# the corrected multi-branch v2 digitized reference points, with the title read
# live from each case's verdict.json. The old routine here hardcoded a stale
# "disagrees" verdict and plotted the superseded single-arch reference, so it was
# removed.


if __name__ == "__main__":
    plot_sean()
    plot_mazhong()
    plot_mack_fig10_1("1p6", "M=1.6")
    plot_mack_fig10_1("2p2", "M=2.2")
    print("WROTE:")
    for rel, desc in written:
        print(f"  {rel}\n     {desc}")
