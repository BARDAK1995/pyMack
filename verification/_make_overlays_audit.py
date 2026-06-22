"""Generate publication-quality verification overlay plots for the pyMack audit.

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
    "axes.labelsize": 15,      # >= 14 pt
    "xtick.labelsize": 13,     # >= 12 pt
    "ytick.labelsize": 13,
    "axes.titlesize": 17,      # >= 16 pt
    "legend.fontsize": 11.5,   # >= 11 pt
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
    "figure.dpi": 110,
})

# Colorblind-friendly (Okabe-Ito)
PYMACK_BLUE = "#0072B2"     # pyMack solid
PYMACK_GREEN = "#009E73"    # pyMack second branch
REF_ORANGE = "#D55E00"      # reference dashed/hollow
REF_VERM = "#CC79A7"
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

    ax.plot(lo_pm, f_pm, "-", color=PYMACK_BLUE, lw=2.6,
            label="pyMack lower branch (x_left)")
    ax.plot(up_pm, f_pm, "-", color=PYMACK_GREEN, lw=2.6,
            label="pyMack upper branch (x_right)")
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
                "pyMack vs Sean independent LST    "
                "upper branch MAE = 3.2 mm (200-600 kHz, span 220 mm)\n"
                "lower branch MAE = 1.3 mm (330-600 kHz, span 19.8 mm)    "
                "topology matches (single closed band)")

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    update_verdict(cd, "verification/second_mode/sean_m5p35/overlay.png")
    written.append(("verification/second_mode/sean_m5p35/overlay.png",
                    "M5.35 dimensional 2nd-mode neutral curve: pyMack lower/upper "
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

    fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.15, 1.0, 0.85))

    ax.fill_between(R, 0, sigma, where=(sigma > 0), color=PYMACK_BLUE,
                    alpha=0.12, label="pyMack unstable band ($\\sigma>0$)")
    ax.plot(R, sigma, "-o", color=PYMACK_BLUE, lw=2.6, ms=5.5, mfc=PYMACK_BLUE,
            label="pyMack spatial growth  $\\sigma=-\\alpha_i$")
    ax.axhline(0, color="black", lw=1.0)

    ax.axvline(bI_pm, color=PYMACK_GREEN, lw=2.4, ls="-",
               label=f"pyMack Branch I (R={bI_pm:.0f})")
    ax.axvline(bII_pm, color=PYMACK_GREEN, lw=2.4, ls="-",
               label=f"pyMack Branch II (R={bII_pm:.0f})")
    ax.axvline(bI_rf, color=REF_ORANGE, lw=2.2, ls="--",
               label=f"Ma & Zhong Branch I (R={bI_rf:.0f})")
    ax.axvline(bII_rf, color=REF_ORANGE, lw=2.2, ls="--",
               label=f"Ma & Zhong Branch II (R={bII_rf:.1f})")

    ax.set_xlabel("Reynolds number  R = $\\sqrt{Re_x}$")
    ax.set_ylabel("Spatial growth rate  $\\sigma = -\\alpha_i$")
    ax.set_title(f"M4.5 2nd-mode growth at fixed F={F:.1e}\nVERDICT: agrees ($\\leq$3.4%)",
                 color="#333333")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(R.min(), R.max())
    ax.margins(y=0.05)
    outside_legend(ax)

    bottom_note(fig,
                "Branch I:  pyMack 831 vs M&Z 806  ($+3.1\\%$)    "
                "Branch II: pyMack 1033 vs M&Z 999.6 ($+3.4\\%$)    "
                "topology: one closed band, two neutral points")

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    update_verdict(cd, "verification/second_mode/mazhong2003_m4p5/overlay.png")
    written.append(("verification/second_mode/mazhong2003_m4p5/overlay.png",
                    "M4.5 2nd-mode spatial growth sigma=-alpha_i vs R at fixed F=2.2e-4: "
                    "pyMack lobe (solid) with its two neutral crossings vs Ma&Zhong Branch "
                    "I=806 / II=999.6 (dashed lines); verdict agrees (~3%)."))


# =============================================================================
# 3. first_mode/mack_fig10_1_m{16,22}  -- first-mode neutral loop F*1e4 vs R
# =============================================================================
def plot_mack_fig10_1(mtag, mlabel, headline):
    cd = os.path.join(ROOT, "first_mode", f"mack_fig10_1_m{mtag}")
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

    sd = np.argsort(x_dep)
    ax.plot(x_dep[sd], y_dep[sd], ":", color=GREY, lw=1.6, alpha=0.8,
            label="old (Dunn-Lin) ref [deprecated]")

    sl = np.argsort(R_ce[lo_mask])
    su = np.argsort(R_ce[up_mask])
    ax.plot(R_ce[lo_mask][sl], F_ce[lo_mask][sl], "--", color=REF_ORANGE, lw=2.0,
            marker="o", mfc="none", mec=REF_ORANGE, ms=5,
            label="Mack Complete-Eqn lower")
    ax.plot(R_ce[up_mask][su], F_ce[up_mask][su], "--", color=REF_VERM, lw=2.0,
            marker="s", mfc="none", mec=REF_VERM, ms=5,
            label="Mack Complete-Eqn upper")
    if nose_mask.any():
        ax.plot(R_ce[nose_mask], F_ce[nose_mask], "*", color=REF_ORANGE, ms=14,
                mfc="none", mec=REF_ORANGE, mew=1.6, label="Mack nose (critical R)")

    m_lo = np.isfinite(Flo)
    m_up = np.isfinite(Fup)
    ax.plot(R_pm[m_lo], Flo[m_lo], "-", color=PYMACK_BLUE, lw=2.6,
            marker="o", ms=6, label="pyMack lower (onset)")
    ax.plot(R_pm[m_up], Fup[m_up], "-", color=PYMACK_GREEN, lw=2.6,
            marker="^", ms=6, label="pyMack upper (cutoff)")

    ax.set_xlabel("Reynolds number  R = $\\sqrt{Re_x}$")
    ax.set_ylabel("Frequency  $F \\times 10^{4}$")
    ax.set_title(f"Mack Fig 10.1  first-mode neutral loop  {mlabel}\n"
                 f"VERDICT: disagrees ({headline})", color="#333333")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.05)
    outside_legend(ax)

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    rel = f"verification/first_mode/mack_fig10_1_m{mtag}/overlay.png"
    update_verdict(cd, rel)
    written.append((rel,
                    f"Mack Fig10.1 {mlabel} first-mode neutral loop F*1e4 vs R: pyMack "
                    f"lower/upper branches (solid) sit wider/higher than Mack's "
                    f"Complete-Equations loop (dashed, hollow) + faint old Dunn-Lin ref; "
                    f"verdict disagrees ({headline})."))


# =============================================================================
# 4. first_mode/ozgen_m{2,3,4,6}  -- alpha vs Re neutral locus + c_i>0 lobe
# =============================================================================
def plot_ozgen(m, headline):
    cd = os.path.join(ROOT, "first_mode", f"ozgen_m{m}")
    pm = load_csv(os.path.join(cd, f"pymack_ozgen_M{m}_neutral.csv"))
    rf = load_csv(os.path.join(cd, f"reference_ozgen_M{m}_neutral.csv"))
    grid = load_csv(os.path.join(cd, f"pymack_ozgen_M{m}_ci_grid.csv"))

    Re_pm = fcol(pm, "Re_L")
    al_pm = fcol(pm, "alpha_neutral_pymack")
    br = [r["branch"].strip() for r in pm]
    lo = np.array([b == "lower" for b in br])
    up = np.array([b == "upper" for b in br])

    x_rf = fcol(rf, "x")   # Re
    y_rf = fcol(rf, "y")   # alpha

    gRe = fcol(grid, "Re_L")
    gAl = fcol(grid, "alpha_L")
    gCi = fcol(grid, "c_i")
    uRe = np.unique(gRe)
    uAl = np.unique(gAl)
    Z = np.full((len(uAl), len(uRe)), np.nan)
    ri = {v: i for i, v in enumerate(uRe)}
    ai = {v: i for i, v in enumerate(uAl)}
    for re_, al_, ci_ in zip(gRe, gAl, gCi):
        Z[ai[al_], ri[re_]] = ci_
    RR, AA = np.meshgrid(uRe, uAl)

    fig, ax = plt.subplots(figsize=(10.8, 6.9), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.13, 1.0, 0.87))

    Zm = np.ma.masked_invalid(Z)
    try:
        ax.contourf(RR, AA, Zm, levels=[0, Zm.max()],
                    colors=[PYMACK_BLUE], alpha=0.13)
        ax.contour(RR, AA, Zm, levels=[0.0], colors=[PYMACK_BLUE],
                   linewidths=1.2, alpha=0.5)
    except Exception:
        pass

    sl = np.argsort(Re_pm[lo])
    su = np.argsort(Re_pm[up])
    ax.plot(Re_pm[lo][sl], al_pm[lo][sl], "-", color=PYMACK_BLUE, lw=2.4,
            marker="o", ms=5, label="pyMack lower (onset)")
    ax.plot(Re_pm[up][su], al_pm[up][su], "-", color=PYMACK_GREEN, lw=2.4,
            marker="^", ms=5, label="pyMack upper (cutoff)")

    sr = np.argsort(x_rf)
    ax.plot(x_rf[sr], y_rf[sr], "--", color=REF_ORANGE, lw=2.2,
            marker="o", mfc="none", mec=REF_ORANGE, ms=6,
            label="Ozgen & Kircali digitized arch")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=PYMACK_BLUE, alpha=0.13))
    labels.append("pyMack unstable region ($c_i>0$)")

    ax.set_xlabel("Reynolds number  $Re_L$")
    ax.set_ylabel("Wavenumber  $\\alpha_L$")
    ax.set_title(f"Ozgen & Kircali Fig 3  first-mode neutral curve  M{m}\n"
                 f"VERDICT: disagrees (median |$\\Delta\\alpha$|/$\\alpha$ = {headline})",
                 color="#333333")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.05)
    outside_legend(ax, handles, labels)

    bottom_note(fig, "open-lobe (pyMack) vs closed-arch (paper) topology")

    out = os.path.join(cd, "overlay.png")
    save(fig, out)
    rel = f"verification/first_mode/ozgen_m{m}/overlay.png"
    update_verdict(cd, rel)
    written.append((rel,
                    f"Ozgen M{m} first-mode neutral alpha vs Re_L: pyMack lower/upper "
                    f"locus (solid) + faint c_i>0 unstable lobe vs the digitized closed "
                    f"arch (dashed, hollow); open-lobe vs closed-arch; disagrees "
                    f"(median {headline})."))


if __name__ == "__main__":
    plot_sean()
    plot_mazhong()
    plot_mack_fig10_1("1p6", "M=1.6", "loop median ~37%")
    plot_mack_fig10_1("2p2", "M=2.2", "loop median ~2x / 128%")
    plot_ozgen(2, "15.8%")
    plot_ozgen(3, "15.8%")
    plot_ozgen(4, "16.6%")
    plot_ozgen(6, "23.9%")
    print("WROTE:")
    for rel, desc in written:
        print(f"  {rel}\n     {desc}")
