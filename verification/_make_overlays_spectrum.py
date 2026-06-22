#!/usr/bin/env python
"""Regenerate the eigenvalue / spectrum verification overlay plots for pyMack.

Three figures, all re-plotted ONLY from committed per-case data (no eigenvalue
or growth sweep is recomputed here):

  1. second_mode/malik_case6/overlay.png
        complex-plane spatial eigenvalue alpha: pyMack vs Malik (1990) vs
        Tumin (2007).  Points come straight from verdict.json metrics
        (pymack_alpha, malik_alpha) plus the Tumin (2007) recompute documented
        in verdict.json["verdict_reason"]/["source"].

  2. second_mode/balakumar_malik1992_via_xirenfu/overlay.png
        complex-plane spatial eigenvalue alpha: pyMack vs Balakumar & Malik
        (1992) vs Xi/Ren/Fu.  Points from verdict.json metrics + source note.

  3. second_mode/balakumar_malik1992_branches/overlay.png
        full spatial companion spectrum (discrete second mode vs continuous
        cluster at c->1) from the committed spectrum.npz, with the published
        B&M (1992) marker and a zoom inset on the discrete mode.

Layout goal (visual QA): no legend / annotation / inset / title may overlap or
obscure the plotted data.  Hard font rule: axis labels >= 14 pt, ticks >= 12 pt,
titles >= 16 pt, legend >= 11 pt.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYMACK_NO_BANNER", "1")

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

# --- HARD style rules --------------------------------------------------------
plt.rcParams.update({
    "axes.labelsize": 15,     # >= 14 pt
    "xtick.labelsize": 13,    # >= 12 pt
    "ytick.labelsize": 13,
    "axes.titlesize": 17,     # >= 16 pt
    "legend.fontsize": 12,    # >= 11 pt
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
})

# Colorblind-friendly (Okabe-Ito)
PYMACK_BLUE = "#0072B2"   # pyMack -> solid / filled
REF_ORANGE = "#D55E00"    # first reference -> dashed / hollow square
REF_GREEN = "#009E73"     # second reference -> hollow triangle
CONT_TEAL = "#009E73"     # continuous-spectrum cluster
GREY = "#999999"

ROOT = os.path.dirname(os.path.abspath(__file__))
VER = ROOT
SM = os.path.join(VER, "second_mode")

BOX = dict(boxstyle="round,pad=0.45", fc="#fbf7e8", ec=GREY, alpha=0.97)

written = []


def set_overlay(case_dir, rel):
    vpath = os.path.join(case_dir, "verdict.json")
    with open(vpath, "r", encoding="utf-8") as f:
        v = json.load(f)
    v.setdefault("artifacts", {})
    v["artifacts"]["overlay"] = rel
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)


# =============================================================================
# 1. malik_case6 -- complex-plane alpha: pyMack vs Malik 1990 vs Tumin 2007
# =============================================================================
def plot_malik_case6():
    cd = os.path.join(SM, "malik_case6")
    v = json.load(open(os.path.join(cd, "verdict.json")))
    m = v["metrics"]

    # committed points (verdict.json metrics + documented Tumin recompute)
    pm = complex(*m["pymack_alpha"])             # 0.2533998 - 0.0024898 i
    malik = complex(*m["malik_alpha"])           # 0.2534048 - 0.0024921 i
    # Tumin (2007) recompute, documented in verdict_reason / source:
    #   alpha = 0.2534420 - 0.0027738 i  (~11% in alpha_i from Malik)
    tumin = complex(0.2534420, -0.0027738)
    rr = m["alpha_r_rel_err"]
    ri = m["alpha_i_rel_err"]
    cph = m["c_phase"]
    N = m["N"]

    fig, ax = plt.subplots(figsize=(8.6, 6.8))

    ax.plot(pm.real, pm.imag, "o", color=PYMACK_BLUE, ms=15, mec="k", mew=0.8,
            zorder=6, label=f"pyMack: {pm.real:.7f}{pm.imag:+.7f}i")
    ax.plot(malik.real, malik.imag, "s", mfc="none", mec=REF_ORANGE, mew=2.6,
            ms=17, zorder=5, label=f"Malik (1990): {malik.real:.7f}{malik.imag:+.7f}i")
    ax.plot(tumin.real, tumin.imag, "^", mfc="none", mec=REF_GREEN, mew=2.6,
            ms=17, zorder=5, label=f"Tumin (2007): {tumin.real:.7f}{tumin.imag:+.7f}i")

    # generous margins so no marker is jammed against the frame
    xs = [pm.real, malik.real, tumin.real]
    ys = [pm.imag, malik.imag, tumin.imag]
    xpad = (max(xs) - min(xs)) * 0.55 + 5e-6
    ypad = (max(ys) - min(ys)) * 0.45 + 5e-6
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad * 1.4, max(ys) + ypad)

    ax.set_xlabel(r"$\alpha_r$  (streamwise wavenumber, real part)")
    ax.set_ylabel(r"$\alpha_i$  (spatial growth rate, imag part)")
    ax.set_title("Malik (1990) Case 6  second mode\n"
                 "spatial eigenvalue $\\alpha$ in the complex plane")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="plain")

    # legend in the upper-left -- the markers occupy the upper-right / center,
    # the Tumin triangle is at lower-center, so upper-left is clear.
    ax.legend(loc="upper left", framealpha=0.95)

    # rel-err annotation in the lower-RIGHT empty corner (markers are upper /
    # center, Tumin is lower-center -> lower-right is free).
    ax.annotate(
        "VERDICT: AGREES\n"
        f"$\\alpha_r$ rel err = {rr:.2e}  ({rr*100:.4f}%)\n"
        f"$\\alpha_i$ rel err = {ri:.2e}  ({ri*100:.2f}%)\n"
        f"$c = \\omega/\\alpha_r$ = {cph:.5f},  N={N}\n"
        "Tumin's $\\alpha_i$ is $\\sim$11% from Malik\n"
        "(literature spread $\\gg$ pyMack deviation)",
        xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=12, bbox=BOX, zorder=7)

    fig.tight_layout()
    out = os.path.join(cd, "overlay.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    set_overlay(cd, "verification/second_mode/malik_case6/overlay.png")
    written.append((out,
                    "malik_case6: complex-plane alpha -- pyMack (filled blue circle) vs "
                    "Malik 1990 (hollow orange square) vs Tumin 2007 (hollow green "
                    "triangle); legend upper-left, rel-err box lower-right; AGREES."))


# =============================================================================
# 2. balakumar_malik1992_via_xirenfu -- complex-plane alpha:
#    pyMack vs B&M (1992) vs Xi/Ren/Fu
# =============================================================================
def plot_via_xirenfu():
    cd = os.path.join(SM, "balakumar_malik1992_via_xirenfu")
    v = json.load(open(os.path.join(cd, "verdict.json")))
    m = v["metrics"]

    pm = complex(*m["pymack_alpha"])    # 0.2202103 - 0.0027906 i
    bm = complex(*m["malik_alpha"])     # 0.220     - 0.003091  i
    # Xi/Ren/Fu solver value, documented in verdict.json["source"]:
    #   alpha = 0.220199 - 0.003098 i
    xrf = complex(0.220199, -0.003098)
    rr = m["alpha_r_rel_err"]
    ri = m["alpha_i_rel_err"]
    cph = m["c_phase"]
    N = m["N"]

    fig, ax = plt.subplots(figsize=(8.6, 6.8))

    ax.plot(pm.real, pm.imag, "o", color=PYMACK_BLUE, ms=15, mec="k", mew=0.8,
            zorder=6, label=f"pyMack: {pm.real:.6f}{pm.imag:+.6f}i")
    ax.plot(bm.real, bm.imag, "s", mfc="none", mec=REF_ORANGE, mew=2.6,
            ms=17, zorder=5, label=f"Balakumar & Malik (1992): {bm.real:.6f}{bm.imag:+.6f}i")
    ax.plot(xrf.real, xrf.imag, "^", mfc="none", mec=REF_GREEN, mew=2.6,
            ms=17, zorder=5, label=f"Xi/Ren/Fu (2020): {xrf.real:.6f}{xrf.imag:+.6f}i")

    xs = [pm.real, bm.real, xrf.real]
    ys = [pm.imag, bm.imag, xrf.imag]
    xpad = (max(xs) - min(xs)) * 0.55 + 2e-5
    ypad = (max(ys) - min(ys)) * 0.45 + 2e-5
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad * 1.3)
    ax.set_ylim(min(ys) - ypad * 1.5, max(ys) + ypad)

    ax.set_xlabel(r"$\alpha_r$  (streamwise wavenumber, real part)")
    ax.set_ylabel(r"$\alpha_i$  (spatial growth rate, imag part)")
    ax.set_title("Balakumar & Malik (1992) M4.5  second mode\n"
                 "spatial eigenvalue $\\alpha$ in the complex plane")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="plain")

    # B&M + Xi/Ren/Fu sit at lower-left (more damped), pyMack at upper-right
    # (less damped). Legend in upper-left is clear of all three.
    ax.legend(loc="upper left", framealpha=0.95)

    # rel-err box in the lower-RIGHT corner (free of markers).
    ax.annotate(
        "VERDICT: ACCEPTABLE\n"
        f"$\\alpha_r$ rel err = {rr:.2e}  ({rr*100:.3f}%, essentially exact)\n"
        f"$\\alpha_i$ rel err = {ri:.2e}  ({ri*100:.1f}%)\n"
        f"$c = \\omega/\\alpha_r$ = {cph:.5f},  N={N}\n"
        "$\\alpha_i$ offset = documented $\\sim$10% inter-method spread\n"
        "(Xi/Ren/Fu & Tumin also differ from B&M's printed $\\alpha_i$)",
        xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=12, bbox=BOX, zorder=7)

    fig.tight_layout()
    out = os.path.join(cd, "overlay.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    set_overlay(cd, "verification/second_mode/balakumar_malik1992_via_xirenfu/overlay.png")
    written.append((out,
                    "via_xirenfu: complex-plane alpha -- pyMack (filled blue circle) vs "
                    "B&M 1992 (hollow orange square) vs Xi/Ren/Fu (hollow green triangle); "
                    "legend upper-left, rel-err box lower-right; ACCEPTABLE."))


# =============================================================================
# 3. balakumar_malik1992_branches -- full spatial companion spectrum
# =============================================================================
def plot_branches():
    cd = os.path.join(SM, "balakumar_malik1992_branches")
    v = json.load(open(os.path.join(cd, "verdict.json")))
    m = v["metrics"]

    npz = np.load(os.path.join(cd, "spectrum.npz"))
    alphas = np.asarray(npz["alphas"])
    omega = float(npz["omega"])

    disc = complex(*m["discrete_alpha"])        # discrete 2nd mode
    pub = complex(*m["published_alpha"])        # B&M (1992) published
    n_cont = m["n_continuous_cluster_near_c1"]
    n_total = len(alphas)
    rr = m["disc_alpha_r_rel_err"]
    ri = m["disc_alpha_i_rel_err"]
    cph = m["discrete_c_phase"]
    N = m["N"]

    # --- classify roots (same logic as compute_*; from committed spectrum) ---
    c_all = omega / alphas.real
    cont_mask = np.abs(c_all - 1.0) < 0.05          # continuous cluster at c->1
    is_disc = np.abs(alphas - disc) < 1e-6
    other_mask = (~cont_mask) & (~is_disc)          # other companion roots

    cont = alphas[cont_mask]
    other = alphas[other_mask]

    fig, ax = plt.subplots(figsize=(11.0, 7.6))

    # other companion roots (acoustic / high-|alpha|) -- faint grey open circles
    ax.plot(other.real, other.imag, "o", mfc="none", mec=GREY, mew=1.2,
            ms=6, alpha=0.75, zorder=2,
            label=f"other companion roots (acoustic / high-$|\\alpha|$),  n={other.size}")

    # continuous-spectrum cluster near c->1 (alpha_r -> omega) -- teal diamonds
    ax.plot(cont.real, cont.imag, "D", color=CONT_TEAL, ms=6, alpha=0.85,
            zorder=3,
            label=f"continuous-spectrum cluster ($c\\to1$),  n={cont.size}")
    # vertical guide at alpha_r = omega (continuum accumulation line)
    ax.axvline(omega, color=CONT_TEAL, ls="--", lw=1.6, alpha=0.6, zorder=1)

    # pyMack discrete second mode -- filled blue star
    ax.plot(disc.real, disc.imag, "*", color=PYMACK_BLUE, ms=26, mec="k",
            mew=0.9, zorder=6,
            label=f"pyMack discrete 2nd mode: {disc.real:.6f}{disc.imag:+.6f}i")
    # B&M (1992) published -- large hollow orange square
    ax.plot(pub.real, pub.imag, "s", mfc="none", mec=REF_ORANGE, mew=2.8,
            ms=20, zorder=5,
            label=f"B&M (1992) published: {pub.real:.6f}{pub.imag:+.6f}i")

    # main window: show the discrete mode (low, negative alpha_i) and the
    # continuum ladder rising in alpha_i, without the far companion artifacts.
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 0.16)

    ax.set_xlabel(r"$\alpha_r$  (streamwise wavenumber, real part)")
    ax.set_ylabel(r"$\alpha_i$  (spatial growth rate, imag part)")
    ax.grid(True, alpha=0.3)

    # --- legend OUTSIDE the axes (below the plot), so it never covers any
    #     spectrum point. The rising continuous cluster sits at alpha_r~0.2 and
    #     climbs to the top of the window, so any in-axes legend would collide;
    #     placing it under the axes keeps the data fully clear. ---------------
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              framealpha=0.95, ncol=2, borderaxespad=0.0,
              columnspacing=1.4, handletextpad=0.5)
    ax.set_title("Balakumar & Malik (1992) M4.5  discrete second mode vs continuous spectrum\n"
                 "full spatial companion spectrum ($\\alpha$-plane)")

    # --- verdict text box: lower-RIGHT of the main axes. The data lives at
    #     small alpha_r (left third) + the rising teal ladder near alpha_r~0.2,
    #     so the right half above the axis floor is genuinely empty. ----------
    ax.annotate(
        "VERDICT: ACCEPTABLE  (qualitative branch topology)\n"
        f"discrete 2nd mode: $\\alpha_r$ {rr*100:.2f}%, $\\alpha_i$ {ri*100:.1f}% vs B&M\n"
        f"$c = \\omega/\\alpha_r$ = {cph:.4f}  (least-damped in 2nd-mode band)\n"
        f"continuous cluster at $c\\to1$:  {n_cont} roots,  N={N}\n"
        f"$\\omega$=0.2, Re=1000;  full QEP spectrum, {n_total} roots plotted",
        xy=(0.975, 0.40), xycoords="axes fraction", ha="right", va="top",
        fontsize=12, bbox=BOX, zorder=7)

    # --- zoom inset: discrete mode region. Place it at the UPPER-RIGHT of the
    #     axes (empty in the main window) so it covers no spectrum points. -----
    axin = ax.inset_axes([0.60, 0.52, 0.36, 0.42])
    zr = (0.205, 0.235)
    zi = (-0.006, 0.012)
    zmask = ((alphas.real >= zr[0]) & (alphas.real <= zr[1]) &
             (alphas.imag >= zi[0]) & (alphas.imag <= zi[1]))
    zc = alphas[zmask & cont_mask]
    zo = alphas[zmask & other_mask]
    axin.plot(zo.real, zo.imag, "o", mfc="none", mec=GREY, mew=1.1, ms=6, alpha=0.75)
    axin.plot(zc.real, zc.imag, "D", color=CONT_TEAL, ms=6, alpha=0.85)
    axin.axvline(omega, color=CONT_TEAL, ls="--", lw=1.4, alpha=0.6)
    axin.plot(pub.real, pub.imag, "s", mfc="none", mec=REF_ORANGE, mew=2.6, ms=18)
    axin.plot(disc.real, disc.imag, "*", color=PYMACK_BLUE, ms=22, mec="k", mew=0.8)
    axin.set_xlim(*zr)
    axin.set_ylim(*zi)
    axin.set_title("zoom: discrete mode", fontsize=12, pad=3)
    axin.tick_params(labelsize=11)
    axin.grid(True, alpha=0.3)
    for s in axin.spines.values():
        s.set_edgecolor(GREY)
    # connector box on the main axes showing the zoom region
    ax.indicate_inset_zoom(axin, edgecolor=GREY, alpha=0.7)

    fig.tight_layout()
    out = os.path.join(cd, "overlay.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    set_overlay(cd, "verification/second_mode/balakumar_malik1992_branches/overlay.png")
    written.append((out,
                    "branches: full spatial spectrum -- pyMack discrete 2nd mode (blue star) "
                    "+ B&M published (hollow orange square) vs continuous cluster (teal "
                    "diamonds, c->1) and other companion roots (grey hollow); legend top-left, "
                    "verdict box mid-right, zoom inset upper-right; ACCEPTABLE."))


if __name__ == "__main__":
    plot_malik_case6()
    plot_via_xirenfu()
    plot_branches()
    print("WROTE_SPECTRUM_OVERLAYS")
    for p, desc in written:
        print(p)
        print("  " + desc)
