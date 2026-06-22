"""Overlay three neutral curves at ~Mach 6 in the (R_L, F x 1e4) plane:

  1. Ozgen & Kircali (2008) Fig 3, M=6  -- FIRST mode (temporal). Digitized as
     (Re_L, alpha_L); converted to reduced frequency F = omega_L/R = alpha_L*c_r/R
     using c_r interpolated from pyMack's own Ozgen M6 c_i grid (data-driven, not
     assumed).
  2. pyMack M=5.85 N2  -- SECOND (Mack) mode, spatial. Native (F, lower/upper
     neutral R_L) from the APS production neutral envelope. Exact, no conversion.
  3. pyMack M=6.0 air  -- SECOND (Mack) mode, spatial. Same, from the canonical
     Mach 6 production neutral envelope.

HONEST NOTE: curves 1 (first mode) and 2/3 (second mode) are DIFFERENT
instabilities; this overlay shows where each sits at ~Mach 6, not a like-for-like
agreement test. Reads pre-computed data only (no recompute).
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
VER = REPO / "verification"
CH = REPO / "chapters/ozgen_kircali_2008/results"

# Okabe-Ito palette
C_OZ, C_585, C_6 = "#d55e00", "#0072b2", "#009e73"


def read_cols(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = [{k: (float(v) if v not in ("", "nan", "NaN") else np.nan)
                 for k, v in row.items()} for row in r]
    return rows


def ozgen_m6_RF():
    """Ozgen M6 first-mode neutral as (Re_L, F*1e4), c_r from the c_i grid."""
    neu = []
    with open(VER / "first_mode/ozgen_m6/reference_ozgen_M6_neutral.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            try:
                neu.append((float(row[0]), float(row[1])))
            except ValueError:
                pass
    neu = np.array(neu)            # (Re, alpha)
    g = []
    with open(VER / "first_mode/_ozgen_compute/ozgen_M6_ci_grid.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                g.append((float(row["Re_L"]), float(row["alpha_L"]), float(row["c_r"])))
            except (ValueError, KeyError):
                pass
    g = np.array(g)
    cr = griddata(g[:, :2], g[:, 2], neu, method="linear")
    nn = griddata(g[:, :2], g[:, 2], neu, method="nearest")
    cr = np.where(np.isfinite(cr), cr, nn)
    F1e4 = neu[:, 1] * cr / neu[:, 0] * 1e4      # F = alpha*c_r/Re
    order = np.argsort(neu[:, 0])
    return neu[order, 0], F1e4[order], float(np.nanmedian(cr))


def pymack_env(path):
    rows = read_cols(path)
    F = np.array([r["F_x1e4"] for r in rows])
    lo = np.array([r["lower_neutral_R_L"] for r in rows])
    hi = np.array([r["upper_neutral_R_L"] for r in rows])
    return F, lo, hi


def closed_loop(F, lo, hi):
    """Trace up the lower branch then down the upper branch -> closed neutral loop."""
    okl = np.isfinite(lo) & np.isfinite(F)
    okh = np.isfinite(hi) & np.isfinite(F)
    Fl, Rl = F[okl], lo[okl]
    Fh, Rh = F[okh], hi[okh]
    sl = np.argsort(Fl); sh = np.argsort(Fh)[::-1]
    R = np.concatenate([Rl[sl], Rh[sh]])
    Ff = np.concatenate([Fl[sl], Fh[sh]])
    return R, Ff


def main():
    oz_R, oz_F, oz_cr = ozgen_m6_RF()
    F585, lo585, hi585 = pymack_env(CH / "aps_dimensional_production/amplification/spatial_fixed_frequency_neutral_envelope.csv")
    F6, lo6, hi6 = pymack_env(CH / "verification_20260608/runner_production/amplification/spatial_fixed_frequency_neutral_envelope.csv")
    R585, FF585 = closed_loop(F585, lo585, hi585)
    R6, FF6 = closed_loop(F6, lo6, hi6)

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.plot(oz_R, oz_F, "o--", color=C_OZ, mfc="none", ms=6, lw=2,
            label=f"Özgen & Kırcalı (2008) M=6 — 1st mode (temporal), c̄ᵣ≈{oz_cr:.2f}")
    ax.plot(R585, FF585, "-", color=C_585, lw=2.4,
            label="pyMack M=5.85 N₂ — 2nd mode (spatial)")
    ax.plot(R6, FF6, "-", color=C_6, lw=2.4,
            label="pyMack M=6.0 air — 2nd mode (spatial)")

    ax.set_xlabel(r"Reynolds number  $R_L=\sqrt{Re_x}$", fontsize=15)
    ax.set_ylabel(r"reduced frequency  $F\times10^{4}\;(F=\omega\nu_e/U_e^2)$", fontsize=15)
    ax.set_title("Neutral curves near Mach 6: Özgen 1st mode vs pyMack 2nd mode\n"
                 "(different instabilities — shown in a common (R, F) plane)",
                 fontsize=16)
    ax.tick_params(labelsize=13)
    ax.margins(0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0., frameon=True)
    fig.text(0.012, 0.012,
             "First-mode (Özgen) and second-mode (pyMack) neutral curves are distinct "
             "instabilities; overlay shows where each sits at ~M6, not a like-for-like test. "
             "Özgen converted via F=α·cᵣ/R with cᵣ from pyMack's Özgen-M6 grid.",
             fontsize=9.5, wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = Path(__file__).resolve().parent / "ozgenM6_vs_pymack_M5p85_M6_neutral.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  Ozgen M6: {len(oz_R)} pts, Re {oz_R.min():.0f}-{oz_R.max():.0f}, "
          f"F*1e4 {np.nanmin(oz_F):.2f}-{np.nanmax(oz_F):.2f}, median c_r={oz_cr:.3f}")
    print(f"  pyMack M5.85: F*1e4 {np.nanmin(F585):.2f}-{np.nanmax(F585):.2f}, "
          f"R {np.nanmin(lo585):.0f}-{np.nanmax(hi585):.0f}")
    print(f"  pyMack M6:    F*1e4 {np.nanmin(F6):.2f}-{np.nanmax(F6):.2f}, "
          f"R {np.nanmin(lo6):.0f}-{np.nanmax(hi6):.0f}")


if __name__ == "__main__":
    main()
