#!/usr/bin/env python3
"""Lightweight, reproducible replot of the Mack Fig. 10.3 / 10.4 first-mode
verification overlays.

VISUAL-QA replot ONLY. This script does NOT recompute any eigenvalue / growth
sweep. It reads each case's COMMITTED data files (pymack_*.json/csv +
reference_*.csv / reference_data/digitized/...) and regenerates a clean
overlay.png in place, with the legend and the condition annotation placed in
genuinely empty regions (never over data), uniform styling, and the font sizes
required by the workspace rule (axis labels >=14, ticks >=12, titles >=16,
legend >=11).

Conventions (verified against the committed data and the original heavier
generators):
  * Fig 10.3 : reference CSV is (x = R*1e-2, y = omega_i*1e3).
               Our LST code -> plot (R*1e-2, omega_i_max*1e3).
               M1.3 Our LST code is a CSV with columns R, omega_i_max (+ Table-10.1
               anchor stars); M2.2 / M3.0 Our LST code is JSON {"rows":[{R, omega_i_max}]}.
  * Fig 10.4 : reference CSV is (x = R, y = omega_i*1e3).
               Our LST code -> plot (R, omega_i_max*1e3). Our LST code JSON is a flat list
               of {R, omega_i_max}.

Style: Our LST code = solid line + filled markers; reference = dashed line + hollow
markers; both clearly labelled. Verdict + headline metric live in the title.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Repo layout
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent          # .../MS_LST
GROWTH = REPO / "verification" / "first_mode"
DIGITIZED = REPO / "reference_data" / "digitized"

# Uniform style ---------------------------------------------------------------
# Readability convention (workspace): reference/digitized paper data = RED and
# slightly LARGER markers; Our LST code computed curve = BLUE and slightly THINNER line.
PYMACK_COLOR = "#000000"     # black, dashed, thick  (Our LST code)
REF_COLOR = "#d62728"        # red, dashed + hollow (reference/digitized)
ANCHOR_COLOR = "#e8741c"     # orange star (Table 10.1 anchor)

FS_TITLE = 18
FS_LABEL = 19
FS_TICK = 16
FS_LEGEND = 15
FS_ANNOT = 12

plt.rcParams.update({
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
})


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------
def load_reference_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column (x, y) reference CSV -> (x, y) arrays."""
    xs, ys = [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.asarray(xs, float), np.asarray(ys, float)


def load_pymack_curve_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Our LST code curve JSON. Accepts either {"rows":[...]} (Fig 10.3
    self-seed) or a flat list (Fig 10.4). Returns (R, omega_i_max)."""
    obj = json.loads(path.read_text())
    rows = obj["rows"] if isinstance(obj, dict) else obj
    R, oi = [], []
    for r in rows:
        val = r.get("omega_i_max")
        R.append(float(r["R"]))
        oi.append(float(val) if val is not None else np.nan)
    return np.asarray(R, float), np.asarray(oi, float)


def load_pymack_overlay_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the M1.3 Our LST code overlay CSV -> (R, omega_i_max)."""
    R, oi = [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            R.append(float(row["R"]))
            oi.append(float(row["omega_i_max"]))
    return np.asarray(R, float), np.asarray(oi, float)


# ----------------------------------------------------------------------------
# Shared plotting helpers
# ----------------------------------------------------------------------------
def _style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.margins(x=0.03)


def _expand_ylim(ax, y_all):
    """Add headroom/footroom so curves+markers aren't jammed on the frame."""
    finite = y_all[np.isfinite(y_all)]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax.set_ylim(lo - 0.10 * span, hi + 0.14 * span)


# ----------------------------------------------------------------------------
# Fig 10.3 cases
# ----------------------------------------------------------------------------
def make_fig10_3(case_dir: Path, ref_name: str, pymack_kind: str,
                 title_l1: str, condition_note: str,
                 anchor_csv: Path | None = None):
    """Render one Fig 10.3 overlay. x = R*1e-2, y = omega_i*1e3.

    pymack_kind: "csv" (M1.3) or "json" (M2.2 / M3.0).
    condition_note is placed BELOW the axes (empty figure margin) so it never
    overlaps the steeply-rising reference curve in the top-left.
    """
    verdict = json.loads((case_dir / "verdict.json").read_text())

    xr, yr = load_reference_csv(case_dir / ref_name)
    if pymack_kind == "csv":
        Rp, oip = load_pymack_overlay_csv(case_dir / "pymack_mack_fig10_3_overlay.csv")
    else:
        Rp, oip = load_pymack_curve_json(case_dir / "pymack_curve.json")
    xp = Rp * 1e-2
    yp = oip * 1e3

    fig, ax = plt.subplots(figsize=(9.4, 6.0))

    # reference: red, larger hollow markers (points emphasised) + thin dashed guide
    ax.plot(xr, yr, ls="--", lw=1.5, color=REF_COLOR, marker="o",
            ms=5.5, mfc="none", mec=REF_COLOR, mew=1.4,
            label=r"Mack (1984) Fig. 10.3 (digitized)", zorder=3)
    # Our LST code: blue, thinner line + smaller filled markers
    ax.plot(xp, yp, ls="--", lw=3.6, color=PYMACK_COLOR, marker="o",
            ms=5.5, mfc=PYMACK_COLOR, mec=PYMACK_COLOR,
            label="Our LST code", zorder=4)

    # Table-10.1 anchor stars (M1.3 only)
    if anchor_csv is not None and anchor_csv.exists():
        # anchor points are a subset of the Our LST code curve; mark R = 500 and 1500
        # (the Layer-3 validated anchors used in the original figure).
        for Ranchor in (500.0, 1500.0):
            idx = int(np.argmin(np.abs(Rp - Ranchor)))
            ax.plot(xp[idx], yp[idx], marker="*", ms=20,
                    mfc=ANCHOR_COLOR, mec="black", mew=0.8,
                    ls="none", zorder=6,
                    label=("Mack Table 10.1 (6th order) anchor"
                           if Ranchor == 500.0 else None))

    _style_axes(ax, r"$R \times 10^{-2}$",
                r"$\omega_i \times 10^{3}$  (Mack $L^{*}$ scale)")

    y_all = np.concatenate([yr, yp[np.isfinite(yp)]])
    _expand_ylim(ax, y_all)
    ax.set_xlim(0.0, max(xr.max(), xp.max()) * 1.05)

    # Case-identifying title only (no verdict word, no metric).
    ax.set_title(title_l1, fontsize=FS_TITLE, pad=10)

    # Legend: bottom-right is the genuinely empty region for these
    # rising-then-plateauing curves (data lives top-left -> top-right).
    ax.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="0.6")

    fig.subplots_adjust(left=0.115, right=0.975, top=0.90, bottom=0.115)

    out = case_dir / "overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Fig 10.4 cases
# ----------------------------------------------------------------------------
def make_fig10_4(case_dir: Path, mach_label: str):
    """Render one Fig 10.4 overlay. x = R, y = omega_i*1e3.

    Reference lives in reference_data/digitized/. Our LST code is a flat list JSON.
    Curves are saturating (rise then plateau), so the genuinely empty region is
    the lower-right; legend goes there.
    """
    verdict = json.loads((case_dir / "verdict.json").read_text())
    ref_path = REPO / verdict["artifacts"]["reference"]

    xr, yr = load_reference_csv(ref_path)
    Rp, oip = load_pymack_curve_json(case_dir / "pymack_curve.json")
    xp = Rp
    yp = oip * 1e3

    fig, ax = plt.subplots(figsize=(9.4, 6.0))

    ax.plot(xr, yr, ls="--", lw=1.5, color=REF_COLOR, marker="o",
            ms=5.5, mfc="none", mec=REF_COLOR, mew=1.4,
            label="Mack (1984) Fig. 10.4 (digitized)", zorder=3)
    ax.plot(xp, yp, ls="--", lw=3.6, color=PYMACK_COLOR, marker="s",
            ms=5.5, mfc=PYMACK_COLOR, mec=PYMACK_COLOR,
            label="Our LST code", zorder=4)

    _style_axes(ax, r"$R$",
                r"$\omega_i \times 10^{3}$  (max over $\alpha,\psi$)")

    y_all = np.concatenate([yr, yp[np.isfinite(yp)]])
    _expand_ylim(ax, y_all)
    ax.set_xlim(0.0, max(xr.max(), xp.max()) * 1.04)

    ax.set_title(f"Mack (1984) Fig. 10.4:  $M={mach_label}$  oblique first mode",
                 fontsize=FS_TITLE, pad=10)

    # Our LST code lies BELOW the reference for these disagree/acceptable cases, and
    # both curves climb to the upper-right, so the lower-right corner is empty.
    ax.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.95,
              edgecolor="0.6")

    fig.subplots_adjust(left=0.115, right=0.975, top=0.91, bottom=0.115)

    out = case_dir / "overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    outputs = []

    # --- Fig 10.3 -----------------------------------------------------------
    outputs.append(make_fig10_3(
        GROWTH / "mack_fig10_3_m1p3",
        ref_name="reference_mack_fig10_3_M13_paper_psi45.csv",
        pymack_kind="csv",
        title_l1=r"Mack (1984) Fig. 10.3:  $M =1.3$, $\psi = 45^{\circ}$ first mode",
        condition_note=("condition: table_11_1 ($T_e^{*}\\!\\approx\\!228$ K, adiabatic); "
                        "exact 8$\\times$8 Appendix-A shooting"),
        anchor_csv=GROWTH / "mack_fig10_3_m1p3" / "pymack_mack_fig10_3_overlay.csv",
    ))

    outputs.append(make_fig10_3(
        GROWTH / "mack_fig10_3_m2p2",
        ref_name="reference_mack_ch10_fig10_3_M22_paper_psi45.csv",
        pymack_kind="json",
        title_l1=r"Mack (1984) Fig. 10.3:  $M =2.2$, $\psi = 45^{\circ}$ first mode",
        condition_note=("condition: table_11_1 ($T_e^{*}\\!\\approx\\!155$ K, adiabatic); "
                        "self-seeded 8$\\times$8 shooting, "
                        "$y_{max}\\!=\\!4\\,\\delta^{*}/L^{*}$, conv-checked"),
        anchor_csv=None,
    ))

    outputs.append(make_fig10_3(
        GROWTH / "mack_fig10_3_m3p0",
        ref_name="reference_mack_ch10_fig10_3_M30_paper_psi60.csv",
        pymack_kind="json",
        title_l1=r"Mack (1984) Fig. 10.3:  $M =3.0$, $\psi = 60^{\circ}$ first mode",
        condition_note=("condition: table_11_1 ($T_e^{*}\\!\\approx\\!109$ K, adiabatic); "
                        "self-seeded 8$\\times$8 shooting, "
                        "$y_{max}\\!=\\!4\\,\\delta^{*}/L^{*}$, conv-checked"),
        anchor_csv=None,
    ))

    # --- Fig 10.4 -----------------------------------------------------------
    for case, mach in (("mack_fig10_4_M45", "4.5"),
                       ("mack_fig10_4_M58", "5.8"),
                       ("mack_fig10_4_M70", "7.0"),
                       ("mack_fig10_4_M100", "10.0")):
        outputs.append(make_fig10_4(GROWTH / case, mach))

    for o in outputs:
        print(o)


if __name__ == "__main__":
    main()
