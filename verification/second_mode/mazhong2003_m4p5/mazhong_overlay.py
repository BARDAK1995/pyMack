"""Overlay pyMack's full first- and second-mode neutral curves (neg_alpha_i=0
contour from the trace grid) on the digitized Ma & Zhong (2003) Fig. 15, in the
(R, omega) plane, with the F=2.2e-4 and 0.6e-4 lines. The companion comparison
for the deck slide."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load_grid():
    import collections
    d = collections.defaultdict(dict)
    with open(HERE / "mazhong_curve_grid.csv") as f:
        for r in csv.DictReader(f):
            v = r["neg_alpha_i"]
            d[r["mode"]][(float(r["R"]), float(r["omega"]))] = (float(v) if v not in ("", "nan") else np.nan)
    out = {}
    for m, dd in d.items():
        R = np.array(sorted({k[0] for k in dd})); om = np.array(sorted({k[1] for k in dd}))
        Z = np.full((len(om), len(R)), np.nan)
        for (Rr, oo), v in dd.items():
            Z[list(om).index(oo), list(R).index(Rr)] = v
        out[m] = (R, om, Z)
    return out


def contour0(R, om, Z):
    Zm = np.where(np.isnan(Z), -9.99, Z)
    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(R, om, Zm, levels=[0.0])
    segs = [s for col in cs.allsegs for s in col if len(s) >= 2]
    plt.close(fig)
    return segs


def ozgen():
    cur = {}
    with open(HERE / "reference_mazhong_fig15.csv") as f:
        for r in csv.DictReader(f):
            cur.setdefault((r["mode"], r["branch"]), []).append((float(r["R"]), float(r["omega"])))
    return {k: np.array(v) for k, v in cur.items()}


def first_cutoff_discrete():
    """pyMack's first-mode CUTOFF branch from the discrete-mode extractor
    (trace_firstmode_discrete.py), replacing the old band-classifier contour."""
    P = HERE / "pymack_firstmode_cutoff_discrete.csv"
    if not P.exists():
        return None
    a = [(float(r["R"]), float(r["omega"])) for r in csv.DictReader(open(P))
         if r["branch"] == "upper"]
    return np.array(sorted(a)) if a else None


def main():
    g = load_grid(); oz = ozgen()
    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    # 1st-mode ONSET region: continuous-spectrum-limited (no clean discrete mode)
    ax.axhspan(0.0, 0.046, color="0.90", alpha=0.7, zorder=0)
    ax.text(70, 0.030, "1st-mode onset:\ncontinuous-spectrum-limited",
            fontsize=10.5, color="0.35", zorder=1)

    # pyMack SECOND-mode neutral loop (grid c_i=0 contour -- matches Fig.15)
    firstlbl = True
    for s in contour0(*g["second"]):
        ax.plot(s[:, 0], s[:, 1], "-", color="#d55e00", lw=2.8,
                label=("pyMack 2nd-mode neutral" if firstlbl else None)); firstlbl = False
    # pyMack FIRST-mode CUTOFF branch (discrete-mode extractor)
    fc = first_cutoff_discrete()
    if fc is not None:
        ax.plot(fc[:, 0], fc[:, 1], "-", color="#0072b2", lw=3.0,
                label="pyMack 1st-mode cutoff (discrete)")

    # Ma & Zhong digitised reference points, by mode
    for (m, br), arr in oz.items():
        a = np.array(sorted(arr.tolist()))
        mk = "s" if m == "second" else "o"
        mec = "#b34700" if m == "second" else "#004c80"
        ax.plot(a[:, 0], a[:, 1], mk, mfc="none", mec=mec, mew=1.5, ms=6, zorder=3)
    ax.plot([], [], "s", mfc="none", mec="#b34700", label="Ma & Zhong 2nd mode (digitised)")
    ax.plot([], [], "o", mfc="none", mec="#004c80", label="Ma & Zhong 1st mode (digitised)")

    Rl = np.linspace(0, 2000, 50)
    ax.plot(Rl, Rl * 2.2e-4, ":", color="0.45", lw=1.4)
    ax.text(1320, 0.258, r"$F=2.2\times10^{-4}$", fontsize=11)
    ax.set_xlim(0, 2000); ax.set_ylim(0, 0.28)
    ax.set_xlabel(r"$R=\sqrt{Re_x}$", fontsize=17); ax.set_ylabel(r"$\omega$", fontsize=17)
    ax.set_title("pyMack vs Ma & Zhong (2003) Fig. 15 — $M=4.5$ neutral curves", fontsize=14)
    ax.tick_params(labelsize=14); ax.legend(fontsize=11.5, loc="lower right"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(HERE / "overlay_fig15_full.png", dpi=160)
    print("wrote overlay_fig15_full.png")
    # report the F=2.2e-4 crossings of pyMack 2nd-mode neutral (should ~ branch I/II)
    R, om, Z = g["second"]
    print("pyMack 2nd-mode unstable omega range per R (neutral band edges):")
    for j, Rr in enumerate(R):
        col = Z[:, j]; un = om[(col > 0) & np.isfinite(col)]
        if un.size:
            print(f"  R={Rr:.0f}: omega_unstable [{un.min():.3f},{un.max():.3f}]")


if __name__ == "__main__":
    main()
