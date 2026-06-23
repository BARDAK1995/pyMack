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


def main():
    g = load_grid(); oz = ozgen()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    cols = {"second": "#d55e00", "first": "#0072b2"}
    for m, (R, om, Z) in g.items():
        first = True
        for s in contour0(R, om, Z):
            ax.plot(s[:, 0], s[:, 1], "-", color=cols[m], lw=2.6,
                    label=(f"pyMack {m} mode neutral" if first else None)); first = False
    for (m, br), arr in oz.items():
        a = np.array(sorted(arr.tolist()))
        ax.plot(a[:, 0], a[:, 1], "o", mfc="none",
                mec=("#b34700" if m == "second" else "#004c80"), mew=1.5, ms=6)
    ax.plot([], [], "o", mfc="none", mec="#333", label="Ma & Zhong (2003) Fig. 15 (digitized)")
    Rl = np.linspace(0, 2000, 50)
    ax.plot(Rl, Rl * 2.2e-4, ":", color="0.45", lw=1.4)
    ax.plot(Rl, Rl * 0.6e-4, ":", color="0.45", lw=1.4)
    ax.text(1320, 0.258, r"$F=2.2\times10^{-4}$", fontsize=11)
    ax.text(1520, 0.108, r"$F=0.6\times10^{-4}$", fontsize=11)
    ax.set_xlim(0, 2000); ax.set_ylim(0, 0.28)
    ax.set_xlabel("$R=\\sqrt{Re_x}$", fontsize=15); ax.set_ylabel(r"$\omega$", fontsize=15)
    ax.set_title("pyMack vs Ma & Zhong (2003) Fig. 15 — M=4.5 neutral curves", fontsize=14)
    ax.tick_params(labelsize=12); ax.legend(fontsize=11, loc="lower right"); ax.grid(True, alpha=0.25)
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
