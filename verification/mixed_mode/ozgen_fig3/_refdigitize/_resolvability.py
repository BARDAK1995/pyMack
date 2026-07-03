"""How much of each Ozgen first-mode lobe can pyMack resolve as a clean discrete
mode (eigenfunction decay + y_max-stability)?  Samples mid-lobe alpha at several
Re for M3, M4, M6 and reports the discrete mode (or CS-contaminated)."""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
from discrete_mode import discrete_mode  # noqa: E402


def lobe(Ma, lobe_name="lower"):
    """Return (upper, lower) branch arrays of the first-mode lobe for Ma."""
    ref = REPO / f"reference_data/digitized/ozgen_fig3_M{Ma}_neutral_v2.csv"
    rows = list(csv.DictReader(open(ref)))
    # first-mode lobe: mode=='first' if present, else the low-alpha lobe
    first = [r for r in rows if r.get("mode", "first") == "first"]
    use = first if first else rows
    up = sorted((float(r["Re"]), float(r["alpha"])) for r in use if r["lobe"] == "upper")
    lo = sorted((float(r["Re"]), float(r["alpha"])) for r in use if r["lobe"] == "lower")
    return np.array(up), np.array(lo)


for Ma in (3, 4, 6):
    up, lo = lobe(Ma)
    if up.size == 0 or lo.size == 0:
        print(f"M{Ma}: missing branch data"); continue
    re_lo = max(up[:, 0].min(), lo[:, 0].min())
    re_hi = min(up[:, 0].max(), lo[:, 0].max())
    print(f"=== M{Ma} first-mode lobe: Re {re_lo:.0f}-{re_hi:.0f} ===")
    n_ok = n_unst = n_cs = 0
    for Re in np.linspace(re_lo * 1.05, re_hi * 0.95, 7):
        a_up = np.interp(Re, up[:, 0], up[:, 1])
        a_lo = np.interp(Re, lo[:, 0], lo[:, 1])
        a_mid = 0.5 * (a_up + a_lo)
        m = discrete_mode(float(Ma), float(Re), float(a_mid), N=240)
        if m is None:
            print(f"  Re={Re:5.0f} a={a_mid:.4f}: NO discrete mode (CS)")
            n_cs += 1
        else:
            v = "UNSTABLE-agrees" if m["c_i"] > 0 else "stable"
            print(f"  Re={Re:5.0f} a={a_mid:.4f}: c={m['c_r']:.3f}{m['c_i']:+.4f}i fs={m['fs']:.3f} {v}")
            n_ok += 1
            n_unst += int(m["c_i"] > 0)
    print(f"  -> resolved {n_ok}/7, of which unstable {n_unst}; CS-blocked {n_cs}")
