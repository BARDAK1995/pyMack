"""Independent adversarial recomputation of the ozgen_m2 headline metric.

Does NOT use verification/compare_ozgen_fig3.py. Reads only the two raw CSVs in
this folder and recomputes, with a few lines of numpy, the headline number the
engine recorded in verdict.json:

    "nearest-crossing median |Delta-alpha|/alpha = 15.8% over 20 in-range points"

Headline definition (as documented in verdict.json pymack_provenance):
  - reference = digitized neutral arch alpha(Re) from Ozgen & Kircali (2008) Fig 3.
  - pyMack neutral locus = per-branch (lower/upper) zero-c_i crossings vs Re.
  - For each digitized Re in the Re-overlap, interpolate BOTH pyMack branches onto
    that Re, pick the crossing NEAREST to the digitized alpha, rel err = |d|/alpha.
  - headline = median of those relative errors.

Run: PYMACK_NO_BANNER=1 python recompute_headline_independent.py
"""
import csv
from collections import defaultdict

import numpy as np

REF = "reference_ozgen_M2_neutral.csv"
PM = "pymack_ozgen_M2_neutral.csv"
RECORDED = 0.15792325773920107


def main():
    # reference digitized arch
    ref = np.genfromtxt(REF, delimiter=",", names=True)
    ref_Re = np.asarray(ref["x"], float)
    ref_a = np.asarray(ref["y"], float)
    o = np.argsort(ref_Re)
    ref_Re, ref_a = ref_Re[o], ref_a[o]

    # pyMack per-branch crossings
    lo_Re, lo_a, up_Re, up_a = [], [], [], []
    for r in csv.DictReader(open(PM)):
        if r["status"].strip() != "ok":
            continue
        Re, a = float(r["Re_L"]), float(r["alpha_neutral_pymack"])
        if r["branch"].strip() == "lower":
            lo_Re.append(Re); lo_a.append(a)
        else:
            up_Re.append(Re); up_a.append(a)
    lo_Re, lo_a, up_Re, up_a = map(np.array, (lo_Re, lo_a, up_Re, up_a))

    # Re overlap
    lo = max(ref_Re.min(), min(lo_Re.min(), up_Re.min()))
    hi = min(ref_Re.max(), max(lo_Re.max(), up_Re.max()))
    m = (ref_Re >= lo) & (ref_Re <= hi)
    rRe, ra = ref_Re[m], ref_a[m]

    # nearest-crossing rel err
    li = np.interp(rRe, lo_Re, lo_a)
    ui = np.interp(rRe, up_Re, up_a)
    cand = np.vstack([li, ui])
    near = cand[np.argmin(np.abs(cand - ra[None, :]), axis=0), np.arange(len(ra))]
    rel = np.abs(near - ra) / np.abs(ra)

    median_rel = float(np.median(rel))
    print(f"n in-range points          = {len(rel)}")
    print(f"recomputed median rel err  = {median_rel*100:.2f}%")
    print(f"recorded median rel err    = {RECORDED*100:.2f}%")
    print(f"relative diff vs recorded  = {abs(median_rel-RECORDED)/RECORDED*100:.2f}%")
    print(f"fraction within 15% band   = {np.mean(rel <= 0.15)*100:.1f}%")
    print(f"MAE                        = {np.mean(np.abs(near-ra)):.5f}")
    ok = abs(median_rel - RECORDED) / RECORDED <= 0.10
    print("MATCHES recorded within 10% relative:" , ok)


if __name__ == "__main__":
    main()
