"""ADVERSARIAL independent recompute of the ozgen_m4 headline metric.

Does NOT import or read compare_ozgen_fig3.py. Reads only the raw CSVs:
  reference_ozgen_M4_neutral.csv  -- digitized arch (x=Re_L, y=alpha)
  pymack_ozgen_M4_neutral.csv     -- pyMack neutral crossings (lower+upper) per Re

Headline metric (per provenance string): for each digitized (Re, alpha_ref)
point, take pyMack's neutral crossing alpha at the SAME Re and match to the
NEAREST crossing (min |alpha_pymack - alpha_ref|). Relative error = that
nearest |delta-alpha| / alpha_ref. Headline = median over in-range points.
"""
import csv
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent

# --- reference digitized arch ---
ref_re, ref_a = [], []
with open(here / "reference_ozgen_M4_neutral.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        ref_re.append(float(row["x"]))
        ref_a.append(float(row["y"]))
ref_re = np.array(ref_re); ref_a = np.array(ref_a)

# --- pyMack crossings keyed by Re ---
py = {}  # Re -> list of alpha crossings (any branch)
with open(here / "pymack_ozgen_M4_neutral.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["status"].strip() != "ok":
            continue
        re_l = float(row["Re_L"])
        a = float(row["alpha_neutral_pymack"])
        py.setdefault(re_l, []).append(a)

# --- nearest-crossing relative error at each digitized Re ---
rel_errs = []
n_no = 0
details = []
for re_l, a_ref in zip(ref_re, ref_a):
    # match by Re exactly (pyMack grid Re's match the digitized Re's here)
    if re_l in py:
        crossings = np.array(py[re_l])
    else:
        # nearest Re fallback
        keys = np.array(sorted(py.keys()))
        nk = keys[np.argmin(np.abs(keys - re_l))]
        crossings = np.array(py[nk])
    if crossings.size == 0:
        n_no += 1
        continue
    dabs = np.abs(crossings - a_ref)
    j = np.argmin(dabs)
    rel = dabs[j] / abs(a_ref)
    rel_errs.append(rel)
    details.append((re_l, a_ref, crossings[j], dabs[j], rel))

rel_errs = np.array(rel_errs)
median_rel = float(np.median(rel_errs))
within_band = float(np.mean(rel_errs <= 0.15))

print("=== Independent ozgen_m4 nearest-crossing recompute ===")
print(f"n in-range points        : {rel_errs.size}")
print(f"n_no_crossing            : {n_no}")
print(f"MEDIAN rel_err_alpha     : {median_rel:.6f}  ({median_rel*100:.2f}%)")
print(f"mean rel_err_alpha       : {rel_errs.mean():.6f}")
print(f"fraction within 15% band : {within_band:.3f}")
print()
print("Recorded in verdict.json : 0.16606158910220395 (16.61%), within-band=0.35, n=20")
print()
rec = 0.16606158910220395
rel_diff = abs(median_rel - rec) / rec
print(f"relative difference vs recorded: {rel_diff*100:.2f}%  -> {'MATCH (<=10%)' if rel_diff<=0.10 else 'MISMATCH (>10%)'}")
print()
print("per-point detail (Re, alpha_ref, nearest_pymack, |d|, rel):")
for d in details:
    print(f"  {d[0]:7.0f}  {d[1]:.4f}  {d[2]:.4f}  {d[3]:.4f}  {d[4]*100:6.2f}%")
