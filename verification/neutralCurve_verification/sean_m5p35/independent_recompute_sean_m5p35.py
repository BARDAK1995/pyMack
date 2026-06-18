"""
ADVERSARIAL independent recomputation of the headline metric for case 'sean_m5p35'.

Does NOT use verification/compare_sean_m5p35.py. Computes the upper-branch MAE
(200-600 kHz) directly from the raw CSVs with numpy interp + mean(abs(diff)),
and cross-checks against the recorded verdict.json value.

Upper branch  = reference x_right_mm  vs  pyMack upper_neutral_x_mm
Lower branch  = reference x_left_mm   vs  pyMack lower_neutral_x_mm
"""
import csv
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "LST_neutral_curve_M5p35.csv"
PM = HERE / "pymack_neutral_envelope_dimensional.csv"
VERDICT = HERE / "verdict.json"


def load(path, cols):
    out = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for c in cols:
                out[c].append(float(row[c]))
    return {c: np.array(v) for c, v in out.items()}


ref = load(REF, ["frequency_khz", "x_left_mm", "x_right_mm"])
pm = load(PM, ["frequency_khz", "lower_neutral_x_mm", "upper_neutral_x_mm"])

rf = ref["frequency_khz"]
ru = ref["x_right_mm"]   # upper branch reference
rl = ref["x_left_mm"]    # lower branch reference

pf = pm["frequency_khz"]
pu = pm["upper_neutral_x_mm"]
pl = pm["lower_neutral_x_mm"]

# Drop any NaN rows in pyMack (use as interpolation source)
def clean(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def mae_band(ref_f, ref_y, test_f, test_y, flo, fhi):
    """Interpolate pyMack (test) onto reference frequencies in [flo,fhi]; MAE."""
    tf, ty = clean(test_f, test_y)
    order = np.argsort(tf)
    tf, ty = tf[order], ty[order]
    mask = (ref_f >= flo) & (ref_f <= fhi)
    rf_b = ref_f[mask]
    ry_b = ref_y[mask]
    pred = np.interp(rf_b, tf, ty)  # outside-range -> clamps to endpoints
    abs_err = np.abs(pred - ry_b)
    mae = float(np.mean(abs_err))
    span = float(np.max(ry_b) - np.min(ry_b))
    return mae, span, int(mask.sum()), abs_err.max()


# ---- Headline: upper branch 200-600 kHz ----
up_mae, up_span, up_n, up_max = mae_band(rf, ru, pf, pu, 200.0, 600.0)
# Lower branch 330-600 kHz
lo_mae_g, lo_span_g, lo_n_g, _ = mae_band(rf, rl, pf, pl, 330.0, 600.0)
# Lower branch full 200-600 kHz
lo_mae_f, lo_span_f, lo_n_f, _ = mae_band(rf, rl, pf, pl, 200.0, 600.0)

print("=== INDEPENDENT RECOMPUTE (numpy interp + MAE) ===")
print(f"UPPER 200-600 kHz : MAE={up_mae:.4f} mm  span={up_span:.3f} mm  "
      f"n={up_n}  ratio={up_mae/up_span:.4f}  (max abs err {up_max:.3f})")
print(f"LOWER 330-600 kHz : MAE={lo_mae_g:.4f} mm  span={lo_span_g:.3f} mm  "
      f"n={lo_n_g}  ratio={lo_mae_g/lo_span_g:.4f}")
print(f"LOWER 200-600 kHz : MAE={lo_mae_f:.4f} mm  span={lo_span_f:.3f} mm  "
      f"n={lo_n_f}  ratio={lo_mae_f/lo_span_f:.4f}")

vd = json.loads(VERDICT.read_text())
rec = vd["metrics"]["upper_branch_MAE_mm_200_600kHz"]
print()
print(f"RECORDED upper-branch MAE (verdict.json) = {rec:.4f} mm")
print(f"INDEPENDENT upper-branch MAE             = {up_mae:.4f} mm")
rel = abs(up_mae - rec) / rec
print(f"Relative difference = {rel*100:.2f}%  -> "
      f"{'MATCH (<=10%)' if rel <= 0.10 else 'MISMATCH (>10%)'}")

# Also cross-check the other two recorded numbers
print()
print("Cross-check secondary metrics:")
print(f"  lower 330-600: recorded {vd['metrics']['lower_branch_MAE_mm_330_600kHz']:.4f} "
      f"vs indep {lo_mae_g:.4f}")
print(f"  lower full   : recorded {vd['metrics']['lower_branch_MAE_mm_full_200kHz']:.4f} "
      f"vs indep {lo_mae_f:.4f}")
print(f"  upper n pts  : recorded {vd['metrics']['upper_branch_n_points']} vs indep {up_n}")
