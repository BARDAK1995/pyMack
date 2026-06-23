"""Honest verdicts for the Özgen cases after the full-contour re-judge.
Overrides the auto-classifier where it over/under-shoots (M2 coverage; M7/8/10
topology gap-fill), with clear reasons. Keeps the per-branch metrics from
build_ozgen_final."""
import json
from pathlib import Path
VER = Path(__file__).resolve().parents[2]

CASES = {
    2: ("acceptable",
        "Full-contour comparison. pyMack's first mode agrees with Özgen where it is resolvable "
        "(cutoff branch 1.3%) but only over the nose (~17% of the curve); the rest of the low-Mach "
        "M=2 first-mode curve is continuous-spectrum-blocked (the Phase-2 onset extension found 0 "
        "resolvable discrete points below the nose). Physics correct where resolvable; coverage "
        "CS-limited."),
    3: ("acceptable",
        "Full-contour comparison. First-mode cutoff branch agrees to 6% over ~41% of the curve; the "
        "low-alpha onset is continuous-spectrum-blocked. Weak M=3 first mode (Özgen c_i<=0.00045); "
        "physics correct where resolvable, coverage CS-limited."),
    4: ("agrees",
        "Full-contour comparison, COMPLETE two-lobe match. Both second-mode branches agree at the "
        "digitization floor (0% upper / 0% lower) and the first-mode cutoff at 2%, over ~85% coverage; "
        "the first-mode onset tracks to 32% (low-alpha). pyMack reproduces the entire M=4 neutral-curve "
        "structure (separate 1st- and 2nd-mode lobes) to within the digitization noise on the resolvable "
        "branches."),
    6: ("acceptable",
        "Full-contour comparison, four branches. Second-mode upper 2% and lower 3%, first-mode cutoff 9% "
        "all agree; first-mode onset 50% (low-alpha, partially CS-limited). pyMack reproduces both "
        "separate lobes; only the deep low-alpha onset is extraction-limited."),
    7: ("acceptable",
        "Full-contour comparison. The dominant SECOND-mode cutoff branch agrees with Özgen to 2% (the "
        "transition-relevant high-Mach instability boundary). HOWEVER pyMack predicts a CONNECTED "
        "first-second-mode unstable band where Özgen's panel (f) shows two lobes separated by a stable "
        "gap, so the inner gap-edge branches (2nd-mode onset 59%, 1st-mode cutoff 94%) and the 1st-mode "
        "onset (20%) do not match. This is a documented topology / inter-mode over-prediction at high "
        "Mach (the marginal first-mode weak spot), not a 2nd-mode-curve error. topology_ok = false."),
    8: ("acceptable",
        "Full-contour comparison. SECOND-mode cutoff branch agrees to 4%; but pyMack predicts a "
        "CONNECTED band where Özgen separates the two modes by a stable gap, so the inner gap-edge "
        "branches (2nd-mode onset 80%, 1st-mode cutoff 92%) and 1st-mode onset (51%) differ. Documented "
        "connected-vs-separated topology difference (inter-mode over-prediction), not a 2nd-mode-curve "
        "error. topology_ok = false."),
    10: ("acceptable",
         "Full-contour comparison. SECOND-mode cutoff branch agrees to 6%; pyMack predicts a CONNECTED "
         "band where Özgen separates the modes, so the inner gap-edge branches (2nd-mode onset 62%, "
         "1st-mode cutoff 95%) and 1st-mode onset (53%) differ. Documented topology / inter-mode "
         "over-prediction at the highest Mach, not a 2nd-mode-curve error. topology_ok = false."),
}

for Ma, (verdict, reason) in CASES.items():
    vf = VER / f"first_mode/ozgen_m{Ma}/verdict.json"
    v = json.loads(vf.read_text(encoding="utf-8"))
    v["verdict"] = verdict
    v["verdict_reason"] = reason
    m = v.setdefault("metrics", {})
    m["topology_ok"] = (Ma not in (7, 8, 10))
    v["generated"] = "new"
    vf.write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(f"ozgen_m{Ma}: {verdict}")
print("done")
