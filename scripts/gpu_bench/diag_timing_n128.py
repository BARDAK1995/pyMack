"""R3-5(a) item 3: N=128 per-stage timing decomposition of the GPU enumerator.

The overlay probe artifact predates the stage-timing surfacing; this captures the
representative per-stage fractions (which stage dominates at dim 4*(N+1)=516) from
a small N=128 sweep, to inform whether the 2.1x headline is attackable via
projection-stage re-optimization.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "gpu" / "benchmarks" / "probe_overlay_n128_timing.json"
prof = pymack.make_flatplate_profile(2.0)
d = delta_star_over_lstar(prof)
F = (CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
     CBand(0.88, 0.99, ci_abs_max=0.05, label="Mack"))
# small representative sweep (fractions are node-count-insensitive)
alphas = np.linspace(0.10, 0.20, 6)
Res = np.logspace(np.log10(1500), np.log10(4500), 6)
res = temporal_sweep(prof, alphas, Res, Ma=2.0, N=128, y_max=6.0 * d,
                     length_scale="L_star", operator="ozgen_2d", families=F,
                     backend="gpu", cpu_workers=1)
genum = res.meta.get("gpu_enumerator", {})
out = {"N": 128, "dim_4Np1": 4 * 129, "n_points": len(alphas) * len(Res),
       "engine_wall_s": res.meta.get("engine_wall_time_s"),
       "timing_s": genum.get("timing_s"),
       "cross_point_batching": genum.get("cross_point_batching")}
OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, indent=2, default=str))
