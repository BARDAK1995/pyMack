"""R3-5(a) drift ROOT-CAUSE: characterize the single N=128 in-sweep audit drift.

The overlay probe's 5% stratified audit (36 random samples, corners always
included) flagged n_drift=1 at N=128.  ozgen_M2.json known_artifacts documents a
weak (c_i~1e-3) SPURIOUS instability at the high-Re/high-alpha corner from the
discretized slow-acoustic continuous spectrum (c_r <= 1-1/Ma overlaps the TS
band), which is y_max-SENSITIVE unlike genuine discrete modes.

This runs a SMALL GPU sweep over the high-alpha x high-Re corner block (the
per-point GPU verdict is box-independent: the affine pencil is evaluated exactly
per point, so a small-box sweep reproduces the full-sweep verdicts at these
nodes), compares each cell's TS/Mack verdict vs fresh CPU _select, and for any
disagreement tests y_max fragility (base vs +8%).  A verdict shift >> match_tol
under a tiny y_max change confirms an ill-defined continuous-spectrum verdict, so
any GPU/CPU disagreement there is a continuous-spectrum artifact, not an engine
defect.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, _select_temporal_mode, temporal_sweep
from pymack.temporal_solver import solve_temporal_2d

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "gpu" / "benchmarks" / "probe_overlay_n128_drift_rootcause.json"


def fams():
    return (CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
            CBand(0.88, 0.99, ci_abs_max=0.05, label="Mack"))


def main():
    Ma = 2.0
    N = 128
    prof = pymack.make_flatplate_profile(Ma)
    d = delta_star_over_lstar(prof)
    y_max = 6.0 * d
    # full overlay grid axes
    Res_full = np.logspace(np.log10(300), np.log10(4500), 24)
    al_full = np.linspace(0.02, 0.24, 30)
    # high-alpha x high-Re corner block (covers the 4 always-sampled corners' hot
    # one + the audit's high-alpha edge bias)
    ia_lo = 24  # alpha ~0.196..0.24
    jr_lo = 18  # Re ~2320..4500
    alphas = al_full[ia_lo:]
    Res = Res_full[jr_lo:]
    F = fams()

    t0 = time.perf_counter()
    res = temporal_sweep(prof, alphas, Res, Ma=Ma, N=N, y_max=y_max,
                         length_scale="L_star", operator="ozgen_2d", families=F,
                         backend="gpu", cpu_workers=1)
    sweep_s = time.perf_counter() - t0

    disagreements = []
    n_checks = 0
    worst = 0.0
    for ia, a in enumerate(alphas):
        for jr, Re in enumerate(Res):
            ev, _v, _y = solve_temporal_2d(prof, float(a), float(Re), Ma, N=N,
                                           y_max=float(y_max), length_scale="L_star")
            for k, band in enumerate(F):
                ridx = _select_temporal_mode(np.asarray(ev, dtype=complex), band)
                gv = res.families[k].c[ia, jr]
                gc = bool(res.families[k].converged[ia, jr])
                n_checks += 1
                cpu_none = ridx is None
                if cpu_none and not gc:
                    continue
                if cpu_none != (not gc):
                    # empty/hit mismatch -> the drift class; test fragility
                    ev2, _, _ = solve_temporal_2d(prof, float(a), float(Re), Ma, N=N,
                                                  y_max=float(y_max * 1.08), length_scale="L_star")
                    r2 = _select_temporal_mode(np.asarray(ev2, dtype=complex), band)
                    c0 = None if cpu_none else complex(ev[ridx])
                    c2 = None if r2 is None else complex(ev2[r2])
                    disagreements.append({
                        "alpha": float(a), "Re": float(Re), "family": band.label,
                        "abs_alpha_idx": ia_lo + ia, "abs_Re_idx": jr_lo + jr,
                        "gpu_converged": gc,
                        "gpu": ([float(gv.real), float(gv.imag)] if gc else None),
                        "cpu_select_ymax_base": (None if c0 is None else [c0.real, c0.imag]),
                        "cpu_select_ymax_plus8pct": (None if c2 is None else [c2.real, c2.imag]),
                        "cpu_verdict_flips_none_under_ymax": (cpu_none != (c2 is None)),
                        "reason": "empty_hit_mismatch",
                    })
                    continue
                if not cpu_none:
                    err = abs(complex(gv) - complex(ev[ridx]))
                    worst = max(worst, float(err))
                    if err > 1e-9 * max(1.0, abs(ev[ridx])):
                        # value disagreement -> test y_max fragility of the CPU verdict
                        ev2, _, _ = solve_temporal_2d(prof, float(a), float(Re), Ma, N=N,
                                                      y_max=float(y_max * 1.08), length_scale="L_star")
                        r2 = _select_temporal_mode(np.asarray(ev2, dtype=complex), band)
                        c0 = complex(ev[ridx]); c2 = None if r2 is None else complex(ev2[r2])
                        frag = None if c2 is None else abs(c0 - c2)
                        disagreements.append({
                            "alpha": float(a), "Re": float(Re), "family": band.label,
                            "abs_alpha_idx": ia_lo + ia, "abs_Re_idx": jr_lo + jr,
                            "gpu_vs_cpu_abs_err": float(err),
                            "gpu": [float(gv.real), float(gv.imag)],
                            "cpu_select_ymax_base": [c0.real, c0.imag],
                            "cpu_select_ymax_plus8pct": (None if c2 is None else [c2.real, c2.imag]),
                            "cpu_ymax_fragility_shift": (None if frag is None else float(frag)),
                            "reason": "value_disagreement",
                        })

    out = {
        "artifact": "probe_overlay_n128_drift_rootcause",
        "purpose": "root-cause the 1 in-sweep audit drift at N=128 overlay",
        "N": N, "Ma": Ma, "y_max_abs": float(y_max),
        "corner_block": {"alpha_range": [float(alphas[0]), float(alphas[-1])],
                         "Re_range": [float(Res[0]), float(Res[-1])],
                         "n_alpha": len(alphas), "n_re": len(Res)},
        "gpu_sweep_s": float(sweep_s),
        "n_point_family_checks": int(n_checks),
        "gpu_vs_cpu_worst_abs_error": float(worst),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "match_tol_reference": 4.0e-3,
        "known_artifact_quote": ("ozgen_M2.json known_artifacts: 'At M=2, high-Re/high-alpha "
                                 "corner: weak (c_i ~ 1e-3) spurious instability from the "
                                 "discretized slow-acoustic continuous spectrum (c_r <= 1 - 1/Ma "
                                 "overlaps the TS band); it is y_max-sensitive, unlike the "
                                 "converged mid-lobe discrete modes.'"),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"gpu_sweep_s": out["gpu_sweep_s"], "n_checks": n_checks,
                      "gpu_vs_cpu_worst": worst, "n_disagreements": len(disagreements),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
