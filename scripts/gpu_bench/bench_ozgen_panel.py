"""Ozgen panel benchmark harness for slice 08."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep
from pymack.temporal_solver import solve_temporal_2d


REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "docs" / "gpu" / "benchmarks" / "ozgen_panel_bench.json"


def _families():
    return (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.45, 0.97, ci_abs_max=0.05, label="Mack"),
    )


def _select(evals, band):
    vals = np.asarray(evals, dtype=np.complex128)
    mask = (
        np.isfinite(vals.real)
        & np.isfinite(vals.imag)
        & (vals.real > band.cr_min)
        & (vals.real < band.cr_max)
        & (np.abs(vals.imag) < band.ci_abs_max)
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return int(idx[np.argmax(vals[idx].imag)])


def _comparison_check(
    result,
    profile,
    alphas,
    Res,
    *,
    Ma,
    N,
    y_max,
    check_eigs,
    sample_frac,
    exhaustive=False,
):
    rng = np.random.default_rng(80508)
    total = len(alphas) * len(Res)
    if exhaustive:
        picks = np.arange(total, dtype=int)
    else:
        n_sample = max(1, int(round(total * sample_frac)))
        picks = rng.choice(total, size=n_sample, replace=False)
    failures = []
    worst = 0.0
    fams = result.families
    family_drift_counts = {str(k): 0 for k in range(len(fams))}
    for flat in picks:
        i = int(flat // len(Res))
        j = int(flat % len(Res))
        evals, _vecs, _y = solve_temporal_2d(
            profile,
            float(alphas[i]),
            float(Res[j]),
            Ma,
            N=N,
            y_max=y_max,
            length_scale="L_star",
        )
        for k, fam in enumerate(fams):
            sel = _select(evals, fam.band)
            got = fam.c[i, j]
            if sel is None:
                if fam.converged[i, j]:
                    family_drift_counts[str(k)] += 1
                    failures.append({
                        "i": i,
                        "j": j,
                        "family": k,
                        "reason": "cpu_empty_gpu_nonempty",
                        "got": [float(np.real(got)), float(np.imag(got))],
                    })
                continue
            if not fam.converged[i, j]:
                family_drift_counts[str(k)] += 1
                ref = complex(evals[sel])
                failures.append({
                    "i": i,
                    "j": j,
                    "family": k,
                    "reason": "cpu_nonempty_gpu_empty",
                    "ref": [float(ref.real), float(ref.imag)],
                })
                continue
            ref = complex(evals[sel])
            err = abs(complex(got) - ref)
            worst = max(worst, float(err))
            if err > check_eigs * max(1.0, abs(ref)):
                family_drift_counts[str(k)] += 1
                failures.append({
                    "i": i,
                    "j": j,
                    "family": k,
                    "reason": "eigenvalue_mismatch",
                    "error": float(err),
                    "ref": [float(ref.real), float(ref.imag)],
                    "got": [float(np.real(got)), float(np.imag(got))],
                })
    return {
        "mode": "full" if exhaustive else "sample",
        "exhaustive": bool(exhaustive),
        "sample_fraction": 1.0 if exhaustive else float(sample_frac),
        "n_sample_points": int(len(picks)),
        "n_panel_points": int(total),
        "n_point_family_checks": int(len(picks) * len(fams)),
        "worst_abs_error": float(worst),
        "tolerance": float(check_eigs),
        "n_drift": int(len(failures)),
        "family_drift_counts": family_drift_counts,
        "passed": not failures,
        "failures": failures[:20],
        "failures_truncated": bool(len(failures) > 20),
    }


def _empty_truth_subsample():
    manifest = json.loads(
        (REPO / "verification" / "gpu_certification" / "hard_cells" / "truth_manifest.json")
        .read_text(encoding="utf-8")
    )
    empty = [c["id"] for c in manifest["cells"] if c["verdict"]["status"] == "no_discrete_mode"]
    return {
        "source": "verification/gpu_certification/hard_cells/truth_manifest.json",
        "n_empty_verdict_cells": len(empty),
        "sample_ids": empty[:5],
        "status": "recorded_from_corpus_not_recomputed_by_panel_bench",
    }


def _run_one_panel(args, *, profile, dstar, y_max, regime_label, comparison_mode):
    Ma = float(args.mach)
    n_re = int(args.n_re)
    n_alpha = int(args.n_alpha)
    if n_re * n_alpha != args.points:
        raise SystemExit(f"n_re*n_alpha must equal --points ({n_re*n_alpha} != {args.points})")
    Res = np.logspace(np.log10(args.re_min), np.log10(args.re_max), n_re)
    alphas = np.linspace(args.alpha_min, args.alpha_max, n_alpha)
    t0 = time.perf_counter()
    result = temporal_sweep(
        profile,
        alphas,
        Res,
        Ma=Ma,
        N=args.N,
        y_max=y_max,
        length_scale="L_star",
        operator="ozgen_2d",
        families=_families(),
        backend="gpu",
        cpu_workers=args.workers,
    )
    full_call_elapsed = time.perf_counter() - t0
    # F2: use engine-only wall (audit moved outside timed engine window; reported separately)
    engine_elapsed = float(result.meta.get("engine_wall_time_s") or result.meta.get("wall_time_s", full_call_elapsed))
    audit_elapsed = float(result.meta.get("audit_wall_time_s", 0.0))
    elapsed = engine_elapsed  # gate uses engine time only (audit is telemetry)
    checks = {}
    if comparison_mode in ("sample", "both"):
        checks["sample_check"] = _comparison_check(
            result,
            profile,
            alphas,
            Res,
            Ma=Ma,
            N=args.N,
            y_max=y_max,
            check_eigs=args.check_eigs,
            sample_frac=args.sample_frac,
            exhaustive=False,
        )
    if comparison_mode in ("full", "both"):
        checks["full_check"] = _comparison_check(
            result,
            profile,
            alphas,
            Res,
            Ma=Ma,
            N=args.N,
            y_max=y_max,
            check_eigs=args.check_eigs,
            sample_frac=args.sample_frac,
            exhaustive=True,
        )
    primary_check = checks.get("full_check") or checks["sample_check"]
    is_device_engine = not str(result.meta.get("gpu_engine_status", "")).startswith("cpu_qz_fallback")
    trusted_timing = args.sysmem_policy == "prefer_no_sysmem_fallback"
    audit_meta = result.meta.get("audit", {})
    audit_passed = bool(audit_meta.get("passed", True))
    gate_pass = bool(elapsed <= args.gate_seconds and primary_check["passed"] and is_device_engine and audit_passed)
    if gate_pass:
        status = "m2_gate_passed_trusted" if trusted_timing else "m2_gate_passed_provisional"
        gate_note = (
            "Device contour projection met the wall-clock and eigenvalue "
            "checks under trusted sysmem-fallback and serial-run provenance."
            if trusted_timing else
            "Device contour projection met the provisional wall-clock and "
            "sample-eigenvalue checks under the recorded GPU-sharing policy."
        )
    elif is_device_engine:
        status = "m2_gate_failed_trusted" if trusted_timing else "device_run_complete_but_m2_gate_not_met"
        gate_note = (
            "Device contour projection ran end-to-end under trusted sysmem-"
            "fallback and serial-run provenance, but M2 failed because the "
            "wall-clock gate or eigenvalue check was not met."
            if trusted_timing else
            "Device contour projection ran end-to-end, but the provisional "
            "M2 gate is not certified because elapsed time exceeded the gate "
            "or the CPU-QZ sample check failed."
        )
    else:
        status = "complete_but_m2_not_certified"
        gate_note = (
            "M2 is not certified when the temporal backend reports CPU-QZ "
            "fallback; the wall number is recorded but not a GPU engine pass."
        )
    if trusted_timing:
        m2_conclusion = "PASS" if gate_pass else "FAIL"
    else:
        m2_conclusion = "PROVISIONAL_PASS" if gate_pass else "NOT_CERTIFIED"
    payload = {
        "schema_version": 1,
        "artifact": "ozgen_panel_bench",
        "status": status,
        "numbers_provisional": not trusted_timing,
        "sysmem_policy": args.sysmem_policy,
        "sysmem_policy_trusted": trusted_timing,
        "gpu_possibly_shared": not trusted_timing,
        "run_provenance": {
            "fresh_python_process_required": trusted_timing,
            "serial_gpu_run": trusted_timing,
            "note": (
                "Trusted R3/M2 timing path: invoked in a fresh Python process "
                "with NVCP sysmem fallback set to Prefer No Sysmem Fallback; "
                "no other GPU work was intentionally launched by this lane."
                if trusted_timing else
                "Development/provisional timing path; GPU sharing or sysmem "
                "fallback policy can still invalidate pass/fail conclusions."
            ),
        },
        "gate_seconds": float(args.gate_seconds),
        "gate_elapsed_s": float(elapsed),
        "engine_wall_time_s": float(engine_elapsed),
        "audit_wall_time_s": float(audit_elapsed),
        "full_call_time_s": float(full_call_elapsed),
        "gate_passed": gate_pass,
        "m2_conclusion": m2_conclusion,
        "gate_note": gate_note,
        "panel": {
            "mach": Ma,
            "N": int(args.N),
            "n_points": int(args.points),
            "n_alpha": n_alpha,
            "n_re": n_re,
            "alpha_range": [float(alphas[0]), float(alphas[-1])],
            "Re_range": [float(Res[0]), float(Res[-1])],
            "y_max": float(y_max),
            "y_max_factor": float(y_max / dstar),
            "regime_label": regime_label,
        },
        "backend_meta": result.meta,
        "comparison_mode": comparison_mode,
        "comparison": primary_check,
        **checks,
        "empty_cells": _empty_truth_subsample(),
        "generated_at_unix": time.time(),
    }
    return payload


def run(args):
    Ma = float(args.mach)
    profile = pymack.make_flatplate_profile(Ma)
    dstar = delta_star_over_lstar(profile)
    comparison_mode = "full" if args.b1_both_ymax_regimes else args.comparison_mode
    if args.b1_both_ymax_regimes:
        panels = [
            _run_one_panel(
                args,
                profile=profile,
                dstar=dstar,
                y_max=6.0 * dstar,
                regime_label="6*dstar",
                comparison_mode=comparison_mode,
            ),
            _run_one_panel(
                args,
                profile=profile,
                dstar=dstar,
                y_max=12.0,
                regime_label="12",
                comparison_mode=comparison_mode,
            ),
        ]
        full_checks = [p.get("full_check", p["comparison"]) for p in panels]
        zero_drift = bool(all(c["passed"] and c["n_drift"] == 0 for c in full_checks))
        trusted_timing = args.sysmem_policy == "prefer_no_sysmem_fallback"
        payload = {
            "schema_version": 2,
            "artifact": "ozgen_panel_b1_full_compare",
            "status": (
                "b1_full_panel_zero_drift"
                if zero_drift else "b1_full_panel_drift_detected"
            ),
            "numbers_provisional": not trusted_timing,
            "sysmem_policy": args.sysmem_policy,
            "sysmem_policy_trusted": trusted_timing,
            "gpu_possibly_shared": not trusted_timing,
            "regime_set": "6*dstar_and_12",
            "check_eigs": float(args.check_eigs),
            "panel_points_per_regime": int(args.points),
            "total_point_family_checks": int(
                sum(c["n_point_family_checks"] for c in full_checks)
            ),
            "zero_drift": zero_drift,
            "panels": panels,
            "generated_at_unix": time.time(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "output": str(args.output),
            "zero_drift": payload["zero_drift"],
            "regimes": [
                {
                    "label": p["panel"]["regime_label"],
                    "gate_elapsed_s": p["gate_elapsed_s"],
                    "drift": p["comparison"]["n_drift"],
                    "worst_abs_error": p["comparison"]["worst_abs_error"],
                }
                for p in panels
            ],
        }, indent=2))
        return

    y_max = args.y_max_factor * dstar
    payload = _run_one_panel(
        args,
        profile=profile,
        dstar=dstar,
        y_max=y_max,
        regime_label=f"{args.y_max_factor:g}*dstar",
        comparison_mode=comparison_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "gate_elapsed_s": payload["gate_elapsed_s"],
        "gate_passed": payload["gate_passed"],
        "backend_status": payload["backend_meta"].get("gpu_engine_status"),
        "comparison_mode": payload["comparison_mode"],
        "comparison_passed": payload["comparison"]["passed"],
        "comparison_drift": payload["comparison"]["n_drift"],
    }, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-seconds", type=float, required=True)
    ap.add_argument("--check-eigs", type=float, required=True)
    ap.add_argument("--sysmem-policy", required=True, choices=["unknown", "prefer_no_sysmem_fallback"])
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--comparison-mode", choices=["sample", "full", "both"], default="sample")
    ap.add_argument("--b1-both-ymax-regimes", action="store_true")
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=31)
    ap.add_argument("--points", type=int, default=720)
    ap.add_argument("--n-alpha", type=int, default=30)
    ap.add_argument("--n-re", type=int, default=24)
    ap.add_argument("--alpha-min", type=float, default=0.02)
    ap.add_argument("--alpha-max", type=float, default=0.24)
    ap.add_argument("--re-min", type=float, default=300.0)
    ap.add_argument("--re-max", type=float, default=4500.0)
    ap.add_argument("--y-max-factor", type=float, default=6.0)
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
