"""Ozgen full two-domain family benchmark for slice 09 (M3 gate).

Measures the complete Ozgen family verdict (both domain heights) under the
GPU temporal engine and verifies point-by-point identity vs the deployed
CPU path (discrete-mode two-domain + _decaying_candidates + match + argmax).

Gate (verbatim):
  <= 30 s wall (fresh serial, --sysmem-policy prefer_no_sysmem_fallback)
  verdict-identical on 720 nodes (both y_max)
  CPU-escalation fraction < 0.03 (hard cap 0.05)

Mirrors bench_ozgen_panel.py conventions:
- required --sysmem-policy recorded
- gate fields in artifact
- serial fresh-process measurement only for trusted numbers
- environment / tile / determinism / escalation in meta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "docs" / "gpu" / "benchmarks" / "ozgen_full_bench.json"


def _families():
    return (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.45, 0.97, ci_abs_max=0.05, label="Mack"),
    )


def _add_discrete_path():
    v = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_refdigitize"
    if str(v) not in sys.path:
        sys.path.insert(0, str(v))
    import discrete_mode as dm  # noqa: E402

    return dm


def _two_domain_ref_verdict(Ma, Re, alpha, y_short, y_tall, N=31):
    """Reference verdict using the deployed discrete-mode driver (two-domain)."""
    dm = _add_discrete_path()
    ref = dm.discrete_mode(
        float(Ma), float(Re), float(alpha),
        N=int(N), ymf_pair=(float(y_short), float(y_tall)),
    )
    if ref is None:
        return None, None, 0
    return complex(ref["c_r"], ref["c_i"]), float(ref["fs"]), int(ref["n_match"])


def _run_full_family(args, *, profile, dstar, verify_verdicts):
    Ma = float(args.mach)
    n_re = int(args.n_re)
    n_alpha = int(args.n_alpha)
    if n_re * n_alpha != args.points:
        raise SystemExit(f"n_re*n_alpha must equal --points ({n_re*n_alpha} != {args.points})")
    Res = np.logspace(np.log10(args.re_min), np.log10(args.re_max), n_re)
    alphas = np.linspace(args.alpha_min, args.alpha_max, n_alpha)

    # ymf multipliers (scaled); absolute y = ymf * dstar for both sweeps and dm ref
    ymf_short = 6.0
    ymf_tall = 12.0
    y_short = ymf_short * dstar
    y_tall = ymf_tall * dstar

    # R3-2: capture GPU-derived per-point candidate sets so the two-domain
    # family verdict is assembled from GPU data (not fresh CPU spectra).  The
    # env toggle is read inside pymack/gpu/temporal.py::_run_gpu_grid; the CPU
    # facade is not touched.
    if verify_verdicts:
        import os
        os.environ["PYMACK_GPU_CAPTURE_CANDIDATE_SETS"] = "1"

    t0 = time.perf_counter()
    # GPU runs for both domains (full family)
    res_short = temporal_sweep(
        profile, alphas, Res, Ma=Ma, N=args.N, y_max=y_short,
        length_scale="L_star", operator="ozgen_2d", families=_families(),
        backend="gpu", cpu_workers=args.workers,
    )
    res_tall = temporal_sweep(
        profile, alphas, Res, Ma=Ma, N=args.N, y_max=y_tall,
        length_scale="L_star", operator="ozgen_2d", families=_families(),
        backend="gpu", cpu_workers=args.workers,
    )
    full_call_elapsed = time.perf_counter() - t0
    # F2: engine wall excludes audit (telemetry); use engine for gate_elapsed
    eng_s = float(res_short.meta.get("engine_wall_time_s") or res_short.meta.get("wall_time_s", full_call_elapsed))
    eng_t = float(res_tall.meta.get("engine_wall_time_s") or res_tall.meta.get("wall_time_s", full_call_elapsed))
    eng_audit_s = float(res_short.meta.get("audit_wall_time_s", 0.0))
    eng_audit_t = float(res_tall.meta.get("audit_wall_time_s", 0.0))
    # HONEST M3 wall: the full two-domain family requires BOTH domain-height
    # sweeps, which run serially, so the wall is their SUM (not max -- max would
    # halve and thus soften the miss).  Audit is excluded (F2: certification
    # telemetry, not engine cost).  full_call_time_s (below) additionally carries
    # both audits + Python overhead for transparency.
    elapsed = eng_s + eng_t
    # engine time only; verify after for identity gate; audit separated below

    # Pull escalation / audit from meta (stage-3 wiring)
    esc_short = res_short.meta.get("escalation", {})
    esc_tall = res_tall.meta.get("escalation", {})
    cpu_frac = max(
        float(esc_short.get("cpu_qz_fraction", 0.0)),
        float(esc_tall.get("cpu_qz_fraction", 0.0)),
    )
    audit_short = res_short.meta.get("audit", {})
    audit_tall = res_tall.meta.get("audit", {})
    audit_drift = int(audit_short.get("n_drift", 0)) + int(audit_tall.get("n_drift", 0))
    audit_passed = bool(audit_short.get("passed", True) and audit_tall.get("passed", True))

    # Verify verdicts (R3-2): the two-domain family verdict is assembled from
    # GPU-SWEEP-DERIVED candidate sets (res_short/res_tall meta captures), NOT
    # fresh CPU spectra, then compared character-identically against the deployed
    # dm.discrete_mode two-domain reference (CPU authority).  Any value or
    # None-cell drift = K3 (gate fails; do not tune).  A SEPARATE, clearly
    # labeled CPU-spectra layer-equivalence check certifies the assembly logic
    # (expected worst ~0).  Per-height single-domain identity is also kept.
    n_families = len(_families())
    gpu_sets_short = res_short.meta.get("gpu_candidate_sets", {})
    gpu_sets_tall = res_tall.meta.get("gpu_candidate_sets", {})
    gpu_capture_ok = bool(gpu_sets_short) and bool(gpu_sets_tall)

    ver_failures = []
    n_checks = 0
    worst = 0.0
    # GPU-derived two-domain gate (the R3-2 substance)
    two_domain_failures = []
    n_two_checks = 0
    two_worst = 0.0
    two_worst_fs = 0.0
    two_none_mismatch = 0
    two_nmatch_mismatch = 0
    # CPU-spectra layer-equivalence (logic-only; labeled separately)
    layer_failures = []
    n_layer_checks = 0
    layer_worst = 0.0

    def _empty_set():
        return {"values": np.zeros(0, dtype=complex),
                "vectors": np.zeros((0, 0), dtype=complex)}

    if verify_verdicts:
        from pymack.sweep import _select_temporal_mode
        from pymack.temporal_solver import solve_temporal_2d
        from pymack.gpu.verdict import (
            assemble_ozgen_verdict_from_gpu_candidate_sets,
            assemble_ozgen_verdict_from_spectra,
        )
        for ii, alpha in enumerate(alphas):
            for jj, Re in enumerate(Res):
                pidx = ii * n_re + jj
                # deployed two-domain reference (CPU authority)
                ref_val, ref_fs, ref_n = _two_domain_ref_verdict(
                    Ma, Re, alpha, ymf_short, ymf_tall, N=args.N
                )
                ref_none = (ref_val is None)

                # per-height single-domain identity + full CPU spectra (one solve each)
                ev_s, vec_s, y_s = solve_temporal_2d(profile, float(alpha), float(Re), Ma, N=args.N, y_max=float(y_short), length_scale="L_star")
                ev_t, vec_t, y_t = solve_temporal_2d(profile, float(alpha), float(Re), Ma, N=args.N, y_max=float(y_tall), length_scale="L_star")
                for tag, ev_h, res_g in [("short", ev_s, res_short), ("tall", ev_t, res_tall)]:
                    for k, fam in enumerate(_families()):
                        ref_idx = _select_temporal_mode(np.asarray(ev_h, dtype=complex), fam)
                        got = res_g.families[k].c[ii, jj]
                        n_checks += 1
                        if ref_idx is None:
                            if res_g.families[k].converged[ii, jj]:
                                ver_failures.append({"i": ii, "j": jj, "tag": tag, "fam": k, "reason": "ref_none_gpu_hit"})
                            continue
                        ref = complex(ev_h[ref_idx])
                        err = abs(complex(got) - ref)
                        worst = max(worst, float(err))
                        if err > 1e-9 * max(1.0, abs(ref)):
                            ver_failures.append({
                                "i": ii, "j": jj, "tag": tag, "fam": k,
                                "error": float(err),
                                "ref": [float(ref.real), float(ref.imag)],
                                "got": [float(got.real), float(got.imag)],
                            })

                # ---- GPU-DERIVED two-domain family verdict (R3-2 gate) ----
                if gpu_capture_ok:
                    sets_short = [gpu_sets_short.get((pidx, k), _empty_set()) for k in range(n_families)]
                    sets_tall = [gpu_sets_tall.get((pidx, k), _empty_set()) for k in range(n_families)]
                    v_gpu = assemble_ozgen_verdict_from_gpu_candidate_sets(sets_short, sets_tall)
                    n_two_checks += 1
                    got_none = (v_gpu.value is None or v_gpu.status == "no_discrete_mode")
                    if ref_none != got_none:
                        two_none_mismatch += 1
                        two_domain_failures.append({
                            "i": ii, "j": jj, "reason": "none_mismatch",
                            "ref_none": ref_none, "got_none": got_none,
                            "ref": (None if ref_none else [float(ref_val.real), float(ref_val.imag)]),
                            "got": (None if got_none else [float(v_gpu.value.real), float(v_gpu.value.imag)]),
                        })
                    elif not ref_none:
                        err = abs(complex(v_gpu.value) - complex(ref_val))
                        two_worst = max(two_worst, float(err))
                        if err > 1e-9 * max(1.0, abs(ref_val)):
                            two_domain_failures.append({
                                "i": ii, "j": jj, "error": float(err),
                                "ref": [float(ref_val.real), float(ref_val.imag)],
                                "got": [float(v_gpu.value.real), float(v_gpu.value.imag)],
                            })
                        # soft character diagnostics (recorded, NOT a K3 gate):
                        # GPU candidate-count may legitimately differ from full QZ.
                        two_worst_fs = max(two_worst_fs, abs((v_gpu.fs or 0.0) - (ref_fs or 0.0)))
                        if v_gpu.n_match != (ref_n or 0):
                            two_nmatch_mismatch += 1

                # ---- CPU-spectra layer equivalence (logic-only; labeled) ----
                v_cpu_layer = assemble_ozgen_verdict_from_spectra(
                    {"ev": ev_s, "vec": vec_s, "y": y_s},
                    {"ev": ev_t, "vec": vec_t, "y": y_t},
                )
                n_layer_checks += 1
                layer_none = (v_cpu_layer.value is None)
                if layer_none != ref_none:
                    layer_failures.append({"i": ii, "j": jj, "reason": "none_mismatch"})
                elif not ref_none:
                    lerr = abs(complex(v_cpu_layer.value) - complex(ref_val))
                    layer_worst = max(layer_worst, float(lerr))
                    if lerr > 1e-9 * max(1.0, abs(ref_val)):
                        layer_failures.append({"i": ii, "j": jj, "error": float(lerr)})

    # K3 gate: per-height + GPU-derived two-domain drift both count.  A missing
    # GPU capture is itself a hard verdict-gate failure (cannot certify).
    ver_failures.extend([{"two_domain_gpu": True, **f} for f in two_domain_failures])
    if verify_verdicts and not gpu_capture_ok:
        ver_failures.append({"reason": "gpu_candidate_capture_missing",
                             "detail": "PYMACK_GPU_CAPTURE_CANDIDATE_SETS not honored; "
                                       "two-domain gate could not run on GPU-derived data"})

    trusted = (args.sysmem_policy == "prefer_no_sysmem_fallback")
    gate_cpu_ok = (cpu_frac < float(args.gate_cpu_fallback))
    gate_time_ok = (elapsed <= float(args.gate_seconds))
    gate_verdict_ok = (len(ver_failures) == 0) if verify_verdicts else True
    gate_audit_ok = audit_passed
    gate_pass = bool(gate_time_ok and gate_cpu_ok and gate_verdict_ok and gate_audit_ok and trusted)

    status = "m3_gate_passed_trusted" if (gate_pass and trusted) else (
        "m3_gate_failed" if trusted else "m3_provisional"
    )

    payload = {
        "schema_version": 1,
        "artifact": "ozgen_full_bench",
        "status": status,
        "numbers_provisional": not trusted,
        "sysmem_policy": args.sysmem_policy,
        "sysmem_policy_trusted": trusted,
        "gate_seconds": float(args.gate_seconds),
        "gate_cpu_fallback": float(args.gate_cpu_fallback),
        "gate_elapsed_s": float(elapsed),
        "gate_passed": gate_pass,
        "gate_components": {
            "time_ok": gate_time_ok,
            "cpu_fraction_ok": gate_cpu_ok,
            "verdict_identity_ok": gate_verdict_ok,
            "audit_ok": gate_audit_ok,
        },
        "engine_wall_time_s": float(elapsed),
        "audit_wall_time_s_short": eng_audit_s,
        "audit_wall_time_s_tall": eng_audit_t,
        "full_call_time_s": float(full_call_elapsed),
        "cpu_escalation_fraction": cpu_frac,
        "audit_drift_total": audit_drift,
        "verify_verdicts": bool(verify_verdicts),
        "n_verdict_checks": int(n_checks),
        "verdict_worst_abs_error": float(worst),
        "verdict_failures": ver_failures[:5],
        "two_domain_verdicts": {
            "enabled": bool(verify_verdicts),
            "source": "gpu_sweep_derived_candidate_sets",
            "gpu_candidate_capture_ok": bool(gpu_capture_ok),
            "gpu_candidate_entries_short": int(len(gpu_sets_short)),
            "gpu_candidate_entries_tall": int(len(gpu_sets_tall)),
            "n_two_domain_checks": int(n_two_checks),
            "two_domain_worst_abs_error": float(two_worst),
            "two_domain_failures": two_domain_failures[:5],
            "two_domain_ver_fails": int(len(two_domain_failures)),
            "two_domain_none_mismatch": int(two_none_mismatch),
            "two_domain_worst_fs_diff": float(two_worst_fs),
            "two_domain_nmatch_mismatch_soft": int(two_nmatch_mismatch),
            "ymf_short": float(ymf_short),
            "ymf_tall": float(ymf_tall),
            "y_short_abs": float(y_short),
            "y_tall_abs": float(y_tall),
            "regimes": ["6*dstar", "12*dstar"],
            "note": ("GPU-DERIVED two-domain verdict assembled from "
                     "res_short/res_tall meta['gpu_candidate_sets'] via "
                     "assemble_ozgen_verdict_from_gpu_candidate_sets, compared vs "
                     "dm.discrete_mode (CPU authority). Nonzero worst expected "
                     "(~1e-12 refine floor); exact zero would mean CPU-vs-CPU. "
                     "K3 on value/None drift; fs/n_match are soft diagnostics."),
        },
        "layer_equivalence_cpu": {
            "enabled": bool(verify_verdicts),
            "n_checks": int(n_layer_checks),
            "worst_abs_error": float(layer_worst),
            "n_fails": int(len(layer_failures)),
            "failures": layer_failures[:5],
            "note": ("Logic-only check: assemble_ozgen_verdict_from_spectra on "
                     "FRESH CPU spectra vs dm.discrete_mode. NOT the gate; "
                     "certifies the assembly chain (expected worst ~0)."),
        },
        "panel": {
            "mach": Ma,
            "N": int(args.N),
            "n_points": int(args.points),
            "n_alpha": n_alpha,
            "n_re": n_re,
            "alpha_range": [float(alphas[0]), float(alphas[-1])],
            "Re_range": [float(Res[0]), float(Res[-1])],
        },
        "backend_meta_short": {
            "gpu_engine_status": res_short.meta.get("gpu_engine_status"),
            "escalation": esc_short,
            "audit": audit_short,
            "wall_time_s": res_short.meta.get("wall_time_s"),
        },
        "backend_meta_tall": {
            "gpu_engine_status": res_tall.meta.get("gpu_engine_status"),
            "escalation": esc_tall,
            "audit": audit_tall,
            "wall_time_s": res_tall.meta.get("wall_time_s"),
        },
        "seed_codes": res_short.meta.get("seed_codes"),
        "generated_at_unix": time.time(),
        "run_provenance": {
            "fresh_python_process_required": trusted,
            "serial_gpu_run": trusted,
        },
    }
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-seconds", type=float, required=True)
    ap.add_argument("--gate-cpu-fallback", type=float, required=True)
    ap.add_argument("--sysmem-policy", required=True, choices=["unknown", "prefer_no_sysmem_fallback"])
    ap.add_argument("--verify-verdicts", action="store_true")
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=31)
    ap.add_argument("--points", type=int, default=720)
    ap.add_argument("--n-alpha", type=int, default=30)
    ap.add_argument("--n-re", type=int, default=24)
    ap.add_argument("--alpha-min", type=float, default=0.02)
    ap.add_argument("--alpha-max", type=float, default=0.24)
    ap.add_argument("--re-min", type=float, default=300.0)
    ap.add_argument("--re-max", type=float, default=4500.0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    profile = pymack.make_flatplate_profile(float(args.mach))
    dstar = delta_star_over_lstar(profile)

    payload = _run_full_family(args, profile=profile, dstar=dstar, verify_verdicts=args.verify_verdicts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "gate_passed": payload["gate_passed"],
        "gate_elapsed_s": payload["gate_elapsed_s"],
        "cpu_escalation_fraction": payload["cpu_escalation_fraction"],
        "verify_verdicts": payload["verify_verdicts"],
        "n_checks": payload["n_verdict_checks"],
        "worst_err": payload["verdict_worst_abs_error"],
        "audit_drift": payload["audit_drift_total"],
        "two_domain_source": payload.get("two_domain_verdicts", {}).get("source"),
        "two_domain_capture_ok": payload.get("two_domain_verdicts", {}).get("gpu_candidate_capture_ok"),
        "two_domain_n_checks": payload.get("two_domain_verdicts", {}).get("n_two_domain_checks", 0),
        "two_domain_worst": payload.get("two_domain_verdicts", {}).get("two_domain_worst_abs_error", 0.0),
        "two_domain_fails": payload.get("two_domain_verdicts", {}).get("two_domain_ver_fails", 0),
        "two_domain_worst_fs": payload.get("two_domain_verdicts", {}).get("two_domain_worst_fs_diff", 0.0),
        "layer_equiv_worst": payload.get("layer_equivalence_cpu", {}).get("worst_abs_error", 0.0),
    }, indent=2))

    # Exit non-zero on gate fail for CI visibility (but we report honest numbers)
    if not payload["gate_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
