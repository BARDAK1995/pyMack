"""R3-5 probe (a): OVERLAY production workload, measurement-only (NO gate).

Ma=2, N=128, single domain y_max = 6*delta_star/L*, the 720-node overlay grid
(Re 300-4500 log x24, alpha 0.02-0.24 lin x30) that mirrors
verification/mixed_mode/ozgen_fig3/_compute/ozgen_M2.json.

Runs temporal_sweep(backend='gpu') at N=128 and records: engine wall, VRAM
behaviour (from batch meta), any OOM / affine fallback, and verdict agreement
against the COMMITTED ozgen_M2_ci_grid.csv (N=128, TS-family single-domain
selection) where comparable, PLUS a GPU-vs-fresh-CPU subset check to isolate GPU
correctness from band-protocol differences.

This probe carries NO pass/fail gate; its deliverable is the honest table for the
user's M3 scope decision.  Trusted numbers require a fresh serial process with
--sysmem-policy prefer_no_sysmem_fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, _select_temporal_mode, temporal_sweep
from pymack.temporal_solver import solve_temporal_2d

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "docs" / "gpu" / "benchmarks" / "probe_overlay_n128.json"
CI_GRID = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_compute" / "ozgen_M2_ci_grid.csv"
OZGEN_M2 = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_compute" / "ozgen_M2.json"


def _committed_baseline_s():
    try:
        d = json.loads(OZGEN_M2.read_text(encoding="utf-8"))
        return float(d.get("total_wall_time_s"))
    except Exception:
        return None


def _families():
    # Bands matching ozgen_M2.json mode_classification: TS c_r<0.45, Mack 0.88-0.99,
    # ci_abs_max=0.05 (so the comparison against the committed grid is apples-to-apples).
    return (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.88, 0.99, ci_abs_max=0.05, label="Mack"),
    )


def _load_ci_grid():
    rows = []
    with open(CI_GRID) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmem-policy", required=True, choices=["unknown", "prefer_no_sysmem_fallback"])
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--n-alpha", type=int, default=30)
    ap.add_argument("--n-re", type=int, default=24)
    ap.add_argument("--alpha-min", type=float, default=0.02)
    ap.add_argument("--alpha-max", type=float, default=0.24)
    ap.add_argument("--re-min", type=float, default=300.0)
    ap.add_argument("--re-max", type=float, default=4500.0)
    ap.add_argument("--cpu-subset", type=int, default=40, help="nodes for GPU-vs-freshCPU check")
    args = ap.parse_args(argv)

    trusted = (args.sysmem_policy == "prefer_no_sysmem_fallback")
    Ma = 2.0
    profile = pymack.make_flatplate_profile(Ma)
    dstar = delta_star_over_lstar(profile)
    y_max = 6.0 * dstar
    Res = np.logspace(np.log10(args.re_min), np.log10(args.re_max), args.n_re)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)
    fams = _families()

    payload = {
        "schema_version": 1,
        "artifact": "probe_overlay_n128",
        "probe": "R3-5(a) overlay workload; measurement-only (no gate)",
        "sysmem_policy": args.sysmem_policy,
        "sysmem_policy_trusted": trusted,
        "numbers_provisional": not trusted,
        "workload": {
            "Ma": Ma, "N": int(args.N), "y_max_rule": "6*delta_star/L*",
            "y_max_abs": float(y_max),
            "n_alpha": int(args.n_alpha), "n_re": int(args.n_re),
            "n_points": int(args.n_alpha * args.n_re),
            "alpha_range": [float(alphas[0]), float(alphas[-1])],
            "Re_range": [float(Res[0]), float(Res[-1])],
            "families": ["TS(-inf,0.45)", "Mack(0.88,0.99)"],
        },
        "generated_at_unix": time.time(),
    }

    t0 = time.perf_counter()
    oom = None
    try:
        res = temporal_sweep(
            profile, alphas, Res, Ma=Ma, N=args.N, y_max=y_max,
            length_scale="L_star", operator="ozgen_2d", families=fams,
            backend="gpu", cpu_workers=1,
        )
        wall = time.perf_counter() - t0
    except Exception as exc:  # OOM / L-sizing / dtype promotion == FINDING, not failure
        payload["status"] = "resource_ceiling_hit"
        payload["finding"] = {
            "where": "temporal_sweep(backend='gpu') N=%d 720 nodes" % args.N,
            "exception": repr(exc),
            "wall_s_before_fail": time.perf_counter() - t0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": payload["status"], "finding": payload["finding"]}, indent=2))
        return

    meta = res.meta
    genum = meta.get("gpu_enumerator", {})
    batch = genum.get("cross_point_batching", {})
    # VRAM observed lives per-family in diagnostics; pull the batch vram_observed if present
    vram = None
    for d in genum.get("diagnostics", []):
        cpb = d.get("cross_point_batch", {})
        if isinstance(cpb, dict) and "vram_observed" in cpb:
            vram = cpb["vram_observed"]
            break
    payload["status"] = "measured_trusted" if trusted else "measured_provisional"
    payload["engine_wall_time_s"] = float(meta.get("engine_wall_time_s", wall))
    baseline = _committed_baseline_s()
    eng = float(meta.get("engine_wall_time_s", wall))
    payload["committed_cpu_baseline_s"] = baseline
    payload["speedup_headline_engine_vs_committed"] = (
        (baseline / eng) if (baseline and eng > 0) else None
    )
    payload["speedup_headline_fullwall_vs_committed"] = (
        (baseline / float(wall)) if (baseline and wall > 0) else None
    )
    payload["audit_wall_time_s"] = float(meta.get("audit_wall_time_s", 0.0))
    payload["full_wall_time_s"] = float(wall)
    payload["gpu_engine_status"] = meta.get("gpu_engine_status")
    payload["affine_reverified"] = meta.get("affine_reverified")
    payload["operator_source"] = meta.get("operator_source")
    payload["vram_observed"] = vram
    payload["batch_meta"] = {
        "batched_family_runs": batch.get("batched_family_runs"),
        "per_point_fallback_runs": batch.get("per_point_fallback_runs"),
        "batch_failures": batch.get("batch_failures"),
    }
    # Stage-timing decomposition (item 3): which stage dominates at dim 4*(N+1)=516?
    payload["stage_timing_s"] = genum.get("timing_s")
    payload["escalation"] = meta.get("escalation")
    # Full audit dict (failures + seed) so any in-sweep verdict drift is
    # characterizable per-node, not just counted.
    payload["audit"] = meta.get("audit")

    # --- comparison vs committed ci_grid (TS single-domain map, N=128) ---
    rows = _load_ci_grid()
    n_re = args.n_re
    n_cmp = 0
    worst = 0.0
    fails = []
    ts_idx = 0  # families[0] == TS
    for r in rows:
        fam = (r.get("family") or "").strip()
        cr = (r.get("c_r") or "").strip()
        ci = (r.get("c_i") or "").strip()
        if fam != "TS" or cr in ("", "nan"):
            continue
        # committed grid row order: alpha outer, Re inner (matches this grid)
        # locate node by nearest (alpha, Re)
        aL = float(r["alpha_L"]); ReL = float(r["Re_L"])
        ia = int(np.argmin(np.abs(alphas - aL)))
        jr = int(np.argmin(np.abs(Res - ReL)))
        got = res.families[ts_idx].c[ia, jr]
        if not res.families[ts_idx].converged[ia, jr]:
            fails.append({"ia": ia, "jr": jr, "reason": "gpu_empty_grid_hit"})
            continue
        ref = complex(float(cr), float(ci))
        err = abs(complex(got) - ref)
        worst = max(worst, float(err))
        n_cmp += 1
        if err > 1e-6 * max(1.0, abs(ref)):
            fails.append({"ia": ia, "jr": jr, "err": float(err),
                          "ref": [ref.real, ref.imag], "got": [float(got.real), float(got.imag)]})
    payload["ci_grid_comparison"] = {
        "protocol": ("GPU families[TS].c vs committed ozgen_M2_ci_grid.csv TS rows, "
                     "matched by nearest (alpha,Re); tol 1e-6-rel. Band-protocol matched "
                     "to ozgen_M2.json (TS c_r<0.45, ci_abs_max=0.05)."),
        "committed_N": 128,
        "n_ts_resolved_in_csv": sum(1 for r in rows if (r.get("family") or "").strip() == "TS"
                                    and (r.get("c_r") or "").strip() not in ("", "nan")),
        "n_compared": int(n_cmp),
        "worst_abs_error": float(worst),
        "n_fails": int(len(fails)),
        "failures": fails[:8],
    }

    # --- GPU vs fresh-CPU subset (isolates GPU correctness from band protocol) ---
    rng = np.random.default_rng(20260706)
    flat = rng.choice(args.n_alpha * n_re, size=min(args.cpu_subset, args.n_alpha * n_re), replace=False)
    sub_worst = 0.0
    sub_checks = 0
    sub_fails = []
    for f in flat:
        ia = int(f // n_re); jr = int(f % n_re)
        ev, _v, _y = solve_temporal_2d(profile, float(alphas[ia]), float(Res[jr]), Ma,
                                       N=args.N, y_max=float(y_max), length_scale="L_star")
        for k, band in enumerate(fams):
            ridx = _select_temporal_mode(np.asarray(ev, dtype=complex), band)
            got = res.families[k].c[ia, jr]
            conv = res.families[k].converged[ia, jr]
            sub_checks += 1
            if ridx is None:
                if conv:
                    sub_fails.append({"ia": ia, "jr": jr, "fam": k, "reason": "cpu_empty_gpu_hit"})
                continue
            if not conv:
                sub_fails.append({"ia": ia, "jr": jr, "fam": k, "reason": "gpu_empty_cpu_hit"})
                continue
            err = abs(complex(got) - complex(ev[ridx]))
            sub_worst = max(sub_worst, float(err))
            if err > 1e-9 * max(1.0, abs(ev[ridx])):
                sub_fails.append({"ia": ia, "jr": jr, "fam": k, "err": float(err)})
    payload["gpu_vs_fresh_cpu_subset"] = {
        "n_nodes": int(len(flat)),
        "n_point_family_checks": int(sub_checks),
        "worst_abs_error": float(sub_worst),
        "n_fails": int(len(sub_fails)),
        "failures": sub_fails[:8],
        "note": "same bands both sides; isolates GPU verdict correctness at N=128.",
    }

    # --- deterministic characterization of ALL GPU Mack-band (0.88-0.99) hits ---
    # The overlay is a TS story (committed grid has 0 Mack rows); any in-sweep
    # audit drift most likely lives in the near-empty Mack band.  Compare every
    # GPU Mack-converged cell against fresh CPU _select (deterministic, bounded).
    mack_idx = 1
    mack_cells = []
    conv = res.families[mack_idx].converged
    for ia in range(len(alphas)):
        for jr in range(n_re):
            if bool(conv[ia, jr]):
                mack_cells.append((ia, jr))
    mack_fails = []
    mack_worst = 0.0
    for (ia, jr) in mack_cells[:120]:
        ev, _v, _y = solve_temporal_2d(profile, float(alphas[ia]), float(Res[jr]), Ma,
                                       N=args.N, y_max=float(y_max), length_scale="L_star")
        ridx = _select_temporal_mode(np.asarray(ev, dtype=complex), fams[mack_idx])
        got = res.families[mack_idx].c[ia, jr]
        if ridx is None:
            mack_fails.append({"ia": ia, "jr": jr, "reason": "cpu_empty_gpu_hit",
                               "gpu": [float(got.real), float(got.imag)]})
            continue
        err = abs(complex(got) - complex(ev[ridx]))
        mack_worst = max(mack_worst, float(err))
        if err > 1e-9 * max(1.0, abs(ev[ridx])):
            mack_fails.append({"ia": ia, "jr": jr, "err": float(err),
                               "gpu": [float(got.real), float(got.imag)],
                               "cpu": [float(ev[ridx].real), float(ev[ridx].imag)]})
    payload["mack_band_full_check"] = {
        "n_gpu_mack_converged_cells": len(mack_cells),
        "n_checked": min(len(mack_cells), 120),
        "worst_abs_error": float(mack_worst),
        "n_fails": int(len(mack_fails)),
        "failures": mack_fails[:20],
        "note": ("Every GPU Mack-band(0.88-0.99) hit vs fresh CPU _select at N=128; "
                 "deterministic. Characterizes the near-empty Mack band where the "
                 "random in-sweep audit drift most plausibly sits."),
    }

    # --- deterministic y_max-fragility test of the known spurious corner ---
    # ozgen_M2.json known_artifacts: the high-Re/high-alpha corner carries a weak
    # (c_i~1e-3) spurious instability from the discretized slow-acoustic continuous
    # spectrum (c_r <= 1-1/Ma overlaps the TS band); it is y_max-SENSITIVE, unlike
    # the converged mid-lobe discrete modes.  A large verdict shift under a small
    # y_max perturbation confirms an ill-defined (non-discrete) verdict there --
    # i.e. any GPU/CPU disagreement at that cell is a continuous-spectrum artifact,
    # not an engine defect.
    corner_cells = [(len(alphas) - 1, n_re - 1), (len(alphas) - 1, n_re - 2),
                    (len(alphas) - 2, n_re - 1), (len(alphas) - 1, 0), (0, n_re - 1)]
    corner_rows = []
    for (ia, jr) in corner_cells:
        a = float(alphas[ia]); Re = float(Res[jr])
        row = {"ia": ia, "jr": jr, "alpha": a, "Re": Re}
        ev0, _, _ = solve_temporal_2d(profile, a, Re, Ma, N=args.N, y_max=float(y_max), length_scale="L_star")
        ev1, _, _ = solve_temporal_2d(profile, a, Re, Ma, N=args.N, y_max=float(y_max * 1.08), length_scale="L_star")
        for k, band in enumerate(fams):
            gv = res.families[k].c[ia, jr]; gc = bool(res.families[k].converged[ia, jr])
            i0 = _select_temporal_mode(np.asarray(ev0, dtype=complex), band)
            i1 = _select_temporal_mode(np.asarray(ev1, dtype=complex), band)
            c0 = None if i0 is None else complex(ev0[i0])
            c1 = None if i1 is None else complex(ev1[i1])
            shift = None if (c0 is None or c1 is None) else abs(c0 - c1)
            gpu_cpu = None if (not gc or c0 is None) else abs(complex(gv) - c0)
            row[band.label] = {
                "gpu": ([float(gv.real), float(gv.imag)] if gc else None),
                "cpu_ymax_base": (None if c0 is None else [c0.real, c0.imag]),
                "cpu_ymax_plus8pct": (None if c1 is None else [c1.real, c1.imag]),
                "cpu_ymax_fragility_shift": (None if shift is None else float(shift)),
                "gpu_vs_cpu_base": (None if gpu_cpu is None else float(gpu_cpu)),
            }
        corner_rows.append(row)
    payload["corner_ymax_fragility"] = {
        "cells": corner_rows,
        "ymax_perturbation": "+8%",
        "match_tol_reference": 4.0e-3,
        "note": ("ozgen_M2.json known_artifacts: high-Re/high-alpha TS spurious "
                 "continuous-spectrum instability is y_max-sensitive. A "
                 "cpu_ymax_fragility_shift >> match_tol (4e-3) confirms an ill-defined "
                 "(non-discrete) verdict -> any GPU/CPU disagreement there is a "
                 "continuous-spectrum artifact, not an engine defect."),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "engine_wall_time_s": payload["engine_wall_time_s"],
        "committed_cpu_baseline_s": payload["committed_cpu_baseline_s"],
        "speedup_engine_vs_committed": payload["speedup_headline_engine_vs_committed"],
        "gpu_engine_status": payload["gpu_engine_status"],
        "vram_peak_used_bytes": (vram or {}).get("peak_used_bytes"),
        "ci_grid_n_compared": payload["ci_grid_comparison"]["n_compared"],
        "ci_grid_worst": payload["ci_grid_comparison"]["worst_abs_error"],
        "ci_grid_fails": payload["ci_grid_comparison"]["n_fails"],
        "gpu_vs_cpu_worst": payload["gpu_vs_fresh_cpu_subset"]["worst_abs_error"],
        "gpu_vs_cpu_fails": payload["gpu_vs_fresh_cpu_subset"]["n_fails"],
        "audit_n_drift": (payload.get("audit") or {}).get("n_drift"),
        "mack_cells": payload["mack_band_full_check"]["n_gpu_mack_converged_cells"],
        "mack_fails": payload["mack_band_full_check"]["n_fails"],
        "mack_worst": payload["mack_band_full_check"]["worst_abs_error"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
