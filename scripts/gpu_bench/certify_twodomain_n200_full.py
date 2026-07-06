"""Certify GPU-derived two-domain Ozgen verdicts on the full Ma=2 N=200 grid.

This is the production 13x15 GRID[2] from
verification/mixed_mode/ozgen_fig3/_refdigitize/build_firstmode_grid.py:
Re=logspace(350, 5500, 13), alpha=linspace(0.010, 0.075, 15).

The GPU side intentionally mirrors scripts/gpu_bench/probe_twodomain_n200.py:
two temporal_sweep(backend="gpu") calls at N=200 with ymf_pair=(35,45),
candidate capture enabled by PYMACK_GPU_CAPTURE_CANDIDATE_SETS, and
two-domain assembly through
pymack.gpu.verdict.assemble_ozgen_verdict_from_gpu_candidate_sets.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep

REPO = Path(__file__).resolve().parents[2]
REF_CSV = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_refdigitize" / "firstmode_grid.csv"
OUT_DEFAULT = REPO / "docs" / "gpu" / "benchmarks" / "certify_twodomain_n200_full.json"

RE_TOL = 5.1e-2
ALPHA_TOL = 5.1e-6
FAIL_TOL = 4.0e-3


def _families():
    return (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.45, 0.97, ci_abs_max=0.05, label="Mack"),
    )


def _grid2():
    return (
        np.logspace(np.log10(350), np.log10(5500), 13),
        np.linspace(0.010, 0.075, 15),
    )


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _row_is_resolved(row: dict[str, str]) -> bool:
    flag = row.get("resolved", "").strip().lower()
    has_value = bool(row.get("c_r", "").strip()) and bool(row.get("c_i", "").strip())
    return flag in {"1", "true", "yes"} and has_value


def _load_reference_grid(ref_csv: Path, Res: np.ndarray, alphas: np.ndarray):
    by_node: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    off_grid: list[dict[str, Any]] = []
    ma2_rows_total = 0

    with ref_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            try:
                Ma = float(row["Ma"])
            except (KeyError, ValueError):
                continue
            if abs(Ma - 2.0) > 1.0e-12:
                continue
            ma2_rows_total += 1
            Re = float(row["Re"])
            alpha = float(row["alpha"])
            jr = int(np.argmin(np.abs(Res - Re)))
            ia = int(np.argmin(np.abs(alphas - alpha)))
            re_delta = float(abs(Res[jr] - Re))
            alpha_delta = float(abs(alphas[ia] - alpha))
            if re_delta > RE_TOL or alpha_delta > ALPHA_TOL:
                off_grid.append({
                    "row_number": int(row_number),
                    "Re": float(Re),
                    "alpha": float(alpha),
                    "nearest_grid": {
                        "j_re": int(jr),
                        "i_alpha": int(ia),
                        "Re": float(Res[jr]),
                        "alpha": float(alphas[ia]),
                        "re_delta": re_delta,
                        "alpha_delta": alpha_delta,
                    },
                })
                continue

            key = (ia, jr)
            resolved = _row_is_resolved(row)
            value = None
            if resolved:
                value = complex(float(row["c_r"]), float(row["c_i"]))
            rec = {
                "row_number": int(row_number),
                "i_alpha": int(ia),
                "j_re": int(jr),
                "Re": float(Re),
                "alpha": float(alpha),
                "resolved": bool(resolved),
                "value": value,
                "c_r": None if value is None else float(value.real),
                "c_i": None if value is None else float(value.imag),
                "fs": None if not row.get("fs", "").strip() else float(row["fs"]),
                "resolved_raw": row.get("resolved", ""),
                "re_delta": re_delta,
                "alpha_delta": alpha_delta,
            }
            if key in by_node:
                duplicates.append({"node": [int(ia), int(jr)], "prior": by_node[key], "new": rec})
            by_node[key] = rec

    missing_nodes = [
        {"i_alpha": int(ia), "j_re": int(jr), "alpha": float(alpha), "Re": float(Re)}
        for ia, alpha in enumerate(alphas)
        for jr, Re in enumerate(Res)
        if (ia, jr) not in by_node
    ]
    n_resolved = sum(1 for rec in by_node.values() if rec["resolved"])
    n_none = sum(1 for rec in by_node.values() if not rec["resolved"])
    return {
        "by_node": by_node,
        "ma2_rows_total": int(ma2_rows_total),
        "n_matched_reference": int(len(by_node)),
        "n_resolved": int(n_resolved),
        "n_none": int(n_none),
        "off_grid_rows": off_grid,
        "duplicates": duplicates,
        "missing_nodes": missing_nodes,
    }


def _vram_from_meta(meta):
    genum = meta.get("gpu_enumerator", {})
    for d in genum.get("diagnostics", []):
        cpb = d.get("cross_point_batch", {})
        if isinstance(cpb, dict) and "vram_observed" in cpb:
            return cpb["vram_observed"]
    return None


def _run_height(profile, alphas, Res, Ma, N, y_max):
    t0 = time.perf_counter()
    res = temporal_sweep(
        profile,
        alphas,
        Res,
        Ma=Ma,
        N=N,
        y_max=y_max,
        length_scale="L_star",
        operator="ozgen_2d",
        families=_families(),
        backend="gpu",
        cpu_workers=1,
    )
    return res, time.perf_counter() - t0


def _empty_set():
    return {"values": np.zeros(0, dtype=complex), "vectors": np.zeros((0, 0), dtype=complex)}


def _value_pair(z: complex | None):
    if z is None:
        return None
    return [float(z.real), float(z.imag)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmem-policy", required=True, choices=["unknown", "prefer_no_sysmem_fallback"])
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--ymf-short", type=float, default=35.0)
    ap.add_argument("--ymf-tall", type=float, default=45.0)
    args = ap.parse_args(argv)

    sys.dont_write_bytecode = True
    trusted = args.sysmem_policy == "prefer_no_sysmem_fallback"
    t_start = time.perf_counter()
    Ma = 2.0
    Res, alphas = _grid2()
    n_grid_nodes = int(len(Res) * len(alphas))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "certify_twodomain_n200_full",
        "status": "started",
        "sysmem_policy": args.sysmem_policy,
        "sysmem_policy_trusted": bool(trusted),
        "numbers_provisional": not trusted,
        "n_grid_nodes": n_grid_nodes,
        "n_matched_reference": 0,
        "n_resolved": 0,
        "n_none": 0,
        "worst_abs_error": 0.0,
        "n_fails": 0,
        "none_mismatches": 0,
        "per_failure_details": [],
        "reference_provenance": (
            "firstmode_grid.csv is a LOCAL gitignored regenerable artifact rebuilt by "
            "build_firstmode_grid.py (deterministic, resumable)."
        ),
        "workload": {
            "Ma": Ma,
            "N": int(args.N),
            "ymf_pair": [float(args.ymf_short), float(args.ymf_tall)],
            "n_re": int(len(Res)),
            "n_alpha": int(len(alphas)),
            "n_points": n_grid_nodes,
            "alpha_range": [float(alphas[0]), float(alphas[-1])],
            "Re_range": [float(Res[0]), float(Res[-1])],
            "node_match_tolerance": {"Re": RE_TOL, "alpha": ALPHA_TOL},
            "failure_abs_tolerance": FAIL_TOL,
        },
        "run_provenance": {
            "fresh_python_process_required": True,
            "serial_gpu_run": True,
            "gpu_candidate_capture_env": "PYMACK_GPU_CAPTURE_CANDIDATE_SETS=1",
        },
        "generated_at_unix": time.time(),
    }

    ref_t0 = time.perf_counter()
    ref = _load_reference_grid(REF_CSV, Res, alphas)
    ref_wall = time.perf_counter() - ref_t0
    payload.update({
        "reference_csv": str(REF_CSV),
        "ma2_reference_rows_total": ref["ma2_rows_total"],
        "n_matched_reference": ref["n_matched_reference"],
        "n_resolved": ref["n_resolved"],
        "n_none": ref["n_none"],
        "reference_match": {
            "matched_of_grid_nodes": [ref["n_matched_reference"], n_grid_nodes],
            "off_grid_rows": ref["off_grid_rows"][:20],
            "n_off_grid_rows": int(len(ref["off_grid_rows"])),
            "duplicates": ref["duplicates"],
            "missing_nodes": ref["missing_nodes"],
            "reference_load_wall_time_s": float(ref_wall),
        },
    })
    print(
        "REFERENCE_MATCH n_grid_nodes=%d matched=%d resolved=%d none=%d ma2_total=%d"
        % (
            n_grid_nodes,
            ref["n_matched_reference"],
            ref["n_resolved"],
            ref["n_none"],
            ref["ma2_rows_total"],
        ),
        flush=True,
    )
    _write_json(args.output, payload)

    if ref["n_matched_reference"] != n_grid_nodes or ref["duplicates"] or ref["missing_nodes"]:
        payload["status"] = "reference_grid_match_failed"
        payload["n_fails"] = int(len(ref["duplicates"]) + len(ref["missing_nodes"]))
        _write_json(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            "n_grid_nodes": n_grid_nodes,
            "n_matched_reference": ref["n_matched_reference"],
            "n_duplicates": len(ref["duplicates"]),
            "n_missing_nodes": len(ref["missing_nodes"]),
            "output": str(args.output),
        }, indent=2), flush=True)
        return 3

    profile = pymack.make_flatplate_profile(Ma)
    dstar = delta_star_over_lstar(profile)
    y_short = float(args.ymf_short * dstar)
    y_tall = float(args.ymf_tall * dstar)
    payload["workload"]["y_short_abs"] = y_short
    payload["workload"]["y_tall_abs"] = y_tall

    os.environ["PYMACK_GPU_CAPTURE_CANDIDATE_SETS"] = "1"
    gpu_t0 = time.perf_counter()
    try:
        res_short, wall_short = _run_height(profile, alphas, Res, Ma, args.N, y_short)
        print("GPU_HEIGHT_DONE tag=short wall_s=%.6f" % wall_short, flush=True)
        res_tall, wall_tall = _run_height(profile, alphas, Res, Ma, args.N, y_tall)
        print("GPU_HEIGHT_DONE tag=tall wall_s=%.6f" % wall_tall, flush=True)
    except Exception as exc:
        payload["status"] = "gpu_run_failed"
        payload["failure"] = {
            "where": "temporal_sweep(backend='gpu') N=%d two heights, 195 nodes" % args.N,
            "exception": repr(exc),
            "wall_s_before_fail": float(time.perf_counter() - gpu_t0),
        }
        payload["wall_times"] = {
            "reference_load_s": float(ref_wall),
            "full_process_s": float(time.perf_counter() - t_start),
        }
        _write_json(args.output, payload)
        print(json.dumps({"status": payload["status"], "failure": payload["failure"], "output": str(args.output)}, indent=2), flush=True)
        return 4

    from pymack.gpu.verdict import assemble_ozgen_verdict_from_gpu_candidate_sets

    sets_short = res_short.meta.get("gpu_candidate_sets", {})
    sets_tall = res_tall.meta.get("gpu_candidate_sets", {})
    capture_ok = bool(sets_short) and bool(sets_tall)
    eng_short = float(res_short.meta.get("engine_wall_time_s", wall_short))
    eng_tall = float(res_tall.meta.get("engine_wall_time_s", wall_tall))
    payload.update({
        "status": "gpu_sweeps_complete",
        "wall_times": {
            "reference_load_s": float(ref_wall),
            "gpu_short_call_s": float(wall_short),
            "gpu_tall_call_s": float(wall_tall),
            "engine_wall_time_s_short": eng_short,
            "engine_wall_time_s_tall": eng_tall,
            "engine_wall_time_s_total": float(eng_short + eng_tall),
            "gpu_sweeps_call_s_total": float(wall_short + wall_tall),
            "gpu_sweeps_enclosed_s": float(time.perf_counter() - gpu_t0),
        },
        "gpu_engine_status_short": res_short.meta.get("gpu_engine_status"),
        "gpu_engine_status_tall": res_tall.meta.get("gpu_engine_status"),
        "affine_reverified_short": res_short.meta.get("affine_reverified"),
        "affine_reverified_tall": res_tall.meta.get("affine_reverified"),
        "vram_observed_short": _vram_from_meta(res_short.meta),
        "vram_observed_tall": _vram_from_meta(res_tall.meta),
        "vram": {
            "short": _vram_from_meta(res_short.meta),
            "tall": _vram_from_meta(res_tall.meta),
        },
        "gpu_candidate_capture_ok": bool(capture_ok),
        "gpu_candidate_entries_short": int(len(sets_short)),
        "gpu_candidate_entries_tall": int(len(sets_tall)),
        "escalation_short": res_short.meta.get("escalation"),
        "escalation_tall": res_tall.meta.get("escalation"),
    })
    _write_json(args.output, payload)

    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    failures: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    none_mismatches = 0
    worst = 0.0
    n_compared = 0
    n_families = len(_families())
    missing_capture_entries = {"short": 0, "tall": 0}
    cmp_t0 = time.perf_counter()
    if not capture_ok:
        failures.append({
            "reason": "gpu_candidate_capture_missing",
            "detail": "PYMACK_GPU_CAPTURE_CANDIDATE_SETS did not produce both candidate-set maps.",
        })
    else:
        for ia, alpha in enumerate(alphas):
            for jr, Re in enumerate(Res):
                rec = ref["by_node"][(ia, jr)]
                pidx = ia * len(Res) + jr
                short_sets = []
                tall_sets = []
                for k in range(n_families):
                    key = (pidx, k)
                    if key not in sets_short:
                        missing_capture_entries["short"] += 1
                    if key not in sets_tall:
                        missing_capture_entries["tall"] += 1
                    short_sets.append(sets_short.get(key, _empty_set()))
                    tall_sets.append(sets_tall.get(key, _empty_set()))
                verdict = assemble_ozgen_verdict_from_gpu_candidate_sets(short_sets, tall_sets)
                got = None if verdict.value is None else complex(verdict.value)
                ref_value = rec["value"]
                ref_none = ref_value is None
                got_none = got is None
                n_compared += 1
                row = {
                    "i_alpha": int(ia),
                    "j_re": int(jr),
                    "alpha": float(alpha),
                    "Re": float(Re),
                    "reference_row_number": rec["row_number"],
                    "reference_resolved": bool(rec["resolved"]),
                    "reference_value": _value_pair(ref_value),
                    "gpu_status": verdict.status,
                    "gpu_value": _value_pair(got),
                    "gpu_fs": None if verdict.fs is None else float(verdict.fs),
                    "gpu_n_match": int(verdict.n_match),
                    "gpu_seed": int(verdict.seed),
                    "gpu_rung": verdict.rung,
                    "abs_error": None,
                }
                failure = None
                if ref_none != got_none:
                    none_mismatches += 1
                    failure = {
                        **row,
                        "reason": "none_mismatch",
                        "reference_none": bool(ref_none),
                        "gpu_none": bool(got_none),
                    }
                elif not ref_none:
                    err = float(abs(got - ref_value))
                    worst = max(worst, err)
                    row["abs_error"] = err
                    if err > FAIL_TOL:
                        failure = {**row, "reason": "abs_error_gt_4e-3"}
                if failure is not None:
                    failures.append(failure)
                comparison_rows.append(row)
                if (n_compared % 25) == 0 or n_compared == n_grid_nodes:
                    print(
                        "COMPARE_PROGRESS compared=%d/%d worst_abs_error=%.9g n_fails=%d none_mismatches=%d"
                        % (n_compared, n_grid_nodes, worst, len(failures), none_mismatches),
                        flush=True,
                    )

    comparison_wall = time.perf_counter() - cmp_t0
    zero_drift = len(failures) == 0 and n_compared == ref["n_matched_reference"]
    payload.update({
        "status": "certified_zero_drift" if zero_drift else "certification_failed",
        "n_nodes_compared": int(n_compared),
        "all_matched_nodes_compared": bool(n_compared == ref["n_matched_reference"]),
        "worst_abs_error": float(worst),
        "n_fails": int(len(failures)),
        "none_mismatches": int(none_mismatches),
        "per_failure_details": failures,
        "missing_capture_entries": missing_capture_entries,
        "zero_drift_held": bool(zero_drift),
    })
    payload["wall_times"]["comparison_s"] = float(comparison_wall)
    payload["wall_times"]["full_process_s"] = float(time.perf_counter() - t_start)
    payload["comparison"] = {
        "source": "gpu_sweep_derived_candidate_sets",
        "assembly": "pymack.gpu.verdict.assemble_ozgen_verdict_from_gpu_candidate_sets",
        "reference": "local firstmode_grid.csv matched to build_firstmode_grid.py GRID[2]",
        "n_rows_recorded": int(len(comparison_rows)),
        "failure_abs_tolerance": FAIL_TOL,
    }
    _write_json(args.output, payload)

    summary = {
        "status": payload["status"],
        "output": str(args.output),
        "n_grid_nodes": payload["n_grid_nodes"],
        "n_matched_reference": payload["n_matched_reference"],
        "n_resolved": payload["n_resolved"],
        "n_none": payload["n_none"],
        "n_nodes_compared": payload["n_nodes_compared"],
        "worst_abs_error": payload["worst_abs_error"],
        "n_fails": payload["n_fails"],
        "none_mismatches": payload["none_mismatches"],
        "zero_drift_held": payload["zero_drift_held"],
        "engine_wall_time_s_total": payload["wall_times"]["engine_wall_time_s_total"],
        "full_process_s": payload["wall_times"]["full_process_s"],
        "gpu_candidate_capture_ok": payload["gpu_candidate_capture_ok"],
        "vram_short_peak_used_bytes": (payload["vram_observed_short"] or {}).get("peak_used_bytes"),
        "vram_tall_peak_used_bytes": (payload["vram_observed_tall"] or {}).get("peak_used_bytes"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        print("FAILURES_BEGIN", flush=True)
        for failure in failures:
            print(json.dumps(failure, default=_jsonable, sort_keys=True), flush=True)
        print("FAILURES_END", flush=True)
    return 0 if zero_drift else 2


if __name__ == "__main__":
    raise SystemExit(main())
