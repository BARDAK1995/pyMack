"""Measure a point-parallel CPU anchor for Mack Fig. 10.4 M10.

This is a measurement harness only. It imports the deployed
``verification/compute_mack_fig10_4.py`` first-mode point function and schedules
the 9 station x 136 coarse points, then the exact 9-point per-station refine
windows, across a process pool.
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import json
import multiprocessing
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from scripts.gpu_bench.measure_fig10_4_m10_cpu_anchor import (
    BLAS_THREAD_VARS,
    _gpu_info,
    _identity_check,
    _load_anchor_summary,
    _load_reference,
    _set_blas_threads,
    _workload_config,
)


REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = (
    REPO / "docs" / "gpu" / "benchmarks" / "cpu_fig10_4_m10_pointparallel_61w.json"
)

_ENGINE = None
_WORKER_BLAS_THREADS = None
_WORKER_STATE = None


def _ensure_repo_on_path() -> None:
    repo = str(REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _import_driver():
    _ensure_repo_on_path()
    from verification import compute_mack_fig10_4 as engine

    return engine


def _threadpool_state() -> dict:
    try:
        from threadpoolctl import threadpool_info
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"available": False, "error": repr(exc)}
    libs = []
    for item in threadpool_info():
        libs.append(
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "num_threads": item.get("num_threads"),
                "prefix": item.get("prefix"),
                "filepath": item.get("filepath"),
            }
        )
    return {"available": True, "libraries": libs}


def _capture_worker_state() -> dict:
    threadpools = _threadpool_state()
    blas_threads_observed = []
    if threadpools.get("available"):
        blas_threads_observed = sorted(
            {
                int(item["num_threads"])
                for item in threadpools.get("libraries", [])
                if item.get("user_api") == "blas" and item.get("num_threads") is not None
            }
        )
    return {
        "pid": os.getpid(),
        "blas_threads_requested": _WORKER_BLAS_THREADS,
        "blas_env_after_driver_import": {
            key: os.environ.get(key) for key in BLAS_THREAD_VARS
        },
        "threadpools": threadpools,
        "blas_threads_observed": blas_threads_observed,
    }


def _worker_init(blas_threads: int | None) -> None:
    global _WORKER_BLAS_THREADS
    _WORKER_BLAS_THREADS = blas_threads
    _ensure_repo_on_path()
    _set_blas_threads(blas_threads)
    os.environ.setdefault("PYMACK_NO_BANNER", "1")


def _get_engine():
    global _ENGINE, _WORKER_STATE
    if _ENGINE is None:
        _ENGINE = _import_driver()
        _WORKER_STATE = _capture_worker_state()
    return _ENGINE


def _solve_point(task: dict) -> dict:
    engine = _get_engine()
    try:
        profile = engine._get_profile(float(task["mach"]))
        beta = float(task["alpha"]) * float(
            engine.np.tan(engine.np.radians(float(task["psi"])))
        )
        oi, c = engine.first_mode_growth(
            profile,
            float(task["alpha"]),
            beta,
            float(task["R"]),
            float(task["mach"]),
            N=int(task["N"]),
            y_max=float(task["y_max"]),
        )
        result = {
            "status": "ok",
            "phase": task["phase"],
            "task_id": int(task["task_id"]),
            "station_index": int(task["station_index"]),
            "R": float(task["R"]),
            "alpha": float(task["alpha"]),
            "psi": float(task["psi"]),
            "beta": beta,
            "point_order": int(task["point_order"]),
            "omega_i": None if oi is None else float(oi),
            "c_r": None if c is None else float(c.real),
            "c_i": None if c is None else float(c.imag),
            "worker_pid": os.getpid(),
        }
    except Exception as exc:  # pragma: no cover - recorded in artifact
        result = {
            "status": "error",
            "phase": task["phase"],
            "task_id": int(task["task_id"]),
            "station_index": int(task["station_index"]),
            "R": float(task["R"]),
            "alpha": float(task["alpha"]),
            "psi": float(task["psi"]),
            "point_order": int(task["point_order"]),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
        }
    result["worker_state"] = _WORKER_STATE
    return result


def _run_tasks(
    executor: _cf.ProcessPoolExecutor,
    tasks: list[dict],
    *,
    phase: str,
    progress_every: int = 50,
) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    results = []
    worker_states = {}
    futures = [executor.submit(_solve_point, task) for task in tasks]
    n_done = 0
    for future in _cf.as_completed(futures):
        res = future.result()
        state = res.pop("worker_state", None)
        if state is not None:
            worker_states[str(state["pid"])] = state
        results.append(res)
        n_done += 1
        if n_done == len(tasks) or n_done % progress_every == 0:
            print(
                f"{phase}: {n_done}/{len(tasks)} points done "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    results.sort(key=lambda r: (r["station_index"], r["point_order"]))
    return results, {
        "phase": phase,
        "n_tasks": len(tasks),
        "wall_time_s": time.perf_counter() - t0,
        "worker_pids_observed": sorted(int(pid) for pid in worker_states),
        "workers_observed": len(worker_states),
        "worker_states": worker_states,
    }


def _best_point(records: list[dict]) -> dict | None:
    best = None
    best_oi = float("-inf")
    for rec in sorted(records, key=lambda r: r["point_order"]):
        if rec["status"] != "ok" or rec["omega_i"] is None:
            continue
        if float(rec["omega_i"]) > best_oi:
            best = rec
            best_oi = float(rec["omega_i"])
    return best


def _same_candidate(left: dict | None, right: dict | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    fields = ("omega_i", "alpha", "psi", "c_r", "c_i")
    return all(left.get(field) == right.get(field) for field in fields)


def _sequential_refine_best(coarse_best: dict | None, refine_records: list[dict]):
    if coarse_best is None:
        return None
    best = dict(coarse_best)
    best_oi = float(best["omega_i"])
    for rec in sorted(refine_records, key=lambda r: r["point_order"]):
        if rec["status"] != "ok" or rec["omega_i"] is None:
            continue
        if float(rec["omega_i"]) > best_oi:
            best = rec
            best_oi = float(rec["omega_i"])
    return best


def _batch_argmax_refine_best(coarse_best: dict | None, refine_records: list[dict]):
    if coarse_best is None:
        return None
    candidates = [coarse_best] + [
        rec
        for rec in sorted(refine_records, key=lambda r: r["point_order"])
        if rec["status"] == "ok" and rec["omega_i"] is not None
    ]
    return max(candidates, key=lambda rec: float(rec["omega_i"]))


def _row_from_best(mach: float, R: float, best: dict | None) -> dict:
    if best is None:
        return {
            "mach": float(mach),
            "R": float(R),
            "omega_i_max": None,
            "alpha_peak": None,
            "psi_peak": None,
            "c_r": None,
            "c_i": None,
        }
    return {
        "mach": float(mach),
        "R": float(R),
        "omega_i_max": float(best["omega_i"]),
        "alpha_peak": float(best["alpha"]),
        "psi_peak": float(best["psi"]),
        "c_r": float(best["c_r"]),
        "c_i": float(best["c_i"]),
    }


def _build_coarse_tasks(engine) -> list[dict]:
    mach = 10.0
    stations = [float(x) for x in engine.DEFAULT_R_SWEEPS[mach]]
    alphas = [float(x) for x in engine.ALPHA_GRID[mach]]
    psis = [float(x) for x in engine.PSI_GRID[mach]]
    tasks = []
    task_id = 0
    for station_index, R in enumerate(stations):
        point_order = 0
        for psi in psis:
            for alpha in alphas:
                tasks.append(
                    {
                        "phase": "coarse",
                        "task_id": task_id,
                        "station_index": station_index,
                        "R": R,
                        "alpha": alpha,
                        "psi": psi,
                        "point_order": point_order,
                        "mach": mach,
                        "N": int(engine.N_BY_MACH[mach]),
                        "y_max": float(engine.Y_MAX_BY_MACH[mach]),
                    }
                )
                task_id += 1
                point_order += 1
    return tasks


def _build_refine_tasks(engine, coarse_best_by_station: dict[int, dict | None]):
    mach = 10.0
    alpha_grid = engine.ALPHA_GRID[mach]
    da = float(alpha_grid[1] - alpha_grid[0])
    tasks = []
    windows = {}
    task_id = 0
    for station_index, best in coarse_best_by_station.items():
        if best is None:
            windows[str(station_index)] = []
            continue
        window = [
            float(a)
            for a in engine.np.arange(
                float(best["alpha"]) - da,
                float(best["alpha"]) + da + 1e-12,
                da / 4.0,
            )
            if float(a) > 0.0
        ]
        windows[str(station_index)] = window
        for point_order, alpha in enumerate(window):
            tasks.append(
                {
                    "phase": "refine",
                    "task_id": task_id,
                    "station_index": int(station_index),
                    "R": float(best["R"]),
                    "alpha": alpha,
                    "psi": float(best["psi"]),
                    "point_order": point_order,
                    "mach": mach,
                    "N": int(engine.N_BY_MACH[mach]),
                    "y_max": float(engine.Y_MAX_BY_MACH[mach]),
                }
            )
            task_id += 1
    return tasks, windows


def _summarize_station_points(records_by_station: dict[int, list[dict]]) -> list[dict]:
    summary = []
    for station_index in sorted(records_by_station):
        records = records_by_station[station_index]
        valid = [r for r in records if r["status"] == "ok" and r["omega_i"] is not None]
        summary.append(
            {
                "station_index": station_index,
                "R": records[0]["R"] if records else None,
                "n_points": len(records),
                "n_valid": len(valid),
                "n_errors": sum(1 for r in records if r["status"] != "ok"),
            }
        )
    return summary


def _group_by_station(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for rec in records:
        grouped.setdefault(int(rec["station_index"]), []).append(rec)
    for station_records in grouped.values():
        station_records.sort(key=lambda r: r["point_order"])
    return grouped


def _load_prior_anchors() -> dict:
    return {
        "serial": _load_anchor_summary("cpu_fig10_4_m10_serial.json"),
        "station_parallel_original": _load_anchor_summary(
            "cpu_fig10_4_m10_61w_blas1t.json"
        ),
        "station_parallel_rerun": _load_anchor_summary(
            "cpu_fig10_4_m10_61w_blas1t_rerun.json"
        ),
    }


def run(args) -> int:
    _ensure_repo_on_path()
    _set_blas_threads(args.blas_threads)
    blas_env_before_driver = {key: os.environ.get(key) for key in BLAS_THREAD_VARS}
    engine = _import_driver()
    import numpy as np
    import scipy
    import pymack

    workload = _workload_config(engine)
    coarse_tasks = _build_coarse_tasks(engine)
    total_t0 = time.perf_counter()
    phase_meta = {}
    worker_states = {}
    ctx = multiprocessing.get_context("spawn")
    n_workers = max(1, min(int(args.workers), len(coarse_tasks)))
    print(
        f"point-parallel M10 coarse: {len(coarse_tasks)} points "
        f"across {n_workers} workers",
        flush=True,
    )
    with _cf.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(args.blas_threads,),
    ) as executor:
        coarse_results, coarse_meta = _run_tasks(
            executor, coarse_tasks, phase="coarse", progress_every=50
        )
        phase_meta["coarse"] = coarse_meta
        worker_states.update(coarse_meta["worker_states"])
        coarse_by_station = _group_by_station(coarse_results)
        coarse_best_by_station = {
            station_index: _best_point(records)
            for station_index, records in coarse_by_station.items()
        }
        refine_tasks, refine_windows = _build_refine_tasks(engine, coarse_best_by_station)
        print(
            f"point-parallel M10 refine: {len(refine_tasks)} points "
            f"across {min(int(args.workers), max(1, len(refine_tasks)))} workers",
            flush=True,
        )
        refine_results, refine_meta = _run_tasks(
            executor, refine_tasks, phase="refine", progress_every=20
        )
        phase_meta["refine"] = refine_meta
        worker_states.update(refine_meta["worker_states"])

    total_wall = time.perf_counter() - total_t0
    refine_by_station = _group_by_station(refine_results)
    stations = [float(x) for x in engine.DEFAULT_R_SWEEPS[10.0]]
    rows = []
    refine_equivalence = []
    fallback_used = False
    for station_index, R in enumerate(stations):
        coarse_best = coarse_best_by_station.get(station_index)
        station_refine = refine_by_station.get(station_index, [])
        sequential_best = _sequential_refine_best(coarse_best, station_refine)
        batch_best = _batch_argmax_refine_best(coarse_best, station_refine)
        matches = _same_candidate(sequential_best, batch_best)
        if not matches:
            fallback_used = True
        final_best = sequential_best
        final_oi = None if final_best is None else final_best.get("omega_i")
        ties_at_final = 0
        if final_oi is not None:
            candidates = [coarse_best] + station_refine
            ties_at_final = sum(
                1
                for rec in candidates
                if rec is not None
                and rec.get("status") == "ok"
                and rec.get("omega_i") == final_oi
            )
        refine_equivalence.append(
            {
                "station_index": station_index,
                "R": R,
                "window_alphas": refine_windows.get(str(station_index), []),
                "batch_argmax_matches_sequential_strict": matches,
                "used_sequential_strict_fallback": not matches,
                "ties_at_final_growth": ties_at_final,
                "coarse_best": None
                if coarse_best is None
                else {
                    "alpha": coarse_best["alpha"],
                    "psi": coarse_best["psi"],
                    "omega_i": coarse_best["omega_i"],
                    "c_r": coarse_best["c_r"],
                    "c_i": coarse_best["c_i"],
                },
            }
        )
        rows.append(_row_from_best(10.0, R, final_best))

    ref_rows, ref_verdict = _load_reference()
    identity = _identity_check(rows, ref_rows, ref_verdict)
    zero_diff_fields = {
        field: float(diff) == 0.0
        for field, diff in identity.get("max_abs_diff", {}).items()
    }
    zero_diff_ok = bool(zero_diff_fields) and all(zero_diff_fields.values())
    point_errors = [
        rec
        for rec in coarse_results + refine_results
        if rec.get("status") != "ok"
    ]
    identity_ok = bool(identity["ok"] and zero_diff_ok and not point_errors)
    status = (
        "valid_point_parallel_cpu_anchor"
        if identity_ok
        else "invalid_point_parallel_cpu_anchor"
    )
    prior_anchors = _load_prior_anchors()
    tuned_original = prior_anchors["station_parallel_original"]
    tuned_rerun = prior_anchors["station_parallel_rerun"]
    comparisons = {
        "speedup_vs_serial": None,
        "speedup_vs_station_parallel_original": None,
        "speedup_vs_station_parallel_rerun": None,
    }
    if prior_anchors["serial"] and prior_anchors["serial"].get("wall_time_s"):
        comparisons["speedup_vs_serial"] = (
            float(prior_anchors["serial"]["wall_time_s"]) / total_wall
        )
    if tuned_original and tuned_original.get("wall_time_s"):
        comparisons["speedup_vs_station_parallel_original"] = (
            float(tuned_original["wall_time_s"]) / total_wall
        )
    if tuned_rerun and tuned_rerun.get("wall_time_s"):
        comparisons["speedup_vs_station_parallel_rerun"] = (
            float(tuned_rerun["wall_time_s"]) / total_wall
        )

    payload = {
        "schema_version": 1,
        "artifact": args.output.name,
        "purpose": "Point-parallel CPU tuned anchor for Mack Fig. 10.4 M10",
        "status": status,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "generated_at_unix": time.time(),
        "repo": str(REPO),
        "command_argv": sys.argv,
        "workers_requested": int(args.workers),
        "workers_effective_scheduled": {
            "coarse": min(int(args.workers), len(coarse_tasks)),
            "refine": min(int(args.workers), len(refine_tasks)),
        },
        "workers_observed": {
            "coarse": phase_meta["coarse"]["workers_observed"],
            "refine": phase_meta["refine"]["workers_observed"],
            "total_unique": len(worker_states),
        },
        "blas_threads_requested": args.blas_threads,
        "blas_env_before_driver_import": blas_env_before_driver,
        "blas_env_after_driver_import": {
            key: os.environ.get(key) for key in BLAS_THREAD_VARS
        },
        "worker_blas_states": worker_states,
        "wall_time_s": total_wall,
        "phase_timings": {
            "coarse_wall_time_s": phase_meta["coarse"]["wall_time_s"],
            "refine_wall_time_s": phase_meta["refine"]["wall_time_s"],
        },
        "workload": {
            **workload,
            "scheduler": "point_parallel_process_pool",
            "coarse_tasks": len(coarse_tasks),
            "refine_tasks": len(refine_tasks),
            "total_first_mode_growth_calls": len(coarse_tasks) + len(refine_tasks),
            "point_solver": "verification.compute_mack_fig10_4.first_mode_growth",
            "refine_parallel_reduction": (
                "evaluate whole station window in parallel, then reduce with "
                "the driver strict-improvement order"
            ),
        },
        "prior_anchor_artifacts": prior_anchors,
        "comparisons_to_prior_anchors": comparisons,
        "identity_check": {
            **identity,
            "zero_diff_fields": zero_diff_fields,
            "zero_diff_required_ok": zero_diff_ok,
            "point_errors": point_errors[:20],
            "point_errors_truncated": len(point_errors) > 20,
        },
        "refine_equivalence": {
            "all_batch_argmax_matches_sequential_strict": all(
                item["batch_argmax_matches_sequential_strict"]
                for item in refine_equivalence
            ),
            "any_sequential_fallback_used": fallback_used,
            "station_checks": refine_equivalence,
        },
        "coarse_summary": _summarize_station_points(coarse_by_station),
        "refine_summary": _summarize_station_points(refine_by_station),
        "rows": rows,
        "coarse_points": coarse_results,
        "refine_points": refine_results,
        "machine": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cores": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pymack_file": str(Path(pymack.__file__).resolve()),
            "driver_file": str(Path(engine.__file__).resolve()),
            "gpu": _gpu_info(),
        },
        "reference_artifacts": {
            "pymack_curve": str(
                (
                    REPO
                    / "verification"
                    / "first_mode"
                    / "mack_fig10_4_M100"
                    / "pymack_curve.json"
                ).relative_to(REPO)
            ),
            "verdict": str(
                (
                    REPO
                    / "verification"
                    / "first_mode"
                    / "mack_fig10_4_M100"
                    / "verdict.json"
                ).relative_to(REPO)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"point-parallel M10 CPU anchor wall={total_wall:.3f}s "
        f"identity_ok={identity_ok} zero_diff_ok={zero_diff_ok} "
        f"artifact={args.output}",
        flush=True,
    )
    if not identity_ok:
        print("point-parallel row INVALID; see artifact identity_check", flush=True)
    return 0 if identity_ok else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=61)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
