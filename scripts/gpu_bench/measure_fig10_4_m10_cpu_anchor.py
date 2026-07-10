"""Measure CPU anchors for Mack Fig. 10.4 M10.

This is a measurement harness only.  It imports the deployed
verification/compute_mack_fig10_4.py driver and compares fresh M10 rows against
the committed verification/first_mode/mack_fig10_4_M100 reference without
regenerating that reference.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
REF_DIR = REPO / "verification" / "first_mode" / "mack_fig10_4_M100"


def _set_blas_threads(n: int | None) -> None:
    if n is None:
        return
    for key in BLAS_THREAD_VARS:
        os.environ[key] = str(int(n))


def _import_driver():
    # Import only after optional BLAS env pinning; the driver itself also
    # setdefault-pins BLAS before importing numpy/scipy.
    from verification import compute_mack_fig10_4 as engine

    return engine


def _gpu_info() -> list[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return [f"nvidia-smi unavailable: {type(exc).__name__}: {exc}"]
    if proc.returncode != 0:
        return [f"nvidia-smi failed rc={proc.returncode}: {proc.stderr.strip()}"]
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_reference() -> tuple[list[dict], dict]:
    return (
        json.loads((REF_DIR / "pymack_curve.json").read_text(encoding="utf-8")),
        json.loads((REF_DIR / "verdict.json").read_text(encoding="utf-8")),
    )


def _load_anchor_summary(filename: str) -> dict | None:
    path = REPO / "docs" / "gpu" / "benchmarks" / filename
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact": str(path.relative_to(REPO)),
        "wall_time_s": payload.get("wall_time_s"),
        "identity_ok": payload.get("identity_check", {}).get("ok"),
        "mode": payload.get("mode"),
        "workers_requested": payload.get("workers_requested"),
        "workers_effective": payload.get("workers_effective"),
        "blas_threads_requested": payload.get("blas_threads_requested"),
    }


def _float_diff(a, b):
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _float_match(a, b, *, rel_tol=5e-13, abs_tol=5e-13) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def _identity_check(rows: list[dict], ref_rows: list[dict], ref_verdict: dict) -> dict:
    fields = ("omega_i_max", "alpha_peak", "psi_peak", "c_r", "c_i")
    station_checks = []
    all_ok = len(rows) == len(ref_rows)
    max_abs = {field: 0.0 for field in fields}
    for i, (got, ref) in enumerate(zip(rows, ref_rows)):
        row_ok = float(got["R"]) == float(ref["R"])
        band_ok = (got.get("omega_i_max") is None) == (ref.get("omega_i_max") is None)
        field_diffs = {}
        field_matches = {}
        for field in fields:
            diff = _float_diff(got.get(field), ref.get(field))
            field_diffs[field] = diff
            if diff is not None:
                max_abs[field] = max(max_abs[field], diff)
            field_matches[field] = _float_match(got.get(field), ref.get(field))
            row_ok = row_ok and field_matches[field]
        row_ok = row_ok and band_ok
        station_checks.append(
            {
                "index": i,
                "R": got.get("R"),
                "reference_R": ref.get("R"),
                "band_decision_match": band_ok,
                "field_matches": field_matches,
                "abs_diffs": field_diffs,
                "ok": row_ok,
            }
        )
        all_ok = all_ok and row_ok

    n_valid = sum(1 for row in rows if row.get("omega_i_max") is not None)
    ref_n_valid = sum(1 for row in ref_rows if row.get("omega_i_max") is not None)
    verdict_ok = (
        ref_verdict.get("case_id") == "mack_fig10_4_M100"
        and ref_verdict.get("verdict") not in (None, "pending")
        and n_valid == ref_n_valid
    )
    all_ok = all_ok and verdict_ok
    return {
        "ok": bool(all_ok),
        "row_count_match": len(rows) == len(ref_rows),
        "station_count": len(rows),
        "reference_station_count": len(ref_rows),
        "n_valid_stations": n_valid,
        "reference_n_valid_stations": ref_n_valid,
        "committed_verdict": ref_verdict.get("verdict"),
        "committed_verdict_identity_ok": verdict_ok,
        "max_abs_diff": max_abs,
        "station_checks": station_checks,
    }


def _workload_config(engine) -> dict:
    mach = 10.0
    alpha_grid = [float(x) for x in engine.ALPHA_GRID[mach]]
    psi_grid = [float(x) for x in engine.PSI_GRID[mach]]
    stations = [float(x) for x in engine.DEFAULT_R_SWEEPS[mach]]
    coarse_points_per_station = len(alpha_grid) * len(psi_grid)
    refine_solves_per_station = 9
    return {
        "driver": "verification/compute_mack_fig10_4.py",
        "mach": mach,
        "stations": stations,
        "n_stations": len(stations),
        "N": int(engine.N_BY_MACH[mach]),
        "y_max": float(engine.Y_MAX_BY_MACH[mach]),
        "matrix_dim": int(5 * (engine.N_BY_MACH[mach] + 1)),
        "alpha_grid": alpha_grid,
        "psi_grid": psi_grid,
        "coarse_points_per_station": coarse_points_per_station,
        "refine_solves_per_station": refine_solves_per_station,
        "total_solves": len(stations)
        * (coarse_points_per_station + refine_solves_per_station),
        "cr_band": [float(engine.CR_LO), float(engine.CR_HI)],
        "ci_cap": float(engine.CI_CAP),
        "refine_rule": "np.arange(best_a - da, best_a + da + 1e-12, da/4), skip a <= 0, strict oi > best_oi",
    }


def _run(mode: str, workers: int | None, engine) -> list[dict]:
    if mode == "serial":
        return engine.compute_curve(10.0, verbose=True)
    if workers is None:
        raise ValueError("parallel mode requires --workers")
    by_mach = engine.compute_curves_parallel([10.0], max_workers=int(workers))
    return by_mach[10.0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("serial", "parallel"), required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--blas-threads", type=int, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    _set_blas_threads(args.blas_threads)
    blas_env_before_driver = {key: os.environ.get(key) for key in BLAS_THREAD_VARS}
    engine = _import_driver()
    import numpy as np
    import scipy
    import pymack

    t0 = time.perf_counter()
    rows = _run(args.mode, args.workers, engine)
    wall = time.perf_counter() - t0

    ref_rows, ref_verdict = _load_reference()
    identity = _identity_check(rows, ref_rows, ref_verdict)
    workload = _workload_config(engine)
    prior_anchors = {
        "serial": _load_anchor_summary("cpu_fig10_4_m10_serial.json"),
        "tuned_station_parallel": _load_anchor_summary(
            "cpu_fig10_4_m10_61w_blas1t.json"
        ),
    }
    original = prior_anchors["tuned_station_parallel"]
    rerun_comparison = None
    if (
        original is not None
        and args.output.name != "cpu_fig10_4_m10_61w_blas1t.json"
        and args.mode == "parallel"
        and args.workers == 61
        and args.blas_threads == 1
    ):
        original_wall = original.get("wall_time_s")
        if original_wall is not None:
            rerun_comparison = {
                "original_wall_time_s": float(original_wall),
                "rerun_wall_time_s": wall,
                "delta_s": wall - float(original_wall),
                "ratio_rerun_over_original": wall / float(original_wall),
            }
    payload = {
        "artifact": args.output.name,
        "purpose": "Stage 1 CPU anchor for Mack Fig. 10.4 M10 flagship workload",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "generated_at_unix": time.time(),
        "repo": str(REPO),
        "command_argv": sys.argv,
        "mode": args.mode,
        "workers_requested": args.workers,
        "workers_effective": 1 if args.mode == "serial" else min(args.workers, len(workload["stations"])),
        "blas_threads_requested": args.blas_threads,
        "blas_env_before_driver_import": blas_env_before_driver,
        "blas_env_after_driver_import": {key: os.environ.get(key) for key in BLAS_THREAD_VARS},
        "wall_time_s": wall,
        "prior_anchor_artifacts": prior_anchors,
        "rerun_comparison": rerun_comparison,
        "workload": workload,
        "identity_check": identity,
        "rows": rows,
        "machine": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cores": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pymack_file": str(Path(pymack.__file__).resolve()),
            "gpu": _gpu_info(),
        },
        "reference_artifacts": {
            "pymack_curve": str((REF_DIR / "pymack_curve.json").relative_to(REPO)),
            "verdict": str((REF_DIR / "verdict.json").relative_to(REPO)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"{args.mode} M10 CPU anchor wall={wall:.3f}s "
        f"identity_ok={identity['ok']} artifact={args.output}"
    )
    return 0 if identity["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
