"""Build Slice 20's deterministic overlay oracle pack.

The NPZ is local/gitignored and stores the complete raw generalized spectrum
plus left/right eigenvectors for all production-box and collar modes.  The
committed summary CSV carries one row per overlay point and must match the
committed 720-row verdict map exactly.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hunt.measurement_lock import measurement_lock  # noqa: E402


COMMITTED_GRID = (
    REPO
    / "verification"
    / "mixed_mode"
    / "ozgen_fig3"
    / "_compute"
    / "ozgen_M2_ci_grid.csv"
)
FLOOR_ARTIFACT = REPO / "docs" / "benchmarks" / "cpu_floor_sweep.json"
SUMMARY_DEFAULT = (
    REPO / "docs" / "benchmarks" / "oracle_pack_overlay_n128_summary.csv"
)
BLAS_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ORACLE_SEED = 20260710
COLLAR_ABS = 0.02
N_POINTS = 720
N_STATE = 516


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _set_blas_threads(n: int = 1) -> None:
    for key in BLAS_VARS:
        os.environ[key] = str(int(n))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _environment() -> dict:
    import numpy as np
    import pymack
    import scipy

    return {
        "generated_at_utc": _utc_now(),
        "repo": str(REPO),
        "git_head": _git_head(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "psutil": psutil.__version__,
        "pymack_file": str(Path(pymack.__file__).resolve()),
        "blas_env": {key: os.environ.get(key) for key in BLAS_VARS},
    }


def _load_expected() -> list[dict[str, str]]:
    with COMMITTED_GRID.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_POINTS:
        raise RuntimeError(f"expected 720 committed rows, found {len(rows)}")
    return rows


def _identity(actual: list[dict]) -> dict:
    expected = _load_expected()
    matched = 0
    mismatches = []
    per_row = []
    tolerance = 1.0e-9
    worst_abs_c_r = 0.0
    worst_abs_c_i = 0.0
    for index, (got, ref) in enumerate(zip(actual, expected)):
        ref_empty = ref["family"] == ""
        got_empty = got["family"] == ""
        ok = ref_empty and got_empty
        if not ref_empty and not got_empty:
            diff_r = abs(float(got["c_r"]) - float(ref["c_r"]))
            diff_i = abs(float(got["c_i"]) - float(ref["c_i"]))
            worst_abs_c_r = max(worst_abs_c_r, diff_r)
            worst_abs_c_i = max(worst_abs_c_i, diff_i)
            ok = (
                got["family"] == ref["family"]
                and diff_i <= tolerance
                and diff_r <= tolerance
            )
        per_row.append(ok)
        if ok:
            matched += 1
        elif len(mismatches) < 20:
            mismatches.append({"index": index, "expected": ref, "actual": got})
    return {
        "ok": matched == N_POINTS and len(actual) == N_POINTS,
        "matched_rows": matched,
        "total_rows": N_POINTS,
        "reference": str(COMMITTED_GRID.relative_to(REPO)),
        "comparison": (
            "family exact plus c_r/c_i absolute tolerance 1e-9 against the "
            "committed 8e CSV fields; empty matches empty"
        ),
        "absolute_tolerance": tolerance,
        "worst_abs_c_r": worst_abs_c_r,
        "worst_abs_c_i": worst_abs_c_i,
        "mismatches_first20": mismatches,
        "per_row": per_row,
    }


def _payload_and_grid():
    import numpy as np
    from pymack import make_ozgen_profile
    from pymack.scales import delta_star_over_lstar
    from pymack.sweep import CBand

    profile = make_ozgen_profile(2.0)
    y_max = 6.0 * float(delta_star_over_lstar(profile))
    alphas = np.linspace(0.02, 0.24, 30)
    res_values = np.logspace(np.log10(300.0), np.log10(4500.0), 24)
    families = (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.88, 0.99, ci_abs_max=0.05, label="Mack"),
    )
    payload = {
        "kind": "temporal",
        "operator": "ozgen_2d",
        "profile": profile,
        "families": families,
        "return_eigenvectors": True,
        "cpu_blas_threads": 1,
        "solver_kwargs": {
            "Ma": 2.0,
            "Pr": 0.72,
            "gamma": 1.4,
            "N": 128,
            "y_max": y_max,
            "L": None,
            "wall_bc": "isothermal",
            "length_scale": "L_star",
            "lambda_mu_ratio": 0.0,
        },
    }
    return payload, alphas, res_values, y_max


_PAYLOAD = None


def _worker_init(payload: dict) -> None:
    global _PAYLOAD
    _set_blas_threads(1)
    from pymack import sweep

    _PAYLOAD = payload
    sweep._worker_init(payload)


def _production_mask(values):
    import numpy as np

    return (np.abs(values.imag) < 0.05) & (
        (values.real < 0.45)
        | ((values.real > 0.88) & (values.real < 0.99))
    )


def _collar_mask(values):
    import numpy as np

    ci = 0.05 + COLLAR_ABS
    ts = (
        (values.real > -0.5 - COLLAR_ABS)
        & (values.real < 0.45 + COLLAR_ABS)
        & (np.abs(values.imag) < ci)
    )
    mack = (
        (values.real > 0.88 - COLLAR_ABS)
        & (values.real < 0.99 + COLLAR_ABS)
        & (np.abs(values.imag) < ci)
    )
    return ts | mack


def _distance_to_box_edge(value: complex) -> float:
    edges = (-0.5, 0.45, 0.88, 0.99)
    return float(
        min(
            *(abs(value.real - edge) for edge in edges),
            abs(value.imag - 0.05),
            abs(value.imag + 0.05),
        )
    )


def _oracle_task(task: tuple[int, int, float, float]) -> dict:
    import numpy as np
    from scipy import linalg
    from pymack import sweep

    i, j, alpha, re_value = task
    try:
        A, B = sweep._assemble_temporal_operator(_PAYLOAD, alpha, re_value)
        values, left, right = linalg.eig(A, B, left=True, right=True)
        if values.shape != (N_STATE,):
            raise RuntimeError(f"raw spectrum shape {values.shape}, expected {(N_STATE,)}")
        finite = np.isfinite(values)
        physical = finite & (
            (values.real > -0.5)
            & (values.real < 1.5)
            & (np.abs(values.imag) < 0.5)
        )
        physical_idx = np.flatnonzero(physical)
        physical_idx = physical_idx[np.argsort(-values[physical_idx].imag)]
        physical_values = values[physical_idx]
        prod_local = _production_mask(physical_values)
        prod_idx = physical_idx[prod_local]
        collar_local = _collar_mask(physical_values)
        collar_idx = physical_idx[collar_local]
        collar_values = values[collar_idx]
        collar_in_box = _production_mask(collar_values)

        if len(prod_idx):
            prod_values = values[prod_idx]
            winner_local = int(np.argmax(prod_values.imag))
            winner_idx = int(prod_idx[winner_local])
            winner = complex(values[winner_idx])
            family = "TS" if winner.real < 0.45 else "Mack"
            verdict = {
                "family": family,
                "c_r": float(winner.real),
                "c_i": float(winner.imag),
                "raw_index": winner_idx,
            }
            sorted_ci = sorted((float(z.imag) for z in prod_values), reverse=True)
            winner_margin = (
                sorted_ci[0] - sorted_ci[1] if len(sorted_ci) > 1 else math.inf
            )
            winner_edge_distance = _distance_to_box_edge(winner)
        else:
            prod_values = np.empty(0, dtype=np.complex128)
            verdict = {"family": "", "c_r": math.nan, "c_i": math.nan}
            winner_margin = math.nan
            winner_edge_distance = math.nan

        finite_physical_values = values[physical_idx]
        min_edge = min(
            (_distance_to_box_edge(complex(z)) for z in finite_physical_values),
            default=math.nan,
        )
        return {
            "status": "ok",
            "i": i,
            "j": j,
            "alpha": alpha,
            "Re": re_value,
            "full_values": values,
            "collar_values": collar_values,
            "collar_left": left[:, collar_idx],
            "collar_right": right[:, collar_idx],
            "collar_in_box": collar_in_box,
            "production_values": prod_values,
            "verdict": verdict,
            "n_finite": int(finite.sum()),
            "n_physical": int(physical.sum()),
            "min_abs_distance_to_box_edge": min_edge,
            "winner_distance_to_box_edge": winner_edge_distance,
            "winner_imag_margin": winner_margin,
            "worker_pid": os.getpid(),
            "worker_affinity": psutil.Process().cpu_affinity(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "i": i,
            "j": j,
            "alpha": alpha,
            "Re": re_value,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
            "worker_affinity": psutil.Process().cpu_affinity(),
        }


def _collect_affinity(records: list[dict], workers: int) -> dict:
    masks: dict[str, list[list[int]]] = {}
    for record in records:
        pid = str(record["worker_pid"])
        mask = record["worker_affinity"]
        if mask not in masks.setdefault(pid, []):
            masks[pid].append(mask)
    return {
        "mechanism": "psutil.Process.cpu_affinity observed inside every oracle worker task",
        "worker_masks_observed": masks,
        "n_worker_pids_observed": len(masks),
        "n_worker_pids_expected": int(workers),
        "all_workers_observed": len(masks) == int(workers),
    }


def _offsets(lengths: list[int]):
    import numpy as np

    out = np.zeros(len(lengths) + 1, dtype=np.int64)
    out[1:] = np.cumsum(lengths, dtype=np.int64)
    return out


def _pack_arrays(records: list[dict], alphas, res_values, metadata: dict) -> dict:
    import numpy as np

    collar_lengths = [len(row["collar_values"]) for row in records]
    prod_lengths = [len(row["production_values"]) for row in records]
    collar_offsets = _offsets(collar_lengths)
    prod_offsets = _offsets(prod_lengths)
    collar_values = np.concatenate(
        [row["collar_values"] for row in records]
    ).astype(np.complex128, copy=False)
    collar_left = np.concatenate(
        [row["collar_left"].T for row in records], axis=0
    ).astype(np.complex128, copy=False)
    collar_right = np.concatenate(
        [row["collar_right"].T for row in records], axis=0
    ).astype(np.complex128, copy=False)
    collar_in_box = np.concatenate(
        [row["collar_in_box"] for row in records]
    ).astype(np.bool_, copy=False)
    production_values = np.concatenate(
        [row["production_values"] for row in records]
    ).astype(np.complex128, copy=False)
    verdict_values = np.array(
        [complex(row["verdict"]["c_r"], row["verdict"]["c_i"]) for row in records],
        dtype=np.complex128,
    )
    family_code = np.array(
        [{"": 0, "TS": 1, "Mack": 2}[row["verdict"]["family"]] for row in records],
        dtype=np.int8,
    )
    return {
        "schema_version": np.array([1], dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        "oracle_seed": np.array([ORACLE_SEED], dtype=np.int64),
        "alpha_values": np.asarray(alphas, dtype=np.float64),
        "Re_values": np.asarray(res_values, dtype=np.float64),
        "point_alpha": np.array([row["alpha"] for row in records]),
        "point_Re": np.array([row["Re"] for row in records]),
        "full_eigenvalues_raw": np.stack([row["full_values"] for row in records]),
        "collar_offsets": collar_offsets,
        "collar_eigenvalues": collar_values,
        "collar_left_eigenvectors": collar_left,
        "collar_right_eigenvectors": collar_right,
        "collar_is_production_in_box": collar_in_box,
        "production_candidate_offsets": prod_offsets,
        "production_candidate_eigenvalues": production_values,
        "verdict_eigenvalues": verdict_values,
        "verdict_family_code": family_code,
        "min_abs_distance_to_box_edge": np.array(
            [row["min_abs_distance_to_box_edge"] for row in records]
        ),
        "winner_distance_to_box_edge": np.array(
            [row["winner_distance_to_box_edge"] for row in records]
        ),
        "winner_imag_margin": np.array(
            [row["winner_imag_margin"] for row in records]
        ),
    }


def _write_summary(
    path: Path, records: list[dict], identity: dict, environment: dict, lock_event: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "point_index",
        "alpha_index",
        "Re_index",
        "alpha",
        "Re",
        "n_raw",
        "n_finite",
        "n_physical",
        "n_production_candidates",
        "n_collar_modes",
        "verdict_family",
        "verdict_c_r",
        "verdict_c_i",
        "winner_imag_margin",
        "min_abs_distance_to_box_edge",
        "winner_distance_to_box_edge",
        "identity_match",
        "worker_pid",
        "worker_affinity",
        "git_head",
        "pymack_file",
        "python",
        "numpy",
        "scipy",
        "lock_owner",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(records):
            writer.writerow(
                {
                    "point_index": index,
                    "alpha_index": row["i"],
                    "Re_index": row["j"],
                    "alpha": f"{row['alpha']:.17g}",
                    "Re": f"{row['Re']:.17g}",
                    "n_raw": len(row["full_values"]),
                    "n_finite": row["n_finite"],
                    "n_physical": row["n_physical"],
                    "n_production_candidates": len(row["production_values"]),
                    "n_collar_modes": len(row["collar_values"]),
                    "verdict_family": row["verdict"]["family"],
                    "verdict_c_r": f"{row['verdict']['c_r']:.17g}",
                    "verdict_c_i": f"{row['verdict']['c_i']:.17g}",
                    "winner_imag_margin": row["winner_imag_margin"],
                    "min_abs_distance_to_box_edge": row[
                        "min_abs_distance_to_box_edge"
                    ],
                    "winner_distance_to_box_edge": row[
                        "winner_distance_to_box_edge"
                    ],
                    "identity_match": identity["per_row"][index],
                    "worker_pid": row["worker_pid"],
                    "worker_affinity": json.dumps(row["worker_affinity"]),
                    "git_head": environment["git_head"],
                    "pymack_file": environment["pymack_file"],
                    "python": environment["python"],
                    "numpy": environment["numpy"],
                    "scipy": environment["scipy"],
                    "lock_owner": lock_event["owner"],
                }
            )


def _run(args) -> int:
    if args.workload not in ("overlay-n128", "overlay_n128"):
        raise ValueError("Slice 20 oracle supports only overlay-n128")
    _set_blas_threads(1)
    floor = json.loads(FLOOR_ARTIFACT.read_text(encoding="utf-8"))
    workers = int(floor["best_honest_overlay_floor"]["workers"])
    payload, alphas, res_values, y_max = _payload_and_grid()
    tasks = [
        (i, j, float(alphas[i]), float(res_values[j]))
        for i in range(len(alphas))
        for j in range(len(res_values))
    ]
    ctx = multiprocessing.get_context("spawn")
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temp_out = args.out.with_name(args.out.name + ".tmp.npz")
    if temp_out.exists():
        temp_out.unlink()

    with measurement_lock(run_label="oracle-build-overlay-n128") as lock_event:
        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(payload,),
        ) as pool:
            futures = [pool.submit(_oracle_task, task) for task in tasks]
            for count, future in enumerate(cf.as_completed(futures), start=1):
                records.append(future.result())
                if count % 120 == 0 or count == len(tasks):
                    print(f"oracle: {count}/{len(tasks)} points complete", flush=True)
        solve_wall = time.perf_counter() - t0
        records.sort(key=lambda row: (row["i"], row["j"]))
        errors = [row for row in records if row["status"] != "ok"]
        if errors:
            raise RuntimeError(f"oracle worker failures: {errors[:3]}")
        actual = [row["verdict"] for row in records]
        identity = _identity(actual)
        affinity = _collect_affinity(records, workers)
        if not identity["ok"]:
            raise RuntimeError(
                f"oracle verdict cross-check {identity['matched_rows']}/720; refusing pack"
            )
        if not affinity["all_workers_observed"]:
            raise RuntimeError(
                "oracle affinity provenance incomplete: "
                f"{affinity['n_worker_pids_observed']}/{workers} workers"
            )
        environment = _environment()
        metadata = {
            "schema_version": 1,
            "artifact": args.out.name,
            "workload": {
                "id": "overlay-n128",
                "operator": "ozgen_2d",
                "N": 128,
                "matrix_dim": N_STATE,
                "n_points": N_POINTS,
                "Ma": 2.0,
                "y_max": y_max,
            },
            "full_spectrum": "all 516 raw generalized eigenvalues including nonfinite",
            "production_boxes": {
                "physical": "-0.5 < Re(c) < 1.5 and |Im(c)| < 0.5",
                "TS": "Re(c) < 0.45 and |Im(c)| < 0.05",
                "Mack": "0.88 < Re(c) < 0.99 and |Im(c)| < 0.05",
                "collar_absolute": COLLAR_ABS,
            },
            "vector_storage": (
                "SciPy generalized left and right eigenvectors for union of strict "
                "production boxes and absolute 0.02 collar; flat arrays plus offsets"
            ),
            "oracle_seed": ORACLE_SEED,
            "workers": workers,
            "blas_threads": 1,
            "solve_wall_time_s": solve_wall,
            "identity_check": {key: value for key, value in identity.items() if key != "per_row"},
            "affinity": affinity,
            "environment": environment,
            "lock": lock_event,
            "lock_release_note": (
                "exception-safe context releases immediately after pack and summary writes"
            ),
            "summary_csv": str(args.summary.relative_to(REPO)),
        }
        arrays = _pack_arrays(records, alphas, res_values, metadata)
        import numpy as np

        np.savez_compressed(temp_out, **arrays)
        _write_summary(args.summary, records, identity, environment, lock_event)
        os.replace(temp_out, args.out)

    print(
        f"oracle valid: identity={identity['matched_rows']}/720 "
        f"vectors={len(arrays['collar_eigenvalues'])} out={args.out}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    args = ap.parse_args(argv)
    try:
        return _run(args)
    except BaseException:
        if args.out.exists():
            args.out.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
