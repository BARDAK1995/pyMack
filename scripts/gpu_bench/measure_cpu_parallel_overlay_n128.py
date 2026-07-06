"""Measure the PARALLEL CPU baseline on the overlay workload (N=128, 720 nodes).

The committed 1598.68 s baseline (verification/mixed_mode/ozgen_fig3/_compute/
ozgen_M2.json, dated 2026-07-02) predates the slice-06 parallel sweep facade and its
per-node arithmetic (2.22 s/node) matches a SERIAL run.  Today's deployed CPU path is
temporal_sweep(backend='cpu') with cpu_workers defaulting to os.cpu_count() (61-clamped
on Windows).  This measurement answers: what does the FAIR multi-core CPU baseline do
on the identical overlay workload?  Measurement only — no gate.

When --blas-threads 1 (or equiv) is passed, the opt-in cpu_blas_threads=1 path
is exercised and a suffixed artifact is written (different FP path; ~149 s class
on this hardware for the pinned case).

    python scripts/gpu_bench/measure_cpu_parallel_overlay_n128.py [--workers W] [--blas-threads 1]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from pymack import make_ozgen_profile
from pymack.sweep import CBand, temporal_sweep

REPO = Path(__file__).resolve().parents[2]

def _artifact_path(n, blas_threads):
    base = f"cpu_parallel_overlay_n{n}"
    if blas_threads is not None:
        base += f"_blas{int(blas_threads)}t"
    return REPO / "docs" / "gpu" / "benchmarks" / f"{base}.json"


OUT = _artifact_path(128, None)  # default (overridable in main)


def _families():
    return (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.88, 0.99, ci_abs_max=0.05, label="Mack"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None,
                    help="cpu_workers passed to the facade (None = deployed default)")
    ap.add_argument("--blas-threads", type=int, default=None,
                    help="cpu_blas_threads passed to temporal_sweep (None = unpinned default path; 1 for pinned single-thread BLAS)")
    args = ap.parse_args()

    Ma = 2.0
    alphas = np.linspace(0.02, 0.24, 30)
    Res = np.logspace(np.log10(300.0), np.log10(4500.0), 24)
    profile = make_ozgen_profile(Ma)
    # y_max rule identical to the committed baseline and probe (a): 6*dstar
    from pymack.scales import delta_star_over_lstar
    dstar = float(delta_star_over_lstar(profile))
    y_max = 6.0 * dstar

    t0 = time.perf_counter()
    res = temporal_sweep(
        profile, alphas, Res, Ma=Ma, N=args.N, y_max=y_max,
        length_scale="L_star", operator="ozgen_2d", families=_families(),
        backend="cpu", cpu_workers=args.workers,
        cpu_blas_threads=args.blas_threads,
    )
    wall = time.perf_counter() - t0

    out_path = _artifact_path(args.N, args.blas_threads)
    payload = {
        "artifact": "cpu_parallel_overlay_n128",
        "purpose": "fair multi-core CPU baseline for the overlay workload (measurement only)",
        "workload": {"Ma": Ma, "N": args.N, "n_points": 720, "y_max_rule": "6*dstar",
                     "alpha_range": [0.02, 0.24], "re_range": [300.0, 4500.0]},
        "cpu_workers_requested": args.workers,
        "cpu_workers_effective": res.meta.get("cpu_workers"),
        "cpu_blas_threads": args.blas_threads,
        "cpu_blas_threads_effective": res.meta.get("cpu_blas_threads_effective"),
        "machine_logical_cores": os.cpu_count(),
        "wall_time_s": wall,
        "committed_serial_baseline_s": 1598.6826644999674,
        "gpu_engine_wall_s_for_reference": [754.845, 743.89],
        "n_converged_per_family": {
            fam.band.label: int(np.sum(fam.converged)) for fam in res.families
        },
        "generated_at_unix": time.time(),
    }
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"CPU parallel overlay N={args.N}: wall={wall:.2f}s "
          f"workers={payload['cpu_workers_effective']} "
          f"blas_threads={payload['cpu_blas_threads_effective']} "
          f"(serial committed 1598.7s; GPU 743.9-754.8s)")
    print(f"artifact: {out_path}")


if __name__ == "__main__":
    main()
