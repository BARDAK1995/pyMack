"""Regression gates for Mack Fig. 10.6 single-Mach verification records."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYMACK_NO_BANNER",
)


def _verifier():
    before = {name: os.environ.get(name) for name in THREAD_ENV}
    try:
        from verification import verify_mack_fig10_6 as verifier
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return verifier


def test_verifier_import_pins_blas_before_numpy():
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env.pop(name, None)
    code = (
        "import json; import verification.verify_mack_fig10_6; "
        "from threadpoolctl import threadpool_info; "
        "print(json.dumps(threadpool_info()))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    pools = json.loads(proc.stdout)
    assert pools
    assert {pool["num_threads"] for pool in pools} == {1}


def test_single_mach_effective_parameters_match_recorded_metadata(
    monkeypatch, tmp_path,
):
    verifier = _verifier()
    captured = {}
    rows = [
        {"R": 300.0, "omega_i_max": 8.8e-4, "alpha_peak": 0.2075,
         "c_r": 0.91, "c_i": 0.0042},
        {"R": 900.0, "omega_i_max": 2.8e-3, "alpha_peak": 0.22,
         "c_r": 0.91, "c_i": 0.0127},
        {"R": 1500.0, "omega_i_max": 3.38e-3, "alpha_peak": 0.2225,
         "c_r": 0.91, "c_i": 0.0152},
    ]

    def fake_compute_curve(mach, *, N, y_max, verbose):
        captured.update(mach=mach, N=N, y_max=y_max, verbose=verbose)
        return rows

    monkeypatch.setattr(verifier, "GROWTH_DIR", tmp_path)
    monkeypatch.setattr(verifier.engine, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(verifier, "hash_files", lambda *args, **kwargs: {})

    verdict = verifier.verify_mach(4.5, force=True)
    effective = verdict["provenance"]["effective_parameters"]

    assert captured["N"] == effective["N"] == verdict["metrics"]["N"] == 120
    assert captured["y_max"] == effective["y_max"] == verdict["metrics"]["y_max"] == 40.0
