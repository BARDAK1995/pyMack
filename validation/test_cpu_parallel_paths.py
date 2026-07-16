"""Slice-19 opt-in CPU paths and default-invocation identity gates."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pymack import make_flatplate_profile
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, TemporalFamilyResult, TemporalSweepResult, temporal_sweep


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (ROOT / "validation" / "data" / "cpu_parallel_default_invocations.json")
    .read_text(encoding="utf-8")
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ozgen_driver():
    from scripts import make_ozgen_fig3_overlay as driver
    return driver


def _mack_driver():
    # This deployed driver intentionally pins BLAS at import. A test import
    # must not leak those environment mutations into later spawned-worker
    # default-path fixtures in the same pytest process.
    import os
    names = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYMACK_NO_BANNER",
    )
    before = {name: os.environ.get(name) for name in names}
    try:
        from verification import compute_mack_fig10_4 as driver
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return driver


def test_ozgen_default_invocation_bytes_are_unchanged():
    args = _ozgen_driver().parse_args([])
    legacy = {
        "quality": args.quality, "panels": args.panels, "engine": args.engine,
        "sweep_backend": args.sweep_backend, "workers": args.workers,
    }
    assert _canonical(legacy).encode() == FIXTURE["ozgen"].encode()
    assert args.blas_threads is None
    assert args.eigenvalues_only is False


def test_mack_default_invocation_bytes_are_unchanged():
    args = _mack_driver().parse_args(["--mach", "10"])
    legacy = {
        "mach": args.mach, "r_list": args.r_list, "N": args.N,
        "y_max": args.y_max, "probe": args.probe,
    }
    assert _canonical(legacy).encode() == FIXTURE["mack_fig10_4"].encode()
    assert args.point_parallel is False
    assert args.workers is None
    assert args.verify_against_committed is False


def test_mack_fresh_process_imports_this_worktree(tmp_path):
    script = ROOT / "verification" / "compute_mack_fig10_4.py"
    code = (
        "import runpy; "
        f"d=runpy.run_path({str(script)!r}); "
        "print(d['pymack'].__file__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        check=True, capture_output=True, text=True,
    )
    assert Path(proc.stdout.strip()).resolve() == (ROOT / "pymack" / "__init__.py").resolve()


def test_mack_identity_source_requires_exact_zero():
    driver = _mack_driver()
    ref_dir = ROOT / "verification" / "first_mode" / "mack_fig10_4_M100"
    rows = json.loads((ref_dir / "pymack_curve.json").read_text(encoding="utf-8"))
    verdict = json.loads((ref_dir / "verdict.json").read_text(encoding="utf-8"))
    check = driver._identity_check(rows, rows, verdict)
    assert check["ok"]
    assert all(value == 0.0 for value in check["max_abs_diff"].values())


def test_cpu_eigenvalues_only_matches_default_selected_values():
    profile = make_flatplate_profile(2.0)
    y_max = 6.0 * delta_star_over_lstar(profile)
    kwargs = dict(
        Ma=2.0, N=31, y_max=y_max, length_scale="L_star",
        operator="ozgen_2d",
        families=(
            CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
            CBand(0.88, 0.99, ci_abs_max=0.05, label="Mack"),
        ),
        backend="cpu", cpu_workers=1,
    )
    default = temporal_sweep(profile, [0.08, 0.12], [900.0], **kwargs)
    values_only = temporal_sweep(
        profile, [0.08, 0.12], [900.0],
        cpu_eigenvalues_only=True, **kwargs)
    for expected, actual in zip(default.families, values_only.families):
        assert np.array_equal(actual.c, expected.c, equal_nan=True)
        assert np.array_equal(actual.mode_index, expected.mode_index)
        assert np.array_equal(actual.converged, expected.converged)
        assert np.all(np.isfinite(actual.residual[actual.converged]))
        assert np.all(np.isfinite(actual.edge_ratio[actual.converged]))
    assert values_only.meta["cpu_eigenvalues_only"]["enabled"] is True
    assert "cpu_eigenvalues_only" not in default.meta


def test_cpu_eigenvalues_only_rejects_3d_leakage_path():
    profile = make_flatplate_profile(2.0)
    with pytest.raises(ValueError, match="full eigenvector leakage filter"):
        temporal_sweep(
            profile, [0.08], [900.0], Ma=2.0, N=15,
            operator="mack_3d", backend="cpu", cpu_workers=1,
            cpu_eigenvalues_only=True,
        )


def test_ozgen_driver_wires_blas_and_eigenvalue_options(monkeypatch):
    driver = _ozgen_driver()
    captured = {}

    class Result:
        meta = {"backend": "cpu", "cpu_workers": 8}
        families = tuple()

    def fake_sweep(*args, **kwargs):
        captured.update(kwargs)
        # Two empty family grids keep the driver's combiner honest.
        shape = (1, 1)
        families = []
        for band in kwargs["families"]:
            families.append(TemporalFamilyResult(
                band=band, c=np.full(shape, complex(np.nan, np.nan)),
                omega_i=np.full(shape, np.nan), converged=np.zeros(shape, bool),
                residual=np.full(shape, np.nan), edge_ratio=np.full(shape, np.nan),
                seed_map=np.full(shape, -1), mode_index=np.full(shape, -1),
            ))
        result = Result()
        result.families = tuple(families)
        return result

    monkeypatch.setattr(driver, "temporal_sweep", fake_sweep)
    driver.compute_panel_sweep(
        2.0, np.array([900.0]), np.array([0.08]), 31,
        workers=8, blas_threads=1, eigenvalues_only=True, verbose=False,
    )
    assert captured["cpu_workers"] == 8
    assert captured["cpu_blas_threads"] == 1
    assert captured["cpu_eigenvalues_only"] is True


def test_ozgen_committed_grid_gate_uses_artifact_precision(tmp_path):
    driver = _ozgen_driver()
    reference = (
        ROOT / "verification" / "mixed_mode" / "ozgen_fig3"
        / "_compute" / "ozgen_M2_ci_grid.csv"
    )
    check = driver.verify_grid_against_committed(reference, reference=reference)
    assert check["ok"]
    assert check["rows_matched"] == 720
    assert check["byte_equal"] is True
    # One adjacent last digit in the committed 8e-format c_r field is exactly
    # the registered 1e-9 artifact-precision boundary. Binary parsing may
    # represent that decimal subtraction a few e-17 above 1e-9.
    boundary = tmp_path / "boundary.csv"
    text = reference.read_text(encoding="utf-8")
    boundary.write_text(
        text.replace("4.31142866e-01", "4.31142865e-01", 1),
        encoding="utf-8",
    )
    boundary_check = driver.verify_grid_against_committed(
        boundary, reference=reference)
    assert boundary_check["ok"]
    assert boundary_check["byte_equal"] is False
