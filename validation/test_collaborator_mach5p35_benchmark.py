"""Layer-5 validation gate: pyMack vs the independent Mach 5.35 LST benchmark.

Compares a committed pyMack production neutral-curve sweep (dimensional, run at
the benchmark's recorded conditions) against the external collaborator
Mach 5.35 nitrogen flat-plate second-mode neutral curve in
``reference_data/collaborator_mach5p35/``.

Validated (gated) regions, from the June 2026 comparison study:

- **Upper neutral branch, 200-600 kHz**: agrees to a few mm (MAE ~3 mm over
  branch locations spanning ~20-250 mm).
- **Lower neutral branch, 330-600 kHz**: agrees to ~1-3 mm.

Known, documented difference (NOT gated): below ~330 kHz the benchmark's lower
branch stays near R ~ 520-860 while pyMack's clean second-mode-window sweep
places onset further downstream. Widening the phase-speed tracking window does
not reconcile it (it degrades tracking instead); the leading explanation is a
mode-family / envelope-definition difference in the region where first- and
second-mode bands interact. Tracked as an open investigation in
``docs/VALIDATION_STRATEGY.md``.

This test is CI-safe: it only reads committed CSV artifacts (no eigenvalue
solves). The pyMack artifact provenance (conditions, tracking window, grid) is
recorded in ``validation/data/collaborator_mach5p35/run_manifest.json``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCHMARK_CSV = (
    REPO / "reference_data" / "collaborator_mach5p35" / "LST_neutral_curve_M5p35.csv"
)
DATA_DIR = REPO / "validation" / "data" / "collaborator_mach5p35"
PYMACK_CSV = DATA_DIR / "pymack_neutral_envelope_dimensional.csv"
MANIFEST = DATA_DIR / "run_manifest.json"


def _read_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = math.nan
            rows.append(out)
    return rows


def _finite_branch(rows, x_key):
    pairs = [
        (r["frequency_khz"], r[x_key])
        for r in rows
        if math.isfinite(r.get("frequency_khz", math.nan))
        and math.isfinite(r.get(x_key, math.nan))
    ]
    pairs.sort()
    return (
        np.array([p[0] for p in pairs]),
        np.array([p[1] for p in pairs]),
    )


def _branch_abs_errors(py_rows, bench_rows, py_key, bench_key, f_lo, f_hi):
    bench_f, bench_x = _finite_branch(bench_rows, bench_key)
    errors = []
    for row in py_rows:
        f = row.get("frequency_khz", math.nan)
        x = row.get(py_key, math.nan)
        if not (math.isfinite(f) and math.isfinite(x)):
            continue
        if not (f_lo <= f <= f_hi):
            continue
        errors.append(abs(x - float(np.interp(f, bench_f, bench_x))))
    return np.array(errors, dtype=float)


def test_artifacts_present_and_provenanced():
    assert BENCHMARK_CSV.exists()
    assert PYMACK_CSV.exists()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    text = json.dumps(manifest)
    # The committed sweep must be the matched-condition Mach 5.35 nitrogen case.
    assert "5.35" in text
    assert "nitrogen" in text.lower()
    assert manifest.get("single_sweep", True) in (True, "true", 1)


def test_upper_branch_matches_benchmark_200_to_600_khz():
    py_rows = _read_csv(PYMACK_CSV)
    bench_rows = _read_csv(BENCHMARK_CSV)
    err = _branch_abs_errors(
        py_rows, bench_rows, "upper_neutral_x_mm", "x_right_mm", 200.0, 600.0
    )
    assert err.size >= 40, "expected dense frequency coverage of the upper branch"
    assert float(np.mean(err)) < 5.0, f"upper-branch MAE {np.mean(err):.2f} mm >= 5 mm"
    assert float(np.max(err)) < 10.0, f"upper-branch max {np.max(err):.2f} mm >= 10 mm"


def test_lower_branch_matches_benchmark_330_to_600_khz():
    py_rows = _read_csv(PYMACK_CSV)
    bench_rows = _read_csv(BENCHMARK_CSV)
    err = _branch_abs_errors(
        py_rows, bench_rows, "lower_neutral_x_mm", "x_left_mm", 330.0, 600.0
    )
    assert err.size >= 25, "expected dense frequency coverage of the lower branch"
    assert float(np.mean(err)) < 3.0, f"lower-branch MAE {np.mean(err):.2f} mm >= 3 mm"
    assert float(np.max(err)) < 6.0, f"lower-branch max {np.max(err):.2f} mm >= 6 mm"
