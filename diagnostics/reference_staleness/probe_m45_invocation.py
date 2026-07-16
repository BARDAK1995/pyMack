"""Bounded R=300 probe of the two M45 driver invocation paths.

Writes scratch evidence only. This does not invoke the verdict writer and never
opens a committed reference for writing.
"""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYMACK_NO_BANNER"] = "1"

import datetime as dt
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verification"))
import compute_mack_fig10_6 as engine  # noqa: E402


KEYS = ("R", "omega_i_max", "alpha_peak", "c_r", "c_i")


def historic_row(revision_path: str) -> dict:
    raw = subprocess.check_output(["git", "show", revision_path], cwd=ROOT, text=True)
    return json.loads(raw)[0]


def compare(row: dict, target: dict) -> dict:
    delta = {key: abs(float(row[key]) - float(target[key])) for key in KEYS}
    return {"component_abs_drift": delta, "max_abs_drift": max(delta.values())}


def main() -> int:
    committed = json.loads(
        (ROOT / "verification/second_mode/mack_fig10_6_M45/pymack_curve.json").read_text()
    )[0]
    june17_default = historic_row(
        "ecdcbda5:verification/growthRate_verification/mack_fig10_6_M45/pymack_curve.json"
    )
    june18_corrected = historic_row(
        "cfa4ffcd:verification/growthRate_verification/mack_fig10_6_M45/pymack_curve.json"
    )

    started = time.perf_counter()
    default_row = engine.compute_curve(
        4.5, r_list=[300.0], N=engine.N_DEFAULT,
        y_max=engine.Y_MAX_DEFAULT, verbose=False,
    )[0]
    default_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    corrected_row = engine.compute_curve(
        4.5, r_list=[300.0], N=engine._N_for(4.5),
        y_max=engine._ymax_for(4.5), verbose=False,
    )[0]
    corrected_elapsed = time.perf_counter() - started

    output = {
        "case": "mack_fig10_6_M45_R300_invocation_probe",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "driver_constants": {
            "N_DEFAULT": engine.N_DEFAULT,
            "Y_MAX_DEFAULT": engine.Y_MAX_DEFAULT,
            "N_for_M45": engine._N_for(4.5),
            "ymax_for_M45": engine._ymax_for(4.5),
            "alpha_scan_M45": list(engine.ALPHA_SCAN[4.5]),
        },
        "default_invocation": {
            "call": "compute_curve(4.5, r_list=[300], N=N_DEFAULT, y_max=Y_MAX_DEFAULT)",
            "elapsed_s": default_elapsed,
            "row": default_row,
            "vs_current_committed": compare(default_row, committed),
            "vs_june17_default": compare(default_row, june17_default),
            "vs_june18_corrected": compare(default_row, june18_corrected),
        },
        "corrected_invocation": {
            "call": "compute_curve(4.5, r_list=[300], N=_N_for(4.5), y_max=_ymax_for(4.5))",
            "elapsed_s": corrected_elapsed,
            "row": corrected_row,
            "vs_current_committed": compare(corrected_row, committed),
            "vs_june17_default": compare(corrected_row, june17_default),
            "vs_june18_corrected": compare(corrected_row, june18_corrected),
        },
        "historic_rows": {
            "current_committed_july2": committed,
            "june17_default": june17_default,
            "june18_corrected": june18_corrected,
        },
        "measurement_lock": {
            "used": False,
            "reason": "two bounded one-station probes expected and observed below two minutes each",
        },
    }
    path = Path(__file__).resolve().parent / "SCRATCH/m45_invocation_probe.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(path.relative_to(ROOT)).replace("\\", "/"),
        "default_elapsed_s": default_elapsed,
        "default_vs_committed_max_abs": output["default_invocation"]["vs_current_committed"]["max_abs_drift"],
        "corrected_elapsed_s": corrected_elapsed,
        "corrected_vs_june18_max_abs": output["corrected_invocation"]["vs_june18_corrected"]["max_abs_drift"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
