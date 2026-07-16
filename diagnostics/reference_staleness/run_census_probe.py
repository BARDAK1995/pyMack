"""Scratch-only regeneration probes for the reference-staleness census.

This script mirrors the committed validation drivers without writing anywhere
under verification/. Run one case at a time so an unexpectedly long case has a
clean external wall. Outputs are written below this diagnostic directory's
SCRATCH/ tree.
"""
from __future__ import annotations

import os

# Pin before NumPy/SciPy import, matching the verification drivers' intent.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYMACK_NO_BANNER"] = "1"

import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRATCH = Path(__file__).resolve().parent / "SCRATCH"


def _read_verdict(relative_path: str) -> tuple[Path, dict]:
    path = ROOT / relative_path
    return path, json.loads(path.read_text(encoding="utf-8"))


def _fixture(factory):
    """Call a pytest fixture factory outside pytest without changing its code."""
    return factory.__wrapped__()


def _numeric_classification(committed: list[float], regenerated: list[float]) -> dict:
    committed = [float(v) for v in committed]
    regenerated = [float(v) for v in regenerated]
    c_bytes = json.dumps(committed, separators=(",", ":"), allow_nan=False).encode()
    r_bytes = json.dumps(regenerated, separators=(",", ":"), allow_nan=False).encode()
    delta = np.abs(np.asarray(regenerated) - np.asarray(committed))
    scale = np.maximum(np.abs(np.asarray(committed)), np.finfo(float).tiny)
    byte_identical = c_bytes == r_bytes
    max_abs = float(np.max(delta)) if delta.size else 0.0
    if byte_identical:
        verdict = "byte-identical"
    elif max_abs <= 1.0e-9:
        verdict = "numeric-within-1e-9"
    else:
        verdict = "drifted"
    return {
        "verdict": verdict,
        "canonical_numeric_payload_byte_identical": byte_identical,
        "max_abs_drift": max_abs,
        "max_rel_component_drift": float(np.max(delta / scale)) if delta.size else 0.0,
        "component_abs_drift": [float(v) for v in delta],
    }


def probe_orszag_spectrum() -> dict:
    from validation.test_orszag_full_spectrum import ORSZAG_TABLE5, poiseuille_spectrum

    path, verdict = _read_verdict("verification/other/orszag_spectrum/verdict.json")
    spec = poiseuille_spectrum(1.0, 10000.0, N=128)
    errs = np.asarray([np.min(np.abs(spec - c)) for c in ORSZAG_TABLE5])
    top = complex(spec[np.argmax(spec.imag)])
    committed = [
        verdict["metrics"]["max_abs_err"],
        verdict["metrics"]["median_abs_err"],
        *verdict["metrics"]["least_stable_c"],
    ]
    regenerated = [float(errs.max()), float(np.median(errs)), top.real, top.imag]
    return {
        "case": "orszag_spectrum",
        "driver": "validation/test_orszag_full_spectrum.py::poiseuille_spectrum",
        "committed_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "quantity": ["max_abs_err", "median_abs_err", "least_stable_c.real", "least_stable_c.imag"],
        "committed": committed,
        "regenerated": regenerated,
        "extra": {
            "n_modes_matched_under_1e5": int(np.sum(errs < 1.0e-5)),
            "n_modes_total": int(errs.size),
        },
        "comparison": _numeric_classification(committed, regenerated),
        "note": "The committed least_stable_c is rounded to the paper's 8 digits; comparison also includes regenerated error metrics.",
    }


def probe_malik_case3() -> dict:
    import validation.test_malik1990_case3_anchor as d
    from pymack.solver import solve_temporal_compressible_3d

    path, verdict = _read_verdict("verification/first_mode/malik_case3/verdict.json")
    profile = _fixture(d.malik_case3_profile)
    values, _, _ = solve_temporal_compressible_3d(
        profile, d.ALPHA, d.BETA, d.R_L, d.MA, d.PR, d.GAMMA,
        N=200, y_max=55.0, wall_bc="isothermal",
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    target = d.MALIK_OMEGA / d.ALPHA
    band = values[(values.real > 0.55) & (values.real < 0.68) & (np.abs(values.imag) < 0.05)]
    mode = complex(band[np.argmin(np.abs(band - target))])
    omega = d.ALPHA * mode
    committed = verdict["metrics"]["pymack_omega"]
    regenerated = [omega.real, omega.imag]
    return _anchor_result("malik_case3", d.__file__, path, "omega", committed, regenerated)


def probe_malik_case4() -> dict:
    import validation.test_malik1990_case4_anchor as d
    from pymack.solver import solve_temporal_compressible

    path, verdict = _read_verdict("verification/second_mode/malik_case4/verdict.json")
    profile = _fixture(d.malik_case4_profile)
    values, _, _ = solve_temporal_compressible(
        profile, d.ALPHA, d.R_L, d.MA, d.PR, d.GAMMA,
        N=200, y_max=75.0, wall_bc="isothermal",
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    target = d.MALIK_OMEGA / d.ALPHA
    band = values[(values.real > 0.85) & (values.real < 0.98) & (np.abs(values.imag) < 0.05)]
    mode = complex(band[np.argmin(np.abs(band - target))])
    omega = d.ALPHA * mode
    committed = verdict["metrics"]["pymack_omega"]
    regenerated = [omega.real, omega.imag]
    return _anchor_result("malik_case4", d.__file__, path, "omega", committed, regenerated)


def probe_malik_case5() -> dict:
    import validation.test_malik1990_case5_anchor as d
    from pymack.solver import solve_temporal_compressible

    path, verdict = _read_verdict("verification/second_mode/malik_case5/verdict.json")
    profile = _fixture(d.malik_case5_profile)
    values, _, _ = solve_temporal_compressible(
        profile, d.ALPHA, d.R_L, d.MA, d.PR, d.GAMMA,
        N=280, y_max=140.0, wall_bc="isothermal",
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    target = d.MALIK_OMEGA / d.ALPHA
    band = values[(values.real > 0.90) & (values.real < 1.0) & (np.abs(values.imag) < 0.03)]
    mode = complex(band[np.argmin(np.abs(band - target))])
    omega = d.ALPHA * mode
    committed = verdict["metrics"]["pymack_omega"]
    regenerated = [omega.real, omega.imag]
    return _anchor_result("malik_case5", d.__file__, path, "omega", committed, regenerated)


def probe_malik_case6() -> dict:
    import validation.test_malik1990_case6_anchor as d
    from pymack.solver import solve_spatial

    path, verdict = _read_verdict("verification/second_mode/malik_case6/verdict.json")
    profile = _fixture(d.malik_profile)
    values, _, _ = solve_spatial(
        profile, d.OMEGA_L, d.R_L, d.MA, d.PR, d.GAMMA,
        N=120, y_max=40.0, wall_bc="isothermal",
        target_alpha=d.OMEGA_L / 0.9076, n_modes=10,
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    phase = d.OMEGA_L / values.real
    band = values[(phase > 0.85) & (phase < 0.97) & (np.abs(values.imag) < 0.05)]
    alpha = complex(band[np.argmin(np.abs(band - d.MALIK_ALPHA))])
    committed = verdict["metrics"]["pymack_alpha"]
    regenerated = [alpha.real, alpha.imag]
    return _anchor_result("malik_case6", d.__file__, path, "alpha", committed, regenerated)


def probe_malik_tablex() -> dict:
    import validation.test_malik1990_tableX_anchor as d
    from pymack.solver import solve_spatial

    path, verdict = _read_verdict("verification/second_mode/malik_tableX/verdict.json")
    profile = _fixture(d.malik_tableX_profile)
    values, _, _ = solve_spatial(
        profile, d.OMEGA_L, d.R_L, d.MA, d.PR, d.GAMMA,
        N=200, y_max=100.0, wall_bc="isothermal",
        target_alpha=d.OMEGA_L / 0.938, n_modes=12,
        length_scale="L_star", lambda_mu_ratio=1.2,
    )
    phase = d.OMEGA_L / values.real
    band = values[(phase > 0.88) & (phase < 0.98) & (np.abs(values.imag) < 0.02)]
    alpha = complex(band[np.argmin(np.abs(band - d.MALIK_ALPHA))])
    committed = verdict["metrics"]["pymack_alpha"]
    regenerated = [alpha.real, alpha.imag]
    return _anchor_result("malik_tableX", d.__file__, path, "alpha", committed, regenerated)


def _anchor_result(case, driver_file, committed_path, quantity, committed, regenerated):
    return {
        "case": case,
        "driver": str(Path(driver_file).resolve().relative_to(ROOT)).replace("\\", "/"),
        "committed_path": str(committed_path.relative_to(ROOT)).replace("\\", "/"),
        "quantity": [f"{quantity}.real", f"{quantity}.imag"],
        "committed": [float(v) for v in committed],
        "regenerated": [float(v) for v in regenerated],
        "comparison": _numeric_classification(committed, regenerated),
    }


PROBES = {
    "orszag_spectrum": probe_orszag_spectrum,
    "malik_case3": probe_malik_case3,
    "malik_case4": probe_malik_case4,
    "malik_case5": probe_malik_case5,
    "malik_case6": probe_malik_case6,
    "malik_tableX": probe_malik_tablex,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(PROBES))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    result = PROBES[args.case]()
    result["elapsed_s"] = time.perf_counter() - started
    result["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_thread_env": {name: os.environ[name] for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        )},
    }
    result["committed_sha256"] = hashlib.sha256(
        (ROOT / result["committed_path"]).read_bytes()
    ).hexdigest()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    output = SCRATCH / f"{args.case}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "case": result["case"],
        "elapsed_s": result["elapsed_s"],
        "verdict": result["comparison"]["verdict"],
        "max_abs_drift": result["comparison"]["max_abs_drift"],
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
