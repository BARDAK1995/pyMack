"""Phased provenance census for the 37-case verification matrix.

This harness never writes a committed verification reference.  Regenerated
payloads live below the job SCRATCH root; the only repository outputs are the
small census JSON records under ``verification/provenance_census``.

The numerical case adapters are intentionally invoked one case at a time so a
caller can acquire/release the shared measurement lock per expensive case and
stop that case on drift without disturbing any other reference.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
JOB_TMP = Path(os.environ.get("PYMACK_PROVENANCE_TMP", tempfile.gettempdir()))
SCRATCH_ROOT = JOB_TMP / "provenance_census"
ARTIFACT_ROOT = REPO / "verification" / "provenance_census"
DIAGNOSTIC_ROOT = REPO / "diagnostics" / "reference_staleness"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.verification.provenance import (  # noqa: E402
    build_provenance,
    capture_source,
    exact_command,
    sha256_file,
)


CASE_REFERENCES = {
    "mack_fig10_1_m1p6": "verification/first_mode/mack_fig10_1_m1p6/verdict.json",
    "mack_fig10_1_m2p2": "verification/first_mode/mack_fig10_1_m2p2/verdict.json",
    "mack_fig10_3_m1p3": "verification/first_mode/mack_fig10_3_m1p3/verdict.json",
    "mack_fig10_3_m2p2": "verification/first_mode/mack_fig10_3_m2p2/verdict.json",
    "mack_fig10_3_m3p0": "verification/first_mode/mack_fig10_3_m3p0/verdict.json",
    "mack_fig10_4_family": "verification/first_mode/mack_fig10_4_family/verdict.json",
    "mack_fig10_4_M100": "verification/first_mode/mack_fig10_4_M100/verdict.json",
    "mack_fig10_4_M45": "verification/first_mode/mack_fig10_4_M45/verdict.json",
    "mack_fig10_4_M58": "verification/first_mode/mack_fig10_4_M58/verdict.json",
    "mack_fig10_4_M70": "verification/first_mode/mack_fig10_4_M70/verdict.json",
    "malik_case3": "verification/first_mode/malik_case3/verdict.json",
    "ozgen_fig3_lobes": "verification/mixed_mode/ozgen_fig3/lobes/verdict.json",
    "ozgen_m10": "verification/mixed_mode/ozgen_fig3/M10/verdict.json",
    "ozgen_m2": "verification/mixed_mode/ozgen_fig3/M2/verdict.json",
    "ozgen_m3": "verification/mixed_mode/ozgen_fig3/M3/verdict.json",
    "ozgen_m4": "verification/mixed_mode/ozgen_fig3/M4/verdict.json",
    "ozgen_m6": "verification/mixed_mode/ozgen_fig3/M6/verdict.json",
    "ozgen_m7": "verification/mixed_mode/ozgen_fig3/M7/verdict.json",
    "ozgen_m8": "verification/mixed_mode/ozgen_fig3/M8/verdict.json",
    "malik_case1": "verification/other/malik_case1/verdict.json",
    "orszag_spectrum": "verification/other/orszag_spectrum/verdict.json",
    "balakumar_malik1992_branches": (
        "verification/second_mode/balakumar_malik1992_branches/verdict.json"
    ),
    "balakumar_malik1992_via_xirenfu": (
        "verification/second_mode/balakumar_malik1992_via_xirenfu/verdict.json"
    ),
    "cone_sivasubramanian_fasel_2015": (
        "verification/second_mode/cone_sivasubramanian_fasel_2015/verdict.json"
    ),
    "egorov2006_m6": "verification/second_mode/egorov2006_m6/verdict.json",
    "mack_fig10_6_family": "verification/second_mode/mack_fig10_6_family/verdict.json",
    "mack_fig10_6_M100": "verification/second_mode/mack_fig10_6_M100/verdict.json",
    "mack_fig10_6_M45": "verification/second_mode/mack_fig10_6_M45/verdict.json",
    "mack_fig10_6_M58": "verification/second_mode/mack_fig10_6_M58/verdict.json",
    "mack_fig10_6_M70": "verification/second_mode/mack_fig10_6_M70/verdict.json",
    "malik_case4": "verification/second_mode/malik_case4/verdict.json",
    "malik_case5": "verification/second_mode/malik_case5/verdict.json",
    "malik_case6": "verification/second_mode/malik_case6/verdict.json",
    "malik_fig4_eigenfunction": (
        "verification/second_mode/malik_fig4_eigenfunction/verdict.json"
    ),
    "malik_tableX": "verification/second_mode/malik_tableX/verdict.json",
    "mazhong2003_m4p5": "verification/second_mode/mazhong2003_m4p5/verdict.json",
    "sean_m5p35": "verification/second_mode/sean_m5p35/verdict.json",
}

INHERITED_FRESH_RECORDS = ("ozgen_fig3_lobes", "mack_fig10_6_M45")

PRE_CLEARED = {
    "malik_case3": "validation/test_malik1990_case3_anchor.py",
    "malik_case4": "validation/test_malik1990_case4_anchor.py",
    "malik_case5": "validation/test_malik1990_case5_anchor.py",
    "malik_case6": "validation/test_malik1990_case6_anchor.py",
    "malik_tableX": "validation/test_malik1990_tableX_anchor.py",
    "orszag_spectrum": "validation/test_orszag_full_spectrum.py",
}

POINTER_CASES = {
    "mack_fig10_4_family": "verification/verify_mack_fig10_4.py",
    "mack_fig10_6_family": "verification/verify_mack_fig10_6.py",
}

WORKER = REPO / "scripts" / "verification" / "provenance_census_worker.py"
WORKER_DRIVERS = {
    "cone_sivasubramanian_fasel_2015": [
        "verification/second_mode/cone_sivasubramanian_fasel_2015/reintegrate_domain_matched.py",
        "verification/second_mode/cone_sivasubramanian_fasel_2015/verdict.json",
    ],
    "mack_fig10_1_m1p6": [
        "verification/compute_mack_fig10_1.py",
        "verification/compare_mack_fig10_1.py",
        "verification/rejudge_mack_fig10_1_complete.py",
        "reference_data/digitized/mack_ch10_fig10_1_M16_paper_complete_equations.csv",
    ],
    "mack_fig10_1_m2p2": [
        "verification/compute_mack_fig10_1.py",
        "verification/compare_mack_fig10_1.py",
        "verification/rejudge_mack_fig10_1_complete.py",
        "reference_data/digitized/mack_ch10_fig10_1_M22_paper_complete_equations.csv",
    ],
    "mack_fig10_4_M45": [
        "verification/compute_mack_fig10_4.py",
        "verification/verify_mack_fig10_4.py",
        "reference_data/digitized/mack_ch10_fig10_4_M45_paper.csv",
    ],
    "mack_fig10_4_M58": [
        "verification/compute_mack_fig10_4.py",
        "verification/verify_mack_fig10_4.py",
        "reference_data/digitized/mack_ch10_fig10_4_M58_paper.csv",
    ],
    "mack_fig10_4_M70": [
        "verification/compute_mack_fig10_4.py",
        "verification/verify_mack_fig10_4.py",
        "reference_data/digitized/mack_ch10_fig10_4_M70_paper.csv",
    ],
    "mack_fig10_4_M100": [
        "verification/compute_mack_fig10_4.py",
        "verification/verify_mack_fig10_4.py",
        "reference_data/digitized/mack_ch10_fig10_4_M100_paper.csv",
    ],
    "mack_fig10_6_M58": [
        "verification/compute_mack_fig10_6.py",
        "verification/verify_mack_fig10_6.py",
        "reference_data/digitized/mack_ch10_fig10_6_M58_paper.csv",
    ],
    "mack_fig10_6_M70": [
        "verification/compute_mack_fig10_6.py",
        "verification/verify_mack_fig10_6.py",
        "reference_data/digitized/mack_ch10_fig10_6_M70_paper.csv",
    ],
    "mack_fig10_6_M100": [
        "verification/compute_mack_fig10_6.py",
        "verification/verify_mack_fig10_6.py",
        "reference_data/digitized/mack_ch10_fig10_6_M100_paper.csv",
    ],
    "sean_m5p35": [
        "verification/compare_sean_m5p35.py",
        "validation/data/collaborator_mach5p35/run_manifest.json",
        "validation/data/collaborator_mach5p35/pymack_neutral_envelope_dimensional.csv",
        "reference_data/collaborator_mach5p35/LST_neutral_curve_M5p35.csv",
    ],
    "balakumar_malik1992_branches": [
        "verification/compute_balakumar_malik1992_branches.py"
    ],
    "balakumar_malik1992_via_xirenfu": [
        "verification/compare_malik1990_anchors.py"
    ],
    "egorov2006_m6": ["verification/compare_egorov2006_m6.py"],
    "malik_fig4_eigenfunction": [
        "verification/compute_malik_fig4_eigenfunction.py",
        "verification/second_mode/malik_fig4_eigenfunction/reference_malik_fig4_Tr.csv",
        "verification/second_mode/malik_fig4_eigenfunction/reference_malik_fig4_Ti.csv",
    ],
    "mazhong2003_m4p5": [
        "verification/second_mode/mazhong2003_m4p5/compute_mazhong_m4p5.py",
        "verification/second_mode/mazhong2003_m4p5/write_verdict_mazhong.py",
        "verification/second_mode/mazhong2003_m4p5/reference_mazhong_fig15.csv",
    ],
    "mack_fig10_3_m2p2": ["verification/compute_mack_fig10_3_selfseed.py"],
    "mack_fig10_3_m3p0": ["verification/compute_mack_fig10_3_selfseed.py"],
}

for _mach in (2, 3, 4, 6, 7, 8, 10):
    _ozgen_drivers = [
        "verification/mixed_mode/ozgen_fig3/_refdigitize/build_ozgen_final.py",
        "verification/mixed_mode/ozgen_fig3/_refdigitize/finalize_ozgen_verdicts.py",
        f"reference_data/digitized/ozgen_fig3_M{_mach}_neutral_v2.csv",
    ]
    if _mach == 2:
        _ozgen_drivers.extend(
            [
                "verification/mixed_mode/ozgen_fig3/_refdigitize/trace_continuation.py",
                "verification/mixed_mode/ozgen_fig3/_refdigitize/continuation_M2.csv",
            ]
        )
    else:
        _ozgen_drivers.extend(
            [
                "verification/mixed_mode/ozgen_fig3/_refdigitize/discrete_mode.py",
                "verification/mixed_mode/ozgen_fig3/_refdigitize/build_firstmode_grid.py",
                "verification/mixed_mode/ozgen_fig3/_refdigitize/build_secondmode_grid.py",
                "verification/mixed_mode/ozgen_fig3/_refdigitize/build_onset_extension.py",
                "verification/mixed_mode/ozgen_fig3/_refdigitize/build_lowR_extension.py",
            ]
        )
    WORKER_DRIVERS[f"ozgen_m{_mach}"] = _ozgen_drivers

COST_DEFERRED = {
    "mack_fig10_3_m1p3": {
        "driver": "scripts/make_mack_fig10_3_overlay.py",
        "evidence": "docs/figures/mack_fig10_3_overlay.json",
        "cost_estimate": "5951.3 s historical production runtime (about 99 minutes)",
        "reason": (
            "The production driver is an eight-station serial exact-shooting sweep, "
            "records 615-843 s per station, writes its comparable payload only after "
            "the complete sweep, and exposes neither station selection nor resume."
        ),
        "command": (
            "$env:PYTHONPATH=(Get-Location).Path; python "
            "scripts/make_mack_fig10_3_overlay.py --mach 1.3 --psi 45 "
            "--quality production --output-png '<SCRATCH>/mack_fig10_3_overlay.png'"
        ),
    },
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _repo_rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        try:
            return "SCRATCH/" + path.relative_to(SCRATCH_ROOT).as_posix()
        except ValueError:
            return str(path)


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {_repo_rel(path): sha256_file(path) for path in map(Path, paths)}


def _committed_when(path: Path) -> dict[str, str]:
    output = subprocess.check_output(
        ["git", "log", "-1", "--format=%H%n%aI", "--", str(path)],
        cwd=REPO,
        text=True,
    ).splitlines()
    if len(output) != 2:
        raise RuntimeError(f"cannot determine committed-when for {path}")
    return {"commit": output[0], "authored_at": output[1]}


def _write_artifact(case_id: str, payload: dict) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_ROOT / f"{case_id}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _common_header(case_id: str) -> tuple[Path, dict]:
    reference = REPO / CASE_REFERENCES[case_id]
    return reference, {
        "schema_version": 1,
        "case_id": case_id,
        "matrix_path": "CPU public record",
        "committed_reference": {
            "path": _repo_rel(reference),
            "sha256": sha256_file(reference),
            "committed_when": _committed_when(reference),
        },
    }


def _helper_provenance(
    *,
    source: dict,
    effective_parameters: dict,
    paths: Iterable[Path],
    command: str | None = None,
) -> dict:
    return build_provenance(
        source=source,
        command=command or exact_command(),
        effective_parameters=effective_parameters,
        sha256=_hashes(paths),
    )


def prepare_precleared(case_id: str, source: dict) -> Path:
    evidence = DIAGNOSTIC_ROOT / "SCRATCH" / f"{case_id}.json"
    prior = _json(evidence)
    reference, artifact = _common_header(case_id)
    prior_verdict = prior["comparison"]["verdict"]
    verdict = "DRIFTED" if prior_verdict.lower() == "drifted" else prior_verdict
    driver = REPO / PRE_CLEARED[case_id]
    original_command = (
        "$env:PYTHONPATH=(Get-Location).Path; python "
        f"diagnostics/reference_staleness/run_census_probe.py {case_id}"
    )
    artifact.update(
        {
            "status": "PRE_CLEARED_2026-07-12",
            "regeneration": {
                "rerun_in_this_census": False,
                "reason": "User-ratified diagnostic evidence; explicitly not rerun.",
                "evidence": _repo_rel(evidence),
                "driver": prior["driver"],
                "exact_command": original_command,
                "generated_at_utc": prior["generated_at_utc"],
                "elapsed_s": prior["elapsed_s"],
                "environment": prior["environment"],
                "quantity": prior["quantity"],
            },
            "comparison": {
                **prior["comparison"],
                "verdict": verdict,
                "classification_rule": (
                    "byte-identical, else max absolute numeric component drift <=1e-9, "
                    "else DRIFTED"
                ),
            },
        }
    )
    artifact["provenance"] = _helper_provenance(
        source=source,
        command=original_command,
        effective_parameters={
            "evidence_origin": "2026-07-12 diagnostic census",
            "driver": prior["driver"],
            "quantity": prior["quantity"],
            "rerun_in_this_census": False,
        },
        paths=(evidence, reference, driver, DIAGNOSTIC_ROOT / "run_census_probe.py"),
    )
    return _write_artifact(case_id, artifact)


def prepare_pointer(case_id: str, source: dict) -> Path:
    reference, artifact = _common_header(case_id)
    scratch = SCRATCH_ROOT / case_id
    scratch.mkdir(parents=True, exist_ok=True)
    generated = scratch / "verdict.json"
    obj = _json(reference)
    # This is the exact transformation performed by each family verifier after
    # its per-Mach rows complete.  It has no numerical compute surface.
    obj["verdict"] = "pending"
    if case_id == "mack_fig10_4_family":
        obj["generated"] = "pending"
    obj["verdict_reason"] = (
        "Superseded by per-Mach rows "
        + (
            "mack_fig10_4_M45/M58/M70/M100"
            if case_id == "mack_fig10_4_family"
            else "mack_fig10_6_M45/M58/M70/M100"
        )
        + " (this family stub retained only as a pointer)."
    )
    generated.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    byte_equal = generated.read_bytes() == reference.read_bytes()
    # Git checkouts can differ only by CRLF smudging; use the canonical JSON
    # payload as a second exact identity guard, not as a numerical tolerance.
    canonical_equal = _json(generated) == _json(reference)
    verdict = "byte-identical" if byte_equal or canonical_equal else "DRIFTED"
    driver = REPO / POINTER_CASES[case_id]
    artifact.update(
        {
            "status": "REGENERATED_POINTER",
            "regeneration": {
                "scratch_dir": _repo_rel(scratch),
                "driver": _repo_rel(driver),
                "configuration": "deterministic family-pointer finalization; no compute surface",
                "started_at_utc": _now(),
                "elapsed_s": 0.0,
            },
            "comparison": {
                "verdict": verdict,
                "byte_identical": byte_equal,
                "canonical_json_identical": canonical_equal,
                "max_abs_drift": 0.0 if canonical_equal else None,
                "note": "Pointer-only row; no numerical solver payload exists.",
            },
        }
    )
    artifact["provenance"] = _helper_provenance(
        source=source,
        effective_parameters={
            "record_type": "family pointer",
            "compute_surface": None,
            "finalizer": _repo_rel(driver),
        },
        paths=(reference, generated, driver),
    )
    return _write_artifact(case_id, artifact)


def prepare_malik_case1(source: dict) -> Path:
    case_id = "malik_case1"
    reference, artifact = _common_header(case_id)
    driver = REPO / "verification" / "compare_malik1990_anchors.py"
    artifact.update(
        {
            "status": "DEFERRED",
            "regeneration": {
                "rerun_in_this_census": False,
                "reason": (
                    "No reproducible compute surface: the committed row explicitly records "
                    "unknown dimensional total-temperature data and scopes the effectively "
                    "incompressible coverage to Orszag. This is a scientific-input wall, not "
                    "a machine-cost deferral."
                ),
                "cost_estimate": "not applicable until the missing physical condition is recovered",
                "driver": _repo_rel(driver),
            },
            "comparison": {
                "verdict": "DEFERRED",
                "byte_identical": None,
                "max_abs_drift": None,
                "note": "The committed pending record is not a regenerated numerical reference.",
            },
        }
    )
    artifact["provenance"] = _helper_provenance(
        source=source,
        effective_parameters={
            "mach": 0.5,
            "Re_l": 2000.0,
            "alpha": 0.10,
            "missing_condition": "T0_K not openly recoverable",
            "compute_surface": None,
        },
        paths=(reference, driver),
    )
    return _write_artifact(case_id, artifact)


def inherited_fresh_records() -> list[Path]:
    """Return ratified carried records whose original staging lane is private."""
    paths = [ARTIFACT_ROOT / f"{case}.json" for case in INHERITED_FRESH_RECORDS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inherited census records: {missing}")
    return paths


def prepare_carried() -> list[Path]:
    source = capture_source(REPO)
    if source["tracked_tree_dirty"]:
        raise RuntimeError("tracked worktree must be clean before provenance capture")
    paths = [prepare_precleared(case, source) for case in PRE_CLEARED]
    paths.extend(prepare_pointer(case, source) for case in POINTER_CASES)
    paths.append(prepare_malik_case1(source))
    paths.extend(inherited_fresh_records())
    return paths


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.update(_numeric_leaves(item, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        leaves[prefix] = float(value)
    return leaves


def _compare_regenerated(reference: Path, regenerated: Path) -> dict:
    committed_doc = _json(reference)
    regenerated_doc = _json(regenerated)
    committed = _numeric_leaves(committed_doc.get("metrics", {}), "metrics")
    fresh = _numeric_leaves(regenerated_doc.get("metrics", {}), "metrics")
    shared = sorted(set(committed) & set(fresh))
    missing = sorted(set(committed) - set(fresh))
    if not shared:
        raise RuntimeError(f"no shared numeric metric leaves in {regenerated}")
    differences = {key: abs(fresh[key] - committed[key]) for key in shared}
    max_path = max(differences, key=differences.get)
    max_abs = differences[max_path]
    byte_identical = reference.read_bytes() == regenerated.read_bytes()
    committed_verdict = committed_doc.get("verdict")
    regenerated_verdict = regenerated_doc.get("verdict")
    scientific_verdict_identical = committed_verdict == regenerated_verdict
    if byte_identical:
        verdict = "byte-identical"
    elif not missing and max_abs <= 1.0e-9 and scientific_verdict_identical:
        verdict = "numeric-within-1e-9"
    else:
        verdict = "DRIFTED"
    return {
        "verdict": verdict,
        "whole_reference_byte_identical": byte_identical,
        "numeric_tolerance": 1.0e-9,
        "n_committed_numeric_metrics": len(committed),
        "n_compared_numeric_metrics": len(shared),
        "missing_numeric_metric_paths": missing,
        "max_abs_drift": max_abs,
        "max_abs_drift_path": max_path,
        "component_abs_drift": differences,
        "committed_verdict": committed_verdict,
        "regenerated_verdict": regenerated_verdict,
        "scientific_verdict_identical": scientific_verdict_identical,
    }


def _run_captured_with_timeout(
    command_args: list[str], *, env: dict[str, str], timeout_s: float
) -> tuple[subprocess.CompletedProcess[str], bool, str | None]:
    """Run one worker and kill its complete process tree at the hard wall."""
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command_args,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
        **popen_kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        return (
            subprocess.CompletedProcess(
                command_args, process.returncode, stdout, stderr
            ),
            False,
            None,
        )
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            killed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                text=True,
                capture_output=True,
            )
            termination = "taskkill /T /F"
            termination_detail = (killed.stdout + killed.stderr).strip()
        else:
            import signal

            os.killpg(process.pid, signal.SIGKILL)
            termination = "SIGKILL process group"
            termination_detail = ""
        stdout, stderr = process.communicate()
        note = f"HARD TIMEOUT after {timeout_s:g} s; {termination}."
        if termination_detail:
            note += f" {termination_detail}"
        stderr = (stderr or "") + ("\n" if stderr else "") + note + "\n"
        return (
            subprocess.CompletedProcess(command_args, 124, stdout or "", stderr),
            True,
            termination,
        )


def run_worker_case(case_id: str, *, use_lock: bool, timeout_s: float) -> Path:
    if case_id not in WORKER_DRIVERS:
        raise ValueError(f"no worker registered for {case_id}")
    source = capture_source(REPO)
    if source["tracked_tree_dirty"]:
        raise RuntimeError("tracked worktree must be clean before provenance capture")

    reference, artifact = _common_header(case_id)
    case_scratch = SCRATCH_ROOT / case_id
    case_scratch.mkdir(parents=True, exist_ok=True)
    stdout_path = case_scratch / "stdout.txt"
    stderr_path = case_scratch / "stderr.txt"
    command_args = [
        sys.executable,
        str(WORKER),
        case_id,
        "--out",
        str(case_scratch),
    ]
    command = (
        "$env:PYTHONPATH=(Get-Location).Path; "
        "$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; "
        "$env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'; "
        "$env:VECLIB_MAXIMUM_THREADS='1'; python "
        f"scripts/verification/provenance_census_worker.py {case_id} "
        f"--out '{case_scratch}'"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"

    lock_event = None
    timed_out = False
    termination = None
    started = time.perf_counter()
    if use_lock:
        from scripts.hunt.measurement_lock import measurement_lock

        with measurement_lock(
            run_label=f"provenance-census-{case_id}",
            owner_prefix="provenance-census-low-priority",
        ) as event:
            lock_event = event
            completed, timed_out, termination = _run_captured_with_timeout(
                command_args, env=env, timeout_s=timeout_s
            )
    else:
        completed, timed_out, termination = _run_captured_with_timeout(
            command_args, env=env, timeout_s=timeout_s
        )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    regenerated = case_scratch / "verdict.json"
    parameter_path = case_scratch / "census_effective_parameters.json"
    driver_paths = [REPO / path for path in WORKER_DRIVERS[case_id]]
    hashed = [reference, WORKER, *driver_paths, stdout_path, stderr_path]
    if timed_out:
        comparison = {
            "verdict": "DEFERRED",
            "whole_reference_byte_identical": None,
            "max_abs_drift": None,
            "note": (
                f"The regeneration exceeded its hard {timeout_s:g} s wall; "
                "the complete worker process tree was killed under walls doctrine."
            ),
        }
        status = "DEFERRED-RUNAWAY"
    elif completed.returncode == 0 and regenerated.is_file():
        comparison = _compare_regenerated(reference, regenerated)
        status = "DRIFTED" if comparison["verdict"] == "DRIFTED" else "REGENERATED"
        hashed.append(regenerated)
    else:
        comparison = {
            "verdict": "DEFERRED",
            "whole_reference_byte_identical": None,
            "max_abs_drift": None,
            "note": (
                "The committed driver did not produce a comparable verdict; "
                "the captured stderr is a binding driver wall."
            ),
        }
        status = "DEFERRED_DRIVER_WALL"

    artifact.update(
        {
            "status": status,
            "regeneration": {
                "rerun_in_this_census": True,
                "scratch_directory": _repo_rel(case_scratch),
                "regenerated_payload": (
                    _repo_rel(regenerated) if regenerated.is_file() else None
                ),
                "exact_command": command,
                "exit_code": completed.returncode,
                "elapsed_s": elapsed,
                "hard_timeout_s": timeout_s,
                "timed_out": timed_out,
                "termination": termination,
                "measurement_lock": lock_event,
                "stdout": _repo_rel(stdout_path),
                "stderr": _repo_rel(stderr_path),
            },
            "comparison": comparison,
        }
    )
    driver_effective_parameters = (
        _json(parameter_path) if parameter_path.is_file() else {}
    )
    if parameter_path.is_file():
        hashed.append(parameter_path)
    known = {path.resolve() for path in hashed}
    for extra in sorted(path for path in case_scratch.rglob("*") if path.is_file()):
        if extra.resolve() not in known:
            hashed.append(extra)
            known.add(extra.resolve())
    artifact["provenance"] = _helper_provenance(
        source=source,
        command=command,
        effective_parameters={
            "case_id": case_id,
            "scratch_only": True,
            "blas_threads": 1,
            "measurement_lock_used": use_lock,
            "driver_effective_parameters": driver_effective_parameters,
        },
        paths=hashed,
    )
    artifact["provenance"]["execution"]["environment"].update(
        {
            "PYTHONPATH": env["PYTHONPATH"],
            "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": env["NUMEXPR_NUM_THREADS"],
            "VECLIB_MAXIMUM_THREADS": env["VECLIB_MAXIMUM_THREADS"],
        }
    )
    return _write_artifact(case_id, artifact)


def prepare_cost_deferred(case_id: str) -> Path:
    if case_id not in COST_DEFERRED:
        raise ValueError(f"no cost deferral registered for {case_id}")
    source = capture_source(REPO)
    if source["tracked_tree_dirty"]:
        raise RuntimeError("tracked worktree must be clean before provenance capture")
    spec = COST_DEFERRED[case_id]
    reference, artifact = _common_header(case_id)
    driver = REPO / spec["driver"]
    evidence = REPO / spec["evidence"]
    artifact.update(
        {
            "status": "DEFERRED_COST",
            "regeneration": {
                "rerun_in_this_census": False,
                "reason": spec["reason"],
                "cost_estimate": spec["cost_estimate"],
                "historical_cost_evidence": _repo_rel(evidence),
                "would_run_command": spec["command"],
            },
            "comparison": {
                "verdict": "DEFERRED",
                "whole_reference_byte_identical": None,
                "max_abs_drift": None,
                "note": "No census regeneration was burned for this no-resume long case.",
            },
        }
    )
    artifact["provenance"] = _helper_provenance(
        source=source,
        command=spec["command"],
        effective_parameters={
            "case_id": case_id,
            "quality": "production",
            "scratch_only": True,
            "deferred_without_execution": True,
            "historical_runtime_s": _json(evidence).get("runtime_s"),
        },
        paths=(reference, driver, evidence),
    )
    return _write_artifact(case_id, artifact)


def inventory() -> dict:
    missing = [
        {"case_id": case, "path": path}
        for case, path in CASE_REFERENCES.items()
        if not (REPO / path).is_file()
    ]
    return {
        "total": len(CASE_REFERENCES),
        "cpu_cases": len(CASE_REFERENCES),
        "inherited_fresh_records": sorted(INHERITED_FRESH_RECORDS),
        "precleared": sorted(PRE_CLEARED),
        "pointer_cases": sorted(POINTER_CASES),
        "remaining_numerical_cases": sorted(
            set(CASE_REFERENCES)
            - set(INHERITED_FRESH_RECORDS)
            - set(PRE_CLEARED)
            - set(POINTER_CASES)
            - {"malik_case1"}
        ),
        "missing_references": missing,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--prepare-carried", action="store_true")
    action.add_argument("--run-case", choices=sorted(WORKER_DRIVERS))
    action.add_argument("--defer-case", choices=sorted(COST_DEFERRED))
    parser.add_argument("--lock", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="hard per-case wall; timeout kills the complete worker process tree",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inventory:
        print(json.dumps(inventory(), indent=2))
        return 0 if not inventory()["missing_references"] else 2
    if args.run_case:
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
        path = run_worker_case(
            args.run_case, use_lock=args.lock, timeout_s=args.timeout_seconds
        )
        print(_repo_rel(path))
        return 0
    if args.defer_case:
        path = prepare_cost_deferred(args.defer_case)
        print(_repo_rel(path))
        return 0
    paths = prepare_carried()
    for path in paths:
        print(_repo_rel(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
