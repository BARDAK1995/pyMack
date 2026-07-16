"""Render the complete provenance-census master table from its JSON records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.verification.provenance_census import CASE_REFERENCES  # noqa: E402


ARTIFACT_ROOT = REPO / "verification" / "provenance_census"
DEFAULT_OUTPUT = REPO / "verification" / "PROVENANCE_CENSUS.md"


def _load_records() -> list[dict[str, Any]]:
    expected = set(CASE_REFERENCES)
    paths = {path.stem: path for path in ARTIFACT_ROOT.glob("*.json")}
    missing = sorted(expected - set(paths))
    unexpected = sorted(set(paths) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"census coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    records = []
    for case_id in expected:
        record = json.loads(paths[case_id].read_text(encoding="utf-8"))
        if record.get("case_id") != case_id:
            raise RuntimeError(f"case-id mismatch in {paths[case_id]}")
        record["_artifact_name"] = paths[case_id].name
        records.append(record)
    return records


def _is_deferred(record: dict[str, Any]) -> bool:
    return record["status"].startswith("DEFERRED")


def _is_drifted(record: dict[str, Any]) -> bool:
    return record.get("comparison", {}).get("verdict") == "DRIFTED"


def _is_repaired(record: dict[str, Any]) -> bool:
    return record["status"] == "REPAIRED-REGENERATED"


def _fmt_number(value: Any) -> str:
    if value is None:
        return "not measured"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.12g}"
    return f"{value:.12f}".rstrip("0").rstrip(".")


def _committed_when(record: dict[str, Any]) -> str:
    committed = record["committed_reference"]["committed_when"]
    return f"{committed['authored_at']} (`{committed['commit'][:7]}`)"


def _config(record: dict[str, Any]) -> str:
    status = record["status"]
    if status == "REPAIRED-REGENERATED":
        effective = record["repair"]["provenance"]["effective_parameters"]
        alpha = effective["alpha_scan"]
        n_alpha = round((alpha[1] - alpha[0]) / alpha[2]) + 1
        return "; ".join(
            [
                "reference repair",
                f"M={effective['mach']}",
                f"N={effective['N']}",
                f"ymax={_fmt_number(effective['y_max'])}",
                f"{len(effective['R_sweep'])} R x {n_alpha} alpha",
                str(effective["sweep_backend"]),
                "BLAS=1",
            ]
        )
    effective = record.get("provenance", {}).get("effective_parameters", {})
    driver = effective.get("driver_effective_parameters", {})
    if driver:
        surface = str(driver.get("execution_surface", "scratch rerun")).replace(
            "_", "-"
        )
        tokens = []
        if driver.get("mach") is not None:
            tokens.append(f"M={driver['mach']}")
        if driver.get("N") is not None:
            tokens.append(f"N={driver['N']}")
        elif driver.get("first_mode_N") is not None:
            second = driver.get("second_mode_N")
            tokens.append(
                f"N={driver['first_mode_N']}"
                + (f"/{second}" if second is not None else "")
            )
        if driver.get("y_max") is not None:
            tokens.append(f"ymax={_fmt_number(driver['y_max'])}")
        if driver.get("n_grid_nodes") is not None:
            tokens.append(f"{driver['n_grid_nodes']} nodes")
        elif driver.get("n_spatial_solves") is not None:
            tokens.append(f"{driver['n_spatial_solves']} solves")
        elif driver.get("R_sweep") is not None:
            alpha = driver.get("alpha_scan") or driver.get("alpha_grid") or []
            n_alpha = (
                round((alpha[1] - alpha[0]) / alpha[2]) + 1
                if len(alpha) == 3 and alpha[2] > 0
                else len(alpha)
            )
            tokens.append(f"{len(driver['R_sweep'])} R x {n_alpha} alpha")
        tokens.append(surface)
        workers = driver.get("workers_effective")
        if workers is None and isinstance(driver.get("scheduler"), dict):
            workers = driver["scheduler"].get("workers_effective")
        if workers is not None:
            tokens.append(f"{workers} workers")
        elif driver.get("station_workers_effective") is not None:
            tokens.append(
                f"{driver['station_workers_effective']} station worker"
            )
        tokens.append(f"BLAS={effective.get('blas_threads', 1)}")
        return "; ".join(tokens)
    if status.startswith("PRE_CLEARED"):
        return f"carried 2026-07-12; `{effective['driver']}`"
    if status == "INHERITED_FRESH_REGENERATION":
        tokens = ["fresh inherited run"]
        for key, label in (("mach", "M"), ("N", "N")):
            if effective.get(key) is not None:
                tokens.append(f"{label}={effective[key]}")
        if effective.get("y_max") is not None:
            tokens.append(f"ymax={_fmt_number(effective['y_max'])}")
        if effective.get("nodes") is not None:
            tokens.append(f"{effective['nodes']} nodes")
        if effective.get("R_sweep") and effective.get("alpha_scan"):
            alpha = effective["alpha_scan"]
            n_alpha = round((alpha[1] - alpha[0]) / alpha[2]) + 1
            tokens.append(f"{len(effective['R_sweep'])} R x {n_alpha} alpha")
        if effective.get("backend") or effective.get("sweep_backend"):
            tokens.append(str(effective.get("backend") or effective["sweep_backend"]))
        if effective.get("workers") is not None:
            tokens.append(f"{effective['workers']} workers")
        if effective.get("blas_threads") is not None:
            tokens.append(f"BLAS={effective['blas_threads']}")
        return "; ".join(tokens)
    if status == "REGENERATED_POINTER":
        return f"pointer-only finalizer; `{effective['finalizer']}`"
    if status == "DEFERRED_COST":
        return (
            "not rerun; exact 8-station production sweep; "
            f"historical wall={effective['historical_runtime_s']} s"
        )
    if status == "DEFERRED":
        return "not runnable; required dimensional total temperature unavailable"
    rerun = record.get("regeneration", {}).get("rerun_in_this_census")
    if rerun:
        return f"scratch rerun; BLAS={effective.get('blas_threads', 1)}; exact command in JSON"
    return "carried provenance; exact configuration in JSON"


def _verdict(record: dict[str, Any]) -> str:
    comparison = record.get("comparison", {})
    result = comparison.get("verdict", record["status"])
    committed = comparison.get("committed_verdict")
    regenerated = comparison.get("regenerated_verdict")
    if result == "REPAIRED-REGENERATED":
        return f"REPAIRED-REGENERATED; {committed} -> {regenerated}"
    if result == "DRIFTED":
        if committed or regenerated:
            return f"DRIFTED; {committed or '?'} -> {regenerated or '?'}"
        return "DRIFTED"
    if result == "DEFERRED":
        return record["status"]
    if regenerated:
        return f"{regenerated} ({result})"
    return str(result)


def _drift(record: dict[str, Any]) -> str:
    comparison = record.get("comparison", {})
    if comparison.get("verdict") == "REPAIRED-REGENERATED":
        return (
            f"was {_fmt_number(comparison['max_abs_drift'])} at "
            f"`{comparison['max_abs_drift_path']}`; replay 0"
        )
    value = comparison.get("max_abs_drift")
    if value is None:
        if comparison.get("verdict") == "byte-identical":
            return "0 (byte-identical)"
        return "not measured"
    suffix = ""
    if comparison.get("max_abs_drift_path"):
        suffix = f" at `{comparison['max_abs_drift_path']}`"
    if comparison.get("verdict") == "byte-identical":
        suffix = " (byte-identical)"
    elif comparison.get("verdict") == "numeric-within-1e-9":
        suffix += " (within 1e-9)"
    return _fmt_number(value) + suffix


def _wall(record: dict[str, Any]) -> str:
    if record["status"] == "REPAIRED-REGENERATED":
        return f"{record['repair']['regeneration']['elapsed_s']:.3f} s"
    regeneration = record.get("regeneration", {})
    elapsed = regeneration.get("elapsed_s")
    if elapsed is None:
        return "not run"
    text = f"{float(elapsed):.3f} s"
    if regeneration.get("timed_out"):
        text += f" / {regeneration['hard_timeout_s']:.0f} s timeout"
    return text


def _deferred_reason(record: dict[str, Any]) -> str:
    regeneration = record.get("regeneration", {})
    comparison = record.get("comparison", {})
    return regeneration.get("reason") or comparison.get("note") or record["status"]


def render(records: list[dict[str, Any]]) -> str:
    drifted = sorted(filter(_is_drifted, records), key=lambda row: row["case_id"])
    repaired = sorted(filter(_is_repaired, records), key=lambda row: row["case_id"])
    deferred = sorted(filter(_is_deferred, records), key=lambda row: row["case_id"])
    remaining = sorted(
        (
            row for row in records
            if not _is_drifted(row) and not _is_repaired(row) and not _is_deferred(row)
        ),
        key=lambda row: row["case_id"],
    )
    ordered = drifted + repaired + deferred + remaining
    lines = [
        "# Verification Provenance Census",
        "",
        (
            f"Coverage: **{len(records)}/{len(CASE_REFERENCES)} cases**. "
            f"Outcome: **{len(drifted)} DRIFTED**, **{len(repaired)} REPAIRED**, "
            f"**{len(deferred)} DEFERRED**, and **{len(remaining)} other "
            "non-drifted records**. Census regeneration was scratch-only; the "
            "user-ratified M58 repair later replaced its mislabeled committed "
            "record while preserving the archived record and DRIFTED history."
        ),
        "",
        f"## DRIFTED ({len(drifted)})",
        "",
    ]
    for record in drifted:
        pointer = f"provenance_census/{record['_artifact_name']}"
        lines.append(
            f"- [`{record['case_id']}`]({pointer}): {_verdict(record)}; "
            f"maximum absolute drift {_drift(record)}."
        )
    lines.extend(["", f"## REPAIRED ({len(repaired)})", ""])
    for record in repaired:
        pointer = f"provenance_census/{record['_artifact_name']}"
        lines.append(
            f"- [`{record['case_id']}`]({pointer}): {_verdict(record)}; "
            f"historical drift and current replay: {_drift(record)}."
        )
    lines.extend(["", f"## DEFERRED ({len(deferred)})", ""])
    for record in deferred:
        pointer = f"provenance_census/{record['_artifact_name']}"
        lines.append(
            f"- [`{record['case_id']}`]({pointer}): **{record['status']}** - "
            f"{_deferred_reason(record)}"
        )
    lines.extend(
        [
            "",
            "## Master table",
            "",
            "| Case | Committed when | Regeneration config | Verdict | Drift | Wall | Provenance |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for record in ordered:
        pointer = f"provenance_census/{record['_artifact_name']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record['case_id']}`",
                    _committed_when(record),
                    _config(record),
                    _verdict(record),
                    _drift(record),
                    _wall(record),
                    f"[census JSON]({pointer})",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Each census JSON is the authoritative provenance pointer: it records the "
            "committed reference and commit time, exact command, effective parameters, "
            "runtime environment, hashes, comparison, wall, timeout, and lock evidence "
            "applicable to that case.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = render(_load_records())
    output = args.output if args.output.is_absolute() else REPO / args.output
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != text:
            print(f"stale or missing: {output.relative_to(REPO)}")
            return 1
        print(f"current: {output.relative_to(REPO)}")
        return 0
    output.write_text(text, encoding="utf-8")
    print(output.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
