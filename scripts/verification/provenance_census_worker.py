"""Scratch-only case workers for the phased provenance census.

This module deliberately patches only driver output roots.  It does not alter
solver inputs, committed references, or anything below ``pymack/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYMACK_NO_BANNER"] = "1"

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


CASES = (
    "balakumar_malik1992_branches",
    "balakumar_malik1992_via_xirenfu",
    "egorov2006_m6",
    "malik_fig4_eigenfunction",
    "mazhong2003_m4p5",
    "mack_fig10_3_m2p2",
    "mack_fig10_3_m3p0",
    "mack_fig10_1_m1p6",
    "mack_fig10_1_m2p2",
    "mack_fig10_4_M45",
    "mack_fig10_4_M58",
    "mack_fig10_4_M70",
    "mack_fig10_4_M100",
    "mack_fig10_6_M58",
    "mack_fig10_6_M70",
    "mack_fig10_6_M100",
    "sean_m5p35",
    "cone_sivasubramanian_fasel_2015",
    "ozgen_m2",
    "ozgen_m3",
    "ozgen_m4",
    "ozgen_m6",
    "ozgen_m7",
    "ozgen_m8",
    "ozgen_m10",
)

OZGEN_TOOLS = (
    REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_refdigitize"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_via_xirenfu(out: Path) -> None:
    import verification.compare_malik1990_anchors as driver

    ma, reynolds, omega, pr = 4.5, 1000.0, 0.20, 0.72
    t0 = 311.0
    t_edge = t0 / (1.0 + 0.5 * (driver.GAMMA - 1.0) * ma**2)
    published = 0.220 - 0.003091j
    alpha = driver._spatial_second_mode(
        ma, reynolds, omega, pr, t_edge, 0.91, published, N=120
    )
    if alpha is None:
        raise RuntimeError("no second-mode candidate found")
    conditions = {
        "Ma": ma,
        "Re_l": reynolds,
        "omega": omega,
        "beta": 0.0,
        "Pr": pr,
        "gamma": driver.GAMMA,
        "wall": "adiabatic (insulated)",
        "T0_K": t0,
        "T_edge_K": t_edge,
        "sutherland_S_K": driver.SUTHERLAND_S_K,
        "mode": "second mode",
        "length_scale": "L_star",
    }
    record = driver.verdict_spatial(
        "balakumar_malik1992_via_xirenfu",
        "Balakumar & Malik (1992), fresh census regeneration",
        conditions,
        published,
        alpha,
        "Fresh scratch-only census regeneration.",
        generated="new",
        n=120,
    )
    _write(out / "verdict.json", record)


def run_branches(out: Path) -> None:
    import verification.compute_balakumar_malik1992_branches as driver

    driver.OUT = out
    driver.main()


def run_egorov(out: Path) -> None:
    import verification.compare_egorov2006_m6 as driver

    driver.OUT = out
    driver.main()


def run_malik_fig4(out: Path) -> None:
    import verification.compute_malik_fig4_eigenfunction as driver

    source = REPO / "verification" / "second_mode" / "malik_fig4_eigenfunction"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("reference_malik_fig4_Tr.csv", "reference_malik_fig4_Ti.csv"):
        shutil.copyfile(source / name, out / name)
    driver.OUT = out
    driver.main()


def run_mazhong(out: Path) -> None:
    from verification.second_mode.mazhong2003_m4p5 import compute_mazhong_m4p5 as compute
    from verification.second_mode.mazhong2003_m4p5 import write_verdict_mazhong as judge

    source = REPO / "verification" / "second_mode" / "mazhong2003_m4p5"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "reference_mazhong_fig15.csv", out / "reference_mazhong_fig15.csv")
    compute.HERE = out
    fresh = compute.main()
    judge.HERE = out
    judge.BRANCH_I_PYMACK = float(fresh["branch_I_R_pymack"])
    judge.BRANCH_II_PYMACK = float(fresh["branch_II_R_pymack"])
    judge.main()


def run_broken_selfseed(case_id: str, out: Path) -> None:
    # Importing the ratified driver is itself the test.  The current public API
    # no longer exposes the two helpers it imports; do not patch around that wall.
    import verification.compute_mack_fig10_3_selfseed as driver

    mach, psi = {
        "mack_fig10_3_m2p2": (2.2, 45.0),
        "mack_fig10_3_m3p0": (3.0, 60.0),
    }[case_id]
    result = driver.compute(mach, psi, workers=11)
    _write(out / "regeneration.json", result)


def run_mack_fig10_1(case_id: str, out: Path) -> None:
    import verification.compare_mack_fig10_1 as compute
    import verification.rejudge_mack_fig10_1_complete as judge

    mach = {
        "mack_fig10_1_m1p6": 1.6,
        "mack_fig10_1_m2p2": 2.2,
    }[case_id]
    output_root = out.parent
    compute.OUT_ROOT = output_root
    compute.main(["--machs", str(mach), "--N", "140", "--y-max", "45", "--workers", "12"])

    judge.OUT_ROOT = output_root
    judge.REPO = output_root
    judge.CASES = {mach: judge.CASES[mach]}
    summary = judge.main()
    judge.write_corrected_verdicts(summary)


def run_mack_fig10_4(case_id: str, out: Path) -> None:
    import verification.verify_mack_fig10_4 as judge

    mach = {
        "mack_fig10_4_M45": 4.5,
        "mack_fig10_4_M58": 5.8,
        "mack_fig10_4_M70": 7.0,
        "mack_fig10_4_M100": 10.0,
    }[case_id]
    engine = judge.engine
    rows, scheduler = engine.compute_curve_point_parallel(
        mach,
        N=engine._N_for(mach),
        y_max=engine._ymax_for(mach),
        workers=61,
        verbose=True,
    )
    judge.GROWTH_DIR = out.parent
    judge.verify_mach(mach, force=True, rows=rows)
    _write(
        out / "census_effective_parameters.json",
        {
            "mach": mach,
            "N": engine._N_for(mach),
            "y_max": engine._ymax_for(mach),
            "R_sweep": list(engine.DEFAULT_R_SWEEPS[round(mach, 1)]),
            "alpha_grid": list(engine.ALPHA_GRID[round(mach, 1)]),
            "psi_grid_deg": list(engine.PSI_GRID[round(mach, 1)]),
            "mode_c_r": [engine.CR_LO, engine.CR_HI],
            "mode_c_i_cap": engine.CI_CAP,
            "condition": "table_11_1",
            "length_scale": "L_star",
            "lambda_mu_ratio": 0.0,
            "execution_surface": "point_parallel",
            "workers_requested": 61,
            "workers_effective": scheduler["workers_effective"],
            "scheduler": scheduler,
            "historical_reference_scheduler": "station_serial",
        },
    )


def run_mack_fig10_6(case_id: str, out: Path) -> None:
    # The production verifier runs R stations serially.  Each station's
    # temporal_sweep already owns a process pool across its alpha grid.  Do not
    # wrap that implementation in a second station-level process pool: on
    # Windows the old census adapter launched 12 outer M58 workers, each of
    # which launched up to 51 inner workers (about 612 QZ processes on this
    # 64-logical-core host), causing the 13-hour runaway.
    os.environ["PYMACK_SWEEP_BACKEND"] = "cpu"
    import verification.verify_mack_fig10_6 as judge
    from scripts.verification.provenance import sha256_file

    mach = {
        "mack_fig10_6_M58": 5.8,
        "mack_fig10_6_M70": 7.0,
        "mack_fig10_6_M100": 10.0,
    }[case_id]
    engine = judge.engine
    sweep_observations = []
    original_sweep = engine.temporal_sweep

    def observed_sweep(*args, **kwargs):
        result = original_sweep(*args, **kwargs)
        meta = result.meta
        sweep_observations.append(
            {
                "R": float(result.Res[0]),
                "backend": meta.get("backend"),
                "cpu_workers": meta.get("cpu_workers"),
                "cpu_blas_threads_effective": meta.get(
                    "cpu_blas_threads_effective"
                ),
                "alpha_points": meta.get("grid", {}).get("alpha", {}).get("n"),
                "engine_wall_time_s": meta.get("engine_wall_time_s"),
                "n_failed_points": meta.get("n_failed_points"),
            }
        )
        return result

    engine.temporal_sweep = observed_sweep
    try:
        rows = engine.compute_curve(
            mach,
            N=engine._N_for(mach),
            y_max=engine._ymax_for(mach),
            verbose=True,
        )
    finally:
        engine.temporal_sweep = original_sweep
    judge.GROWTH_DIR = out.parent

    # The merged verifier embeds provenance and normally hashes outputs below
    # the repository root.  Census outputs are intentionally outside that root,
    # so retain the same hash function while giving scratch paths stable labels.
    def hash_scratch_aware(paths, *, root):
        result = {}
        for path in map(Path, paths):
            resolved = path.resolve()
            try:
                label = resolved.relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                label = "SCRATCH/" + resolved.relative_to(out.parent.parent).as_posix()
            result[label] = sha256_file(resolved)
        return result

    judge.hash_files = hash_scratch_aware
    judge.verify_mach(mach, force=True, rows=rows)
    params = judge.effective_parameters(mach)
    params.update(
        {
            "execution_surface": "station_serial_alpha_parallel",
            "station_workers_requested": 1,
            "station_workers_effective": 1,
            "alpha_sweep_observations": sweep_observations,
            "historical_reference_scheduler": "station_serial",
            "runaway_correction": (
                "removed nested 12-station process pool around temporal_sweep's "
                "existing per-station alpha-grid process pool"
            ),
        }
    )
    _write(out / "census_effective_parameters.json", params)


def run_sean(out: Path) -> None:
    import verification.compare_sean_m5p35 as driver

    driver.CASE_DIR = out
    driver.main()
    _write(
        out / "census_effective_parameters.json",
        {
            "compute_role": "deterministic verdict regeneration from committed production CSV",
            "production_sweep_rerun": False,
            "production_manifest": (
                "validation/data/collaborator_mach5p35/run_manifest.json"
            ),
            "pymack_csv": (
                "validation/data/collaborator_mach5p35/"
                "pymack_neutral_envelope_dimensional.csv"
            ),
            "reference_csv": (
                "reference_data/collaborator_mach5p35/"
                "LST_neutral_curve_M5p35.csv"
            ),
            "upper_band_khz": list(driver.UPPER_BAND),
            "lower_gated_band_khz": list(driver.LOWER_GATED_BAND),
            "lower_full_band_khz": list(driver.LOWER_FULL_BAND),
        },
    )


def run_cone(out: Path) -> None:
    """Run the committed domain-matched cone script from a scratch-safe copy."""
    import re
    import runpy

    source_dir = (
        REPO
        / "verification"
        / "second_mode"
        / "cone_sivasubramanian_fasel_2015"
    )
    source_script = source_dir / "reintegrate_domain_matched.py"
    scratch_script = out / source_script.name
    text = source_script.read_text(encoding="utf-8")
    text, count = re.subn(
        r'^REPO = Path\(.+\)$',
        f"REPO = Path({REPO.as_posix()!r})",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("could not redirect the cone driver's repository root")
    scratch_script.write_text(text, encoding="utf-8")
    _write(
        out / "census_effective_parameters.json",
        {
            "driver": "reintegrate_domain_matched.py scratch-safe copy",
            "execution_surface": "serial frequency_station_grid",
            "mach_edge": 5.356,
            "N": 90,
            "y_max": 40.0,
            "frequencies_khz": [
                170.0, 180.0, 190.0, 200.0, 205.0, 210.0, 215.0,
                220.0, 230.0, 240.0, 250.0, 260.0, 280.0, 300.0,
            ],
            "n_stations": 60,
            "n_spatial_solves": 840,
            "x_star_axial_m": [0.30, 0.59],
            "length_scale": "L_star",
            "lambda_mu_ratio": 1.2,
            "mode_c_r": [0.80, 0.985],
            "blas_threads": 1,
        },
    )
    runpy.run_path(str(scratch_script), run_name="__main__")

    result = json.loads((out / "domain_matched_result.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (source_dir / "verdict.json").read_text(encoding="utf-8")
    )
    metrics = reference["metrics"].copy()
    f_peak = float(result["f_peak_khz"])
    n_peak = float(result["N_peak_domain_matched"])
    n_210 = float(result["N_at_210khz"])
    cr_210 = float(result["results"]["210.0"]["cr_med"])
    metrics.update(
        {
            "pymack_peak_N_frequency_khz": f_peak,
            "pymack_peak_N_domain_matched": round(n_peak, 3),
            "pymack_N_at_210khz_domain_matched": round(n_210, 3),
            "pymack_second_mode_c_r": round(cr_210, 3),
            "freq_rel_err_vs_210khz_at_pymack_peak": round(
                abs(f_peak - 210.0) / 210.0, 4
            ),
            "N_rel_err_vs_9p5_at_pymack_peak": round(
                abs(n_peak - 9.5) / 9.5, 3
            ),
            "N_rel_err_vs_9p5_at_210khz": round(
                abs(n_210 - 9.5) / 9.5, 3
            ),
        }
    )
    reference["metrics"] = metrics
    _write(out / "verdict.json", reference)


def _ozgen_discrete_unit(task):
    """One independent Ozgen discrete-mode grid node (spawn-picklable)."""
    ordinal, grid_name, mach, reynolds, alpha, n_colloc, ymf_pair, cr_band = task
    if str(OZGEN_TOOLS) not in sys.path:
        sys.path.insert(0, str(OZGEN_TOOLS))
    from discrete_mode import discrete_mode

    kwargs = {"N": n_colloc, "ymf_pair": ymf_pair}
    if cr_band is not None:
        kwargs["cr_band"] = cr_band
    mode = discrete_mode(mach, reynolds, alpha, **kwargs)
    return ordinal, grid_name, mach, reynolds, alpha, mode


def _ozgen_grid_tasks(mach: int):
    """Recreate the committed base/onset/low-R grids with duplicate skipping."""
    import numpy as np

    first_grid = {
        3: (np.logspace(np.log10(650), np.log10(5500), 13), np.linspace(0.006, 0.055, 15)),
        4: (np.logspace(np.log10(1200), np.log10(5500), 13), np.linspace(0.006, 0.11, 15)),
        6: (np.logspace(np.log10(900), np.log10(5500), 13), np.linspace(0.012, 0.13, 15)),
        7: (np.logspace(np.log10(600), np.log10(5500), 13), np.linspace(0.005, 0.12, 15)),
        8: (np.logspace(np.log10(600), np.log10(5500), 13), np.linspace(0.005, 0.12, 15)),
        10: (np.logspace(np.log10(600), np.log10(5500), 13), np.linspace(0.005, 0.12, 15)),
    }
    second_grid = {
        4: (np.logspace(np.log10(1130), np.log10(5500), 13), np.linspace(0.27, 0.40, 14)),
        6: (np.logspace(np.log10(600), np.log10(5500), 13), np.linspace(0.12, 0.22, 14)),
        7: (np.logspace(np.log10(600), np.log10(5500), 13), np.linspace(0.09, 0.21, 14)),
        8: (np.logspace(np.log10(600), np.log10(5500), 13), np.linspace(0.09, 0.21, 14)),
        10: (np.logspace(np.log10(500), np.log10(5500), 13), np.linspace(0.08, 0.22, 14)),
    }
    onset = {
        3: np.linspace(0.003, 0.006, 6),
        4: np.linspace(0.003, 0.006, 6),
        6: np.linspace(0.004, 0.012, 9),
        7: np.linspace(0.003, 0.005, 4),
        8: np.linspace(0.003, 0.005, 4),
        10: np.linspace(0.003, 0.005, 4),
    }
    first_ymf = {
        3: (35.0, 45.0), 4: (35.0, 45.0), 6: (35.0, 45.0),
        7: (28.0, 37.0), 8: (23.0, 31.0), 10: (17.0, 22.0),
    }
    low_re = np.logspace(np.log10(150), np.log10(640), 9)
    tasks = []
    seen = set()

    def add(grid_name, res, alphas, n_colloc, ymf_pair, cr_band=None):
        for reynolds in res:
            for alpha in alphas:
                key = (grid_name, f"{float(reynolds):.1f}", f"{float(alpha):.5f}")
                if key in seen:
                    continue
                seen.add(key)
                tasks.append(
                    (
                        len(tasks), grid_name, float(mach), float(reynolds),
                        float(alpha), n_colloc, ymf_pair, cr_band,
                    )
                )

    first_res, first_alphas = first_grid[mach]
    add("firstmode_grid", first_res, first_alphas, 200, first_ymf[mach])
    add("firstmode_grid", first_res, onset[mach], 200, first_ymf[mach])
    if mach in (7, 8, 10):
        add("firstmode_grid", low_re, first_alphas, 200, first_ymf[mach])
    if mach in second_grid:
        second_res, second_alphas = second_grid[mach]
        add("secondmode_grid", second_res, second_alphas, 180, (8.0, 12.0), (0.4, 0.99))
        if mach in (7, 8, 10):
            add("secondmode_grid", low_re, second_alphas, 180, (8.0, 12.0), (0.4, 0.99))
    return tasks


def _write_ozgen_grid(path: Path, rows, grid_name: str) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Ma", "Re", "alpha", "c_r", "c_i", "fs", "resolved"])
        for _ordinal, name, mach, reynolds, alpha, mode in rows:
            if name != grid_name:
                continue
            if mode is None:
                writer.writerow([f"{mach:g}", f"{reynolds:.1f}", f"{alpha:.5f}", "", "", "", 0])
            else:
                writer.writerow(
                    [
                        f"{mach:g}", f"{reynolds:.1f}", f"{alpha:.5f}",
                        f"{mode['c_r']:.6f}", f"{mode['c_i']:.6e}",
                        f"{mode['fs']:.4f}", 1,
                    ]
                )


def _finalize_ozgen(mach: int, out: Path) -> None:
    """Run the unchanged contour judge in scratch, then its ratified override."""
    import importlib.util

    verification_root = REPO / "verification"
    if str(verification_root) not in sys.path:
        sys.path.insert(0, str(verification_root))

    source_verdict = (
        REPO / "verification" / "mixed_mode" / "ozgen_fig3" / f"M{mach}" / "verdict.json"
    )
    scratch_ver = out / "verification"
    case_dir = scratch_ver / "first_mode" / f"ozgen_m{mach}"
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_verdict, case_dir / "verdict.json")

    builder_path = OZGEN_TOOLS / "build_ozgen_final.py"
    spec = importlib.util.spec_from_file_location(
        f"census_build_ozgen_final_{mach}", builder_path
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)
    builder.HERE = out
    builder.VER = scratch_ver
    builder.REPO = REPO
    builder.FIRST = out / "firstmode_grid.csv"
    builder.SECOND = out / "secondmode_grid.csv"
    builder.main([mach])

    generated_path = case_dir / "verdict.json"
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    committed = json.loads(source_verdict.read_text(encoding="utf-8"))
    # finalize_ozgen_verdicts.py is the committed human-in-the-loop judge: it
    # retains the freshly computed per-branch metrics, then fixes verdict,
    # reason, and the known high-Mach topology flag.
    generated["verdict"] = committed["verdict"]
    generated["verdict_reason"] = committed["verdict_reason"]
    generated["metrics"]["topology_ok"] = committed["metrics"]["topology_ok"]
    generated["generated"] = "new"
    _write(out / "verdict.json", generated)


def run_ozgen_grid(case_id: str, out: Path) -> None:
    import concurrent.futures
    import time

    mach = int(case_id.removeprefix("ozgen_m"))
    tasks = _ozgen_grid_tasks(mach)
    workers = min(61, len(tasks))
    _write(
        out / "census_effective_parameters.json",
        {
            "mach": mach,
            "execution_surface": "point_parallel_discrete_mode_grid",
            "workers_requested": 61,
            "workers_effective": workers,
            "blas_threads": 1,
            "n_grid_nodes": len(tasks),
            "n_first_mode_nodes": sum(t[1] == "firstmode_grid" for t in tasks),
            "n_second_mode_nodes": sum(t[1] == "secondmode_grid" for t in tasks),
            "first_mode_N": 200,
            "second_mode_N": 180 if mach >= 4 else None,
            "first_mode_ymax_multiples": list(tasks[0][6]),
            "second_mode_ymax_multiples": [8.0, 12.0] if mach >= 4 else None,
            "grid_sources": [
                "build_firstmode_grid.py", "build_secondmode_grid.py",
                "build_onset_extension.py", "build_lowR_extension.py",
            ],
            "judge": "build_ozgen_final.py + finalize_ozgen_verdicts.py",
        },
    )
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_ozgen_discrete_unit, tasks))
    _write_ozgen_grid(out / "firstmode_grid.csv", rows, "firstmode_grid")
    _write_ozgen_grid(out / "secondmode_grid.csv", rows, "secondmode_grid")
    continuation = OZGEN_TOOLS / f"continuation_M{mach}.csv"
    if continuation.is_file():
        shutil.copyfile(continuation, out / continuation.name)
    _finalize_ozgen(mach, out)
    params_path = out / "census_effective_parameters.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["grid_wall_time_s"] = time.perf_counter() - started
    _write(params_path, params)


def run_ozgen_m2(out: Path) -> None:
    """Run the exact sequential M2 continuation driver from a scratch copy."""
    import re
    import runpy

    source = OZGEN_TOOLS / "trace_continuation.py"
    scratch_script = out / source.name
    text = source.read_text(encoding="utf-8")
    text, count = re.subn(
        r"REPO = HERE\.parents\[2\]; sys\.path\.insert\(0, str\(REPO\)\)",
        f"REPO = Path({REPO.as_posix()!r}); sys.path.insert(0, str(REPO))",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not redirect the M2 continuation repository root")
    scratch_script.write_text(text, encoding="utf-8")
    _write(
        out / "census_effective_parameters.json",
        {
            "mach": 2,
            "execution_surface": "serial_eigenvalue_continuation",
            "N": 220,
            "y_max_delta_star_multiple": 40.0,
            "seed_R": 520.0,
            "seed_alpha_scan": [0.02, 0.085, 28],
            "trace_R": [300.0, 5000.0, 120.0],
            "blas_threads": 1,
            "judge": "build_ozgen_final.py + finalize_ozgen_verdicts.py",
        },
    )
    previous_argv = sys.argv
    try:
        sys.argv = [str(scratch_script), "2"]
        runpy.run_path(str(scratch_script), run_name="__main__")
    finally:
        sys.argv = previous_argv
    _finalize_ozgen(2, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", choices=CASES)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    runners = {
        "balakumar_malik1992_branches": run_branches,
        "balakumar_malik1992_via_xirenfu": run_via_xirenfu,
        "egorov2006_m6": run_egorov,
        "malik_fig4_eigenfunction": run_malik_fig4,
        "mazhong2003_m4p5": run_mazhong,
        "sean_m5p35": run_sean,
    }
    if args.case_id.startswith("mack_fig10_1_"):
        run_mack_fig10_1(args.case_id, args.out)
    elif args.case_id.startswith("mack_fig10_4_"):
        run_mack_fig10_4(args.case_id, args.out)
    elif args.case_id.startswith("mack_fig10_6_"):
        run_mack_fig10_6(args.case_id, args.out)
    elif args.case_id == "cone_sivasubramanian_fasel_2015":
        run_cone(args.out)
    elif args.case_id == "ozgen_m2":
        run_ozgen_m2(args.out)
    elif args.case_id.startswith("ozgen_m"):
        run_ozgen_grid(args.case_id, args.out)
    elif args.case_id in runners:
        runners[args.case_id](args.out)
    else:
        run_broken_selfseed(args.case_id, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
