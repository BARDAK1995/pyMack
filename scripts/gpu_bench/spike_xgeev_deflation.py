"""Slice 03 probe: cuSOLVER Xgeev availability and exact BC deflation.

Read-only pyMack probe. Writes one JSON evidence file
(docs/gpu/benchmarks/spike_xgeev_deflation.json) and exits 0 only when the
slice-03 contract gate passes on all three anchors.

FACT 1 (Xgeev): does the installed stack expose cusolverDnXgeev usably from
CuPy? Probed one fact per isolated child process (presence, single-solve
functional tests, and a deliberate reproduction of the crash signature),
because the cusolver Xgeev host path on this stack has a heap-layout-
dependent native crash (0xC0000005): a small eigensolve followed by a large
one in the same process dies inside the large call, while either alone
succeeds. Isolation keeps the deflation half and the exit code immune to it
and turns the crash itself into recorded evidence. Present or absent is
recorded either way -- absence is NOT a failure of this slice.

FACT 2 (deflation): temporal pencils (A, B) carry BC rows whose B-row is
identically zero (6 rows for the 2D 4n-state pencils, 8 for the 3D 5n-state
pencil; all are unit rows here because every anchor uses isothermal
perturbation BCs -- the "wall-temperature linear constraint" of the slice
text reduces to a Dirichlet row in these production cases). Eliminating the
constrained dofs yields an (m-k) pencil whose characteristic polynomial
equals the original's up to sign -- for unit rows the reduced pencil is an
exact submatrix (verified bitwise). The probe then compares the finite
spectra of both pencils via scipy QZ:

  * finite/infinite classification uses QZ's homogeneous (alpha, beta) output
    (|beta|/||(alpha,beta)|| <= 1e-12 => infinite class); the observed
    classification gap spans many orders of magnitude on every anchor;
  * finite eigenvalues are matched one-to-one by linear-sum assignment;
  * a control experiment (row/column-permuted ORIGINAL pencil vs the original,
    i.e. no deflation at all) measures the intrinsic scipy-QZ reproducibility
    noise floor for each pencil.

Contract gate (spec as amended 2026-07-04 by the orchestrator after this
probe's permutation control proved the original absolute 1e-10 gate
unimplementable -- see the amendment note in
specs/pymack-gpu/slices/03-xgeev-deflation-probe.md): per anchor, finite
counts identical AND the reduction is an exact bitwise submatrix AND matched
max |dc| <= that anchor's measured QZ noise-floor control, with 1e-6 as an
absolute backstop. See the JSON's "gate_calibration_history".

The pencils are captured read-only: 2D anchors by intercepting
scipy.linalg.eig around the unchanged public solvers (robust to slice 04's
pending assembly extraction landing or not), the 3D anchor via the public
assemble/apply-BC functions exactly as verification/compute_mack_fig10_4.py
uses them.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "docs" / "gpu" / "benchmarks" / "spike_xgeev_deflation.json"
MAX_DC_BACKSTOP = 1.0e-6      # absolute backstop of the amended contract gate
INF_BETA_THRESH = 1.0e-12     # |beta|/||(alpha,beta)|| below this => infinite class
PHYS_BOX = (-0.5, 1.5, 0.5)   # production physical filter: c_r in (lo,hi), |c_i|<cap
XGEEV_CHILD_FLAG = "--xgeev-child"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot JSON serialize {type(obj)!r}")


def _write_json(record: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _dist_version("numpy"),
            "scipy": _dist_version("scipy"),
            "cupy": _dist_version("cupy"),
            "cupy-cuda12x": _dist_version("cupy-cuda12x"),
            "nvidia-cusolver-cu12": _dist_version("nvidia-cusolver-cu12"),
        },
        "nvcp_sysmem_fallback": (
            "not_queried: this probe is functional fact-finding, not a timing "
            "benchmark; no performance number here feeds a milestone gate"
        ),
    }


# ======================================================================
# FACT 1 -- Xgeev availability (runs inside an isolated child process)
# ======================================================================

def xgeev_child_main(checkpoint_path: Path, task: str) -> int:
    """Child-process body: run ONE Xgeev probe task, checkpointing to a file.

    Tasks (one per process, because the cusolver Xgeev host path has a
    heap-layout-dependent ACCESS_VIOLATION on this stack -- see the
    "sequence" task -- so each fact is collected in a fresh process):

      presence        -- imports, API surface, raw bindings, ctypes DLL check
      functional:N    -- single cupy.linalg.eigvals(N x N complex128) vs numpy
      sequence:M,N    -- eigvals(M) then eigvals(N) in ONE process; on this
                         stack M=64, N=997 deterministically dies with
                         0xC0000005 (the crash signature being documented)
    """
    out: dict = {"task": task, "last_completed_step": None}

    def checkpoint(step: str) -> None:
        out["last_completed_step"] = step
        checkpoint_path.write_text(
            json.dumps(out, default=_json_default), encoding="utf-8"
        )

    try:
        import numpy as np
        import cupy
    except Exception as exc:
        out["error"] = f"cupy import: {type(exc).__name__}: {exc}"
        checkpoint("import_failed")
        return 0
    checkpoint("imports")

    def eig_case(n: int) -> dict:
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        t0 = time.perf_counter()
        w_gpu = cupy.asnumpy(cupy.linalg.eigvals(cupy.asarray(a)))
        cupy.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        w_cpu = np.linalg.eigvals(a)
        cost = np.abs(w_gpu[:, None] - w_cpu[None, :])
        ri, ci = linear_sum_assignment(cost)
        return {
            "n": n,
            "dtype": "complex128",
            "elapsed_s": float(dt),
            "max_abs_dw_vs_numpy": float(cost[ri, ci].max()),
        }

    if task == "presence":
        info: dict = {"cupy_version": cupy.__version__}
        try:
            info["cuda_runtime_version"] = int(cupy.cuda.runtime.runtimeGetVersion())
            info["cuda_driver_version"] = int(cupy.cuda.runtime.driverGetVersion())
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            info["device_name"] = (
                name.decode() if isinstance(name, bytes) else str(name)
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            info["device_error"] = f"{type(exc).__name__}: {exc}"
        out["cupy_import"] = info
        out["high_level_api"] = {
            "cupy.linalg.eig": bool(hasattr(cupy.linalg, "eig")),
            "cupy.linalg.eigvals": bool(hasattr(cupy.linalg, "eigvals")),
        }
        checkpoint("high_level_api")

        # Raw bindings. The extension module only imports after CuPy has
        # preloaded the cusolver DLL from the wheel layout; a cusolver-touching
        # op triggers that (a bare import in a fresh process fails with a DLL
        # load error -- recorded fact on this machine).
        raw: dict = {}
        try:
            cupy.linalg.qr(cupy.eye(4))  # force cusolver preload
            mod = importlib.import_module("cupy_backends.cuda.libs.cusolver")
            raw["module"] = "cupy_backends.cuda.libs.cusolver"
            raw["geev_symbols"] = sorted(
                n for n in dir(mod) if "geev" in n.lower()
            )
            raw["xgeev_present"] = any(
                "xgeev" in n.lower() for n in raw["geev_symbols"]
            )
        except Exception as exc:
            raw["error"] = f"{type(exc).__name__}: {exc}"
            raw["xgeev_present"] = False
        out["raw_bindings"] = raw
        checkpoint("raw_bindings")

        # Secondary evidence: ctypes against the wheel's own DLL. The wheel
        # layout keeps cusolver64_*.dll under site-packages/nvidia/cusolver/
        # bin, NOT on PATH; its dependencies live in sibling nvidia/*/bin
        # dirs, so each must be registered via os.add_dll_directory before
        # loading (this is exactly why a bare
        # ctypes.WinDLL("cusolver64_12.dll") fails on this machine).
        ct: dict = {"feasible_without_building_bindings": False}
        try:
            spec = importlib.util.find_spec("nvidia")
            if spec and spec.submodule_search_locations:
                base = Path(list(spec.submodule_search_locations)[0])
                bin_dirs = sorted(
                    str(p) for p in base.glob("*/bin") if p.is_dir()
                )
                ct["dll_directories_registered"] = bin_dirs
                if os.name == "nt":
                    for d in bin_dirs:
                        os.add_dll_directory(d)
                dlls = sorted(base.glob("cusolver/bin/cusolver64_*.dll"))
                ct["candidate_dlls"] = [str(p) for p in dlls]
                for dll_path in dlls:
                    if "Mg" in dll_path.name:
                        continue
                    dll = (
                        ctypes.WinDLL(str(dll_path))
                        if os.name == "nt"
                        else ctypes.CDLL(str(dll_path))
                    )
                    entry = {
                        "dll": str(dll_path),
                        "loaded": True,
                        "cusolverDnXgeev": bool(hasattr(dll, "cusolverDnXgeev")),
                        "cusolverDnXgeev_bufferSize": bool(
                            hasattr(dll, "cusolverDnXgeev_bufferSize")
                        ),
                    }
                    try:
                        ver = ctypes.c_int(0)
                        rc = dll.cusolverGetVersion(ctypes.byref(ver))
                        entry["cusolverGetVersion"] = {
                            "rc": int(rc),
                            "version": int(ver.value),
                        }
                    except Exception as exc:  # pragma: no cover
                        entry["cusolverGetVersion"] = f"{type(exc).__name__}: {exc}"
                    ct["probe"] = entry
                    ct["feasible_without_building_bindings"] = bool(
                        entry["cusolverDnXgeev"]
                        and entry["cusolverDnXgeev_bufferSize"]
                    )
                    break
        except Exception as exc:
            ct["error"] = f"{type(exc).__name__}: {exc}"
        out["ctypes_wheel_dll"] = ct
        checkpoint("presence_complete")

    elif task.startswith("functional:"):
        n = int(task.split(":", 1)[1])
        checkpoint(f"functional_start_n{n}")
        try:
            out["case"] = eig_case(n)
            checkpoint("functional_complete")
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            checkpoint("functional_python_error")

    elif task.startswith("sequence:"):
        sizes = [int(s) for s in task.split(":", 1)[1].split(",")]
        out["cases"] = []
        checkpoint("sequence_start")
        try:
            for n in sizes:
                out["cases"].append(eig_case(n))
                checkpoint(f"sequence_done_n{n}")
            checkpoint("sequence_complete")
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            checkpoint("sequence_python_error")

    else:
        out["error"] = f"unknown task {task!r}"
        checkpoint("unknown_task")
    return 0


def _run_xgeev_child(task: str) -> dict:
    """Launch one child task; return {exit_code, payload or None, stderr_tail}."""
    ckpt = OUT_JSON.parent / ".spike_xgeev_child_checkpoint.json"
    ckpt.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                XGEEV_CHILD_FLAG,
                str(ckpt),
                task,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        result: dict = {"task": task, "exit_code": proc.returncode}
        if ckpt.exists():
            try:
                result["payload"] = json.loads(ckpt.read_text(encoding="utf-8"))
            except Exception as exc:
                result["checkpoint_parse_error"] = f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            result["stderr_tail"] = proc.stderr[-500:]
    except Exception as exc:
        result = {"task": task, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        ckpt.unlink(missing_ok=True)
    return result


def probe_xgeev_isolated() -> dict:
    """Collect the Xgeev facts, one isolated child process per fact.

    One process per fact because the cusolver Xgeev host path on this stack
    has a heap-layout-dependent native crash (0xC0000005): eigvals(64)
    followed by eigvals(997) in one process dies deterministically inside the
    second call, while either size alone succeeds deterministically (and even
    a pure-CPU numpy eigvals(64) beforehand arms the crash, implicating a
    layout-sensitive host-side overrun, not the GPU kernel). The probe
    records the working single-solve facts AND reproduces the crash signature
    deliberately, since both determine how a slice-09 Xgeev rung would have
    to be engineered (sandboxed, uniform-size usage).
    """
    record: dict = {
        "question": "Does this Python/CUDA stack expose cusolverDnXgeev usably?",
        "isolation": (
            "one child process per fact with file checkpointing; keeps "
            "native CUDA crashes from corrupting the deflation result or "
            "this script's exit code"
        ),
        "pass_for_slice": True,  # present or absent, recorded either way
    }

    presence = _run_xgeev_child("presence")
    record["presence"] = presence
    payload = presence.get("payload") or {}
    for key in ("cupy_import", "high_level_api", "raw_bindings", "ctypes_wheel_dll"):
        if key in payload:
            record[key] = payload[key]

    # Functional single-solve tests, one process per size. n=997 is the
    # deflated M10 anchor size (m - 8 = 5*201 - 8).
    functional = []
    for n in (64, 997):
        res = _run_xgeev_child(f"functional:{n}")
        entry: dict = {"n": n, "exit_code": res.get("exit_code")}
        pl = res.get("payload") or {}
        if pl.get("case"):
            entry.update(pl["case"])
            entry["completed"] = pl.get("last_completed_step") == "functional_complete"
        else:
            entry["completed"] = False
            entry["last_completed_step"] = pl.get("last_completed_step")
            if "stderr_tail" in res:
                entry["stderr_tail"] = res["stderr_tail"]
        functional.append(entry)
    record["functional_single_solve"] = {
        "api": "cupy.linalg.eigvals",
        "cases": functional,
    }

    # Deliberate crash-signature sampling: 64 then 997 in one process,
    # repeated. During probe development this sequence died 3/3 times with
    # 0xC0000005 inside the n=997 call (and a pure-CPU numpy eigvals(64)
    # beforehand armed it equally), then later runs completed cleanly -- the
    # trigger is heap-layout/state dependent and drifts over time, so a
    # repeated sample is recorded instead of a single shot.
    seq_runs = []
    n_crashed = 0
    for rep in range(1, 6):
        seq = _run_xgeev_child("sequence:64,997")
        code = seq.get("exit_code")
        pl = seq.get("payload") or {}
        crashed = code not in (0, None)
        n_crashed += int(crashed)
        seq_runs.append(
            {
                "repeat": rep,
                "exit_code": code,
                "exit_code_hex": (
                    hex(code & 0xFFFFFFFF) if isinstance(code, int) else None
                ),
                "last_completed_step": pl.get("last_completed_step"),
            }
        )
    record["crash_signature_sequence_64_997"] = {
        "runs": seq_runs,
        "crash_count": n_crashed,
        "run_count": len(seq_runs),
        "session_history_note": (
            "earlier in this probe's development session the identical "
            "sequence crashed deterministically (3/3 in-probe, 8/8 across "
            "bisection variants, including with a pure-CPU numpy "
            "eigvals(64) as the arming step), while every single-solve "
            "process succeeded; the trigger is heap-layout dependent and "
            "varies with machine state"
        ),
        "interpretation": (
            "0xC0000005 reproduces the layout-dependent host-side crash "
            "inside the n=997 cusolverDnXgeev call"
            if n_crashed
            else "no crash in this sample; instability remains documented "
            "from the session history above"
        ),
    }

    single_ok = all(c.get("completed") for c in functional)
    api_ok = bool(record.get("high_level_api", {}).get("cupy.linalg.eig"))
    raw_ok = bool(record.get("raw_bindings", {}).get("xgeev_present"))
    crashed_seq = n_crashed > 0
    if api_ok and single_ok:
        record["verdict"] = "xgeev_usable_via_cupy_linalg_eig_single_solve"
        record["stability_caveat"] = (
            "cusolverDnXgeev on this stack is functionally correct per "
            "single-solve process but has an observed heap-layout-dependent "
            "native crash (0xC0000005) when a small eigensolve precedes a "
            "large one in the same process "
            + (
                f"(reproduced live: {n_crashed}/{len(seq_runs)} sample runs)"
                if crashed_seq
                else "(not reproduced in this sample; documented from "
                "session history)"
            )
            + "; any slice-09 Xgeev rung must run it sandboxed (worker "
            "process, uniform matrix size) and treat worker death as an "
            "escalation to CPU QZ"
        )
    elif raw_ok:
        record["verdict"] = "xgeev_bindings_present_functional_test_failed"
    elif record.get("ctypes_wheel_dll", {}).get("feasible_without_building_bindings"):
        record["verdict"] = "xgeev_symbol_in_wheel_dll_ctypes_feasible"
    else:
        record["verdict"] = "xgeev_absent"
    return record


# ======================================================================
# FACT 2 -- exact BC deflation on three anchors
# ======================================================================

def capture_solver_pencil(np, scipy_linalg, run_solver):
    """Capture the (A, B) pencil a solver hands to scipy.linalg.eig.

    Interception is used (rather than the pending slice-04 _assemble_*
    extraction) so the probe captures exactly the production pencil whether
    or not that uncommitted refactor survives review. Arrays are copied
    BEFORE the original call in case a caller ever passes overwrite flags.
    """
    calls = []
    original = scipy_linalg.eig

    def wrapped(A, B=None, *args, **kwargs):
        if B is not None:
            calls.append((np.array(A, copy=True), np.array(B, copy=True)))
        return original(A, B, *args, **kwargs)

    scipy_linalg.eig = wrapped
    try:
        run_solver()
    finally:
        scipy_linalg.eig = original
    if len(calls) != 1:
        raise RuntimeError(
            f"expected exactly one scipy.linalg.eig(A, B) call, saw {len(calls)}"
        )
    return calls[0]


def build_malik_anchor(np, scipy_linalg) -> dict:
    from pymack import CompressibleBlasiusProfile
    from pymack.solver import solve_temporal_compressible

    ma, r_l, alpha, pr, gamma = 10.0, 2000.0, 0.105, 0.7, 1.4
    t0_k = 4200.0 * 5.0 / 9.0
    t_edge_k = t0_k / (1.0 + 0.5 * (gamma - 1.0) * ma**2)
    t_rec_k = t_edge_k * (1.0 + 0.5 * (gamma - 1.0) * pr**0.5 * ma**2)
    n_cheb = 128
    profile = CompressibleBlasiusProfile(
        Ma=ma, T_edge=t_edge_k, T_wall=0.1 * t_rec_k, gamma=gamma, Pr=pr,
        wall_bc="isothermal", viscosity_model="sutherland",
        sutherland_S=198.6 / 1.8, n_points=4000, eta_max=40.0,
    )
    A, B = capture_solver_pencil(
        np, scipy_linalg,
        lambda: solve_temporal_compressible(
            profile, alpha, r_l, ma, pr, gamma, N=n_cheb, y_max=75.0,
            wall_bc="isothermal", length_scale="L_star", lambda_mu_ratio=1.2,
        ),
    )
    n = A.shape[0] // 4
    return {
        "id": "malik_case4_2d_temporal_N128",
        "source_reference": "validation/test_malik1990_case4_anchor.py (N=128 per slice)",
        "parameters": {
            "Ma": ma, "R_L": r_l, "alpha": alpha, "Pr": pr, "gamma": gamma,
            "N": n_cheb, "y_max": 75.0, "wall_bc": "isothermal",
            "length_scale": "L_star", "lambda_mu_ratio": 1.2,
        },
        "A": A, "B": B,
        "expected_bc_rows": sorted([0, n - 1, n, 2 * n - 1, 2 * n, 3 * n - 1]),
    }


def build_ozgen_anchor(np, scipy_linalg) -> dict:
    import pymack
    from pymack.scales import delta_star_over_lstar
    from pymack.temporal_solver import solve_temporal_2d

    ma = 2.0
    profile = pymack.make_flatplate_profile(ma)
    dstar = float(delta_star_over_lstar(profile))
    re_grid = np.logspace(np.log10(350.0), np.log10(5500.0), 13)
    alpha_grid = np.linspace(0.010, 0.075, 15)
    r_l = float(re_grid[len(re_grid) // 2])
    alpha = float(alpha_grid[len(alpha_grid) // 2])
    y_max = 35.0 * dstar
    A, B = capture_solver_pencil(
        np, scipy_linalg,
        lambda: solve_temporal_2d(
            profile, alpha, r_l, ma, N=200, y_max=y_max, length_scale="L_star",
        ),
    )
    n = A.shape[0] // 4
    return {
        "id": "ozgen_m2_firstmode_midgrid_N200",
        "source_reference": (
            "verification/mixed_mode/ozgen_fig3/_refdigitize/build_firstmode_grid.py "
            "(GRID[2] mid-node, N=200, short domain ymf=35)"
        ),
        "parameters": {
            "Ma": ma, "R_L": r_l, "alpha": alpha, "Pr": 0.72, "gamma": 1.4,
            "N": 200, "ymf": 35.0, "delta_star_over_lstar": dstar,
            "y_max": y_max, "length_scale": "L_star",
        },
        "A": A, "B": B,
        "expected_bc_rows": sorted([0, n - 1, n, 2 * n - 1, 2 * n, 3 * n - 1]),
    }


def build_m10_anchor(np) -> dict:
    import pymack
    from pymack.solver import (
        apply_dirichlet_freestream_bc_3d,
        apply_wall_bc_3d,
        assemble_temporal_compressible_3d_evp,
    )

    ma, r_l, pr, gamma = 10.0, 1500.0, 0.72, 1.4
    alpha = 0.030   # mid of ALPHA_GRID[10.0] = arange(0.010, 0.050, 0.0025), 17 pts
    psi_deg = 54.0  # mid of PSI_GRID[10.0] = arange(42, 63, 3), 8 pts
    beta = float(alpha * np.tan(np.radians(psi_deg)))
    profile = pymack.make_mack_profile(ma, condition="table_11_1")
    A, B, _y, D1, n, _, _, _ = assemble_temporal_compressible_3d_evp(
        profile, alpha, beta, r_l, ma, pr, gamma,
        N=200, y_max=150.0, length_scale="L_star", lambda_mu_ratio=0.0,
    )
    apply_wall_bc_3d(A, B, D1, n)
    apply_dirichlet_freestream_bc_3d(A, B, n)
    return {
        "id": "mack_fig10_4_m10_3d_R1500_midgrid_N200",
        "source_reference": (
            "verification/compute_mack_fig10_4.py (R=1500, mid alpha/psi grid, "
            "N_BY_MACH[10.0]=200, Y_MAX_BY_MACH[10.0]=150)"
        ),
        "parameters": {
            "Ma": ma, "R_L": r_l, "alpha": alpha, "psi_deg": psi_deg,
            "beta": beta, "Pr": pr, "gamma": gamma, "N": 200, "y_max": 150.0,
            "length_scale": "L_star", "lambda_mu_ratio": 0.0,
        },
        "A": A, "B": B,
        "expected_bc_rows": sorted(
            [n - 1, 2 * n - 1, 3 * n - 1, 4 * n - 1, 0, n, 2 * n, 3 * n]
        ),
    }


def eliminate_bc_rows(np, scipy_linalg, A, B, bc_rows) -> dict:
    """Build the (m-k) reduced pencil from the zero-B constraint rows.

    Constraint rows say A[r, :] x = c * 0 = 0, so finite-c eigenvectors live
    in null(C), C = A[bc_rows, :]. When every constraint row is a unit row
    (all three anchors: pure Dirichlet after isothermal BCs), the null-space
    basis is an exact 0/1 selection and the reduced pencil is bitwise equal
    to a submatrix of (A, B); the characteristic polynomials then agree up to
    sign by Laplace expansion along the unit rows -- deflation is exact by
    construction, before any eigensolve. A pivoted-QR general path covers
    non-unit constraint rows (e.g. adiabatic walls), unexercised here.
    """
    m = A.shape[0]
    k = len(bc_rows)
    C = A[bc_rows, :]
    unit_rows = all(
        np.count_nonzero(C[i]) == 1 and C[i][np.flatnonzero(C[i])[0]] == 1.0
        for i in range(k)
    )
    dyn = [i for i in range(m) if i not in set(map(int, bc_rows))]

    if unit_rows:
        pivot_cols = sorted(int(np.flatnonzero(C[i])[0]) for i in range(k))
        free_cols = [j for j in range(m) if j not in set(pivot_cols)]
        A_red = A[np.ix_(dyn, free_cols)]
        B_red = B[np.ix_(dyn, free_cols)]
        exact = True
        cz_residual = 0.0
    else:  # general linear constraints: pivoted-QR null-space basis
        _q, r_, piv = scipy_linalg.qr(C, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r_))
        if int((diag > np.finfo(float).eps * max(C.shape) * diag[0]).sum()) != k:
            raise RuntimeError("constraint rows are rank deficient")
        pivot_cols = list(map(int, piv[:k]))
        free_cols = [j for j in range(m) if j not in set(pivot_cols)]
        X = -scipy_linalg.solve(C[:, pivot_cols], C[:, free_cols])
        Z = np.zeros((m, m - k), dtype=A.dtype)
        Z[pivot_cols, :] = X
        Z[free_cols, :] = np.eye(m - k)
        A_red = A[dyn, :] @ Z
        B_red = B[dyn, :] @ Z
        exact = False
        cz_residual = float(np.abs(C @ Z).max())

    return {
        "A_red": A_red,
        "B_red": B_red,
        "constraint_rows_all_unit": bool(unit_rows),
        "eliminated_dof_columns": list(pivot_cols),
        "reduced_is_exact_submatrix": bool(exact),
        "constraint_nullspace_residual_max": cz_residual,
    }


def homogeneous_spectrum(np, scipy_linalg, A, B):
    """QZ with homogeneous (alpha, beta) output -> (c, beta_normalized)."""
    ab = scipy_linalg.eig(
        A, B, right=False, homogeneous_eigvals=True, check_finite=False
    )
    al, be = ab[0], ab[1]
    norm = np.hypot(np.abs(al), np.abs(be))
    beta_n = np.abs(be) / norm
    with np.errstate(divide="ignore", invalid="ignore"):
        c = al / be
    return c, beta_n


def classify_finite(np, c, beta_n) -> dict:
    inf_mask = beta_n <= INF_BETA_THRESH
    fin = ~inf_mask
    gap_lo = float(beta_n[inf_mask].max()) if inf_mask.any() else 0.0
    gap_hi = float(beta_n[fin].min()) if fin.any() else math.inf
    return {
        "finite_values": c[fin],
        "n_infinite": int(inf_mask.sum()),
        "n_finite": int(fin.sum()),
        "classification_gap": {
            "largest_infinite_class_beta_n": gap_lo,
            "smallest_finite_class_beta_n": gap_hi,
            "threshold": INF_BETA_THRESH,
            "unambiguous": bool(gap_lo * 1e3 <= INF_BETA_THRESH <= gap_hi / 1e3),
        },
    }


def match_spectra(np, w_a, w_b) -> dict:
    """One-to-one assignment matching between equal-size finite spectra."""
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(w_a[:, None] - w_b[None, :])
    ri, ci = linear_sum_assignment(cost)
    d = cost[ri, ci]
    box = (
        (w_b[ci].real > PHYS_BOX[0])
        & (w_b[ci].real < PHYS_BOX[1])
        & (np.abs(w_b[ci].imag) < PHYS_BOX[2])
    )
    worst = np.argsort(d)[-5:][::-1]
    return {
        "n_matched": int(len(d)),
        "max_abs_dc": float(d.max()),
        "median_abs_dc": float(np.median(d)),
        "physical_box": {
            "definition": "c_r in (-0.5, 1.5), |c_i| < 0.5 (production filter)",
            "n_pairs": int(box.sum()),
            "max_abs_dc": float(d[box].max()) if box.any() else None,
        },
        "worst_pairs": [
            {"c": complex(w_b[ci[j]]), "abs_dc": float(d[j])} for j in worst
        ],
    }


def reduced_b_invertibility(np, scipy_linalg, B_red) -> dict:
    sv = scipy_linalg.svdvals(B_red, check_finite=False)
    smax = float(sv[0]) if sv.size else 0.0
    smin = float(sv[-1]) if sv.size else 0.0
    tol = np.finfo(float).eps * max(B_red.shape) * max(smax, 1.0)
    rank = int((sv > tol).sum())
    return {
        "invertible": bool(rank == B_red.shape[0]),
        "rank": rank,
        "size": int(B_red.shape[0]),
        "rank_tol": float(tol),
        "sigma_max": smax,
        "sigma_min": smin,
        "cond_2": float(smax / smin) if smin > 0.0 else math.inf,
        "exactly_zero_columns": int((np.linalg.norm(B_red, axis=0) == 0.0).sum()),
    }


def check_deflation_anchor(np, scipy_linalg, anchor: dict) -> dict:
    t_start = time.perf_counter()
    A = anchor.pop("A")
    B = anchor.pop("B")
    expected_rows = anchor.pop("expected_bc_rows")
    m = A.shape[0]

    bc_rows = np.flatnonzero(np.linalg.norm(B, axis=1) == 0.0)
    rows_match = list(map(int, bc_rows)) == list(expected_rows)

    elim = eliminate_bc_rows(np, scipy_linalg, A, B, bc_rows)
    A_red, B_red = elim.pop("A_red"), elim.pop("B_red")

    t0 = time.perf_counter()
    c_full, bn_full = homogeneous_spectrum(np, scipy_linalg, A, B)
    t1 = time.perf_counter()
    c_red, bn_red = homogeneous_spectrum(np, scipy_linalg, A_red, B_red)
    t2 = time.perf_counter()

    cls_full = classify_finite(np, c_full, bn_full)
    cls_red = classify_finite(np, c_red, bn_red)
    w_full = cls_full.pop("finite_values")
    w_red = cls_red.pop("finite_values")

    counts_ok = (
        len(w_full) == len(w_red)
        and cls_full["n_infinite"] == cls_red["n_infinite"] + len(bc_rows)
    )
    comparison = (
        match_spectra(np, w_red, w_full)
        if len(w_full) == len(w_red)
        else {"error": "finite counts differ", "max_abs_dc": math.inf}
    )

    # Control: permuted ORIGINAL pencil vs the original -- identical spectrum
    # by construction, so its matched |dc| is the pure scipy-QZ
    # reproducibility noise floor for this pencil (no deflation involved).
    rng = np.random.default_rng(12345)
    P, Q = rng.permutation(m), rng.permutation(m)
    c_ctl, bn_ctl = homogeneous_spectrum(
        np, scipy_linalg, A[np.ix_(P, Q)], B[np.ix_(P, Q)]
    )
    cls_ctl = classify_finite(np, c_ctl, bn_ctl)
    w_ctl = cls_ctl.pop("finite_values")
    control = (
        match_spectra(np, w_ctl, w_full)
        if len(w_ctl) == len(w_full)
        else {"error": "control finite count differs", "max_abs_dc": math.inf}
    )

    invert = reduced_b_invertibility(np, scipy_linalg, B_red)

    max_dc = float(comparison["max_abs_dc"])
    noise_dominated = bool(
        counts_ok
        and elim["reduced_is_exact_submatrix"]
        and max_dc <= float(control["max_abs_dc"])
    )
    # Amended contract gate (spec amendment 2026-07-04): exact-submatrix
    # reduction AND max|dc| within the anchor's own measured QZ noise floor,
    # with an absolute 1e-6 backstop so a pathological control can never
    # excuse a real discrepancy.
    contract_gate_passed = bool(noise_dominated and max_dc <= MAX_DC_BACKSTOP)

    return {
        **anchor,
        "matrix_shape": [int(m), int(m)],
        "bc_rows_detected": list(map(int, bc_rows)),
        "bc_rows_match_expected_structure": bool(rows_match),
        "elimination": elim,
        "reduced_shape": [int(A_red.shape[0]), int(A_red.shape[1])],
        "spectrum_counts": {
            "full": {
                "n_finite": cls_full["n_finite"],
                "n_infinite": cls_full["n_infinite"],
                "classification_gap": cls_full["classification_gap"],
            },
            "reduced": {
                "n_finite": cls_red["n_finite"],
                "n_infinite": cls_red["n_infinite"],
                "classification_gap": cls_red["classification_gap"],
            },
            "finite_counts_equal": bool(len(w_full) == len(w_red)),
            "infinite_count_difference_equals_k": bool(
                cls_full["n_infinite"] == cls_red["n_infinite"] + len(bc_rows)
            ),
        },
        "spectrum_comparison": comparison,
        "qz_noise_control": {
            "description": (
                "row/column-permuted ORIGINAL pencil vs the original: same "
                "exact spectrum, so matched |dc| here is scipy QZ's own "
                "reproducibility floor -- no deflation involved"
            ),
            **control,
        },
        "reduced_B": invert,
        "contract_gate": {
            "criterion": (
                "counts identical AND exact bitwise submatrix AND "
                "max|dc| <= measured QZ noise-floor control AND "
                "max|dc| <= 1e-6 absolute backstop "
                "(spec amended 2026-07-04)"
            ),
            "backstop_abs_dc": MAX_DC_BACKSTOP,
            "control_max_abs_dc": float(control["max_abs_dc"]),
        },
        "contract_gate_passed": contract_gate_passed,
        "deflation_error_below_qz_noise_floor": noise_dominated,
        "timings_s": {
            "qz_full": float(t1 - t0),
            "qz_reduced": float(t2 - t1),
            "total": float(time.perf_counter() - t_start),
        },
    }


def run_deflation_probe() -> dict:
    import numpy as np
    import scipy.linalg as scipy_linalg

    anchors = [
        build_malik_anchor(np, scipy_linalg),
        build_ozgen_anchor(np, scipy_linalg),
        build_m10_anchor(np),
    ]
    checked = [check_deflation_anchor(np, scipy_linalg, a) for a in anchors]
    all_contract = bool(all(a["contract_gate_passed"] for a in checked))
    all_noise = bool(
        all(a["deflation_error_below_qz_noise_floor"] for a in checked)
    )
    all_exact = bool(
        all(a["elimination"]["reduced_is_exact_submatrix"] for a in checked)
    )
    return {
        "question": "Is exact structural deflation spectrum-identical?",
        "method": (
            "zero-B constraint rows -> exact submatrix elimination (unit "
            "rows) or pivoted-QR null-space projection (general rows); "
            "finite/infinite split via homogeneous QZ (alpha, beta); "
            "one-to-one assignment matching; permuted-pencil control for "
            "the QZ reproducibility floor; reduced-B SVD for invertibility"
        ),
        "anchors": checked,
        "all_contract_gates_passed": all_contract,
        "all_deflation_exact_by_construction": all_exact,
        "all_deflation_error_below_qz_noise_floor": all_noise,
        "gate_calibration_history": (
            "The slice's original absolute gate (max|dc| <= 1e-10) was "
            "proven unimplementable by the permutation control: scipy QZ's "
            "own reproducibility error on these pencils (no deflation "
            "involved) violates it on every anchor (2.1e-10 / 2.4e-9 / "
            "1.3e-7 for m = 516 / 804 / 1005). The orchestrator amended the "
            "spec on 2026-07-04: the gate is now exact-submatrix reduction "
            "AND max|dc| <= the anchor's measured QZ noise-floor control, "
            "with 1e-6 as an absolute backstop."
        ),
    }


def main() -> int:
    started = time.perf_counter()
    record = {
        "schema": "pymack-gpu-slice-03-xgeev-deflation-probe-v2",
        "slice": "03-xgeev-deflation-probe",
        "generated_at_utc": _utc_now(),
        "command": "python scripts/gpu_bench/spike_xgeev_deflation.py",
        "environment": environment_record(),
        "xgeev": probe_xgeev_isolated(),
        "deflation": None,
        "status": "running",
    }

    exit_code = 1
    try:
        record["deflation"] = run_deflation_probe()
        if record["deflation"]["all_contract_gates_passed"]:
            record["status"] = "pass_gate_noise_floor_calibrated"
            exit_code = 0
        else:
            record["status"] = "fail_amended_gate"
            exit_code = 1
    except Exception as exc:
        record["status"] = "error"
        record["deflation"] = {
            "all_contract_gates_passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 2
    finally:
        record["elapsed_s"] = float(time.perf_counter() - started)
        _write_json(record)

    dfl = record["deflation"] or {}
    print(f"status: {record['status']}")
    for a in dfl.get("anchors", []):
        print(
            f"  {a['id']}: m={a['matrix_shape'][0]} k={len(a['bc_rows_detected'])}"
            f" exact_submatrix={a['elimination']['reduced_is_exact_submatrix']}"
            f" max|dc|={a['spectrum_comparison']['max_abs_dc']:.3e}"
            f" qz_floor={a['qz_noise_control']['max_abs_dc']:.3e}"
            f" reduced_B_invertible={a['reduced_B']['invertible']}"
        )
    print(f"xgeev verdict: {record['xgeev'].get('verdict')}")
    print(f"JSON: {OUT_JSON}")
    return exit_code


if __name__ == "__main__":
    if XGEEV_CHILD_FLAG in sys.argv:
        idx = sys.argv.index(XGEEV_CHILD_FLAG)
        raise SystemExit(
            xgeev_child_main(Path(sys.argv[idx + 1]), sys.argv[idx + 2])
        )
    raise SystemExit(main())
