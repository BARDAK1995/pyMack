"""Slice 00 benchmark: batched complex LU and triangular solves on CuPy.

Kill criterion K1 for pyMack-GPU: does equilibrated batched complex64 LU reach
~1 TFLOPS effective at n=645 on the RTX 4070 Ti under Windows/WDDM?

This script is intentionally standalone. It imports no pyMack modules and writes
one JSON report under docs/gpu/benchmarks/.

Measurement notes (audited 2026-07-04 against the installed CuPy 14.1.1):
- Low-level cuBLAS bindings live in ``cupy_backends.cuda.libs.cublas``
  (``import cupy.cuda.cublas`` fails in CuPy 14; the lazy attribute
  ``cupy.cuda.cublas`` still resolves to the backend module). Backend modules
  MUST be reached via package attribute access — the package __getattr__
  preloads the NVIDIA wheel DLL via cuda.pathfinder; a direct submodule
  import dies with "DLL load failed" on Windows.
- ``getrfBatched`` writes its info to a DEVICE int array; ``getrsBatched``
  writes its info to a HOST int pointer (see cupy/cublas.py batched_gesv).
- CuPy 14.1.1 raises NotImplementedError when a cuBLAS API is called during
  stream capture, so the CUDA-graph experiment additionally attempts a direct
  ctypes call into cublas64_12.dll (bypassing the wrapper guard) and always
  measures a capturable proxy kernel loop to quantify WDDM launch overhead.
- Matrices are filled with uniform random complex entries WITHOUT diagonal
  dominance so that partial pivoting performs genuine row swaps; correctness
  of every configuration is attested by getrf info == 0 plus an FP64 residual
  check recorded in the JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_JSON = ROOT / "docs" / "gpu" / "benchmarks" / "bench_batched_lu.json"
N_VALUES = (516, 645, 804, 1005, 1285)
BATCH_VALUES = (16, 64, 256)
DTYPES = ("complex64", "complex128")
DTYPE_ITEMSIZES = {"complex64": 8, "complex128": 16}
SOLVE_RHS_VALUES = (1, 16, 32)
SOLVE_TRANS_VALUES = ("N", "C")
WARMUP_REPS = 2
TIMED_REPS = 5
SOLVE_TARGET_WINDOW_MS = 3.0
SOLVE_MAX_ITERS = 64
VRAM_HEADROOM_BYTES = 512 * 1024**2
GRAPH_LOOP_ITERS = 8
GRAPH_PROXY_LAUNCHES = 256
GPU_BUSY_MAX_WAITS = 4
GPU_BUSY_WAIT_S = 30.0
# gate attestation: worst acceptable relative FP64 solve residual for the
# gate config (complex64; measured worst on this card is ~3e-3 at n=1285)
GATE_RESIDUAL_CEILING = 1e-2


def log(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot JSON serialize {type(obj)!r}")


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=30
        )
        return {
            "command": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"command": args, "error_type": type(exc).__name__, "error": str(exc)}


def safe_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def nvidia_smi_record() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,driver_model.current,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    out = run_command(query)
    record: dict[str, Any] = {"query": out}
    if out.get("returncode") == 0 and out.get("stdout"):
        parts = [part.strip() for part in out["stdout"].splitlines()[0].split(",")]
        if len(parts) >= 5:
            record.update(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "driver_model": parts[2],
                    "memory_total_mib": safe_int(parts[3]),
                    "compute_cap": parts[4],
                }
            )
    return record


def other_python_compute_processes() -> list[dict[str, Any]]:
    """List python-like processes on the GPU other than this one.

    Another lane may briefly probe the GPU (e.g. a light cusolver probe); timing
    through it would corrupt the measurement. WDDM drivers report many graphics
    clients under compute-apps, so filter to python/conda executables only.
    """
    out = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ]
    )
    procs: list[dict[str, Any]] = []
    if out.get("returncode") != 0 or not out.get("stdout"):
        return procs
    own_pid = os.getpid()
    for line in out["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        pid = safe_int(parts[0])
        name = parts[1].lower()
        if pid == own_pid:
            continue
        if "python" in name or "conda" in name:
            procs.append({"pid": pid, "process_name": parts[1]})
    return procs


def wait_for_idle_gpu(busy_log: list[dict[str, Any]], context: str) -> None:
    for attempt in range(GPU_BUSY_MAX_WAITS + 1):
        procs = other_python_compute_processes()
        if not procs:
            if attempt > 0:
                busy_log.append(
                    {
                        "context": context,
                        "at_utc": utc_now(),
                        "resolved_after_waits": attempt,
                    }
                )
            return
        if attempt == GPU_BUSY_MAX_WAITS:
            busy_log.append(
                {
                    "context": context,
                    "at_utc": utc_now(),
                    "timed_while_busy": True,
                    "processes": procs,
                }
            )
            log(f"WARNING: timing '{context}' with busy GPU: {procs}")
            return
        log(
            f"GPU busy before '{context}' ({procs}); waiting "
            f"{GPU_BUSY_WAIT_S:.0f}s (attempt {attempt + 1}/{GPU_BUSY_MAX_WAITS})"
        )
        time.sleep(GPU_BUSY_WAIT_S)


def environment_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "nvidia_smi": nvidia_smi_record(),
        "sysmem_policy": args.sysmem_policy,
        "numbers_provisional": args.sysmem_policy != "prefer-no-fallback",
        "sysmem_policy_note": (
            "NVCP 'CUDA - Sysmem Fallback Policy' state at run time. Under "
            "'unknown' the numbers are PROVISIONAL: a sysmem spill can only "
            "deflate measured throughput, so a PASS is a safe PASS, while a "
            "marginal FAIL (0.5-1.0 TFLOPS) requires a re-run after the user "
            "confirms 'Prefer No Sysmem Fallback' before being declared a K1 kill."
        ),
        "packages": {
            "cupy": dist_version("cupy"),
            "cupy-cuda12x": dist_version("cupy-cuda12x"),
            "numpy": dist_version("numpy"),
            "scipy": dist_version("scipy"),
            "pytest": dist_version("pytest"),
            "nvidia-cublas-cu12": dist_version("nvidia-cublas-cu12"),
            "nvidia-cusolver-cu12": dist_version("nvidia-cusolver-cu12"),
            "nvidia-cuda-runtime-cu12": dist_version("nvidia-cuda-runtime-cu12"),
            "nvidia-cuda-nvrtc-cu12": dist_version("nvidia-cuda-nvrtc-cu12"),
        },
    }


def import_cupy() -> tuple[Any | None, dict[str, Any] | None]:
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:
        return None, {"error_type": type(exc).__name__, "error": str(exc)}
    return cp, None


def cuda_version_string(value: int) -> str:
    return f"{value // 1000}.{(value % 1000) // 10}.{value % 10}"


def decode_cuda_bytes(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def cupy_environment(cp: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"cupy_version": getattr(cp, "__version__", None)}
    for name, func in (
        ("cuda_runtime_version", cp.cuda.runtime.runtimeGetVersion),
        ("cuda_driver_version", cp.cuda.runtime.driverGetVersion),
    ):
        try:
            value = func()
            record[name] = value
            record[f"{name}_string"] = cuda_version_string(value)
        except Exception as exc:  # pragma: no cover
            record[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        record["device"] = {
            "id": int(device.id),
            "name": decode_cuda_bytes(props.get("name")),
            "compute_capability": [
                int(props.get("major", -1)),
                int(props.get("minor", -1)),
            ],
            "total_global_mem_bytes": int(props.get("totalGlobalMem", 0)),
        }
        free, total = cp.cuda.runtime.memGetInfo()
        record["mem_info_at_start"] = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "free_gib": bytes_to_gib(int(free)),
        }
    except Exception as exc:  # pragma: no cover
        record["device_error"] = f"{type(exc).__name__}: {exc}"
    return record


# ---------------------------------------------------------------------------
# Primitive inventory
# ---------------------------------------------------------------------------

def module_inventory(module_name: str) -> dict[str, Any]:
    """Import a module, trying the direct submodule path then the parent
    attribute path.

    The distinction is a real finding on this stack (CuPy 14.1.1, Windows,
    pip wheels): lazy package __getattr__ hooks preload the NVIDIA DLLs via
    cuda.pathfinder, so `from cupy_backends.cuda.libs import cublas` works
    while `import cupy_backends.cuda.libs.cublas` fails with "DLL load
    failed" unless the DLL is already in the process.
    """
    direct_error: str | None = None
    mode = "direct_import"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        direct_error = f"{type(exc).__name__}: {exc}"
        module = None
        if "." in module_name:
            parent_name, leaf = module_name.rsplit(".", 1)
            try:
                module = getattr(importlib.import_module(parent_name), leaf)
                mode = "parent_attribute"
            except Exception:
                module = None
    if module is None:
        return {
            "module": module_name,
            "imported": False,
            "direct_import_error": direct_error,
            "symbols": [],
        }
    record: dict[str, Any] = {
        "module": module_name,
        "imported": True,
        "import_mode": mode,
        "symbols": sorted(dir(module)),
    }
    if direct_error is not None:
        record["direct_import_error"] = direct_error
    return record


def symbols_containing(modules: list[dict[str, Any]], needles: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    lowered = tuple(needle.lower() for needle in needles)
    for module in modules:
        if not module.get("imported"):
            continue
        for name in module.get("symbols", []):
            low = name.lower()
            if all(needle in low for needle in lowered):
                found.append(f"{module['module']}.{name}")
    return sorted(found)


def smoke_eigvalsh(cp: Any) -> dict[str, Any]:
    """Functional smoke: batched Hermitian eigenvalues, checked against numpy."""
    import numpy as np

    record: dict[str, Any] = {"attempted": True}
    try:
        rng = np.random.default_rng(11)
        raw = rng.random((4, 16, 16)) + 1j * rng.random((4, 16, 16))
        herm = ((raw + np.conj(np.transpose(raw, (0, 2, 1)))) / 2).astype(np.complex64)
        got = cp.asnumpy(cp.linalg.eigvalsh(cp.asarray(herm)))
        want = np.linalg.eigvalsh(herm)
        record.update(
            {
                "works_batched": True,
                "max_abs_diff_vs_numpy": float(np.max(np.abs(got - want))),
            }
        )
    except Exception as exc:
        record.update(
            {"works_batched": False, "error_type": type(exc).__name__, "error": str(exc)}
        )
    return record


def smoke_batched_qr(cp: Any) -> dict[str, Any]:
    """Functional smoke: high-level batched QR (low-level geqrfBatched is unwrapped)."""
    import numpy as np

    record: dict[str, Any] = {"attempted": True}
    try:
        rng = np.random.default_rng(13)
        a = (rng.random((4, 16, 16)) + 1j * rng.random((4, 16, 16))).astype(np.complex64)
        q, r = cp.linalg.qr(cp.asarray(a))
        recon = cp.asnumpy(q @ r)
        record.update(
            {
                "cupy_linalg_qr_batched_works": True,
                "max_abs_reconstruction_error": float(np.max(np.abs(recon - a))),
            }
        )
    except Exception as exc:
        record.update(
            {
                "cupy_linalg_qr_batched_works": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return record


def smoke_gesvdj_batched(cp: Any) -> dict[str, Any]:
    """Functional smoke: low-level cusolver cgesvdjBatched at n=16 (limit is 32)."""
    import numpy as np

    record: dict[str, Any] = {"attempted": True}
    params = 0
    try:
        from cupy.cuda import device
        from cupy_backends.cuda.libs import cusolver

        m = n = 16
        bs = 4
        rng = np.random.default_rng(17)
        a_np = (rng.random((bs, n, m)) + 1j * rng.random((bs, n, m))).astype(np.complex64)
        a = cp.asarray(a_np)  # column-major m x n per matrix as seen by cusolver
        s = cp.empty((bs, min(m, n)), dtype=cp.float32)
        u = cp.empty((bs, m, m), dtype=cp.complex64)
        v = cp.empty((bs, n, n), dtype=cp.complex64)
        info = cp.empty((bs,), dtype=cp.int32)
        handle = device.get_cusolver_handle()
        jobz = getattr(cusolver, "CUSOLVER_EIG_MODE_VECTOR", 1)
        params = cusolver.createGesvdjInfo()
        lwork = cusolver.cgesvdjBatched_bufferSize(
            handle, jobz, m, n, a.data.ptr, m, s.data.ptr, u.data.ptr, m,
            v.data.ptr, n, params, bs,
        )
        work = cp.empty((max(int(lwork), 1),), dtype=cp.complex64)
        cusolver.cgesvdjBatched(
            handle, jobz, m, n, a.data.ptr, m, s.data.ptr, u.data.ptr, m,
            v.data.ptr, n, work.data.ptr, int(lwork), info.data.ptr, params, bs,
        )
        cp.cuda.get_current_stream().synchronize()
        got = cp.asnumpy(s)
        # singular values are transpose-invariant, so the layout does not matter
        want = np.stack([np.linalg.svd(a_np[i], compute_uv=False) for i in range(bs)])
        record.update(
            {
                "works": True,
                "info": [int(x) for x in cp.asnumpy(info)],
                "lwork": int(lwork),
                "max_abs_sigma_diff_vs_numpy": float(np.max(np.abs(got - want))),
                "documented_size_limit": "m, n <= 32",
            }
        )
    except Exception as exc:
        record.update({"works": False, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if params:
            try:
                from cupy_backends.cuda.libs import cusolver

                cusolver.destroyGesvdjInfo(params)
            except Exception:
                pass
    return record


def primitive_inventory(cp: Any | None) -> dict[str, Any]:
    if cp is None:
        return {"cupy_imported": False}

    modules = [
        module_inventory("cupy.cuda.cublas"),       # removed in CuPy 14: a finding
        module_inventory("cupy.cuda.cusolver"),     # removed in CuPy 14: a finding
        module_inventory("cupy_backends.cuda.libs.cublas"),
        module_inventory("cupy_backends.cuda.libs.cusolver"),
    ]

    def present(*needles: str) -> dict[str, Any]:
        symbols = symbols_containing(modules, needles)
        return {"present": bool(symbols), "symbols": symbols}

    getrf_getrs = sorted(
        set(
            symbol
            for prefix in ("c", "z")
            for suffix in ("getrfBatched", "getrsBatched")
            for symbol in symbols_containing(modules, (f"{prefix}{suffix}",))
        )
    )
    inventory: dict[str, Any] = {
        "cupy_imported": True,
        "low_level_getrfBatched_getrsBatched": {
            "present": all(
                any(symbol.endswith(f".{prefix}{suffix}") for symbol in getrf_getrs)
                for prefix in ("c", "z")
                for suffix in ("getrfBatched", "getrsBatched")
            ),
            "symbols": getrf_getrs,
            "functional_note": "exercised directly by this benchmark",
        },
        "gesvdjBatched": {
            **present("gesvdj", "batched"),
            "functional_smoke": smoke_gesvdj_batched(cp),
        },
        "batched_geqrf": {
            **present("geqrfbatched"),
            "single_matrix_geqrf": present("geqrf"),
            "note": (
                "cgeqrfBatched/zgeqrfBatched are NOT wrapped by "
                "cupy_backends.cuda.libs.cublas in CuPy 14.1.1 (present in the "
                "cuBLAS C API but unreachable without ctypes); only "
                "single-matrix cusolver geqrf wrappers exist"
            ),
            "high_level_qr_smoke": smoke_batched_qr(cp),
        },
        "eigvalsh": {
            "present": bool(hasattr(cp.linalg, "eigvalsh")),
            "symbols": ["cupy.linalg.eigvalsh"] if hasattr(cp.linalg, "eigvalsh") else [],
            "functional_smoke": smoke_eigvalsh(cp),
        },
        "gemmBatched": present("gemm", "batched"),
        "trsmBatched": present("trsm", "batched"),
        "xgeev": {
            **present("xgeev"),
            "note": "cusolverDnXgeev binding present; probed further in slice 03",
        },
        "modules": [
            {key: value for key, value in module.items() if key != "symbols"}
            | {"symbol_count": len(module.get("symbols", []))}
            for module in modules
        ],
    }
    return inventory


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

def estimate_peak_bytes(n: int, batch: int, itemsize: int, max_rhs: int = 32) -> int:
    matrix = batch * n * n * itemsize
    rhs = batch * n * max_rhs * itemsize
    pointer_arrays = 4 * batch * 8
    pivots = batch * n * 4
    infos = 2 * batch * 4
    safety_scratch = 256 * 1024**2
    return 2 * matrix + 2 * rhs + pointer_arrays + pivots + infos + safety_scratch


def bytes_to_gib(value: int) -> float:
    return value / 1024**3


def lu_flops(n: int, batch: int) -> float:
    # complex LU: ~n^3/3 complex multiplies (6 real flops) + n^3/3 complex adds
    # (2 real flops) per matrix -> (8/3) n^3 real flops.
    return (8.0 / 3.0) * n**3 * batch


def getrs_flops(n: int, batch: int, nrhs: int) -> float:
    # two triangular solves per RHS: n^2/2 complex mul + n^2/2 complex add each
    # -> 8 n^2 real flops per RHS per matrix.
    return 8.0 * n**2 * nrhs * batch


def skipped_for_vram(
    dtype_name: str, n: int, batch: int, itemsize: int, budget_bytes: int
) -> dict[str, Any] | None:
    estimate = estimate_peak_bytes(n, batch, itemsize)
    if estimate <= budget_bytes:
        return None
    return {
        "dtype": dtype_name,
        "n": n,
        "batch": batch,
        "estimated_memory_bytes": estimate,
        "estimated_memory_gib": bytes_to_gib(estimate),
        "budget_bytes": budget_bytes,
        "budget_gib": bytes_to_gib(budget_bytes),
        "reason": "estimated peak allocation exceeds free-VRAM budget",
    }


# ---------------------------------------------------------------------------
# Batched LU runner (audited against CuPy 14.1.1 / cupy/cublas.py batched_gesv)
# ---------------------------------------------------------------------------

class BatchedLuRunner:
    def __init__(self, cp: Any, dtype_name: str, n: int, batch: int):
        import numpy as np

        self.cp = cp
        self.np = np
        self.dtype_name = dtype_name
        self.n = n
        self.batch = batch
        self.dtype = cp.dtype(dtype_name)
        self.real_dtype = cp.float32 if dtype_name == "complex64" else cp.float64
        prefix = "c" if dtype_name == "complex64" else "z"
        # Access via package attribute, NOT importlib.import_module on the
        # dotted name: cupy_backends/cuda/libs/__init__.__getattr__ preloads
        # the NVIDIA wheel DLL through cuda.pathfinder; a direct submodule
        # import bypasses that and dies with "DLL load failed" on Windows
        # unless the DLL is already loaded (measured, CuPy 14.1.1).
        self.cublas = getattr(
            importlib.import_module("cupy_backends.cuda.libs"), "cublas"
        )
        device = importlib.import_module("cupy.cuda.device")
        self.handle = device.get_cublas_handle()
        self.getrf_func = getattr(self.cublas, f"{prefix}getrfBatched")
        self.getrs_func = getattr(self.cublas, f"{prefix}getrsBatched")
        self.op_n = self.cublas.CUBLAS_OP_N
        self.op_c = self.cublas.CUBLAS_OP_C
        # getrsBatched reports parameter errors through a HOST int pointer
        # (cuBLAS docs; confirmed against cupy/cublas.py batched_gesv).
        self.host_info = np.zeros((1,), dtype=np.int32)

    def bind_stream(self, stream_ptr: int) -> None:
        self.cublas.setStream(self.handle, stream_ptr)

    def alloc_matrix(self) -> Any:
        return self.cp.empty((self.batch, self.n, self.n), dtype=self.dtype, order="C")

    def alloc_rhs(self, nrhs: int) -> Any:
        # C-order (batch, nrhs, n): python row j is column j of the column-major
        # n x nrhs cuBLAS matrix, ldb = n, contiguous per batch entry.
        return self.cp.empty((self.batch, nrhs, self.n), dtype=self.dtype, order="C")

    def fill_random(self, array: Any, seed: int) -> None:
        # Fill the interleaved re/im float view with uniform values in
        # [-0.5, 0.5): no diagonal dominance, so pivoting stays honest.
        #
        # MEASURED PITFALL (2026-07-04, CuPy 14.1.1): Generator.random(out=view)
        # on a float view of a complex array silently writes a degenerate
        # pattern (values duplicated 4x, real == imag) that makes every matrix
        # exactly singular. Allocate with the generator and copy in instead,
        # chunked over the batch axis to bound temporary memory.
        cp = self.cp
        flat = array.view(self.real_dtype)  # (batch, n, 2n), C-contiguous
        rng = cp.random.default_rng(seed)
        per_matrix_bytes = flat.shape[1] * flat.shape[2] * flat.dtype.itemsize
        chunk = max(1, min(self.batch, (256 * 1024**2) // max(per_matrix_bytes, 1)))
        for i0 in range(0, self.batch, chunk):
            i1 = min(self.batch, i0 + chunk)
            vals = rng.random(
                (i1 - i0, flat.shape[1], flat.shape[2]), dtype=self.real_dtype
            )
            vals -= 0.5
            flat[i0:i1] = vals
        del vals

    def pointer_array(self, array: Any) -> Any:
        cp = self.cp
        step = array.strides[0]
        start = array.data.ptr
        return cp.arange(start, start + step * self.batch, step, dtype=cp.uintp)

    def getrf(self, a_ptrs: Any, pivots: Any, dev_info: Any) -> None:
        self.getrf_func(
            self.handle, self.n, a_ptrs.data.ptr, self.n,
            pivots.data.ptr, dev_info.data.ptr, self.batch,
        )

    def getrs(self, trans: str, nrhs: int, a_ptrs: Any, pivots: Any, b_ptrs: Any) -> None:
        trans_code = self.op_c if trans == "C" else self.op_n
        self.getrs_func(
            self.handle, trans_code, self.n, nrhs, a_ptrs.data.ptr, self.n,
            pivots.data.ptr, b_ptrs.data.ptr, self.n,
            self.host_info.ctypes.data, self.batch,
        )
        if int(self.host_info[0]) != 0:
            raise RuntimeError(f"getrsBatched host info = {int(self.host_info[0])}")


# ---------------------------------------------------------------------------
# Timing helpers (CUDA events, warmup, median of >= 5)
# ---------------------------------------------------------------------------

def event_time_once_ms(cp: Any, stream: Any, operation) -> float:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    operation()
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end))


def median_event_time_ms(
    cp: Any,
    operation,
    setup=None,
    reps: int = TIMED_REPS,
    warmup: int = WARMUP_REPS,
    iters: int = 1,
) -> dict[str, Any]:
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        if setup is not None:
            setup()
            stream.synchronize()
        operation()
        stream.synchronize()

    samples: list[float] = []
    for _ in range(reps):
        if setup is not None:
            setup()
            stream.synchronize()

        def block() -> None:
            for _ in range(iters):
                operation()

        samples.append(event_time_once_ms(cp, stream, block) / iters)
    return {
        "median_ms": float(statistics.median(samples)),
        "samples_ms": samples,
        "timed_repetitions": reps,
        "warmup_repetitions": warmup,
        "iters_per_sample": iters,
        "timer": "cupy.cuda.Event",
    }


def choose_solve_iters(cp: Any, stream: Any, operation) -> int:
    est_ms = event_time_once_ms(cp, stream, operation)
    if est_ms >= SOLVE_TARGET_WINDOW_MS:
        return 1
    return max(1, min(SOLVE_MAX_ITERS, math.ceil(SOLVE_TARGET_WINDOW_MS / max(est_ms, 1e-3))))


# ---------------------------------------------------------------------------
# Correctness attestation (FP64 residuals on sampled batch entries)
# ---------------------------------------------------------------------------

def solve_residual(
    cp: Any, runner: BatchedLuRunner, a_base: Any, b_base: Any, x: Any, trans: str
) -> float:
    """Max relative FP64 residual over sampled batch entries.

    cuBLAS sees the C-order python matrix a_base[i] as its transpose M = a[i].T,
    so trans='N' solves a[i].T x = b and trans='C' solves conj(a[i]) x = b.
    """
    worst = 0.0
    for i in (0, runner.batch - 1):
        a64 = a_base[i].astype(cp.complex128)
        x64 = x[i].astype(cp.complex128)  # (nrhs, n): row j is solution column j
        b64 = b_base[i].astype(cp.complex128)
        if trans == "C":
            r = cp.conj(a64) @ x64.T - b64.T
        else:
            r = a64.T @ x64.T - b64.T
        num = float(cp.linalg.norm(r))
        den = float(cp.linalg.norm(b64)) + 1e-300
        worst = max(worst, num / den)
    return worst


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def benchmark_config(cp: Any, dtype_name: str, n: int, batch: int) -> dict[str, Any]:
    runner = BatchedLuRunner(cp, dtype_name, n, batch)
    stream = cp.cuda.get_current_stream()
    runner.bind_stream(stream.ptr)
    estimate = estimate_peak_bytes(n, batch, runner.dtype.itemsize)
    result: dict[str, Any] = {
        "dtype": dtype_name,
        "n": n,
        "batch": batch,
        "memory_estimate_bytes": estimate,
        "memory_estimate_gib": bytes_to_gib(estimate),
    }

    cp.get_default_memory_pool().free_all_blocks()
    a_base = runner.alloc_matrix()
    a_work = runner.alloc_matrix()
    runner.fill_random(a_base, seed=17 + n + batch)
    a_ptrs = runner.pointer_array(a_work)
    pivots = cp.empty((batch, n), dtype=cp.int32)
    dev_info = cp.empty((batch,), dtype=cp.int32)
    stream.synchronize()

    def setup_lu() -> None:
        cp.copyto(a_work, a_base)

    def lu_once() -> None:
        runner.getrf(a_ptrs, pivots, dev_info)

    lu_timing = median_event_time_ms(cp, lu_once, setup=setup_lu)
    stream.synchronize()
    info_nonzero = int(cp.count_nonzero(dev_info).get())
    median_lu_s = lu_timing["median_ms"] / 1000.0
    lu_gflops = lu_flops(n, batch) / median_lu_s / 1.0e9 if median_lu_s > 0 else math.inf
    result["lu"] = {
        **lu_timing,
        "flop_model": "complex LU factorization: (8/3)*n^3 real flops per matrix",
        "gflops": lu_gflops,
        "tflops": lu_gflops / 1000.0,
        "getrf_info_nonzero_count": info_nonzero,
    }
    log(
        f"{dtype_name} n={n} batch={batch}: LU {lu_timing['median_ms']:.2f} ms "
        f"-> {lu_gflops / 1000.0:.3f} TFLOPS (info!=0: {info_nonzero})"
    )

    # refactor once so a_work holds valid LU factors for the solve section
    cp.copyto(a_work, a_base)
    runner.getrf(a_ptrs, pivots, dev_info)
    stream.synchronize()

    solves: list[dict[str, Any]] = []
    for nrhs in SOLVE_RHS_VALUES:
        b_base = runner.alloc_rhs(nrhs)
        b_work = runner.alloc_rhs(nrhs)
        runner.fill_random(b_base, seed=101 + nrhs + n)
        b_ptrs = runner.pointer_array(b_work)
        stream.synchronize()

        for trans in SOLVE_TRANS_VALUES:
            def setup_solve() -> None:
                cp.copyto(b_work, b_base)

            def solve_once(trans_value: str = trans, k: int = nrhs) -> None:
                runner.getrs(trans_value, k, a_ptrs, pivots, b_ptrs)

            setup_solve()
            stream.synchronize()
            solve_once()  # untimed warm call, also feeds the iters estimate
            stream.synchronize()
            iters = choose_solve_iters(cp, stream, solve_once)
            timing = median_event_time_ms(cp, solve_once, setup=setup_solve, iters=iters)
            stream.synchronize()

            # dedicated correctness pass: fresh RHS -> single solve -> residual
            setup_solve()
            stream.synchronize()
            solve_once()
            stream.synchronize()
            residual = solve_residual(cp, runner, a_base, b_base, b_work, trans)

            median_s = timing["median_ms"] / 1000.0
            gflops = getrs_flops(n, batch, nrhs) / median_s / 1.0e9 if median_s > 0 else math.inf
            solves.append(
                {
                    "nrhs": nrhs,
                    "trans": trans,
                    **timing,
                    "flop_model": "complex LU triangular solves: 8*n^2*nrhs real flops per matrix",
                    "gflops": gflops,
                    "tflops": gflops / 1000.0,
                    "max_rel_residual_fp64_sampled": residual,
                }
            )
        del b_base, b_work, b_ptrs
        cp.get_default_memory_pool().free_all_blocks()

    result["getrs"] = solves
    del a_base, a_work, a_ptrs, pivots, dev_info
    cp.get_default_memory_pool().free_all_blocks()
    return result


# ---------------------------------------------------------------------------
# CUDA graph experiments
# ---------------------------------------------------------------------------

def find_cublas_dll() -> ctypes.WinDLL | None:
    # cuBLAS is already loaded into this process by CuPy; LoadLibrary by name
    # binds to the loaded module. Fall back to the nvidia wheel directory.
    candidates = ["cublas64_12.dll", "cublas64_13.dll"]
    for name in candidates:
        try:
            return ctypes.WinDLL(name)
        except OSError:
            continue
    try:
        import nvidia

        for base in nvidia.__path__:
            for dll in Path(base).glob("cublas/bin/cublas64_*.dll"):
                try:
                    return ctypes.WinDLL(str(dll))
                except OSError:
                    continue
    except Exception:
        pass
    return None


def graph_launch(graph: Any, stream: Any) -> None:
    try:
        graph.launch(stream)
        return
    except TypeError:
        pass
    graph.launch()


def time_graph_replay(
    cp: Any, stream: Any, graph: Any, dev_info: Any
) -> dict[str, Any]:
    for _ in range(WARMUP_REPS):
        graph_launch(graph, stream)
        stream.synchronize()
    graph_samples: list[float] = []
    graph_wall: list[float] = []
    for _ in range(TIMED_REPS):
        wall0 = time.perf_counter()
        graph_samples.append(
            event_time_once_ms(cp, stream, lambda: graph_launch(graph, stream))
        )
        graph_wall.append((time.perf_counter() - wall0) * 1000.0)
    stream.synchronize()
    return {
        "median_event_ms_per_iteration": float(statistics.median(graph_samples))
        / GRAPH_LOOP_ITERS,
        "median_wall_ms_per_iteration": float(statistics.median(graph_wall))
        / GRAPH_LOOP_ITERS,
        "samples_event_ms": graph_samples,
        "getrf_info_nonzero_after_replay": int(cp.count_nonzero(dev_info).get()),
    }


def cuda_graph_experiment(cp: Any, n: int, batch: int) -> dict[str, Any]:
    """Eager vs captured LU+solve loop, plus a capturable launch-overhead proxy.

    CuPy 14.1.1 refuses cuBLAS calls during stream capture (NotImplementedError),
    so three tiers are attempted and all outcomes recorded, in increasing order
    of danger (a broken ctypes-captured graph replay can raise
    cudaErrorIllegalAddress and poison the CUDA context, which is why the whole
    experiment runs in a subprocess):
      1. capture through the CuPy wrapper (expected to be refused),
      2. a pure-CuPy elementwise kernel loop (always capturable) to quantify
         WDDM launch-latency amortization,
      3. capture with direct ctypes calls into cublas64_12.dll (LAST).
    """
    record: dict[str, Any] = {
        "attempted": True,
        "isolation": "subprocess",
        "config": {
            "dtype": "complex64",
            "n": n,
            "batch": batch,
            "nrhs": 32,
            "trans": "N",
            "loop_iterations": GRAPH_LOOP_ITERS,
        },
    }
    runner = BatchedLuRunner(cp, "complex64", n, batch)
    stream = cp.cuda.Stream(non_blocking=True)
    nrhs = 32

    a_base = runner.alloc_matrix()
    a_work = runner.alloc_matrix()
    b_base = runner.alloc_rhs(nrhs)
    b_work = runner.alloc_rhs(nrhs)
    runner.fill_random(a_base, seed=211)
    runner.fill_random(b_base, seed=307)
    a_ptrs = runner.pointer_array(a_work)
    b_ptrs = runner.pointer_array(b_work)
    pivots = cp.empty((batch, n), dtype=cp.int32)
    dev_info = cp.empty((batch,), dtype=cp.int32)

    def one_iteration() -> None:
        cp.copyto(a_work, a_base)
        cp.copyto(b_work, b_base)
        runner.getrf(a_ptrs, pivots, dev_info)
        runner.getrs("N", nrhs, a_ptrs, pivots, b_ptrs)

    def loop_body() -> None:
        for _ in range(GRAPH_LOOP_ITERS):
            one_iteration()

    try:
        with stream:
            runner.bind_stream(stream.ptr)
            loop_body()  # warm eager pass on the same stream (lazy init)
            stream.synchronize()

            eager_samples: list[float] = []
            eager_wall: list[float] = []
            for _ in range(TIMED_REPS):
                wall0 = time.perf_counter()
                eager_samples.append(event_time_once_ms(cp, stream, loop_body))
                eager_wall.append((time.perf_counter() - wall0) * 1000.0)
            record["eager"] = {
                "median_event_ms_per_iteration": float(statistics.median(eager_samples))
                / GRAPH_LOOP_ITERS,
                "median_wall_ms_per_iteration": float(statistics.median(eager_wall))
                / GRAPH_LOOP_ITERS,
                "samples_event_ms": eager_samples,
            }

            # tier 1: capture through the CuPy wrapper
            try:
                stream.begin_capture()
                loop_body()
                graph = stream.end_capture()
                tier1: dict[str, Any] = {"supported": True}
            except Exception as exc:
                tier1 = {
                    "supported": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                graph = None
                try:
                    if getattr(stream, "is_capturing", lambda: False)():
                        stream.end_capture()
                except Exception:
                    pass
            record["cupy_wrapper_capture"] = tier1

            if graph is not None:
                record["captured"] = time_graph_replay(cp, stream, graph, dev_info)

            # tier 2: pure-CuPy launch-overhead proxy (safe, before the ctypes
            # bypass which can poison the context)
            record["launch_overhead_proxy"] = launch_overhead_proxy(cp)

            # tier 3: direct ctypes calls into the cuBLAS DLL, bypassing the
            # CuPy guard; capture AND replay may fail destructively.
            if graph is None:
                tier3 = ctypes_capture_attempt(
                    cp, runner, stream, a_base, a_work, b_base, b_work,
                    a_ptrs, b_ptrs, pivots, dev_info, nrhs,
                )
                graph = tier3.pop("_graph", None)
                record["ctypes_bypass_capture"] = tier3
                if graph is not None:
                    try:
                        record["captured"] = time_graph_replay(
                            cp, stream, graph, dev_info
                        )
                    except Exception as exc:
                        tier3["replay_error_type"] = type(exc).__name__
                        tier3["replay_error"] = str(exc)
                        tier3["replay_note"] = (
                            "capture succeeded but replay failed; a "
                            "cudaErrorIllegalAddress here poisons the CUDA "
                            "context (subprocess isolation contains it)"
                        )

            if "captured" in record and "eager" in record:
                record["speedup_captured_vs_eager_event"] = (
                    record["eager"]["median_event_ms_per_iteration"]
                    / record["captured"]["median_event_ms_per_iteration"]
                )
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        try:
            runner.bind_stream(cp.cuda.Stream.null.ptr)
        except Exception:
            pass
        del a_base, a_work, b_base, b_work, a_ptrs, b_ptrs, pivots, dev_info
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return record


def run_graph_experiment_subprocess(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CUDA-graph experiment in a child process.

    A broken captured-graph replay can raise cudaErrorIllegalAddress, which
    irrecoverably poisons the CUDA context; isolating the experiment protects
    the main sweep results (this happened on the first full run)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--graph-experiment-internal",
        "--gate-tflops", str(args.gate_tflops),
        "--n", str(args.n),
        "--batch", str(args.batch),
        "--sysmem-policy", args.sysmem_policy,
    ]
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=600
        )
    except Exception as exc:
        return {
            "attempted": True,
            "isolation": "subprocess",
            "subprocess_failed": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                record = json.loads(line)
                record["subprocess_returncode"] = proc.returncode
                return record
            except json.JSONDecodeError:
                break
    return {
        "attempted": True,
        "isolation": "subprocess",
        "subprocess_failed": True,
        "subprocess_returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def ctypes_capture_attempt(
    cp: Any,
    runner: BatchedLuRunner,
    stream: Any,
    a_base: Any,
    a_work: Any,
    b_base: Any,
    b_work: Any,
    a_ptrs: Any,
    b_ptrs: Any,
    pivots: Any,
    dev_info: Any,
    nrhs: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {"attempted": True}
    lib = find_cublas_dll()
    if lib is None:
        record.update({"supported": False, "reason": "cublas64_*.dll not loadable"})
        return record
    try:
        c_int, c_void_p = ctypes.c_int, ctypes.c_void_p
        lib.cublasSetStream_v2.argtypes = [c_void_p, c_void_p]
        lib.cublasSetStream_v2.restype = c_int
        lib.cublasCgetrfBatched.argtypes = [
            c_void_p, c_int, c_void_p, c_int, c_void_p, c_void_p, c_int
        ]
        lib.cublasCgetrfBatched.restype = c_int
        lib.cublasCgetrsBatched.argtypes = [
            c_void_p, c_int, c_int, c_int, c_void_p, c_int, c_void_p,
            c_void_p, c_int, c_void_p, c_int,
        ]
        lib.cublasCgetrsBatched.restype = c_int
        handle = c_void_p(runner.handle)
        host_info = ctypes.c_int(0)
        n, batch = runner.n, runner.batch

        status = lib.cublasSetStream_v2(handle, c_void_p(stream.ptr))
        if status != 0:
            record.update({"supported": False, "reason": f"cublasSetStream status {status}"})
            return record

        def one_iteration_ctypes() -> None:
            cp.copyto(a_work, a_base)
            cp.copyto(b_work, b_base)
            s1 = lib.cublasCgetrfBatched(
                handle, n, c_void_p(a_ptrs.data.ptr), n,
                c_void_p(pivots.data.ptr), c_void_p(dev_info.data.ptr), batch,
            )
            s2 = lib.cublasCgetrsBatched(
                handle, 0, n, nrhs, c_void_p(a_ptrs.data.ptr), n,
                c_void_p(pivots.data.ptr), c_void_p(b_ptrs.data.ptr), n,
                ctypes.byref(host_info), batch,
            )
            if s1 != 0 or s2 != 0:
                raise RuntimeError(f"cublas status getrf={s1} getrs={s2}")

        one_iteration_ctypes()  # eager warm pass through the ctypes path
        stream.synchronize()

        stream.begin_capture()
        for _ in range(GRAPH_LOOP_ITERS):
            one_iteration_ctypes()
        graph = stream.end_capture()
        record.update({"supported": True, "_graph": graph})
    except Exception as exc:
        record.update(
            {"supported": False, "error_type": type(exc).__name__, "error": str(exc)}
        )
        try:
            if getattr(stream, "is_capturing", lambda: False)():
                stream.end_capture()
        except Exception:
            pass
    return record


def launch_overhead_proxy(cp: Any) -> dict[str, Any]:
    """Eager vs captured loop of tiny elementwise kernels: pure WDDM launch cost."""
    record: dict[str, Any] = {
        "attempted": True,
        "kernel": "x *= 1.000001 on 1024 float32 elements",
        "launches_per_loop": GRAPH_PROXY_LAUNCHES,
    }
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            x = cp.ones((1024,), dtype=cp.float32)

            def loop_body() -> None:
                for _ in range(GRAPH_PROXY_LAUNCHES):
                    cp.multiply(x, cp.float32(1.000001), out=x)

            loop_body()
            stream.synchronize()
            eager_event: list[float] = []
            eager_wall: list[float] = []
            for _ in range(TIMED_REPS):
                wall0 = time.perf_counter()
                eager_event.append(event_time_once_ms(cp, stream, loop_body))
                eager_wall.append((time.perf_counter() - wall0) * 1000.0)

            stream.begin_capture()
            loop_body()
            graph = stream.end_capture()
            for _ in range(WARMUP_REPS):
                graph_launch(graph, stream)
                stream.synchronize()
            graph_event: list[float] = []
            graph_wall: list[float] = []
            for _ in range(TIMED_REPS):
                wall0 = time.perf_counter()
                graph_event.append(
                    event_time_once_ms(cp, stream, lambda: graph_launch(graph, stream))
                )
                graph_wall.append((time.perf_counter() - wall0) * 1000.0)

            eager_us = float(statistics.median(eager_event)) * 1000.0 / GRAPH_PROXY_LAUNCHES
            graph_us = float(statistics.median(graph_event)) * 1000.0 / GRAPH_PROXY_LAUNCHES
            record.update(
                {
                    "supported": True,
                    "eager_event_us_per_launch": eager_us,
                    "eager_wall_us_per_launch": float(statistics.median(eager_wall))
                    * 1000.0 / GRAPH_PROXY_LAUNCHES,
                    "captured_event_us_per_launch": graph_us,
                    "captured_wall_us_per_launch": float(statistics.median(graph_wall))
                    * 1000.0 / GRAPH_PROXY_LAUNCHES,
                    "speedup_event": eager_us / graph_us if graph_us > 0 else math.inf,
                }
            )
    except Exception as exc:
        record.update(
            {"supported": False, "error_type": type(exc).__name__, "error": str(exc)}
        )
    finally:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    return record


# ---------------------------------------------------------------------------
# Sweep + tables + gate
# ---------------------------------------------------------------------------

def make_empty_tables() -> dict[str, Any]:
    return {dtype: {"lu": {}, "getrs": {}} for dtype in DTYPES}


def add_to_tables(tables: dict[str, Any], result: dict[str, Any]) -> None:
    dtype = result["dtype"]
    key = f"n{result['n']}_batch{result['batch']}"
    if "lu" in result:
        tables[dtype]["lu"][key] = {
            "gflops": result["lu"]["gflops"],
            "tflops": result["lu"]["tflops"],
            "median_ms": result["lu"]["median_ms"],
        }
    for solve in result.get("getrs", []):
        solve_key = f"{key}_L{solve['nrhs']}_trans{solve['trans']}"
        tables[dtype]["getrs"][solve_key] = {
            "gflops": solve["gflops"],
            "tflops": solve["tflops"],
            "median_ms": solve["median_ms"],
        }


def sweep_grid(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    grid = [
        (dtype_name, n, batch)
        for dtype_name in DTYPES
        for n in N_VALUES
        for batch in BATCH_VALUES
    ]
    gate_config = ("complex64", args.n, args.batch)
    if gate_config not in grid:
        grid.insert(0, gate_config)
    return grid


def run_benchmarks(cp: Any, args: argparse.Namespace, record: dict[str, Any]) -> None:
    free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    budget_bytes = max(free_bytes - VRAM_HEADROOM_BYTES, 0)
    record["vram_budget"] = {
        "free_bytes_at_start": free_bytes,
        "headroom_bytes": VRAM_HEADROOM_BYTES,
        "budget_bytes": budget_bytes,
        "budget_gib": bytes_to_gib(budget_bytes),
    }

    tables = make_empty_tables()
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    busy_log: list[dict[str, Any]] = []

    for dtype_name, n, batch in sweep_grid(args):
        itemsize = DTYPE_ITEMSIZES[dtype_name]
        skip = skipped_for_vram(dtype_name, n, batch, itemsize, budget_bytes)
        if skip is not None:
            skipped.append(skip)
            log(f"SKIP {dtype_name} n={n} batch={batch}: {skip['estimated_memory_gib']:.2f} GiB > budget")
            continue
        wait_for_idle_gpu(busy_log, f"{dtype_name} n={n} batch={batch}")
        try:
            result = benchmark_config(cp, dtype_name, n, batch)
            results.append(result)
            add_to_tables(tables, result)
        except cp.cuda.memory.OutOfMemoryError as exc:
            skipped.append(
                {
                    "dtype": dtype_name,
                    "n": n,
                    "batch": batch,
                    "estimated_memory_bytes": estimate_peak_bytes(n, batch, itemsize),
                    "estimated_memory_gib": bytes_to_gib(
                        estimate_peak_bytes(n, batch, itemsize)
                    ),
                    "budget_bytes": budget_bytes,
                    "budget_gib": bytes_to_gib(budget_bytes),
                    "reason": f"runtime CuPy OutOfMemoryError: {exc}",
                }
            )
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            errors.append(
                {
                    "dtype": dtype_name,
                    "n": n,
                    "batch": batch,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            log(f"ERROR {dtype_name} n={n} batch={batch}: {type(exc).__name__}: {exc}")
            cp.get_default_memory_pool().free_all_blocks()

    record["results"] = results
    record["tables"] = tables
    record["skipped_configs"] = skipped
    record["benchmark_errors"] = errors
    record["gpu_busy_checks"] = busy_log

    # settle the gate and persist BEFORE the CUDA-graph experiment so a
    # context-poisoning graph replay can never lose the sweep results
    record["gate"] = gate_record(args, results, skipped, errors)
    write_json(args.output, record)

    wait_for_idle_gpu(busy_log, "cuda_graph_experiment")
    record["cuda_graph"] = run_graph_experiment_subprocess(args)


def gate_record(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "dtype": "complex64",
        "n": args.n,
        "batch": args.batch,
        "threshold_tflops": args.gate_tflops,
        "sysmem_policy": args.sysmem_policy,
        "measured_tflops": None,
        "passed": False,
        "verdict": "FAIL",
    }
    for result in results:
        if (
            result["dtype"] == "complex64"
            and result["n"] == args.n
            and result["batch"] == args.batch
        ):
            measured = result["lu"]["tflops"]
            info_nonzero = result["lu"]["getrf_info_nonzero_count"]
            residuals = [
                solve["max_rel_residual_fp64_sampled"]
                for solve in result.get("getrs", [])
                if "max_rel_residual_fp64_sampled" in solve
            ]
            worst_residual = max(residuals) if residuals else None
            gate["measured_tflops"] = measured
            gate["measured_gflops"] = result["lu"]["gflops"]
            gate["median_ms"] = result["lu"]["median_ms"]
            gate["getrf_info_nonzero_count"] = info_nonzero
            gate["worst_solve_residual_fp64"] = worst_residual
            gate["residual_ceiling"] = GATE_RESIDUAL_CEILING

            # A throughput number only counts if the work it timed was real:
            # every getrf must have factored (info == 0) and, when solve
            # residuals were recorded for the gate config, they must be sane.
            attestation_failures: list[str] = []
            if info_nonzero != 0:
                attestation_failures.append(
                    f"getrf info != 0 for {info_nonzero} of {args.batch} matrices"
                )
            if worst_residual is not None and not (
                worst_residual <= GATE_RESIDUAL_CEILING
            ):
                # `not (x <= ceiling)` also trips on NaN residuals
                attestation_failures.append(
                    f"worst FP64 solve residual {worst_residual!r} exceeds "
                    f"ceiling {GATE_RESIDUAL_CEILING}"
                )
            gate["attestations_ok"] = not attestation_failures
            if attestation_failures:
                gate["attestation_failures"] = attestation_failures

            gate["passed"] = (
                bool(measured >= args.gate_tflops) and not attestation_failures
            )
            if gate["passed"]:
                gate["verdict"] = "PASS"
            elif attestation_failures:
                # never PASS or MARGINAL on an unattested measurement
                gate["verdict"] = "FAIL"
                gate["reason"] = (
                    "correctness attestation failed for the gate config: "
                    + "; ".join(attestation_failures)
                )
            elif (
                measured >= 0.5 * args.gate_tflops
                and args.sysmem_policy != "prefer-no-fallback"
            ):
                # a sysmem spill can only deflate the number: a marginal miss
                # under an unconfirmed NVCP policy is not yet a K1 kill.
                gate["verdict"] = "MARGINAL_FAIL_NEEDS_NVCP_CONFIRMATION"
                gate["reason"] = (
                    "measured between 0.5x and 1.0x of the gate under sysmem "
                    f"policy '{args.sysmem_policy}'; re-run after the user sets "
                    "NVCP 'Prefer No Sysmem Fallback' before declaring K1 dead"
                )
            else:
                gate["verdict"] = "FAIL"
            return gate
    for skip in skipped:
        if skip["dtype"] == "complex64" and skip["n"] == args.n and skip["batch"] == args.batch:
            gate["reason"] = "gate config was skipped"
            gate["skip_record"] = skip
            return gate
    for error in errors:
        if error["dtype"] == "complex64" and error["n"] == args.n and error["batch"] == args.batch:
            gate["reason"] = "gate config errored"
            gate["error_record"] = error
            return gate
    gate["reason"] = "gate config was not found in results, skips, or errors"
    return gate


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-tflops", type=float, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--sysmem-policy",
        choices=("prefer-no-fallback", "default", "unknown"),
        required=True,
        help=(
            "NVCP 'CUDA - Sysmem Fallback Policy' state (provenance). A spill "
            "can only deflate numbers: PASS under 'unknown' is a safe PASS; a "
            "marginal FAIL under 'unknown' needs NVCP confirmation."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--graph-experiment-internal",
        action="store_true",
        help=argparse.SUPPRESS,  # child-process mode used for context isolation
    )
    return parser.parse_args(argv)


def graph_experiment_child(args: argparse.Namespace) -> int:
    """Child-process entry: run only the CUDA-graph experiment, print JSON."""
    cp, import_error = import_cupy()
    if cp is None:
        print(json.dumps({"attempted": True, "isolation": "subprocess",
                          "cupy_import_failed": True, **(import_error or {})}))
        return 1
    try:
        record = cuda_graph_experiment(cp, args.n, args.batch)
    except Exception as exc:
        record = {
            "attempted": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    record["isolation"] = "subprocess"
    print(json.dumps(record, default=json_default))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.graph_experiment_internal:
        return graph_experiment_child(args)
    command = " ".join([Path(sys.executable).name, sys.argv[0], *sys.argv[1:]])
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": "pymack-gpu-slice-00-batched-lu-benchmark-v2",
        "slice": "00-gpu-lu-spike",
        "generated_at_utc": utc_now(),
        "command": command,
        "requested_command": (
            "python scripts/gpu_bench/bench_batched_lu.py "
            f"--gate-tflops {args.gate_tflops} --n {args.n} --batch {args.batch} "
            f"--sysmem-policy {args.sysmem_policy}"
        ),
        "environment": environment_record(args),
        "cupy": None,
        "primitive_inventory": None,
        "results": [],
        "tables": make_empty_tables(),
        "skipped_configs": [],
        "benchmark_errors": [],
        "cuda_graph": {"attempted": False},
        "gate": {
            "dtype": "complex64",
            "n": args.n,
            "batch": args.batch,
            "threshold_tflops": args.gate_tflops,
            "measured_tflops": None,
            "passed": False,
            "verdict": "FAIL",
        },
        "status": "running",
    }

    exit_code = 1
    try:
        cp, import_error = import_cupy()
        if cp is None:
            record["status"] = "blocked"
            record["cupy"] = {"imported": False, **(import_error or {})}
            record["primitive_inventory"] = primitive_inventory(None)
            record["gate"]["reason"] = "CuPy import failed; benchmark could not run"
            return exit_code

        record["cupy"] = {"imported": True, **cupy_environment(cp)}
        record["primitive_inventory"] = primitive_inventory(cp)
        run_benchmarks(cp, args, record)
        record["status"] = record["gate"]["verdict"].lower()
        exit_code = 0 if record["gate"]["passed"] else 1
        return exit_code
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"error_type": type(exc).__name__, "error": str(exc)}
        record["gate"]["reason"] = "benchmark script raised an unexpected error"
        return exit_code
    finally:
        record["elapsed_s"] = float(time.perf_counter() - started)
        write_json(args.output, record)
        log(
            f"gate: {record['gate'].get('verdict')} "
            f"measured={record['gate'].get('measured_tflops')} TFLOPS "
            f"(threshold {args.gate_tflops}); JSON -> {args.output}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
