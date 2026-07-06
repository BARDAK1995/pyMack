"""VRAM-adaptive tiling, pinned double-buffered staging, and the CUDA-graph
decision for the pyMack GPU sweep.

The scheduler resolves ONE tile size for the whole sweep over
``(points x contour nodes)`` from the free VRAM and the per-lane working set,
records it in ``meta`` (so results carry the tile size the determinism contract
is keyed on), stages host inputs through pinned memory on two alternating
streams (double-buffered so tile ``i+1``'s upload overlaps tile ``i``'s
compute), and records the CUDA-graph decision.

CUDA graphs (slice-00 inheritance).  CuPy's wrapper REFUSES cuBLAS capture
(``NotImplementedError``); only a direct-ctypes cuBLAS path captures and replays.
Graphs amortise ~1.00x on the compute-bound LU loop and ~17.7x on tiny-launch
phases, and a broken captured replay once raised ``cudaErrorIllegalAddress`` and
poisoned the CUDA context.  Therefore: the default inner loop runs EAGER (the
decision is recorded with a measured launch-overhead proxy that is safe to run
in-process); the risky ctypes cuBLAS capture is available (:meth:`capture_inner_loop`)
and is exercised only subprocess-isolated (:func:`probe_ctypes_graph`), never in
the caller's process.
"""

from __future__ import annotations

import ctypes
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "TilePlan",
    "TileScheduler",
    "probe_ctypes_graph",
]


# ---------------------------------------------------------------------------
# Tile plan
# ---------------------------------------------------------------------------


@dataclass
class TilePlan:
    tile_size: int
    n_tiles: int
    total_lanes: int
    n: int
    bytes_per_lane: int
    shared_bytes: int
    budget_bytes: int
    meta: dict = field(default_factory=dict)

    def slices(self):
        for start in range(0, self.total_lanes, self.tile_size):
            yield start, min(start + self.tile_size, self.total_lanes)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TileScheduler:
    """Resolve tiling once per sweep and drive pinned, double-buffered tiles."""

    def __init__(self, ops, config):
        self.ops = ops
        self.config = config
        self._graph_decision = None

    # -- memory model ------------------------------------------------------

    def bytes_per_lane(self, n, nrhs=1):
        """Per-lane device working set for the c64 factor + FP64 refinement,
        INCLUDING the worst-case (100%) z128 promotion reserve.

        The base c64 solve holds the equilibrated ``M``, its transpose-stored
        LU buffer, the c64 RHS staging, and the FP64 solution + residual
        temporaries.  Promotion allocates a FRESH complex128 ``Mz`` + ``luz``
        and z128 solve buffers ALONGSIDE the live c64 tile for the promoted
        lanes; this budgets for every lane promoting, so an auto-sized tile
        never OOMs.  (Measured: the earlier c64-only 1.5x model undercounted an
        all-promote tile by ~2.03x and hard-crashed above ~25% promotion.  The
        once-per-sweep tile-size determinism contract forbids mid-sweep tile
        halving, so the safe upper-bound model -- not OOM-retry -- is the fix.)
        A 1.25x factor covers pool fragmentation and pointer/scalar overhead.
        """
        c64_matrix = 8 * n * n              # equilibrated M
        c64_lu = 8 * n * n                  # transpose-stored LU buffer
        c64_rhs = 8 * n * nrhs
        fp64_x = 16 * n * nrhs
        fp64_res = 16 * n * nrhs            # residual temporaries
        z128_matrix = 16 * n * n            # promotion: fresh Mz
        z128_lu = 16 * n * n                # promotion: fresh luz
        z128_rhs = 16 * n * nrhs            # promotion: z128 solve buffers
        pivots = 8 * n + 128                # c64 + z128 pivots/info
        per = (c64_matrix + c64_lu + c64_rhs + fp64_x + fp64_res
               + z128_matrix + z128_lu + z128_rhs + pivots)
        return int(math.ceil(1.25 * per))

    def shared_bytes(self, n, n_terms):
        """Family-shared device memory (c64 + c128 merged term stacks)."""
        return int(n_terms * n * n * (8 + 16))

    def resolve_tile(self, n, n_terms, nrhs=1, free_bytes=None):
        """Resolve the single sweep tile size (lanes).  Records the decision."""
        cp = self.ops.cp
        pool_reusable = 0
        raw_free_bytes = free_bytes
        if free_bytes is None:
            raw_free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
            try:
                pool = cp.get_default_memory_pool()
                pool_reusable = max(
                    int(pool.total_bytes()) - int(pool.used_bytes()), 0
                )
            except Exception:
                pool_reusable = 0
            free_bytes = int(raw_free_bytes) + int(pool_reusable)
        else:
            raw_free_bytes = int(free_bytes)
        budget = max(int(free_bytes) - self.config.vram_headroom_bytes, 0)
        shared = self.shared_bytes(n, n_terms)
        per = self.bytes_per_lane(n, nrhs)
        avail = max(budget - shared, 0)
        auto_tile = max(1, avail // per)
        clamped = False
        env_tile = int(self.config.tile)
        if env_tile > 0:
            # A PYMACK_GPU_TILE override must never silently exceed the VRAM
            # cap: clamp to auto_tile and record the clamp (loud provenance).
            tile = min(env_tile, int(auto_tile))
            source = "env_override"
            clamped = tile < env_tile
        else:
            tile = int(auto_tile)
            source = "vram_adaptive"
        sizing = {
            "tile_source": source,
            "free_bytes": int(free_bytes),
            "raw_device_free_bytes": int(raw_free_bytes),
            "pool_reusable_bytes": int(pool_reusable),
            "budget_bytes": int(budget),
            "shared_bytes": int(shared),
            "bytes_per_lane": int(per),
            "bytes_per_lane_note": "includes 100% z128 promotion reserve",
            "auto_tile_lanes": int(auto_tile),
        }
        if env_tile > 0:
            sizing["env_tile_requested"] = env_tile
            sizing["env_tile_clamped_to_vram_cap"] = bool(clamped)
        return tile, sizing

    def plan(self, total_lanes, n, n_terms, nrhs=1, free_bytes=None):
        """One tile plan for the whole sweep (tile resolved ONCE, recorded)."""
        tile, sizing = self.resolve_tile(n, n_terms, nrhs, free_bytes)
        tile = min(tile, max(1, total_lanes))
        n_tiles = int(math.ceil(total_lanes / tile)) if total_lanes else 0
        meta = {
            "tile_size": int(tile),
            "n_tiles": int(n_tiles),
            "total_lanes": int(total_lanes),
            "n": int(n),
            "n_terms": int(n_terms),
            "nrhs": int(nrhs),
            "base_dtype": self.config.base_dtype,
            "vram_headroom_mib": int(self.config.vram_headroom_mib),
            **sizing,
            "graph_decision": self.graph_decision(),
        }
        return TilePlan(
            tile_size=tile, n_tiles=n_tiles, total_lanes=total_lanes, n=n,
            bytes_per_lane=sizing["bytes_per_lane"],
            shared_bytes=sizing["shared_bytes"],
            budget_bytes=sizing["budget_bytes"], meta=meta,
        )

    # -- pinned double-buffered staging ------------------------------------

    def _to_pinned(self, arr):
        import cupyx

        arr = np.ascontiguousarray(arr)
        pinned = cupyx.empty_pinned(arr.shape, arr.dtype)
        pinned[...] = arr
        return pinned

    def stage_and_run(self, inputs, compute_fn, plan):
        """Run ``compute_fn`` over tiles with pinned, double-buffered upload.

        ``inputs`` maps names to host arrays whose leading axis is the lane
        axis (length ``plan.total_lanes``).  ``compute_fn(dev_inputs, sl)``
        receives the device tile inputs and the ``(start, stop)`` slice, and
        returns a device array whose leading axis is the tile lanes.  Results
        are gathered on the host and returned as one array.
        """
        cp = self.ops.cp
        pinned = {k: self._to_pinned(v) for k, v in inputs.items()}
        streams = [cp.cuda.Stream(non_blocking=True),
                   cp.cuda.Stream(non_blocking=True)]
        result = None
        for i, (s0, s1) in enumerate(plan.slices()):
            stream = streams[i % 2]
            with stream:
                dev_inputs = {}
                for k, host in pinned.items():
                    tile_host = host[s0:s1]
                    d = cp.empty(tile_host.shape, tile_host.dtype)
                    d.set(tile_host, stream=stream)
                    dev_inputs[k] = d
                out = compute_fn(dev_inputs, (s0, s1))
                out = cp.asarray(out)
                if result is None:
                    result = np.empty((plan.total_lanes,) + out.shape[1:],
                                      dtype=out.dtype)
                out.get(stream=stream, out=result[s0:s1])
        for st in streams:
            st.synchronize()
        return result

    # -- CUDA-graph decision (safe, in-process) ----------------------------

    def graph_decision(self):
        """Record the CUDA-graph decision for the inner loop (cached).

        HONESTY CONTRACT (slice-08 consumes this provenance): ``used`` is True
        ONLY when :meth:`stage_and_run` actually executes a captured graph.
        Today the tiled inner loop runs EAGER -- the ctypes cuBLAS capture is a
        verified capability (:func:`probe_ctypes_graph`, subprocess-isolated)
        but is NOT wired into the execution path -- so ``used`` is False
        regardless of ``PYMACK_GPU_GRAPHS``.  The steady-state loop is LU-bound,
        where graphs amortise ~1.00x (slice 00) at real WDDM context-poisoning
        risk; graphs are reserved for launch-bound stages.  A safe
        elementwise-kernel proxy quantifies the launch-overhead headroom and
        confirms capture+replay is bitwise-exact for safe kernels.
        """
        if self._graph_decision is not None:
            return self._graph_decision
        proxy = self._safe_graph_proxy()
        self._graph_decision = {
            "used": False,                       # eager execution path only
            "execution_path": "eager",
            "graphs_env": bool(self.config.graphs),
            "reason": (
                "inner loop is compute-bound batched LU: graphs amortise "
                "~1.00x there (slice 00), cuBLAS capture is CuPy-refused / "
                "ctypes-only with a measured context-poisoning risk, and the "
                "ctypes capture is not wired into stage_and_run's execution "
                "path -- so nothing runs a graph. Eager by default; graphs "
                "reserved for launch-bound stages."
            ),
            "path": "ctypes-direct-cublas (capture verified, NOT wired into "
                    "stage_and_run)",
            "proxy": proxy,
        }
        return self._graph_decision

    def _safe_graph_proxy(self, launches=128):
        """Eager vs captured loop of tiny elementwise kernels (no cuBLAS).

        Safe to run in-process: no cuBLAS is captured, so the context-poisoning
        path (slice 00) cannot trigger.  Returns speedup + bitwise-match.
        """
        cp = self.ops.cp
        try:
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                x = cp.ones((1024,), dtype=cp.float32)

                def loop():
                    for _ in range(launches):
                        cp.multiply(x, cp.float32(1.0000001), out=x)

                loop()
                stream.synchronize()
                eager_ref = x.copy()

                def timed(fn):
                    s = cp.cuda.Event(); e = cp.cuda.Event()
                    s.record(stream); fn(); e.record(stream); e.synchronize()
                    return float(cp.cuda.get_elapsed_time(s, e))

                x[...] = 1.0
                eager_ms = statistics.median([timed(loop) for _ in range(3)])

                x[...] = 1.0
                stream.begin_capture()
                loop()
                graph = stream.end_capture()
                for _ in range(2):
                    graph.launch(stream); stream.synchronize()
                x[...] = 1.0
                graph_ms = statistics.median(
                    [timed(lambda: graph.launch(stream)) for _ in range(3)]
                )
                # bitwise check: one eager loop vs one graph replay from the
                # same start must reproduce identical device values.
                x[...] = 1.0
                loop(); stream.synchronize(); one_eager = x.copy()
                x[...] = 1.0
                graph.launch(stream); stream.synchronize(); one_graph = x.copy()
                match = bool(cp.array_equal(one_eager, one_graph))
            return {
                "attempted": True,
                "supported": True,
                "launches": launches,
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
                "speedup": (eager_ms / graph_ms) if graph_ms > 0 else None,
                "bitwise_match": match,
                "kernel": "elementwise multiply x1024 (launch-overhead proxy)",
            }
        except Exception as exc:  # pragma: no cover - environment-specific
            return {"attempted": True, "supported": False,
                    "error_type": type(exc).__name__, "error": str(exc)}

    # -- ctypes cuBLAS capture (opt-in, risky; prefer subprocess) ----------

    def capture_inner_loop(self, iteration, iters, stream):
        """Capture ``iters`` calls of ``iteration()`` into a CUDA graph.

        ``iteration`` must issue only capturable work on ``stream`` (the caller
        uses direct-ctypes cuBLAS to bypass CuPy's capture refusal).  Returns
        the captured graph.  Prefer :func:`probe_ctypes_graph` (subprocess) --
        a broken replay can poison the CUDA context.
        """
        iteration()                      # warm eager pass (lazy init)
        stream.synchronize()
        stream.begin_capture()
        for _ in range(iters):
            iteration()
        return stream.end_capture()


# ---------------------------------------------------------------------------
# Subprocess-isolated ctypes cuBLAS graph probe
# ---------------------------------------------------------------------------


def _find_cublas_dll():
    for name in ("cublas64_12.dll", "cublas64_13.dll"):
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


def _ctypes_graph_child(n=64, batch=32, iters=8):
    """Child-process body: capture+replay a c64 LU inner loop, check bitwise."""
    import cupy as cp

    from pymack.gpu.kernels import CupyBatchedLinalgOps

    ops = CupyBatchedLinalgOps(cp=cp)
    cublas = ops.cublas
    lib = _find_cublas_dll()
    if lib is None:
        return {"attempted": True, "supported": False,
                "reason": "cublas DLL not loadable"}
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((batch, n, n))
         + 1j * rng.standard_normal((batch, n, n))
         + (n * np.eye(n))[None]).astype(np.complex64)
    b = (rng.standard_normal((batch, n, 1))
         + 1j * rng.standard_normal((batch, n, 1))).astype(np.complex64)

    def solve_once():
        lu = ops.lu_factor(cp.asarray(A))
        return ops.lu_solve(lu, cp.asarray(b), trans="N")

    eager = cp.asnumpy(solve_once())

    # ctypes capture of a factor+solve loop on a private stream
    stream = cp.cuda.Stream(non_blocking=True)
    c_int, c_void_p = ctypes.c_int, ctypes.c_void_p
    lib.cublasSetStream_v2.argtypes = [c_void_p, c_void_p]
    lib.cublasSetStream_v2.restype = c_int
    lib.cublasCgetrfBatched.argtypes = [
        c_void_p, c_int, c_void_p, c_int, c_void_p, c_void_p, c_int]
    lib.cublasCgetrfBatched.restype = c_int
    lib.cublasCgetrsBatched.argtypes = [
        c_void_p, c_int, c_int, c_int, c_void_p, c_int, c_void_p, c_void_p,
        c_int, c_void_p, c_int]
    lib.cublasCgetrsBatched.restype = c_int
    handle = c_void_p(ops.handle)
    host_info = ctypes.c_int(0)

    # Pre-stage every source on the DEVICE: no H2D transfer may occur inside a
    # stream capture, so the reset copies below are device-to-device.
    Ad = cp.asarray(A)
    Ad_T = cp.ascontiguousarray(Ad.transpose(0, 2, 1))
    b_dev = cp.asarray(b)
    bc_src = cp.ascontiguousarray(b_dev.transpose(0, 2, 1))
    lu_buf = Ad_T.copy()
    a_ptrs = ops._pointer_array(lu_buf)
    piv = cp.empty((batch, n), dtype=cp.int32)
    dev_info = cp.empty((batch,), dtype=cp.int32)
    bc = bc_src.copy()
    b_ptrs = ops._pointer_array(bc)

    def iteration():
        cp.copyto(lu_buf, Ad_T)
        cp.copyto(bc, bc_src)
        lib.cublasCgetrfBatched(handle, n, c_void_p(a_ptrs.data.ptr), n,
                                c_void_p(piv.data.ptr),
                                c_void_p(dev_info.data.ptr), batch)
        lib.cublasCgetrsBatched(handle, 0, n, 1, c_void_p(a_ptrs.data.ptr), n,
                                c_void_p(piv.data.ptr), c_void_p(b_ptrs.data.ptr),
                                n, ctypes.byref(host_info), batch)

    with stream:
        lib.cublasSetStream_v2(handle, c_void_p(stream.ptr))
        try:
            iteration()
            stream.synchronize()
            stream.begin_capture()
            for _ in range(iters):
                iteration()
            graph = stream.end_capture()
        except Exception as exc:
            try:
                if getattr(stream, "is_capturing", lambda: False)():
                    stream.end_capture()
            except Exception:
                pass
            return {"attempted": True, "supported": False,
                    "error_type": type(exc).__name__, "error": str(exc)}
        try:
            graph.launch(stream)
            stream.synchronize()
        except Exception as exc:
            return {"attempted": True, "captured": True, "replayed": False,
                    "error_type": type(exc).__name__, "error": str(exc)}
        replayed = cp.asnumpy(bc.transpose(0, 2, 1))
    lib.cublasSetStream_v2(handle, c_void_p(cp.cuda.Stream.null.ptr))

    match = bool(np.allclose(replayed, eager, rtol=0, atol=0))
    return {"attempted": True, "supported": True, "captured": True,
            "replayed": True, "bitwise_match_vs_eager": match,
            "max_abs_diff": float(np.max(np.abs(replayed - eager))),
            "n": n, "batch": batch, "iters": iters}


def probe_ctypes_graph(n=64, batch=32, iters=8, timeout=180):
    """Run the ctypes cuBLAS graph capture in a CHILD process (isolation).

    A broken captured replay can raise ``cudaErrorIllegalAddress`` and poison
    the CUDA context; isolating it protects the caller's process (slice 00).
    Returns the child's JSON record (capture/replay/bitwise-match).
    """
    import subprocess

    code = (
        "import json,sys;"
        "from pymack.gpu.batch import _ctypes_graph_child;"
        f"print(json.dumps(_ctypes_graph_child({n},{batch},{iters})))"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], check=False,
                              capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {"attempted": True, "isolation": "subprocess",
                "subprocess_failed": True, "error_type": type(exc).__name__,
                "error": str(exc)}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                rec["isolation"] = "subprocess"
                rec["returncode"] = proc.returncode
                return rec
            except json.JSONDecodeError:
                break
    return {"attempted": True, "isolation": "subprocess",
            "subprocess_failed": True, "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-1500:]}
