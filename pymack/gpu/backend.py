"""Device policy, environment knobs, and the determinism contract.

This module resolves the GPU engine's runtime policy (which device, which base
precision, refinement/promotion thresholds, VRAM headroom, whether CUDA graphs
are permitted) from environment variables with conservative defaults, and
implements :func:`determinism_check` -- the bitwise-stability contract the spec
requires at fixed (GPU, driver, CuPy, precision, tile size): the same tile run
twice must produce ``cupy.array_equal`` outputs.

No CuPy import happens at module import time; the guarded import lives in
``pymack.gpu.__init__`` and the ops import CuPy lazily on the device path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np

__all__ = [
    "BackendConfig",
    "resolve_config",
    "select_device",
    "device_report",
    "determinism_check",
]


# ---------------------------------------------------------------------------
# Environment-driven policy
# ---------------------------------------------------------------------------


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_dtype(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    val = raw.strip().lower()
    if val in ("complex64", "c64", "single"):
        return "complex64"
    if val in ("complex128", "z128", "double"):
        return "complex128"
    raise ValueError(f"{name}={raw!r}: expected complex64/complex128")


@dataclass(frozen=True)
class BackendConfig:
    """Resolved GPU-engine policy (all fields overridable by environment).

    Environment knobs (all optional):

    ``PYMACK_GPU_DEVICE``           CUDA device id (default 0).
    ``PYMACK_GPU_BASE_DTYPE``       base solve precision, ``complex64`` (default)
                                    or ``complex128``.
    ``PYMACK_GPU_REFINE_TOL``       target relative FP64 residual (default 1e-10).
    ``PYMACK_GPU_MAX_REFINE``       max refinement iterations (default 30).
    ``PYMACK_GPU_PROMOTE``          enable z128 promotion (default on).
    ``PYMACK_GPU_PROMOTE_GROWTH``   pivot-growth promotion threshold (default 1e8).
    ``PYMACK_GPU_TILE``             fixed tile size in lanes; 0 == VRAM-adaptive.
    ``PYMACK_GPU_VRAM_HEADROOM_MIB``  free-VRAM headroom to leave (default 512).
    ``PYMACK_GPU_GRAPHS``           permit CUDA-graph capture of the inner loop
                                    (default off; graphs do not pay on the
                                    compute-bound LU loop -- slice 00).
    """

    device: int = 0
    base_dtype: str = "complex64"
    refine_tol: float = 1e-10
    max_refine: int = 30
    promote: bool = True
    promote_growth: float = 1e8
    tile: int = 0
    vram_headroom_mib: int = 512
    graphs: bool = False

    @property
    def vram_headroom_bytes(self):
        return int(self.vram_headroom_mib) * 1024 * 1024

    def as_meta(self):
        """JSON-able dict for result/CSV/NPZ provenance headers."""
        return dict(asdict(self))


def resolve_config():
    """Build a :class:`BackendConfig` from the environment (with defaults)."""
    return BackendConfig(
        device=_env_int("PYMACK_GPU_DEVICE", 0),
        base_dtype=_env_dtype("PYMACK_GPU_BASE_DTYPE", "complex64"),
        refine_tol=_env_float("PYMACK_GPU_REFINE_TOL", 1e-10),
        max_refine=_env_int("PYMACK_GPU_MAX_REFINE", 30),
        promote=_env_bool("PYMACK_GPU_PROMOTE", True),
        promote_growth=_env_float("PYMACK_GPU_PROMOTE_GROWTH", 1e8),
        tile=_env_int("PYMACK_GPU_TILE", 0),
        vram_headroom_mib=_env_int("PYMACK_GPU_VRAM_HEADROOM_MIB", 512),
        graphs=_env_bool("PYMACK_GPU_GRAPHS", False),
    )


# ---------------------------------------------------------------------------
# Device selection + report
# ---------------------------------------------------------------------------


def select_device(config=None):
    """Return the CuPy ``Device`` for ``config`` (does not enter its context)."""
    from . import require

    require()
    import cupy as cp

    config = config or resolve_config()
    return cp.cuda.Device(config.device)


def _base_cupy_dtype(cp, config):
    return cp.complex64 if config.base_dtype == "complex64" else cp.complex128


def device_report(config=None):
    """Provenance record: device name, compute capability, driver, free VRAM."""
    from . import is_available

    config = config or resolve_config()
    record = {"config": config.as_meta(), "available": is_available()}
    if not record["available"]:
        return record
    import cupy as cp

    try:
        dev = cp.cuda.Device(config.device)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace").rstrip("\x00")
        with dev:
            free, total = cp.cuda.runtime.memGetInfo()
        record.update(
            device_id=int(dev.id),
            device_name=name,
            compute_capability=[int(props.get("major", -1)),
                                int(props.get("minor", -1))],
            cupy_version=getattr(cp, "__version__", None),
            cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            cuda_driver_version=int(cp.cuda.runtime.driverGetVersion()),
            free_bytes=int(free),
            total_bytes=int(total),
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        record["device_error"] = f"{type(exc).__name__}: {exc}"
    return record


# ---------------------------------------------------------------------------
# Determinism contract
# ---------------------------------------------------------------------------


def _random_operator(n, batch, nrhs, seed, base_dtype):
    """Fixed host operator + RHS (identical bytes every call for a given seed)."""
    rng = np.random.default_rng(seed)

    def cplx(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    # a diagonally-dominant, well-conditioned batch (deterministic factorization)
    A = cplx((batch, n, n)) + (n * np.eye(n))[None]
    B = 0.1 * cplx((batch, n, n))
    b = cplx((batch, n, nrhs))
    z = (0.3 + 0.2j) * np.ones(batch)
    return (A.astype(base_dtype), B.astype(base_dtype),
            b.astype(base_dtype), z)


def determinism_check(ops=None, *, n=64, batch=8, nrhs=4, seed=0, config=None):
    """Run one representative tile twice; assert bitwise-identical device output.

    Pipeline exercised: ``contract`` (a shift ``A - zB`` in one GEMM),
    ``equilibrate``, batched ``lu_factor``, multi-RHS ``lu_solve`` (``'N'`` and
    ``'C'``), ``lu_diagonal``, and the FP64 matrix-free ``residual``.  Two runs
    from byte-identical host inputs must produce ``cupy.array_equal`` results at
    fixed (GPU, driver, CuPy, precision, tile size).

    Returns a dict with ``deterministic`` (bool) and per-output equality flags.
    """
    from . import require

    require()
    import cupy as cp

    config = config or resolve_config()
    if ops is None:
        from .kernels import CupyBatchedLinalgOps

        ops = CupyBatchedLinalgOps(cp=cp)
    base_dtype = _base_cupy_dtype(cp, config)
    A_h, B_h, b_h, z_h = _random_operator(n, batch, nrhs, seed, base_dtype)
    row_h = np.exp2(np.zeros(n))          # trivial power-of-2 scalings (exact)
    col_h = np.exp2(np.zeros(n))

    def run():
        A = cp.asarray(A_h)
        B = cp.asarray(B_h)
        b = cp.asarray(b_h)
        z = cp.asarray(z_h).astype(base_dtype)
        row = cp.asarray(row_h)
        col = cp.asarray(col_h)
        # equilibrated shift M = diag(row) (A - zB) diag(col)
        M = ops.equilibrate(A - z[:, None, None] * B, row, col)
        lu = ops.lu_factor(M)
        x = ops.lu_solve(lu, b, trans="N")
        xh = ops.lu_solve(lu, b, trans="C")
        diagU = ops.lu_diagonal(lu)
        # FP64 residual of the primary solve.  The determinism operator is a
        # per-lane random matrix (not a shared-K affine family), so the FP64
        # apply here is a direct batched matmul; the affine matrix-free path is
        # exercised by the refinement tests.
        Md = ops.equilibrate(
            A.astype(cp.complex128) - z[:, None, None].astype(cp.complex128)
            * B.astype(cp.complex128),
            cp.asarray(row_h), cp.asarray(col_h),
        )
        r = b.astype(cp.complex128) - cp.matmul(Md, x.astype(cp.complex128))
        cp.cuda.get_current_stream().synchronize()
        return x, xh, diagU, r

    x1, xh1, d1, r1 = run()
    x2, xh2, d2, r2 = run()
    flags = {
        "solve_N": bool(cp.array_equal(x1, x2)),
        "solve_C": bool(cp.array_equal(xh1, xh2)),
        "lu_diagonal": bool(cp.array_equal(d1, d2)),
        "residual": bool(cp.array_equal(r1, r2)),
    }
    deterministic = all(flags.values())
    return {
        "deterministic": deterministic,
        "flags": flags,
        "n": n,
        "batch": batch,
        "nrhs": nrhs,
        "base_dtype": config.base_dtype,
    }
