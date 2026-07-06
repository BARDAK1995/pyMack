"""Batched mixed-precision linear-algebra core for the pyMack GPU engines.

This module freezes the :class:`BatchedLinalgOps` protocol -- the single
device primitive surface every GPU engine (temporal, spatial, verdict) consumes
and the future C++/CUDA extension implements -- and provides the CuPy/cuBLAS
reference implementation :class:`CupyBatchedLinalgOps`.

The primitives are, deliberately, only the ones the measured method needs
(slice-00 primitive inventory; slice-02 method verdict):

* ``contract``    scalars x constant K-tensors -> a batch of assembled matrices
                  in ONE GEMM (the affine-operator contraction of slice 05);
* ``equilibrate`` fused two-sided power-of-2 scaling, consuming the cache's
                  baked ``row_scale``/``col_scale`` (exact in binary FP);
* ``lu_factor``   batched dense LU via cuBLAS ``{c,z}getrfBatched``
                  (complex64 or complex128, chosen from the array dtype);
* ``lu_solve``    batched multi-RHS triangular solves via ``{c,z}getrsBatched``
                  with ``trans in {'N','C'}`` -- ``'N'`` solves ``A x = b``,
                  ``'C'`` solves ``A^H x = b`` (the polisher's adjoint solve);
* ``lu_diagonal`` / ``lu_pivot_swaps`` the U-diagonal and permutation parity
                  the winding-number absence certificate reads off the factors;
* ``affine_matvec`` / ``residual`` FP64 matrix-free operator application: the
                  operator is applied straight from the affine K-tensors, so no
                  per-lane FP64 dense matrix is ever materialised (slice-README
                  invariant: every candidate carries an FP64 matrix-free
                  residual; B is singular so nothing forms ``B^{-1} A``).

Column-major convention (the load-bearing subtlety).  cuBLAS is column-major;
a C-contiguous CuPy ``(n, n)`` buffer is seen by cuBLAS as its transpose.  To
make cuBLAS factor the mathematical matrix ``A`` (so that ``trans='N'`` means
``A x = b`` and ``trans='C'`` means ``A^H x = b``, with no surprise transpose
leaking into the protocol) the factor buffer stores ``A`` transposed per matrix
(``A^T`` row-major == ``A`` column-major).  The U-diagonal is then the buffer
diagonal.  This was validated end-to-end before use (N-solve, C-solve, and
``prod(diag U) == det`` all confirmed on random and real anchor operators).

Measured device traps inherited from slice 00 and honoured here:

* ``cupy_backends`` submodules load ONLY via package attribute access
  (``getattr(import_module('cupy_backends.cuda.libs'), 'cublas')``); a direct
  ``import_module('cupy_backends.cuda.libs.cublas')`` dies with "DLL load
  failed" on Windows because the package ``__getattr__`` preloads the wheel DLL.
* ``getrfBatched`` writes ``info`` to a DEVICE int array (one per matrix);
  ``getrsBatched`` writes ``info`` to a HOST int pointer -- passing a device
  pointer there corrupts host memory.  Both are handled below.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BatchedLinalgOps",
    "LuFactorization",
    "CupyBatchedLinalgOps",
    "get_cupy",
]


# ---------------------------------------------------------------------------
# Guarded CuPy access (this module is only imported on the device path)
# ---------------------------------------------------------------------------


def get_cupy():
    """Return the imported ``cupy`` module or raise a clear error.

    Import is deferred to call time so that merely importing this module never
    forces CuPy (the package ``__init__`` guard owns availability); engines
    import :class:`CupyBatchedLinalgOps` lazily on the device path.
    """
    try:
        import cupy  # noqa: F401
    except Exception as exc:  # pragma: no cover - GPU-less CI never gets here
        raise RuntimeError(
            "CupyBatchedLinalgOps requires CuPy and a CUDA device; import the "
            "CPU affine path (pymack.gpu.affine) for GPU-less use."
        ) from exc
    return importlib.import_module("cupy")


def _cublas():
    """cuBLAS backend via the ONLY load path that works on Windows wheels."""
    return getattr(importlib.import_module("cupy_backends.cuda.libs"), "cublas")


# ---------------------------------------------------------------------------
# The frozen protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BatchedLinalgOps(Protocol):
    """The device primitive surface every GPU engine consumes.

    All arrays are CuPy ``ndarray`` on the device path (a future C++/CUDA
    extension implements the same signatures).  Batched arrays lead with the
    batch axis; matrices are ``(batch, n, n)``; right-hand sides and solutions
    are ``(batch, n, nrhs)`` (a plain ``(batch, n)`` is accepted as ``nrhs=1``).
    """

    def contract(self, scalars, terms):
        """``matrices[p] = sum_j scalars[p, j] * terms[j]`` in one GEMM.

        ``scalars`` is ``(batch, K)``; ``terms`` is ``(K, n, n)``; returns
        ``(batch, n, n)``.  This is the affine-operator contraction of slice 05.
        """

    def equilibrate(self, matrices, row_scale, col_scale):
        """Fused two-sided scaling ``diag(row) @ M @ diag(col)`` per matrix.

        ``row_scale``/``col_scale`` are the cache's baked ``(n,)`` power-of-2
        vectors; the multiply is exact in binary FP so results stay bitwise
        comparable through the scaling.  Works for any leading dims (a matrix
        batch or a ``(K, n, n)`` term stack).
        """

    def lu_factor(self, matrices):
        """Batched dense LU; returns a reusable :class:`LuFactorization`."""

    def lu_solve(self, lu, rhs, trans="N"):
        """Batched multi-RHS solve reusing ``lu``.

        ``trans='N'`` solves ``A x = rhs``; ``trans='C'`` solves
        ``A^H x = rhs`` (``A`` the matrix handed to :meth:`lu_factor`).
        ``trans='T'`` (solve ``A^T x = rhs``) is a verified extension BEYOND
        the frozen protocol (Opus review, 5.6e-17); ``'N'``/``'C'`` are the
        frozen surface.
        """

    def lu_diagonal(self, lu):
        """Return ``diag(U)`` per lane, ``(batch, n)`` -- winding certificate."""

    def lu_pivot_swaps(self, lu):
        """Return the number of row swaps per lane, ``(batch,)`` int (det sign)."""

    def affine_matvec(self, scalars, terms, x):
        """FP64 matrix-free ``y[p] = (sum_j scalars[p, j] terms[j]) @ x[p]``."""

    def residual(self, scalars, terms, x, b):
        """FP64 matrix-free residual ``b - (affine operator) @ x``."""


# ---------------------------------------------------------------------------
# Reusable factorization handle
# ---------------------------------------------------------------------------


@dataclass
class LuFactorization:
    """Opaque handle to a batched LU factorization (reused across solves).

    Attributes are held so the underlying device buffers stay alive as long as
    the handle does (the pointer array references ``lu``'s base address).
    """

    lu: object          # (batch, n, n) packed factors, cuBLAS column-major view
    piv: object         # (batch, n) int32 device pivots (1-based)
    info: object        # (batch,) int32 device getrf info (0 == success)
    a_ptrs: object      # (batch,) uintp device pointer array into ``lu``
    n: int
    batch: int
    dtype: object       # cupy complex dtype used for the factorization

    def singular_lanes(self):
        """Boolean host array, True where getrf reported a zero pivot."""
        return np.asarray(self.info.get()) != 0


# ---------------------------------------------------------------------------
# CuPy / cuBLAS implementation
# ---------------------------------------------------------------------------


class CupyBatchedLinalgOps:
    """cuBLAS-backed :class:`BatchedLinalgOps` (complex64 and complex128).

    One instance dispatches both precisions from the array dtype, so the
    refinement loop can factor c64 and promote a sub-batch to z128 through the
    same ops object.
    """

    def __init__(self, cp=None):
        self.cp = cp if cp is not None else get_cupy()
        self.cublas = _cublas()
        device = importlib.import_module("cupy.cuda.device")
        self.handle = device.get_cublas_handle()
        # getrsBatched reports parameter errors through a HOST int pointer.
        self._host_info = np.zeros((1,), dtype=np.int32)

    # -- dtype dispatch ----------------------------------------------------

    def _prefix(self, dtype):
        cp = self.cp
        if dtype == cp.complex64:
            return "c"
        if dtype == cp.complex128:
            return "z"
        raise TypeError(
            f"batched LU supports complex64/complex128, got {dtype}"
        )

    def _bind_stream(self):
        # Bind to whatever stream is current (double-buffering / capture safe).
        self.cublas.setStream(self.handle, self.cp.cuda.get_current_stream().ptr)

    def _pointer_array(self, array):
        cp = self.cp
        step = array.strides[0]
        start = array.data.ptr
        return cp.arange(start, start + step * array.shape[0], step,
                         dtype=cp.uintp)

    # -- contraction (one GEMM) -------------------------------------------

    def contract(self, scalars, terms):
        cp = self.cp
        terms = cp.asarray(terms)
        K = terms.shape[0]
        n = terms.shape[1]
        scalars = cp.asarray(scalars)
        if scalars.ndim == 1:
            scalars = scalars[None, :]
        if scalars.shape[1] != K:
            raise ValueError(
                f"scalars last dim {scalars.shape[1]} != #terms {K}"
            )
        scalars = scalars.astype(terms.dtype, copy=False)
        flat = terms.reshape(K, n * n)
        return (scalars @ flat).reshape(scalars.shape[0], n, n)

    # -- equilibration (fused, exact powers of two) -----------------------

    def equilibrate(self, matrices, row_scale, col_scale):
        cp = self.cp
        matrices = cp.asarray(matrices)
        row = cp.asarray(row_scale)
        col = cp.asarray(col_scale)
        return matrices * row[:, None] * col[None, :]

    # -- batched LU factorization -----------------------------------------

    def lu_factor(self, matrices):
        cp = self.cp
        matrices = cp.asarray(matrices)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("lu_factor expects (batch, n, n)")
        batch, n, _ = matrices.shape
        dtype = matrices.dtype
        getrf = getattr(self.cublas, self._prefix(dtype) + "getrfBatched")
        self._bind_stream()
        # Transpose-store: buffer holds A^T row-major == A column-major, so
        # cuBLAS factors the mathematical A (see module docstring).
        lu = cp.ascontiguousarray(matrices.transpose(0, 2, 1))
        a_ptrs = self._pointer_array(lu)
        piv = cp.empty((batch, n), dtype=cp.int32)
        info = cp.empty((batch,), dtype=cp.int32)
        getrf(self.handle, n, a_ptrs.data.ptr, n, piv.data.ptr,
              info.data.ptr, batch)
        return LuFactorization(lu=lu, piv=piv, info=info, a_ptrs=a_ptrs,
                               n=n, batch=batch, dtype=dtype)

    # -- batched multi-RHS solve ------------------------------------------

    def lu_solve(self, lu, rhs, trans="N"):
        cp = self.cp
        rhs = cp.asarray(rhs)
        squeeze = False
        if rhs.ndim == 2:               # (batch, n) -> (batch, n, 1)
            rhs = rhs[:, :, None]
            squeeze = True
        if rhs.ndim != 3 or rhs.shape[1] != lu.n or rhs.shape[0] != lu.batch:
            raise ValueError(
                f"rhs shape {rhs.shape} incompatible with (batch={lu.batch}, "
                f"n={lu.n})"
            )
        nrhs = rhs.shape[2]
        getrs = getattr(self.cublas, self._prefix(lu.dtype) + "getrsBatched")
        if trans == "N":
            tcode = self.cublas.CUBLAS_OP_N
        elif trans == "C":
            tcode = self.cublas.CUBLAS_OP_C
        elif trans == "T":
            tcode = self.cublas.CUBLAS_OP_T
        else:
            raise ValueError(f"trans must be 'N', 'C' or 'T', got {trans!r}")
        self._bind_stream()
        # cuBLAS RHS is column-major (n, nrhs): a C-contiguous (nrhs, n) buffer
        # presents exactly that.  Copy (never overwrite the caller's array so
        # the same LU can be re-solved against fresh residuals).
        bc = cp.ascontiguousarray(
            rhs.astype(lu.dtype, copy=False).transpose(0, 2, 1)
        )
        b_ptrs = self._pointer_array(bc)
        self._host_info[0] = 0
        getrs(self.handle, tcode, lu.n, nrhs, lu.a_ptrs.data.ptr, lu.n,
              lu.piv.data.ptr, b_ptrs.data.ptr, lu.n,
              self._host_info.ctypes.data, lu.batch)
        # getrsBatched writes info to a HOST pointer; under a non-null (or
        # captured) stream the read races the launch, and a stale 0 would mask
        # an illegal-argument launch error into silent garbage.  Synchronise the
        # bound stream before reading.  (lu_solve is never called during graph
        # capture -- the ctypes probe path bypasses it -- so the sync is safe.)
        self.cp.cuda.get_current_stream().synchronize()
        if int(self._host_info[0]) != 0:
            raise RuntimeError(
                f"getrsBatched reported illegal argument info="
                f"{int(self._host_info[0])}"
            )
        x = bc.transpose(0, 2, 1)       # back to (batch, n, nrhs)
        return x[:, :, 0] if squeeze else x

    # -- LU-diagonal / permutation parity (winding certificate) -----------

    def lu_diagonal(self, lu):
        cp = self.cp
        return cp.diagonal(lu.lu, axis1=1, axis2=2)

    def lu_pivot_swaps(self, lu):
        cp = self.cp
        n = lu.n
        idx = cp.arange(1, n + 1, dtype=lu.piv.dtype)[None, :]
        # getrf pivots are 1-based row indices; a swap happened wherever
        # piv[k] != k+1.
        return cp.count_nonzero(lu.piv != idx, axis=1)

    def lu_logdet(self, lu):
        """Complex log-determinant per lane; imaginary part folded to (-pi, pi].

        ``det(A) = (-1)^{swaps} * prod(diag U)``.  The real part is
        ``log|det|``; the imaginary part is the PRINCIPAL argument of ``det``
        folded to ``(-pi, pi]`` via ``angle(exp(i*.))`` -- so a winding consumer
        never receives a spurious non-integer 2*pi increment from summing many
        per-factor ``log`` arguments.  ``exp(lu_logdet) == det`` exactly (fold
        is exp-invariant).

        NOTE: this helper is OUTSIDE the frozen ``BatchedLinalgOps`` protocol
        (currently uncalled).  Slice 08 defines the winding consumer and the
        exact branch-continuation convention it needs along the contour; this
        principal-value fold is the single-point convention until then.
        """
        cp = self.cp
        diagU = self.lu_diagonal(lu)
        logsum = cp.sum(cp.log(diagU), axis=1)
        swaps = self.lu_pivot_swaps(lu)
        sign = cp.where((swaps % 2) == 0, 0.0, cp.pi)
        logdet = logsum + 1j * sign
        folded_imag = cp.angle(cp.exp(1j * cp.imag(logdet)))
        return cp.real(logdet) + 1j * folded_imag

    # -- FP64 matrix-free operator application ----------------------------

    def affine_matvec(self, scalars, terms, x):
        cp = self.cp
        terms = cp.asarray(terms)
        x = cp.asarray(x)
        squeeze = False
        if x.ndim == 2:
            x = x[:, :, None]
            squeeze = True
        scalars = cp.asarray(scalars)
        if scalars.ndim == 1:
            scalars = scalars[None, :]
        K = terms.shape[0]
        if scalars.shape[1] != K:
            raise ValueError("affine_matvec scalars/terms length mismatch")
        # Never materialise a per-lane dense operator: accumulate the shared
        # terms applied to x, weighted per lane.  K is small (<= 18).
        out = cp.zeros(x.shape, dtype=cp.result_type(terms.dtype, x.dtype,
                                                     scalars.dtype))
        for j in range(K):
            out += scalars[:, j, None, None] * cp.matmul(terms[j], x)
        return out[:, :, 0] if squeeze else out

    def residual(self, scalars, terms, x, b):
        cp = self.cp
        b = cp.asarray(b)
        return b - self.affine_matvec(scalars, terms, x)


def default_ops(cp=None):
    """Convenience factory for the CuPy ops (device path only)."""
    return CupyBatchedLinalgOps(cp=cp)
