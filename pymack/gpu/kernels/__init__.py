"""Device linear-algebra kernels for the pyMack GPU engines.

Public surface:

* :class:`~pymack.gpu.kernels.cupy_ops.BatchedLinalgOps` -- the frozen protocol
  every engine consumes (and the future C++/CUDA extension implements).
* :class:`~pymack.gpu.kernels.cupy_ops.CupyBatchedLinalgOps` -- the CuPy/cuBLAS
  reference implementation (complex64 + complex128, mixed-precision).
* :class:`~pymack.gpu.kernels.cupy_ops.LuFactorization` -- reusable LU handle.

Importing this package does NOT import CuPy; the ops import it lazily on the
device path (:func:`~pymack.gpu.kernels.cupy_ops.get_cupy`), so GPU-less CI can
still import ``pymack.gpu.kernels`` for the protocol type.
"""

from __future__ import annotations

from .cupy_ops import (
    BatchedLinalgOps,
    CupyBatchedLinalgOps,
    LuFactorization,
    default_ops,
    get_cupy,
)

__all__ = [
    "BatchedLinalgOps",
    "CupyBatchedLinalgOps",
    "LuFactorization",
    "default_ops",
    "get_cupy",
]
