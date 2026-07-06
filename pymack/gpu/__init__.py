"""Optional GPU subpackage for pyMack batch sweep engines.

Import policy (design_architecture.md section 1):

- ``import pymack`` never imports this package; nothing under ``pymack.gpu``
  is required for the validated CPU path.
- This ``__init__`` performs the single sanctioned guarded CuPy import.  On a
  machine without CuPy the package still imports cleanly and
  :func:`is_available` returns ``False``.
- ``pymack.gpu.affine`` and ``pymack.gpu.assemblers`` are pure numpy by design
  and must stay unit-testable on GPU-less CI.
"""

try:  # the single sanctioned guarded import
    import cupy
except ImportError:  # pragma: no cover - exercised on GPU-less CI
    cupy = None


class GpuNotAvailable(RuntimeError):
    """Raised when a GPU execution path is requested without a usable GPU."""


_INSTALL_HINT = (
    "pyMack GPU support requires CuPy and a visible CUDA device. "
    "Install the optional dependency with 'pip install pymack[gpu]' "
    "(or 'pip install cupy-cuda12x' matching your CUDA runtime) and check "
    "python -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\". "
    "All results remain available through the CPU path."
)


def is_available():
    """Return True when CuPy imports and at least one CUDA device exists."""
    if cupy is None:
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def require():
    """Raise :class:`GpuNotAvailable` with an install hint if no GPU is usable."""
    if not is_available():
        raise GpuNotAvailable(_INSTALL_HINT)
