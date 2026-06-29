"""Deprecated module path. Use :mod:`pymack.temporal_solver` instead.

This shim preserves the old import path (the solver was formerly named after
the Ozgen & Kircali (2008) reference whose equation arrangement it follows);
the method is standard compressible LST. Kept for backwards compatibility.
"""
from .temporal_solver import (  # noqa: F401
    solve_temporal_2d,
    solve_temporal_ozgen_2d,
)
