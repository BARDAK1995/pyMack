"""Deprecated module path -- use :mod:`pymack.temporal_solver` instead.

The temporal 2-D solver was formerly named after the Ozgen & Kircali (2008)
reference whose equation arrangement it follows; the method itself is
standard compressible LST.
"""

import warnings

warnings.warn(
    'pymack.ozgen_solver is deprecated; import pymack.temporal_solver instead',
    DeprecationWarning,
    stacklevel=2,
)

from .temporal_solver import (  # noqa: F401,E402
    solve_temporal_2d,
    solve_temporal_ozgen_2d,
)
