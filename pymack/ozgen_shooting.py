"""Deprecated module path -- use :mod:`pymack.temporal_shooting` instead."""

import warnings

warnings.warn(
    'pymack.ozgen_shooting is deprecated; import pymack.temporal_shooting instead',
    DeprecationWarning,
    stacklevel=2,
)

from .temporal_shooting import *  # noqa: F401,F403,E402
