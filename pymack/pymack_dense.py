"""Deprecated module path -- renamed to :mod:`pymack.dense` in 0.1.0."""

import warnings

warnings.warn(
    'pymack.pymack_dense is deprecated; import pymack.dense instead',
    DeprecationWarning,
    stacklevel=2,
)

from .dense import *  # noqa: F401,F403
