"""Deprecated module path. Use :mod:`pymack.temporal_shooting` instead.

This shim preserves the old import path (the module was formerly named after
the Ozgen & Kircali (2008) reference whose first-order system it follows).
Kept for backwards compatibility.
"""
from .temporal_shooting import *  # noqa: F401,F403
