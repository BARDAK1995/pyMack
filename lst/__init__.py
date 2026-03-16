"""
Compressible Linear Stability Theory (LST) Solver
==================================================

Spatial stability analysis for compressible boundary layers using
Chebyshev spectral collocation.

Modules
-------
spectral   : Chebyshev differentiation matrices and domain mapping
baseflow   : Mean flow profile interfaces (Blasius, compressible, tabulated)
equations  : Compressible stability equation coefficient matrices
solver     : Eigenvalue solver with mode filtering and tracking
analysis   : Parameter sweeps, neutral curves, N-factor integration
plotting   : Publication-quality visualization
"""

from .spectral import chebyshev_points, chebyshev_D, map_domain, physical_derivatives
from .baseflow import BlasiusProfile, CompressibleBlasiusProfile
from .solver import solve_temporal_os, solve_spatial
from .analysis import frequency_sweep, neutral_curve, nfactor

__version__ = "0.1.0"
