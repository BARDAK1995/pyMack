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
from .baseflow import (
    BlasiusProfile,
    CompressibleBlasiusProfile,
    OzgenFlatPlateProfile,
    make_ozgen_profile,
    ozgen_adiabatic_wall_temperature,
)
from .mack_conditions import (
    make_mack_profile,
    mack_figure_edge_temperature,
    mack_table_11_1_edge_temperature,
)
from .scales import (
    delta_star_over_lstar,
    delta_star_to_eta,
    delta_star_to_lstar,
    eta_to_delta_star,
    eta_to_lstar,
    lstar_to_delta_star,
    lstar_to_eta,
    momentum_thickness_over_lstar,
    rescale_baseflow_derivatives,
)
from .asymptotic import (
    mack_freestream_characteristic_values,
    mack_freestream_decay_basis,
    mack_freestream_subspace_residual,
)
from .mack_shooting import (
    mack_first_order_matrix_3d,
    temporal_shooting_residual_3d,
    temporal_shooting_sigma_min_3d,
    solve_temporal_mode_3d_shooting,
    solve_temporal_mode_3d_shooting_sigma_min,
    continue_temporal_mode_3d_shooting_sigma_min,
)
from .solver import solve_temporal_os, solve_spatial
from .analysis import (
    critical_reynolds_from_growth_series,
    critical_reynolds_curve,
    find_temporal_mode_anchor_3d_shooting,
    frequency_sweep,
    most_unstable_wave_angle,
    neutral_curve,
    nfactor,
    search_temporal_roots_3d_shooting,
    spatial_growth_curve,
    spatial_growth_map,
    temporal_growth_curve,
    temporal_growth_map,
    temporal_growth_scan_3d_shooting,
    temporal_growth_scan_3d_shooting_from_anchor,
    temporal_neutral_points_from_scan,
    trace_spatial_neutral_curve,
    trace_temporal_neutral_curve,
    trace_temporal_neutral_curve_shooting,
)
from .reference_data import (
    find_paper_target,
    load_paper_target_registry,
    load_reference_csv,
    reference_data_root,
)

__version__ = "0.1.0"
