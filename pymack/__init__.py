"""pyMack: local linear stability of compressible and hypersonic boundary layers.

Quickstart
----------
::

    import pymack as pm

    bl   = pm.flat_plate(Ma=6.0)                       # self-similar base flow
    mode = pm.temporal_mode(bl, alpha=0.174, Re=5500)   # discrete Mack mode
    print(mode)          # c = 0.9301+0.0200j, omega_i = +3.5e-03, unstable

    mode = pm.spatial_mode(bl, omega=0.162, Re=5500)    # spatial counterpart
    mode.sigma           # spatial growth rate -Im(alpha)

The package is layered; each layer is fully public:

===================  ==========================================================
Layer                Modules
===================  ==========================================================
Facade               :mod:`pymack.api` (re-exported here)
Base flows           :mod:`pymack.baseflow`, :mod:`pymack.boundary_layer`
Operators            :mod:`pymack.spectral`, :mod:`pymack.equations`
Eigenvalue engines   :mod:`pymack.temporal_solver`, :mod:`pymack.solver`,
                     :mod:`pymack.mack_shooting`, :mod:`pymack.temporal_shooting`,
                     :mod:`pymack.dense`
Workflows            :mod:`pymack.analysis` (sweeps, neutral curves, N-factors)
Scaling / units      :mod:`pymack.scales`, :mod:`pymack.cone`
Reference data       :mod:`pymack.reference_data`, :mod:`pymack.mack_conditions`,
                     :mod:`pymack.mack_table_10_1`, :mod:`pymack.asymptotic`
===================  ==========================================================

Importing :mod:`pymack` has no side effects. If pyMack contributes to your
work, ``pymack.cite()`` prints the reference (please do -- it keeps the
project going).
"""

from __future__ import annotations

__version__ = "0.1.0"

__citation__ = (
    "Mert Senkardesler, pyMack: local linear stability solver for "
    "compressible and hypersonic boundary layers (2026). "
    "DOI: 10.5281/zenodo.20588214. https://github.com/BARDAK1995/pyMack"
)

# --- Facade: the one-obvious-way entry points --------------------------------
from .api import ModeResult, flat_plate, spatial_mode, temporal_mode

# --- Base flows ---------------------------------------------------------------
from .baseflow import (
    BlasiusProfile,
    CompressibleBlasiusProfile,
    FlatPlateProfile,
    make_flatplate_profile,
)
from .boundary_layer import (
    BoundaryLayerResult,
    DimensionalBoundaryLayer,
    generate_boundary_layer,
)
from .mack_conditions import (
    make_mack_profile,
    mack_figure_edge_temperature,
    mack_table_10_1_edge_temperature,
    mack_table_11_1_edge_temperature,
)

# --- Scaling and dimensional conversions --------------------------------------
from .scales import (
    DimensionalEdgeState,
    F_to_frequency_khz,
    R_L_to_x_m,
    R_L_to_x_mm,
    alpha_L_to_per_m,
    alpha_L_to_per_mm,
    delta_star_over_lstar,
    delta_star_to_eta,
    delta_star_to_lstar,
    eta_to_delta_star,
    eta_to_lstar,
    frequency_khz_to_F,
    lstar_m_from_R_L,
    lstar_to_delta_star,
    lstar_to_eta,
    momentum_thickness_over_lstar,
    rescale_baseflow_derivatives,
    sample_baseflow,
    sigma_L_to_per_m,
    sigma_L_to_per_mm,
    wavelength_L_to_mm,
    x_mm_to_R_L,
)

# --- Eigenvalue engines --------------------------------------------------------
from .temporal_solver import solve_temporal_2d
from .solver import (
    solve_spatial,
    solve_spatial_from_temporal,
    solve_spatial_full_spectrum,
    solve_temporal_compressible,
    solve_temporal_compressible_3d,
    solve_temporal_os,
)
from .mack_shooting import (
    continue_temporal_mode_3d_shooting_sigma_min,
    continue_temporal_mode_6_shooting_sigma_min,
    mack_first_order_matrix_3d,
    mack_first_order_matrix_6,
    solve_temporal_mode_3d_shooting,
    solve_temporal_mode_3d_shooting_sigma_min,
    solve_temporal_mode_6_shooting,
    solve_temporal_mode_6_shooting_sigma_min,
)
from .dense import (
    DenseBaseFlowConfig,
    DenseGasModel,
    DenseLSTConfig,
    prepare_dense_case,
    solve_mack_branch,
)

# --- Workflows -------------------------------------------------------------------
from .analysis import (
    critical_reynolds_by_max_growth,
    critical_reynolds_curve,
    critical_reynolds_from_growth_series,
    frequency_sweep,
    integrate_n_factor,
    maximize_growth_over_parameter,
    most_unstable_wave_angle,
    n_factor_curve,
    neutral_curve,
    neutral_points_from_growth_map,
    spatial_growth_curve,
    spatial_growth_map,
    temporal_growth_curve,
    temporal_growth_map,
    trace_spatial_neutral_curve,
    trace_temporal_neutral_curve,
    trace_temporal_neutral_curve_shooting,
)

# --- Geometry --------------------------------------------------------------------
from .cone import ConeGeometry, cone_n_factor, cone_n_factor_multiplier

# --- Reference data ----------------------------------------------------------------
from .reference_data import (
    load_mack_table_10_1_cases,
    load_paper_target_registry,
    load_reference_csv,
    select_mack_table_10_1_cases,
)

__all__ = [
    # meta
    '__version__', '__citation__', 'cite',
    # facade
    'ModeResult', 'flat_plate', 'temporal_mode', 'spatial_mode',
    # base flows
    'BlasiusProfile', 'CompressibleBlasiusProfile', 'FlatPlateProfile',
    'make_flatplate_profile', 'make_mack_profile', 'generate_boundary_layer',
    'BoundaryLayerResult', 'DimensionalBoundaryLayer',
    'mack_figure_edge_temperature', 'mack_table_10_1_edge_temperature',
    'mack_table_11_1_edge_temperature',
    # scaling
    'DimensionalEdgeState', 'delta_star_over_lstar',
    'momentum_thickness_over_lstar', 'sample_baseflow',
    'rescale_baseflow_derivatives',
    'frequency_khz_to_F', 'F_to_frequency_khz', 'lstar_m_from_R_L',
    'R_L_to_x_m', 'R_L_to_x_mm', 'x_mm_to_R_L',
    'alpha_L_to_per_m', 'alpha_L_to_per_mm',
    'sigma_L_to_per_m', 'sigma_L_to_per_mm', 'wavelength_L_to_mm',
    'eta_to_lstar', 'lstar_to_eta', 'eta_to_delta_star', 'delta_star_to_eta',
    'lstar_to_delta_star', 'delta_star_to_lstar',
    # engines
    'solve_temporal_2d', 'solve_temporal_os',
    'solve_temporal_compressible', 'solve_temporal_compressible_3d',
    'solve_spatial', 'solve_spatial_full_spectrum', 'solve_spatial_from_temporal',
    'mack_first_order_matrix_3d', 'mack_first_order_matrix_6',
    'solve_temporal_mode_3d_shooting', 'solve_temporal_mode_6_shooting',
    'solve_temporal_mode_3d_shooting_sigma_min',
    'solve_temporal_mode_6_shooting_sigma_min',
    'continue_temporal_mode_3d_shooting_sigma_min',
    'continue_temporal_mode_6_shooting_sigma_min',
    'DenseBaseFlowConfig', 'DenseGasModel', 'DenseLSTConfig',
    'prepare_dense_case', 'solve_mack_branch',
    # workflows
    'neutral_curve', 'n_factor_curve', 'integrate_n_factor',
    'temporal_growth_curve', 'temporal_growth_map',
    'spatial_growth_curve', 'spatial_growth_map',
    'trace_temporal_neutral_curve', 'trace_temporal_neutral_curve_shooting',
    'trace_spatial_neutral_curve', 'neutral_points_from_growth_map',
    'frequency_sweep', 'maximize_growth_over_parameter',
    'most_unstable_wave_angle',
    'critical_reynolds_from_growth_series', 'critical_reynolds_by_max_growth',
    'critical_reynolds_curve',
    # geometry
    'ConeGeometry', 'cone_n_factor', 'cone_n_factor_multiplier',
    # reference data
    'load_reference_csv', 'load_mack_table_10_1_cases',
    'select_mack_table_10_1_cases', 'load_paper_target_registry',
]


def cite() -> None:
    """Print the recommended citation for pyMack (text + BibTeX)."""
    print(
        "\nHow to cite pyMack\n"
        "------------------\n"
        f"  {__citation__}\n\n"
        "BibTeX:\n"
        "  @software{pymack,\n"
        "    author  = {Senkardesler, Mert},\n"
        "    title   = {pyMack: local linear stability solver for compressible\n"
        "               and hypersonic boundary layers},\n"
        "    year    = {2026},\n"
        f"    version = {{{__version__}}},\n"
        "    doi     = {10.5281/zenodo.20588214},\n"
        "    url     = {https://github.com/BARDAK1995/pyMack}\n"
        "  }\n\n"
        "Archived release DOI: 10.5281/zenodo.20588214 (a JOSS paper is planned).\n"
    )


# --- Deprecated names (PEP 562 lazy forwarding with warnings) -----------------
_DEPRECATED = {
    'solve_temporal_ozgen_2d':
        ('pymack.solve_temporal_2d', lambda: solve_temporal_2d),
    'make_ozgen_profile':
        ('pymack.make_flatplate_profile', lambda: make_flatplate_profile),
    'OzgenFlatPlateProfile':
        ('pymack.FlatPlateProfile', lambda: FlatPlateProfile),
    'nfactor':
        ('pymack.n_factor_curve', lambda: n_factor_curve),
}


def __getattr__(name):
    if name in _DEPRECATED:
        import warnings

        replacement, resolve = _DEPRECATED[name]
        warnings.warn(
            f'pymack.{name} is deprecated; use {replacement}',
            DeprecationWarning,
            stacklevel=2,
        )
        return resolve()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(__all__) | set(globals()) | set(_DEPRECATED))
