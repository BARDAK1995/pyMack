"""
Local Linear Stability Solver for Compressible and Hypersonic Boundary Layers (Mack Modes)
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
    ozgen_conductivity_ratio,
    ozgen_cp_ratio,
    ozgen_local_prandtl,
    ozgen_viscosity_ratio,
)
from .mack_conditions import (
    make_mack_profile,
    mack_figure_edge_temperature,
    mack_table_10_1_edge_temperature,
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
    mack_first_order_matrix_6,
    temporal_shooting_residual_3d,
    temporal_shooting_residual_6,
    temporal_shooting_sigma_min_3d,
    temporal_shooting_sigma_min_6,
    solve_temporal_mode_3d_shooting,
    solve_temporal_mode_6_shooting,
    solve_temporal_mode_3d_shooting_sigma_min,
    solve_temporal_mode_6_shooting_sigma_min,
    continue_temporal_mode_3d_shooting_sigma_min,
    continue_temporal_mode_6_shooting_sigma_min,
)
from .solver import solve_temporal_os, solve_spatial
from .pymack_dense import (
    DenseBaseFlowConfig,
    DenseGasModel,
    DenseLSTConfig,
    prepare_dense_case,
    solve_mack_branch,
)
from .ozgen_solver import solve_temporal_ozgen_2d
from .analysis import (
    critical_reynolds_from_growth_series,
    critical_reynolds_by_max_growth,
    critical_reynolds_curve,
    find_temporal_mode_anchor_3d_shooting,
    frequency_sweep,
    maximize_growth_over_parameter,
    most_unstable_wave_angle,
    neutral_curve,
    neutral_points_from_growth_map,
    nfactor,
    integrate_n_factor,
    search_temporal_roots_3d_shooting,
    search_temporal_roots_6_shooting,
    spatial_growth_curve,
    spatial_growth_map,
    temporal_growth_curve,
    temporal_growth_map,
    temporal_growth_scan_3d_shooting,
    temporal_growth_scan_3d_shooting_from_anchor,
    temporal_neutral_points_from_scan,
    track_complex_branch,
    trace_spatial_neutral_curve,
    trace_temporal_neutral_curve,
    trace_temporal_neutral_curve_shooting,
)
from .reference_data import (
    find_paper_target,
    load_mack_table_10_1_cases,
    load_paper_target_registry,
    load_reference_csv,
    mack_table_10_1_case_key,
    reference_data_root,
    select_mack_table_10_1_cases,
)
from .mack_table_10_1 import (
    DEFAULT_TABLE_10_1_CONDITION,
    DEFAULT_TABLE_10_1_WALL_BC,
    evaluate_table_10_1_exact_shooting,
    load_low_mid_table_10_1_families,
)

__version__ = "0.0.1"

# --- Citation -----------------------------------------------------------------
__citation__ = (
    "Mert Senkardesler, pyMack: local linear stability solver for "
    "compressible and hypersonic boundary layers (2026). "
    "DOI: 10.5281/zenodo.20588214. https://github.com/BARDAK1995/pyMack"
)

#: Set to ``True`` (or call :func:`mark_cited`) to quietly retire the friendly
#: citation reminders. Only flip this once pyMack has actually been cited in a
#: paper, report, or write-up that used its results.
CITED_IN_PAPER = False


def mark_cited() -> None:
    """Retire the citation reminders for this session.

    Call this (or set ``pymack.CITED_IN_PAPER = True``) after you've made sure
    a proper citation for pyMack appears where it was used. You'll get a small
    thank-you the first time.
    """
    global CITED_IN_PAPER
    if not CITED_IN_PAPER:
        CITED_IN_PAPER = True
        try:
            import sys
            sys.stderr.write("[pyMack] Thanks! Citation reminders are off for this session. :)\n")
        except Exception:
            pass


def cite() -> None:
    """Print the recommended citation for pyMack.

    If pyMack contributes to your work, citing it helps others find the tool
    (and makes the author happy).
    """
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
        "    version = {0.0.1},\n"
        "    doi     = {10.5281/zenodo.20588214},\n"
        "    url     = {https://github.com/BARDAK1995/pyMack}\n"
        "  }\n\n"
        "Archived release DOI: 10.5281/zenodo.20588214 (a JOSS paper is planned).\n"
        "\nThanks for using pyMack — citations really help the project. :)\n"
    )


# We keep the reminders deliberately light and non-intrusive:
# - A tiny friendly note on import (to stderr)
# - A single gentle nudge only when you call the main "results" functions
#   that are most likely to end up in a paper (N-factors, neutral curves, etc.)
# Both are suppressed by CITED_IN_PAPER or PYMACK_NO_BANNER=1.
# No more periodic chatter during long parameter sweeps.

_nudged_this_session = [False]


def _nudge() -> None:
    """Emit a single, friendly citation reminder per session (stderr)."""
    import os
    import sys

    if CITED_IN_PAPER or os.environ.get("PYMACK_NO_BANNER"):
        return
    if _nudged_this_session[0]:
        return
    _nudged_this_session[0] = True
    try:
        sys.stderr.write(
            "[pyMack] If pyMack contributed to these results, citing it would be lovely :)\n"
            "         See pymack.cite() or CITATION.cff for the details.\n"
        )
    except Exception:
        pass


def _with_nudge(_fn):
    import functools

    @functools.wraps(_fn)
    def _wrapped(*args, **kwargs):
        _nudge()
        return _fn(*args, **kwargs)

    return _wrapped


# Only wrap the higher-level analysis functions that typically produce
# publishable outputs. We deliberately avoid wrapping the low-level solvers
# (solve_spatial, solve_mack_branch, etc.) so long sweeps stay quiet.
for _name in (
    "neutral_curve",
    "nfactor",
    "integrate_n_factor",
    "trace_spatial_neutral_curve",
    "trace_temporal_neutral_curve",
):
    if _name in globals() and callable(globals()[_name]):
        globals()[_name] = _with_nudge(globals()[_name])


def _print_citation_banner() -> None:
    """Print a small, friendly one-time note on first import (stderr)."""
    import os
    import sys

    if CITED_IN_PAPER or os.environ.get("PYMACK_NO_BANNER"):
        return
    try:
        sys.stderr.write(
            "[pyMack] If this helps your research, citing it would be appreciated :)\n"
            "         pymack.cite() has the details (or set PYMACK_NO_BANNER=1 to hush).\n"
        )
    except Exception:
        pass


_print_citation_banner()
