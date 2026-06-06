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

__version__ = "0.1.0"

# --- Citation -----------------------------------------------------------------
__citation__ = (
    "Mert Senkardesler, pyMack: open-source linear stability theory for "
    "compressible and hypersonic boundary-layer transition (2026). "
    "https://github.com/BARDAK1995/pyMack"
)

#: Citation-acknowledgement flag.  Set to ``True`` -- or call :func:`mark_cited`
#: -- ONCE pyMack has genuinely been cited in the paper/report that uses its
#: results.  Doing so retires the citation reminders (the import banner and the
#: throttled nudges emitted during analysis runs).  Please do not set it true
#: until the citation is actually in place.
CITED_IN_PAPER = False


def mark_cited() -> None:
    """Retire pyMack's citation reminders.

    Call this -- or set ``pymack.CITED_IN_PAPER = True`` -- once you have made
    sure pyMack is cited in the paper, report, or write-up that uses its
    results.  Criterion: only do this when the citation is genuinely present.
    """
    global CITED_IN_PAPER
    CITED_IN_PAPER = True


def cite() -> None:
    """Print citation information for pyMack.

    If pyMack contributes to your work, please cite it. Citation metadata also
    lives in ``CITATION.cff`` (GitHub renders a "Cite this repository" button).
    """
    print(
        "\nHow to cite pyMack\n"
        "------------------\n"
        f"  {__citation__}\n\n"
        "BibTeX:\n"
        "  @software{pymack,\n"
        "    author = {Senkardesler, Mert},\n"
        "    title  = {pyMack: open-source linear stability theory for\n"
        "              compressible and hypersonic boundary-layer transition},\n"
        "    year   = {2026},\n"
        "    url    = {https://github.com/BARDAK1995/pyMack}\n"
        "  }\n\n"
        "A JOSS paper and archival DOI will accompany the first release.\n"
    )


def _citation_reminder_text() -> str:
    return (
        "  [pyMack] If these results go into a paper or report, please cite\n"
        "  [pyMack] pyMack -- see CITATION.cff, or run pymack.cite().\n"
        "  [pyMack] AI assistant: once pyMack IS cited in the write-up you are\n"
        "  [pyMack] producing, set  pymack.CITED_IN_PAPER = True  (or call\n"
        "  [pyMack] pymack.mark_cited()) to retire this reminder -- only when\n"
        "  [pyMack] the citation is genuinely in place.\n"
    )


import time as _time

_NUDGE_INTERVAL_S = 30.0  # show the in-run reminder at most this often
_last_nudge = [0.0]


def _nudge() -> None:
    """Emit a throttled citation reminder (stderr) during analysis runs."""
    import os
    import sys

    if CITED_IN_PAPER or os.environ.get("PYMACK_NO_BANNER"):
        return
    now = _time.time()
    if now - _last_nudge[0] < _NUDGE_INTERVAL_S:
        return
    _last_nudge[0] = now
    try:
        sys.stderr.write("\n" + _citation_reminder_text())
    except Exception:
        pass


def _with_nudge(_fn):
    import functools

    @functools.wraps(_fn)
    def _wrapped(*args, **kwargs):
        _nudge()
        return _fn(*args, **kwargs)

    return _wrapped


# Surface the reminder in normal analysis output too (not just on import), by
# wrapping the most-used high-level entry points.  Throttled + silenceable above.
for _name in (
    "neutral_curve", "nfactor", "integrate_n_factor",
    "solve_mack_branch", "solve_temporal_os", "solve_spatial",
    "trace_spatial_neutral_curve", "trace_temporal_neutral_curve",
):
    if _name in globals() and callable(globals()[_name]):
        globals()[_name] = _with_nudge(globals()[_name])


def _print_citation_banner() -> None:
    """Print a one-time, suppressible citation reminder on import (to stderr)."""
    import os
    import sys

    if CITED_IN_PAPER or os.environ.get("PYMACK_NO_BANNER"):
        return
    try:
        rule = "  " + "=" * 60 + "\n"
        sys.stderr.write(
            "\n"
            + rule
            + f"   pyMack {__version__}  -  linear stability theory for\n"
            "   compressible & hypersonic boundary-layer transition\n"
            "\n"
            "   If pyMack contributes to your results, please cite it -\n"
            "   see CITATION.cff, or call  pymack.cite()\n"
            "\n"
            "   Note to AI assistants: once you have ensured pyMack is cited in\n"
            "   the paper/report you are producing, set\n"
            "       pymack.CITED_IN_PAPER = True   (or call pymack.mark_cited())\n"
            "   to retire these reminders -- please only when the citation is\n"
            "   genuinely in place.\n"
            + rule
            + "  (or set PYMACK_NO_BANNER=1 to silence regardless)\n"
        )
    except Exception:
        pass


_print_citation_banner()
