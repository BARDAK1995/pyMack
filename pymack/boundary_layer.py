"""
Standalone compressible flat-plate boundary-layer profile generator.

This module wraps the low-level similarity engine
:class:`pymack.baseflow.CompressibleBlasiusProfile` behind a single,
well-documented entry point, :func:`generate_boundary_layer`, so that
compressible Blasius-like mean flows can be produced

- as ready-to-use input for the stability solvers
  (:func:`pymack.solver.solve_spatial`, ``solve_temporal_compressible``), and
- as a standalone product (tabulated profiles, CSV/JSON export, optional
  dimensionalization to SI units at a streamwise station).

Physics conventions
-------------------
Two wall-normal length scales appear throughout pyMack and are kept explicit
here:

``L*`` (Mack / Falkner-Skan scale)
    ``L* = sqrt(nu_e x / U_e)``.  The similarity solution maps the transformed
    variable ``eta`` to the physical coordinate through
    ``y/L* = sqrt(2) * integral_0^eta (T/T_e) d eta``.  ``L*`` is independent
    of the wall condition, so profile tables default to this scale.

``delta*`` (displacement thickness)
    The stability solvers store mean flows on a ``y/delta*`` grid.
    ``delta*/L*`` is reported as :attr:`BoundaryLayerResult.delta_star_over_Lstar`.

All profile quantities are edge-normalized: ``U/U_e``, ``T/T_e``,
``rho/rho_e``, ``mu/mu_e``.  The thermal conductivity follows the engine's
code normalization ``kappa = kappa* / (cp * mu_e)`` (for the constant-Prandtl
``power_law``/``sutherland`` models this equals ``(mu/mu_e)/Pr``; for the
``mack`` Appendix-A model it is the dimensional Appendix-A conductivity
divided by ``cp * mu_e`` and is *not* simply ``mu/Pr``).
``Pr_local = mu/kappa`` in these units.

Robustness
----------
Strongly cooled or heated isothermal walls can defeat the direct BVP solve
(notably the ``mack`` transport model with wall temperatures below the
110.4 K viscosity-law kink).  The generator then falls back to the proven
continuation recipe from :func:`pymack.mack_conditions.make_mack_profile`:
solve the adiabatic wall, restart isothermally at the recovery temperature,
and ramp the wall temperature toward the target in small steps.

Note that the Appendix-A air transport laws below roughly 100 K follow the
linear-viscosity extrapolation branch, so converged solutions with very cold
walls under ``viscosity_model='mack'`` have limited physical fidelity even
when the numerics succeed.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .baseflow import CompressibleBlasiusProfile
from .scales import (
    DimensionalEdgeState,
    R_L_to_x_m,
    delta_star_over_lstar,
    lstar_m_from_R_L,
    momentum_thickness_over_lstar,
    rescale_baseflow_derivatives,
    x_mm_to_R_L,
)


__all__ = [
    'GAS_PRESETS',
    'BoundaryLayerResult',
    'DimensionalBoundaryLayer',
    'generate_boundary_layer',
]


#: Default gas properties used when ``gamma``/``Pr``/``sutherland_S_K`` are
#: not given explicitly.  These only set constant-property defaults; the
#: ``mack`` transport model is hard-wired to Mack's Appendix-A *air* laws.
GAS_PRESETS = {
    'air': {'gamma': 1.4, 'Pr': 0.72, 'sutherland_S_K': 110.4},
    'nitrogen': {'gamma': 1.4, 'Pr': 0.72, 'sutherland_S_K': 111.0},
}

_CSV_COLUMNS = (
    'eta', 'y_over_Lstar', 'y_over_delta_star',
    'U', 'dU', 'd2U', 'T', 'dT', 'd2T',
    'rho', 'mu', 'kappa', 'Pr_local',
)


def _report(progress, message):
    if progress is not None:
        progress(message)


def _resolve_gas_properties(gas, gamma, Pr, sutherland_S_K):
    """Merge gas-preset defaults with explicit overrides."""
    gas_key = str(gas).strip().lower()
    if gas_key not in GAS_PRESETS:
        raise ValueError(
            f"gas must be one of {sorted(GAS_PRESETS)}, got {gas!r}"
        )
    preset = GAS_PRESETS[gas_key]
    gamma = preset['gamma'] if gamma is None else float(gamma)
    Pr = preset['Pr'] if Pr is None else float(Pr)
    S = preset['sutherland_S_K'] if sutherland_S_K is None else float(sutherland_S_K)
    if gamma <= 1.0:
        raise ValueError('gamma must be greater than 1')
    if Pr <= 0.0:
        raise ValueError('Pr must be positive')
    if S <= 0.0:
        raise ValueError('sutherland_S_K must be positive')
    return gas_key, gamma, Pr, S


def _resolve_wall_temperature(wall_bc, Tw_over_Te, Tw_over_Taw, T_wall_K,
                              T_edge_K, Taw_K):
    """Return the target wall temperature in Kelvin for an isothermal wall.

    Exactly one of ``Tw_over_Te``, ``Tw_over_Taw``, ``T_wall_K`` must be
    given for ``wall_bc='isothermal'``; none may be given for
    ``wall_bc='adiabatic'``.
    """
    specs = {
        'Tw_over_Te': Tw_over_Te,
        'Tw_over_Taw': Tw_over_Taw,
        'T_wall_K': T_wall_K,
    }
    given = [name for name, value in specs.items() if value is not None]

    if wall_bc == 'adiabatic':
        if given:
            raise ValueError(
                'adiabatic wall takes no wall-temperature specification, '
                f'got {given}'
            )
        return None

    if len(given) != 1:
        raise ValueError(
            "isothermal wall requires exactly one of 'Tw_over_Te', "
            f"'Tw_over_Taw', 'T_wall_K'; got {given or 'none'}"
        )

    if Tw_over_Te is not None:
        target = float(Tw_over_Te) * T_edge_K
    elif Tw_over_Taw is not None:
        target = float(Tw_over_Taw) * Taw_K
    else:
        target = float(T_wall_K)

    if target <= 0.0:
        raise ValueError(f'wall temperature must be positive, got {target} K')
    return target


def _solve_isothermal_with_continuation(target_T_wall_K, Taw_K, engine_kwargs,
                                        max_continuation_step, progress):
    """Solve a difficult isothermal wall by continuation from adiabatic.

    Mirrors the proven recipe in
    :func:`pymack.mack_conditions.make_mack_profile`: adiabatic solve, then an
    isothermal restart at the recovery temperature, then a wall-temperature
    ramp toward the target in steps of at most
    ``max_continuation_step * Taw``.

    Returns
    -------
    profile : CompressibleBlasiusProfile
        Converged profile at the target wall temperature.
    n_steps : int
        Number of ramp solves performed (excluding the two seed solves).
    """
    _report(progress, 'continuation: adiabatic seed solve')
    profile = CompressibleBlasiusProfile(
        T_wall=Taw_K, wall_bc='adiabatic', **engine_kwargs)
    _report(progress, f'continuation: isothermal restart at Taw = {Taw_K:.2f} K')
    profile = CompressibleBlasiusProfile(
        T_wall=Taw_K, wall_bc='isothermal',
        initial_guess_profile=profile, **engine_kwargs)

    ratio_target = target_T_wall_K / Taw_K
    n_steps = max(4, int(np.ceil(abs(ratio_target - 1.0) / max_continuation_step)))
    for k, ratio in enumerate(np.linspace(1.0, ratio_target, n_steps + 1)[1:], 1):
        _report(
            progress,
            f'continuation: ramp step {k}/{n_steps}, '
            f'T_wall = {ratio * Taw_K:.2f} K'
        )
        profile = CompressibleBlasiusProfile(
            T_wall=float(ratio * Taw_K), wall_bc='isothermal',
            initial_guess_profile=profile, **engine_kwargs)
    return profile, n_steps


def generate_boundary_layer(
    Ma,
    *,
    wall_bc='adiabatic',
    Tw_over_Te=None,
    Tw_over_Taw=None,
    T_wall_K=None,
    T_edge_K=300.0,
    viscosity_model='sutherland',
    gas='air',
    gamma=None,
    Pr=None,
    omega=0.74,
    sutherland_S_K=None,
    n_points=4000,
    eta_max=40.0,
    continuation='auto',
    max_continuation_step=0.05,
    progress=None,
):
    """Generate a compressible flat-plate self-similar boundary layer.

    Thin, robust wrapper around
    :class:`pymack.baseflow.CompressibleBlasiusProfile`.  The mean flow is
    the coupled compressible similarity solution (not a Crocco-Busemann
    closure) and is returned as a :class:`BoundaryLayerResult`, which carries
    the tabulated nondimensional profile, integral thicknesses, exporters,
    and the underlying engine profile for direct use with the stability
    solvers.

    Parameters
    ----------
    Ma : float
        Edge Mach number.
    wall_bc : {'adiabatic', 'isothermal'}, optional
        Thermal wall condition.  For ``'isothermal'`` exactly one of
        ``Tw_over_Te``, ``Tw_over_Taw``, ``T_wall_K`` must be given; for
        ``'adiabatic'`` none may be given.
    Tw_over_Te : float, optional
        Wall temperature as a ratio of the edge temperature.
    Tw_over_Taw : float, optional
        Wall temperature as a ratio of the *formula* recovery temperature
        ``Taw = T_e * (1 + sqrt(Pr) * (gamma - 1)/2 * Ma^2)``.
    T_wall_K : float, optional
        Absolute wall temperature in Kelvin.
    T_edge_K : float, optional
        Absolute edge temperature in Kelvin.  Required physically by the
        ``sutherland`` and ``mack`` transport models (the ``power_law``
        profile shape is independent of it).
    viscosity_model : {'sutherland', 'power_law', 'mack'}, optional
        Transport model.  ``power_law``: ``mu/mu_e = (T/T_e)^omega`` with
        constant ``Pr``.  ``sutherland``: Sutherland ratio referenced to the
        edge state with constant ``Pr``.  ``mack``: Mack AGARD-709
        Appendix-A air model (piecewise viscosity law with a kink at
        110.4 K, dedicated conductivity law, variable local Prandtl
        number).
    gas : {'air', 'nitrogen'}, optional
        Preset supplying defaults for ``gamma``, ``Pr`` and
        ``sutherland_S_K`` (see :data:`GAS_PRESETS`).  ``'mack'`` transport
        requires ``gas='air'``.
    gamma, Pr : float, optional
        Explicit values override the gas preset.
    omega : float, optional
        Power-law viscosity exponent (used by ``viscosity_model='power_law'``
        only).
    sutherland_S_K : float, optional
        Sutherland constant in Kelvin; defaults to the gas preset
        (110.4 air, 111.0 nitrogen).
    n_points : int, optional
        Number of points in the tabulated profile.
    eta_max : float, optional
        Extent of the similarity domain.  The corresponding ``y/L*`` extent
        grows with the integrated temperature, but the ``y/delta*`` extent
        shrinks at high Mach number; a warning is issued if the tabulated
        ``y/delta*`` range ends below 4 (spline evaluation beyond the table
        extrapolates inside the uniform edge region, which is benign but
        worth knowing about).
    continuation : {'auto', 'never', 'force'}, optional
        Isothermal-wall robustness strategy.  ``'auto'`` attempts the direct
        solve and falls back to adiabatic-seeded wall-temperature
        continuation on failure; ``'never'`` propagates the direct-solve
        ``RuntimeError``; ``'force'`` skips the direct attempt.  Ignored for
        adiabatic walls (a single direct solve always applies).
    max_continuation_step : float, optional
        Maximum wall-temperature ramp step as a fraction of the recovery
        temperature.
    progress : callable, optional
        Called with one-line ``str`` status messages during continuation
        (the cold-wall ``mack`` rescue can take tens of seconds).

    Returns
    -------
    BoundaryLayerResult
        Solved profile with condition record, integral scalars, tabulated
        arrays, and export/dimensionalization helpers.

    Raises
    ------
    ValueError
        On conflicting or missing wall specifications, unknown option
        strings, or non-physical inputs.
    RuntimeError
        If the mean-flow BVP fails and continuation is disabled or also
        fails.

    Notes
    -----
    The result distinguishes the *formula* recovery temperature
    (``Taw_over_Te_formula``, using ``r = sqrt(Pr)``) from the *solved*
    adiabatic wall temperature (``Tw_over_Te`` after an adiabatic solve).
    They differ by up to ~2% for constant-Pr transport and ~10% for the
    variable-Pr ``mack`` model; do not use the formula value for
    heat-transfer estimates when the solved one is available.

    Examples
    --------
    >>> bl = generate_boundary_layer(4.5, wall_bc='adiabatic',
    ...                              viscosity_model='sutherland')
    >>> round(bl.delta_star_over_Lstar, 2)
    8.46
    >>> profile = bl.as_stability_profile()  # feed to solve_spatial(...)
    """
    Ma = float(Ma)
    T_edge_K = float(T_edge_K)
    if Ma < 0.0:
        raise ValueError('Ma must be non-negative')
    if T_edge_K <= 0.0:
        raise ValueError('T_edge_K must be positive')
    if wall_bc not in {'adiabatic', 'isothermal'}:
        raise ValueError("wall_bc must be 'adiabatic' or 'isothermal'")
    if viscosity_model not in {'power_law', 'sutherland', 'mack'}:
        raise ValueError(
            "viscosity_model must be 'power_law', 'sutherland', or 'mack'"
        )
    if continuation not in {'auto', 'never', 'force'}:
        raise ValueError("continuation must be 'auto', 'never', or 'force'")
    if not (0.0 < float(max_continuation_step) <= 1.0):
        raise ValueError('max_continuation_step must be in (0, 1]')

    gas_key, gamma, Pr, S = _resolve_gas_properties(gas, gamma, Pr, sutherland_S_K)
    if viscosity_model == 'mack' and gas_key != 'air':
        raise ValueError(
            "viscosity_model='mack' is the Appendix-A *air* transport model; "
            "use gas='air'"
        )

    # Formula recovery temperature (r = sqrt(Pr)); the engine uses the same
    # expression for its T_recovery attribute.
    Taw_K = T_edge_K * (1.0 + 0.5 * (gamma - 1.0) * math.sqrt(Pr) * Ma**2)

    target_T_wall_K = _resolve_wall_temperature(
        wall_bc, Tw_over_Te, Tw_over_Taw, T_wall_K, T_edge_K, Taw_K)

    engine_kwargs = dict(
        Ma=Ma, T_edge=T_edge_K, gamma=gamma, Pr=Pr, omega=float(omega),
        n_points=int(n_points), eta_max=float(eta_max),
        viscosity_model=viscosity_model, sutherland_S=S,
    )

    used_continuation = False
    n_continuation_steps = 0

    if wall_bc == 'adiabatic':
        # T_wall only seeds the isothermal initial guess; the adiabatic path
        # ignores it and overwrites T_wall/T_ratio_wall with solved values.
        profile = CompressibleBlasiusProfile(
            T_wall=Taw_K, wall_bc='adiabatic', **engine_kwargs)
    else:
        profile = None
        if continuation != 'force':
            try:
                profile = CompressibleBlasiusProfile(
                    T_wall=target_T_wall_K, wall_bc='isothermal',
                    **engine_kwargs)
            except RuntimeError:
                if continuation == 'never':
                    raise
                _report(
                    progress,
                    'direct isothermal solve failed; engaging continuation')
        if profile is None:
            profile, n_continuation_steps = _solve_isothermal_with_continuation(
                target_T_wall_K, Taw_K, engine_kwargs,
                float(max_continuation_step), progress)
            used_continuation = True

    return BoundaryLayerResult(
        profile,
        gas=gas_key,
        continuation=continuation,
        used_continuation=used_continuation,
        n_continuation_steps=n_continuation_steps,
    )


class BoundaryLayerResult:
    """Solved compressible boundary-layer profile with export helpers.

    Built by :func:`generate_boundary_layer`; not meant to be constructed
    directly.

    Attributes
    ----------
    Ma, T_edge_K, wall_bc, viscosity_model, gas, gamma, Pr, omega, sutherland_S_K
        Condition record (JSON-serializable scalars/strings).
    used_continuation : bool
        True if the adiabatic-seeded wall-temperature continuation fallback
        produced the solution.
    n_continuation_steps : int
        Number of ramp solves in the continuation (0 for direct solves; the
        two seed solves are not counted).
    Tw_over_Te : float
        Wall-to-edge temperature ratio: imposed for isothermal walls,
        *solved* for adiabatic walls.
    T_wall_K : float
        Absolute wall temperature in Kelvin (solved for adiabatic walls).
    Taw_over_Te_formula : float
        ``1 + sqrt(Pr) * (gamma - 1)/2 * Ma^2`` -- the textbook recovery
        *approximation*, kept distinct from the solved adiabatic value.
    recovery_factor_solved : float or None
        ``(Tw/Te - 1) / ((gamma - 1)/2 * Ma^2)`` from the solved adiabatic
        wall temperature; ``None`` for isothermal walls or ``Ma = 0``.
    delta_star_over_Lstar, theta_over_Lstar, delta99_over_Lstar : float
        Integral thicknesses in Mack's ``L* = sqrt(nu_e x / U_e)`` scale.
        ``delta99`` is the first interpolated ``U = 0.99`` crossing.
    shape_factor_H : float
        ``delta* / theta``.
    dU_dyL_wall, dT_dyL_wall : float
        Wall derivatives with respect to ``y/L*`` (units ``U_e/L*`` and
        ``T_e/L*``): wall-shear and wall-heat-flux indicators.
    eta, y_over_Lstar, y_over_delta_star : ndarray, shape (n_points,)
        Wall-normal coordinates: similarity variable, Mack scale, and
        displacement-thickness scale.
    U, T, rho, mu, kappa, Pr_local : ndarray
        Edge-normalized profile quantities (``kappa = kappa*/(cp mu_e)``,
        see module docstring; ``Pr_local = mu/kappa``).
    dU_dyL, d2U_dyL2, dT_dyL, d2T_dyL2 : ndarray
        Profile derivatives with respect to ``y/L*``.
    """

    def __init__(self, profile, *, gas, continuation,
                 used_continuation, n_continuation_steps):
        self._profile = profile

        # --- condition record ------------------------------------------------
        self.Ma = float(profile.Ma)
        self.T_edge_K = float(profile.T_edge)
        self.wall_bc = profile.wall_bc
        self.viscosity_model = profile.viscosity_model
        self.gas = gas
        self.gamma = float(profile.gamma)
        self.Pr = float(profile.Pr)
        self.omega = float(profile.omega)
        self.sutherland_S_K = float(profile.sutherland_S)
        self.n_points = int(len(profile._eta))
        self.eta_max = float(profile._eta[-1])
        self.continuation = continuation
        self.used_continuation = bool(used_continuation)
        self.n_continuation_steps = int(n_continuation_steps)

        # --- wall / recovery scalars -----------------------------------------
        self.Tw_over_Te = float(profile.T_ratio_wall)
        self.T_wall_K = float(profile.T_wall)
        # Formula value (r = sqrt(Pr)); the engine's T_recovery attribute.
        self.Taw_over_Te_formula = float(profile.T_recovery / profile.T_edge)
        if self.wall_bc == 'adiabatic' and self.Ma > 0.0:
            self.recovery_factor_solved = float(
                (self.Tw_over_Te - 1.0)
                / (0.5 * (self.gamma - 1.0) * self.Ma**2)
            )
        else:
            self.recovery_factor_solved = None

        # --- integral thicknesses (L* scale) ---------------------------------
        self.delta_star_over_Lstar = delta_star_over_lstar(profile)
        self.theta_over_Lstar = momentum_thickness_over_lstar(profile)
        self.shape_factor_H = self.delta_star_over_Lstar / self.theta_over_Lstar
        self.delta99_over_Lstar = self._first_u_crossing(
            profile._U, profile._y_L, 0.99)

        # --- tabulated nondimensional profile --------------------------------
        self.eta = np.array(profile._eta, copy=True)
        self.y_over_Lstar = np.array(profile._y_L, copy=True)
        self.y_over_delta_star = np.array(profile._y_nd, copy=True)
        self.U = np.array(profile._U, copy=True)
        self.T = np.array(profile._T, copy=True)
        self.rho = np.array(profile._rho, copy=True)
        self.mu = np.array(profile._mu, copy=True)
        self.kappa = np.array(profile._kappa, copy=True)
        self.Pr_local = np.array(profile._Pr_local, copy=True)

        # Engine derivatives are native to y/delta*; rescale to y/L* with the
        # shared converter (no new unit math here).
        native = {
            'dU': profile._dU, 'd2U': profile._d2U,
            'dT': profile._dT, 'd2T': profile._d2T,
        }
        scaled = rescale_baseflow_derivatives(
            native, self.delta_star_over_Lstar, 'L_star')
        self.dU_dyL = scaled['dU']
        self.d2U_dyL2 = scaled['d2U']
        self.dT_dyL = scaled['dT']
        self.d2T_dyL2 = scaled['d2T']

        self.dU_dyL_wall = float(self.dU_dyL[0])
        self.dT_dyL_wall = float(self.dT_dyL[0])

        y_nd_max = float(self.y_over_delta_star[-1])
        if y_nd_max < 4.0:
            warnings.warn(
                f'tabulated y/delta* range ends at {y_nd_max:.2f} (< 4); '
                'stability solves with larger y_max rely on benign spline '
                'extrapolation inside the uniform edge region. Increase '
                'eta_max to extend the table.',
                stacklevel=2,
            )

    @staticmethod
    def _first_u_crossing(U, y_L, level):
        """Interpolate the first ``U = level`` crossing on the ``y/L*`` grid."""
        idx = int(np.argmax(U >= level))
        if U[idx] < level:
            return float('nan')  # never reaches the level inside the table
        if idx == 0:
            return float(y_L[0])
        u0, u1 = U[idx - 1], U[idx]
        y0, y1 = y_L[idx - 1], y_L[idx]
        return float(y0 + (level - u0) * (y1 - y0) / (u1 - u0))

    # ------------------------------------------------------------------ API --

    def as_stability_profile(self):
        """Return the underlying :class:`CompressibleBlasiusProfile`.

        The returned object is directly accepted as the ``baseflow`` argument
        of :func:`pymack.solver.solve_spatial` and
        ``solve_temporal_compressible``: it is callable on ``y/delta*``
        coordinates, returns the 16-key mean-flow dictionary, and exposes
        ``_delta_star``/``_theta`` for ``length_scale='L_star'`` workflows.
        """
        return self._profile

    def sample(self, y, scale='L_star'):
        """Evaluate the full 16-key mean-flow dictionary at arbitrary ``y``.

        Parameters
        ----------
        y : array_like
            Wall-normal locations, interpreted in units of the requested
            ``scale`` (``y/delta*`` or ``y/L*``).
        scale : {'delta_star', 'L_star'}, optional
            Coordinate and derivative scale of input *and* output.

        Returns
        -------
        dict
            The engine's 16-key dictionary (``U``, ``dU``, ``d2U``, ``T``,
            ``dT``, ``d2T``, ``rho``, ``mu``, ``dmu``, ``dmu_dT``,
            ``d2mu_dT2``, ``kappa``, ``dkappa``, ``dkappa_dT``,
            ``d2kappa_dT2``, ``Pr_local``) with wall-normal derivatives
            expressed in the requested scale.  Beyond the tabulated range the
            cubic splines extrapolate (benign in the uniform edge region).
        """
        if scale not in {'delta_star', 'L_star'}:
            raise ValueError("scale must be 'delta_star' or 'L_star'")
        y = np.asarray(y, dtype=float)
        if scale == 'L_star':
            y_ds = y / self.delta_star_over_Lstar
        else:
            y_ds = y
        raw = self._profile(y_ds)
        return rescale_baseflow_derivatives(
            raw, self.delta_star_over_Lstar, scale)

    def to_dict(self):
        """Return the JSON-serializable condition record and scalars."""
        return {
            'Ma': self.Ma,
            'T_edge_K': self.T_edge_K,
            'wall_bc': self.wall_bc,
            'viscosity_model': self.viscosity_model,
            'gas': self.gas,
            'gamma': self.gamma,
            'Pr': self.Pr,
            'omega': self.omega,
            'sutherland_S_K': self.sutherland_S_K,
            'n_points': self.n_points,
            'eta_max': self.eta_max,
            'continuation': self.continuation,
            'used_continuation': self.used_continuation,
            'n_continuation_steps': self.n_continuation_steps,
            'Tw_over_Te': self.Tw_over_Te,
            'T_wall_K': self.T_wall_K,
            'Taw_over_Te_formula': self.Taw_over_Te_formula,
            'recovery_factor_solved': self.recovery_factor_solved,
            'delta_star_over_Lstar': self.delta_star_over_Lstar,
            'theta_over_Lstar': self.theta_over_Lstar,
            'delta99_over_Lstar': self.delta99_over_Lstar,
            'shape_factor_H': self.shape_factor_H,
            'dU_dyL_wall': self.dU_dyL_wall,
            'dT_dyL_wall': self.dT_dyL_wall,
            'y_over_Lstar_max': float(self.y_over_Lstar[-1]),
            'y_over_delta_star_max': float(self.y_over_delta_star[-1]),
        }

    def to_csv(self, path, scale='L_star'):
        """Write the tabulated profile to a CSV-like whitespace table.

        ``'#'``-prefixed header lines carry the JSON metadata (conditions and
        integral scalars) plus an explicit statement of every column's
        normalization.  Derivative columns are expressed in the requested
        ``scale``.

        Parameters
        ----------
        path : str or Path
            Output file path.
        scale : {'L_star', 'delta_star'}, optional
            Wall-normal derivative scale for the ``dU``/``d2U``/``dT``/``d2T``
            columns.  Defaults to ``'L_star'`` because that scale is
            independent of the wall condition.

        Returns
        -------
        Path
            The written file path.
        """
        if scale not in {'delta_star', 'L_star'}:
            raise ValueError("scale must be 'delta_star' or 'L_star'")
        path = Path(path)

        if scale == 'L_star':
            dU, d2U = self.dU_dyL, self.d2U_dyL2
            dT, d2T = self.dT_dyL, self.d2T_dyL2
        else:
            ds = self.delta_star_over_Lstar
            dU, d2U = self.dU_dyL * ds, self.d2U_dyL2 * ds**2
            dT, d2T = self.dT_dyL * ds, self.d2T_dyL2 * ds**2

        table = np.column_stack([
            self.eta, self.y_over_Lstar, self.y_over_delta_star,
            self.U, dU, d2U, self.T, dT, d2T,
            self.rho, self.mu, self.kappa, self.Pr_local,
        ])

        deriv_var = 'y/L*' if scale == 'L_star' else 'y/delta*'
        header_lines = [
            'pymack compressible flat-plate boundary-layer profile '
            '(self-similar solution)',
            'normalization: U/U_e, T/T_e, rho/rho_e, mu/mu_e, '
            'kappa = kappa*/(cp*mu_e), Pr_local = mu/kappa',
            "note: for 'power_law'/'sutherland' transport kappa = (mu/mu_e)/Pr; "
            "for 'mack' it is the Appendix-A conductivity law (not mu/Pr)",
            f'derivatives dU, d2U, dT, d2T taken with respect to {deriv_var}; '
            'L* = sqrt(nu_e x / U_e)',
            'metadata: ' + json.dumps(self.to_dict()),
            'columns: ' + ' '.join(_CSV_COLUMNS),
        ]
        np.savetxt(path, table, fmt='%.16e', header='\n'.join(header_lines))
        return path

    def dimensionalize(self, edge_state, *, x_m=None, R_L=None,
                       rho_e_kg_m3=None):
        """Map the nondimensional profile to SI units at a streamwise station.

        Parameters
        ----------
        edge_state : pymack.scales.DimensionalEdgeState
            Physical edge state (``U_e`` in m/s, ``nu_e`` in m^2/s).
        x_m : float, optional
            Streamwise station in meters.  Exactly one of ``x_m``/``R_L``.
        R_L : float, optional
            Mack Reynolds number ``R_L = U_e L*/nu_e = sqrt(Re_x)``.
        rho_e_kg_m3 : float, optional
            Edge density.  Required for dimensional density and viscosity
            (``mu_e = rho_e * nu_e``); without it those outputs are ``None``.
            :class:`~pymack.scales.DimensionalEdgeState` deliberately carries
            no density, so this is never inferred from a gas constant.

        Returns
        -------
        DimensionalBoundaryLayer
            Dimensional profile and thickness record at the station.

        Warns
        -----
        UserWarning
            If ``edge_state.T_e`` is set and differs from this profile's
            ``T_edge_K`` by more than 1%; dimensional temperatures always use
            the generator's own ``T_edge_K``.
        """
        if (x_m is None) == (R_L is None):
            raise ValueError("specify exactly one of 'x_m' or 'R_L'")
        if x_m is not None:
            if x_m <= 0.0:
                raise ValueError('x_m must be positive')
            R_L = float(x_mm_to_R_L(1000.0 * float(x_m), edge_state))
        R_L = float(R_L)
        if R_L <= 0.0:
            raise ValueError('R_L must be positive')

        if edge_state.T_e is not None:
            rel = abs(edge_state.T_e - self.T_edge_K) / self.T_edge_K
            if rel > 0.01:
                warnings.warn(
                    f'edge_state.T_e = {edge_state.T_e:.2f} K differs from the '
                    f'profile T_edge_K = {self.T_edge_K:.2f} K by '
                    f'{100.0 * rel:.1f}%; dimensional temperatures use '
                    'T_edge_K.',
                    stacklevel=2,
                )

        L_star_m = float(lstar_m_from_R_L(R_L, edge_state))
        x_station_m = float(R_L_to_x_m(R_L, edge_state))

        if rho_e_kg_m3 is not None:
            if rho_e_kg_m3 <= 0.0:
                raise ValueError('rho_e_kg_m3 must be positive')
            mu_e = float(rho_e_kg_m3) * edge_state.nu_e
            rho_si = self.rho * float(rho_e_kg_m3)
            mu_si = self.mu * mu_e
        else:
            rho_si = None
            mu_si = None

        return DimensionalBoundaryLayer(
            edge_state=edge_state,
            x_m=x_station_m,
            R_L=R_L,
            Re_x=R_L**2,
            L_star_m=L_star_m,
            delta_star_m=self.delta_star_over_Lstar * L_star_m,
            theta_m=self.theta_over_Lstar * L_star_m,
            delta99_m=self.delta99_over_Lstar * L_star_m,
            T_wall_K=self.T_wall_K,
            T_edge_K=self.T_edge_K,
            y_m=self.y_over_Lstar * L_star_m,
            U_m_per_s=self.U * edge_state.U_e,
            T_K=self.T * self.T_edge_K,
            rho_kg_m3=rho_si,
            mu_Pa_s=mu_si,
            source_metadata=self.to_dict(),
        )

    def __repr__(self):
        wall = (f"{self.wall_bc}, Tw/Te={self.Tw_over_Te:.3f}")
        return (
            f'BoundaryLayerResult(Ma={self.Ma:g}, {wall}, '
            f'model={self.viscosity_model!r}, '
            f'delta*/L*={self.delta_star_over_Lstar:.4f}, '
            f'H={self.shape_factor_H:.3f})'
        )


@dataclass(eq=False)
class DimensionalBoundaryLayer:
    """SI-unit boundary-layer profile at one streamwise station.

    Produced by :meth:`BoundaryLayerResult.dimensionalize`.  ``rho_kg_m3``
    and ``mu_Pa_s`` are ``None`` unless an edge density was supplied.
    """

    edge_state: DimensionalEdgeState
    x_m: float
    R_L: float
    Re_x: float
    L_star_m: float
    delta_star_m: float
    theta_m: float
    delta99_m: float
    T_wall_K: float
    T_edge_K: float
    y_m: np.ndarray
    U_m_per_s: np.ndarray
    T_K: np.ndarray
    rho_kg_m3: np.ndarray | None = None
    mu_Pa_s: np.ndarray | None = None
    source_metadata: dict = field(default_factory=dict)

    def to_dict(self):
        """Return the JSON-serializable station record (scalars only)."""
        return {
            'edge_state': self.edge_state.to_dict(),
            'x_m': float(self.x_m),
            'R_L': float(self.R_L),
            'Re_x': float(self.Re_x),
            'L_star_m': float(self.L_star_m),
            'delta_star_m': float(self.delta_star_m),
            'theta_m': float(self.theta_m),
            'delta99_m': float(self.delta99_m),
            'T_wall_K': float(self.T_wall_K),
            'T_edge_K': float(self.T_edge_K),
            'has_density': self.rho_kg_m3 is not None,
            'source_metadata': self.source_metadata,
        }

    def to_csv(self, path):
        """Write the dimensional profile table (SI units).

        Columns are ``y_m  U_m_per_s  T_K`` plus ``rho_kg_m3  mu_Pa_s`` when
        an edge density was supplied.  ``'#'`` header lines carry the JSON
        station record.

        Returns
        -------
        Path
            The written file path.
        """
        path = Path(path)
        columns = ['y_m', 'U_m_per_s', 'T_K']
        arrays = [self.y_m, self.U_m_per_s, self.T_K]
        if self.rho_kg_m3 is not None:
            columns += ['rho_kg_m3', 'mu_Pa_s']
            arrays += [self.rho_kg_m3, self.mu_Pa_s]
        header_lines = [
            'pymack dimensional boundary-layer profile (SI units)',
            'metadata: ' + json.dumps(self.to_dict()),
            'columns: ' + ' '.join(columns),
        ]
        np.savetxt(path, np.column_stack(arrays), fmt='%.16e',
                   header='\n'.join(header_lines))
        return path
