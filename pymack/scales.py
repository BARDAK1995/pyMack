"""
Length-scale conversions for Mack-style stability calculations.

The collocation solver stores compressible mean flows on a ``y / delta*`` grid,
but many paper figures and tables use Mack's Falkner-Skan scale

    L* = sqrt(nu_e x / U_e).

This module makes those conversions explicit instead of letting each chapter
script handle them ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_FIRST_DERIVATIVE_KEYS = (
    'dU', 'dT', 'drho', 'dmu', 'dkappa',
)
_SECOND_DERIVATIVE_KEYS = (
    'd2U', 'd2T',
)


@dataclass(frozen=True)
class DimensionalEdgeState:
    """Physical edge state used to map Mack ``R_L``/``F`` to SI units.

    The LST solver remains nondimensional.  This object only supplies the
    flat-plate similarity scale

        L* = nu_e R_L / U_e,   x = nu_e R_L**2 / U_e,

    and the fixed-frequency relation

        F = omega nu_e / U_e**2 = 2*pi*f_Hz*nu_e/U_e**2.
    """

    U_e: float
    nu_e: float
    T_e: float | None = None
    M_e: float | None = None
    gamma: float = 1.4
    gas: str = "air"
    unit_reynolds_per_m: float | None = None

    def __post_init__(self):
        if self.U_e <= 0.0:
            raise ValueError("U_e must be positive")
        if self.nu_e <= 0.0:
            raise ValueError("nu_e must be positive")
        if self.T_e is not None and self.T_e <= 0.0:
            raise ValueError("T_e must be positive when provided")
        if self.M_e is not None and self.M_e <= 0.0:
            raise ValueError("M_e must be positive when provided")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")
        if self.unit_reynolds_per_m is not None and self.unit_reynolds_per_m <= 0.0:
            raise ValueError("unit_reynolds_per_m must be positive when provided")

    @property
    def computed_unit_reynolds_per_m(self) -> float:
        """Return ``U_e / nu_e`` in 1/m."""
        return float(self.U_e / self.nu_e)

    @property
    def unit_reynolds_consistency_error(self) -> float | None:
        """Relative mismatch between supplied and implied unit Reynolds number."""
        if self.unit_reynolds_per_m is None:
            return None
        return float(
            (self.computed_unit_reynolds_per_m - self.unit_reynolds_per_m)
            / self.unit_reynolds_per_m
        )

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "U_e_m_per_s": float(self.U_e),
            "nu_e_m2_per_s": float(self.nu_e),
            "T_e_K": None if self.T_e is None else float(self.T_e),
            "M_e": None if self.M_e is None else float(self.M_e),
            "gamma": float(self.gamma),
            "gas": self.gas,
            "unit_reynolds_per_m": (
                None
                if self.unit_reynolds_per_m is None
                else float(self.unit_reynolds_per_m)
            ),
            "computed_unit_reynolds_per_m": self.computed_unit_reynolds_per_m,
            "unit_reynolds_consistency_error": self.unit_reynolds_consistency_error,
        }


def _asarray(value):
    return np.asarray(value, dtype=float)


def _maybe_scalar(value, template):
    array = np.asarray(value)
    if np.isscalar(template):
        return float(array)
    return array


def lstar_m_from_R_L(R_L, edge_state: DimensionalEdgeState):
    """Return ``L*`` in meters for Mack ``R_L = U_e L* / nu_e``."""
    out = edge_state.nu_e * _asarray(R_L) / edge_state.U_e
    return _maybe_scalar(out, R_L)


def R_L_to_x_m(R_L, edge_state: DimensionalEdgeState):
    """Convert Mack ``R_L = sqrt(Re_x)`` to streamwise distance in meters."""
    R = _asarray(R_L)
    out = edge_state.nu_e * R**2 / edge_state.U_e
    return _maybe_scalar(out, R_L)


def R_L_to_x_mm(R_L, edge_state: DimensionalEdgeState):
    """Convert Mack ``R_L = sqrt(Re_x)`` to streamwise distance in millimeters."""
    out = 1000.0 * _asarray(R_L_to_x_m(R_L, edge_state))
    return _maybe_scalar(out, R_L)


def x_mm_to_R_L(x_mm, edge_state: DimensionalEdgeState):
    """Convert streamwise distance in millimeters to Mack ``R_L``."""
    x_m = 1.0e-3 * _asarray(x_mm)
    if np.any(x_m < 0.0):
        raise ValueError("x_mm must be non-negative")
    out = np.sqrt(edge_state.U_e * x_m / edge_state.nu_e)
    return _maybe_scalar(out, x_mm)


def frequency_khz_to_F(frequency_khz, edge_state: DimensionalEdgeState):
    """Convert cyclic frequency in kHz to Mack fixed-frequency parameter ``F``."""
    f_hz = 1000.0 * _asarray(frequency_khz)
    if np.any(f_hz < 0.0):
        raise ValueError("frequency_khz must be non-negative")
    out = 2.0 * math.pi * f_hz * edge_state.nu_e / edge_state.U_e**2
    return _maybe_scalar(out, frequency_khz)


def F_to_frequency_khz(F, edge_state: DimensionalEdgeState):
    """Convert Mack fixed-frequency parameter ``F`` to cyclic frequency in kHz."""
    out = _asarray(F) * edge_state.U_e**2 / (2.0 * math.pi * edge_state.nu_e) / 1000.0
    return _maybe_scalar(out, F)


def sigma_L_to_per_m(sigma_L, R_L, edge_state: DimensionalEdgeState):
    """Convert dimensionless spatial growth ``sigma_L=-Im(alpha_L)`` to 1/m."""
    out = _asarray(sigma_L) / _asarray(lstar_m_from_R_L(R_L, edge_state))
    return _maybe_scalar(out, sigma_L)


def sigma_L_to_per_mm(sigma_L, R_L, edge_state: DimensionalEdgeState):
    """Convert dimensionless spatial growth ``sigma_L`` to 1/mm."""
    out = _asarray(sigma_L_to_per_m(sigma_L, R_L, edge_state)) / 1000.0
    return _maybe_scalar(out, sigma_L)


def alpha_L_to_per_m(alpha_L, R_L, edge_state: DimensionalEdgeState):
    """Convert an ``L*``-scaled wavenumber to 1/m."""
    out = _asarray(alpha_L) / _asarray(lstar_m_from_R_L(R_L, edge_state))
    return _maybe_scalar(out, alpha_L)


def alpha_L_to_per_mm(alpha_L, R_L, edge_state: DimensionalEdgeState):
    """Convert an ``L*``-scaled wavenumber to 1/mm."""
    out = _asarray(alpha_L_to_per_m(alpha_L, R_L, edge_state)) / 1000.0
    return _maybe_scalar(out, alpha_L)


def wavelength_L_to_mm(wavelength_L, R_L, edge_state: DimensionalEdgeState):
    """Convert wavelength expressed in ``L*`` units to millimeters."""
    out = _asarray(wavelength_L) * _asarray(lstar_m_from_R_L(R_L, edge_state)) * 1000.0
    return _maybe_scalar(out, wavelength_L)


def delta_star_over_lstar(profile) -> float:
    """Return ``delta* / L*`` for a compressible Mack-style base flow."""
    if hasattr(profile, '_delta_star'):
        return float(profile._delta_star)
    raise AttributeError(
        'profile does not expose _delta_star, so delta*/L* is unavailable'
    )


def momentum_thickness_over_lstar(profile) -> float:
    """Return ``theta / L*`` for a compressible flat-plate base flow."""
    if hasattr(profile, '_theta'):
        return float(profile._theta)
    raise AttributeError(
        'profile does not expose _theta, so theta/L* is unavailable'
    )


def lstar_to_delta_star(value, delta_over_l):
    """Convert an ``L*``-based scalar/array to ``delta*`` scaling."""
    return np.asarray(value) / float(delta_over_l)


def delta_star_to_lstar(value, delta_over_l):
    """Convert a ``delta*``-based scalar/array to ``L*`` scaling."""
    return np.asarray(value) * float(delta_over_l)


def eta_to_lstar(profile, eta):
    """Map the similarity coordinate ``eta`` to physical ``y/L*``."""
    if not hasattr(profile, '_eta') or not hasattr(profile, '_y_L'):
        raise AttributeError('profile does not expose eta and y/L* data')
    return np.interp(np.asarray(eta, dtype=float), profile._eta, profile._y_L)


def lstar_to_eta(profile, y_lstar):
    """Map physical ``y/L*`` to the similarity coordinate ``eta``."""
    if not hasattr(profile, '_eta') or not hasattr(profile, '_y_L'):
        raise AttributeError('profile does not expose eta and y/L* data')
    return np.interp(np.asarray(y_lstar, dtype=float), profile._y_L, profile._eta)


def eta_to_delta_star(profile, eta):
    """Map the similarity coordinate ``eta`` to solver ``y/delta*``."""
    delta_over_l = delta_star_over_lstar(profile)
    return lstar_to_delta_star(eta_to_lstar(profile, eta), delta_over_l)


def delta_star_to_eta(profile, y_delta_star):
    """Map solver ``y/delta*`` coordinates back to the similarity variable."""
    delta_over_l = delta_star_over_lstar(profile)
    return lstar_to_eta(profile, delta_star_to_lstar(y_delta_star, delta_over_l))


def sample_baseflow(baseflow, y, length_scale='delta_star'):
    """Sample a base-flow profile at ``y`` in the requested length scale.

    This is the single canonical resampling helper used by every solver in
    pyMack. Profiles natively store fields on a ``y / delta*`` grid; for
    ``length_scale='L_star'`` the query points are converted with
    ``delta*/L*`` and the wall-normal derivatives rescaled accordingly.

    Parameters
    ----------
    baseflow : callable
        Profile object; calling it with an array of heights returns the
        base-flow dictionary (``U``, ``dU``, ``T``, ... ).
    y : float or array
        Wall-normal position(s), in units of ``length_scale``.
    length_scale : {'delta_star', 'L_star'}
        Length scale of ``y`` (and of the returned derivatives).

    Returns
    -------
    dict
        Base-flow fields sampled at ``y``, with derivatives expressed in
        ``length_scale`` units.
    """
    y = np.asarray(y, dtype=float)
    if length_scale == 'delta_star':
        return baseflow(y)
    if length_scale != 'L_star':
        raise ValueError("length_scale must be 'delta_star' or 'L_star'")
    delta_over_l = delta_star_over_lstar(baseflow)
    sampled = baseflow(y / delta_over_l)
    return rescale_baseflow_derivatives(sampled, delta_over_l, target_scale='L_star')


def rescale_baseflow_derivatives(baseflow_dict, delta_over_l, target_scale):
    """Rescale wall-normal derivatives in a sampled base-flow dictionary.

    Parameters
    ----------
    baseflow_dict : dict
        Output of ``profile(y)`` sampled on the solver's ``y / delta*`` grid.
    delta_over_l : float
        Ratio ``delta* / L*``.
    target_scale : {"delta_star", "L_star"}
        Desired wall-normal derivative scale.

    Returns
    -------
    dict
        Copy of ``baseflow_dict`` with derivative quantities expressed in the
        requested wall-normal scale.
    """
    if target_scale not in {'delta_star', 'L_star'}:
        raise ValueError("target_scale must be 'delta_star' or 'L_star'")

    if target_scale == 'delta_star':
        return {k: np.array(v, copy=True) for k, v in baseflow_dict.items()}

    scale = float(delta_over_l)
    out = {k: np.array(v, copy=True) for k, v in baseflow_dict.items()}

    for key in _FIRST_DERIVATIVE_KEYS:
        if key in out:
            out[key] = out[key] / scale
    for key in _SECOND_DERIVATIVE_KEYS:
        if key in out:
            out[key] = out[key] / scale**2

    return out
