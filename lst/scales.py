"""
Length-scale conversions for Mack-style stability calculations.

The collocation solver stores compressible mean flows on a ``y / delta*`` grid,
but many paper figures and tables use Mack's Falkner-Skan scale

    L* = sqrt(nu_e x / U_e).

This module makes those conversions explicit instead of letting each chapter
script handle them ad hoc.
"""

from __future__ import annotations

import numpy as np


_FIRST_DERIVATIVE_KEYS = (
    'dU', 'dT', 'drho', 'dmu', 'dkappa',
)
_SECOND_DERIVATIVE_KEYS = (
    'd2U', 'd2T',
)


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
