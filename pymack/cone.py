"""
Sharp-cone (Mangler) dimensional bookkeeping for Mack-style LST.

Scope
-----
This module maps a SHARP cone at ZERO incidence onto the equivalent flat-plate
problem through the Mangler transformation, with transverse-curvature operator
terms OMITTED.  Under that scope the local-parallel LST eigenproblem at cone
surface station ``s`` is *exactly* the flat-plate eigenproblem evaluated at the
equivalent Reynolds number ``R_eq`` -- no solver or operator changes are
needed, only the station bookkeeping below.

Derivation of the Mangler factor (``MANGLER_FACTOR = 3``)
---------------------------------------------------------
The Mangler transformation maps an axisymmetric boundary layer (transverse
curvature neglected) to a two-dimensional one:

    x_bar = (1/L^2) * integral_0^s r_w(s')^2 ds',     y_bar = (r_w / L) * y.

For a sharp cone the wetted radius grows linearly along the surface ray,
``r_w(s) = s * sin(theta_c)`` with half-angle ``theta_c``, so

    x_bar = s^3 * sin^2(theta_c) / (3 L^2).

Choosing the local radius as the reference length, ``L = r_w(s)``, the
half-angle cancels EXACTLY:

    x_eq = s / 3        (independent of theta_c).

Hence ``Re_x_eq = Re_s / 3`` and, with Mack's ``R = sqrt(Re_x)``,

    R_eq = sqrt(Re_s / 3) = R_s / sqrt(3),

which is the classic result ``delta_cone(s) = delta_plate(s) / sqrt(3)``
(White, *Viscous Fluid Flow*; Schlichting).  Because a sharp-cone conical
flow has zero edge pressure gradient, the self-similar flat-plate profile
family applies unchanged.

The half-angle therefore enters the stability problem ONLY through the
post-shock edge state (e.g. from a Taylor-Maccoll solution) that the user
supplies; it never appears in the station mapping.

Reuse contract with :mod:`pymack.scales`
----------------------------------------
With ``x_eq = s/3`` the cone similarity length is

    L*_eq = sqrt(nu_e * x_eq / U_e) = nu_e * R_eq / U_e,

i.e. the SAME formula as the plate.  Consequently every per-``L*`` converter
in :mod:`pymack.scales` is correct for the cone UNCHANGED provided the ``R``
argument is ``R_eq``:

    ``lstar_m_from_R_L``, ``sigma_L_to_per_m``, ``sigma_L_to_per_mm``,
    ``alpha_L_to_per_m``, ``alpha_L_to_per_mm``, ``wavelength_L_to_mm``,

and ``frequency_khz_to_F`` / ``F_to_frequency_khz`` are geometry-independent
outright (they involve only ``U_e`` and ``nu_e``).  The thin ``cone_*``
wrappers below exist so cone scripts read unambiguously; they delegate
verbatim.  The only converters whose MATH changes are the station maps:

    s(R_eq)  = 3 * nu_e * R_eq**2 / U_e          (vs x = nu_e R^2 / U_e)
    ds       = 6 * L*_eq * dR_eq                 (vs dx = 2 L* dR)
    N(s)     = integral sigma_phys ds
             = integral 6 * sigma_L dR_eq
             = integral 2*sqrt(3) * sigma_L dR_s
             = 3 * N_plate over the same R_eq window.

At a fixed PHYSICAL station ``s`` the cone ``L*`` is ``1/sqrt(3)`` times the
plate value, so the second-mode frequency at fixed ``omega_L`` is ``sqrt(3)``
times higher on the cone (``CONE_FREQUENCY_RATIO_AT_SAME_S``).

Validity and traps
------------------
* ``delta << r_local = s*sin(theta_c)`` is REQUIRED (transverse curvature
  omitted).  Check it at your lowest station; near the tip it always fails.
* Sharp tip, zero incidence only.  Nose bluntness (entropy layer) or angle of
  attack breaks the plate/cone similarity entirely.
* The :class:`pymack.scales.DimensionalEdgeState` you pass in MUST be the
  POST-SHOCK cone edge state (Taylor-Maccoll), not the freestream.  Nothing
  here can validate that for you.
* In cone outputs the solver's ``R_L`` column means ``R_eq``; comparing cone
  and plate runs "at the same R_L" compares different physical stations
  (factor 3 in distance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pymack.scales import (
    DimensionalEdgeState,
    F_to_frequency_khz,
    R_L_to_x_m,
    R_L_to_x_mm,
    alpha_L_to_per_m,
    alpha_L_to_per_mm,
    frequency_khz_to_F,
    lstar_m_from_R_L,
    sigma_L_to_per_m,
    sigma_L_to_per_mm,
    wavelength_L_to_mm,
    x_mm_to_R_L,
)


#: Sharp-cone Mangler factor: ``x_eq = s / MANGLER_FACTOR``.  The half-angle
#: cancels exactly in the derivation (see module docstring), so this is a
#: pure number, not a function of geometry.
MANGLER_FACTOR = 3.0

#: Flat-plate streamwise multiplier in this repository's convention
#: (``R = sqrt(Re_x)`` => ``dx = 2 L* dR``), as used by
#: ``scripts/postprocess_spatial_amplification.py --r-convention sqrt_re_x``.
PLATE_DX_OVER_LSTAR_PER_DR = 2.0

#: Second-mode frequency ratio cone/plate at the SAME physical station ``s``
#: and fixed ``omega_L`` (because ``L*_cone(s) = L*_plate(s)/sqrt(3)``).
CONE_FREQUENCY_RATIO_AT_SAME_S = math.sqrt(3.0)


@dataclass(frozen=True)
class ConeGeometry:
    """Sharp-cone LST bookkeeping via the Mangler transformation.

    ``half_angle_deg`` is METADATA ONLY: it never enters the station mapping
    (the half-angle cancels exactly; see module docstring).  It influences the
    physics only through the post-shock edge state the user supplies.
    """

    half_angle_deg: float | None = None

    def __post_init__(self):
        if self.half_angle_deg is not None and not (
            0.0 < float(self.half_angle_deg) < 90.0
        ):
            raise ValueError("half_angle_deg must lie in (0, 90) when provided")

    def to_dict(self) -> dict:
        return {
            "type": "sharp_cone_mangler",
            "mangler_factor": MANGLER_FACTOR,
            "half_angle_deg": (
                None if self.half_angle_deg is None else float(self.half_angle_deg)
            ),
            "half_angle_role": (
                "metadata only; enters physics solely via the user-supplied "
                "post-shock edge state (Taylor-Maccoll)"
            ),
            "transverse_curvature_terms": "omitted",
            "validity": "delta << r_local = s*sin(half_angle); sharp tip; zero incidence",
            "R_L_meaning": "R_eq = sqrt(Re_s/3) = R_s/sqrt(3)",
            "edge_state_meaning": (
                "post-shock cone edge (e.g. Taylor-Maccoll), supplied by user"
            ),
        }


# Shared scalar/array passthrough — single source of truth in pymack.scales.
from pymack.scales import _maybe_scalar  # noqa: E402


# ---------------------------------------------------------------------------
# Station mapping: the only converters whose math differs from pymack.scales.
# ---------------------------------------------------------------------------

def R_eq_from_R_s(R_s):
    """Equivalent-plate ``R_eq = R_s / sqrt(3)`` from the surface ``R_s = sqrt(Re_s)``."""
    out = np.asarray(R_s, dtype=float) / math.sqrt(MANGLER_FACTOR)
    return _maybe_scalar(out, R_s)


def R_s_from_R_eq(R_eq):
    """Surface ``R_s = sqrt(Re_s) = R_eq * sqrt(3)`` from the equivalent-plate ``R_eq``."""
    out = np.asarray(R_eq, dtype=float) * math.sqrt(MANGLER_FACTOR)
    return _maybe_scalar(out, R_eq)


def cone_R_eq_to_s_m(R_eq, edge_state: DimensionalEdgeState):
    """Cone surface distance ``s = 3 * nu_e * R_eq**2 / U_e`` in meters."""
    out = MANGLER_FACTOR * np.asarray(R_L_to_x_m(R_eq, edge_state), dtype=float)
    return _maybe_scalar(out, R_eq)


def cone_R_eq_to_s_mm(R_eq, edge_state: DimensionalEdgeState):
    """Cone surface distance along the ray in millimeters."""
    out = MANGLER_FACTOR * np.asarray(R_L_to_x_mm(R_eq, edge_state), dtype=float)
    return _maybe_scalar(out, R_eq)


def cone_s_mm_to_R_eq(s_mm, edge_state: DimensionalEdgeState):
    """Equivalent-plate ``R_eq = sqrt(Re_s / 3)`` from cone surface distance in mm."""
    scaled = np.asarray(s_mm, dtype=float) / MANGLER_FACTOR
    out = np.asarray(x_mm_to_R_L(scaled, edge_state), dtype=float)
    return _maybe_scalar(out, s_mm)


# ---------------------------------------------------------------------------
# Verbatim delegations: identical math given R = R_eq (see reuse contract).
# Provided so cone-facing code reads unambiguously; do NOT add cone factors
# here -- the geometry lives entirely in the station maps and the N multiplier.
# ---------------------------------------------------------------------------

def cone_lstar_m_from_R_eq(R_eq, edge_state: DimensionalEdgeState):
    """Cone ``L*_eq = nu_e * R_eq / U_e`` in meters (same formula as the plate)."""
    return lstar_m_from_R_L(R_eq, edge_state)


def cone_frequency_khz_to_F(frequency_khz, edge_state: DimensionalEdgeState):
    """``F = 2*pi*f*nu_e/U_e**2`` -- geometry-independent; delegates verbatim."""
    return frequency_khz_to_F(frequency_khz, edge_state)


def cone_F_to_frequency_khz(F, edge_state: DimensionalEdgeState):
    """Inverse of :func:`cone_frequency_khz_to_F`; geometry-independent."""
    return F_to_frequency_khz(F, edge_state)


def cone_sigma_L_to_per_m(sigma_L, R_eq, edge_state: DimensionalEdgeState):
    """``sigma_phys = sigma_L / L*_eq`` in 1/m (identical to plate at R = R_eq)."""
    return sigma_L_to_per_m(sigma_L, R_eq, edge_state)


def cone_sigma_L_to_per_mm(sigma_L, R_eq, edge_state: DimensionalEdgeState):
    """``sigma_phys`` in 1/mm (identical to plate at R = R_eq)."""
    return sigma_L_to_per_mm(sigma_L, R_eq, edge_state)


def cone_alpha_L_to_per_m(alpha_L, R_eq, edge_state: DimensionalEdgeState):
    """``L*``-scaled wavenumber in 1/m (identical to plate at R = R_eq)."""
    return alpha_L_to_per_m(alpha_L, R_eq, edge_state)


def cone_alpha_L_to_per_mm(alpha_L, R_eq, edge_state: DimensionalEdgeState):
    """``L*``-scaled wavenumber in 1/mm (identical to plate at R = R_eq)."""
    return alpha_L_to_per_mm(alpha_L, R_eq, edge_state)


def cone_wavelength_L_to_mm(wavelength_L, R_eq, edge_state: DimensionalEdgeState):
    """Wavelength in mm (identical to plate at R = R_eq)."""
    return wavelength_L_to_mm(wavelength_L, R_eq, edge_state)


# ---------------------------------------------------------------------------
# N-factor: the cone path integral.
# ---------------------------------------------------------------------------

def cone_n_factor_multiplier(plate_multiplier: float = PLATE_DX_OVER_LSTAR_PER_DR) -> float:
    """Streamwise multiplier ``m`` in ``N = integral m * sigma_L dR_eq`` for the cone.

    ``ds = 6 L*_eq dR_eq`` versus the plate's ``dx = 2 L* dR``, so the cone
    multiplier is always ``MANGLER_FACTOR`` times the plate multiplier of the
    R-convention in use -- never a bare constant:

    * repo default ``R = sqrt(Re_x)`` (plate multiplier 2.0)  -> 6.0
    * ``R = sqrt(2 Re_x)``            (plate multiplier 1.0)  -> 3.0
    """
    plate_multiplier = float(plate_multiplier)
    if plate_multiplier <= 0.0:
        raise ValueError("plate_multiplier must be positive")
    return MANGLER_FACTOR * plate_multiplier


def cone_n_factor(
    sigma_L,
    R_eq,
    *,
    plate_multiplier: float = PLATE_DX_OVER_LSTAR_PER_DR,
    clip_negative: bool = True,
):
    """Integrate the cone N-factor ``N(s) = integral sigma_phys ds`` over an ``R_eq`` path.

    Uses ``ds = 6 L*_eq dR_eq``, i.e. trapezoidal integration of
    ``cone_n_factor_multiplier(plate_multiplier) * sigma_L`` over ``R_eq``.
    Equals ``MANGLER_FACTOR`` times the plate N over the same ``R_eq`` window.

    Parameters
    ----------
    sigma_L : array_like
        Dimensionless spatial growth ``-Im(alpha_L)`` sampled along the path.
    R_eq : array_like
        Strictly increasing equivalent-plate Reynolds path (the solver's R grid).
    plate_multiplier : float
        Plate ``dx/L*`` per ``dR`` of the R-convention in use (2.0 for the
        repo's ``R = sqrt(Re_x)``).
    clip_negative : bool
        If true (transition-envelope convention), damped segments do not
        reduce N.  If false, signed growth is integrated.

    Returns
    -------
    dict
        ``{"R_eq", "N", "sigma_L", "multiplier", "clip_negative"}`` with ``N``
        the cumulative N-factor array, ``N[0] = 0``.
    """
    R = np.asarray(R_eq, dtype=float)
    sigma = np.asarray(sigma_L, dtype=float)
    if R.ndim != 1 or sigma.shape != R.shape:
        raise ValueError("sigma_L and R_eq must be 1-D arrays of the same length")
    if R.size == 0:
        raise ValueError("sigma_L and R_eq must be non-empty")
    if np.any(np.diff(R) <= 0.0):
        raise ValueError("R_eq path must be strictly increasing")

    multiplier = cone_n_factor_multiplier(plate_multiplier)
    integrand = multiplier * sigma
    if clip_negative:
        integrand = np.maximum(integrand, 0.0)

    N_vals = np.zeros(R.size, dtype=float)
    for i in range(1, R.size):
        N_vals[i] = N_vals[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * (R[i] - R[i - 1])

    return {
        "R_eq": R,
        "N": N_vals,
        "sigma_L": sigma,
        "multiplier": float(multiplier),
        "clip_negative": bool(clip_negative),
    }
