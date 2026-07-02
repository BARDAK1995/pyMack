"""High-level facade: the one-obvious-way entry points for pyMack.

This module wraps the verified numerical kernels with a small, friendly
surface for the most common tasks::

    import pymack as pm

    bl   = pm.flat_plate(Ma=6.0)                      # self-similar base flow
    mode = pm.temporal_mode(bl, alpha=0.174, Re=5500)  # -> ModeResult
    mode.growth_rate, mode.phase_speed, mode.unstable

    mode = pm.spatial_mode(bl, omega=0.23, Re=1500)    # -> ModeResult
    mode.sigma                                          # spatial growth -Im(alpha)

Design notes
------------
* The facade never hides the kernels: everything here delegates to
  :mod:`pymack.temporal_solver`, :mod:`pymack.solver`, and
  :mod:`pymack.baseflow`, which remain fully public for power users.
* A :class:`ModeResult` is immutable and carries the eigenfunctions, the grid,
  and the exact parameters used, so a result is always reproducible from its
  own metadata.
* Mode selection applies the freestream-decay test described in
  ``docs/numerical_methods``: a genuine boundary-layer eigenfunction decays
  toward the freestream, continuous-spectrum artifacts do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baseflow import make_flatplate_profile
from .scales import delta_star_over_lstar
from .solver import solve_spatial
from .temporal_solver import solve_temporal_2d

__all__ = ['ModeResult', 'flat_plate', 'temporal_mode', 'spatial_mode']

#: Fraction of the peak eigenfunction amplitude allowed at the freestream
#: boundary for a mode to count as a discrete (physical) boundary-layer mode.
DEFAULT_EDGE_DECAY_RATIO = 0.1


@dataclass(frozen=True, repr=False)
class ModeResult:
    """A single stability eigenmode with its eigenfunctions and provenance.

    Attributes
    ----------
    kind : {'temporal', 'spatial'}
        Which eigenvalue problem produced the mode.
    eigenvalue : complex
        The raw eigenvalue: the phase speed ``c`` (temporal) or the
        streamwise wavenumber ``alpha`` (spatial).
    alpha, omega : complex
        Wavenumber and frequency of the mode (one of them is the fixed,
        real input; the other contains the eigenvalue).
    growth_rate : float
        Temporal growth ``omega_i = alpha * Im(c)`` or spatial growth
        ``sigma = -Im(alpha)``. Positive means unstable.
    phase_speed : float
        ``Re(c)`` (temporal) or ``omega / Re(alpha)`` (spatial). First
        (TS) modes sit near 0.3-0.6; second (Mack) modes near 1 - 1/Ma.
    y : ndarray
        Wall-normal grid (freestream first, wall last -- collocation order).
    u, v, T, p : ndarray
        Complex eigenfunction components on ``y``, normalized so that the
        largest ``|u|`` sample equals 1 with zero phase.
    params : dict
        Exact solver inputs (Re, Ma, N, y_max, wall_bc, length_scale, ...)
        for reproducibility.
    edge_ratio : float
        Freestream-to-peak amplitude ratio used by the decay test
        (small = well-confined mode).
    """

    kind: str
    eigenvalue: complex
    alpha: complex
    omega: complex
    growth_rate: float
    phase_speed: float
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    T: np.ndarray
    p: np.ndarray
    params: dict = field(default_factory=dict)
    edge_ratio: float = float('nan')

    @property
    def unstable(self) -> bool:
        """True when the growth rate is positive."""
        return self.growth_rate > 0.0

    @property
    def c(self) -> complex:
        """Complex phase speed ``omega / alpha``."""
        return complex(self.omega) / complex(self.alpha)

    @property
    def sigma(self) -> float:
        """Spatial growth rate ``-Im(alpha)`` (meaningful for spatial modes)."""
        return -float(np.imag(self.alpha))

    @property
    def omega_i(self) -> float:
        """Temporal growth rate ``Im(omega)`` (meaningful for temporal modes)."""
        return float(np.imag(self.omega))

    def __repr__(self) -> str:  # concise, physics-first
        tag = 'unstable' if self.unstable else 'stable'
        if self.kind == 'temporal':
            head = (f'c = {self.eigenvalue.real:.4f}{self.eigenvalue.imag:+.4f}j, '
                    f'omega_i = {self.growth_rate:+.3e}')
        else:
            head = (f'alpha = {self.eigenvalue.real:.4f}{self.eigenvalue.imag:+.4f}j, '
                    f'sigma = {self.growth_rate:+.3e}')
        return (f'ModeResult({self.kind}, {head}, c_r = {self.phase_speed:.3f}, '
                f'{tag}, Re = {self.params.get("Re")}, Ma = {self.params.get("Ma")})')


def flat_plate(Ma, *, T_wall_over_T_edge=None, T_edge=288.0, gamma=1.4,
               Pr=0.72, n_points=4000, eta_max=40.0):
    """Build a self-similar compressible flat-plate base flow.

    Parameters
    ----------
    Ma : float
        Edge Mach number.
    T_wall_over_T_edge : float, optional
        Isothermal wall temperature ratio ``Tw/Te``. ``None`` (default)
        gives an adiabatic wall.
    T_edge : float
        Edge temperature in Kelvin (enters the Sutherland-type transport laws).
    gamma, Pr : float
        Ratio of specific heats and edge Prandtl number.
    n_points, eta_max : int, float
        Similarity-grid resolution and truncation.

    Returns
    -------
    FlatPlateProfile
        Callable profile object: ``profile(y)`` samples the mean flow.
        Accepted by every solver in pyMack.
    """
    return make_flatplate_profile(
        Ma,
        T_edge=T_edge,
        T_wall=T_wall_over_T_edge,
        gamma=gamma,
        Pr=Pr,
        n_points=n_points,
        eta_max=eta_max,
    )


def _resolve_Ma(profile, Ma):
    if Ma is not None:
        return float(Ma)
    Ma_attr = getattr(profile, 'Ma', None)
    if Ma_attr is None:
        raise ValueError(
            'Ma was not given and the profile does not carry a .Ma attribute; '
            'pass Ma=... explicitly.'
        )
    return float(Ma_attr)


def _default_y_max(profile, length_scale, multiple=10.0):
    """Domain height as a multiple of the boundary-layer scale delta*."""
    if length_scale == 'L_star':
        return multiple * float(delta_star_over_lstar(profile))
    return multiple


def _split_eigenfunction(phi, n):
    u, v, T, p = (np.asarray(phi[k * n:(k + 1) * n]) for k in range(4))
    # Normalize: largest |u| sample -> 1 + 0j.
    pivot = u[int(np.argmax(np.abs(u)))]
    if pivot != 0:
        u, v, T, p = u / pivot, v / pivot, T / pivot, p / pivot
    return u, v, T, p


def _edge_ratio(phi, n, n_edge=4):
    """Freestream-to-peak amplitude ratio of the stacked eigenvector."""
    fields = np.abs(phi.reshape(4, n))
    profile_amp = fields.max(axis=0)
    peak = float(profile_amp.max())
    if peak == 0.0:
        return float('inf')
    return float(profile_amp[:n_edge].mean() / peak)


def _needs_stationarity(check_stationarity, guess):
    """'auto' -> check exactly when there is no guess to anchor the branch."""
    if check_stationarity == 'auto':
        return guess is None
    return bool(check_stationarity)


def _select_mode(eigs, vecs, n, guess, decay_ratio, reference_eigs=None,
                 stationarity_tol=2e-3):
    """Pick the physical mode.

    Acceptance mirrors the two tests of ``docs/numerical_methods``:

    1. *Freestream decay* -- the eigenfunction amplitude at the outer boundary
       must be a small fraction of its peak (``decay_ratio``).
    2. *Domain stationarity* (optional) -- when ``reference_eigs`` from a
       taller-domain solve are supplied, the eigenvalue must reappear there
       within ``stationarity_tol``; continuous-spectrum artifacts move with
       the box, genuine boundary-layer modes do not.

    With a ``guess`` the nearest acceptable eigenvalue is returned
    (continuation); otherwise the most unstable acceptable one.
    """
    order = range(len(eigs))
    passing = [k for k in order if _edge_ratio(vecs[:, k], n) < decay_ratio]
    if reference_eigs is not None and len(reference_eigs):
        ref = np.asarray(reference_eigs)
        passing = [k for k in passing
                   if np.min(np.abs(ref - eigs[k])) < stationarity_tol]
    pool = passing if passing else list(order)
    if guess is not None:
        k_best = min(pool, key=lambda k: abs(eigs[k] - guess))
    else:
        k_best = pool[0]  # inputs arrive sorted most-unstable-first
    return k_best, (k_best in passing)


def temporal_mode(profile, alpha, Re, *, Ma=None, c_guess=None, N=128,
                  y_max=None, wall_bc='isothermal', length_scale='L_star',
                  Pr=0.72, gamma=1.4,
                  decay_ratio=DEFAULT_EDGE_DECAY_RATIO,
                  check_stationarity='auto'):
    """Solve the temporal problem and return the discrete boundary-layer mode.

    Given a real wavenumber ``alpha``, finds the complex phase speed ``c`` of
    the physical (freestream-decaying) mode. Growth: ``omega_i = alpha*Im(c)``.

    Parameters
    ----------
    profile : callable
        Base-flow profile (e.g. from :func:`flat_plate`).
    alpha : float
        Real streamwise wavenumber, in ``length_scale`` units.
    Re : float
        Reynolds number in ``length_scale`` units (R = sqrt(Re_x) for L*).
    Ma : float, optional
        Edge Mach number; defaults to ``profile.Ma``.
    c_guess : complex, optional
        Select the returned eigenvalue nearest this guess (e.g. a
        continuation predictor); default picks the most unstable
        freestream-decaying mode.
    N, y_max, wall_bc, length_scale, Pr, gamma
        Discretization and physics options, passed to
        :func:`pymack.temporal_solver.solve_temporal_2d`. ``y_max`` defaults
        to ``10 * delta*/L*`` -- a domain that scales with the boundary layer.
    decay_ratio : float
        Freestream-decay acceptance threshold (see module docs).
    check_stationarity : bool or 'auto'
        Re-solve on a 30% taller domain and require the eigenvalue to
        persist -- rejects continuous-spectrum artifacts that decay but
        drift with the box. 'auto' (default) applies the test exactly when
        no ``c_guess`` anchors the branch; it doubles the solve cost.

    Returns
    -------
    ModeResult
    """
    Ma = _resolve_Ma(profile, Ma)
    if y_max is None:
        y_max = _default_y_max(profile, length_scale)
    eigs, vecs, y = solve_temporal_2d(
        profile, alpha, Re, Ma, Pr=Pr, gamma=gamma, N=N, y_max=y_max,
        wall_bc=wall_bc, length_scale=length_scale,
    )
    if len(eigs) == 0:
        raise RuntimeError('temporal solve returned no eigenvalues in the physical band')
    n = len(y)
    reference_eigs = None
    if _needs_stationarity(check_stationarity, c_guess):
        reference_eigs, _, _ = solve_temporal_2d(
            profile, alpha, Re, Ma, Pr=Pr, gamma=gamma, N=N,
            y_max=1.3 * y_max, wall_bc=wall_bc, length_scale=length_scale,
        )
    k, decayed = _select_mode(eigs, vecs, n, c_guess, decay_ratio,
                              reference_eigs=reference_eigs)
    c = complex(eigs[k])
    u, v, T, p = _split_eigenfunction(vecs[:, k], n)
    params = dict(Re=Re, Ma=Ma, alpha=float(alpha), N=N, y_max=float(y_max),
                  wall_bc=wall_bc, length_scale=length_scale, Pr=Pr,
                  gamma=gamma, decay_test_passed=decayed,
                  stationarity_checked=reference_eigs is not None)
    return ModeResult(
        kind='temporal', eigenvalue=c, alpha=complex(alpha),
        omega=complex(alpha) * c, growth_rate=float(alpha) * c.imag,
        phase_speed=c.real, y=y, u=u, v=v, T=T, p=p, params=params,
        edge_ratio=_edge_ratio(vecs[:, k], n),
    )


def spatial_mode(profile, omega, Re, *, Ma=None, alpha_guess=None, N=128,
                 y_max=None, wall_bc='isothermal', length_scale='L_star',
                 Pr=0.72, gamma=1.4, n_modes=20,
                 decay_ratio=DEFAULT_EDGE_DECAY_RATIO,
                 check_stationarity='auto'):
    """Solve the spatial problem and return the discrete boundary-layer mode.

    Given a real frequency ``omega``, finds the complex wavenumber ``alpha``
    of the physical mode. Growth: ``sigma = -Im(alpha)`` (positive = growing
    downstream); amplitude ratio integrates to the N-factor.

    Parameters mirror :func:`temporal_mode`; ``alpha_guess`` doubles as the
    shift-invert target of :func:`pymack.solver.solve_spatial`.

    Returns
    -------
    ModeResult
    """
    Ma = _resolve_Ma(profile, Ma)
    if y_max is None:
        y_max = _default_y_max(profile, length_scale)
    eigs, vecs, y = solve_spatial(
        profile, omega, Re, Ma, Pr, gamma, N=N, y_max=y_max,
        wall_bc=wall_bc, target_alpha=alpha_guess, n_modes=n_modes,
        length_scale=length_scale,
    )
    if len(eigs) == 0:
        raise RuntimeError('spatial solve returned no eigenvalues near the target')
    n = len(y)
    reference_eigs = None
    if _needs_stationarity(check_stationarity, alpha_guess):
        reference_eigs, _, _ = solve_spatial(
            profile, omega, Re, Ma, Pr, gamma, N=N, y_max=1.3 * y_max,
            wall_bc=wall_bc, target_alpha=alpha_guess, n_modes=n_modes,
            length_scale=length_scale,
        )
    k, decayed = _select_mode(eigs, vecs, n, alpha_guess, decay_ratio,
                              reference_eigs=reference_eigs)
    a = complex(eigs[k])
    u, v, T, p = _split_eigenfunction(vecs[:, k], n)
    params = dict(Re=Re, Ma=Ma, omega=float(omega), N=N, y_max=float(y_max),
                  wall_bc=wall_bc, length_scale=length_scale, Pr=Pr,
                  gamma=gamma, n_modes=n_modes, decay_test_passed=decayed,
                  stationarity_checked=reference_eigs is not None)
    return ModeResult(
        kind='spatial', eigenvalue=a, alpha=a, omega=complex(omega),
        growth_rate=-a.imag, phase_speed=float(omega) / a.real if a.real else float('nan'),
        y=y, u=u, v=v, T=T, p=p, params=params,
        edge_ratio=_edge_ratio(vecs[:, k], n),
    )
