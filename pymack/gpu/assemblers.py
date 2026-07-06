"""Probe adapters binding the affine cache to the validated CPU assemblers.

PURE NUMPY.  This is the only GPU-side file that knows CPU assembly
internals; nothing here re-implements physics.  Each adapter pre-binds the
structural inputs (profile, grid, BCs, gas constants) and exposes an
``assemble(**probe_params)`` callable that takes ONLY the sweep parameters,
plus the declared :class:`~pymack.gpu.affine.ParameterBasis` and the
structural cache key.

The four production assemblers (design_architecture.md section 2):

- ``pymack.solver._assemble_spatial_qep`` -- spatial QEP, already probe-able,
  returns BC-applied (C0, C1, C2, y).
- ``pymack.solver._assemble_temporal_2d_evp`` -- Mack enthalpy 2D temporal
  form (slice-04 extraction, bitwise-certified).
- ``pymack.temporal_solver._assemble_temporal_ozgen_2d_evp`` -- Ozgen 2D
  temporal form (slice-04 extraction, bitwise-certified).
- ``pymack.solver.assemble_temporal_compressible_3d_evp`` +
  ``apply_wall_bc_3d`` + ``apply_dirichlet_freestream_bc_3d`` -- 3D
  wave-aligned temporal form (Dirichlet freestream: the production fig10.4
  path; the asymptotic freestream BC is NOT affine and stays on CPU).

Bases (measured, design_redteam.md): the declared sets below are generic
supersets -- dead terms prune to numerically-zero K tensors, missing terms
would trip the held-out gate.  The 3D basis MUST be (k, cos psi, sin psi) x
{1, invRe}: the operator is not polynomial in (alpha, beta) (measured probe
residual 6.1e-4 vs 9.1e-14 in the wave-aligned variables).

Cache keys hash the ``sample_baseflow(profile, y, length_scale)`` OUTPUT
arrays -- the actual assembly input -- so any profile mutation invalidates
the cache correctly.
"""

from __future__ import annotations

import hashlib
from typing import Callable, NamedTuple

import numpy as np

from .. import solver as _solver
from .. import temporal_solver as _temporal_solver
from ..equations import DEFAULT_LAMBDA_MU_RATIO
from ..scales import sample_baseflow
from ..spectral import chebyshev_D, physical_derivatives
from .affine import ParameterBasis

__all__ = [
    "AdapterBundle",
    "SPATIAL_QEP_BASIS",
    "TEMPORAL_2D_BASIS",
    "TEMPORAL_3D_BASIS",
    "spatial_qep_assembler",
    "temporal_2d_assembler",
    "temporal_ozgen_2d_assembler",
    "temporal_3d_assembler",
]


class AdapterBundle(NamedTuple):
    """What a probe adapter hands to ``AffineOperatorCache.from_probe``."""

    assemble: Callable
    basis: ParameterBasis
    key: dict
    y: np.ndarray


# -- declared bases ----------------------------------------------------------

#: Spatial QEP over (omega, invRe).  Measured: C0 = K1 + w*Kw + r*Kr,
#: C1 = K1' + r*Kr', C2 = r*K'' -- the omega*invRe cross term is dead and
#: prunes; probing it generically keeps the extraction basis-declared.
SPATIAL_QEP_BASIS = ParameterBasis.from_monomials(
    ("omega", "invRe"),
    [
        {},
        {"omega": 1},
        {"invRe": 1},
        {"omega": 1, "invRe": 1},
    ],
)

#: 2D temporal (both Mack and Ozgen forms) over (alpha, invRe):
#: {1, a, a^2} x {1, invRe}.  Measured: B = a*B1; the Re-free a^2 term of A
#: is dead in both formulations (a^2 always enters through visc/cond ~ 1/Re).
TEMPORAL_2D_BASIS = ParameterBasis.from_monomials(
    ("alpha", "invRe"),
    [
        {},
        {"alpha": 1},
        {"alpha": 2},
        {"invRe": 1},
        {"alpha": 1, "invRe": 1},
        {"alpha": 2, "invRe": 1},
    ],
)

#: 3D wave-aligned temporal over (k, psi, invRe): the full generic tensor
#: {1, k, k^2} x {1, cos(psi), sin(psi)} x {1, invRe}.  NEVER (alpha, beta)
#: monomials (measured failure 6.1e-4 vs 9.1e-14).  Dead members (e.g.
#: Re-free k^2, k*sin(psi) ~ beta) prune to zero.
TEMPORAL_3D_BASIS = ParameterBasis.from_monomials(
    ("k", "psi", "invRe"),
    [
        {
            **({"k": kp} if kp else {}),
            **({trig: 1} if trig else {}),
            **({"invRe": 1} if rp else {}),
        }
        for kp in (0, 1, 2)
        for trig in (None, "cos(psi)", "sin(psi)")
        for rp in (0, 1)
    ],
)


# -- shared helpers -----------------------------------------------------------


def _resolve_y_max(y_max, Ma):
    """Mirror the solvers' default: 6.0 above Ma=2, else 12.0."""
    if y_max is None:
        return 6.0 if Ma > 2.0 else 12.0
    return float(y_max)


def _grid(N, y_max, L):
    D_eta = chebyshev_D(N)
    return physical_derivatives(D_eta, y_max, N, L)


def _profile_fingerprint(profile, y, length_scale):
    """sha256 of the sample_baseflow output arrays -- the assembly input."""
    bf = sample_baseflow(profile, y, length_scale)
    h = hashlib.sha256()
    arr_y = np.ascontiguousarray(np.asarray(y))
    h.update(b"y")
    h.update(arr_y.dtype.str.encode())
    h.update(str(arr_y.shape).encode())
    h.update(arr_y.tobytes())
    for name in sorted(bf):
        arr = np.ascontiguousarray(np.asarray(bf[name]))
        h.update(name.encode())
        h.update(arr.dtype.str.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _cache_key(adapter, profile, y, *, N, y_max, L, wall_bc, length_scale,
               Ma, Pr, gamma, lambda_mu_ratio, basis, extra=None):
    key = {
        "adapter": adapter,
        "profile_fingerprint": _profile_fingerprint(profile, y, length_scale),
        "N": int(N),
        "y_max": float(y_max),
        "L": None if L is None else float(L),
        "wall_bc": str(wall_bc),
        "length_scale": str(length_scale),
        "Ma": float(Ma),
        "Pr": float(Pr),
        "gamma": float(gamma),
        "lambda_mu_ratio": (
            None if lambda_mu_ratio is None else float(lambda_mu_ratio)
        ),
        "basis": list(basis.labels()),
    }
    if extra:
        key.update(extra)
    return key


# -- adapters -----------------------------------------------------------------


def spatial_qep_assembler(profile, *, N, y_max=None, L=None,
                          wall_bc="isothermal", length_scale="delta_star",
                          Ma, Pr, gamma,
                          lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Adapter for ``pymack.solver._assemble_spatial_qep``.

    ``assemble(omega, invRe)`` returns the BC-applied ``{'C0','C1','C2'}`` of
    the spatial quadratic EVP (C0 + alpha*C1 + alpha^2*C2) phi = 0.  Note the
    QEP eigen-parameter derivative dT/dalpha = C1 + 2*alpha*C2 comes from the
    QEP structure itself; the cache's ``evaluate_derivative`` covers the
    probe parameters (omega, invRe).
    """
    y_max = _resolve_y_max(y_max, Ma)
    y, _D1, _D2 = _grid(N, y_max, L)

    def assemble(omega, invRe):
        C0, C1, C2, _y = _solver._assemble_spatial_qep(
            profile, float(omega), 1.0 / float(invRe), Ma, Pr, gamma,
            N, y_max, L, wall_bc, length_scale, lambda_mu_ratio,
        )
        return {"C0": C0, "C1": C1, "C2": C2}

    key = _cache_key(
        "spatial_qep", profile, y, N=N, y_max=y_max, L=L, wall_bc=wall_bc,
        length_scale=length_scale, Ma=Ma, Pr=Pr, gamma=gamma,
        lambda_mu_ratio=lambda_mu_ratio, basis=SPATIAL_QEP_BASIS,
    )
    return AdapterBundle(assemble, SPATIAL_QEP_BASIS, key, y)


def temporal_2d_assembler(profile, *, N, y_max=None, L=None,
                          wall_bc="isothermal", length_scale="delta_star",
                          Ma, Pr, gamma,
                          lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO):
    """Adapter for the Mack-enthalpy 2D temporal form (slice-04 extraction).

    Pre-binds the exact discretized inputs ``(bf, y, D1, D2)`` that
    ``solve_temporal_compressible`` builds, then probes
    ``pymack.solver._assemble_temporal_2d_evp``; ``assemble(alpha, invRe)``
    returns the BC-applied ``{'A', 'B'}`` with A phi = c B phi.
    """
    y_max = _resolve_y_max(y_max, Ma)
    y, D1, D2 = _grid(N, y_max, L)
    bf, D1, D2 = _solver._scaled_compressible_problem(
        profile, y, D1, D2, length_scale
    )

    def assemble(alpha, invRe):
        A, B = _solver._assemble_temporal_2d_evp(
            bf, y, D1, D2, float(alpha), 1.0 / float(invRe), Ma, Pr, gamma,
            wall_bc=wall_bc, lambda_mu_ratio=lambda_mu_ratio,
        )
        return {"A": A, "B": B}

    key = _cache_key(
        "temporal_2d_mack", profile, y, N=N, y_max=y_max, L=L,
        wall_bc=wall_bc, length_scale=length_scale, Ma=Ma, Pr=Pr,
        gamma=gamma, lambda_mu_ratio=lambda_mu_ratio,
        basis=TEMPORAL_2D_BASIS,
    )
    return AdapterBundle(assemble, TEMPORAL_2D_BASIS, key, y)


def temporal_ozgen_2d_assembler(profile, *, N, y_max=None, L=None,
                                wall_bc="isothermal",
                                length_scale="delta_star",
                                Ma, Pr, gamma):
    """Adapter for the Ozgen 2D temporal form (slice-04 extraction).

    Pre-binds the exact discretized inputs that ``solve_temporal_2d`` builds,
    then probes ``pymack.temporal_solver._assemble_temporal_ozgen_2d_evp``;
    ``assemble(alpha, invRe)`` returns the BC-applied ``{'A', 'B'}``.
    """
    y_max = _resolve_y_max(y_max, Ma)
    y, D1, D2 = _grid(N, y_max, L)
    bf, D1, D2 = _temporal_solver._scaled_problem(profile, y, D1, D2,
                                                  length_scale)

    def assemble(alpha, invRe):
        A, B = _temporal_solver._assemble_temporal_ozgen_2d_evp(
            bf, y, D1, D2, float(alpha), 1.0 / float(invRe), Ma, Pr, gamma,
            wall_bc=wall_bc,
        )
        return {"A": A, "B": B}

    key = _cache_key(
        "temporal_2d_ozgen", profile, y, N=N, y_max=y_max, L=L,
        wall_bc=wall_bc, length_scale=length_scale, Ma=Ma, Pr=Pr,
        gamma=gamma, lambda_mu_ratio=None, basis=TEMPORAL_2D_BASIS,
    )
    return AdapterBundle(assemble, TEMPORAL_2D_BASIS, key, y)


def temporal_3d_assembler(profile, *, N, y_max=None, L=None,
                          wall_bc="isothermal", length_scale="delta_star",
                          Ma, Pr, gamma,
                          lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
                          include_spanwise_dissipation_coupling=True):
    """Adapter for the 3D wave-aligned temporal form with production BCs.

    ``assemble(k, psi, invRe)`` maps (k, psi) -> (alpha, beta) =
    (k cos psi, k sin psi), calls
    ``pymack.solver.assemble_temporal_compressible_3d_evp``, applies
    ``apply_wall_bc_3d`` + ``apply_dirichlet_freestream_bc_3d`` (the
    production fig10.4 BC stack), and returns ``{'A', 'B'}``.

    The probe parameters are (k, psi, invRe) because the operator is exactly
    affine in (k, cos psi, sin psi) but NOT polynomial in (alpha, beta) --
    probing (alpha, beta) monomials fails at 6.1e-4 (design_redteam.md).
    ``psi`` is in radians; probe boxes must keep ``k > 0``.
    """
    y_max = _resolve_y_max(y_max, Ma)
    y, _D1, _D2 = _grid(N, y_max, L)

    def assemble(k, psi, invRe):
        alpha = float(k) * float(np.cos(psi))
        beta = float(k) * float(np.sin(psi))
        A, B, _y, D1, n, _a, _b, _bf = (
            _solver.assemble_temporal_compressible_3d_evp(
                profile, alpha, beta, 1.0 / float(invRe), Ma, Pr, gamma,
                N=N, y_max=y_max, L=L,
                include_spanwise_dissipation_coupling=(
                    include_spanwise_dissipation_coupling
                ),
                length_scale=length_scale,
                lambda_mu_ratio=lambda_mu_ratio,
            )
        )
        _solver.apply_wall_bc_3d(A, B, D1, n, wall_bc=wall_bc)
        _solver.apply_dirichlet_freestream_bc_3d(A, B, n)
        return {"A": A, "B": B}

    key = _cache_key(
        "temporal_3d", profile, y, N=N, y_max=y_max, L=L, wall_bc=wall_bc,
        length_scale=length_scale, Ma=Ma, Pr=Pr, gamma=gamma,
        lambda_mu_ratio=lambda_mu_ratio, basis=TEMPORAL_3D_BASIS,
        extra={
            "freestream_bc": "dirichlet",
            "include_spanwise_dissipation_coupling": bool(
                include_spanwise_dissipation_coupling
            ),
        },
    )
    return AdapterBundle(assemble, TEMPORAL_3D_BASIS, key, y)
