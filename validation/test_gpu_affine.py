"""Affine operator cache certification (pymack-gpu slice 05).

Constructs an :class:`~pymack.gpu.affine.AffineOperatorCache` for EVERY
production operator family by probing the validated CPU assemblers at the
production drivers' real constants, and gates each on the mandatory held-out
self-verification residual (<= 1e-12).  Also certifies the failure path (a
structural parameter varying with the probe parameters must raise
:class:`~pymack.gpu.affine.AffineExtractionError`), pruning correctness
(dead monomials such as the omega/Re cross term come out numerically zero),
the measured 3D basis correction (probing (alpha, beta) monomials FAILS;
(k, cos psi, sin psi) is exact -- design_redteam.md), fingerprint stability,
and the exactness of the power-of-2 equilibration.

Production family provenance (constants copied from the drivers, cited
per family below):

- Ozgen fig3 grids: verification/mixed_mode/ozgen_fig3/_refdigitize/
  build_firstmode_grid.py (N=200, YMF_BY_MACH) + discrete_mode.py
  (two domain heights per node, L* scale).
- Mack fig10.4: verification/compute_mack_fig10_4.py (N_BY_MACH,
  Y_MAX_BY_MACH, ALPHA_GRID, PSI_GRID, DEFAULT_R_SWEEPS, 3D solver +
  wall/Dirichlet BCs, lambda_mu_ratio=0.0, L*).
- Mack fig10.6 (the Mack-enthalpy 2D form in production):
  verification/compute_mack_fig10_6.py (solve_temporal_compressible;
  the sweep uses the per-Mach N_BY_MACH/Y_MAX_BY_MACH tables via
  _N_for/_ymax_for, overriding the N_DEFAULT=110/Y_MAX_DEFAULT=30 module
  constants; ALPHA_SCAN, DEFAULT_R_SWEEPS, lambda_mu_ratio=0.0, L*).
- Malik (1990) case 5: verification/compute_malik1990_cases35.py (the
  beta=0 case of the pair -> solve_temporal_compressible at N=280,
  y_max=140, Pr=0.7, isothermal disturbance BC, lambda_mu_ratio=1.2, L*;
  a fixed-point anchor, probed over a bracketing box).
- Ma & Zhong: verification/second_mode/mazhong2003_m4p5/
  compute_mazhong_m4p5.py + trace_mazhong_curves.py (spatial QEP, N=130,
  y_max=40, lambda_mu_ratio=1.2, L*).
- Mach-6 eN: scripts/compute_mach6_growth_nfactor.py (Ozgen temporal form,
  N=64 default; the optional convergence check re-solves each node at
  max(40, N - convergence_delta), whose floor is N=40 -- see the family
  builder for the accurate citation; delta* scale).
- Cone: verification/second_mode/cone_sivasubramanian_fasel_2015/
  compute_cone_sf2015.py (spatial QEP, N=90, y_max=40, lambda_mu_ratio=1.2,
  L*, Taylor-Maccoll edge state).

Runs entirely GPU-less; ``pymack.gpu.affine`` and ``pymack.gpu.assemblers``
are pure numpy.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pytest

import pymack
from pymack import CompressibleBlasiusProfile, cone as pymack_cone
from pymack.scales import (
    DimensionalEdgeState,
    delta_star_over_lstar,
    frequency_khz_to_F,
)
from pymack.gpu.affine import (
    AffineExtractionError,
    AffineOperatorCache,
    ParameterBasis,
)
from pymack.gpu import assemblers as gpu_asm

REPO = Path(__file__).resolve().parents[1]
HELD_OUT_RTOL = 1e-12


# ===========================================================================
# Profile builders (cached: several families share a profile)
# ===========================================================================


@lru_cache(maxsize=None)
def _ozgen_profile(mach):
    # discrete_mode.py:_profile -- make_flatplate_profile(Ma), all defaults.
    return pymack.make_flatplate_profile(float(mach))


@lru_cache(maxsize=None)
def _mack_profile(mach):
    # compute_mack_fig10_4.py:make_profile
    return pymack.make_mack_profile(float(mach), condition="table_11_1")


@lru_cache(maxsize=None)
def _mazhong_profile():
    # compute_mazhong_m4p5.py:build_profile (adiabatic base flow)
    t_rec = 65.15 * (1.0 + 0.5 * (1.4 - 1.0) * 0.72**0.5 * 4.5**2)
    return CompressibleBlasiusProfile(
        Ma=4.5, T_edge=65.15, T_wall=t_rec, gamma=1.4, Pr=0.72,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=110.33, n_points=3000, eta_max=20.0,
    )


@lru_cache(maxsize=None)
def _mach6_en_profile():
    # compute_mach6_growth_nfactor.py:_worker_init with argparse defaults:
    # ma=6, t_edge=288, tw_over_te=None (adiabatic), gas=nitrogen -> S=111.0,
    # profile_points=3000, eta_max=40.
    return pymack.make_flatplate_profile(
        6.0, T_edge=288.0, T_wall=None, gamma=1.4, Pr=0.72, S=111.0,
        n_points=3000, eta_max=40.0,
    )


@lru_cache(maxsize=None)
def _malik_case5_profile():
    # compute_malik1990_cases35.py:66-73 with the malik_case5 config
    # (M=10, T0=4200 degR, adiabatic mean wall, Sutherland S=198.6/1.8 K,
    # GAMMA, PR = 1.4, 0.7 -- note Pr=0.7, not 0.72).
    gamma, pr = 1.4, 0.7
    T_edge = (4200.0 * 5.0 / 9.0) / (1.0 + 0.5 * (gamma - 1.0) * 10.0**2)
    t_rec = T_edge * (1.0 + 0.5 * (gamma - 1.0) * pr**0.5 * 10.0**2)
    return CompressibleBlasiusProfile(
        Ma=10.0, T_edge=T_edge, T_wall=t_rec, gamma=gamma, Pr=pr,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=198.6 / 1.8, n_points=6000, eta_max=70.0,
    )


@lru_cache(maxsize=None)
def _cone_profile():
    # compute_cone_sf2015.py: post-shock Taylor-Maccoll edge state
    return CompressibleBlasiusProfile(
        Ma=5.356, T_edge=63.78, T_wall=300.0, gamma=1.4, Pr=0.72,
        wall_bc="isothermal", viscosity_model="sutherland",
        sutherland_S=110.4, n_points=3000, eta_max=20.0,
    )


def _cone_edge_state():
    U_E, NU_E = 857.5, 6.0628e-5
    return DimensionalEdgeState(
        U_e=U_E, nu_e=NU_E, T_e=63.78, M_e=5.356, gamma=1.4, gas="air",
        unit_reynolds_per_m=U_E / NU_E,
    )


# ===========================================================================
# Production family definitions
# ===========================================================================


class Family(NamedTuple):
    name: str
    kind: str  # 'spatial' | 'temporal2d' | 'temporal3d'
    build: Callable


def _ozgen_family(ymf):
    # build_firstmode_grid.py:46-51: N=200, YMF_BY_MACH[4]=(35., 45.);
    # discrete_mode.py:83-87: y_max = ymf * delta*/L*, length_scale='L_star',
    # solve_temporal_2d defaults (isothermal, Pr=0.72, gamma=1.4).
    # Sweep box: GRID[4] -- Re in [1200, 5500], alpha in [0.006, 0.11] (L*).
    prof = _ozgen_profile(4.0)
    d = delta_star_over_lstar(prof)
    bundle = gpu_asm.temporal_ozgen_2d_assembler(
        prof, N=200, y_max=ymf * d, wall_bc="isothermal",
        length_scale="L_star", Ma=4.0, Pr=0.72, gamma=1.4,
    )
    box = {"alpha": (0.006, 0.11), "invRe": (1.0 / 5500.0, 1.0 / 1200.0)}
    return bundle, box


_FIG104 = {
    # compute_mack_fig10_4.py:90-132 -- ALPHA_GRID / PSI_GRID (degrees) /
    # DEFAULT_R_SWEEPS / N_BY_MACH / Y_MAX_BY_MACH per Mach panel.
    4.5: dict(N=130, y_max=42.0, alpha=(0.020, 0.100), psi=(45.0, 66.0),
              R=(200.0, 2000.0)),
    5.8: dict(N=150, y_max=64.0, alpha=(0.015, 0.080), psi=(45.0, 66.0),
              R=(220.0, 2000.0)),
    7.0: dict(N=170, y_max=86.0, alpha=(0.012, 0.065), psi=(45.0, 66.0),
              R=(240.0, 2000.0)),
    10.0: dict(N=200, y_max=150.0, alpha=(0.010, 0.050), psi=(42.0, 63.0),
               R=(450.0, 2000.0)),
}


def _fig104_family(mach):
    # first_mode_growth (compute_mack_fig10_4.py:166-171): 3D assembly +
    # apply_wall_bc_3d (isothermal default) + Dirichlet freestream,
    # length_scale='L_star', lambda_mu_ratio=0.0, Pr=0.72, gamma=1.4.
    g = _FIG104[mach]
    prof = _mack_profile(mach)
    bundle = gpu_asm.temporal_3d_assembler(
        prof, N=g["N"], y_max=g["y_max"], wall_bc="isothermal",
        length_scale="L_star", Ma=mach, Pr=0.72, gamma=1.4,
        lambda_mu_ratio=0.0,
    )
    psi_lo, psi_hi = np.deg2rad(g["psi"][0]), np.deg2rad(g["psi"][1])
    # beta = alpha*tan(psi) -> k = alpha/cos(psi); bounding box of the sweep
    k_lo = g["alpha"][0] / np.cos(psi_lo)
    k_hi = g["alpha"][1] / np.cos(psi_hi)
    box = {
        "k": (float(k_lo), float(k_hi)),
        "psi": (float(psi_lo), float(psi_hi)),
        "invRe": (1.0 / g["R"][1], 1.0 / g["R"][0]),
    }
    return bundle, box


_FIG106 = {
    # compute_mack_fig10_6.py:77-108 -- DEFAULT_R_SWEEPS / ALPHA_SCAN /
    # Y_MAX_BY_MACH / N_BY_MACH.  The sweep resolves N and y_max through
    # _N_for/_ymax_for (lines 111-116, applied at line 225), i.e. the
    # per-Mach tables, NOT the N_DEFAULT=110 / Y_MAX_DEFAULT=30 constants.
    4.5: dict(N=120, y_max=40.0, alpha=(0.08, 0.40), R=(300.0, 2000.0)),
    10.0: dict(N=200, y_max=140.0, alpha=(0.02, 0.22), R=(300.0, 2000.0)),
}


def _fig106_family(mach):
    # second_mode_growth (compute_mack_fig10_6.py:132-134): the Mack-enthalpy
    # 2D form, solve_temporal_compressible(profile, alpha, R, M, 0.72, 1.4,
    # N=N, y_max=y_max, length_scale='L_star', lambda_mu_ratio=0.0) with the
    # solver's default isothermal disturbance wall BC.
    g = _FIG106[mach]
    prof = _mack_profile(mach)
    bundle = gpu_asm.temporal_2d_assembler(
        prof, N=g["N"], y_max=g["y_max"], wall_bc="isothermal",
        length_scale="L_star", Ma=mach, Pr=0.72, gamma=1.4,
        lambda_mu_ratio=0.0,
    )
    box = {
        "alpha": g["alpha"],
        "invRe": (1.0 / g["R"][1], 1.0 / g["R"][0]),
    }
    return bundle, box


def _malik_case5_family():
    # compute_malik1990_cases35.py: the beta=0 case runs
    # solve_temporal_compressible(prof, 0.12, 1000.0, 10.0, 0.7, 1.4, N=280,
    # y_max=140.0, wall_bc='isothermal', length_scale='L_star',
    # lambda_mu_ratio=1.2) (lines 86-89).  This is a single-point anchor
    # (alpha=0.12, R=1000), so the probe box brackets it: alpha 0.5x-1.5x,
    # R +-20% (the extraction is exact, the box only sets conditioning).
    bundle = gpu_asm.temporal_2d_assembler(
        _malik_case5_profile(), N=280, y_max=140.0, wall_bc="isothermal",
        length_scale="L_star", Ma=10.0, Pr=0.7, gamma=1.4,
        lambda_mu_ratio=1.2,
    )
    box = {"alpha": (0.06, 0.18), "invRe": (1.0 / 1200.0, 1.0 / 800.0)}
    return bundle, box


def _mazhong_family():
    # compute_mazhong_m4p5.py:43-56 + trace_mazhong_curves.py:26-31:
    # solve_spatial at N=130, y_max=40, isothermal disturbance BC,
    # lambda_mu_ratio=1.2, L*; grid R in [240, 2000], omega in
    # [0.010, 0.235] (both mode bands).
    bundle = gpu_asm.spatial_qep_assembler(
        _mazhong_profile(), N=130, y_max=40.0, wall_bc="isothermal",
        length_scale="L_star", Ma=4.5, Pr=0.72, gamma=1.4,
        lambda_mu_ratio=1.2,
    )
    box = {"omega": (0.010, 0.235), "invRe": (1.0 / 2000.0, 1.0 / 240.0)}
    return bundle, box


def _mach6_en_family(N):
    # compute_mach6_growth_nfactor.py argparse defaults: N=64, y_max_delta=10,
    # wall_bc='adiabatic', delta* scale (solve_temporal_2d default), grid
    # R_L in [100, 1200], alpha_L in [0.02, 0.40] converted to delta* units
    # (lines 83-95).  The OPTIONAL convergence check (only active when
    # --convergence-delta is nonzero; the argparse default is 0, line 329)
    # re-solves each node at max(40, N - convergence_delta) (lines 102-112).
    # N=40 is that code path's floor, so the N=64/N=40 pair bounds the N
    # values this driver runs at its default N.
    prof = _mach6_en_profile()
    d = delta_star_over_lstar(prof)
    bundle = gpu_asm.temporal_ozgen_2d_assembler(
        prof, N=N, y_max=10.0, wall_bc="adiabatic",
        length_scale="delta_star", Ma=6.0, Pr=0.72, gamma=1.4,
    )
    box = {
        "alpha": (0.02 * d, 0.40 * d),
        "invRe": (1.0 / (1200.0 * d), 1.0 / (100.0 * d)),
    }
    return bundle, box


def _cone_family():
    # compute_cone_sf2015.py: N=90, y_max=40, isothermal, lambda_mu_ratio=1.2,
    # L*; stations s in [120, 520] mm -> R_eq = sqrt(Re_s/3); frequencies
    # 150-400 kHz -> omega_L = F * R_eq.
    edge = _cone_edge_state()
    r_lo = float(pymack_cone.cone_s_mm_to_R_eq(120.0, edge))
    r_hi = float(pymack_cone.cone_s_mm_to_R_eq(520.0, edge))
    f_lo = float(frequency_khz_to_F(150.0, edge))
    f_hi = float(frequency_khz_to_F(400.0, edge))
    bundle = gpu_asm.spatial_qep_assembler(
        _cone_profile(), N=90, y_max=40.0, wall_bc="isothermal",
        length_scale="L_star", Ma=5.356, Pr=0.72, gamma=1.4,
        lambda_mu_ratio=1.2,
    )
    box = {
        "omega": (f_lo * r_lo, f_hi * r_hi),
        "invRe": (1.0 / r_hi, 1.0 / r_lo),
    }
    return bundle, box


def _slow(family):
    return pytest.param(family, id=family.name, marks=pytest.mark.slow)


def _fast(family):
    return pytest.param(family, id=family.name)


PRODUCTION_FAMILIES = [
    _slow(Family("ozgen_fig3_M4_short", "temporal2d",
                 lambda: _ozgen_family(35.0))),
    _slow(Family("ozgen_fig3_M4_tall", "temporal2d",
                 lambda: _ozgen_family(45.0))),
    _slow(Family("fig10_4_M4.5", "temporal3d", lambda: _fig104_family(4.5))),
    _slow(Family("fig10_4_M5.8", "temporal3d", lambda: _fig104_family(5.8))),
    _slow(Family("fig10_4_M7.0", "temporal3d", lambda: _fig104_family(7.0))),
    _slow(Family("fig10_4_M10.0", "temporal3d",
                 lambda: _fig104_family(10.0))),
    _fast(Family("fig10_6_M4.5", "temporal2d",
                 lambda: _fig106_family(4.5))),
    _slow(Family("fig10_6_M10.0", "temporal2d",
                 lambda: _fig106_family(10.0))),
    _slow(Family("malik1990_case5", "temporal2d", _malik_case5_family)),
    _fast(Family("mazhong_m4p5", "spatial", _mazhong_family)),
    _fast(Family("mach6_eN_N64", "temporal2d",
                 lambda: _mach6_en_family(64))),
    _fast(Family("mach6_eN_N40", "temporal2d",
                 lambda: _mach6_en_family(40))),
    _fast(Family("cone_sf2015", "spatial", _cone_family)),
]


@pytest.fixture(scope="module", params=PRODUCTION_FAMILIES,
                ids=lambda f: f.name)
def family_cache(request):
    """One cache per production family, built once and shared by the tests."""
    family = request.param
    bundle, box = family.build()
    cache = AffineOperatorCache.from_probe(
        bundle.assemble, bundle.basis, box, key=bundle.key,
        rtol=HELD_OUT_RTOL,
    )
    return family, cache


# ===========================================================================
# The production gate: held-out residual <= 1e-12 for EVERY family
# ===========================================================================


def test_family_heldout_gate(family_cache):
    family, cache = family_cache
    assert cache.max_heldout_residual <= HELD_OUT_RTOL, (
        f"{family.name}: held-out residual {cache.max_heldout_residual:.3e}"
    )
    # every matrix of the family individually passes
    for name in cache.names:
        assert max(cache.heldout_residuals[name]) <= HELD_OUT_RTOL
    # design-conditioning diagnostics recorded for slice 07 provenance
    assert np.isfinite(cache.diagnostics["design_cond_full"])
    for name in cache.names:
        cond_kept = cache.diagnostics["design_cond_kept"][name]
        assert np.isfinite(cond_kept) and cond_kept >= 1.0


def test_family_verify_at_center(family_cache):
    """Runtime spot-check API: direct assembly vs contraction at box center."""
    family, cache = family_cache
    assert cache.verify_at(**cache.center) <= HELD_OUT_RTOL


# ===========================================================================
# Pruning correctness (dead terms are numerically zero)
# ===========================================================================

# Measured kept sets (labels are canonically sorted within each monomial).
_EXPECT_KEPT = {
    "spatial": {
        "C0": {"1", "omega", "invRe"},
        "C1": {"1", "invRe"},
        "C2": {"invRe"},
    },
    "temporal2d": {
        "A": {"1", "alpha", "invRe", "alpha*invRe", "alpha^2*invRe"},
        "B": {"alpha"},
    },
    "temporal3d": {
        "A": {"1", "invRe", "cos(psi)", "cos(psi)*invRe", "sin(psi)",
              "invRe*sin(psi)", "k", "invRe*k", "cos(psi)*k",
              "cos(psi)*invRe*k", "invRe*k^2"},
        "B": {"cos(psi)*k"},
    },
}


def test_family_pruning(family_cache):
    family, cache = family_cache
    expected = _EXPECT_KEPT[family.kind]
    for name in cache.names:
        assert set(cache.kept_labels(name)) == expected[name], (
            f"{family.name}/{name}: kept {cache.kept_labels(name)}"
        )


def test_family_dead_terms_numerically_zero(family_cache):
    """Contract examples: the omega/Re cross term of the spatial QEP, the
    Re-free alpha^2 of the 2D temporal forms, and the (k*sin psi ~ beta)
    and Re-free k^2 members of the 3D basis are numerically zero."""
    family, cache = family_cache
    labels = cache.basis.labels()
    dead_by_kind = {
        "spatial": {"C0": ["invRe*omega"], "C1": ["invRe*omega", "omega"],
                    "C2": ["invRe*omega", "omega", "1"]},
        "temporal2d": {"A": ["alpha^2"],
                       "B": ["1", "alpha^2", "invRe"]},
        "temporal3d": {"A": ["k*sin(psi)", "k^2", "cos(psi)*k^2"],
                       "B": ["1", "k", "k^2", "sin(psi)", "invRe"]},
    }
    for name, dead_labels in dead_by_kind[family.kind].items():
        contrib = cache.term_contributions[name]
        scale = contrib.max()
        for lab in dead_labels:
            j = labels.index(lab)
            ratio = contrib[j] / scale
            assert ratio < 1e-10, (
                f"{family.name}/{name}: dead term {lab} has relative "
                f"box-scale contribution {ratio:.3e}"
            )


# ===========================================================================
# Equilibration: exact powers of two, reversible, nontrivial
# ===========================================================================


def _assert_power_of_two(arr):
    mant, _ = np.frexp(np.asarray(arr, dtype=float))
    assert np.all(arr > 0) and np.all(mant == 0.5)


def test_family_equilibration(family_cache):
    family, cache = family_cache
    row, col = cache.equilibration
    _assert_power_of_two(row)
    _assert_power_of_two(col)
    # nontrivial for these badly-scaled operators
    assert not (np.all(row == 1.0) and np.all(col == 1.0))

    central = cache.evaluate(**cache.center)
    once = cache.apply_equilibration(central)
    twice = cache.apply_equilibration(once)
    name = cache.names[0]
    # applying the stored scalings twice is NOT the same as once
    assert not np.array_equal(twice[name], once[name])
    # power-of-2 scaling is exact in binary FP: bitwise round-trip
    back = cache.remove_equilibration(once)
    for n in cache.names:
        assert np.array_equal(back[n], central[n])
    # the equilibrated magnitude matrix is balanced (row/col maxima ~ 1)
    magnitude = np.zeros((cache.nn, cache.nn))
    for n in cache.names:
        magnitude += np.abs(central[n])
    W = row[:, None] * magnitude * col[None, :]
    assert 0.25 <= W.max(axis=1).max() <= 4.0
    assert 0.25 <= W.max(axis=0).max() <= 4.0


# ===========================================================================
# Fingerprint and analytic derivatives
# ===========================================================================


def test_family_fingerprint_shape(family_cache):
    family, cache = family_cache
    fp = cache.fingerprint()
    assert isinstance(fp, str) and len(fp) == 64
    assert fp == cache.fingerprint()  # deterministic


def test_family_analytic_derivative(family_cache):
    """Analytic d(matrix)/d(param) from the basis vs central differences of
    the affine evaluation (exact for quadratics, O(h^2) for trig)."""
    family, cache = family_cache
    params = cache.basis.params
    point = dict(cache.center)
    for param in params:
        lo, hi = cache.probe_box[param]
        h = 1e-4 * (hi - lo)
        up = dict(point, **{param: point[param] + h})
        dn = dict(point, **{param: point[param] - h})
        analytic = cache.evaluate_derivative(param, **point)
        plus, minus = cache.evaluate(**up), cache.evaluate(**dn)
        for name in cache.names:
            fd = (plus[name] - minus[name]) / (2.0 * h)
            scale = np.linalg.norm(fd)
            err = np.linalg.norm(analytic[name] - fd)
            if scale == 0.0:
                assert np.linalg.norm(analytic[name]) < 1e-12
            else:
                assert err / scale < 1e-6, (
                    f"{family.name}/{name}: d/d{param} rel err {err/scale:.2e}"
                )


# ===========================================================================
# Failure paths (the gate must catch structural and basis errors)
# ===========================================================================


def test_ymax_variation_raises():
    """A structural parameter (y_max) varying with the probe parameters is
    NOT affine and must trip the held-out gate."""
    prof = _mach6_en_profile()
    d = delta_star_over_lstar(prof)

    def bad_assemble(alpha, invRe):
        bundle = gpu_asm.temporal_ozgen_2d_assembler(
            prof, N=40, y_max=10.0 * (1.0 + 0.15 * float(alpha)),
            wall_bc="adiabatic", length_scale="delta_star",
            Ma=6.0, Pr=0.72, gamma=1.4,
        )
        return bundle.assemble(alpha, invRe)

    box = {"alpha": (0.02 * d, 0.40 * d),
           "invRe": (1.0 / (1200.0 * d), 1.0 / (100.0 * d))}
    with pytest.raises(AffineExtractionError):
        AffineOperatorCache.from_probe(
            bad_assemble, gpu_asm.TEMPORAL_2D_BASIS, box, rtol=HELD_OUT_RTOL,
        )


def test_3d_alpha_beta_monomials_fail():
    """The red-team correction: the 3D wave-aligned operator is NOT
    polynomial in (alpha, beta) (measured 6.1e-4) -- probing those monomials
    must raise; the (k, cos psi, sin psi) basis on the same operator is
    exact (covered by the fig10.4 family gates)."""
    prof = _mack_profile(4.5)
    bundle = gpu_asm.temporal_3d_assembler(
        prof, N=48, y_max=42.0, wall_bc="isothermal", length_scale="L_star",
        Ma=4.5, Pr=0.72, gamma=1.4, lambda_mu_ratio=0.0,
    )

    def ab_assemble(alpha, beta, invRe):
        k = float(np.hypot(alpha, beta))
        psi = float(np.arctan2(beta, alpha))
        return bundle.assemble(k=k, psi=psi, invRe=invRe)

    ab_basis = ParameterBasis.from_monomials(
        ("alpha", "beta", "invRe"),
        [
            {**({"alpha": ap} if ap else {}), **({"beta": bp} if bp else {}),
             **({"invRe": 1} if rp else {})}
            for ap in (0, 1, 2)
            for bp in (0, 1, 2)
            for rp in (0, 1)
            if ap + bp <= 2
        ],
    )
    # the (alpha, psi) sweep region in (alpha, beta): beta = alpha*tan(psi)
    psi_lo, psi_hi = np.deg2rad(45.0), np.deg2rad(66.0)
    box = {
        "alpha": (0.020, 0.100),
        "beta": (0.020 * np.tan(psi_lo), 0.100 * np.tan(psi_hi)),
        "invRe": (1.0 / 2000.0, 1.0 / 200.0),
    }
    with pytest.raises(AffineExtractionError):
        AffineOperatorCache.from_probe(
            ab_assemble, ab_basis, box, rtol=HELD_OUT_RTOL,
        )


def test_synthetic_nonaffine_raises():
    rng = np.random.default_rng(3)
    K0, K1 = rng.standard_normal((2, 4, 4))

    def assemble(alpha, invRe):
        return {"A": (K0 + np.exp(alpha) * K1).astype(complex)}

    basis = ParameterBasis.from_monomials(
        ("alpha", "invRe"), [{}, {"alpha": 1}, {"invRe": 1}]
    )
    with pytest.raises(AffineExtractionError):
        AffineOperatorCache.from_probe(
            assemble, basis, {"alpha": (0.1, 2.0), "invRe": (1e-4, 1e-2)},
        )


# ===========================================================================
# Fingerprint semantics (cheap real family: eN N=40)
# ===========================================================================


def test_fingerprint_stability_and_key_sensitivity():
    bundle, box = _mach6_en_family(40)
    build = lambda key: AffineOperatorCache.from_probe(  # noqa: E731
        bundle.assemble, bundle.basis, box, key=key, rtol=HELD_OUT_RTOL,
    )
    c1 = build(bundle.key)
    c2 = build(bundle.key)
    assert c1.fingerprint() == c2.fingerprint()  # bitwise-stable rebuild
    mutated = dict(bundle.key, N=bundle.key["N"] + 1)
    c3 = build(mutated)
    assert c3.fingerprint() != c1.fingerprint()  # key is hashed


def test_fingerprint_carries_probe_box_extent():
    """A cache reused outside its validated range must not alias."""
    rng = np.random.default_rng(19)
    K0 = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    K1 = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))

    def assemble(alpha, invRe):
        return {"A": K0 + alpha * K1}

    basis = ParameterBasis.from_monomials(
        ("alpha", "invRe"), [{}, {"alpha": 1}]
    )
    key = {"same": "structural-inputs"}
    c1 = AffineOperatorCache.from_probe(
        assemble, basis, {"alpha": (0.1, 0.2), "invRe": (1e-4, 2e-4)},
        key=key, rtol=HELD_OUT_RTOL,
    )
    c2 = AffineOperatorCache.from_probe(
        assemble, basis, {"alpha": (0.1, 0.3), "invRe": (1e-4, 2e-4)},
        key=key, rtol=HELD_OUT_RTOL,
    )
    assert c1.fingerprint() != c2.fingerprint()


def test_cache_key_hashes_baseflow_arrays():
    """The profile fingerprint hashes sample_baseflow OUTPUT arrays, so two
    physically different profiles at identical solver settings must key
    differently (profile mutations invalidate)."""
    b_adiabatic, _ = _mach6_en_family(40)
    prof_cold = pymack.make_flatplate_profile(
        6.0, T_edge=288.0, T_wall=200.0, gamma=1.4, Pr=0.72, S=111.0,
        n_points=3000, eta_max=40.0,
    )
    b_cold = gpu_asm.temporal_ozgen_2d_assembler(
        prof_cold, N=40, y_max=10.0, wall_bc="adiabatic",
        length_scale="delta_star", Ma=6.0, Pr=0.72, gamma=1.4,
    )
    assert (b_cold.key["profile_fingerprint"]
            != b_adiabatic.key["profile_fingerprint"])


# ===========================================================================
# ParameterBasis unit tests + synthetic exact recovery
# ===========================================================================


def test_parameter_basis_labels_and_values():
    basis = ParameterBasis.from_monomials(
        ("k", "psi"), [{}, {"k": 2}, {"k": 1, "cos(psi)": 1}]
    )
    assert basis.labels() == ("1", "k^2", "cos(psi)*k")
    pts = np.array([[2.0, np.pi / 3.0]])
    V = basis.evaluate_monomials(pts)
    assert np.allclose(V, [[1.0, 4.0, 2.0 * 0.5]])
    dV = basis.evaluate_monomial_derivatives(pts, "psi")
    assert np.allclose(dV, [[0.0, 0.0, -2.0 * np.sin(np.pi / 3.0)]])
    dVk = basis.evaluate_monomial_derivatives(pts, "k")
    assert np.allclose(dVk, [[0.0, 4.0, 0.5]])


def test_parameter_basis_validation():
    with pytest.raises(ValueError):
        ParameterBasis.from_monomials(("a",), [{"b": 1}])  # unknown symbol
    with pytest.raises(ValueError):
        ParameterBasis.from_monomials(("a",), [{"tan(a)": 1}])  # bad func
    with pytest.raises(ValueError):
        ParameterBasis.from_monomials(("a",), [{}, {}])  # duplicate
    with pytest.raises(ValueError):
        ParameterBasis.from_monomials(("a", "a"), [{}])  # dup param


def test_synthetic_exact_recovery_and_pruning():
    rng = np.random.default_rng(7)
    K = rng.standard_normal((3, 5, 5)) + 1j * rng.standard_normal((3, 5, 5))

    def assemble(alpha, invRe):
        # spans {1, alpha, alpha*invRe}; declared basis adds a dead invRe
        return {"A": K[0] + alpha * K[1] + alpha * invRe * K[2]}

    basis = ParameterBasis.from_monomials(
        ("alpha", "invRe"),
        [{}, {"alpha": 1}, {"invRe": 1}, {"alpha": 1, "invRe": 1}],
    )
    box = {"alpha": (0.5, 2.0), "invRe": (1e-4, 1e-2)}
    cache = AffineOperatorCache.from_probe(assemble, basis, box, key={"t": 1})
    assert cache.max_heldout_residual < 1e-13
    assert set(cache.kept_labels("A")) == {"1", "alpha", "alpha*invRe"}
    assert cache.pruned_labels("A") == ("invRe",)
    # recovered tensors match the ground truth
    for lab, truth in [("1", K[0]), ("alpha", K[1]), ("alpha*invRe", K[2])]:
        j = list(cache.kept_labels("A")).index(lab)
        assert np.allclose(cache.terms["A"][j], truth, atol=1e-12)
    # evaluate() reproduces direct assembly
    got = cache.evaluate(alpha=1.3, invRe=3e-3)["A"]
    want = assemble(alpha=1.3, invRe=3e-3)["A"]
    assert np.allclose(got, want, rtol=0, atol=1e-13)
    assert cache.verify_at(alpha=0.9, invRe=7e-3) < 1e-13


def test_scalars_batch_shapes():
    bundle, box = _mach6_en_family(40)
    cache = AffineOperatorCache.from_probe(
        bundle.assemble, bundle.basis, box, key=bundle.key,
    )
    alphas = np.linspace(*box["alpha"], 5)
    invres = np.linspace(*box["invRe"], 3)
    vals = cache.scalars(
        {"alpha": alphas[:, None], "invRe": invres[None, :]}
    )
    assert vals.shape == (5, 3, cache.basis.n_terms)
    # spot value: monomial alpha^2*invRe at [2, 1]
    j = cache.basis.labels().index("alpha^2*invRe")
    assert np.isclose(vals[2, 1, j], alphas[2] ** 2 * invres[1])
    # contraction through kept columns == evaluate()
    w = vals[2, 1][cache.kept["A"]]
    A = np.tensordot(w, cache.terms["A"], axes=(0, 0))
    ref = cache.evaluate(alpha=float(alphas[2]), invRe=float(invres[1]))["A"]
    assert np.allclose(A, ref, rtol=0, atol=1e-12)


# ===========================================================================
# Import hygiene: GPU-less CI runs everything; pymack stays CuPy-free
# ===========================================================================


def test_import_pymack_is_cupy_free():
    code = (
        "import sys; import pymack; "
        "bad = [m for m in sys.modules if m == 'cupy' "
        "or m.startswith('cupy.') or m.startswith('pymack.gpu')]; "
        "assert not bad, bad; print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(REPO), check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_affine_modules_are_pure_numpy():
    """No cupy import statement (or attribute use) in the pure-numpy files;
    prose mentions in docstrings are fine."""
    import ast

    for mod in ("affine.py", "assemblers.py"):
        src = (REPO / "pymack" / "gpu" / mod).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("cupy"), (
                    f"{mod} imports {name}; it must stay pure numpy"
                )


def test_gpu_import_guard():
    import pymack.gpu as gpu

    assert isinstance(gpu.is_available(), bool)
    assert issubclass(gpu.GpuNotAvailable, RuntimeError)
    if gpu.is_available():
        gpu.require()  # must not raise
    else:
        with pytest.raises(gpu.GpuNotAvailable, match="cupy|CuPy|GPU"):
            gpu.require()


def test_gpu_package_importable_without_cupy():
    """The guard path: pymack.gpu (and the pure-numpy modules) must import
    cleanly when cupy is unimportable."""
    code = "\n".join([
        "import builtins",
        "real = builtins.__import__",
        "def deny(name, *a, **k):",
        "    if name == 'cupy' or name.startswith('cupy.'):",
        "        raise ImportError('cupy disabled for test')",
        "    return real(name, *a, **k)",
        "builtins.__import__ = deny",
        "import pymack.gpu as g",
        "assert g.cupy is None and g.is_available() is False",
        "try:",
        "    g.require()",
        "except g.GpuNotAvailable:",
        "    pass",
        "else:",
        "    raise SystemExit('require() must raise')",
        "import pymack.gpu.affine, pymack.gpu.assemblers",
        "print('guarded-ok')",
    ])
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(REPO), check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "guarded-ok" in out.stdout
