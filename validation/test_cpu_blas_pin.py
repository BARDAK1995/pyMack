"""CPU BLAS pinning opt-in for temporal_sweep (cpu-blas-pin mini-slice).

Contract (per CPU_PIN_BRIEF):
- DEFAULT (cpu_blas_threads=None / unset) is bitwise-identical to historical
  path: all existing fixtures and np.array_equal checks in
  test_sweep_cpu_backend.py stay green.
- When enabled, workers run with the requested BLAS thread count (verified
  *from inside* a worker process and recorded in SweepResult.meta as
  'cpu_blas_threads_effective').
- The pinned config is a *different floating-point path*. We only claim
  semantic identity: on a small grid, pinned vs default agree on converged
  values to rtol=1e-9 (and identical NaN/None-status).
- Plumbing works on Windows spawn (parent env mutation is exception-safe
  and restored; children see it at import time).
- No new hard dependencies (threadpoolctl is optional; graceful fallback).
- No edits to per-point solvers (temporal_solver.py, solver.py, ...).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymack import make_flatplate_profile
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep


def _small_grid():
    # ~6x4 = 24 nodes, N=31 as suggested in brief for semantic test
    alphas = np.linspace(0.05, 0.20, 6)
    Res = np.linspace(600.0, 1200.0, 4)
    return alphas, Res


def _ts_family():
    return (CBand(float('-inf'), 0.45, ci_abs_max=0.05, label='TS'),)


@pytest.mark.parametrize("workers", [2, 3])
def test_pinned_vs_default_semantic_identity(workers):
    """Pinned (1-thread BLAS) vs default: same convergence mask + values ~1e-9 rel.

    This is *semantic* identity only; we do not assert bitwise or ulp equality.
    The test forces a small pool so that workers are used and inside-worker
    verification is exercised.
    """
    profile = make_flatplate_profile(2.0)
    y_max = 6.0 * delta_star_over_lstar(profile)
    alphas, Res = _small_grid()
    families = _ts_family()

    res_def = temporal_sweep(
        profile, alphas, Res, Ma=2.0, N=31, y_max=y_max,
        length_scale='L_star', operator='ozgen_2d', families=families,
        backend='cpu', cpu_workers=workers, cpu_blas_threads=None,
    )
    res_pin = temporal_sweep(
        profile, alphas, Res, Ma=2.0, N=31, y_max=y_max,
        length_scale='L_star', operator='ozgen_2d', families=families,
        backend='cpu', cpu_workers=workers, cpu_blas_threads=1,
    )

    # meta records
    assert res_def.meta['cpu_blas_threads'] is None
    assert res_pin.meta['cpu_blas_threads'] == 1
    assert res_pin.meta['cpu_blas_threads_effective'] == 1
    assert res_def.meta['cpu_workers'] == min(workers, len(alphas) * len(Res))
    assert res_pin.meta['cpu_workers'] == min(workers, len(alphas) * len(Res))

    # identical NaN / convergence pattern (None-status identical)
    for fd, fp in zip(res_def.families, res_pin.families):
        c_def = fd.c
        c_pin = fp.c
        assert np.array_equal(np.isnan(c_def), np.isnan(c_pin), equal_nan=True)
        # semantic tol on finite values
        # use isclose with rtol on the complex values (compares modulus)
        assert np.allclose(c_def, c_pin, rtol=1e-9, atol=1e-12, equal_nan=True)

    # effective reported only when pinned (or observed); at least for pin path
    assert res_pin.meta['cpu_blas_threads_effective'] == 1


def test_env_var_sets_pinning(monkeypatch):
    """PYMACK_SWEEP_CPU_BLAS_THREADS env is honored (like cpu_workers)."""
    monkeypatch.setenv('PYMACK_SWEEP_CPU_BLAS_THREADS', '1')
    profile = make_flatplate_profile(2.0)
    y_max = 6.0 * delta_star_over_lstar(profile)
    alphas, Res = _small_grid()
    families = _ts_family()

    # Use workers=2 so pool path is taken
    res = temporal_sweep(
        profile, alphas, Res, Ma=2.0, N=31, y_max=y_max,
        length_scale='L_star', operator='ozgen_2d', families=families,
        backend='cpu', cpu_workers=2,
        # no explicit arg: env should win
    )
    assert res.meta['cpu_blas_threads'] == 1
    assert res.meta['cpu_blas_threads_effective'] == 1


def test_cpu_blas_threads_invalid_raises():
    profile = make_flatplate_profile(2.0)
    with pytest.raises(ValueError):
        temporal_sweep(
            profile, [0.1], [900.0], Ma=2.0, N=31, y_max=6.0,
            length_scale='L_star', operator='ozgen_2d',
            families=_ts_family(), backend='cpu', cpu_blas_threads=0,
        )


def test_default_path_does_not_force_single_thread(monkeypatch):
    """Default (no env, no arg) must not inject pinning; reported None or >1 observed.

    We do not assert a specific >1 because it is platform/BLAS dependent,
    only that we did not force 1 when the user asked for default behavior.
    """
    # ensure no accidental env leak in this process
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.delenv(v, raising=False)

    profile = make_flatplate_profile(2.0)
    y_max = 6.0 * delta_star_over_lstar(profile)
    res = temporal_sweep(
        profile, [0.1, 0.15, 0.2], [800.0, 900.0], Ma=2.0, N=31, y_max=y_max,
        length_scale='L_star', operator='ozgen_2d',
        families=_ts_family(), backend='cpu', cpu_workers=2,
    )
    # default requested
    assert res.meta['cpu_blas_threads'] is None
    # on the default path the observed value must now be None (observer not run
    # unless payload cpu_blas_threads is non-None); meta key absent or None
    assert res.meta.get('cpu_blas_threads_effective') is None
