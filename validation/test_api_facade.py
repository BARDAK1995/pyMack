"""Tests for the high-level facade (pymack.api).

The facade must (a) find the documented worked-example Mack mode, (b) apply
the freestream-decay mode selection, and (c) round-trip its own provenance.
"""

import sys

import numpy as np


import pymack as pm


def _profile():
    if not hasattr(_profile, 'cached'):
        _profile.cached = pm.flat_plate(Ma=6.0)
    return _profile.cached


def test_import_is_side_effect_free():
    import contextlib
    import importlib
    import io

    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        importlib.reload(sys.modules['pymack'])
    assert buf_out.getvalue() == ''
    assert buf_err.getvalue() == ''


def test_temporal_mode_finds_worked_example_mack_mode():
    mode = pm.temporal_mode(_profile(), alpha=0.174, Re=5500, N=140)
    assert mode.kind == 'temporal'
    assert abs(mode.eigenvalue - (0.9301 + 0.0200j)) < 5e-3
    assert mode.unstable
    assert mode.params['decay_test_passed']
    assert mode.edge_ratio < 1e-3
    # growth definition: omega_i = alpha * Im(c)
    assert np.isclose(mode.growth_rate, 0.174 * mode.eigenvalue.imag)
    # eigenfunction normalization: max |u| == 1
    assert np.isclose(np.max(np.abs(mode.u)), 1.0)
    # pressure is wall-confined for a Mack mode: |p| peaks near the wall
    assert np.argmax(np.abs(mode.p)) > len(mode.y) // 2


def test_spatial_mode_is_consistent_with_temporal():
    tm = pm.temporal_mode(_profile(), alpha=0.174, Re=5500, N=140)
    sm = pm.spatial_mode(_profile(), omega=tm.omega.real, Re=5500, N=140)
    assert sm.kind == 'spatial'
    assert sm.unstable
    # the spatial twin of the temporal mode: same phase speed to ~1%
    assert abs(sm.phase_speed - tm.phase_speed) < 0.01
    assert abs(sm.alpha.real - 0.174) < 0.005
    assert np.isclose(sm.sigma, -sm.alpha.imag)


def test_ma_defaults_from_profile_and_explicit_override():
    mode = pm.temporal_mode(_profile(), alpha=0.174, Re=5500, N=100)
    assert mode.params['Ma'] == 6.0
    try:
        pm.temporal_mode(lambda y: None, alpha=0.1, Re=100)
    except ValueError as err:
        assert 'Ma' in str(err)
    else:  # pragma: no cover
        raise AssertionError('expected ValueError for profile without Ma')


def test_deprecated_aliases_warn_and_forward():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert pm.make_ozgen_profile is pm.make_flatplate_profile
        assert pm.solve_temporal_ozgen_2d is pm.solve_temporal_2d
    assert sum('deprecated' in str(w.message) for w in caught) >= 2


def test_unguided_spatial_selection_rejects_domain_artifacts():
    """Without a guess, auto domain-stationarity must reject the spurious
    strong-growth root (it drifts ~30% with box height) and return the
    physical Mack mode instead."""
    mode = pm.spatial_mode(_profile(), omega=3.0e-5 * 5600, Re=5600, N=100)
    assert mode.params['stationarity_checked']
    assert mode.params['decay_test_passed']
    assert 0.0 < mode.sigma < 8e-3          # artifact sits near 1.3e-2
    assert abs(mode.alpha.real - 0.182) < 5e-3


def test_guided_calls_skip_stationarity_by_default():
    mode = pm.spatial_mode(_profile(), omega=3.0e-5 * 5600, Re=5600, N=100,
                           alpha_guess=0.180 - 0.004j)
    assert not mode.params['stationarity_checked']
    assert abs(mode.alpha.real - 0.182) < 5e-3
