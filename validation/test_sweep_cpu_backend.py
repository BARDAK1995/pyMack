"""Gate for the ``pymack.sweep`` CPU backend (pyMack-GPU spec slice 06).

Asserts, per the slice contract:

- the sweep facade reproduces the direct per-point loop of the adopted
  Ozgen Fig. 3 driver EXACTLY (bit-equal selected modes, identical
  no-mode/NaN pattern, identical family labels) on a 3x3 sub-grid of the
  production grid, at two domain heights;
- the production-height sub-grid also matches the committed production CSV
  (identical family labels/NaN pattern; values to CSV-formatting
  precision -- see the committed-CSV test docstring for the measured
  1-ulp-of-format environment drift that rules out byte identity);
- the ``PYMACK_SWEEP_BACKEND`` env override is honored for ``backend='auto'``
  and never overrides an explicit ``backend=`` argument;
- ``backend='gpu'`` dispatches to the temporal GPU hook when available, while
  spatial GPU remains unsupported until its slice lands;
- result meta is complete enough to reproduce the sweep;
- CSV/NPZ round-trips embed the meta as headers and recover the exact arrays;
- importing ``pymack.sweep`` never imports ``cupy`` or ``pymack.gpu``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from pymack import make_flatplate_profile  # noqa: E402
from pymack.scales import delta_star_over_lstar  # noqa: E402
from pymack.solver import solve_spatial  # noqa: E402
from pymack.sweep import (  # noqa: E402
    CBand,
    SEED_NO_ADMISSIBLE_MODE,
    SEED_QZ_FULL_SPECTRUM,
    TemporalFamilyResult,
    TemporalSweepResult,
    load_csv,
    spatial_sweep,
    temporal_sweep,
)
from pymack.temporal_solver import solve_temporal_2d  # noqa: E402

# ---------------------------------------------------------------------------
# The adopted driver, loaded from its real location so the test exercises the
# deployed classification (classify_most_unstable) and both compute engines.
# ---------------------------------------------------------------------------

_DRIVER_PATH = REPO_ROOT / 'scripts' / 'make_ozgen_fig3_overlay.py'
_COMMITTED_CSV = REPO_ROOT / 'docs' / 'figures' / 'ozgen_fig3_overlay_ci_grid.csv'


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        'make_ozgen_fig3_overlay', _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def driver():
    return _load_driver()


@pytest.fixture(scope='module')
def profile_m2():
    return make_flatplate_profile(2.0)


# 3x3 sub-grid drawn from the exact production grid of the driver
# (24 log-spaced R_L x 30 linear alpha_L; see QUALITY_SETTINGS/RE_RANGE).
_RE_PROD = np.logspace(np.log10(300.0), np.log10(4500.0), 24)
_ALPHA_PROD = np.linspace(0.02, 0.24, 30)
RE_SUB = _RE_PROD[[0, 12, 23]]
ALPHA_SUB = _ALPHA_PROD[[0, 15, 29]]
N_PROD = 128


def _families(driver):
    """The two CBand families that partition classify_most_unstable's set."""
    return (
        CBand(float('-inf'), driver.TS_CR_MAX,
              ci_abs_max=driver.CI_ABS_MAX, label='TS'),
        CBand(driver.MACK_CR_MIN, driver.MACK_CR_MAX,
              ci_abs_max=driver.CI_ABS_MAX, label='Mack'),
    )


def _combine(result):
    """Cross-family global argmax over Im(c) -- compute_panel_sweep's
    combiner, tie-broken by solver-order mode_index (deployed semantics)."""
    n_a, n_r = result.families[0].c.shape
    ci = np.full((n_a, n_r), np.nan)
    cr = np.full((n_a, n_r), np.nan)
    fam = [['' for _ in range(n_r)] for _ in range(n_a)]
    for i in range(n_a):
        for j in range(n_r):
            best_k, best_ci, best_idx = -1, -np.inf, None
            for k, family in enumerate(result.families):
                v = family.c[i, j].imag
                if not np.isfinite(v):
                    continue
                idx_k = int(family.mode_index[i, j])
                if v > best_ci or (v == best_ci and idx_k < best_idx):
                    best_k, best_ci, best_idx = k, v, idx_k
            if best_k >= 0:
                ci[i, j] = result.families[best_k].c[i, j].imag
                cr[i, j] = result.families[best_k].c[i, j].real
                fam[i][j] = result.families[best_k].band.label
    return ci, cr, fam


def _run_pair(driver, profile, alphas, res_values, N, y_max, cpu_workers):
    """Sweep result + direct-loop spectra/classification at one height."""
    result = temporal_sweep(
        profile, alphas, res_values,
        Ma=2.0, N=N, y_max=y_max, length_scale='L_star',
        operator='ozgen_2d', families=_families(driver),
        backend='cpu', cpu_workers=cpu_workers,
    )
    spectra = {}
    ci = np.full((len(alphas), len(res_values)), np.nan)
    cr = np.full_like(ci, np.nan)
    fam = [['' for _ in res_values] for _ in alphas]
    for i, alpha in enumerate(alphas):
        for j, Re in enumerate(res_values):
            evals, _vecs, _y = solve_temporal_2d(
                profile, float(alpha), float(Re), 2.0,
                N=N, y_max=y_max, length_scale='L_star',
            )
            spectra[(i, j)] = evals
            ci[i, j], cr[i, j], fam[i][j] = driver.classify_most_unstable(
                evals)
    return result, spectra, ci, cr, fam


@pytest.fixture(scope='module')
def pair_production_height(driver, profile_m2):
    """3x3 production sub-grid at the driver's y_max (6 * delta*/L*), N=128,
    sweep through the spawn ProcessPool path."""
    y_max = driver.Y_MAX_FACTOR * delta_star_over_lstar(profile_m2)
    return _run_pair(driver, profile_m2, ALPHA_SUB, RE_SUB, N_PROD, y_max,
                     cpu_workers=3)


def test_sweep_equals_point_loop_production_height(pair_production_height):
    result, spectra, ci_ref, cr_ref, fam_ref = pair_production_height
    ci_sw, cr_sw, fam_sw = _combine(result)
    assert np.array_equal(ci_sw, ci_ref, equal_nan=True)
    assert np.array_equal(cr_sw, cr_ref, equal_nan=True)
    assert fam_sw == fam_ref
    # Per-family: every converged value is bit-equal to an eigenvalue of the
    # direct solve at that point, inside the family window; every point
    # carries an honest provenance code and a certified residual.
    for family in result.families:
        band = family.band
        n_a, n_r = family.c.shape
        for i in range(n_a):
            for j in range(n_r):
                if family.converged[i, j]:
                    c = complex(family.c[i, j])
                    assert any(c == complex(e) for e in spectra[(i, j)])
                    # mode_index is the exact solver-order index of c.
                    mi = int(family.mode_index[i, j])
                    assert mi >= 0
                    assert complex(spectra[(i, j)][mi]) == c
                    assert band.cr_min < c.real < band.cr_max
                    assert abs(c.imag) < band.ci_abs_max
                    assert family.seed_map[i, j] == SEED_QZ_FULL_SPECTRUM
                    assert family.residual[i, j] < 1e-8
                    assert np.isfinite(family.edge_ratio[i, j])
                    assert family.omega_i[i, j] == (
                        float(result.alphas[i]) * c.imag)
                else:
                    assert family.seed_map[i, j] == SEED_NO_ADMISSIBLE_MODE
                    assert np.isnan(family.c[i, j].real)
                    assert np.isnan(family.residual[i, j])
                    assert family.mode_index[i, j] == -1


def test_tie_break_matches_deployed_global_order(driver):
    """On bitwise-equal cross-family Im(c), the combiner reproduces the
    deployed classify_most_unstable global-argmax tie-break (lowest solver
    index wins), NOT a first-family preference.

    Synthetic spectrum with one admissible TS mode and one admissible Mack
    mode carrying IDENTICAL c_i, at controlled solver-order positions; the
    reference is the real deployed classifier, run on the same spectrum."""
    from pymack.sweep import _select_temporal_mode

    ci_tie = 0.01
    ts_c = complex(0.30, ci_tie)     # admissible TS (c_r < TS_CR_MAX)
    mack_c = complex(0.92, ci_tie)   # admissible Mack (MACK band)
    filler = complex(0.60, 0.30)     # inadmissible (excluded band + big c_i)

    def combine_synthetic(evals):
        """Build a 1x1 sweep result from the per-family CPU selection, then
        run the production combiner on it (the code under test)."""
        fams = []
        for band in _families(driver):
            sel = _select_temporal_mode(evals, band)
            converged = sel is not None
            c_val = complex(evals[sel]) if converged else complex(
                np.nan, np.nan)
            fams.append(TemporalFamilyResult(
                band=band,
                c=np.array([[c_val]]),
                omega_i=np.array([[0.0]]),
                converged=np.array([[converged]]),
                residual=np.array([[np.nan]]),
                edge_ratio=np.array([[np.nan]]),
                seed_map=np.array([[0 if converged else -1]]),
                mode_index=np.array([[sel if converged else -1]]),
            ))
        result = TemporalSweepResult(
            alphas=np.array([0.1]), Res=np.array([1000.0]),
            families=tuple(fams), meta={})
        _ci, cr, fam = _combine(result)
        return fam[0][0]

    # Case A: Mack mode earlier in solver order -> deployed picks Mack.
    evals_a = np.array([mack_c, filler, ts_c])
    _ci, _cr, ref_a = driver.classify_most_unstable(evals_a)
    assert ref_a == 'Mack'
    assert combine_synthetic(evals_a) == ref_a

    # Case B: TS mode earlier in solver order -> deployed picks TS. A naive
    # first-family combiner would ALSO say TS here, so this case alone is
    # not decisive; Case A is (first family is TS, deployed picks Mack).
    evals_b = np.array([ts_c, filler, mack_c])
    _ci, _cr, ref_b = driver.classify_most_unstable(evals_b)
    assert ref_b == 'TS'
    assert combine_synthetic(evals_b) == ref_b


def test_sweep_matches_committed_production_csv(pair_production_height):
    """Adoption A/B + executable drift attribution vs the committed CSV
    (cefb440, 2026-06-10).

    The COMMITTED artifact differs from a fresh recompute in the 9th
    significant digit of its %.8e formatting at a few points -- numpy/BLAS
    build drift since 2026-06-10.  This test encodes the attribution as
    code (accepted review item 4): the sweep engine and a FRESH legacy loop
    are (a) bit-identical to each other, and (b) disagree with the committed
    artifact at the SAME set of grid locations.  Identical mismatch sets +
    bit-equal engines prove the drift is common-mode environment drift, NOT
    a facade-introduced regression (a facade defect would move or add
    mismatch locations relative to the untouched legacy loop)."""
    result, _spectra, ci_leg, cr_leg, fam_leg = pair_production_height
    ci_sw, cr_sw, fam_sw = _combine(result)

    # (a) sweep == fresh legacy loop, bitwise, on the full sub-grid.
    assert np.array_equal(ci_sw, ci_leg, equal_nan=True)
    assert np.array_equal(cr_sw, cr_leg, equal_nan=True)
    assert fam_sw == fam_leg

    committed = {}
    with open(_COMMITTED_CSV, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            committed[(row['Ma'], row['Re_L'], row['alpha_L'])] = (
                row['c_i'], row['c_r'], row['family'])

    def mismatch_locations(ci_grid, cr_grid, fam_grid):
        """Keys where this engine's %.8e strings differ from committed."""
        locs = set()
        for i, alpha in enumerate(ALPHA_SUB):
            for j, Re in enumerate(RE_SUB):
                key = ('2', f'{Re:.6f}', f'{alpha:.6f}')
                assert key in committed, f'missing committed row {key}'
                ci_ref, cr_ref, fam_ref = committed[key]
                got = (f'{ci_grid[i, j]:.8e}', f'{cr_grid[i, j]:.8e}',
                       fam_grid[i][j])
                # environment-independent content must always hold:
                assert fam_grid[i][j] == fam_ref, (
                    f'{key}: family {fam_grid[i][j]!r} != {fam_ref!r}')
                cf = float(ci_ref)
                if np.isnan(cf):
                    assert np.isnan(ci_grid[i, j]) and np.isnan(cr_grid[i, j])
                else:
                    assert np.isclose(ci_grid[i, j], cf, rtol=5e-9, atol=1e-12)
                    assert np.isclose(cr_grid[i, j], float(cr_ref),
                                      rtol=5e-9, atol=1e-12)
                if got != (ci_ref, cr_ref, fam_ref):
                    locs.add(key)
        return locs

    sweep_mismatches = mismatch_locations(ci_sw, cr_sw, fam_sw)
    legacy_mismatches = mismatch_locations(ci_leg, cr_leg, fam_leg)
    # (b) the drift is common-mode: both engines miss the artifact at the
    # SAME locations. This refutes a facade-specific regression.
    assert sweep_mismatches == legacy_mismatches


def test_sweep_equals_point_loop_second_height(driver, profile_m2):
    """Same sub-grid at a second domain height (4 * delta*/L*), serial path,
    plus the driver-level integration: compute_panel_sweep == compute_panel."""
    y_max = 4.0 * delta_star_over_lstar(profile_m2)
    result, _spectra, ci_ref, cr_ref, fam_ref = _run_pair(
        driver, profile_m2, ALPHA_SUB, RE_SUB, 64, y_max, cpu_workers=1)
    ci_sw, cr_sw, fam_sw = _combine(result)
    assert np.array_equal(ci_sw, ci_ref, equal_nan=True)
    assert np.array_equal(cr_sw, cr_ref, equal_nan=True)
    assert fam_sw == fam_ref


def test_driver_sweep_engine_equals_point_engine(driver):
    """End-to-end driver A/B: compute_panel_sweep vs compute_panel (N=64)."""
    panel_pt = driver.compute_panel(2.0, RE_SUB, ALPHA_SUB, 64, verbose=False)
    panel_sw = driver.compute_panel_sweep(
        2.0, RE_SUB, ALPHA_SUB, 64, backend='cpu', workers=2, verbose=False)
    assert np.array_equal(panel_pt['ci_grid'], panel_sw['ci_grid'],
                          equal_nan=True)
    assert np.array_equal(panel_pt['cr_grid'], panel_sw['cr_grid'],
                          equal_nan=True)
    assert panel_pt['family_grid'] == panel_sw['family_grid']
    for key in ('Ma', 'N', 'delta_star_over_lstar', 'y_max_lstar'):
        assert panel_pt[key] == panel_sw[key]


def test_env_backend_override(monkeypatch, profile_m2):
    import pymack.sweep as sweep_mod

    families = (CBand(float('-inf'), 0.45, ci_abs_max=0.05, label='TS'),)
    kwargs = dict(Ma=2.0, N=31, y_max=12.0, length_scale='L_star',
                  operator='ozgen_2d', families=families, cpu_workers=1)
    # env override steers backend='auto'
    monkeypatch.setenv('PYMACK_SWEEP_BACKEND', 'cpu')
    result = temporal_sweep(profile_m2, [0.08], [900.0], backend='auto',
                            **kwargs)
    assert result.meta['backend'] == 'cpu'
    assert result.meta['backend_requested'] == 'auto'
    assert result.meta['env_backend_override'] == 'cpu'
    # env='gpu' + backend='auto' steers to the GPU hook when available.
    monkeypatch.setenv('PYMACK_SWEEP_BACKEND', 'gpu')
    if sweep_mod._gpu_backend_available():
        result = temporal_sweep(profile_m2, [0.08], [900.0],
                                backend='auto', **kwargs)
        assert result.meta['backend'] == 'gpu'
        assert result.meta['backend_requested'] == 'auto'
        assert result.meta['env_backend_override'] == 'gpu'
    else:
        with pytest.raises(NotImplementedError, match='no usable GPU'):
            temporal_sweep(profile_m2, [0.08], [900.0],
                           backend='auto', **kwargs)
    # an explicit backend argument always wins over the environment
    result = temporal_sweep(profile_m2, [0.08], [900.0], backend='cpu',
                            **kwargs)
    assert result.meta['backend'] == 'cpu'
    assert result.meta['backend_requested'] == 'cpu'


def test_gpu_backend_dispatches_cleanly(monkeypatch, profile_m2):
    """backend='gpu' either dispatches to the temporal hook or raises a clear
    availability error; spatial GPU remains unsupported."""
    import pymack.sweep as sweep_mod

    monkeypatch.delenv('PYMACK_SWEEP_BACKEND', raising=False)
    kwargs = dict(Ma=2.0, N=31, y_max=12.0, length_scale='L_star',
                  operator='ozgen_2d',
                  families=(CBand(float('-inf'), 0.45,
                                  ci_abs_max=0.05, label='TS'),),
                  cpu_workers=1)
    if sweep_mod._gpu_backend_available():
        result = temporal_sweep(profile_m2, [0.08], [900.0],
                                backend='gpu', **kwargs)
        assert result.meta['backend'] == 'gpu'
        assert result.meta['gpu_engine_status']
    else:
        with pytest.raises(NotImplementedError, match='no usable GPU'):
            temporal_sweep(profile_m2, [0.08], [900.0],
                           backend='gpu', **kwargs)
    with pytest.raises(NotImplementedError, match='spatial|no usable GPU'):
        spatial_sweep(profile_m2, [0.1], [1000.0], Ma=4.5, backend='gpu')
    with pytest.raises(ValueError, match='backend'):
        temporal_sweep(profile_m2, [0.08], [900.0], backend='tpu', **kwargs)


def test_auto_resolves_to_cpu_when_no_engine(monkeypatch, profile_m2):
    """backend='auto' resolves to CPU when the engine probe is false."""
    import pymack.sweep as sweep_mod

    monkeypatch.delenv('PYMACK_SWEEP_BACKEND', raising=False)
    monkeypatch.setattr(sweep_mod, '_gpu_backend_available', lambda: False)
    result = temporal_sweep(
        profile_m2, [0.08], [900.0], Ma=2.0, N=31, y_max=12.0,
        length_scale='L_star', operator='ozgen_2d',
        families=(CBand(float('-inf'), 0.45, ci_abs_max=0.05, label='TS'),),
        backend='auto', cpu_workers=1)
    assert result.meta['backend'] == 'cpu'
    assert result.meta['backend_requested'] == 'auto'
    assert result.meta['env_backend_override'] is None


def test_shared_solve_failure_is_contained_and_diagnosable(profile_m2):
    """A per-point solver exception degrades exactly that point to seed -2,
    records the exception repr in meta['errors'] (items 2 + 3), and the
    sweep still returns."""
    families = (CBand(float('-inf'), 0.45, ci_abs_max=0.05, label='TS'),)
    result = temporal_sweep(
        profile_m2, [0.08], [900.0], Ma=2.0, N=31, y_max=12.0,
        length_scale='L_star', operator='ozgen_2d', wall_bc='bogus',
        families=families, backend='cpu', cpu_workers=1)
    fam = result.families[0]
    from pymack.sweep import SEED_SOLVER_FAILED
    assert fam.seed_map[0, 0] == SEED_SOLVER_FAILED
    assert not fam.converged[0, 0]
    assert np.isnan(fam.c[0, 0].real)
    assert fam.mode_index[0, 0] == -1
    assert result.meta['n_failed_points'] == 1
    errors = result.meta['errors']
    assert len(errors) == 1
    err = errors[0]
    assert err['i'] == 0 and err['j'] == 0 and err['family'] == 0
    assert 'wall_bc' in err['error']  # the deployed ValueError text


def test_post_solve_exception_is_contained(monkeypatch, profile_m2):
    """A failure AFTER the solve (item 3: whole per-point record path is
    wrapped) degrades one point to -2 without killing the sweep; a
    neighboring point still converges."""
    import pymack.sweep as sweep_mod

    real_edge_ratio = sweep_mod._edge_ratio
    calls = {'n': 0}

    def flaky_edge_ratio(phi, n, n_edge=4):
        calls['n'] += 1
        if calls['n'] == 1:          # blow up on the first selected mode only
            raise RuntimeError('synthetic post-solve failure')
        return real_edge_ratio(phi, n, n_edge=n_edge)

    monkeypatch.setattr(sweep_mod, '_edge_ratio', flaky_edge_ratio)
    families = (CBand(float('-inf'), 0.45, ci_abs_max=0.05, label='TS'),)
    result = temporal_sweep(
        profile_m2, [0.08, 0.10], [900.0], Ma=2.0, N=48, y_max=12.0,
        length_scale='L_star', operator='ozgen_2d',
        families=families, backend='cpu', cpu_workers=1)
    fam = result.families[0]
    from pymack.sweep import SEED_SOLVER_FAILED
    seeds = fam.seed_map.ravel().tolist()
    assert SEED_SOLVER_FAILED in seeds, 'post-solve failure was not contained'
    assert result.meta['n_failed_points'] >= 1
    assert any('synthetic post-solve failure' in e['error']
               for e in result.meta['errors'])
    # containment, not collapse: at least one point still converged normally.
    assert fam.converged.any()


_REQUIRED_META_KEYS = (
    'api', 'kind', 'schema_version', 'backend', 'backend_requested',
    'env_backend_override', 'operator', 'operator_source', 'precision',
    'precision_requested', 'tile_size', 'tile_size_requested', 'seeds',
    'cpu_workers', 'families', 'solver_kwargs', 'grid',
    'return_eigenvectors', 'seed_codes', 'n_failed_points', 'errors',
    'residual_definition', 'edge_ratio_definition', 'versions', 'platform',
    'timestamp_utc', 'wall_time_s',
)


def test_meta_completeness(pair_production_height):
    result, _s, _ci, _cr, _f = pair_production_height
    meta = result.meta
    for key in _REQUIRED_META_KEYS:
        assert key in meta, f'meta missing {key!r}'
    assert meta['kind'] == 'temporal'
    assert meta['backend'] == 'cpu'
    assert meta['operator'] == 'ozgen_2d'
    assert meta['operator_source'] == 'per_point_cpu_qz'
    assert meta['precision'] == 'fp64'          # effective CPU precision
    assert meta['precision_requested'] == 'mixed'
    assert meta['cpu_workers'] == 3
    assert meta['wall_time_s'] > 0.0
    assert [f['label'] for f in meta['families']] == ['TS', 'Mack']
    for lib in ('pymack', 'numpy', 'scipy', 'python'):
        assert lib in meta['versions']
    for kw in ('Ma', 'N', 'y_max', 'L', 'wall_bc', 'length_scale', 'Pr',
               'gamma', 'lambda_mu_ratio'):
        assert kw in meta['solver_kwargs']
    assert meta['grid']['alpha']['n'] == 3
    assert meta['grid']['Re']['n'] == 3
    # the seed_codes legend documents the per-backend contract
    assert '0' in meta['seed_codes'] and '-1' in meta['seed_codes']
    assert '-2' in meta['seed_codes'] and 'reserved_positive' in meta['seed_codes']
    # a clean sweep has no failed points but still publishes the error list
    assert meta['n_failed_points'] == 0
    assert meta['errors'] == []
    # meta must be JSON-serializable (it is embedded in CSV/NPZ headers)
    json.dumps(meta)


def test_csv_npz_roundtrip(tmp_path, pair_production_height):
    result, _s, _ci, _cr, _f = pair_production_height

    npz_path = tmp_path / 'sweep.npz'
    result.to_npz(npz_path)
    loaded = TemporalSweepResult.from_npz(npz_path)
    assert loaded.meta == result.meta
    assert np.array_equal(loaded.alphas, result.alphas)
    assert np.array_equal(loaded.Res, result.Res)
    assert len(loaded.families) == len(result.families)
    for fam_in, fam_out in zip(result.families, loaded.families):
        assert fam_out.band == fam_in.band
        assert np.array_equal(fam_out.c, fam_in.c, equal_nan=True)
        assert np.array_equal(fam_out.omega_i, fam_in.omega_i, equal_nan=True)
        assert np.array_equal(fam_out.converged, fam_in.converged)
        assert np.array_equal(fam_out.residual, fam_in.residual,
                              equal_nan=True)
        assert np.array_equal(fam_out.edge_ratio, fam_in.edge_ratio,
                              equal_nan=True)
        assert np.array_equal(fam_out.seed_map, fam_in.seed_map)
        assert np.array_equal(fam_out.mode_index, fam_in.mode_index)

    csv_path = tmp_path / 'sweep.csv'
    result.to_csv(csv_path)
    meta, rows = load_csv(csv_path)
    assert meta == result.meta
    n_points = len(result.alphas) * len(result.Res)
    assert len(rows) == len(result.families) * n_points
    for row in rows:
        fam = result.families[row['family']]
        i = int(np.flatnonzero(result.alphas == row['alpha'])[0])
        j = int(np.flatnonzero(result.Res == row['Re'])[0])
        assert row['family_label'] == fam.band.label
        for got, ref in (
            (row['c_re'], fam.c[i, j].real),
            (row['c_im'], fam.c[i, j].imag),
            (row['omega_i'], fam.omega_i[i, j]),
            (row['residual'], fam.residual[i, j]),
            (row['edge_ratio'], fam.edge_ratio[i, j]),
        ):
            assert (got == ref) or (np.isnan(got) and np.isnan(ref))
        assert row['converged'] == bool(fam.converged[i, j])
        assert row['seed'] == int(fam.seed_map[i, j])
        assert row['mode_index'] == int(fam.mode_index[i, j])


def test_cpu_worker_resolution_respects_windows_cap():
    """cpu_workers=None on a >61-core Windows box must not crash the pool
    (ProcessPoolExecutor's WaitForMultipleObjects limit is 61 workers)."""
    from pymack.sweep import _resolve_cpu_workers

    assert _resolve_cpu_workers(1, 9) == 1
    assert _resolve_cpu_workers(8, 3) == 3          # capped by task count
    resolved = _resolve_cpu_workers(None, 720)      # machine default
    assert 1 <= resolved <= (61 if sys.platform == 'win32' else 720)
    assert _resolve_cpu_workers(500, 720) <= (
        61 if sys.platform == 'win32' else 500)
    with pytest.raises(ValueError):
        _resolve_cpu_workers(0, 9)


def test_import_is_numpy_safe():
    """`import pymack, pymack.sweep` must not pull in cupy or pymack.gpu."""
    code = (
        'import sys; import pymack, pymack.sweep; '
        "assert 'cupy' not in sys.modules, 'cupy imported'; "
        "assert not any(m.startswith('pymack.gpu') for m in sys.modules), "
        "'pymack.gpu imported'; "
        'print(\'ok\')'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith('ok')


def test_spatial_sweep_matches_direct_loop(profile_m2):
    """spatial_sweep (serial CPU) reproduces the deployed per-point pattern:
    solve_spatial at the family target + band filter + argmin Im(alpha)."""
    omega, res_values = 0.08, [800.0, 1500.0]
    band = CBand(0.05, 0.95, label='spatial-smoke', target=0.4)
    ai_cap = 0.5
    result = spatial_sweep(
        profile_m2, [omega], res_values,
        Ma=2.0, N=48, y_max=12.0, length_scale='L_star',
        families=(band,), n_modes=25, alpha_i_abs_max=ai_cap,
        backend='cpu', cpu_workers=1,
    )
    family = result.families[0]
    n_converged = 0
    for j, Re in enumerate(res_values):
        alphas, _modes, _y = solve_spatial(
            profile_m2, float(omega), float(Re), 2.0, 0.72, 1.4,
            N=48, y_max=12.0, wall_bc='isothermal',
            target_alpha=float(omega) / band.target + 0j, n_modes=25,
            length_scale='L_star',
        )
        c = omega / alphas.real
        mask = ((c > band.cr_min) & (c < band.cr_max)
                & (np.abs(alphas.imag) < ai_cap) & (alphas.real > 0.0))
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            assert not family.converged[0, j]
            continue
        expected = complex(alphas[idx[np.argmin(alphas.imag[idx])]])
        assert family.converged[0, j]
        assert complex(family.alpha[0, j]) == expected
        assert family.sigma[0, j] == -expected.imag
        assert family.residual[0, j] < 1e-6
        n_converged += 1
    assert n_converged > 0, 'smoke test never selected a mode (vacuous)'
