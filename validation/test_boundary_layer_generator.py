"""
Validation: standalone compressible boundary-layer generator.

Tests (fast suite, CI budget < 30 s):
 1. Pure-wrapper equality with CompressibleBlasiusProfile + measured anchors.
 2. Mack Table 11.1 displacement-thickness regression (subset, 1%).
 3. Incompressible Blasius limit (delta*, theta, H).
 4. Exact Pr=1/omega=1 thermodynamics (Taw, Crocco-Busemann temperature).
 5. Solved recovery factor versus the r = sqrt(Pr) formula.
 6. Continuation machinery (forced ramp equals direct solve; mode validation).
 7. Stability-solver round trip through as_stability_profile().
 8. Cross-backend consistency against pymack_dense.solve_base_flow.
 9. Dimensionalization arithmetic (exact synthetic edge state).
10. CSV/JSON round trip with parseable metadata header.
11. Input validation and wall-specification equivalence.
12. CLI smoke test (subprocess, CSV + JSON outputs).

Slow tests (set PYMACK_RUN_SLOW=1 to enable):
13. Cold-wall 'mack' continuation rescue (Ma=6, Tw/Te=0.2, ~15 s).
14. Full 105-case matrix sweep (Ma x wall x transport model, minutes).
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymack.baseflow import CompressibleBlasiusProfile
from pymack.boundary_layer import generate_boundary_layer
from pymack.mack_conditions import mack_table_11_1_edge_temperature
from pymack.pymack_dense import (
    DenseBaseFlowConfig,
    DenseGasModel,
    solve_base_flow,
)
from pymack.scales import DimensionalEdgeState
from pymack.solver import solve_spatial


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SLOW = os.environ.get('PYMACK_RUN_SLOW', '') not in {'', '0'}


def _skip_slow():
    """Mark a slow test as skipped (visible under pytest, printed under main)."""
    msg = 'set PYMACK_RUN_SLOW=1 to run'
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print(f'  SKIPPED ({msg})\n')

# Fast-suite resolution: anchors below were verified at this n_points.
N_FAST = 1500

_CACHE = {}


def gen(**kwargs):
    """Cache generator results so profiles are shared across tests."""
    key = tuple(sorted(kwargs.items()))
    if key not in _CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # narrow-table warning at Ma=10
            _CACHE[key] = generate_boundary_layer(**kwargs)
    return _CACHE[key]


def test_wrapper_equality():
    """The generator must be a pure wrapper around the engine."""
    print('Test 1: wrapper equality with CompressibleBlasiusProfile')

    r = gen(Ma=4.5, wall_bc='adiabatic', viscosity_model='sutherland',
            T_edge_K=300.0, n_points=N_FAST)
    p = CompressibleBlasiusProfile(
        Ma=4.5, T_wall=300.0, T_edge=300.0, wall_bc='adiabatic',
        viscosity_model='sutherland', n_points=N_FAST)

    assert r.delta_star_over_Lstar == p._delta_star, 'not a pure wrap (d*)'
    assert r.theta_over_Lstar == p._theta, 'not a pure wrap (theta)'
    assert isinstance(r.as_stability_profile(), CompressibleBlasiusProfile)
    assert r.as_stability_profile()._delta_star == r.delta_star_over_Lstar

    # Measured anchors (probe, n_points=1500). theta uses the corrected
    # momentum-thickness integrand theta/L* = sqrt(2) int U(1-U) d_eta
    # (the pre-2026-06 U(T-U) integrand was a bug); the incompressible limit
    # of this value is checked against Blasius 0.6641 in test below.
    assert abs(r.delta_star_over_Lstar - 8.4636) < 0.01
    assert abs(r.theta_over_Lstar - 0.5468) < 0.005
    assert abs(r.Tw_over_Te - 4.3984) < 0.001
    assert r.used_continuation is False
    assert r.n_continuation_steps == 0
    print(f'  d*/L* = {r.delta_star_over_Lstar:.4f}, '
          f'theta/L* = {r.theta_over_Lstar:.4f}, Tw/Te = {r.Tw_over_Te:.4f}')
    print('  PASSED\n')


def test_mack_table_11_1_subset():
    """Mack Table 11.1 d* values must hold through the generator (1%)."""
    print('Test 2: Mack Table 11.1 displacement thickness (subset)')

    refs = {2.2: 3.72, 4.5: 10.34, 10.0: 36.88}
    for Ma, ref in refs.items():
        r = gen(Ma=Ma, wall_bc='adiabatic', viscosity_model='mack',
                T_edge_K=mack_table_11_1_edge_temperature(Ma),
                n_points=N_FAST)
        rel_err = abs(r.delta_star_over_Lstar - ref) / ref
        print(f'  M={Ma:4.1f}: d*={r.delta_star_over_Lstar:8.3f}, '
              f'ref={ref:8.3f}, rel_err={100.0 * rel_err:6.3f}%')
        assert rel_err < 0.01, f'Table 11.1 mismatch at M={Ma}'
    print('  PASSED\n')


def test_incompressible_blasius_limit():
    """Ma -> 0 with Tw = Te must recover the incompressible Blasius layer."""
    print('Test 3: incompressible Blasius limit')

    r = gen(Ma=1e-3, wall_bc='isothermal', Tw_over_Te=1.0,
            viscosity_model='power_law', n_points=N_FAST)
    print(f'  d*/L* = {r.delta_star_over_Lstar:.6f} (Blasius 1.720788)')
    print(f'  theta/L* = {r.theta_over_Lstar:.6f} (Blasius 0.664115)')
    print(f'  H = {r.shape_factor_H:.5f} (Blasius 2.591)')
    assert abs(r.delta_star_over_Lstar - 1.720788) < 5e-4
    assert abs(r.theta_over_Lstar - 0.664115) < 5e-4
    assert abs(r.shape_factor_H - 2.591) < 2e-3
    assert r.recovery_factor_solved is None  # isothermal wall
    print('  PASSED\n')


def test_exact_pr1_thermodynamics():
    """Pr=1, omega=1: Taw/Te = 1 + 0.2 Ma^2 and Crocco-Busemann hold exactly."""
    print('Test 4: exact Pr=1/omega=1 thermodynamics at Ma=4')

    r = gen(Ma=4.0, wall_bc='adiabatic', Pr=1.0, omega=1.0,
            viscosity_model='power_law', T_edge_K=300.0, n_points=N_FAST)
    taw_exact = 1.0 + 0.2 * 16.0
    print(f'  solved Tw/Te = {r.Tw_over_Te:.7f} (exact {taw_exact})')
    assert abs(r.Tw_over_Te - taw_exact) < 1e-4
    assert abs(r.Taw_over_Te_formula - taw_exact) < 1e-12  # sqrt(1) = 1

    T_cb = 1.0 + 0.2 * 16.0 * (1.0 - r.U**2)
    max_err = float(np.max(np.abs(r.T - T_cb)))
    print(f'  max |T - Crocco-Busemann| = {max_err:.2e}')
    assert max_err < 1e-5
    print('  PASSED\n')


def test_recovery_factor_vs_formula():
    """Solved recovery factor must track (not equal) r = sqrt(Pr)."""
    print('Test 5: solved recovery factor versus sqrt(Pr) formula')

    r_formula = math.sqrt(0.72)
    cases = [
        dict(Ma=2.0, viscosity_model='power_law'),
        dict(Ma=10.0, viscosity_model='power_law'),
        dict(Ma=10.0, viscosity_model='sutherland'),
    ]
    for case in cases:
        r = gen(wall_bc='adiabatic', T_edge_K=300.0, n_points=N_FAST, **case)
        solved = r.recovery_factor_solved
        print(f"  Ma={case['Ma']:4.1f} {case['viscosity_model']:>10s}: "
              f'r_solved={solved:.5f} (formula {r_formula:.5f})')
        assert solved is not None
        assert abs(solved - r_formula) < 0.02
        # The formula and solved values must be reported as distinct numbers.
        assert abs(r.Taw_over_Te_formula - r.Tw_over_Te) > 1e-6

    r = gen(Ma=4.5, wall_bc='adiabatic', viscosity_model='mack',
            T_edge_K=300.0, n_points=N_FAST)
    print(f'  Ma= 4.5       mack: r_solved={r.recovery_factor_solved:.5f} '
          '(variable Pr)')
    assert 0.70 < r.recovery_factor_solved < 0.85
    print('  PASSED\n')


def test_continuation_machinery():
    """Forced continuation must reproduce the direct solve; modes validated."""
    print('Test 6: continuation machinery')

    kwargs = dict(Ma=4.5, wall_bc='isothermal', Tw_over_Te=0.5,
                  viscosity_model='sutherland', T_edge_K=300.0, n_points=1000)
    direct = gen(**kwargs)
    assert direct.used_continuation is False
    assert direct.n_continuation_steps == 0

    messages = []
    forced = generate_boundary_layer(
        continuation='force', progress=messages.append, **kwargs)
    assert forced.used_continuation is True
    assert forced.n_continuation_steps >= 4
    assert len(messages) >= forced.n_continuation_steps

    rel = abs(forced.delta_star_over_Lstar - direct.delta_star_over_Lstar) \
        / direct.delta_star_over_Lstar
    print(f'  direct d* = {direct.delta_star_over_Lstar:.6f}, '
          f'forced d* = {forced.delta_star_over_Lstar:.6f} '
          f'(rel diff {rel:.2e}, {forced.n_continuation_steps} ramp steps)')
    assert rel < 1e-4, 'continuation drifted from the direct solve'

    try:
        generate_boundary_layer(2.0, continuation='banana')
        raise AssertionError('bad continuation mode accepted')
    except ValueError:
        pass
    print('  PASSED\n')


def test_stability_solver_round_trip():
    """as_stability_profile() must plug straight into solve_spatial."""
    print('Test 7: stability-solver round trip (Ma=4.5 adiabatic mack)')

    bf = gen(Ma=4.5, wall_bc='adiabatic', viscosity_model='mack',
             T_edge_K=300.0, n_points=N_FAST).as_stability_profile()
    alphas, _, _ = solve_spatial(
        bf, omega=0.1, Re=2000, Ma=4.5, Pr=0.72, gamma=1.4, N=80)
    assert len(alphas) > 0
    alpha = min(alphas, key=lambda a: abs(a - 0.1))
    print(f'  {len(alphas)} modes; nearest to 0.1: '
          f'alpha = {alpha.real:.5f} {alpha.imag:+.2e}j')
    assert 0.08 < alpha.real < 0.12
    assert abs(alpha.imag) < 0.01
    print('  PASSED\n')


def test_cross_backend_consistency():
    """Generator must agree with the independent dense Lees-Dorodnitsyn solver."""
    print('Test 8: cross-backend consistency (Ma=6, Tw/Te=5.88, sutherland)')

    r = gen(Ma=6.0, wall_bc='isothermal', Tw_over_Te=5.88,
            viscosity_model='sutherland', T_edge_K=300.0, n_points=N_FAST)
    assert abs(r.delta_star_over_Lstar - 11.451) / 11.451 < 1e-3
    assert r.used_continuation is False

    dense = solve_base_flow(
        DenseBaseFlowConfig(mach_edge=6.0, Tw_Te=5.88),
        DenseGasModel(T_edge_K=300.0))
    worst = 0.0
    for y_L in [1.0, 3.0, 6.0, 10.0]:
        U_gen = float(r.sample(y_L, scale='L_star')['U'])
        U_dense = float(np.interp(y_L, dense.y, dense.U))
        worst = max(worst, abs(U_gen - U_dense))
        print(f'  y/L*={y_L:5.1f}: U_gen={U_gen:.6f}, U_dense={U_dense:.6f}, '
              f'diff={abs(U_gen - U_dense):.2e}')
    assert worst < 2e-4
    print('  PASSED\n')


def test_dimensionalization_arithmetic():
    """Exact synthetic edge state: pure multiplications, no hidden unit math."""
    print('Test 9: dimensionalization arithmetic')

    r = gen(Ma=4.5, wall_bc='adiabatic', viscosity_model='sutherland',
            T_edge_K=300.0, n_points=N_FAST)
    edge = DimensionalEdgeState(U_e=1000.0, nu_e=1e-5)

    d = r.dimensionalize(edge, R_L=2000.0)
    assert d.L_star_m == 2.0e-5
    assert d.x_m == 0.04
    assert d.Re_x == 4.0e6
    assert d.delta_star_m == r.delta_star_over_Lstar * 2.0e-5
    assert d.theta_m == r.theta_over_Lstar * 2.0e-5
    assert d.T_K[0] == r.Tw_over_Te * r.T_edge_K
    assert d.U_m_per_s[-1] == 1000.0
    assert d.rho_kg_m3 is None and d.mu_Pa_s is None
    print(f'  L* = {d.L_star_m:.3e} m, x = {d.x_m} m, '
          f'delta* = {d.delta_star_m:.4e} m')

    d2 = r.dimensionalize(edge, R_L=2000.0, rho_e_kg_m3=0.1)
    assert np.allclose(d2.rho_kg_m3, r.rho * 0.1, rtol=0, atol=0)
    assert np.allclose(d2.mu_Pa_s, r.mu * (0.1 * 1e-5), rtol=0, atol=0)

    # x_m route must agree with the R_L route.
    d3 = r.dimensionalize(edge, x_m=0.04)
    assert abs(d3.R_L - 2000.0) / 2000.0 < 1e-12
    assert abs(d3.L_star_m - 2.0e-5) / 2.0e-5 < 1e-12

    for bad_kwargs in [dict(x_m=0.04, R_L=2000.0), dict()]:
        try:
            r.dimensionalize(edge, **bad_kwargs)
            raise AssertionError(f'accepted bad station spec {bad_kwargs}')
        except ValueError:
            pass
    print('  PASSED\n')


def test_csv_json_round_trip():
    """CSV table and JSON metadata must survive a write/read cycle."""
    print('Test 10: CSV/JSON round trip')

    r = gen(Ma=4.5, wall_bc='adiabatic', viscosity_model='sutherland',
            T_edge_K=300.0, n_points=N_FAST)
    with tempfile.TemporaryDirectory() as tmp:
        path = r.to_csv(Path(tmp) / 'profile.csv')
        assert path.exists()

        data = np.loadtxt(path)
        assert data.shape == (N_FAST, 13)
        assert np.all(np.diff(data[:, 1]) > 0), 'y/L* not strictly increasing'
        assert abs(r.U[0]) < 1e-12
        assert abs(r.U[-1] - 1.0) < 1e-6
        assert abs(r.T[-1] - 1.0) < 1e-6
        in_memory = np.column_stack([
            r.eta, r.y_over_Lstar, r.y_over_delta_star,
            r.U, r.dU_dyL, r.d2U_dyL2, r.T, r.dT_dyL, r.d2T_dyL2,
            r.rho, r.mu, r.kappa, r.Pr_local,
        ])
        assert np.allclose(data, in_memory, rtol=1e-12, atol=0.0)

        meta_lines = [line for line in path.read_text().splitlines()
                      if line.startswith('# metadata: ')]
        assert len(meta_lines) == 1
        meta = json.loads(meta_lines[0][len('# metadata: '):])
        for key in ['Ma', 'wall_bc', 'Tw_over_Te', 'delta_star_over_Lstar']:
            assert key in meta, f'metadata missing {key}'
        assert meta['Ma'] == 4.5
        assert meta['delta_star_over_Lstar'] == r.delta_star_over_Lstar

        # to_dict must be JSON-serializable as-is.
        record = json.loads(json.dumps(r.to_dict()))
        assert record['wall_bc'] == 'adiabatic'

        # Dimensional CSV with density columns.
        edge = DimensionalEdgeState(U_e=1000.0, nu_e=1e-5)
        d = r.dimensionalize(edge, R_L=2000.0, rho_e_kg_m3=0.1)
        si_path = d.to_csv(Path(tmp) / 'profile_si.csv')
        si_data = np.loadtxt(si_path)
        assert si_data.shape == (N_FAST, 5)
        assert np.allclose(si_data[:, 0], d.y_m, rtol=1e-12)
    print(f'  {N_FAST} rows x 13 columns round-tripped, metadata parsed')
    print('  PASSED\n')


def test_input_validation():
    """Conflicting or missing wall specifications must raise ValueError."""
    print('Test 11: input validation and wall-spec equivalence')

    bad_calls = [
        dict(wall_bc='isothermal'),                                # none
        dict(wall_bc='isothermal', Tw_over_Te=0.5, T_wall_K=150.0),  # two
        dict(wall_bc='adiabatic', Tw_over_Te=0.5),                 # extra
        dict(wall_bc='radiative'),                                 # bad bc
        dict(viscosity_model='carbon'),                            # bad model
        dict(viscosity_model='mack', gas='nitrogen'),              # air only
        dict(gas='argon'),                                         # bad preset
        dict(T_edge_K=-10.0),                                      # bad T_e
        dict(wall_bc='isothermal', Tw_over_Te=-0.5),               # bad T_w
    ]
    for kwargs in bad_calls:
        try:
            generate_boundary_layer(2.0, **kwargs)
            raise AssertionError(f'accepted invalid input {kwargs}')
        except ValueError:
            pass
    print(f'  {len(bad_calls)} invalid inputs rejected')

    # Tw_over_Taw and the equivalent absolute T_wall_K must agree.
    # (Expression order mirrors the generator so the routes match bitwise.)
    taw_K = 300.0 * (1.0 + 0.5 * (1.4 - 1.0) * math.sqrt(0.72) * 2.0**2)
    r1 = gen(Ma=2.0, wall_bc='isothermal', Tw_over_Taw=0.7,
             viscosity_model='sutherland', T_edge_K=300.0, n_points=800)
    r2 = gen(Ma=2.0, wall_bc='isothermal', T_wall_K=0.7 * taw_K,
             viscosity_model='sutherland', T_edge_K=300.0, n_points=800)
    rel = abs(r1.delta_star_over_Lstar - r2.delta_star_over_Lstar) \
        / r1.delta_star_over_Lstar
    print(f'  Tw_over_Taw vs T_wall_K route: rel diff {rel:.2e}')
    assert rel < 1e-10
    print('  PASSED\n')


def test_cli_smoke():
    """The CLI must produce CSV + JSON and report the measured d* anchor."""
    print('Test 12: CLI smoke test (Ma=2, Tw/Te=0.5, sutherland)')

    script = REPO_ROOT / 'scripts' / 'generate_boundary_layer.py'
    env = {**os.environ, 'PYMACK_NO_BANNER': '1'}
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / 'bl.csv'
        json_path = Path(tmp) / 'bl.json'
        proc = subprocess.run(
            [sys.executable, str(script), '--ma', '2',
             '--wall', 'isothermal', '--tw-over-te', '0.5',
             '--csv', str(csv_path), '--json', str(json_path),
             '--n-points', '1000'],
            capture_output=True, text=True, env=env, timeout=120,
        )
        assert proc.returncode == 0, f'CLI failed:\n{proc.stderr}'
        assert csv_path.exists() and json_path.exists()
        record = json.loads(json_path.read_text())
        d_star = record['delta_star_over_Lstar']
        print(f'  CLI d*/L* = {d_star:.4f} (anchor 1.250)')
        assert abs(d_star - 1.250) / 1.250 < 0.01
        assert 'delta*/L*' in proc.stdout
    print('  PASSED\n')


def test_slow_cold_wall_mack_rescue():
    """Cold 'mack' wall: direct solve fails, continuation must rescue it."""
    print('Test 13 (slow): cold-wall mack continuation rescue')
    if not RUN_SLOW:
        _skip_slow()
        return

    # continuation='never' must surface the direct-solve failure.
    try:
        generate_boundary_layer(
            6.0, wall_bc='isothermal', Tw_over_Te=0.2,
            viscosity_model='mack', T_edge_K=300.0,
            continuation='never', n_points=N_FAST)
        raise AssertionError('direct cold-wall mack solve unexpectedly passed')
    except RuntimeError:
        print('  continuation=never raised RuntimeError as expected')

    r = generate_boundary_layer(
        6.0, wall_bc='isothermal', Tw_over_Te=0.2,
        viscosity_model='mack', T_edge_K=300.0, n_points=N_FAST)
    print(f'  rescued: d* = {r.delta_star_over_Lstar:.4f} (anchor 4.019), '
          f'theta = {r.theta_over_Lstar:.4f} (anchor 0.6196)')
    assert r.used_continuation is True
    assert abs(r.delta_star_over_Lstar - 4.019) / 4.019 < 0.01
    # theta anchor corrected to the U(1-U) momentum-thickness integrand (commit
    # b6ed898); the old 2.3147 was the pre-fix U(T-U) value, stale (this slow
    # test is gated off by default, so it was missed when the fix landed).
    assert abs(r.theta_over_Lstar - 0.6196) / 0.6196 < 0.01

    # Counter-case: sutherland handles the same cold wall directly at Ma=10.
    r2 = gen(Ma=10.0, wall_bc='isothermal', Tw_over_Te=0.2,
             viscosity_model='sutherland', T_edge_K=300.0, n_points=N_FAST)
    print(f'  sutherland Ma=10 direct: d* = {r2.delta_star_over_Lstar:.4f} '
          '(anchor 9.527)')
    assert r2.used_continuation is False
    assert abs(r2.delta_star_over_Lstar - 9.527) / 9.527 < 0.005
    print('  PASSED\n')


def test_slow_full_matrix_sweep():
    """The full probe matrix must complete without failures (auto mode)."""
    print('Test 14 (slow): full 105-case matrix sweep')
    if not RUN_SLOW:
        _skip_slow()
        return

    failures = []
    n_total = 0
    for Ma in [0.5, 2.0, 4.5, 6.0, 10.0]:
        for model in ['power_law', 'sutherland', 'mack']:
            walls = [dict(wall_bc='adiabatic')] + [
                dict(wall_bc='isothermal', Tw_over_Te=ratio)
                for ratio in [0.2, 0.5, 1.0, 2.0, 4.0, 6.0]
            ]
            for wall in walls:
                n_total += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        generate_boundary_layer(
                            Ma, viscosity_model=model, T_edge_K=300.0,
                            n_points=N_FAST, **wall)
                except Exception as exc:  # noqa: BLE001 - sweep report
                    failures.append((Ma, model, wall, str(exc)))
    print(f'  {n_total - len(failures)}/{n_total} cases converged')
    assert not failures, f'matrix sweep failures: {failures}'
    print('  PASSED\n')


if __name__ == '__main__':
    print('=' * 60)
    print('BOUNDARY-LAYER GENERATOR VALIDATION')
    print('=' * 60 + '\n')

    test_wrapper_equality()
    test_mack_table_11_1_subset()
    test_incompressible_blasius_limit()
    test_exact_pr1_thermodynamics()
    test_recovery_factor_vs_formula()
    test_continuation_machinery()
    test_stability_solver_round_trip()
    test_cross_backend_consistency()
    test_dimensionalization_arithmetic()
    test_csv_json_round_trip()
    test_input_validation()
    test_cli_smoke()
    test_slow_cold_wall_mack_rescue()
    test_slow_full_matrix_sweep()

    print('=' * 60)
    print('BOUNDARY-LAYER GENERATOR TESTS COMPLETE')
    print('=' * 60)
