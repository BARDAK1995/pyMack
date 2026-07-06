"""Slice-08 GPU temporal enumerator and backend-contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import pymack.gpu as pygpu  # noqa: E402

if not pygpu.is_available():  # pragma: no cover - GPU-less CI path
    pytest.skip("no CUDA device available", allow_module_level=True)

import pymack  # noqa: E402
from pymack.gpu.affine import AffineExtractionError  # noqa: E402
from pymack.gpu.enumerator import (  # noqa: E402
    OFFICIAL_CONTOUR_VARIANTS,
    _inside_rect,
    _raw_filter_pad,
    candidate_prefilter,
)
from pymack.gpu.temporal import _production_temporal_candidate_mask  # noqa: E402
from pymack.sweep import CBand, temporal_sweep  # noqa: E402
from pymack.temporal_solver import (  # noqa: E402
    _assemble_temporal_ozgen_2d_evp,
    _scaled_problem,
    solve_temporal_2d,
)
from pymack.spectral import chebyshev_D, physical_derivatives  # noqa: E402
from scipy import linalg  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[1]


def _small_ozgen_kwargs():
    return dict(
        Ma=2.0,
        N=31,
        y_max=12.0,
        length_scale="L_star",
        operator="ozgen_2d",
        families=(CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),),
        cpu_workers=1,
    )


def test_official_contour_variants_match_tournament_report():
    report = json.loads(
        (REPO / "docs" / "gpu" / "benchmarks" / "tournament_report.json")
        .read_text(encoding="utf-8")
    )
    official = report["official_contour_variant_per_family"]
    assert official["ozgen_first_pair"] == "hankel"
    assert official["ozgen_second_pair"] == "hankel"
    assert official["mack_fig10_4_m10_3d"] == "single"
    assert official["mach6_spatial_dense"] == "single"
    assert official["mazhong_first_spatial"] == "window"
    assert official["mazhong_second_spatial"] == "window"
    assert OFFICIAL_CONTOUR_VARIANTS["ozgen_first_pair"].L == 48
    assert OFFICIAL_CONTOUR_VARIANTS["ozgen_first_pair"].K == 3
    assert OFFICIAL_CONTOUR_VARIANTS["ozgen_second_pair"].L == 48
    assert OFFICIAL_CONTOUR_VARIANTS["ozgen_second_pair"].K == 3


def test_production_temporal_candidate_mask_pins_solver_spectrum_gate():
    profile = pymack.make_flatplate_profile(2.0)
    alpha = 0.08
    Re = 900.0
    Ma = 2.0
    N = 31
    y_max = 12.0
    length_scale = "L_star"
    production_values, _vectors, _y = solve_temporal_2d(
        profile,
        alpha,
        Re,
        Ma,
        N=N,
        y_max=y_max,
        length_scale=length_scale,
    )

    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, None)
    bf, D1, D2 = _scaled_problem(profile, y, D1, D2, length_scale)
    A, B = _assemble_temporal_ozgen_2d_evp(
        bf, y, D1, D2, alpha, Re, Ma, 0.72, 1.4
    )
    raw_values, _raw_vectors = linalg.eig(A, B)

    mask = _production_temporal_candidate_mask(raw_values, "ozgen_2d")
    expected_mask = (
        np.isfinite(raw_values.real)
        & np.isfinite(raw_values.imag)
        & (raw_values.real > -0.5)
        & (raw_values.real < 1.5)
        & (np.abs(raw_values.imag) < 0.5)
    )
    assert np.array_equal(np.flatnonzero(mask), np.flatnonzero(expected_mask))

    kept = raw_values[mask]
    kept = kept[np.argsort(-kept.imag)]
    assert np.array_equal(kept, production_values)


def _hard_cell(cell_id):
    manifest = json.loads(
        (REPO / "verification" / "gpu_certification" / "hard_cells" / "truth_manifest.json")
        .read_text(encoding="utf-8")
    )
    return next(c for c in manifest["cells"] if c["id"] == cell_id)


def test_hc035_regression_live_adaptive_resplit_matches_cpu_qz():
    cell = _hard_cell("hc_035")
    p = cell["params"]
    profile = pymack.make_flatplate_profile(float(p["Ma"]))
    band = CBand(
        float(p["cr_band"][0]),
        float(p["cr_band"][1]),
        ci_abs_max=float(p.get("phys_ci_abs", p.get("ci_cap", 0.5))),
        label="hc_035",
    )
    kwargs = dict(
        Ma=float(p["Ma"]),
        N=int(p["N"]),
        y_max=float(p["y_max"]),
        Pr=float(p["Pr"]),
        gamma=float(p["gamma"]),
        lambda_mu_ratio=float(p["lambda_mu_ratio"]),
        beta=float(p["beta"]),
        operator="mack_3d",
        families=(band,),
        cpu_workers=1,
    )
    gpu_res = temporal_sweep(
        profile,
        [float(p["alpha"])],
        [float(p["R"])],
        backend="gpu",
        **kwargs,
    )
    cpu_res = temporal_sweep(
        profile,
        [float(p["alpha"])],
        [float(p["R"])],
        backend="cpu",
        **kwargs,
    )
    gpu_family = gpu_res.families[0]
    cpu_family = cpu_res.families[0]
    assert np.array_equal(gpu_family.converged, cpu_family.converged)
    assert abs(gpu_family.c[0, 0] - cpu_family.c[0, 0]) <= 1.0e-9
    assert gpu_family.residual[0, 0] <= 1.0e-8
    diag = gpu_res.meta["gpu_enumerator"]["diagnostics"][0]
    assert diag["adaptive_response"]["triggered"] is True
    assert diag["adaptive_response"]["n_resplits"] >= 1
    assert diag["rank"]["rank_clipped"] is True


def test_gpu_backend_reverifies_affine_and_runs_device_contour_path():
    profile = pymack.make_flatplate_profile(2.0)
    gpu_res = temporal_sweep(
        profile,
        [0.08],
        [900.0],
        backend="gpu",
        **_small_ozgen_kwargs(),
    )
    cpu_res = temporal_sweep(
        profile,
        [0.08],
        [900.0],
        backend="cpu",
        **_small_ozgen_kwargs(),
    )
    assert gpu_res.meta["backend"] == "gpu"
    assert gpu_res.meta["gpu_engine_status"] == "device_contour_projection"
    assert gpu_res.meta["operator_source"] == "gpu_contour_projection_device"
    assert gpu_res.meta["affine_reverified"] is True
    assert gpu_res.meta["affine_probe_box_in_fingerprint"] is True
    assert len(gpu_res.meta["affine_fingerprint"]) == 64
    assert "1" in gpu_res.meta["seed_codes"]
    assert gpu_res.meta["n_failed_points"] == 0
    for gpu_family, cpu_family in zip(gpu_res.families, cpu_res.families):
        assert np.array_equal(gpu_family.converged, cpu_family.converged)
        assert np.allclose(
            gpu_family.c[gpu_family.converged],
            cpu_family.c[cpu_family.converged],
            rtol=1.0e-9,
            atol=1.0e-9,
        )
        assert np.all(gpu_family.seed_map[gpu_family.converged] > 0)
        assert np.all(gpu_family.residual[gpu_family.converged] <= 1.0e-8)


@pytest.mark.parametrize(
    ("alpha", "Re"),
    [
        (0.11, 820.0),
        (0.111, 300.0),
    ],
)
def test_b1_ozgen_ymax12_drift_regression_matches_cpu(alpha, Re):
    profile = pymack.make_flatplate_profile(2.0)
    gpu_res = temporal_sweep(
        profile,
        [alpha],
        [Re],
        backend="gpu",
        **_small_ozgen_kwargs(),
    )
    cpu_res = temporal_sweep(
        profile,
        [alpha],
        [Re],
        backend="cpu",
        **_small_ozgen_kwargs(),
    )
    gpu_family = gpu_res.families[0]
    cpu_family = cpu_res.families[0]
    assert np.array_equal(gpu_family.converged, cpu_family.converged)
    if cpu_family.converged[0, 0]:
        assert abs(gpu_family.c[0, 0] - cpu_family.c[0, 0]) <= 1.0e-9
    else:
        assert gpu_family.seed_map[0, 0] == -1


def test_affine_trip_falls_back_to_cpu_with_error_meta(monkeypatch):
    import pymack.gpu.temporal as gpu_temporal

    def forced_trip(*args, **kwargs):
        raise AffineExtractionError("forced held-out trip")

    monkeypatch.setattr(gpu_temporal, "build_temporal_affine_cache", forced_trip)
    profile = pymack.make_flatplate_profile(2.0)
    result = temporal_sweep(
        profile,
        [0.08],
        [900.0],
        backend="gpu",
        **_small_ozgen_kwargs(),
    )
    assert result.meta["backend"] == "gpu"
    assert result.meta["gpu_engine_status"] == "cpu_qz_fallback_after_affine_trip"
    assert result.meta["affine_reverified"] is False
    assert "forced held-out trip" in result.meta["affine_error"]
    assert result.meta["operator_source"] == "cpu_qz_fallback_affine_trip"


def test_gpu_backend_is_deterministic_for_fixed_small_sweep():
    profile = pymack.make_flatplate_profile(2.0)
    r1 = temporal_sweep(profile, [0.08], [900.0], backend="gpu", **_small_ozgen_kwargs())
    r2 = temporal_sweep(profile, [0.08], [900.0], backend="gpu", **_small_ozgen_kwargs())
    f1, f2 = r1.families[0], r2.families[0]
    assert np.array_equal(f1.c, f2.c, equal_nan=True)
    assert np.array_equal(f1.residual, f2.residual, equal_nan=True)
    assert np.array_equal(f1.seed_map, f2.seed_map)
    assert np.array_equal(f1.mode_index, f2.mode_index)


def test_candidate_prefilter_only_removes_certainly_inadmissible_values():
    values = np.array([0.7 + 1e-3j, 0.2 + 1e-3j, 0.8 + 0.2j, np.nan + 0j])
    mask = candidate_prefilter(
        values,
        cr_band=(0.4, 0.95),
        ci_abs_max=0.05,
        bv_norm=np.array([1e-8, 1e-8, 1e-8, 1e-8]),
        bv_floor=1e-10,
        residual=np.array([1e-12, 1e-12, 1e-12, 1e-12]),
        residual_max=1e-8,
    )
    assert mask.tolist() == [True, False, False, False]
    mask = candidate_prefilter(
        np.array([0.7 + 1e-3j]),
        cr_band=(0.4, 0.95),
        bv_norm=np.array([1e-12]),
        bv_floor=1e-10,
    )
    assert mask.tolist() == [False]


def test_raw_projection_collar_keeps_band_edge_candidate_for_polish():
    rect = {"real": (0.0, 1.0), "imag": (-0.1, 0.1)}
    pad = _raw_filter_pad(rect)
    raw_edge = complex(1.0 + 0.5 * pad, 0.0)
    assert not _inside_rect(raw_edge, rect, pad=0.0)
    assert _inside_rect(raw_edge, rect, pad=pad)

    polished = np.array([0.999999999 + 0.0j, 1.000000001 + 0.0j])
    strict = candidate_prefilter(
        polished,
        cr_band=(0.0, 1.0),
        ci_abs_max=0.1,
        bv_norm=np.array([1.0, 1.0]),
        bv_floor=1.0e-10,
        residual=np.array([1.0e-12, 1.0e-12]),
        residual_max=1.0e-8,
    )
    assert strict.tolist() == [True, False]


def test_empty_cell_truth_recorded_in_corpus_subsample():
    manifest = json.loads(
        (REPO / "verification" / "gpu_certification" / "hard_cells" / "truth_manifest.json")
        .read_text(encoding="utf-8")
    )
    empty = [c for c in manifest["cells"] if c["verdict"]["status"] == "no_discrete_mode"]
    assert empty, "hard-cell corpus must contain empty production verdicts"
    assert any(c["id"] == "hc_012" for c in empty)
