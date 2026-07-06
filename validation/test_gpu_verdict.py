"""Slice-09 GPU verdict filter tests (M3 gate surface).

Filter decisions must be bit-identical to CPU _decaying_candidates on recorded
spectra (or spectra obtained from the deployed solver). Empty-cell verdicts
(None / no_discrete_mode) are reproduced exactly.

GPU-found candidates go through the CPU filter code verbatim (via
select_from_gpu_candidates and the Ozgen two-domain path).

Pinning tests cover transcribed constants against the live discrete-mode driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import pymack  # noqa: E402
from pymack.gpu.verdict import (  # noqa: E402
    OZGEN_CI_ABS_MAX,
    OZGEN_CR_BAND,
    OZGEN_FS_THRESH,
    OZGEN_MATCH_TOL,
    Verdict,
    _decaying_candidates,
    assemble_ozgen_verdict_from_spectra,
    select_from_gpu_candidates,
    two_domain_match,
)
from pymack.sweep import CBand, SEED_NO_ADMISSIBLE_MODE  # noqa: E402
from pymack.temporal_solver import solve_temporal_2d  # noqa: E402

pytestmark = [pytest.mark.gpu]

REPO = Path(__file__).resolve().parents[1]


def _add_verification_path():
    vpath = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_refdigitize"
    if str(vpath) not in sys.path:
        sys.path.insert(0, str(vpath))
    return vpath


def _load_discrete_mode():
    vpath = _add_verification_path()
    import discrete_mode as dm  # noqa: E402

    return dm


def _small_ozgen_point():
    """Representative Ozgen point (hc_001-ish, inside lobe)."""
    return dict(Ma=2.0, alpha=0.08, Re=900.0, N=31, y_max=12.0)


def test_ozgen_constants_pinned_to_driver():
    """Pinning test: our transcribed constants match the LIVE production driver defaults.

    Introspects inspect.signature on discrete_mode (and cross-checks _decaying_candidates
    call behavior). A change to driver defaults must fail this test.
    Source citations: discrete_mode.py:69 (the signature defaults).
    """
    import inspect

    dm = _load_discrete_mode()
    sig = inspect.signature(dm.discrete_mode)
    params = sig.parameters
    # Introspect live defaults from driver (must match our pinned consts)
    assert OZGEN_FS_THRESH == params["fs_thresh"].default
    assert abs(OZGEN_MATCH_TOL - params["match_tol"].default) < 1e-12
    assert OZGEN_CI_ABS_MAX == params["ci_abs_max"].default
    assert OZGEN_CR_BAND == params["cr_band"].default
    # Also cross-check a call uses the same numbers (no silent drift in driver)
    # Source: discrete_mode.py:44 and :89 for _decaying_candidates call site
    prof = pymack.make_flatplate_profile(2.0)
    ev, vec, y = solve_temporal_2d(
        prof, 0.08, 900.0, 2.0, N=31, y_max=12.0, length_scale="L_star"
    )
    # Use introspected defaults in the driver call too
    fs_d = params["fs_thresh"].default
    ci_d = params["ci_abs_max"].default
    cr_d = params["cr_band"].default
    cands_driver = dm._decaying_candidates(
        ev, vec, y, ci_abs_max=ci_d, cr_band=cr_d, fs_thresh=fs_d
    )
    cands_ours = _decaying_candidates(
        ev, vec, y, ci_abs_max=OZGEN_CI_ABS_MAX, cr_band=OZGEN_CR_BAND, fs_thresh=OZGEN_FS_THRESH
    )
    assert len(cands_driver) == len(cands_ours)
    for (cd, fd), (co, fo) in zip(cands_driver, cands_ours):
        assert abs(cd - co) < 1e-12
        assert abs(fd - fo) < 1e-12


def test_decaying_candidates_bit_identical_on_spectra():
    """_decaying_candidates on solver spectra must match CPU driver exactly."""
    dm = _load_discrete_mode()
    prof = pymack.make_flatplate_profile(2.0)
    p = _small_ozgen_point()
    ev, vec, y = solve_temporal_2d(
        prof, p["alpha"], p["Re"], p["Ma"], N=p["N"], y_max=p["y_max"], length_scale="L_star"
    )
    ref = dm._decaying_candidates(
        ev, vec, y, ci_abs_max=0.05, cr_band=(0.05, 0.97), fs_thresh=0.06
    )
    got = _decaying_candidates(
        ev, vec, y, ci_abs_max=0.05, cr_band=(0.05, 0.97), fs_thresh=0.06
    )
    assert len(ref) == len(got)
    for (cr, fr), (cg, fg) in zip(ref, got):
        assert abs(cr - cg) < 1e-12
        assert abs(fr - fg) < 1e-12


def test_two_domain_empty_cell_reproduced():
    """Empty verdict (no discrete mode) cases are reproduced exactly (None cells)."""
    dm = _load_discrete_mode()
    prof = pymack.make_flatplate_profile(2.0)
    # Use a point known to be stable or CS-contaminated for both domains (low Re)
    alpha = 0.01
    Re = 200.0
    ev_s, vec_s, y_s = solve_temporal_2d(prof, alpha, Re, 2.0, N=31, y_max=6.0 * 0.3, length_scale="L_star")
    ev_t, vec_t, y_t = solve_temporal_2d(prof, alpha, Re, 2.0, N=31, y_max=12.0, length_scale="L_star")
    ca = _decaying_candidates(ev_s, vec_s, y_s, ci_abs_max=0.05, cr_band=(0.05, 0.97), fs_thresh=0.06)
    cb = _decaying_candidates(ev_t, vec_t, y_t, ci_abs_max=0.05, cr_band=(0.05, 0.97), fs_thresh=0.06)
    matched = two_domain_match(ca, cb, match_tol=4e-3)
    assert matched == []  # empty after match

    v_short = {"ev": ev_s, "vec": vec_s, "y": y_s}
    v_tall = {"ev": ev_t, "vec": vec_t, "y": y_t}
    verdict = assemble_ozgen_verdict_from_spectra(v_short, v_tall)
    assert verdict.status == "no_discrete_mode"
    assert verdict.value is None
    assert verdict.seed == SEED_NO_ADMISSIBLE_MODE


def test_two_domain_discrete_mode_reproduced():
    """When a y_max-stable decaying mode exists, most-unstable selection matches driver."""
    dm = _load_discrete_mode()
    prof = pymack.make_flatplate_profile(2.0)
    p = _small_ozgen_point()
    # short domain ~6*dstar for M2 (dstar~0.3-ish, use 2.0 absolute for small N)
    y_short = 2.0
    ev_s, vec_s, y_s = solve_temporal_2d(
        prof, p["alpha"], p["Re"], p["Ma"], N=p["N"], y_max=y_short, length_scale="L_star"
    )
    ev_t, vec_t, y_t = solve_temporal_2d(
        prof, p["alpha"], p["Re"], p["Ma"], N=p["N"], y_max=p["y_max"], length_scale="L_star"
    )
    ref = dm.discrete_mode(2.0, p["Re"], p["alpha"], N=p["N"], ymf_pair=(y_short, p["y_max"]))
    v_short = {"ev": ev_s, "vec": vec_s, "y": y_s}
    v_tall = {"ev": ev_t, "vec": vec_t, "y": y_t}
    got = assemble_ozgen_verdict_from_spectra(v_short, v_tall)
    if ref is None:
        assert got.status == "no_discrete_mode"
        assert got.value is None
    else:
        assert got.status == "discrete_mode"
        assert abs(got.value.real - ref["c_r"]) < 1e-9
        assert abs(got.value.imag - ref["c_i"]) < 1e-9
        assert got.n_match >= 1


def test_select_from_gpu_candidates_uses_cpu_filter_verbatim():
    """select_from_gpu_candidates must delegate to _select_temporal_mode exactly."""
    band = CBand(-np.inf, 0.45, ci_abs_max=0.05, label="TS")
    # Simulate GPU-found candidates (duck type with .value)
    class C:
        def __init__(self, z):
            self.value = z
    cands = [C(0.1 + 0.01j), C(0.2 + 0.03j), C(0.3 + 0.001j)]
    sel, seed = select_from_gpu_candidates(cands, band)
    # The most unstable in band is the second
    assert sel is not None
    assert abs(sel.value - (0.2 + 0.03j)) < 1e-12
    # Empty case
    sel2, seed2 = select_from_gpu_candidates([], band)
    assert sel2 is None
    assert seed2 == SEED_NO_ADMISSIBLE_MODE


def test_gpu_verdict_empty_cell_on_out_of_band_spectrum():
    """Out-of-band full spectrum produces honest empty verdict (no fake mode)."""
    # Construct a synthetic spectrum outside any reasonable band
    ev = np.array([0.1 + 0.1j, 2.0 + 0.01j, -1.0 + 0.2j])
    # dummy vec/y that will still compute amp (will be filtered by cr/ci)
    n = 8
    vec = np.random.randn(3 * n, 3) + 1j * np.random.randn(3 * n, 3)
    y = np.linspace(12.0, 0.0, n)
    cands = _decaying_candidates(
        ev, vec, y, ci_abs_max=0.05, cr_band=(0.05, 0.97), fs_thresh=0.06
    )
    assert cands == []
    matched = two_domain_match(cands, cands)
    assert matched == []


# ---------------------------------------------------------------------------
# Stage-2 escalation ladder tests (minimal surface; fractions and provenance)
# ---------------------------------------------------------------------------

from pymack.gpu.escalate import (  # noqa: E402
    R0,
    R4,
    EscalationRecord,
    compute_escalation_fractions,
    escalate_one_point,
    is_ambiguous_or_collar,
    run_async_cpu_qz,
)

from pymack.sweep import CBand  # noqa: E402


def test_is_ambiguous_or_collar_basic():
    band = CBand(0.05, 0.97, ci_abs_max=0.05)
    assert is_ambiguous_or_collar(0.96 + 0.01j, band) is True   # near high edge
    assert is_ambiguous_or_collar(0.50 + 0.01j, band) is False
    assert is_ambiguous_or_collar(None, band) is False


def test_is_ambiguous_or_collar_unbounded_ts():
    """Collar must not nan on -inf (TS band); distance to finite edge."""
    band = CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS")
    # near the finite edge 0.45 should collar; far should not
    assert is_ambiguous_or_collar(0.44 + 0.01j, band) is True
    assert is_ambiguous_or_collar(0.30 + 0.01j, band) is False
    assert is_ambiguous_or_collar(0.10 + 0.01j, band) is False


def test_use_two_domain_no_unboundlocal_and_reproduces():
    """F3: use_two_domain=True must not raise UnboundLocalError('sel'); exercises layer."""
    from pymack.gpu.verdict import assemble_ozgen_verdict_from_spectra as _asm
    prof = pymack.make_flatplate_profile(2.0)
    # small spectra that yield a discrete
    ev_s, vec_s, y_s = solve_temporal_2d(prof, 0.08, 900.0, 2.0, N=31, y_max=2.0, length_scale="L_star")
    ev_t, vec_t, y_t = solve_temporal_2d(prof, 0.08, 900.0, 2.0, N=31, y_max=12.0, length_scale="L_star")
    s_short = {"ev": ev_s, "vec": vec_s, "y": y_s}
    s_tall = {"ev": ev_t, "vec": vec_t, "y": y_t}
    band = CBand(float("-inf"), 0.45, ci_abs_max=0.05)
    # pass full ctx not needed for two-domain R0
    rec = escalate_one_point([], band, rung_context={"point": 5, "family": 0},
                             operator="ozgen_2d", use_two_domain=True,
                             spectra_short=s_short, spectra_tall=s_tall)
    assert rec.rung == R0
    assert rec.value is not None or rec.meta.get("ambiguous") is not None  # either discrete or none
    # also that assemble would have run
    v = _asm(s_short, s_tall)
    assert (rec.value is None) == (v.value is None)


def test_escalate_one_point_r0_default_and_live_rungs():
    band = CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS")
    class C:
        def __init__(self, z, er=0.01):
            self.value = z
            self.edge_ratio = er
    cands = [C(0.1 + 0.01j)]
    rec = escalate_one_point(cands, band, rung_context={"point": 0, "family": 0}, operator="ozgen_2d")
    assert rec.rung == R0
    assert rec.value is not None
    assert rec.seed == 1 or rec.seed in (1, 10)  # R0 base
    # collar + explicit R1 refusal (no reproject) -> rung is refusal, recorded
    cands_collar = [C(0.44 + 0.01j, er=0.05)]
    rec2 = escalate_one_point(cands_collar, band, rung_context={"point": 1, "family": 0}, operator="ozgen_2d")
    assert rec2.rung == "enlarged_reproject_refused"
    assert rec2.meta.get("r1_executed") is False
    # test R4 live trigger via provided reproject that keeps ambiguous + sufficient ctx for R4 execute
    prof = pymack.make_flatplate_profile(2.0)
    def still_collar_reproj(enlarged=False):
        # return cands that remain collar so ladder continues to R4
        class CC:
            def __init__(self, z):
                self.value = z
                self.edge_ratio = 0.01
        return [CC(0.44 + 0.009j)]
    rec4 = escalate_one_point(
        cands_collar, band,
        rung_context={
            "point": 2, "family": 0,
            "profile": prof, "alpha": 0.08, "Re": 900.0, "Ma": 2.0, "N": 31, "y_max": 12.0,
            "reproject": still_collar_reproj,
        },
        operator="ozgen_2d",
    )
    assert rec4.rung == R4
    assert rec4.meta.get("r4_executed") is True
    assert "r1" in rec4.meta and rec4.meta["r1"].get("r1_executed") is True


def test_gpu_candidate_sets_captured_and_two_domain_matches_dm():
    """R3-2: two-domain verdict assembled from GPU-SWEEP-DERIVED candidate sets
    (values + eigenvectors captured in meta) reproduces dm.discrete_mode, with
    SMALL NONZERO deviation (not exact zero, which would mean CPU-vs-CPU).
    """
    import os

    from pymack.scales import delta_star_over_lstar
    from pymack.sweep import CBand as _CBand, temporal_sweep
    from pymack.gpu.verdict import assemble_ozgen_verdict_from_gpu_candidate_sets

    dm = _load_discrete_mode()
    prof = pymack.make_flatplate_profile(2.0)
    dstar = delta_star_over_lstar(prof)
    families = (
        _CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        _CBand(0.45, 0.97, ci_abs_max=0.05, label="Mack"),
    )
    # tiny grid mixing empty cells and RESOLVING Mack-mode nodes at N=31
    # (a=0.17 resolves at both Re; a=0.11/Re=866 is empty -> None-cell character).
    alphas = np.array([0.11, 0.17])
    Res = np.array([866.0, 4500.0])
    N = 31
    ymf_short, ymf_tall = 6.0, 12.0
    y_short, y_tall = ymf_short * dstar, ymf_tall * dstar

    prev = os.environ.get("PYMACK_GPU_CAPTURE_CANDIDATE_SETS")
    os.environ["PYMACK_GPU_CAPTURE_CANDIDATE_SETS"] = "1"
    try:
        res_s = temporal_sweep(prof, alphas, Res, Ma=2.0, N=N, y_max=y_short,
                               length_scale="L_star", operator="ozgen_2d",
                               families=families, backend="gpu", cpu_workers=1)
        res_t = temporal_sweep(prof, alphas, Res, Ma=2.0, N=N, y_max=y_tall,
                               length_scale="L_star", operator="ozgen_2d",
                               families=families, backend="gpu", cpu_workers=1)
    finally:
        if prev is None:
            os.environ.pop("PYMACK_GPU_CAPTURE_CANDIDATE_SETS", None)
        else:
            os.environ["PYMACK_GPU_CAPTURE_CANDIDATE_SETS"] = prev

    sets_s = res_s.meta.get("gpu_candidate_sets")
    sets_t = res_t.meta.get("gpu_candidate_sets")
    assert sets_s and sets_t, "GPU candidate sets were not captured"
    # capture carries eigenvectors (4n rows) not just values
    any_vec = next(v for v in sets_s.values() if v["values"].size)
    assert any_vec["vectors"].shape[0] % 4 == 0
    assert any_vec["vectors"].shape[1] == any_vec["values"].size

    n_re = len(Res)
    n_families = len(families)
    saw_nonzero = False
    for ii, alpha in enumerate(alphas):
        for jj, Re in enumerate(Res):
            pidx = ii * n_re + jj
            empty = {"values": np.zeros(0, dtype=complex), "vectors": np.zeros((0, 0), dtype=complex)}
            ss = [sets_s.get((pidx, k), empty) for k in range(n_families)]
            st = [sets_t.get((pidx, k), empty) for k in range(n_families)]
            v_gpu = assemble_ozgen_verdict_from_gpu_candidate_sets(ss, st)
            ref = dm.discrete_mode(2.0, float(Re), float(alpha), N=N, ymf_pair=(ymf_short, ymf_tall))
            # None-cell character identity (hard)
            assert (v_gpu.value is None) == (ref is None), (pidx, alpha, Re)
            if ref is not None:
                err = abs(complex(v_gpu.value) - complex(ref["c_r"], ref["c_i"]))
                assert err < 1e-6, (pidx, err)  # same mode, refine-floor apart
                if err > 0.0:
                    saw_nonzero = True
    # the tell that this is GPU-derived, not CPU-vs-CPU
    assert saw_nonzero, "expected nonzero GPU-vs-CPU deviation on at least one node"


def test_r0_seed_is_rung0_and_distinct_from_escalated():
    """R3-1 reconciliation: select_from_gpu_candidates returns the R0 seed (=1),
    numerically equal to escalate.RUNG_SEED_BASE[R0] and documented by the legend;
    escalated rungs carry DISTINCT seeds readable from EscalationRecord.seed.
    """
    from pymack.gpu.verdict import SEED_R0_DEVICE_FILTER
    from pymack.gpu.escalate import RUNG_SEED_BASE, R0, R4
    from pymack.gpu.enumerator import seed_code_legend

    # (1) the constant is coherent with the escalate legend base
    assert SEED_R0_DEVICE_FILTER == RUNG_SEED_BASE[R0] == 1
    # (2) it is the value returned on a real R0 hit
    band = CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS")

    class C:
        def __init__(self, z):
            self.value = z

    sel, seed = select_from_gpu_candidates([C(0.2 + 0.03j), C(0.1 + 0.01j)], band)
    assert sel is not None
    assert seed == SEED_R0_DEVICE_FILTER == 1
    # (3) the seed_code_legend documents code 1 as the enumerator candidate source
    assert "1" in seed_code_legend()
    assert "enumerator" in seed_code_legend()["1"].lower()
    # (4) escalated rungs carry DISTINCT provenance seeds (readable from record)
    assert RUNG_SEED_BASE[R4] != RUNG_SEED_BASE[R0]
    assert len(set(RUNG_SEED_BASE.values())) == len(RUNG_SEED_BASE)  # all distinct


def test_escalation_fractions_shape():
    recs = [
        EscalationRecord(0, 0, R0, 1, 0.1+0.01j, 3),
        EscalationRecord(1, 0, R0, 1, 0.2+0.02j, 2),
        EscalationRecord(2, 1, R4, 40, None, 0),
    ]
    fr = compute_escalation_fractions(recs, n_points=2)
    # base keys always present; extra ok
    for k in ("device_filter", "enlarged_reproject", "z128_recheck", "xgeev_deflated", "cpu_qz_async"):
        assert k in fr
    assert abs(fr["device_filter"] - 2/3) < 1e-9
    assert fr["cpu_qz_async"] > 0.0


def test_run_async_cpu_qz_smoke_and_windows_clamp(monkeypatch):
    """Smoke: async CPU rung runs, returns records, clamp applied on win32. Live R4."""
    import pymack.gpu.escalate as esc
    prof = pymack.make_flatplate_profile(2.0)
    # tiny grid
    recs = esc.run_async_cpu_qz(
        prof, [0.08], [900.0], Ma=2.0, N=31, y_max=12.0,
        families=(CBand(-np.inf, 0.45, ci_abs_max=0.05, label="TS"),),
        cpu_workers=2,
    )
    assert len(recs) == 1
    assert recs[0].rung == R4
    assert recs[0].meta.get("worker") == "cpu_qz" or "status" in recs[0].meta
    # clamp path (no crash)
    assert esc._clamp_workers(200) <= 61 if sys.platform == "win32" else True


def test_r3_skip_recorded_for_ozgen():
    """R3 for ozgen must be explicit recorded refusal, not silent."""
    band = CBand(0.45, 0.97, ci_abs_max=0.05)
    class C:
        def __init__(self, z, er=0.01):
            self.value = z
            self.edge_ratio = er
    # collar near 0.97 to force past R0 (if not ambig early)
    cands_col = [C(0.96 + 0.01j, er=0.05)]
    # with reproject that keeps ambig -> R1 executed -> fall to R3 refusal recorded in meta of R4
    def keep_ambig(enlarged=False):
        class CC:
            def __init__(self):
                self.value = 0.96 + 0.01j
                self.edge_ratio = 0.01
        return [CC()]
    prof = pymack.make_flatplate_profile(2.0)
    rec = escalate_one_point(
        cands_col, band,
        rung_context={"point":0,"family":1,"profile":prof,"alpha":0.1,"Re":1000,"reproject":keep_ambig},
        operator="ozgen_2d",
    )
    assert rec.rung == R4
    r3 = rec.meta.get("r3", {})
    assert r3.get("r3_executed") is False
    assert "Ozgen" in str(r3.get("reason", "")) or "singular" in str(r3)

