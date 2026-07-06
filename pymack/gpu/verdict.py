"""Verdict assembly + escalation support for the GPU temporal engine (slice 09).

GPU-found candidates ALWAYS pass through the EXISTING CPU filter code VERBATIM
(_select_temporal_mode for per-height/family in-sweep path; full two-domain/fs/c_r
+ match + argmax chain via assemble_ozgen_verdict_from_spectra / two_domain_match
in the verdict layer for tests, audit, bench --verify-verdicts, and escalate
use_two_domain=True path). Device-side filters are remove-only pre-screens.

Two-domain Ozgen family matching (match_tol=4e-3, fs=0.06) and empty-cell
(None) semantics are character-identical to the deployed discrete-mode path
(verification/mixed_mode/ozgen_fig3/_refdigitize/discrete_mode.py:66).

Escalation ladder (escalate.py) feeds ambiguous/collar cells upward; each rung
updates provenance in seed_map / meta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pymack.sweep import (
    SEED_NO_ADMISSIBLE_MODE,
    SEED_QZ_FULL_SPECTRUM,
    _select_temporal_mode,
)

# ---------------------------------------------------------------------------
# Rung-0 provenance seed (R3-1 reconciliation).
#
# select_from_gpu_candidates is the RUNG-0 (device_filter) selection.  It always
# returns the R0 engine seed, which the slice-06 legend and seed_code_legend()
# document as code 1 = "GPU contour enumerator candidate source".  This is the
# SAME value as escalate.RUNG_SEED_BASE[R0]; the two are kept numerically
# consistent by test_r0_seed_is_rung0_and_distinct_from_escalated.  Distinct
# rung provenance for ESCALATED points (R1..R4) is carried by
# EscalationRecord.seed (RUNG_SEED_BASE 10/20/30/40 + family), recorded into the
# per-point diagnostics/seed_map by temporal.py, NOT by this function.  We
# hardcode the literal here (rather than importing escalate) to break the
# verdict<->escalate import cycle; the test enforces they never drift apart.
SEED_R0_DEVICE_FILTER = 1

# ---------------------------------------------------------------------------
# Pinned constants (transcribed from deployed Ozgen discrete-mode driver).
# Any change requires a pinning test that re-derives against the live source.
# Source at time of implementation (with line citations):
#   verification/mixed_mode/ozgen_fig3/_refdigitize/discrete_mode.py:69
#     (defaults: fs_thresh=0.06, match_tol=4e-3, ci_abs_max=0.05, cr_band=(0.05,0.97))
#   verification/gpu_certification/hard_cells/build_truth_set.py (corpus)
# ---------------------------------------------------------------------------
OZGEN_FS_THRESH = 0.06          # fs_thresh (discrete_mode.py:69)
OZGEN_MATCH_TOL = 4.0e-3        # match_tol (discrete_mode.py:69)
OZGEN_CI_ABS_MAX = 0.05         # default for Ozgen families (discrete_mode.py:69)
OZGEN_CR_BAND = (0.05, 0.97)    # broad physical band used for decaying filter (discrete_mode.py:69)

# Default two-domain pair (short ~6*dstar, tall=12.0 absolute in L* units
# for the family gate; actual y_max passed at call time).
DEFAULT_OZGEN_YMF_PAIR = (6.0, 12.0)


@dataclass
class Verdict:
    """One family verdict from GPU candidates (or spectrum for tests)."""

    status: str                 # "discrete_mode" | "no_discrete_mode"
    value: complex | None
    fs: float | None
    n_match: int
    seed: int                   # provenance code (universal <=0 or engine rung)
    rung: str = "cpu_filter"    # which escalation rung produced it


def _decaying_candidates(
    ev: np.ndarray,
    vec: np.ndarray,
    y: np.ndarray,
    *,
    ci_abs_max: float,
    cr_band: tuple[float, float],
    fs_thresh: float,
) -> list[tuple[complex, float]]:
    """Verbatim filter from discrete_mode.py:_decaying_candidates.

    GPU candidates or full spectra are passed here; the logic is the CPU
    authority for freestream decay + cr/ci band admission.  Returns list of
    (c, fs_ratio) for modes that pass.  Character-identical None/empty handling.
    """
    ev = np.asarray(ev, dtype=complex)
    vec = np.asarray(vec, dtype=complex)
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n == 0 or ev.size == 0:
        return []
    ntop = max(2, n // 10)
    # Ozgen 2D temporal state blocks: u, v, T (3 blocks).  Support both
    # (3n, n_eig) and (n_eig, 3n) layouts by detecting.
    if vec.ndim == 1:
        vec = vec.reshape(-1, 1)
    if vec.shape[0] == 3 * n:
        n_eig = vec.shape[1]
        get_block = lambda k, j: vec[k * n : (k + 1) * n, j]
    elif vec.shape[1] == 3 * n:
        n_eig = vec.shape[0]
        get_block = lambda k, j: vec[j, k * n : (k + 1) * n]
    else:
        # Fallback: treat as generic stacked; use first 3n if larger.
        m = min(vec.shape[0], 3 * n)
        n_eig = 1 if vec.ndim == 1 else vec.shape[1 if vec.shape[0] > vec.shape[1] else 0]
        # For safety fall back to amplitude on leading slice
        get_block = lambda k, j: vec[k * n : (k + 1) * n, j] if vec.shape[0] >= 3 * n else vec[:n, j] if vec.ndim > 1 else vec[:n]

    out: list[tuple[complex, float]] = []
    for k in range(ev.size):
        c = ev[k] if ev.ndim == 1 else ev[k]
        if not (abs(c.imag) < ci_abs_max and cr_band[0] < c.real < cr_band[1]):
            continue
        try:
            u = get_block(0, k)
            v = get_block(1, k)
            T = get_block(2, k)
        except Exception:
            # conservative: skip if vec layout unexpected
            continue
        amp = np.abs(u) + np.abs(v) + np.abs(T)
        mx = float(amp.max()) if amp.size else 0.0
        if mx <= 0.0:
            continue
        fs = float(amp[:ntop].max() / mx)
        if fs < fs_thresh:
            out.append((complex(c), fs))
    return out


def two_domain_match(
    cands_a: list[tuple[complex, float]],
    cands_b: list[tuple[complex, float]],
    *,
    match_tol: float = OZGEN_MATCH_TOL,
) -> list[tuple[complex, float]]:
    """Y_max-stationary match across two domains (verbatim discrete_mode logic).

    A candidate from a (short) matches if min |c - bc| < match_tol for bc in b.
    Returns the matched list from a, preserving fs.
    """
    if not cands_a or not cands_b:
        return []
    bc = np.array([c for c, _ in cands_b], dtype=complex)
    matched: list[tuple[complex, float]] = []
    for c, fs in cands_a:
        if np.min(np.abs(c - bc)) < float(match_tol):
            matched.append((c, fs))
    return matched


def assemble_ozgen_verdict_from_spectra(
    spectra_short: dict[str, Any],
    spectra_tall: dict[str, Any],
    *,
    ci_abs_max: float = OZGEN_CI_ABS_MAX,
    cr_band: tuple[float, float] = OZGEN_CR_BAND,
    fs_thresh: float = OZGEN_FS_THRESH,
    match_tol: float = OZGEN_MATCH_TOL,
) -> Verdict:
    """Full two-domain Ozgen discrete verdict from two spectra (for tests/audit).

    spectra_* are dicts with keys "ev", "vec", "y" (as produced by solve_temporal_2d).
    Uses _decaying_candidates + two_domain_match + most-unstable argmax verbatim.
    Returns Verdict with status "discrete_mode" or "no_discrete_mode" (None cell).
    """
    ev_s = np.asarray(spectra_short["ev"], dtype=complex)
    vec_s = np.asarray(spectra_short["vec"], dtype=complex)
    y_s = np.asarray(spectra_short["y"], dtype=float)
    ev_t = np.asarray(spectra_tall["ev"], dtype=complex)
    vec_t = np.asarray(spectra_tall["vec"], dtype=complex)
    y_t = np.asarray(spectra_tall["y"], dtype=float)

    ca = _decaying_candidates(
        ev_s, vec_s, y_s, ci_abs_max=ci_abs_max, cr_band=cr_band, fs_thresh=fs_thresh
    )
    cb = _decaying_candidates(
        ev_t, vec_t, y_t, ci_abs_max=ci_abs_max, cr_band=cr_band, fs_thresh=fs_thresh
    )
    matched = two_domain_match(ca, cb, match_tol=match_tol)
    if not matched:
        return Verdict(
            status="no_discrete_mode",
            value=None,
            fs=None,
            n_match=0,
            seed=SEED_NO_ADMISSIBLE_MODE,
            rung="cpu_filter",
        )
    # most-unstable (largest imag part), stable tie-break by input order
    matched.sort(key=lambda r: -r[0].imag)
    c, fs = matched[0]
    return Verdict(
        status="discrete_mode",
        value=c,
        fs=float(fs),
        n_match=len(matched),
        seed=SEED_QZ_FULL_SPECTRUM,
        rung="cpu_filter",
    )


def _union_gpu_candidate_set(sets: list[dict[str, Any]]):
    """Union per-family GPU candidate ``{values, vectors}`` into one spectrum.

    ``sets`` are the per-(family) GPU production-filtered candidate captures for
    ONE domain height at ONE node (values shape (m,), vectors shape (4n, m) in
    the Ozgen [u, v, T, p] block layout).  The TS and Mack contour rectangles
    overlap by a small guard near c_r=0.45, so duplicates are removed by value.
    Returns ``(ev, vec)`` ready for the verbatim decaying/match/argmax chain.
    """
    vals: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    for s in sets:
        v = np.asarray(s["values"], dtype=complex).ravel()
        if v.size == 0:
            continue
        m = np.asarray(s["vectors"], dtype=complex)
        if m.ndim != 2 or m.shape[1] != v.size:
            raise ValueError(
                f"GPU candidate vectors shape {m.shape} inconsistent with "
                f"{v.size} values"
            )
        vals.append(v)
        vecs.append(m)
    if not vals:
        return np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex)
    allv = np.concatenate(vals)
    allm = np.concatenate(vecs, axis=1)
    keep: list[int] = []
    seen: list[complex] = []
    for i, z in enumerate(allv):
        if all(abs(z - s) > 1.0e-9 * max(1.0, abs(s)) for s in seen):
            keep.append(i)
            seen.append(complex(z))
    idx = np.asarray(keep, dtype=int)
    return allv[idx], allm[:, idx]


def assemble_ozgen_verdict_from_gpu_candidate_sets(
    sets_short: list[dict[str, Any]],
    sets_tall: list[dict[str, Any]],
    *,
    ci_abs_max: float = OZGEN_CI_ABS_MAX,
    cr_band: tuple[float, float] = OZGEN_CR_BAND,
    fs_thresh: float = OZGEN_FS_THRESH,
    match_tol: float = OZGEN_MATCH_TOL,
) -> Verdict:
    """Two-domain Ozgen verdict assembled from GPU-SWEEP-DERIVED candidates.

    This is the R3-2 gate path: the values + eigenvectors are the ones the GPU
    engine actually found and filtered (captured in
    ``meta['gpu_candidate_sets']``), NOT fresh CPU spectra.  The freestream
    decay ratio, two-domain match, and most-unstable argmax are the pinned
    verbatim chain (``assemble_ozgen_verdict_from_spectra``); the only new work
    is unioning the per-family captures per height.  Because the GPU eigenpairs
    differ from the deployed full-QZ ones at the ~1e-12 refinement floor, this
    produces SMALL NONZERO deviations vs ``dm.discrete_mode`` (exact zeros would
    mean CPU-vs-CPU).
    """
    ev_s, vec_s = _union_gpu_candidate_set(sets_short)
    ev_t, vec_t = _union_gpu_candidate_set(sets_tall)
    n_s = vec_s.shape[0] // 4 if vec_s.size else 0
    n_t = vec_t.shape[0] // 4 if vec_t.size else 0
    return assemble_ozgen_verdict_from_spectra(
        {"ev": ev_s, "vec": vec_s, "y": np.zeros(n_s, dtype=float)},
        {"ev": ev_t, "vec": vec_t, "y": np.zeros(n_t, dtype=float)},
        ci_abs_max=ci_abs_max,
        cr_band=cr_band,
        fs_thresh=fs_thresh,
        match_tol=match_tol,
    )


def select_from_gpu_candidates(
    candidates: list[Any],
    band: Any,
    *,
    operator: str = "ozgen_2d",
) -> tuple[Any | None, int]:
    """Run EXISTING CPU _select_temporal_mode verbatim on GPU-found candidates.

    candidates: list of objects with .value (Candidate or duck-type).
    Returns (selected_candidate_or_None, seed_code).
    Device candidates have already been pre-screened; this is the authority.
    """
    if not candidates:
        return None, SEED_NO_ADMISSIBLE_MODE
    values = np.asarray([c.value for c in candidates], dtype=complex)
    idx = _select_temporal_mode(values, band)
    if idx is None:
        return None, SEED_NO_ADMISSIBLE_MODE
    # R0 (device_filter) engine seed; escalated rungs override provenance via
    # EscalationRecord.seed in escalate.py (F3(i) / R3-1 reconciliation).
    return candidates[int(idx)], int(SEED_R0_DEVICE_FILTER)


__all__ = [
    "Verdict",
    "SEED_R0_DEVICE_FILTER",
    "OZGEN_FS_THRESH",
    "OZGEN_MATCH_TOL",
    "OZGEN_CI_ABS_MAX",
    "OZGEN_CR_BAND",
    "_decaying_candidates",
    "two_domain_match",
    "assemble_ozgen_verdict_from_spectra",
    "assemble_ozgen_verdict_from_gpu_candidate_sets",
    "select_from_gpu_candidates",
]
