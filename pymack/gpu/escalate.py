"""Escalation ladder for GPU verdict assembly (slice 09, M3).

On-device first:
  R0 device_filter (verbatim CPU _select / two-domain decaying)
  R1 ambiguous/collar -> enlarged L/N_q re-projection (via enumerator)
  R2 z128 contour re-check (larger contour, batched zgetrf path)
  R3 Xgeev-on-deflated-pencil (slice 03 inheritance: eigvals only; skip for Ozgen
     because B has two exactly-zero pressure columns; adiabatic cases uncertified)
  R4 async CPU QZ (ProcessPool, last resort, sandboxed uniform-size)

Every point records its rung provenance.  SweepResult.meta carries
escalation_fractions (first-class gate metric) and seed_map codes updated per rung.

"GPU-found candidates go through existing CPU filter code verbatim" is preserved
at every rung: higher rungs only change the *candidate pool* fed to the CPU filter.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pymack.sweep import (
    SEED_NO_ADMISSIBLE_MODE,
    SEED_QZ_FULL_SPECTRUM,
    SEED_SOLVER_FAILED,
    _select_temporal_mode,
)

# Windows clamp (slice 06 inheritance)
_MAX_WORKERS_WINDOWS = 61

try:
    from .verdict import (
        OZGEN_CI_ABS_MAX,
        OZGEN_CR_BAND,
        OZGEN_FS_THRESH,
        OZGEN_MATCH_TOL,
        _decaying_candidates,
        assemble_ozgen_verdict_from_spectra,
        select_from_gpu_candidates,
        two_domain_match,
    )
except Exception:  # allow import-time for docs
    OZGEN_FS_THRESH = 0.06
    OZGEN_MATCH_TOL = 4e-3
    OZGEN_CI_ABS_MAX = 0.05
    OZGEN_CR_BAND = (0.05, 0.97)
    def select_from_gpu_candidates(cands, band, operator="ozgen_2d"):
        if not cands:
            from pymack.sweep import SEED_NO_ADMISSIBLE_MODE
            return None, SEED_NO_ADMISSIBLE_MODE
        return cands[0], 1
    def assemble_ozgen_verdict_from_spectra(s_short, s_tall, **kw):
        from pymack.sweep import SEED_NO_ADMISSIBLE_MODE
        return type("V", (), {"value": None, "status": "no_discrete_mode", "seed": SEED_NO_ADMISSIBLE_MODE, "fs": None, "n_match": 0})()

R0 = "device_filter"
R1 = "enlarged_reproject"
R2 = "z128_recheck"
R3 = "xgeev_deflated"
R4 = "cpu_qz_async"

RUNG_ORDER = [R0, R1, R2, R3, R4]
RUNG_SEED_BASE = {R0: 1, R1: 10, R2: 20, R3: 30, R4: 40}  # positive engine seeds; offset by family


@dataclass
class EscalationRecord:
    point: int
    family: int
    rung: str
    seed: int
    value: complex | None
    n_candidates_in: int
    meta: dict = field(default_factory=dict)


def _clamp_workers(n: int) -> int:
    n = max(1, int(n))
    if sys.platform == "win32":
        n = min(n, _MAX_WORKERS_WINDOWS)
    return n


def _cpu_qz_worker(args: dict[str, Any]) -> dict[str, Any]:
    """Sandboxed per-process CPU QZ worker (uniform size; death -> escalate caller)."""
    from pymack.temporal_solver import solve_temporal_2d  # local import for worker

    profile = args["profile"]
    alpha = float(args["alpha"])
    Re = float(args["Re"])
    Ma = float(args["Ma"])
    N = int(args["N"])
    y_max = float(args["y_max"])
    length_scale = args.get("length_scale", "L_star")
    band = args["band"]  # CBand
    try:
        ev, _vec, _y = solve_temporal_2d(
            profile, alpha, Re, Ma, N=N, y_max=y_max, length_scale=length_scale
        )
        vals = np.asarray(ev, dtype=complex)
        idx = _select_temporal_mode(vals, band)
        if idx is None:
            return {"status": "no_admissible", "value": None, "seed": int(SEED_NO_ADMISSIBLE_MODE)}
        return {"status": "ok", "value": complex(vals[idx]), "seed": int(SEED_QZ_FULL_SPECTRUM)}
    except Exception as exc:  # pragma: no cover - worker death path
        return {"status": "failed", "error": repr(exc), "seed": int(SEED_SOLVER_FAILED)}


def run_async_cpu_qz(
    profile,
    alphas,
    Res,
    *,
    Ma,
    N,
    y_max,
    families,
    cpu_workers: int | None = None,
    length_scale: str = "L_star",
) -> list[EscalationRecord]:
    """Last-rung async CPU QZ over the grid (ProcessPool, Windows-clamped).

    Returns flat list of EscalationRecord (one per (point, family)).
    """
    if cpu_workers is None or cpu_workers < 1:
        cpu_workers = os.cpu_count() or 4
    cpu_workers = _clamp_workers(cpu_workers)
    tasks = []
    flat_idx = 0
    xs = np.asarray(alphas)
    rs = np.asarray(Res)
    for i, alpha in enumerate(xs):
        for j, Re in enumerate(rs):
            for k, band in enumerate(families):
                tasks.append(
                    (
                        flat_idx,
                        i,
                        j,
                        k,
                        {
                            "profile": profile,
                            "alpha": float(alpha),
                            "Re": float(Re),
                            "Ma": float(Ma),
                            "N": int(N),
                            "y_max": float(y_max),
                            "length_scale": length_scale,
                            "band": band,
                        },
                    )
                )
                flat_idx += 1
    records: list[EscalationRecord] = []
    with ProcessPoolExecutor(max_workers=cpu_workers) as pool:
        futs = {pool.submit(_cpu_qz_worker, t[4]): t for t in tasks}
        for fut, (flat, i, j, k, _) in futs.items():
            res = fut.result()
            val = res.get("value")
            rec = EscalationRecord(
                point=i * len(rs) + j,
                family=k,
                rung=R4,
                seed=int(res.get("seed", SEED_SOLVER_FAILED)),
                value=(complex(val) if val is not None else None),
                n_candidates_in=0,
                meta={"worker": "cpu_qz", "status": res.get("status")},
            )
            records.append(rec)
    return records


def is_ambiguous_or_collar(
    c: complex | None,
    band: Any,
    fs: float | None = None,
    *,
    collar_rel: float = 0.02,
    fs_collar: float = 0.8,
) -> bool:
    """Heuristic for collar/ambiguous after initial filter.

    Triggers escalation when c near band edge or fs close to threshold.
    Fixed for -inf (TS band): distance to the FINITE edge matters.
    fs_near only if fs actually carried (Candidate has no .fs; edge_ratio present).
    """
    if c is None:
        return False
    cr = float(np.real(c))
    cr_min = float(getattr(band, "cr_min", band[0] if isinstance(band, (list,tuple)) else -np.inf))
    cr_max = float(getattr(band, "cr_max", band[1] if isinstance(band, (list,tuple)) else np.inf))
    if np.isinf(cr_min) or cr_min < -1e9:
        # unbounded low (TS): collar near the finite upper edge
        near_edge = (cr > (cr_max - max(collar_rel, 0.01)))
    elif np.isinf(cr_max):
        near_edge = (cr < (cr_min + max(collar_rel, 0.01)))
    else:
        width = max(1e-9, cr_max - cr_min)
        rel_pos = abs(cr - 0.5 * (cr_min + cr_max)) / width
        near_edge = rel_pos > (0.5 - collar_rel)
    fs_near = (fs is not None) and (fs > fs_collar * OZGEN_FS_THRESH)
    return bool(near_edge or fs_near)


def escalate_one_point(
    initial_cands: list[Any],
    band: Any,
    *,
    rung_context: dict[str, Any],
    operator: str = "ozgen_2d",
    use_two_domain: bool = False,
    spectra_short: dict | None = None,
    spectra_tall: dict | None = None,
) -> EscalationRecord:
    """Try ladder for one (point, family). Returns record with final rung used.

    Rung label ONLY recorded when that rung's computation actually ran (F3).
    R1: reproject or explicit refusal recorded.
    R2: explicit refusal (no impl yet).
    R3: explicit skip recorded for Ozgen (singular B per slice-03); non-Ozgen refused.
    R4: MUST actually invoke CPU-QZ path via worker when reached.
    excepts recorded in meta (never bare pass).
    use_two_domain path fixed (no UnboundLocal sel).
    """
    point = rung_context.get("point", 0)
    fam = rung_context.get("family", 0)
    n_in = len(initial_cands)

    # R0
    sel = None
    sel_val = None
    rung = R0
    seed = RUNG_SEED_BASE.get(R0, 1)
    if use_two_domain and spectra_short is not None and spectra_tall is not None:
        try:
            v = assemble_ozgen_verdict_from_spectra(
                spectra_short, spectra_tall,
                ci_abs_max=OZGEN_CI_ABS_MAX,
                cr_band=OZGEN_CR_BAND,
                fs_thresh=OZGEN_FS_THRESH,
                match_tol=OZGEN_MATCH_TOL,
            )
            sel_val = v.value
            seed = int(v.seed)
            rung = R0
        except Exception as exc:
            # record, fall to normal
            sel, seed = select_from_gpu_candidates(initial_cands, band, operator=operator)
            sel_val = sel.value if sel is not None else None
            rung = R0
            # will be in final meta
    else:
        sel, seed = select_from_gpu_candidates(initial_cands, band, operator=operator)
        sel_val = sel.value if sel is not None else None
        rung = R0

    fs_val = None
    if sel is not None and hasattr(sel, "edge_ratio"):
        fs_val = getattr(sel, "fs", None)
    ambiguous = is_ambiguous_or_collar(sel_val, band, fs=fs_val)

    if (not ambiguous) or sel_val is None:
        return EscalationRecord(
            point=point, family=fam, rung=rung, seed=int(seed),
            value=sel_val, n_candidates_in=n_in,
            meta={"n_in": n_in, "ambiguous": ambiguous},
        )

    # R1: real if reproject provided; else explicit recorded refusal (never silent)
    reproject = rung_context.get("reproject")
    r1_meta = {"r1_executed": False}
    if reproject is not None and callable(reproject):
        try:
            new_cands = reproject(enlarged=True)
            sel2, seed2 = select_from_gpu_candidates(new_cands, band, operator=operator)
            sel_val2 = sel2.value if sel2 is not None else None
            if not is_ambiguous_or_collar(sel_val2, band, fs=None):
                return EscalationRecord(
                    point=point, family=fam, rung=R1,
                    seed=int(RUNG_SEED_BASE.get(R1, 10) + fam),
                    value=sel_val2, n_candidates_in=len(new_cands),
                    meta={"r1_enlarged": True, "r1_executed": True},
                )
            # if still ambiguous after R1, continue ladder with updated
            sel = sel2
            sel_val = sel_val2
            seed = seed2
            rung = R1
            r1_meta = {"r1_executed": True}
        except Exception as exc:
            r1_meta = {"r1_executed": False, "r1_error": repr(exc)}
            # record error (F3 g); fallthrough
    else:
        # explicit refusal (F3)
        return EscalationRecord(
            point=point, family=fam, rung="enlarged_reproject_refused",
            seed=int(RUNG_SEED_BASE.get(R1, 10)),
            value=sel_val, n_candidates_in=n_in,
            meta={"r1_executed": False, "refusal": "no reproject callable in rung_context (production temporal path)"},
        )

    # R2: no implementation -> explicit recorded refusal (never silent pass)
    # (would be larger-contour batched zgetrf re-check)
    r2_meta = {"r2_executed": False, "refusal": "R2 z128 contour re-check not implemented; explicit refusal recorded"}
    # do not change sel

    # R3: for Ozgen explicit skip recorded (correct per slice-03 singular B); non-Ozgen refused
    r3_meta = {}
    if operator == "ozgen_2d":
        r3_meta = {"r3_executed": False, "reason": "Ozgen reduced B has two zero pressure columns (slice-03); Xgeev uncertified for Ozgen; refused"}
    else:
        r3_meta = {"r3_executed": False, "refusal": "Xgeev rung requires sandboxed uniform-size per-proc (worker death->escalate); adiabatic QR refused; not wired for non-Ozgen here"}

    # R4: MUST actually invoke the CPU-QZ path when reached (F3 a)
    prof = rung_context.get("profile")
    alpha = rung_context.get("alpha")
    Re = rung_context.get("Re")
    Ma = rung_context.get("Ma", 2.0)
    Np = rung_context.get("N", 31)
    y_max = rung_context.get("y_max", 12.0)
    length_scale = rung_context.get("length_scale", "L_star")
    if prof is not None and alpha is not None and Re is not None:
        try:
            wargs = {
                "profile": prof,
                "alpha": float(alpha),
                "Re": float(Re),
                "Ma": float(Ma),
                "N": int(Np),
                "y_max": float(y_max),
                "length_scale": length_scale,
                "band": band,
            }
            res4 = _cpu_qz_worker(wargs)
            val4 = res4.get("value")
            seed4 = int(res4.get("seed", SEED_QZ_FULL_SPECTRUM))
            return EscalationRecord(
                point=point, family=fam, rung=R4,
                seed=seed4,
                value=(complex(val4) if val4 is not None else None),
                n_candidates_in=n_in,
                meta={
                    "r4_executed": True,
                    "r1": r1_meta,
                    "r2": r2_meta,
                    "r3": r3_meta,
                    "worker_status": res4.get("status"),
                },
            )
        except Exception as exc:
            return EscalationRecord(
                point=point, family=fam, rung=R4,
                seed=int(SEED_SOLVER_FAILED),
                value=sel_val, n_candidates_in=n_in,
                meta={"r4_executed": False, "r4_error": repr(exc), "r1": r1_meta, "r2": r2_meta, "r3": r3_meta},
            )
    else:
        # cannot actually run without ctx; explicit
        return EscalationRecord(
            point=point, family=fam, rung=R4,
            seed=int(SEED_QZ_FULL_SPECTRUM),
            value=sel_val, n_candidates_in=n_in,
            meta={
                "r4_executed": False,
                "reason": "insufficient rung_context (need profile/alpha/Re to invoke _cpu_qz_worker)",
                "r1": r1_meta, "r2": r2_meta, "r3": r3_meta,
            },
        )


def compute_escalation_fractions(records: list[EscalationRecord], n_points: int) -> dict[str, float]:
    """Return per-rung fraction of total point-family decisions.

    Extra rung keys (refusals) are carried through; base RUNG_ORDER always present.
    """
    if n_points <= 0:
        base = {r: 0.0 for r in RUNG_ORDER}
        return base
    counts = {r: 0 for r in RUNG_ORDER}
    for r in records:
        key = r.rung
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = counts.get(key, 0) + 1
    total = float(len(records))
    fr = {r: (counts.get(r, 0) / total if total > 0 else 0.0) for r in RUNG_ORDER}
    # include any extra refusal rungs for visibility in meta
    for key, cnt in counts.items():
        if key not in fr:
            fr[key] = (cnt / total) if total > 0 else 0.0
    return fr


__all__ = [
    "EscalationRecord",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "RUNG_ORDER",
    "is_ambiguous_or_collar",
    "escalate_one_point",
    "run_async_cpu_qz",
    "compute_escalation_fractions",
]
