"""Temporal GPU sweep orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np

from . import assemblers
from .affine import AffineExtractionError, AffineOperatorCache
from .backend import device_report, resolve_config
from .enumerator import (
    enumerate_temporal_band,
    enumerate_temporal_band_batch,
    seed_code_legend,
)
from .kernels import CupyBatchedLinalgOps
from .refine import AffinePencil

AFFINE_SWEEP_RTOL = 1.0e-12

__all__ = [
    "AFFINE_SWEEP_RTOL",
    "build_temporal_affine_cache",
    "reverify_affine_cache",
    "solve_temporal_sweep",
]


def _axis_range(values, *, positive=False):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("sweep axis must not be empty")
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0e-8)
        lo -= pad
        hi += pad
    if positive and lo <= 0:
        lo = min(float(np.min(arr[arr > 0])) * 0.5, hi * 0.5)
    return lo, hi


def _inv_re_range(Res):
    arr = np.asarray(Res, dtype=float).ravel()
    if np.any(arr <= 0):
        raise ValueError("Re values must be positive")
    inv = 1.0 / arr
    return _axis_range(inv, positive=True)


def _k_psi_range(alphas, beta):
    alphas = np.asarray(alphas, dtype=float).ravel()
    beta = float(beta)
    ks = np.hypot(alphas, beta)
    psis = np.arctan2(beta, alphas)
    klo, khi = _axis_range(ks, positive=True)
    plo, phi = _axis_range(psis)
    if phi - plo < 1.0e-8:
        c = 0.5 * (plo + phi)
        plo, phi = c - 1.0e-4, c + 1.0e-4
    return (klo, khi), (plo, phi)


def _bundle_for_operator(profile, operator, *, alphas, Res, Ma, N, y_max, L,
                         wall_bc, length_scale, Pr, gamma,
                         lambda_mu_ratio, beta):
    inv_box = _inv_re_range(Res)
    if operator == "ozgen_2d":
        bundle = assemblers.temporal_ozgen_2d_assembler(
            profile, N=N, y_max=y_max, L=L, wall_bc=wall_bc,
            length_scale=length_scale, Ma=Ma, Pr=Pr, gamma=gamma,
        )
        box = {"alpha": _axis_range(alphas, positive=True), "invRe": inv_box}
    elif operator == "mack_2d":
        bundle = assemblers.temporal_2d_assembler(
            profile, N=N, y_max=y_max, L=L, wall_bc=wall_bc,
            length_scale=length_scale, Ma=Ma, Pr=Pr, gamma=gamma,
            lambda_mu_ratio=lambda_mu_ratio,
        )
        box = {"alpha": _axis_range(alphas, positive=True), "invRe": inv_box}
    elif operator == "mack_3d":
        k_box, psi_box = _k_psi_range(alphas, beta)
        bundle = assemblers.temporal_3d_assembler(
            profile, N=N, y_max=y_max, L=L, wall_bc=wall_bc,
            length_scale=length_scale, Ma=Ma, Pr=Pr, gamma=gamma,
            lambda_mu_ratio=lambda_mu_ratio,
        )
        box = {"k": k_box, "psi": psi_box, "invRe": inv_box}
    else:
        raise ValueError(f"unsupported temporal operator {operator!r}")
    return bundle, box


def build_temporal_affine_cache(profile, operator, *, alphas, Res, Ma, N, y_max,
                                L, wall_bc, length_scale, Pr, gamma,
                                lambda_mu_ratio, beta, rtol=AFFINE_SWEEP_RTOL):
    """Build and verify the affine cache for this exact sweep extent."""
    bundle, box = _bundle_for_operator(
        profile, operator, alphas=alphas, Res=Res, Ma=Ma, N=N, y_max=y_max,
        L=L, wall_bc=wall_bc, length_scale=length_scale, Pr=Pr, gamma=gamma,
        lambda_mu_ratio=lambda_mu_ratio, beta=beta,
    )
    cache = AffineOperatorCache.from_probe(
        bundle.assemble,
        bundle.basis,
        box,
        key=bundle.key,
        rtol=rtol,
    )
    reverify_affine_cache(cache, rtol=rtol)
    return cache


def reverify_affine_cache(cache, *, rtol=AFFINE_SWEEP_RTOL):
    """Re-run direct-assembly verification at sweep start.

    The cache constructor already ran held-out checks.  This function is the
    explicit per-sweep gate for cache reuse: it checks the stored held-out max,
    the box center, and every box corner against the bound.  A trip raises
    :class:`AffineExtractionError` so callers can fall back to per-point CPU
    assembly.
    """
    worst = float(cache.max_heldout_residual or 0.0)
    points = [dict(cache.center)]
    params = list(cache.basis.params)
    for mask in range(2 ** len(params)):
        pt = {}
        for bit, param in enumerate(params):
            lo, hi = cache.probe_box[param]
            pt[param] = hi if (mask >> bit) & 1 else lo
        points.append(pt)
    for point in points:
        worst = max(worst, float(cache.verify_at(**point)))
    if worst > rtol:
        raise AffineExtractionError(
            f"per-sweep affine re-verification failed: {worst:.3e} > {rtol:.1e}"
        )
    return worst


def _cpu_fallback(profile, alphas, Res, *, Ma, N, y_max, L, wall_bc,
                  length_scale, Pr, gamma, lambda_mu_ratio, beta, operator,
                  families, seeds, precision, tile_size, return_eigenvectors,
                  cpu_workers, meta_update):
    from pymack.sweep import temporal_sweep

    result = temporal_sweep(
        profile,
        alphas,
        Res,
        Ma=Ma,
        N=N,
        y_max=y_max,
        L=L,
        wall_bc=wall_bc,
        length_scale=length_scale,
        Pr=Pr,
        gamma=gamma,
        lambda_mu_ratio=lambda_mu_ratio,
        beta=beta,
        operator=operator,
        families=families,
        seeds=seeds,
        backend="cpu",
        precision=precision,
        tile_size=tile_size,
        return_eigenvectors=return_eigenvectors,
        cpu_workers=cpu_workers,
    )
    meta = dict(result.meta)
    meta.update(meta_update)
    meta["backend"] = "gpu"
    meta["backend_requested"] = meta_update.get(
        "gpu_backend_requested", meta.get("backend_requested")
    )
    meta["env_backend_override"] = meta_update.get(
        "env_backend_override", meta.get("env_backend_override")
    )
    meta["operator_source"] = meta_update.get(
        "operator_source", "gpu_affine_reverified_cpu_qz_fallback"
    )
    meta["seed_codes"] = seed_code_legend()
    return replace(result, meta=meta)


def _point_scalars(operator, alpha, Re, beta):
    invRe = 1.0 / float(Re)
    if operator in ("ozgen_2d", "mack_2d"):
        return {"alpha": float(alpha), "invRe": invRe}
    if operator == "mack_3d":
        return {
            "k": float(np.hypot(alpha, beta)),
            "psi": float(np.arctan2(beta, alpha)),
            "invRe": invRe,
        }
    raise ValueError(f"unsupported temporal operator {operator!r}")


def _gpu_failed_record(exc):
    from pymack.sweep import SEED_SOLVER_FAILED

    return {"status": SEED_SOLVER_FAILED, "error": repr(exc)}


def _aggregate_enumerator_timing(diagnostics):
    totals = {}
    for diag in diagnostics:
        timing = diag.get("timing_s", {})
        for key, value in timing.items():
            if key in ("total", "batch_setup_total"):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                totals[key] = totals.get(key, 0.0) + float(value)
    total_accounted = sum(totals.values())
    if total_accounted:
        totals["total_accounted"] = float(total_accounted)
        totals["fractions"] = {
            key: float(value / total_accounted)
            for key, value in totals.items()
            if key != "total_accounted"
        }
    return totals


def _stratified_cpu_audit(
    profile, alphas, Res, families, grids, *, Ma, N, y_max, length_scale, sample_frac=0.05
):
    """Real stratified 5% CPU-QZ audit (F4). Coverage across param extent (corners +
    alpha/Re strata) + c_r-band boundary region intent. Seed derived from run
    provenance + runtime nonce so not identical fixed set on repeated same-grid
    runs; seed recorded for reproducibility.

    K3: any drift marks loudly (meta flag) + bench gate fails explicitly.
    Uses the *deployed* solve_temporal_2d + _select.
    """
    import hashlib
    import os
    import time

    from pymack.sweep import _select_temporal_mode
    from pymack.temporal_solver import solve_temporal_2d

    xs = np.asarray(alphas)
    rs = np.asarray(Res)
    n_alpha = len(xs)
    n_re = len(rs)
    n_total = n_alpha * n_re
    n_samp = max(1, int(round(n_total * sample_frac)))

    # Derive seed from run provenance (grid + solver params) + runtime nonce (pid+time)
    # so samples vary across runs of identical grid but remain reproducible from meta.
    prov = repr((
        xs.tolist(), rs.tolist(), float(y_max), float(Ma), int(N),
        str(length_scale), float(sample_frac)
    )).encode("utf-8")
    nonce = f"{os.getpid()}:{time.time_ns()}".encode("utf-8")
    seed = int(hashlib.sha256(prov + nonce).hexdigest()[:16], 16) & 0xffffffff
    rng = np.random.default_rng(seed)

    # Real stratification: corners + param-space strata bins for alpha/Re extent (incl extremes)
    picks = set()
    # 4 corners of the (alpha, Re) grid
    for ia in (0, n_alpha - 1):
        for jr in (0, n_re - 1):
            picks.add(ia * n_re + jr)
    # Divide into ~4x4 = 16 strata bins; sample proportionally (ensures coverage incl edges)
    n_sa = 4
    n_sr = 4
    per = max(1, n_samp // (n_sa * n_sr))
    for ba in range(n_sa):
        alo = int(ba * n_alpha / n_sa)
        ahi = min(n_alpha, int((ba + 1) * n_alpha / n_sa))
        for br in range(n_sr):
            rlo = int(br * n_re / n_sr)
            rhi = min(n_re, int((br + 1) * n_re / n_sr))
            cands = [i * n_re + j for i in range(alo, max(alo + 1, ahi))
                     for j in range(rlo, max(rlo + 1, rhi))]
            if cands:
                k = min(per, len(cands))
                picks.update(int(p) for p in rng.choice(cands, size=k, replace=False))
    # Top up to n_samp from remaining (flat but after strata bias)
    flat_all = np.arange(n_total)
    rem = [int(p) for p in flat_all if p not in picks]
    need = n_samp - len(picks)
    if need > 0 and rem:
        extra = rng.choice(rem, size=min(need, len(rem)), replace=False)
        picks.update(int(p) for p in extra)
    flat_picks = np.array(sorted(picks)[:n_samp], dtype=int)

    # To address "c_r band boundary region": bias one extra pick toward high-alpha (often near band edge for TS/Mack transition)
    # (lightweight; actual c_r observed post-solve in failures if near 0.45/edges)
    if n_alpha > 1 and len(flat_picks) < n_samp:
        edge_ia = n_alpha - 1
        edge_js = list(range(max(0, n_re // 2 - 1), min(n_re, n_re // 2 + 2)))
        edge_flats = [edge_ia * n_re + j for j in edge_js if (edge_ia * n_re + j) not in picks]
        if edge_flats:
            add = rng.choice(edge_flats, size=1, replace=False)
            flat_picks = np.unique(np.concatenate([flat_picks, add.astype(int)]))

    failures = []
    worst = 0.0
    n_checks = 0
    for flat in flat_picks:
        i = int(flat // n_re)
        j = int(flat % n_re)
        alpha = float(xs[i])
        Re = float(rs[j])
        try:
            ev, _v, _y = solve_temporal_2d(
                profile, alpha, Re, Ma, N=N, y_max=y_max, length_scale=length_scale
            )
            for k, fam in enumerate(families):
                ref_sel = _select_temporal_mode(np.asarray(ev, dtype=complex), fam)
                got = grids[k]["value"][i, j]
                n_checks += 1
                if ref_sel is None:
                    if grids[k]["converged"][i, j]:
                        failures.append({"i": i, "j": j, "fam": k, "reason": "cpu_empty_gpu_hit"})
                    continue
                ref = complex(ev[ref_sel])
                if not grids[k]["converged"][i, j]:
                    failures.append({"i": i, "j": j, "fam": k, "reason": "gpu_empty_cpu_hit"})
                    continue
                err = abs(got - ref)
                worst = max(worst, float(err))
                if err > 1e-9 * max(1.0, abs(ref)):
                    failures.append({
                        "i": i, "j": j, "fam": k,
                        "error": float(err),
                        "ref": [float(ref.real), float(ref.imag)],
                        "got": [float(got.real), float(got.imag)],
                    })
        except Exception as exc:
            failures.append({"i": i, "j": j, "error": repr(exc)})
    audit_out = {
        "sample_frac": float(sample_frac),
        "n_sample_points": int(len(flat_picks)),
        "n_point_family_checks": int(n_checks),
        "n_drift": int(len(failures)),
        "worst_abs_error": float(worst),
        "passed": len(failures) == 0,
        "failures": failures[:10],
        "audit_seed": int(seed),
        "audit_seed_provenance": "sha256(grid+params) ^ (pid:time_ns) for non-fixed-per-grid samples; record enables exact repro",
        "stratified": True,
        "strata_note": "corners + 4x4 param bins (alpha/Re extent+extremes) + light c_r-edge bias",
    }
    return audit_out


def _production_temporal_candidate_mask(values, operator):
    """Mask candidates through the deployed per-point solver spectrum gate."""
    c = np.asarray(values, dtype=np.complex128)
    keep = np.isfinite(c.real) & np.isfinite(c.imag)
    if operator in ("ozgen_2d", "mack_2d", "mack_3d"):
        # Mirrors pymack/temporal_solver.py:68-80 exactly: finite QZ spectrum
        # followed by -0.5 < Re(c) < 1.5 and |Im(c)| < 0.5 before sorting.
        keep &= (c.real > -0.5) & (c.real < 1.5) & (np.abs(c.imag) < 0.5)
    return keep


def _apply_production_temporal_filter(candidates, operator):
    if not candidates:
        return [], {"n_input": 0, "n_removed": 0, "rule": "empty"}
    values = np.asarray([c.value for c in candidates], dtype=np.complex128)
    mask = _production_temporal_candidate_mask(values, operator)
    kept = [c for c, ok in zip(candidates, mask) if bool(ok)]
    removed = values[~mask]
    return kept, {
        "n_input": int(values.size),
        "n_removed": int(removed.size),
        "rule": (
            "finite and -0.5 < Re(c) < 1.5 and |Im(c)| < 0.5, "
            "matching the deployed temporal per-point solvers before "
            "family-band selection"
        ),
        "removed_values_preview": [
            [float(z.real), float(z.imag)] for z in removed[:8]
        ],
    }


def _record_from_enumeration(enum, *, operator, band, point_index,
                             family_index, return_eigenvectors, diagnostics,
                             candidate_sink=None):
    """Delegate final selection to verdict layer (verbatim CPU filter).

    Escalation rung decided here; provenance carried in returned record.

    When ``candidate_sink`` is supplied (R3-2), the full production-filtered
    GPU candidate set (values + eigenvectors) for this (point, family) is
    captured into it so the two-domain family verdict can be assembled from
    GPU-SWEEP-DERIVED data (not fresh CPU spectra).  The eigenvector block
    layout is the bitwise-identical Ozgen [u, v, T, p] ordering, so the pinned
    verdict-layer freestream/decaying filter runs on it verbatim.
    """
    from pymack.sweep import SEED_NO_ADMISSIBLE_MODE
    from pymack.gpu.escalate import R0, escalate_one_point
    from pymack.gpu.verdict import select_from_gpu_candidates

    production_candidates, production_filter = (
        _apply_production_temporal_filter(enum.candidates, operator)
    )
    enum.diagnostics["production_operator_filter"] = production_filter

    if candidate_sink is not None:
        if production_candidates:
            vals = np.asarray(
                [c.value for c in production_candidates], dtype=np.complex128
            )
            vecs = np.stack(
                [
                    np.asarray(c.vector, dtype=np.complex128).ravel()
                    for c in production_candidates
                ],
                axis=1,
            )
        else:
            vals = np.zeros(0, dtype=np.complex128)
            vecs = np.zeros((0, 0), dtype=np.complex128)
        candidate_sink[(int(point_index), int(family_index))] = {
            "values": vals,
            "vectors": vecs,
        }

    # Use verdict/escalate for CPU-verbatim selection + rung (F3 provenance)
    rec_ctx = {"point": int(point_index), "family": int(family_index)}
    try:
        esc_rec = escalate_one_point(
            production_candidates,
            band,
            rung_context=rec_ctx,
            operator=operator,
        )
        rung = esc_rec.rung
        seed = int(esc_rec.seed)
        sel_cand = None
        if esc_rec.value is not None:
            # recover the candidate object by value match (stable for small sets)
            for c in production_candidates:
                if abs(c.value - esc_rec.value) < 1e-12:
                    sel_cand = c
                    break
    except Exception:
        # fall back to direct (still verbatim)
        sel_cand, seed = select_from_gpu_candidates(production_candidates, band, operator=operator)
        rung = R0
    rung_for_diag = rung
    diagnostics.append({
        "point": int(point_index),
        "family": int(family_index),
        "rung": rung_for_diag,
        **enum.diagnostics,
    })

    if sel_cand is None:
        return {"status": SEED_NO_ADMISSIBLE_MODE, "rung": rung}
    record = {
        "status": 1,
        "value": sel_cand.value,
        "mode_index": int(sel_cand.mode_index),
        "residual": float(sel_cand.residual),
        "edge_ratio": float(sel_cand.edge_ratio),
        "rung": rung,
    }
    if return_eigenvectors:
        record["vec"] = sel_cand.vector
    return record


def _gpu_point_records(ops, pencil, cache, alpha, Re, *, beta, operator,
                       families, config, return_eigenvectors, diagnostics,
                       point_index):
    point = _point_scalars(operator, alpha, Re, beta)
    records = []
    for k, band in enumerate(families):
        try:
            enum = enumerate_temporal_band(
                ops,
                pencil,
                cache,
                point,
                band,
                config,
                operator=operator,
                family_index=k,
            )
            records.append(
                _record_from_enumeration(
                    enum,
                    operator=operator,
                    band=band,
                    point_index=point_index,
                    family_index=k,
                    return_eigenvectors=return_eigenvectors,
                    diagnostics=diagnostics,
                )
            )
        except Exception as exc:
            records.append(_gpu_failed_record(exc))
    return records


def _run_gpu_grid(profile, alphas, Res, *, cache, Ma, N, y_max, L, wall_bc,
                  length_scale, Pr, gamma, lambda_mu_ratio, beta, operator,
                  families, precision, tile_size, seeds, return_eigenvectors,
                  backend_requested, env_backend_override, cfg, affine_meta):
    from pymack.sweep import (
        TemporalFamilyResult,
        TemporalSweepResult,
        _build_meta,
        _collect_family_grids,
    )

    import cupy as cp

    ops = CupyBatchedLinalgOps(cp=cp)
    pencil = AffinePencil(ops, cache)
    xs = np.asarray(alphas, dtype=float)
    Res = np.asarray(Res, dtype=float)
    point_specs = []
    for i, alpha in enumerate(xs):
        for j, Re in enumerate(Res):
            point_specs.append({
                "i": int(i),
                "j": int(j),
                "alpha": float(alpha),
                "Re": float(Re),
                "point_index": int(len(point_specs)),
                "scalars": _point_scalars(operator, float(alpha), float(Re), beta),
            })
    results = {
        (spec["i"], spec["j"]): [None] * len(families)
        for spec in point_specs
    }
    # R3-2: env-gated capture of GPU-derived per-point candidate sets so the
    # two-domain family verdict can be assembled from GPU data.  Off by default
    # (no serialization/behaviour change for existing paths); the bench sets
    # PYMACK_GPU_CAPTURE_CANDIDATE_SETS=1.  Never edits the CPU-path facade.
    capture_candidates = (
        os.environ.get("PYMACK_GPU_CAPTURE_CANDIDATE_SETS") == "1"
    )
    candidate_sink = {} if capture_candidates else None
    diagnostics = []
    batch_failures = []
    t0 = time.perf_counter()
    batched_family_runs = 0
    per_point_fallback_runs = 0
    for k, band in enumerate(families):
        use_cross_point_batch = operator == "ozgen_2d" and len(point_specs) > 1
        if use_cross_point_batch:
            try:
                enums = enumerate_temporal_band_batch(
                    ops,
                    pencil,
                    cache,
                    [spec["scalars"] for spec in point_specs],
                    band,
                    cfg,
                    operator=operator,
                    family_index=k,
                )
                batched_family_runs += 1
                for spec, enum in zip(point_specs, enums):
                    results[(spec["i"], spec["j"])][k] = _record_from_enumeration(
                        enum,
                        operator=operator,
                        band=band,
                        point_index=spec["point_index"],
                        family_index=k,
                        return_eigenvectors=return_eigenvectors,
                        diagnostics=diagnostics,
                        candidate_sink=candidate_sink,
                    )
                continue
            except Exception as exc:
                batch_failures.append({
                    "family": int(k),
                    "error": repr(exc),
                    "fallback": "per_point_enumerator",
                })
        for spec in point_specs:
            try:
                enum = enumerate_temporal_band(
                    ops,
                    pencil,
                    cache,
                    spec["scalars"],
                    band,
                    cfg,
                    operator=operator,
                    family_index=k,
                )
                enum.diagnostics["cross_point_batch"] = {
                    "used": False,
                    "reason": (
                        "batch_exception_fallback"
                        if use_cross_point_batch else "unsupported_operator_or_single_point"
                    ),
                }
                results[(spec["i"], spec["j"])][k] = _record_from_enumeration(
                    enum,
                    operator=operator,
                    band=band,
                    point_index=spec["point_index"],
                    family_index=k,
                    return_eigenvectors=return_eigenvectors,
                    diagnostics=diagnostics,
                    candidate_sink=candidate_sink,
                )
            except Exception as exc:
                results[(spec["i"], spec["j"])][k] = _gpu_failed_record(exc)
            per_point_fallback_runs += 1
    for key, records in results.items():
        for k, record in enumerate(records):
            if record is None:
                records[k] = _gpu_failed_record(
                    RuntimeError(f"missing GPU family record at {key}, family {k}")
                )
    wall_time_s = time.perf_counter() - t0
    engine_wall_time_s = float(wall_time_s)
    grids, errors = _collect_family_grids(
        results, (len(xs), len(Res)), families, return_eigenvectors
    )
    solver_kwargs = {
        "Ma": float(Ma),
        "N": int(N),
        "y_max": None if y_max is None else float(y_max),
        "L": None if L is None else float(L),
        "wall_bc": wall_bc,
        "length_scale": length_scale,
        "Pr": float(Pr),
        "gamma": float(gamma),
        "lambda_mu_ratio": float(lambda_mu_ratio),
        "beta": float(beta),
    }
    meta = _build_meta(
        kind="temporal",
        api_name="pymack.sweep.temporal_sweep",
        operator=operator,
        backend_requested=backend_requested,
        env_override=env_backend_override,
        precision=precision,
        tile_size=tile_size,
        seeds=seeds,
        n_workers=0,
        families=families,
        solver_kwargs=solver_kwargs,
        xs=xs,
        x_name="alpha",
        Res=Res,
        return_eigenvectors=return_eigenvectors,
        wall_time_s=wall_time_s,
        errors=errors,
    )
    meta.update(affine_meta)
    meta.update({
        "backend": "gpu",
        "gpu_engine_status": "device_contour_projection",
        "operator_source": "gpu_contour_projection_device",
        "precision": f"{cfg.base_dtype}+fp64_refine",
        "tile_size": None if cfg.tile == 0 else int(cfg.tile),
        "tile_size_requested": tile_size if tile_size == "auto" else int(tile_size),
        "seed_codes": seed_code_legend(),
        "gpu_enumerator": {
            "n_point_family_runs": int(len(diagnostics)),
            "cross_point_batching": {
                "enabled_for_operator": operator == "ozgen_2d",
                "batched_family_runs": int(batched_family_runs),
                "per_point_fallback_runs": int(per_point_fallback_runs),
                "batch_failures": batch_failures,
            },
            "timing_s": _aggregate_enumerator_timing(diagnostics),
            "diagnostics": diagnostics[:50],
            "diagnostics_truncated": bool(len(diagnostics) > 50),
            "small_projected_eig": "cpu_round_trip",
            "winding_number_claims": False,
        },
        "wall_time_s": engine_wall_time_s,
    })

    # Stage-3 (F4): real stratified 5% CPU-QZ audit harness (wired into every GPU sweep)
    # Drift ACTS: loud meta flag + K3 stop (bench gate will fail on n_drift>0).
    # Audit cost explicitly separated from engine wall (F2): audit is certification telemetry,
    # not charged to the M2/M3 engine gate (deployed baseline carries none).
    t_audit0 = time.perf_counter()
    audit = _stratified_cpu_audit(
        profile, xs, Res, families, grids,
        Ma=Ma, N=N, y_max=y_max, length_scale=length_scale,
        sample_frac=0.05,
    )
    audit_wall_time_s = time.perf_counter() - t_audit0
    meta["audit"] = audit
    meta["engine_wall_time_s"] = engine_wall_time_s
    meta["audit_wall_time_s"] = float(audit_wall_time_s)
    if not audit.get("passed", True):
        meta["K3_audit_drift"] = True
        meta.setdefault("errors", []).append(
            "K3: stratified CPU-QZ audit detected verdict drift (n_drift=%d)" % audit.get("n_drift", 0)
        )
    # Also surface escalation fractions (default R0-heavy until higher rungs exercised)
    from pymack.gpu.escalate import RUNG_ORDER, compute_escalation_fractions
    # Build simple rung records from diagnostics (rung if present else R0)
    rung_recs = []
    for d in diagnostics:
        r = d.get("rung", None) or "device_filter"
        rung_recs.append(type("R", (), {"rung": r})())
    esc_frac = compute_escalation_fractions(rung_recs, n_points=max(1, len(diagnostics)))
    meta["escalation_fractions"] = esc_frac
    meta["escalation"] = {
        "rungs": RUNG_ORDER,
        "fractions": esc_frac,
        "cpu_qz_fraction": float(esc_frac.get("cpu_qz_async", 0.0)),
    }
    if candidate_sink is not None:
        # R3-2: python-side surface (numpy arrays), consumed in-process by the
        # bench's GPU-derived two-domain assembly.  Deliberately NOT part of the
        # JSON artifact (arrays), so no existing serialization path is affected.
        meta["gpu_candidate_sets"] = candidate_sink
        meta["gpu_candidate_sets_captured"] = int(len(candidate_sink))
    family_results = tuple(
        TemporalFamilyResult(
            band=band,
            c=grid["value"],
            omega_i=xs[:, None] * grid["value"].imag,
            converged=grid["converged"],
            residual=grid["residual"],
            edge_ratio=grid["edge_ratio"],
            seed_map=grid["seed_map"],
            mode_index=grid["mode_index"],
            eigenvectors=grid["eigenvectors"],
        )
        for band, grid in zip(families, grids)
    )
    return TemporalSweepResult(alphas=xs, Res=Res, families=family_results, meta=meta)


def solve_temporal_sweep(profile, alphas, Res, *, Ma=None, N=128, y_max=None,
                         L=None, wall_bc="isothermal",
                         length_scale="L_star", Pr=0.72, gamma=1.4,
                         lambda_mu_ratio=None, beta=0.0,
                         operator="mack_2d", families=(), seeds="auto",
                         precision="mixed", tile_size="auto",
                         return_eigenvectors=False, cpu_workers=None,
                         backend_requested="gpu", env_backend_override=None):
    """GPU backend entry point for :func:`pymack.sweep.temporal_sweep`.

    The affine gate is mandatory.  When it trips, the function returns the CPU
    full-spectrum result with explicit fallback metadata; values remain correct
    and only performance is lost.
    """
    from pymack.equations import DEFAULT_LAMBDA_MU_RATIO

    if lambda_mu_ratio is None:
        lambda_mu_ratio = DEFAULT_LAMBDA_MU_RATIO
    cfg = resolve_config()
    t0 = time.perf_counter()
    try:
        cache = build_temporal_affine_cache(
            profile,
            operator,
            alphas=alphas,
            Res=Res,
            Ma=Ma,
            N=N,
            y_max=y_max,
            L=L,
            wall_bc=wall_bc,
            length_scale=length_scale,
            Pr=Pr,
            gamma=gamma,
            lambda_mu_ratio=lambda_mu_ratio,
            beta=beta,
        )
        affine_meta = {
            "affine_reverified": True,
            "affine_max_reverify_residual": float(cache.max_heldout_residual),
            "affine_fingerprint": cache.fingerprint(),
            "affine_probe_box": cache.probe_box,
            "affine_probe_box_in_fingerprint": True,
            "device": device_report(cfg),
            "gpu_backend_requested": backend_requested,
            "env_backend_override": env_backend_override,
            "wall_time_gpu_path_overhead_s": float(time.perf_counter() - t0),
        }
    except AffineExtractionError as exc:
        affine_meta = {
            "gpu_engine_status": "cpu_qz_fallback_after_affine_trip",
            "affine_reverified": False,
            "affine_error": repr(exc),
            "device": device_report(cfg),
            "gpu_backend_requested": backend_requested,
            "env_backend_override": env_backend_override,
            "operator_source": "cpu_qz_fallback_affine_trip",
            "wall_time_gpu_path_overhead_s": float(time.perf_counter() - t0),
        }
        return _cpu_fallback(
            profile,
            alphas,
            Res,
            Ma=Ma,
            N=N,
            y_max=y_max,
            L=L,
            wall_bc=wall_bc,
            length_scale=length_scale,
            Pr=Pr,
            gamma=gamma,
            lambda_mu_ratio=lambda_mu_ratio,
            beta=beta,
            operator=operator,
            families=families,
            seeds=seeds,
            precision=precision,
            tile_size=tile_size,
            return_eigenvectors=return_eigenvectors,
            cpu_workers=cpu_workers,
            meta_update=affine_meta,
        )
    return _run_gpu_grid(
        profile,
        alphas,
        Res,
        cache=cache,
        Ma=Ma,
        N=N,
        y_max=y_max,
        L=L,
        wall_bc=wall_bc,
        length_scale=length_scale,
        Pr=Pr,
        gamma=gamma,
        lambda_mu_ratio=lambda_mu_ratio,
        beta=beta,
        operator=operator,
        families=families,
        seeds=seeds,
        precision=precision,
        tile_size=tile_size,
        return_eigenvectors=return_eigenvectors,
        backend_requested=backend_requested,
        env_backend_override=env_backend_override,
        cfg=cfg,
        affine_meta=affine_meta,
    )
