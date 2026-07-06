"""Contour-enumerator configuration and prefilters for the GPU temporal path.

The production verdict remains owned by :mod:`pymack.sweep` / the existing CPU
filter stack.  This module only describes the official contour variants and
implements device-safe candidate prefilters that can remove certainly invalid
candidates before CPU verdict assembly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .batch import TileScheduler
from .polish import batched_two_sided_rqi
from .refine import solve_pencil

__all__ = [
    "Candidate",
    "ContourVariant",
    "TemporalEnumeration",
    "OFFICIAL_CONTOUR_VARIANTS",
    "rectangle_gauss_legendre_nodes",
    "candidate_prefilter",
    "enumerate_temporal_band",
    "enumerate_temporal_band_batch",
    "seed_code_legend",
]

_PROJECTION_WORKER_CAP = 8


@dataclass(frozen=True)
class ContourVariant:
    """Pinned contour projection variant for a production family."""

    family: str
    variant: str
    L: int
    K: int
    geometry: str
    note: str


@dataclass
class Candidate:
    """One polished, certified temporal candidate from the GPU enumerator."""

    value: complex
    vector: np.ndarray
    residual: float
    edge_ratio: float
    bv_norm: float
    mode_index: int
    source: str = "gpu_contour"


@dataclass
class TemporalEnumeration:
    """Candidates plus diagnostic metadata for one point/family band."""

    candidates: list[Candidate]
    diagnostics: dict


OFFICIAL_CONTOUR_VARIANTS = {
    "ozgen_first_pair": ContourVariant(
        family="ozgen_first_pair",
        variant="hankel",
        L=48,
        K=3,
        geometry="rectangle",
        note="higher-K Hankel official D1 variant",
    ),
    "ozgen_second_pair": ContourVariant(
        family="ozgen_second_pair",
        variant="hankel",
        L=48,
        K=3,
        geometry="rectangle",
        note="higher-K Hankel official D1 variant",
    ),
    "mack_fig10_4_m10_3d": ContourVariant(
        family="mack_fig10_4_m10_3d",
        variant="single",
        L=112,
        K=1,
        geometry="rectangle",
        note="single-box official D1 variant",
    ),
    "mach6_spatial_dense": ContourVariant(
        family="mach6_spatial_dense",
        variant="single",
        L=56,
        K=1,
        geometry="rectangle",
        note="single-box official D1 variant",
    ),
    "mazhong_first_spatial": ContourVariant(
        family="mazhong_first_spatial",
        variant="window",
        L=112,
        K=1,
        geometry="target_centered_window",
        note="target-centred production-window official D1 variant",
    ),
    "mazhong_second_spatial": ContourVariant(
        family="mazhong_second_spatial",
        variant="window",
        L=112,
        K=1,
        geometry="target_centered_window",
        note="target-centred production-window official D1 variant",
    ),
}


def rectangle_gauss_legendre_nodes(rect, n_per_edge):
    """Counter-clockwise Gauss-Legendre nodes on each rectangle edge.

    The returned weights already include ``dz / (2 pi i)`` for contour-moment
    accumulation.
    """
    (x0, x1), (y0, y1) = rect["real"], rect["imag"]
    xi, wq = np.polynomial.legendre.leggauss(int(n_per_edge))
    corners = [
        complex(x0, y0),
        complex(x1, y0),
        complex(x1, y1),
        complex(x0, y1),
    ]
    zs, ws = [], []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        zs.append(mid + half * xi)
        ws.append(wq * half / (2.0j * np.pi))
    return np.concatenate(zs), np.concatenate(ws)


def _finite_rect_for_band(band, *, guard=0.02):
    lo = float(band.cr_min)
    hi = float(band.cr_max)
    if not np.isfinite(lo):
        lo = min(-0.25, hi - 1.0)
    if not np.isfinite(hi):
        hi = max(1.5, lo + 1.0)
    if not hi > lo:
        raise ValueError(f"invalid temporal c_r band: {(band.cr_min, band.cr_max)}")
    width = hi - lo
    lo -= guard * max(width, 1.0e-3)
    hi += guard * max(width, 1.0e-3)
    ci = float(getattr(band, "ci_abs_max", np.inf))
    if not np.isfinite(ci):
        ci = 0.5
    ci = max(ci, 1.0e-3)
    ci *= 1.0 + guard
    return {"real": (lo, hi), "imag": (-ci, ci)}


def _operator_variant(operator, n):
    if operator == "ozgen_2d":
        return {"L": min(48, max(4, n - 1)), "K": 3, "nq_edge": 12,
                "variant": "higher_k_hankel_rectangle_cpu_projected"}
    if operator == "mack_3d":
        return {"L": min(112, max(4, n - 1)), "K": 1, "nq_edge": 12,
                "variant": "single_box_rectangle_cpu_projected"}
    return {"L": min(96, max(4, n - 1)), "K": 1, "nq_edge": 12,
            "variant": "single_box_rectangle_cpu_projected"}


def _stable_seed(point, band, family_index):
    meta = getattr(band, "as_meta", lambda: repr(band))()
    payload = {
        "point": {k: float(v) for k, v in sorted(point.items())},
        "band": meta,
        "family_index": int(family_index),
    }
    blob = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") % (2**32)


def _normalized_probe_block(n, L, seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L))
    norm = np.linalg.norm(V, axis=0)
    norm[norm == 0.0] = 1.0
    return (V / norm[None, :]).astype(np.complex128, copy=False)


def _project_from_moments(moments, *, K, center, scale, extract_tol=1.0e-8,
                          strong_tol=1.0e-3, saturation_slack=2):
    m = moments[0].shape[0]
    bh0 = np.vstack([
        np.hstack([moments[i + j] for j in range(K)])
        for i in range(K)
    ])
    bh1 = np.vstack([
        np.hstack([moments[i + j + 1] for j in range(K)])
        for i in range(K)
    ])
    U, s, Vh = np.linalg.svd(bh0, full_matrices=False)
    s0 = s[0] if s.size and s[0] > 0 else 1.0
    rel = s / s0
    cap = min(bh0.shape)
    rank = int(min(np.count_nonzero(rel > extract_tol), cap - 1))
    strong_rank = int(np.count_nonzero(rel > strong_tol))
    saturated = bool(strong_rank >= cap - saturation_slack)
    gap = float(rel[rank - 1] / rel[rank]) if 0 < rank < s.size else math.inf
    diag = {
        "rank": rank,
        "strong_rank": strong_rank,
        "rank_gap": None if not math.isfinite(gap) else gap,
        "rank_saturated": saturated,
        "capacity": int(cap),
        "sv_tail_rel": [float(x) for x in rel[-3:]],
    }
    if rank == 0:
        return np.zeros(0, dtype=np.complex128), np.zeros((m, 0), dtype=np.complex128), diag
    Uk = U[:, :rank]
    Vk = Vh[:rank, :].conj().T
    Mk = Uk.conj().T @ bh1 @ Vk / s[:rank]
    mu, small_vecs = np.linalg.eig(Mk)
    vals = center + scale * mu
    vecs = Uk[:m, :] @ small_vecs
    nv = np.linalg.norm(vecs, axis=0)
    nv[nv == 0.0] = 1.0
    return vals, vecs / nv[None, :], diag


def _projection_worker_count(n_items):
    if n_items <= 1:
        return 1
    env = os.environ.get("PYMACK_GPU_PROJECTION_WORKERS")
    if env:
        try:
            requested = int(env)
        except ValueError:
            requested = _PROJECTION_WORKER_CAP
        if requested <= 0:
            return 1
        return max(1, min(int(n_items), requested))
    return max(1, min(int(n_items), os.cpu_count() or 1, _PROJECTION_WORKER_CAP))


def _project_many_from_moments(moments_by_point, *, K, center, scale):
    n_items = len(moments_by_point)
    workers = _projection_worker_count(n_items)

    def run(idx):
        t0 = time.perf_counter()
        values, vectors, diag = _project_from_moments(
            moments_by_point[idx], K=K, center=center, scale=scale
        )
        return idx, values, vectors, diag, time.perf_counter() - t0

    t_wall = time.perf_counter()
    if workers == 1:
        results = [run(idx) for idx in range(n_items)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, range(n_items)))
    wall_s = time.perf_counter() - t_wall
    results.sort(key=lambda item: item[0])
    return results, {
        "strategy": "thread_pool" if workers > 1 else "serial",
        "workers": int(workers),
        "n_items": int(n_items),
        "wall_s": float(wall_s),
        "sum_worker_s": float(sum(item[4] for item in results)),
    }


def _inside_rect(z, rect, pad=0.0):
    return (
        rect["real"][0] - pad <= z.real <= rect["real"][1] + pad
        and rect["imag"][0] - pad <= z.imag <= rect["imag"][1] + pad
    )


def _split_rect(rect):
    real = tuple(map(float, rect["real"]))
    imag = tuple(map(float, rect["imag"]))
    if (real[1] - real[0]) >= (imag[1] - imag[0]):
        mid = 0.5 * (real[0] + real[1])
        return [
            {"real": (real[0], mid), "imag": imag},
            {"real": (mid, real[1]), "imag": imag},
        ]
    mid = 0.5 * (imag[0] + imag[1])
    return [
        {"real": real, "imag": (imag[0], mid)},
        {"real": real, "imag": (mid, imag[1])},
    ]


def _rect_center_scale(rect):
    center = complex(
        0.5 * (rect["real"][0] + rect["real"][1]),
        0.5 * (rect["imag"][0] + rect["imag"][1]),
    )
    scale = max(
        0.5 * abs(complex(rect["real"][1] - rect["real"][0],
                          rect["imag"][1] - rect["imag"][0])),
        1.0e-12,
    )
    return center, scale


def _raw_filter_pad(rect, fraction=0.02):
    real_width = float(rect["real"][1] - rect["real"][0])
    imag_width = float(rect["imag"][1] - rect["imag"][0])
    return float(fraction) * max(real_width, imag_width, 1.0e-12)


def _dedupe_polished(items, *, tol=1.0e-8):
    keep = []
    for item in sorted(items, key=lambda r: (r.value.real, r.value.imag, r.residual)):
        if all(abs(item.value - prev.value) > tol * max(1.0, abs(prev.value)) for prev in keep):
            keep.append(item)
    return keep


def _solve_diag_for_indices(solve, cp, idx):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return {
            "promoted": 0,
            "precision_counts": {"complex64": 0, "complex128": 0},
            "trigger_counts": {"stall": 0, "pivot_growth": 0, "getrf_info": 0},
            "max_best_residual": 0.0,
            "max_exit_residual": 0.0,
            "max_growth": 0.0,
        }
    idx_d = cp.asarray(idx)
    return {
        "promoted": int(np.count_nonzero(np.asarray(solve.promoted)[idx])),
        "precision_counts": {
            "complex64": int(np.count_nonzero(np.asarray(solve.precision)[idx] == "complex64")),
            "complex128": int(np.count_nonzero(np.asarray(solve.precision)[idx] == "complex128")),
        },
        "trigger_counts": {
            "stall": int(np.count_nonzero(np.asarray(solve.trigger)[idx] == "stall")),
            "pivot_growth": int(np.count_nonzero(np.asarray(solve.trigger)[idx] == "pivot_growth")),
            "getrf_info": int(np.count_nonzero(np.asarray(solve.trigger)[idx] == "getrf_info")),
        },
        "max_best_residual": float(cp.asnumpy(cp.max(solve.best[idx_d]))),
        "max_exit_residual": float(cp.asnumpy(cp.max(solve.rel[idx_d]))),
        "max_growth": float(cp.asnumpy(cp.max(solve.growth[idx_d]))),
    }


def _solve_diag(solve, cp):
    return _solve_diag_for_indices(solve, cp, np.arange(len(solve.precision)))


def _vram_snapshot(cp):
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def _merge_solve_diags(diags):
    merged = {
        "promoted": 0,
        "precision_counts": {"complex64": 0, "complex128": 0},
        "trigger_counts": {"stall": 0, "pivot_growth": 0, "getrf_info": 0},
        "max_best_residual": 0.0,
        "max_exit_residual": 0.0,
        "max_growth": 0.0,
    }
    for diag in diags:
        merged["promoted"] += int(diag["promoted"])
        for key in merged["precision_counts"]:
            merged["precision_counts"][key] += int(diag["precision_counts"].get(key, 0))
        for key in merged["trigger_counts"]:
            merged["trigger_counts"][key] += int(diag["trigger_counts"].get(key, 0))
        merged["max_best_residual"] = max(
            merged["max_best_residual"], float(diag["max_best_residual"])
        )
        merged["max_exit_residual"] = max(
            merged["max_exit_residual"], float(diag["max_exit_residual"])
        )
        merged["max_growth"] = max(
            merged["max_growth"], float(diag["max_growth"])
        )
    return merged


def _rank_summary(rect_diags):
    if not rect_diags:
        return {
            "rank": 0,
            "strong_rank": 0,
            "rank_gap": None,
            "rank_saturated": False,
            "rank_clipped": False,
            "capacity": 0,
            "sv_tail_rel": [],
            "rects": [],
            "adaptive_resplits": 0,
        }
    final = [d for d in rect_diags if not d.get("resplit")]
    basis = final or rect_diags
    return {
        "rank": int(max(d["rank"] for d in basis)),
        "strong_rank": int(max(d["strong_rank"] for d in basis)),
        "rank_gap": min(
            (d["rank_gap"] for d in basis if d["rank_gap"] is not None),
            default=None,
        ),
        "rank_saturated": bool(any(d["rank_saturated"] for d in rect_diags)),
        "rank_clipped": bool(any(d.get("rank_clipped", False) for d in rect_diags)),
        "capacity": int(min(d["capacity"] for d in basis)),
        "sv_tail_rel": basis[-1].get("sv_tail_rel", []),
        "rects": rect_diags,
        "adaptive_resplits": int(sum(1 for d in rect_diags if d.get("resplit"))),
    }


def _enumerate_rect_raw(ops, pencil, point, rhs_phys, rect, variant, K, config):
    cp = ops.cp
    nodes, weights = rectangle_gauss_legendre_nodes(rect, variant["nq_edge"])
    center, scale = _rect_center_scale(rect)
    rhs = np.repeat(rhs_phys[None, :, :], len(nodes), axis=0)
    node_points = {
        p: np.full(len(nodes), float(point[p]), dtype=float)
        for p in pencil.basis.params
    }
    rhs_eq = pencil.scale_rhs(cp.asarray(rhs))
    solve = solve_pencil(ops, pencil, node_points, nodes, rhs_eq, config)
    Y = pencil.unscale_solution(solve.x).astype(cp.complex128, copy=False)
    phi = cp.asarray((nodes - center) / scale, dtype=cp.complex128)
    w = cp.asarray(weights, dtype=cp.complex128)
    moments = [
        cp.asnumpy(cp.tensordot(w * (phi ** p), Y, axes=(0, 0)))
        for p in range(2 * K)
    ]
    values, vectors, rank_diag = _project_from_moments(
        moments, K=K, center=center, scale=scale
    )
    rank_diag = dict(rank_diag)
    rank_diag["rank_clipped"] = bool(rank_diag["rank"] >= rank_diag["capacity"] - 1)
    return values, vectors, rank_diag, _solve_diag(solve, cp), int(len(nodes))


def _polish_and_filter_candidates(ops, pencil, point, A, B, values, vectors,
                                  config, band, operator, residual_max,
                                  bv_floor):
    n = A.shape[0]
    polished = batched_two_sided_rqi(
        ops,
        pencil,
        point,
        A,
        B,
        values,
        vectors,
        config,
        max_iter=2,
        tol=1.0e-11,
        n_physical=(n // 5 if operator == "mack_3d" else n // 4),
    )
    mask = candidate_prefilter(
        [p.value for p in polished],
        cr_band=(band.cr_min, band.cr_max),
        ci_abs_max=band.ci_abs_max,
        bv_norm=[p.bv_norm for p in polished],
        bv_floor=bv_floor,
        residual=[p.residual for p in polished],
        residual_max=residual_max,
    )
    kept = [
        Candidate(
            value=p.value,
            vector=p.vector,
            residual=p.residual,
            edge_ratio=p.edge_ratio,
            bv_norm=p.bv_norm,
            mode_index=i,
        )
        for i, (p, ok) in enumerate(zip(polished, mask))
        if bool(ok) and bool(p.converged)
    ]
    return _dedupe_polished(kept), polished


def enumerate_temporal_band(ops, pencil, cache, point, band, config, *,
                            operator="ozgen_2d", family_index=0,
                            residual_max=1.0e-8, bv_floor=1.0e-10):
    """Enumerate and polish candidates for one temporal point and c-band.

    The expensive shifted solves are the real slice-07 device path:
    ``A - z_q B`` is contraction-assembled by :class:`AffinePencil`, factored
    by batched cuBLAS LU, and refined with FP64 residuals.  Only the small
    projected eig/SVD is round-tripped to the CPU.
    """
    cp = ops.cp
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    matrices = cache.evaluate(**point)
    A = np.asarray(matrices["A"], dtype=np.complex128)
    B = np.asarray(matrices["B"], dtype=np.complex128)
    n = A.shape[0]
    variant = _operator_variant(operator, n)
    L = int(variant["L"])
    K = int(variant["K"])
    rect = _finite_rect_for_band(band)
    seed = _stable_seed(point, band, family_index)
    V = _normalized_probe_block(n, L, seed)
    rhs_phys = B @ V
    assembly_rhs_s = time.perf_counter() - t0
    max_resplits = 2
    queue = [rect]
    n_resplit = 0
    raw_values = []
    raw_vectors = []
    rect_diags = []
    solve_diags = []
    n_nodes_total = 0

    while queue:
        active = queue.pop(0)
        t0 = time.perf_counter()
        values_i, vectors_i, rank_diag, solve_diag, n_nodes = _enumerate_rect_raw(
            ops, pencil, point, rhs_phys, active, variant, K, config
        )
        enumerate_rect_s = time.perf_counter() - t0
        solve_diags.append(solve_diag)
        n_nodes_total += n_nodes
        rank_clipped = bool(rank_diag["rank_clipped"])
        if (rank_diag["rank_saturated"] or rank_clipped) and n_resplit < max_resplits:
            n_resplit += 1
            rect_diags.append({
                **rank_diag,
                "rect": active,
                "resplit": True,
                "timing_s": {"enumerate_rect": float(enumerate_rect_s)},
                "resplit_reason": (
                    "rank_saturated" if rank_diag["rank_saturated"]
                    else "rank_clipped_at_capacity_minus_one"
                ),
                "n_raw_projected_before_filter": int(values_i.size),
            })
            queue.extend(_split_rect(active))
            continue

        raw_pad = _raw_filter_pad(active)
        if values_i.size:
            inside = np.array(
                [_inside_rect(z, active, pad=raw_pad) for z in values_i],
                dtype=bool,
            )
            raw_values.extend(complex(z) for z in values_i[inside])
            raw_vectors.append(vectors_i[:, inside])
            n_inside = int(np.count_nonzero(inside))
        else:
            n_inside = 0
        rect_diags.append({
            **rank_diag,
            "rect": active,
            "resplit": False,
            "timing_s": {"enumerate_rect": float(enumerate_rect_s)},
            "raw_filter_pad": raw_pad,
            "n_raw_projected_before_filter": int(values_i.size),
            "n_raw_projected_in_collar": n_inside,
        })

    if raw_values:
        values = np.asarray(raw_values, dtype=np.complex128)
        vectors = np.concatenate(raw_vectors, axis=1)
    else:
        values = np.zeros(0, dtype=np.complex128)
        vectors = np.zeros((n, 0), dtype=np.complex128)
    t0 = time.perf_counter()
    kept, polished = _polish_and_filter_candidates(
        ops,
        pencil,
        point,
        A,
        B,
        values,
        vectors,
        config,
        band,
        operator,
        residual_max,
        bv_floor,
    )
    polish_s = time.perf_counter() - t0
    diag = {
        "operator": operator,
        "variant": variant["variant"],
        "rect": rect,
        "n_nodes": int(n_nodes_total),
        "L": L,
        "K": K,
        "rank": _rank_summary(rect_diags),
        "adaptive_response": {
            "max_resplits": max_resplits,
            "n_resplits": int(n_resplit),
            "triggered": bool(n_resplit),
            "rule": (
                "split long axis when the projection rank saturates or clips "
                "at capacity-1; raw projected values are padded to the guard "
                "collar and strict band filtering happens after polish"
            ),
        },
        "n_raw_projected": int(values.size),
        "n_polished": int(len(polished)),
        "n_kept": int(len(kept)),
        "node_solve": _merge_solve_diags(solve_diags),
        "timing_s": {
            "assembly_rhs": float(assembly_rhs_s),
            "enumerate_rect": float(sum(
                d.get("timing_s", {}).get("enumerate_rect", 0.0)
                for d in rect_diags
            )),
            "polish": float(polish_s),
            "total": float(time.perf_counter() - t_total),
        },
    }
    return TemporalEnumeration(candidates=kept, diagnostics=diag)


def enumerate_temporal_band_batch(ops, pencil, cache, points, band, config, *,
                                  operator="ozgen_2d", family_index=0,
                                  residual_max=1.0e-8, bv_floor=1.0e-10):
    """Enumerate one family for multiple points by batching contour-node solves.

    This is the R3 cross-point path.  It batches the initial rectangle's
    ``points x contour nodes`` shifted solves, then preserves the previous
    per-point semantics for rank-revealing, polishing, filtering, and adaptive
    resplits.  A point whose initial projected rank saturates leaves the batch
    and reruns through :func:`enumerate_temporal_band` for the remedial split.
    """
    points = [dict(p) for p in points]
    if not points:
        return []
    if len(points) == 1:
        enum = enumerate_temporal_band(
            ops,
            pencil,
            cache,
            points[0],
            band,
            config,
            operator=operator,
            family_index=family_index,
            residual_max=residual_max,
            bv_floor=bv_floor,
        )
        enum.diagnostics["cross_point_batch"] = {
            "used": False,
            "reason": "single_point",
        }
        return [enum]

    cp = ops.cp
    t_assembly = time.perf_counter()
    matrices = []
    rhs_by_point = []
    assembly_rhs_by_point = []
    first = cache.evaluate(**points[0])
    n = int(np.asarray(first["A"]).shape[0])
    variant = _operator_variant(operator, n)
    L = int(variant["L"])
    K = int(variant["K"])
    rect = _finite_rect_for_band(band)
    nodes, weights = rectangle_gauss_legendre_nodes(rect, variant["nq_edge"])
    n_nodes = int(len(nodes))
    for point in points:
        t0 = time.perf_counter()
        mats = cache.evaluate(**point)
        A = np.asarray(mats["A"], dtype=np.complex128)
        B = np.asarray(mats["B"], dtype=np.complex128)
        if A.shape != (n, n) or B.shape != (n, n):
            raise ValueError("cross-point temporal batch requires uniform matrix size")
        V = _normalized_probe_block(n, L, _stable_seed(point, band, family_index))
        rhs_by_point.append(B @ V)
        matrices.append((A, B))
        assembly_rhs_by_point.append(time.perf_counter() - t0)
    assembly_total_s = time.perf_counter() - t_assembly

    scheduler = TileScheduler(ops, config)
    n_terms = int(pencil.kA + pencil.kB)
    plan = scheduler.plan(
        total_lanes=len(points) * n_nodes,
        n=n,
        n_terms=n_terms,
        nrhs=L,
    )
    point_tile = max(1, int(plan.tile_size) // n_nodes)
    point_tile = min(point_tile, len(points))
    vram_start = _vram_snapshot(cp)
    vram_min_free = [
        vram_start.get("free_bytes")
        if isinstance(vram_start.get("free_bytes"), int) else None
    ]

    def observe_vram():
        snap = _vram_snapshot(cp)
        free = snap.get("free_bytes")
        if isinstance(free, int):
            if vram_min_free[0] is None:
                vram_min_free[0] = free
            else:
                vram_min_free[0] = min(vram_min_free[0], free)
        return snap

    batch_meta = {
        "used": True,
        "strategy": "point_tiles_x_contour_nodes",
        "point_tile_size": int(point_tile),
        "point_tile_source": "tile_scheduler_lane_cap",
        "contour_nodes_per_point": n_nodes,
        "requested_lane_tile_size": int(plan.tile_size),
        "effective_lane_tile_size": int(point_tile * n_nodes),
        "n_point_tiles": int(math.ceil(len(points) / point_tile)),
        "tile_plan": plan.meta,
        "resplit_policy": (
            "rank-saturated or capacity-clipped points leave the cross-point "
            "tile and rerun through the existing per-point adaptive splitter"
        ),
    }
    rhs_by_point = np.asarray(rhs_by_point, dtype=np.complex128)
    moments_by_point = [
        [np.zeros((n, L), dtype=np.complex128) for _ in range(2 * K)]
        for _ in points
    ]
    solve_diags_by_point = [[] for _ in points]
    solve_s_by_point = [0.0 for _ in points]
    moment_s_by_point = [0.0 for _ in points]
    phi = np.asarray((nodes - _rect_center_scale(rect)[0]) / _rect_center_scale(rect)[1],
                     dtype=np.complex128)
    w = np.asarray(weights, dtype=np.complex128)
    coeffs = [
        cp.asarray(w * (phi ** power), dtype=cp.complex128)
        for power in range(2 * K)
    ]

    for start in range(0, len(points), point_tile):
        stop = min(start + point_tile, len(points))
        chunk_points = points[start:stop]
        n_chunk = stop - start
        node_points = {
            p: np.repeat([float(point[p]) for point in chunk_points], n_nodes)
            for p in pencil.basis.params
        }
        z = np.tile(nodes, n_chunk)
        rhs_tile = np.repeat(rhs_by_point[start:stop], n_nodes, axis=0)
        t0 = time.perf_counter()
        rhs_eq = pencil.scale_rhs(cp.asarray(rhs_tile))
        solve = solve_pencil(ops, pencil, node_points, z, rhs_eq, config)
        Y = pencil.unscale_solution(solve.x).astype(cp.complex128, copy=False)
        Y = Y.reshape(n_chunk, n_nodes, n, L)
        observe_vram()
        solve_s = time.perf_counter() - t0
        for local in range(n_chunk):
            solve_s_by_point[start + local] += solve_s / n_chunk
        t0 = time.perf_counter()
        for power, coeff in enumerate(coeffs):
            part = cp.tensordot(coeff, Y, axes=(0, 1))
            part_h = cp.asnumpy(part)
            for local in range(n_chunk):
                moments_by_point[start + local][power] += part_h[local]
        observe_vram()
        moment_s = time.perf_counter() - t0
        for local in range(n_chunk):
            moment_s_by_point[start + local] += moment_s / n_chunk
        for local in range(n_chunk):
            idx = np.arange(local * n_nodes, (local + 1) * n_nodes)
            solve_diags_by_point[start + local].append(
                _solve_diag_for_indices(solve, cp, idx)
            )

    center, scale = _rect_center_scale(rect)
    projected, projection_meta = _project_many_from_moments(
        moments_by_point, K=K, center=center, scale=scale
    )
    projection_wall_share_s = (
        projection_meta["wall_s"] / len(points) if points else 0.0
    )
    batch_meta["projection"] = projection_meta
    vram_end = _vram_snapshot(cp)
    if isinstance(vram_start.get("total_bytes"), int) and vram_min_free[0] is not None:
        total_bytes = int(vram_start["total_bytes"])
        min_free = int(vram_min_free[0])
        start_free = int(vram_start.get("free_bytes", min_free))
        batch_meta["vram_observed"] = {
            "start": vram_start,
            "end": vram_end,
            "min_free_bytes": min_free,
            "peak_used_bytes": int(total_bytes - min_free),
            "peak_delta_bytes": int(max(0, start_free - min_free)),
        }
    else:
        batch_meta["vram_observed"] = {
            "start": vram_start,
            "end": vram_end,
        }
    enumerations = []
    for idx, point in enumerate(points):
        _proj_idx, values_i, vectors_i, rank_diag, projection_worker_s = projected[idx]
        projection_s = float(projection_wall_share_s)
        rank_diag = dict(rank_diag)
        rank_diag["projection_worker_s"] = float(projection_worker_s)
        rank_diag["rank_clipped"] = bool(rank_diag["rank"] >= rank_diag["capacity"] - 1)
        rank_clipped = bool(rank_diag["rank_clipped"])
        if rank_diag["rank_saturated"] or rank_clipped:
            t0 = time.perf_counter()
            enum = enumerate_temporal_band(
                ops,
                pencil,
                cache,
                point,
                band,
                config,
                operator=operator,
                family_index=family_index,
                residual_max=residual_max,
                bv_floor=bv_floor,
            )
            remedial_s = time.perf_counter() - t0
            enum.diagnostics["cross_point_batch"] = {
                **batch_meta,
                "remedial_per_point": True,
                "initial_rank": rank_diag,
            }
            timing = dict(enum.diagnostics.get("timing_s", {}))
            timing.update({
                "assembly_rhs": float(assembly_rhs_by_point[idx]),
                "batched_solve": float(solve_s_by_point[idx]),
                "moment_transfer": float(moment_s_by_point[idx]),
                "projection": float(projection_s),
                "remedial_per_point": float(remedial_s),
                "batch_setup_total": float(assembly_total_s),
            })
            timing["total"] = float(sum(
                v for k, v in timing.items()
                if k not in ("total", "batch_setup_total")
            ))
            enum.diagnostics["timing_s"] = timing
            enumerations.append(enum)
            continue

        raw_pad = _raw_filter_pad(rect)
        if values_i.size:
            inside = np.array(
                [_inside_rect(z, rect, pad=raw_pad) for z in values_i],
                dtype=bool,
            )
            values = np.asarray(values_i[inside], dtype=np.complex128)
            vectors = vectors_i[:, inside]
            n_inside = int(np.count_nonzero(inside))
        else:
            values = np.zeros(0, dtype=np.complex128)
            vectors = np.zeros((n, 0), dtype=np.complex128)
            n_inside = 0
        A, B = matrices[idx]
        t0 = time.perf_counter()
        kept, polished = _polish_and_filter_candidates(
            ops,
            pencil,
            point,
            A,
            B,
            values,
            vectors,
            config,
            band,
            operator,
            residual_max,
            bv_floor,
        )
        polish_s = time.perf_counter() - t0
        rect_diag = {
            **rank_diag,
            "rect": rect,
            "resplit": False,
            "raw_filter_pad": raw_pad,
            "n_raw_projected_before_filter": int(values_i.size),
            "n_raw_projected_in_collar": n_inside,
        }
        diag = {
            "operator": operator,
            "variant": variant["variant"],
            "rect": rect,
            "n_nodes": n_nodes,
            "L": L,
            "K": K,
            "rank": _rank_summary([rect_diag]),
            "adaptive_response": {
                "max_resplits": 2,
                "n_resplits": 0,
                "triggered": False,
                "rule": (
                    "initial rectangle was solved in a cross-point tile; "
                    "strict band filtering happens after polish"
                ),
            },
            "n_raw_projected": int(values.size),
            "n_polished": int(len(polished)),
            "n_kept": int(len(kept)),
            "node_solve": _merge_solve_diags(solve_diags_by_point[idx]),
            "timing_s": {
                "assembly_rhs": float(assembly_rhs_by_point[idx]),
                "batched_solve": float(solve_s_by_point[idx]),
                "moment_transfer": float(moment_s_by_point[idx]),
                "projection": float(projection_s),
                "polish": float(polish_s),
                "total": float(
                    assembly_rhs_by_point[idx]
                    + solve_s_by_point[idx]
                    + moment_s_by_point[idx]
                    + projection_s
                    + polish_s
                ),
                "batch_setup_total": float(assembly_total_s),
            },
            "cross_point_batch": {
                **batch_meta,
                "remedial_per_point": False,
            },
        }
        enumerations.append(TemporalEnumeration(candidates=kept, diagnostics=diag))
    return enumerations


def candidate_prefilter(values, vectors=None, *, cr_band=None,
                        ci_abs_max=np.inf, bv_norm=None, bv_floor=0.0,
                        residual=None, residual_max=np.inf,
                        edge_ratio=None, edge_ratio_max=np.inf):
    """Return a boolean keep mask for certainly inadmissible candidates.

    This function never decides a final verdict.  It only applies monotone
    filters that remove values outside the requested phase-speed band, outside
    the absolute growth cap, below the singular-B leakage floor, above the FP64
    residual cap, or above the freestream-decay cap.
    """
    z = np.asarray(values, dtype=np.complex128)
    keep = np.isfinite(z.real) & np.isfinite(z.imag)
    if cr_band is not None:
        lo, hi = cr_band
        keep &= (z.real > float(lo)) & (z.real < float(hi))
    if np.isfinite(ci_abs_max):
        keep &= np.abs(z.imag) < float(ci_abs_max)
    if bv_norm is not None:
        keep &= np.asarray(bv_norm, dtype=float) >= float(bv_floor)
    if residual is not None:
        keep &= np.asarray(residual, dtype=float) <= float(residual_max)
    if edge_ratio is not None:
        keep &= np.asarray(edge_ratio, dtype=float) <= float(edge_ratio_max)
    if vectors is not None:
        vec = np.asarray(vectors)
        if vec.ndim >= 2:
            keep &= np.linalg.norm(vec.reshape(vec.shape[0], -1), axis=1) > 0
    return keep


def seed_code_legend():
    """Seed/provenance code legend for GPU temporal artifacts."""
    return {
        "0": "full-spectrum CPU QZ at this point",
        "-1": "no admissible mode in the family window",
        "-2": "per-point record construction raised; exception repr in meta['errors']",
        "1": "GPU contour enumerator candidate source",
        "reserved_positive": "positive ints are reserved for GPU seed or candidate-chain ids",
    }
