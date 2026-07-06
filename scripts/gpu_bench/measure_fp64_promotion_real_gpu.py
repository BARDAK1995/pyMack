"""Measure real-GPU z128 promotion rates for slice 08.

This is intentionally a measurement script, not the production enumerator.
It uses the committed hard-cell corpus parameters, assembles representative
production matrices on the CPU, then runs the actual CuPy/cuBLAS batched
complex64 LU plus FP64 residual-refinement path on the local GPU.  The output
is the binding first deliverable for slice 08:

    docs/gpu/benchmarks/fp64_promotion_real_gpu.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORPUS = REPO / "verification" / "gpu_certification" / "hard_cells"
for path in (HERE, CORPUS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_truth_set as bts  # noqa: E402
import spike_tournament_core as core  # noqa: E402
import spike_tournament_families as fams  # noqa: E402
from pymack.gpu import backend  # noqa: E402
from pymack.gpu.kernels import CupyBatchedLinalgOps  # noqa: E402


OUT_DEFAULT = REPO / "docs" / "gpu" / "benchmarks" / "fp64_promotion_real_gpu.json"


def _load_manifest():
    return json.loads((CORPUS / "truth_manifest.json").read_text(encoding="utf-8"))


def _cell_by_id(cell_id):
    for cell in _load_manifest()["cells"]:
        if cell["id"] == cell_id:
            return cell
    raise KeyError(cell_id)


def _json_pair(z):
    z = complex(z)
    return [float(z.real), float(z.imag)]


def _rect_center(rect):
    return complex(
        0.5 * (rect["real"][0] + rect["real"][1]),
        0.5 * (rect["imag"][0] + rect["imag"][1]),
    )


def _linear_case(cell, variant_name):
    p = cell["params"]
    cfg = fams.CFG["families"][cell["family"]]["contour"]
    variant = cfg["variants"][variant_name]
    rect = fams._guard_rect(bts.box_for_cell(cell), cfg["guard"], 1.0)
    nodes, _weights = core.rect_nodes(rect, int(cfg["nq_edge"]))
    if cell["kind"] == "ozgen_pair":
        ymf = float(p["ymf_pair"][0])
        A, B, _y = fams.assemble_ozgen(p, ymf, float(p["alpha"]))
        params = {"alpha": float(p["alpha"]), "Re": float(p["Re"]), "ymf": ymf}
    elif cell["kind"] == "mack_3d":
        A, B, _y = fams.assemble_mack(p, float(p["alpha"]), float(p["beta"]))
        params = {
            "alpha": float(p["alpha"]),
            "beta": float(p["beta"]),
            "R": float(p["R"]),
            "psi_deg": float(p["psi_deg"]),
        }
    else:
        raise ValueError(f"not a linear temporal cell: {cell['kind']}")
    L = int(variant["L"])
    seed = fams._seed_for("promotion", cell["id"], variant_name)
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((A.shape[0], L)) + 1j * rng.standard_normal((A.shape[0], L))
    rhs = B @ V
    matrices = np.stack([A - z * B for z in nodes])
    scale_matrix = A - _rect_center(rect) * B
    return {
        "cell": cell,
        "variant": variant_name,
        "operator_kind": "linear_pencil",
        "A": A,
        "B": B,
        "nodes": nodes,
        "matrices": matrices,
        "rhs": np.repeat(rhs[None, :, :], len(nodes), axis=0),
        "scale_matrix": scale_matrix,
        "rhs_block_columns": L,
        "params": params,
        "contour": {
            "type": "rectangle_gauss_legendre_edges",
            "rect": rect,
            "n_nodes": int(len(nodes)),
            "nq_edge": int(cfg["nq_edge"]),
        },
    }


def _mazhong_case(cell):
    p = cell["params"]
    cfg = fams.CFG["families"][cell["family"]]["contour"]
    C0, C1, C2, _y = fams.assemble_mazhong(p, float(p["omega"]))
    a0 = complex(float(p["omega"]) / float(p["c_guess"]))
    a_lo = float(p["omega"]) / float(p["c_hi"])
    a_hi = float(p["omega"]) / float(p["c_lo"])
    radius = max(abs(a0 - a_lo), abs(a0 - a_hi))
    nodes, _weights = core.circle_nodes(a0, radius, int(cfg["disk_nq"]))
    L = int(cfg["disk_L"])
    seed = fams._seed_for("promotion", cell["id"], "window")
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((C0.shape[0], L)) + 1j * rng.standard_normal((C0.shape[0], L))
    matrices = np.stack([C0 + z * C1 + z * z * C2 for z in nodes])
    scale_matrix = C0 + a0 * C1 + a0 * a0 * C2
    return {
        "cell": cell,
        "variant": "window",
        "operator_kind": "quadratic_pencil",
        "C0": C0,
        "C1": C1,
        "C2": C2,
        "nodes": nodes,
        "matrices": matrices,
        "rhs": np.repeat(V[None, :, :], len(nodes), axis=0),
        "scale_matrix": scale_matrix,
        "rhs_block_columns": L,
        "params": {
            "omega": float(p["omega"]),
            "R": float(p["R"]),
            "c_guess": float(p["c_guess"]),
            "mode": str(p["mode"]),
        },
        "contour": {
            "type": "target_centered_disk",
            "center": _json_pair(a0),
            "radius": float(radius),
            "n_nodes": int(len(nodes)),
        },
    }


def _sample_cases():
    return [
        _linear_case(_cell_by_id("hc_028"), "single"),
        _linear_case(_cell_by_id("hc_001"), "hankel"),
        _mazhong_case(_cell_by_id("hc_040")),
    ]


def _filter_eigen_bases(raw, *, max_bases=12):
    raw = np.asarray(raw, dtype=complex)
    finite = raw[np.isfinite(raw)]
    phys = finite[
        (finite.real > -0.5)
        & (finite.real < 1.5)
        & (np.abs(finite.imag) < 0.5)
    ]
    vals = phys if phys.size else finite[:max_bases]
    if vals.size == 0:
        vals = np.asarray([0.9 + 0.0j])
    order = np.argsort(-vals.imag)
    vals = vals[order]
    if vals.size > max_bases:
        vals = vals[:max_bases]
    return np.asarray(vals, dtype=complex)


def _computed_eigen_bases(case, *, max_bases=12):
    import scipy.linalg as sla

    if case["operator_kind"] == "linear_pencil":
        raw = sla.eigvals(case["A"], case["B"])
        return _filter_eigen_bases(raw, max_bases=max_bases)
    C0 = case["C0"]
    C1 = case["C1"]
    C2 = case["C2"]
    n = C0.shape[0]
    Z = np.zeros_like(C0)
    I = np.eye(n)
    A = np.block([[Z, I], [-C0, -C1]])
    B = np.block([[I, Z], [Z, C2]])
    raw = sla.eigvals(A, B)
    return _filter_eigen_bases(raw, max_bases=max_bases)


def _stored_eigen_bases(cell, *, max_bases=12):
    _cell, spectra, _verdict, _kappa, _cross = bts.load_npz(CORPUS / cell["npz"])
    vals = []
    for spec in spectra:
        if cell["kind"] == "ozgen_pair":
            raw = np.asarray(spec["prod_eigenvalues"], dtype=complex)
        elif "prod_candidates" in spec:
            raw = np.asarray(spec["prod_candidates"], dtype=complex)
        else:
            raw = np.asarray(spec["raw_eigenvalues"], dtype=complex)
        vals.extend(_filter_eigen_bases(raw, max_bases=max_bases).tolist())
    return _filter_eigen_bases(vals, max_bases=max_bases)


def _sample_near_eigen_nodes(case, n_samples):
    try:
        bases = _stored_eigen_bases(case["cell"])
        source = "stored_hard_cell_spectrum"
    except FileNotFoundError:
        bases = _computed_eigen_bases(case)
        source = "direct_qz_fallback_from_assembled_matrices"
    rng = np.random.default_rng(
        fams._seed_for("promotion", case["cell"]["id"], "near_eigen")
    )
    chosen = bases[np.arange(n_samples) % len(bases)]
    scales = np.logspace(-2, -9, 8)
    scale = np.resize(np.repeat(scales, max(1, n_samples // len(scales) + 1)), n_samples)
    direction = rng.normal(size=n_samples) + 0.7j * rng.normal(size=n_samples)
    direction /= np.maximum(np.abs(direction), 1.0e-300)
    return chosen + direction * scale, source


def _sample_corner_nodes(case, n_samples):
    contour = case["contour"]
    rng = np.random.default_rng(fams._seed_for("promotion", case["cell"]["id"], "corners"))
    if "rect" in contour:
        rect = contour["rect"]
        corners = np.asarray([
            complex(rect["real"][0], rect["imag"][0]),
            complex(rect["real"][1], rect["imag"][0]),
            complex(rect["real"][1], rect["imag"][1]),
            complex(rect["real"][0], rect["imag"][1]),
        ])
        real_w = rect["real"][1] - rect["real"][0]
        imag_w = rect["imag"][1] - rect["imag"][0]
        jitter = 1.0e-4 * max(real_w, imag_w, 1.0e-12)
        base = corners[np.arange(n_samples) % len(corners)]
        offsets = jitter * (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples))
        return base + offsets
    center = complex(*contour["center"])
    radius = float(contour["radius"])
    theta = 0.5 * np.pi * (np.arange(n_samples) % 4)
    theta = theta + rng.normal(scale=1.0e-4, size=n_samples)
    return center + radius * np.exp(1j * theta)


def _matrices_for_nodes(case, nodes):
    if case["operator_kind"] == "linear_pencil":
        A = case["A"]
        B = case["B"]
        return np.stack([A - z * B for z in nodes])
    C0 = case["C0"]
    C1 = case["C1"]
    C2 = case["C2"]
    return np.stack([C0 + z * C1 + z * z * C2 for z in nodes])


def _case_for_regime(case, regime):
    if regime == "contour_generic":
        return case
    n_samples = int(len(case["nodes"]))
    if regime == "contour_corners":
        nodes = _sample_corner_nodes(case, n_samples)
        note = (
            "Contour-corner stress points adapted from the track-A corner "
            "regime for this hard-cell matrix surface."
        )
    elif regime == "near_eigen_polish":
        nodes, source = _sample_near_eigen_nodes(case, n_samples)
        note = (
            "Near-eigen shifts sampled from stored hard-cell spectra; this "
            "approximates the RQI polish regime where A-zB is nearly singular. "
            f"basis_source={source}."
        )
    else:
        raise ValueError(regime)
    rhs0 = case["rhs"][0]
    matrices = _matrices_for_nodes(case, nodes)
    center_matrix = _matrices_for_nodes(case, np.asarray([np.mean(nodes)]))[0]
    out = dict(case)
    out["nodes"] = nodes
    out["matrices"] = matrices
    out["rhs"] = np.repeat(rhs0[None, :, :], n_samples, axis=0)
    out["scale_matrix"] = center_matrix
    out["contour"] = {
        "type": regime,
        "n_nodes": n_samples,
        "note": note,
        "node_preview": [_json_pair(z) for z in nodes[:4]],
    }
    return out


def _relative_residual(cp, M, x, b):
    r = b - cp.matmul(M, x)
    rn = cp.linalg.norm(r.reshape(r.shape[0], -1), axis=1)
    bn = cp.linalg.norm(b.reshape(b.shape[0], -1), axis=1)
    bn = cp.where(bn == 0, cp.asarray(1.0), bn)
    return rn / bn, r


def _run_dense_refine(ops, matrices, rhs, scale_matrix, *, tol, max_iter, growth_limit, tile):
    cp = ops.cp
    n_lanes = int(matrices.shape[0])
    n = int(matrices.shape[1])
    nrhs = int(rhs.shape[2])
    dr, dc = core.pow2_scalings(scale_matrix)
    rows = []
    totals = {"stall": 0, "pivot_growth": 0, "info_nonzero": 0}
    promoted_total = 0
    rel_max = 0.0
    best_max = 0.0
    tile = max(1, min(int(tile), n_lanes))
    for start in range(0, n_lanes, tile):
        stop = min(start + tile, n_lanes)
        Mh = matrices[start:stop]
        bh = rhs[start:stop]
        M64 = cp.asarray((Mh * dr[:, None]) * dc[None, :]).astype(cp.complex64)
        M128 = cp.asarray((Mh * dr[:, None]) * dc[None, :]).astype(cp.complex128)
        b64 = cp.asarray(bh * dr[None, :, None]).astype(cp.complex64)
        b128 = cp.asarray(bh * dr[None, :, None]).astype(cp.complex128)

        lu = ops.lu_factor(M64)
        x = ops.lu_solve(lu, b64, trans="N").astype(cp.complex128)
        best = cp.full((stop - start,), cp.inf, dtype=cp.float64)
        rel = None
        for _ in range(max_iter):
            rel, r = _relative_residual(cp, M128, x, b128)
            best = cp.minimum(best, rel)
            active = rel > tol
            if not bool(cp.any(active)):
                break
            dx = ops.lu_solve(lu, r.astype(cp.complex64), trans="N").astype(cp.complex128)
            x = x + cp.where(active[:, None, None], dx, 0)
        if rel is None:
            rel, _r = _relative_residual(cp, M128, x, b128)
        info = cp.asnumpy(lu.info)
        lu_max = cp.max(cp.abs(lu.lu).reshape(stop - start, -1), axis=1)
        m_max = cp.max(cp.abs(M64).reshape(stop - start, -1), axis=1)
        growth = cp.asnumpy(lu_max / cp.maximum(m_max, cp.asarray(1.0)))
        best_h = cp.asnumpy(best)
        rel_h = cp.asnumpy(rel)
        info_mask = info != 0
        growth_mask = growth > growth_limit
        stall_mask = best_h > tol
        promoted = info_mask | growth_mask | stall_mask
        promoted_total += int(np.count_nonzero(promoted))
        totals["info_nonzero"] += int(np.count_nonzero(info_mask))
        totals["pivot_growth"] += int(np.count_nonzero(~info_mask & growth_mask))
        totals["stall"] += int(np.count_nonzero(~info_mask & ~growth_mask & stall_mask))
        rel_max = max(rel_max, float(np.nanmax(rel_h)))
        best_max = max(best_max, float(np.nanmax(best_h)))
        rows.append({
            "lane_start": int(start),
            "lane_stop": int(stop),
            "promoted": int(np.count_nonzero(promoted)),
            "info_nonzero": int(np.count_nonzero(info_mask)),
            "pivot_growth": int(np.count_nonzero(~info_mask & growth_mask)),
            "stall": int(np.count_nonzero(~info_mask & ~growth_mask & stall_mask)),
            "max_growth": float(np.nanmax(growth)),
            "max_best_residual": float(np.nanmax(best_h)),
            "max_exit_residual": float(np.nanmax(rel_h)),
        })
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "lanes": n_lanes,
        "matrix_dim": n,
        "rhs_block_columns": nrhs,
        "batch_size": int(tile),
        "promoted": int(promoted_total),
        "promotion_rate": float(promoted_total / n_lanes if n_lanes else math.nan),
        "trigger_counts": totals,
        "max_best_residual": float(best_max),
        "max_exit_residual": float(rel_max),
        "tiles": rows,
        "equilibration": "shared power-of-two scaling from contour center matrix",
    }


def measure(*, output, tile, tol, max_iter, sysmem_policy):
    import cupy as cp

    ops = CupyBatchedLinalgOps(cp=cp)
    cfg = backend.resolve_config()
    started = time.perf_counter()
    cases = []
    regimes = ("contour_generic", "contour_corners", "near_eigen_polish")
    for base_case in _sample_cases():
        cell = base_case["cell"]
        regime_breakdown = {}
        elapsed = 0.0
        for regime in regimes:
            case = _case_for_regime(base_case, regime)
            t0 = time.perf_counter()
            stats = _run_dense_refine(
                ops,
                case["matrices"],
                case["rhs"],
                case["scale_matrix"],
                tol=tol,
                max_iter=max_iter,
                growth_limit=cfg.promote_growth,
                tile=tile,
            )
            dt = time.perf_counter() - t0
            elapsed += dt
            regime_breakdown[regime] = {
                "stats": stats,
                "contour": case["contour"],
                "elapsed_s": float(dt),
            }
        generic_stats = regime_breakdown["contour_generic"]["stats"]
        cases.append({
            "id": cell["id"],
            "family": cell["family"],
            "kind": cell["kind"],
            "variant": base_case["variant"],
            "operator_kind": base_case["operator_kind"],
            "params": base_case["params"],
            "contour": base_case["contour"],
            "stats": generic_stats,
            "regime_breakdown": regime_breakdown,
            "elapsed_s": float(elapsed),
        })
    total_lanes = sum(
        r["stats"]["lanes"]
        for c in cases
        for r in c["regime_breakdown"].values()
    )
    total_promoted = sum(
        r["stats"]["promoted"]
        for c in cases
        for r in c["regime_breakdown"].values()
    )
    summary_by_regime = {}
    for regime in regimes:
        lanes = sum(c["regime_breakdown"][regime]["stats"]["lanes"] for c in cases)
        promoted = sum(c["regime_breakdown"][regime]["stats"]["promoted"] for c in cases)
        summary_by_regime[regime] = {
            "lanes": int(lanes),
            "promoted": int(promoted),
            "promotion_rate": float(promoted / lanes if lanes else math.nan),
        }
    near_rates = {
        c["id"]: c["regime_breakdown"]["near_eigen_polish"]["stats"]["promotion_rate"]
        for c in cases
    }
    payload = {
        "schema_version": 2,
        "artifact": "fp64_promotion_real_gpu",
        "status": "complete",
        "sysmem_policy": sysmem_policy,
        "numbers_provisional": True,
        "measurement_note": (
            "Counts and rates are valid under GPU sharing; elapsed timings are "
            "provisional until the card is run serially with sysmem fallback policy confirmed."
        ),
        "engine_surface": "real CuPy/cuBLAS c64 getrfBatched/getrsBatched plus FP64 residual refinement",
        "promotion_policy": {
            "tol": float(tol),
            "max_iter": int(max_iter),
            "promote_growth": float(cfg.promote_growth),
            "triggers": ["stall", "pivot_growth", "info_nonzero"],
        },
        "overall": {
            "lanes": int(total_lanes),
            "promoted": int(total_promoted),
            "promotion_rate": float(total_promoted / total_lanes if total_lanes else math.nan),
        },
        "summary_by_regime": summary_by_regime,
        "explanation": (
            "C4 grafted the track-A three-regime study into this worktree. "
            "The original slice-08 measurement covered only contour_generic "
            f"nodes and found zero promotions; this rerun still measures "
            f"contour_generic={summary_by_regime['contour_generic']['promoted']}/"
            f"{summary_by_regime['contour_generic']['lanes']} and "
            f"contour_corners={summary_by_regime['contour_corners']['promoted']}/"
            f"{summary_by_regime['contour_corners']['lanes']}. The added "
            "near_eigen_polish regime deliberately places shifts close to "
            "hard-cell eigenvalues, matching the RQI-polish conditioning "
            f"story from track A; this rerun measures "
            f"{summary_by_regime['near_eigen_polish']['promoted']}/"
            f"{summary_by_regime['near_eigen_polish']['lanes']} promoted "
            f"({summary_by_regime['near_eigen_polish']['promotion_rate']:.6f}), "
            f"with per-case rates {near_rates}. That is the relevant "
            "promotion budget for polish, while generic contour enumeration "
            "remains low-promotion. Counts are valid under GPU sharing; "
            "timings are provisional."
        ),
        "cases": cases,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "platform": platform.platform(),
        },
        "device": backend.device_report(cfg),
        "elapsed_s": float(time.perf_counter() - started),
        "generated_at_unix": time.time(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1.0e-10)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--sysmem-policy", required=True, choices=["unknown", "prefer_no_sysmem_fallback"])
    args = ap.parse_args(argv)
    payload = measure(
        output=args.output,
        tile=args.tile,
        tol=args.tol,
        max_iter=args.max_iter,
        sysmem_policy=args.sysmem_policy,
    )
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "overall": payload["overall"],
        "cases": [
            {
                "id": c["id"],
                "family": c["family"],
                "lanes": c["stats"]["lanes"],
                "promoted": c["stats"]["promoted"],
                "promotion_rate": c["stats"]["promotion_rate"],
                "trigger_counts": c["stats"]["trigger_counts"],
                "regime_rates": {
                    name: r["stats"]["promotion_rate"]
                    for name, r in c["regime_breakdown"].items()
                },
            }
            for c in payload["cases"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
