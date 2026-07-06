"""Build the slice-08 enumerator bake-off artifact.

The artifact intentionally separates two evidence classes:

* the inherited D1 candidate matrix over the required hard-cell subset; and
* current real-GPU replay through ``temporal_sweep(backend='gpu')`` for cells
  that are representable by the public temporal sweep facade.

The Ma/Zhong spatial window and Ozgen ymf-pair hard-cell mechanisms are not
the public temporal sweep surface, so they are not silently reported as fresh
real-GPU temporal measurements here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pymack
from pymack.sweep import CBand, temporal_sweep


REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs" / "gpu" / "benchmarks" / "tournament_report.json"
MANIFEST = REPO / "verification" / "gpu_certification" / "hard_cells" / "truth_manifest.json"
OUT = REPO / "docs" / "gpu" / "benchmarks" / "enumerator_bakeoff.json"
SUBSET = ["hc_035", "hc_001", "hc_040", "hc_028"]


def _flat_diag(diag):
    out = []
    for item in diag or []:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def _variant_summary(row, variant_name, variant):
    recall = variant.get("recall", {})
    ledger = variant.get("ledger_base", {})
    polish = variant.get("polish_stats", [])
    diag = _flat_diag(variant.get("diag"))
    return {
        "cell": row["id"],
        "family": row["family"],
        "variant": variant_name,
        "recall": float(recall.get("recall", 0.0)),
        "n_truth": int(recall.get("n_truth", 0)),
        "n_recalled": int(recall.get("n_recalled", 0)),
        "lu_equivalents": float(ledger.get("lu_equivalents", 0.0)),
        "wall_proxy": "factorization_lu_equivalents_from_d1",
        "failure_taxonomy": {
            "rank_saturated": int(sum(1 for d in diag if d.get("rank_saturated"))),
            "resplit": int(sum(1 for d in diag if d.get("resplit"))),
            "unconverged_polish": int(sum(s.get("n_unconverged", 0) for s in polish)),
            "infinite_leakage": int(sum(s.get("n_infinite_leakage", 0) for s in polish)),
        },
        "certified_residual_max": variant.get("certified_residual_max"),
        "stability_pass": bool(variant.get("stability", {}).get("pass", False)),
    }


def _d1_candidate_matrix():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in report["cells"]["contour"]}
    official = report["official_contour_variant_per_family"]
    candidates = {}
    for cell_id in SUBSET:
        row = rows[cell_id]
        for variant_name, variant in row["variants"].items():
            label = {
                "single": "one_shot_beyn_rectangle_cpu_projected",
                "split4": "split_rectangle_beyn_cpu_projected",
                "hankel": "higher_k_hankel_rectangle_cpu_projected",
                "window": "target_window_disk_cpu_projected",
            }.get(variant_name, variant_name)
            candidates.setdefault(label, []).append(
                _variant_summary(row, variant_name, variant)
            )
    totals = {}
    for name, items in candidates.items():
        n_truth = sum(i["n_truth"] for i in items)
        n_recalled = sum(i["n_recalled"] for i in items)
        totals[name] = {
            "n_cells": len(items),
            "recall": float(n_recalled / n_truth) if n_truth else 1.0,
            "n_truth": n_truth,
            "n_recalled": n_recalled,
            "lu_equivalents": float(sum(i["lu_equivalents"] for i in items)),
            "rank_saturated": int(sum(i["failure_taxonomy"]["rank_saturated"] for i in items)),
            "resplit": int(sum(i["failure_taxonomy"]["resplit"] for i in items)),
            "unconverged_polish": int(sum(i["failure_taxonomy"]["unconverged_polish"] for i in items)),
            "infinite_leakage": int(sum(i["failure_taxonomy"]["infinite_leakage"] for i in items)),
        }
    shipped = {
        rows[cell_id]["family"]: official[rows[cell_id]["family"]]
        for cell_id in SUBSET
    }
    return {
        "source": str(REPORT.relative_to(REPO)),
        "source_status": report.get("status"),
        "subset": SUBSET,
        "official_variant_per_family": shipped,
        "candidates": candidates,
        "totals": totals,
    }


def _cell(cell_id):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return next(c for c in manifest["cells"] if c["id"] == cell_id)


def _run_mack_public_replay(cell_id):
    cell = _cell(cell_id)
    p = cell["params"]
    profile = pymack.make_flatplate_profile(float(p["Ma"]))
    band = CBand(
        float(p["cr_band"][0]),
        float(p["cr_band"][1]),
        ci_abs_max=float(p.get("phys_ci_abs", p.get("ci_cap", 0.5))),
        label=cell_id,
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
    t0 = time.perf_counter()
    gpu = temporal_sweep(
        profile,
        [float(p["alpha"])],
        [float(p["R"])],
        backend="gpu",
        **kwargs,
    )
    elapsed = time.perf_counter() - t0
    cpu = temporal_sweep(
        profile,
        [float(p["alpha"])],
        [float(p["R"])],
        backend="cpu",
        **kwargs,
    )
    g = gpu.families[0]
    c = cpu.families[0]
    got = complex(g.c[0, 0])
    ref = complex(c.c[0, 0])
    err = abs(got - ref)
    return {
        "cell": cell_id,
        "family": cell["family"],
        "kind": cell["kind"],
        "surface": "public_temporal_sweep_mack_3d",
        "backend_status": gpu.meta.get("gpu_engine_status"),
        "operator_source": gpu.meta.get("operator_source"),
        "converged": bool(g.converged[0, 0]),
        "seed": int(g.seed_map[0, 0]),
        "gpu_value": [float(got.real), float(got.imag)],
        "cpu_qz_value": [float(ref.real), float(ref.imag)],
        "abs_error_vs_cpu_qz": float(err),
        "passed_1e_9": bool(err <= 1.0e-9 * max(1.0, abs(ref))),
        "residual": float(g.residual[0, 0]),
        "elapsed_s": float(elapsed),
        "n_failed_points": int(gpu.meta.get("n_failed_points", -1)),
        "diagnostics": gpu.meta.get("gpu_enumerator", {}),
    }


def _run_ozgen_public_smoke():
    profile = pymack.make_flatplate_profile(2.0)
    families = (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.45, 0.97, ci_abs_max=0.05, label="Mack"),
    )
    kwargs = dict(
        Ma=2.0,
        N=31,
        y_max=12.0,
        length_scale="L_star",
        operator="ozgen_2d",
        families=families,
        cpu_workers=1,
    )
    t0 = time.perf_counter()
    gpu = temporal_sweep(profile, [0.08], [900.0], backend="gpu", **kwargs)
    elapsed = time.perf_counter() - t0
    cpu = temporal_sweep(profile, [0.08], [900.0], backend="cpu", **kwargs)
    rows = []
    for k, (gf, cf) in enumerate(zip(gpu.families, cpu.families)):
        got = complex(gf.c[0, 0])
        ref = complex(cf.c[0, 0])
        err = abs(got - ref)
        rows.append({
            "family_index": int(k),
            "label": gf.label,
            "converged": bool(gf.converged[0, 0]),
            "seed": int(gf.seed_map[0, 0]),
            "gpu_value": [float(got.real), float(got.imag)],
            "cpu_qz_value": [float(ref.real), float(ref.imag)],
            "abs_error_vs_cpu_qz": float(err),
            "passed_1e_9": bool(err <= 1.0e-9 * max(1.0, abs(ref))),
            "residual": float(gf.residual[0, 0]),
        })
    return {
        "cell": "ozgen_public_smoke_N31",
        "family": "ozgen_public",
        "kind": "ozgen_2d_public",
        "surface": "public_temporal_sweep_ozgen_2d",
        "backend_status": gpu.meta.get("gpu_engine_status"),
        "operator_source": gpu.meta.get("operator_source"),
        "elapsed_s": float(elapsed),
        "n_failed_points": int(gpu.meta.get("n_failed_points", -1)),
        "families": rows,
        "diagnostics": gpu.meta.get("gpu_enumerator", {}),
    }


def build(output=OUT):
    started = time.perf_counter()
    real_gpu = {
        "status": "partial_real_gpu_public_temporal_replay",
        "note": (
            "Fresh real-GPU replay covers public temporal sweep surfaces. "
            "The Ozgen ymf-pair hard-cell and Ma/Zhong spatial-window cells "
            "remain represented by the D1 matrix below, not by fresh temporal "
            "backend measurements."
        ),
        "cases": [
            _run_mack_public_replay("hc_028"),
            _run_mack_public_replay("hc_035"),
            _run_ozgen_public_smoke(),
        ],
    }
    d1 = _d1_candidate_matrix()
    payload = {
        "schema_version": 2,
        "artifact": "enumerator_bakeoff",
        "status": "partial_real_gpu_replay_with_d1_candidate_matrix",
        "numbers_provisional": True,
        "sysmem_policy": "unknown",
        "real_gpu_replay": real_gpu,
        "d1_candidate_matrix": d1,
        "winner": {
            "production_rule": "pinned official D1 variants per family",
            "selected_by_family": d1["official_variant_per_family"],
            "current_temporal_backend": "device_contour_projection",
            "reason": (
                "The production temporal backend now uses the device contour "
                "projection path. The full hard-cell bake-off remains the D1 "
                "candidate matrix until non-public Ozgen/MaZhong hard-cell "
                "surfaces are wired into the production GPU runner."
            ),
        },
        "elapsed_s": float(time.perf_counter() - started),
        "generated_at_unix": time.time(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "real_gpu_cases": [
            c["cell"] for c in payload["real_gpu_replay"]["cases"]
        ],
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args(argv)
    build(args.output)


if __name__ == "__main__":
    main()
