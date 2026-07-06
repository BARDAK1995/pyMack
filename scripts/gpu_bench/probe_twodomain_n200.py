"""R3-5 probe (b): TWO-DOMAIN GRID production workload, measurement-only (NO gate).

Ma=2, N=200, ymf_pair=(35,45), the 13x15=195-node GRID[2] from
verification/mixed_mode/ozgen_fig3/_refdigitize/build_firstmode_grid.py
(Re logspace(350,5500,13) x alpha linspace(0.010,0.075,15)).

Runs both domain heights on GPU at N=200, applies the R3-2 GPU-side two-domain
assembly (assemble_ozgen_verdict_from_gpu_candidate_sets over the captured GPU
candidate sets), and compares a subset (<=20 nodes) against freshly-computed
dm.discrete_mode(N=200) -- the committed firstmode_grid.csv is ABSENT (only the
builder script exists), so a subset comparison is used and stated.

If the engine cannot run N=200 (VRAM, L sizing, dtype promotion), that is a
FINDING, not a failure: the exact break point + resource numbers are recorded.

NO pass/fail gate; deliverable is the honest table.  Trusted numbers require a
fresh serial process with --sysmem-policy prefer_no_sysmem_fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import pymack
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "docs" / "gpu" / "benchmarks" / "probe_twodomain_n200.json"


def _families():
    return (
        CBand(float("-inf"), 0.45, ci_abs_max=0.05, label="TS"),
        CBand(0.45, 0.97, ci_abs_max=0.05, label="Mack"),
    )


def _discrete_mode():
    v = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_refdigitize"
    if str(v) not in sys.path:
        sys.path.insert(0, str(v))
    import discrete_mode as dm  # noqa: E402
    return dm


def _vram_from_meta(meta):
    genum = meta.get("gpu_enumerator", {})
    for d in genum.get("diagnostics", []):
        cpb = d.get("cross_point_batch", {})
        if isinstance(cpb, dict) and "vram_observed" in cpb:
            return cpb["vram_observed"]
    return None


def _run_height(profile, alphas, Res, Ma, N, y_max):
    fams = _families()
    t0 = time.perf_counter()
    res = temporal_sweep(
        profile, alphas, Res, Ma=Ma, N=N, y_max=y_max,
        length_scale="L_star", operator="ozgen_2d", families=fams,
        backend="gpu", cpu_workers=1,
    )
    return res, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmem-policy", required=True, choices=["unknown", "prefer_no_sysmem_fallback"])
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--ymf-short", type=float, default=35.0)
    ap.add_argument("--ymf-tall", type=float, default=45.0)
    ap.add_argument("--subset", type=int, default=12, help="nodes for GPU-vs-fresh-dm check")
    args = ap.parse_args(argv)

    trusted = (args.sysmem_policy == "prefer_no_sysmem_fallback")
    Ma = 2.0
    profile = pymack.make_flatplate_profile(Ma)
    dstar = delta_star_over_lstar(profile)
    # GRID[2] from build_firstmode_grid.py
    Res = np.logspace(np.log10(350), np.log10(5500), 13)
    alphas = np.linspace(0.010, 0.075, 15)
    y_short = args.ymf_short * dstar
    y_tall = args.ymf_tall * dstar

    payload = {
        "schema_version": 1,
        "artifact": "probe_twodomain_n200",
        "probe": "R3-5(b) two-domain grid workload; measurement-only (no gate)",
        "sysmem_policy": args.sysmem_policy,
        "sysmem_policy_trusted": trusted,
        "numbers_provisional": not trusted,
        "committed_reference": "firstmode_grid.csv ABSENT (only builder present); "
                               "subset compared vs fresh dm.discrete_mode(N=200).",
        "workload": {
            "Ma": Ma, "N": int(args.N),
            "ymf_pair": [args.ymf_short, args.ymf_tall],
            "y_short_abs": float(y_short), "y_tall_abs": float(y_tall),
            "n_re": len(Res), "n_alpha": len(alphas), "n_points": len(Res) * len(alphas),
            "alpha_range": [float(alphas[0]), float(alphas[-1])],
            "Re_range": [float(Res[0]), float(Res[-1])],
        },
        "generated_at_unix": time.time(),
    }

    os.environ["PYMACK_GPU_CAPTURE_CANDIDATE_SETS"] = "1"
    t0 = time.perf_counter()
    try:
        res_s, wall_s = _run_height(profile, alphas, Res, Ma, args.N, y_short)
        res_t, wall_t = _run_height(profile, alphas, Res, Ma, args.N, y_tall)
    except Exception as exc:  # OOM / L sizing / dtype promotion == FINDING
        payload["status"] = "resource_ceiling_hit"
        payload["finding"] = {
            "where": "temporal_sweep(backend='gpu') N=%d two heights, 195 nodes" % args.N,
            "exception": repr(exc),
            "wall_s_before_fail": time.perf_counter() - t0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": payload["status"], "finding": payload["finding"]}, indent=2))
        return

    from pymack.gpu.verdict import assemble_ozgen_verdict_from_gpu_candidate_sets

    sets_s = res_s.meta.get("gpu_candidate_sets", {})
    sets_t = res_t.meta.get("gpu_candidate_sets", {})
    capture_ok = bool(sets_s) and bool(sets_t)
    payload["status"] = "measured_trusted" if trusted else "measured_provisional"
    payload["engine_wall_time_s_short"] = float(res_s.meta.get("engine_wall_time_s", wall_s))
    payload["engine_wall_time_s_tall"] = float(res_t.meta.get("engine_wall_time_s", wall_t))
    payload["engine_wall_time_s_total"] = (
        payload["engine_wall_time_s_short"] + payload["engine_wall_time_s_tall"]
    )
    payload["full_wall_time_s"] = float(time.perf_counter() - t0)
    payload["gpu_engine_status_short"] = res_s.meta.get("gpu_engine_status")
    payload["gpu_engine_status_tall"] = res_t.meta.get("gpu_engine_status")
    payload["affine_reverified_short"] = res_s.meta.get("affine_reverified")
    payload["affine_reverified_tall"] = res_t.meta.get("affine_reverified")
    payload["vram_observed_short"] = _vram_from_meta(res_s.meta)
    payload["vram_observed_tall"] = _vram_from_meta(res_t.meta)
    payload["gpu_candidate_capture_ok"] = capture_ok
    payload["escalation_short"] = res_s.meta.get("escalation")
    payload["escalation_tall"] = res_t.meta.get("escalation")

    # Both N=200 sweeps SUCCEEDED here.  Persist a PARTIAL artifact NOW so the
    # expensive sweep results survive even if the CPU subset phase (many N=200 QZ
    # solves) later exhausts memory or hard-faults.  (A prior run died in this
    # phase without an artifact.)
    print("SWEEPS_DONE short=%.1fs tall=%.1fs; writing partial artifact"
          % (payload["engine_wall_time_s_short"], payload["engine_wall_time_s_tall"]),
          flush=True)
    payload["status"] = "measured_partial_sweeps_only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Release the device pool before the host-heavy CPU subset phase.
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    # --- GPU-derived two-domain vs fresh dm.discrete_mode on a subset ---
    n_re = len(Res)
    n_families = len(_families())
    empty = {"values": np.zeros(0, dtype=complex), "vectors": np.zeros((0, 0), dtype=complex)}
    cmp_rows = []
    worst = 0.0
    n_none_mismatch = 0
    saw_nonzero = False
    if capture_ok:
        dm = _discrete_mode()
        rng = np.random.default_rng(20260706)
        n_total = len(alphas) * len(Res)
        pick = rng.choice(n_total, size=min(args.subset, n_total), replace=False)
        picks_sorted = sorted(pick)
        for ni, f in enumerate(picks_sorted):
            ii = int(f // n_re); jj = int(f % n_re)
            pidx = ii * n_re + jj
            ss = [sets_s.get((pidx, k), empty) for k in range(n_families)]
            st = [sets_t.get((pidx, k), empty) for k in range(n_families)]
            v = assemble_ozgen_verdict_from_gpu_candidate_sets(ss, st)
            ref = dm.discrete_mode(Ma, float(Res[jj]), float(alphas[ii]), N=args.N,
                                   ymf_pair=(args.ymf_short, args.ymf_tall))
            got_none = (v.value is None)
            ref_none = (ref is None)
            row = {"ia": ii, "jr": jj, "alpha": float(alphas[ii]), "Re": float(Res[jj]),
                   "gpu_none": got_none, "ref_none": ref_none}
            if got_none != ref_none:
                n_none_mismatch += 1
                row["reason"] = "none_mismatch"
            elif not ref_none:
                err = abs(complex(v.value) - complex(ref["c_r"], ref["c_i"]))
                worst = max(worst, float(err))
                row["err"] = float(err)
                if err > 0.0:
                    saw_nonzero = True
            cmp_rows.append(row)
            print("subset %d/%d node(a=%.4f,Re=%.1f) gpu_none=%s ref_none=%s"
                  % (ni + 1, len(picks_sorted), float(alphas[ii]), float(Res[jj]),
                     got_none, ref_none), flush=True)
            # incremental persistence: update artifact every 3 nodes
            if (ni % 3) == 2:
                payload["gpu_derived_two_domain_vs_dm_partial"] = {
                    "n_done": ni + 1, "worst_abs_error": float(worst),
                    "n_none_mismatch": int(n_none_mismatch), "rows": cmp_rows,
                }
                args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["gpu_derived_two_domain_vs_dm"] = {
        "n_subset": len(cmp_rows),
        "worst_abs_error": float(worst),
        "n_none_mismatch": int(n_none_mismatch),
        "gpu_derived_nonzero_seen": bool(saw_nonzero),
        "rows": cmp_rows,
        "note": ("GPU-derived two-domain verdict (from captured candidate sets) vs "
                 "fresh dm.discrete_mode(N=200). Nonzero worst expected (GPU refine "
                 "floor); exact 0 would mean CPU-vs-CPU. K3 territory if value/None drift."),
    }

    payload["status"] = "measured_trusted" if trusted else "measured_provisional"
    payload.pop("gpu_derived_two_domain_vs_dm_partial", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "engine_wall_time_s_total": payload["engine_wall_time_s_total"],
        "gpu_engine_status_short": payload["gpu_engine_status_short"],
        "gpu_engine_status_tall": payload["gpu_engine_status_tall"],
        "vram_short": (payload["vram_observed_short"] or {}).get("peak_used_bytes"),
        "capture_ok": capture_ok,
        "subset_worst": payload["gpu_derived_two_domain_vs_dm"]["worst_abs_error"],
        "subset_none_mismatch": payload["gpu_derived_two_domain_vs_dm"]["n_none_mismatch"],
        "subset_nonzero_seen": payload["gpu_derived_two_domain_vs_dm"]["gpu_derived_nonzero_seen"],
        "output": str(payload.get("output", args.output)),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
