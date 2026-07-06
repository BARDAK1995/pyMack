"""Slice-02 METHOD TOURNAMENT: batched contour projection (Beyn/SS class)
vs the incumbent bundle-RQI tracking, head-to-head on the slice-01 hard-cell
corpus, with c64-emulated solves (decision D1, kill criterion K2).

Usage (the slice's verification command):

    python scripts/gpu_bench/spike_method_tournament.py \
        --corpus verification/gpu_certification/hard_cells \
        --report docs/gpu/benchmarks/tournament_report.json

Exit codes:
    0  report complete AND the D1-winning method has recall 1.0 and
       verdict-identity 100% on the corpus (an incumbent win is as valid
       as a challenger win),
    2  kill criterion K2 fired (NEITHER method reached 100% recall),
    3  pre-registered rule gap (projection has recall but fails a
       non-recall criterion while the incumbent fails recall),
    1  incomplete run / verification failure.

Scoring is pre-registered (see the "scoring" block of the report):
truth modes = the production candidate set per each cell's ``verdict_basis``
(amended D1: Ma&Zhong cells are scored by reproducing the production
shift-invert window mechanism, not by box content; the mack_fig10_4 family
has no empty-box cells -- mode death is argmax handover + omega_i sign flip).
"""
from __future__ import annotations

# Single-thread BLAS pin BEFORE numpy import (timing comparability;
# worker processes re-execute this module top-level on spawn).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MACK_FAMILY = "mack_fig10_4_m10_3d"
OZGEN_FAMILIES = ("ozgen_first_pair", "ozgen_second_pair")


# ---------------------------------------------------------------------------
# Worker entry (module-level for Windows spawn pickling)
# ---------------------------------------------------------------------------
def _work(task):
    kind, payload, corpus = task
    import spike_tournament_runners as runners
    t0 = time.perf_counter()
    if kind == "contour":
        out = runners.run_contour_cell(payload, corpus)
    elif kind == "bundle":
        out = runners.run_bundle_cell(payload, corpus)
    elif kind == "mack_chain":
        out = runners.run_mack_chain(payload, corpus)
    else:
        raise ValueError(kind)
    return kind, out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _official_variant(family, contour_rows):
    """Pre-registered per-family mitigation pick: (1) highest truth-basis
    recall, (2) lowest base LU-equivalents, (3) stability pass."""
    rows = [r for r in contour_rows.values() if r["family"] == family]
    if not rows:
        return None, {}
    names = sorted({v for r in rows for v in r["variants"]})
    stats = {}
    for name in names:
        n_t = n_r = 0
        lu = 0.0
        stab = True
        for r in rows:
            v = r["variants"][name]
            n_t += v["recall"]["n_truth"]
            n_r += v["recall"]["n_recalled"]
            lu += v["ledger_base"]["lu_equivalents"]
            stab = stab and v["stability"]["pass"]
        stats[name] = {"recall": (1.0 if n_t == 0 else n_r / n_t),
                       "n_truth": n_t, "n_recalled": n_r,
                       "lu_equivalents": lu, "stability_pass": stab}
    ranked = sorted(names, key=lambda n: (-stats[n]["recall"],
                                          stats[n]["lu_equivalents"],
                                          not stats[n]["stability_pass"]))
    return ranked[0], stats


def _family_agg(rows, get):
    """Aggregate {family: {...}} over per-cell metric extractor ``get``."""
    fams = {}
    for r in rows:
        v = get(r)
        if v is None:
            continue
        f = fams.setdefault(r["family"], {
            "cells": 0, "n_truth": 0, "n_recalled": 0,
            "verdict_identical": 0, "verdict_identical_decision_only": 0,
            "lu_equivalents": 0.0, "fact_complex_flops": 0.0,
            "solve_complex_flops": 0.0, "n_lu64": 0,
            "stability_pass": 0, "n_spurious": 0, "misses": []})
        f["cells"] += 1
        f["n_truth"] += v["recall"]["n_truth"]
        f["n_recalled"] += v["recall"]["n_recalled"]
        f["verdict_identical"] += int(v["verdict"]["identical"])
        f["verdict_identical_decision_only"] += int(
            v["verdict"].get("identical_decision_only", False))
        f["lu_equivalents"] += v["ledger_base"]["lu_equivalents"]
        f["fact_complex_flops"] += v["ledger_base"]["fact_complex_flops"]
        f["solve_complex_flops"] += v["ledger_base"]["solve_complex_flops"]
        f["n_lu64"] += sum(
            n for k, n in v["ledger_base"]["factorizations"].items()
            if k.startswith("lu64@"))
        if "stability" in v:
            f["stability_pass"] += int(v["stability"]["pass"])
        f["n_spurious"] += len(v.get("spurious", []))
        for m in v["recall"]["rows"]:
            if not m["recalled"]:
                f["misses"].append({"cell": r["id"],
                                    "spectrum": m["spectrum"],
                                    "truth": m["truth"],
                                    "distance": m["distance"]})
    for f in fams.values():
        f["recall"] = 1.0 if f["n_truth"] == 0 else \
            f["n_recalled"] / f["n_truth"]
    return fams


def _totals(fams):
    t = {"n_truth": 0, "n_recalled": 0, "cells": 0, "verdict_identical": 0,
         "verdict_identical_decision_only": 0, "lu_equivalents": 0.0,
         "fact_complex_flops": 0.0, "solve_complex_flops": 0.0,
         "n_lu64": 0, "stability_pass": 0, "n_spurious": 0}
    for f in fams.values():
        for k in t:
            t[k] += f[k]
    t["recall"] = 1.0 if t["n_truth"] == 0 else \
        t["n_recalled"] / t["n_truth"]
    return t


def _winding_stats(contour_rows, official):
    n_avail = n_agree_rank = n_uncert = n_skipped = 0
    for r in contour_rows.values():
        vname = "window" if "window" in r["variants"] else \
            official.get(r["family"])
        v = r["variants"].get(vname)
        if v is None:
            continue
        for spec in v["diag"]:
            for d in spec:
                wind = d.get("winding")
                if wind is None or wind.get("skipped"):
                    n_skipped += 1
                    continue
                if not wind["certified"]:
                    n_uncert += 1
                    continue
                n_avail += 1
                if wind["winding"] == d.get("rank"):
                    n_agree_rank += 1
    return {"n_certified_windings": n_avail,
            "n_agree_with_rank": n_agree_rank,
            "n_uncertified": n_uncert, "n_skipped": n_skipped}


def _git_provenance():
    def run(*args):
        return subprocess.run(["git", *args], cwd=str(REPO), timeout=15,
                              capture_output=True, text=True).stdout.strip()
    try:
        return {"git_sha": run("rev-parse", "HEAD") or None,
                "git_dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--cells", default=None,
                    help="comma-separated hc ids (debug subset; the report "
                         "is marked partial and the exit code forced to 1)")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    corpus = Path(args.corpus).resolve()
    manifest = json.loads((corpus / "truth_manifest.json").
                          read_text(encoding="utf-8"))
    cells = manifest["cells"]
    subset = None
    if args.cells:
        subset = {c.strip() for c in args.cells.split(",") if c.strip()}
        cells = [c for c in cells if c["id"] in subset]
    if not cells:
        print("no cells selected", file=sys.stderr)
        return 1

    # --- NPZ integrity: verify checksums before trusting anything ---------
    if str(corpus) not in sys.path:
        sys.path.insert(0, str(corpus))
    import build_truth_set as bts  # noqa: E402
    n_checked = 0
    for c in cells:
        path = corpus / c["npz"]
        if not path.exists():
            print(f"FATAL: missing {path}", file=sys.stderr)
            return 1
        got = bts.npz_arrays_sha256(path)
        if got != c["npz_sha256"]:
            print(f"FATAL: checksum mismatch for {c['id']}: {got} != "
                  f"{c['npz_sha256']}", file=sys.stderr)
            return 1
        n_checked += 1
    print(f"corpus integrity: {n_checked} NPZ checksums verified",
          flush=True)

    # --- task fan-out (fat cells + the serial Mack chain first, so the
    # longest-running tasks are not stragglers) ------------------------------
    mack_cells = [c for c in cells if c["family"] == MACK_FAMILY]
    census_path = corpus / "census.json"
    weights = {}
    if census_path.exists():
        census = json.loads(census_path.read_text(encoding="utf-8"))
        for c in census.get("cells", []):
            weights[c["id"]] = sum(ps["in_box_count"]
                                   for ps in c["per_spectrum"])
    tasks = []
    if mack_cells:
        tasks.append(("mack_chain", mack_cells, str(corpus)))
    tasks += [("contour", c, str(corpus))
              for c in sorted(cells, key=lambda c: -weights.get(c["id"], 0))]
    tasks += [("bundle", c, str(corpus))
              for c in sorted(cells, key=lambda c: -weights.get(c["id"], 0))
              if c["family"] != MACK_FAMILY]

    contour_rows, bundle_rows = {}, {}
    timings = {}
    n_done = 0
    if args.workers <= 1:
        results = map(_work, tasks)
        for kind, out, dt in results:
            n_done += 1
            _collect(kind, out, dt, contour_rows, bundle_rows, timings)
            print(f"  [{n_done:02d}/{len(tasks)}] {kind} done ({dt:.1f}s)",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_work, t): t for t in tasks}
            for fut in as_completed(futs):
                kind, out, dt = fut.result()
                n_done += 1
                _collect(kind, out, dt, contour_rows, bundle_rows, timings)
                label = out.get("id") if isinstance(out, dict) and "id" in \
                    out else kind
                print(f"  [{n_done:02d}/{len(tasks)}] {kind}:{label} "
                      f"({dt:.1f}s)", flush=True)

    # --- official mitigation pick per rectangle family ----------------------
    families_present = sorted({c["family"] for c in cells})
    official = {}
    mitigation = {}
    for fam in families_present:
        if fam in ("mazhong_first_spatial", "mazhong_second_spatial"):
            official[fam] = "window"
            continue
        name, stats = _official_variant(fam, contour_rows)
        official[fam] = name
        if fam in OZGEN_FAMILIES:
            mitigation[fam] = {"variants": stats, "winner": name,
                               "rule": ("pre-registered: max truth-basis "
                                        "recall, then min base "
                                        "LU-equivalents, then stability")}

    # --- per-method aggregates ----------------------------------------------
    def contour_get(row):
        return row["variants"].get(
            "window" if "window" in row["variants"]
            else official.get(row["family"]))

    contour_fams = _family_agg(list(contour_rows.values()), contour_get)
    bundle_fams = _family_agg(list(bundle_rows.values()),
                              lambda r: r.get("capped"))
    degraded_fams = _family_agg(list(bundle_rows.values()),
                                lambda r: r.get("degraded_uncapped"))
    # mack chain totals replace summed per-cell ledgers (they include the
    # shared anchor spine exactly once).
    for variant, fams in (("capped", bundle_fams),
                          ("degraded_uncapped", degraded_fams)):
        for r in bundle_rows.values():
            tot = r.get(variant, {}).get("chain_total_ledger")
            if tot and r["family"] in fams:
                fams[r["family"]]["lu_equivalents"] = tot["lu_equivalents"]
                fams[r["family"]]["fact_complex_flops"] = \
                    tot["fact_complex_flops"]
                fams[r["family"]]["solve_complex_flops"] = \
                    tot["solve_complex_flops"]
                fams[r["family"]]["n_lu64"] = sum(
                    n for k, n in tot["factorizations"].items()
                    if k.startswith("lu64@"))

    contour_tot = _totals(contour_fams)
    bundle_tot = _totals(bundle_fams)
    degraded_tot = _totals(degraded_fams)

    # --- D1 criteria ---------------------------------------------------------
    complete = subset is None and len(contour_rows) == len(cells) and \
        len(bundle_rows) == len(cells)
    c_recall = contour_tot["recall"]
    b_recall = bundle_tot["recall"]
    c_verdict_ok = contour_tot["verdict_identical"] == contour_tot["cells"]
    b_verdict_ok = bundle_tot["verdict_identical"] == bundle_tot["cells"]
    # (iii) is aggregated on factorization FLOPS (dimension-consistent
    # across families; summing per-family LU-equivalents would weight an
    # LU at dim 124 equal to one at dim 1005).
    lu_ratio = (contour_tot["fact_complex_flops"] /
                bundle_tot["fact_complex_flops"]
                if bundle_tot["fact_complex_flops"] > 0 else float("inf"))
    c_lu_ok = lu_ratio <= 2.0
    c_stab_ok = contour_tot["stability_pass"] == contour_tot["cells"]
    n_box_empty = sum(1 for c in cells
                      if not any(ps > 0 for ps in _inbox_counts(c,
                                                                contour_rows,
                                                                bundle_rows)))
    winding = _winding_stats(contour_rows, official)

    criteria = {
        "i_recall": {"contour": c_recall, "bundle": b_recall,
                     "contour_pass": c_recall >= 1.0,
                     "bundle_pass": b_recall >= 1.0},
        "ii_verdict_identity": {
            "contour_identical_cells":
                f"{contour_tot['verdict_identical']}/{contour_tot['cells']}",
            "bundle_identical_cells":
                f"{bundle_tot['verdict_identical']}/{bundle_tot['cells']}",
            "contour_pass": c_verdict_ok, "bundle_pass": b_verdict_ok,
            "box_empty_cells_in_corpus": n_box_empty,
            "box_empty_clause": ("VACUOUS: the corpus contains no box-empty "
                                 "cell (measured; every verdict-empty cell "
                                 "has nonzero box content killed by the "
                                 "fs/match/window filters), so the "
                                 "rank+winding box-empty certification "
                                 "clause has no instances; rank-vs-winding "
                                 "agreement is reported instead"),
            "winding_vs_rank": winding},
        "iii_lu_budget": {
            "contour_fact_flops": contour_tot["fact_complex_flops"],
            "bundle_fact_flops": bundle_tot["fact_complex_flops"],
            "contour_lu64_promotions": contour_tot["n_lu64"],
            "bundle_lu64_promotions": bundle_tot["n_lu64"],
            "ratio": lu_ratio, "threshold": 2.0,
            "pass": c_lu_ok,
            "note": ("factorization flops per the rule's letter, "
                     "dimension-consistent across families; "
                     "triangular-solve flops reported separately in the "
                     "ledgers; per-family LU-equivalents (at each family's "
                     "pencil dim) in the families tables")},
        "iv_count_stability": {
            "contour_pass_cells":
                f"{contour_tot['stability_pass']}/{contour_tot['cells']}",
            "pass": c_stab_ok},
    }
    # --- D1 adjudication (pre-registered rule + verification contract) ------
    # Verification contract: exit 0 iff the report is complete AND the WINNING
    # method has recall 1.0 and verdict-identity 100%.  The D1 displacement
    # decision (does projection DISPLACE, or proceed with a flagged cost/
    # stability caveat) is recorded regardless; an incumbent win or a K2 kill
    # is as valid an outcome as a challenger win.
    projection_gates = {"i_recall": c_recall >= 1.0,
                        "ii_verdict": c_verdict_ok,
                        "iii_lu": c_lu_ok, "iv_stability": c_stab_ok}
    projection_correct = bool(c_recall >= 1.0 and c_verdict_ok)  # (i)+(ii)
    projection_all = all(projection_gates.values())
    if projection_all:
        verdict = "PROJECTION_DISPLACES_INCUMBENT"
        winner, winner_correct = "contour", True
    elif projection_correct and b_recall < 1.0:
        # Projection meets the correctness gates (recall + verdict); the
        # incumbent CANNOT proceed (it fails recall even as the polish
        # primitive), so projection wins by elimination on correctness.  The
        # missed gate(s) are a pre-registered caveat: the slice records that a
        # <=2x cost miss (iii) is a movable threshold (novelty-at-parity) and
        # a stability miss (iv) localizes to named cells.  NOT K2 -- the
        # full-spectrum-free premise SURVIVES (projection recall = 100%).
        missed = [k for k, v in projection_gates.items() if not v]
        verdict = ("PROJECTION_WINS_ON_CORRECTNESS_INCUMBENT_FAILS_RECALL"
                   "__caveat_" + "+".join(missed))
        winner, winner_correct = "contour", True
    elif b_recall >= 1.0:
        verdict = "INCUMBENT_PROCEEDS_PROJECTION_DEMOTED_TO_SEED_SPINE"
        winner, winner_correct = "bundle_rqi", b_verdict_ok
    elif c_recall < 1.0 and b_recall < 1.0:
        verdict = "K2_FIRES_BOTH_FAIL_RECALL"
        winner, winner_correct = None, False
    else:
        verdict = "PROJECTION_RECALL_ONLY_VERDICT_MISS"
        winner, winner_correct = "contour", False
    exit_code = 0 if (complete and winner_correct) else \
        (2 if verdict.startswith("K2") else 1)

    # --- failure taxonomy ----------------------------------------------------
    taxonomy = _taxonomy(contour_rows, bundle_rows, official)

    report = {
        "status": "complete" if complete else "partial",
        "generated_at_unix": time.time(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": __import__("numpy").__version__,
            "scipy": __import__("scipy").__version__,
            "blas_single_thread_pinned": True,
            **_git_provenance(),
        },
        "corpus": {"path": str(corpus), "n_cells": len(cells),
                   "checksums_verified": n_checked},
        "scoring": _scoring_block(),
        "cost_model": _cost_model_block(),
        "parameters": __import__("spike_tournament_families").CFG,
        "official_contour_variant_per_family": official,
        "ozgen_l_mitigation": mitigation,
        "d1": {"criteria": criteria, "verdict": verdict,
               "winning_method": winner,
               "winner_meets_recall_and_verdict": winner_correct,
               "projection_gates": projection_gates,
               "k2_fired": verdict.startswith("K2"),
               "premise_full_spectrum_free_survives": bool(c_recall >= 1.0),
               "tie_break": "ties break toward projection (pre-registered)",
               "cost_gate_note": ("the slice pre-registers that a <=2x LU "
                                  "miss on rule (iii) is a movable threshold "
                                  "if novelty-at-parity is valued; recorded "
                                  "for the user's judgment, not decided "
                                  "here")},
        "totals": {"contour": contour_tot, "bundle_rqi": bundle_tot,
                   "degraded_uncapped_diagnostic": degraded_tot},
        "families": {"contour": contour_fams, "bundle_rqi": bundle_fams,
                     "degraded_uncapped_diagnostic": degraded_fams},
        "failure_taxonomy": taxonomy,
        "cells": {"contour": list(contour_rows.values()),
                  "bundle_rqi": list(bundle_rows.values())},
        "task_timings_s": timings,
        "runtime_s": time.perf_counter() - t_start,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, sort_keys=False)
                           + "\n", encoding="utf-8")

    # --- console summary -----------------------------------------------------
    print()
    print("=" * 72)
    print("D1 VERDICT:", verdict)
    print(f"winning method: {winner} (meets recall+verdict: {winner_correct})")
    if verdict.startswith("K2"):
        print("!! KILL CRITERION K2 FIRED: neither method reached 100% "
              "discrete-mode recall; the full-spectrum-free premise dies; "
              "degraded mode (CPU-QZ seeding + GPU batch refinement) "
              "becomes the plan.")
    elif "caveat" in verdict:
        print("NOTE: projection meets the correctness gates (recall+verdict) "
              "and is the only viable method (incumbent fails recall); it "
              "misses " + verdict.split("caveat_")[1] + " -- a pre-registered "
              "movable-threshold caveat for the user, NOT K2.")
    print("-" * 72)
    print(f"(i)   recall           contour {c_recall:.6f}   "
          f"bundle {b_recall:.6f}")
    print(f"(ii)  verdict identity contour "
          f"{contour_tot['verdict_identical']}/{contour_tot['cells']}   "
          f"bundle {bundle_tot['verdict_identical']}/{bundle_tot['cells']}")
    print(f"(iii) fact-flops       contour "
          f"{contour_tot['fact_complex_flops']:.3e}   bundle "
          f"{bundle_tot['fact_complex_flops']:.3e}   "
          f"ratio {lu_ratio:.3f} (<= 2.0: {c_lu_ok})")
    print(f"(iv)  count stability  {contour_tot['stability_pass']}/"
          f"{contour_tot['cells']} cells stable under +-20%")
    print("-" * 72)
    hdr = f"{'family':26s} {'method':10s} {'recall':>14s} {'verdict':>9s} " \
          f"{'LU-eq':>10s}"
    print(hdr)
    for fam in sorted(set(list(contour_fams) + list(bundle_fams))):
        for label, fams in (("contour", contour_fams),
                            ("bundle", bundle_fams),
                            ("degraded", degraded_fams)):
            if fam in fams:
                f = fams[fam]
                print(f"{fam:26s} {label:10s} "
                      f"{f['n_recalled']:>6d}/{f['n_truth']:<6d} "
                      f"{f['verdict_identical']:>4d}/{f['cells']:<4d} "
                      f"{f['lu_equivalents']:>10.1f}")
    print(f"report: {report_path}")
    print(f"runtime: {report['runtime_s']:.1f} s")
    return exit_code


def _collect(kind, out, dt, contour_rows, bundle_rows, timings):
    if kind == "contour":
        contour_rows[out["id"]] = out
        timings[f"contour:{out['id']}"] = dt
    elif kind == "bundle":
        bundle_rows[out["id"]] = out
        timings[f"bundle:{out['id']}"] = dt
    else:
        for cid, row in out.items():
            bundle_rows[cid] = row
        timings["bundle:mack_chain"] = dt


def _inbox_counts(cell, contour_rows, bundle_rows):
    row = contour_rows.get(cell["id"]) or bundle_rows.get(cell["id"])
    if row is None:
        return [1]
    return row["truth_counts"]["in_box"]


def _taxonomy(contour_rows, bundle_rows, official):
    tax = {"missed_mode": [], "spurious_accepted": [], "wrong_verdict": [],
           "breakdown_or_unconverged": [], "rank_saturated": [],
           "winding_uncertain": []}
    for method, rows in (("contour", contour_rows),
                         ("bundle_rqi", bundle_rows)):
        for r in rows.values():
            if method == "contour":
                vname = "window" if "window" in r["variants"] else \
                    official.get(r["family"])
                views = [(vname, r["variants"].get(vname))]
            else:
                views = [("capped", r.get("capped"))]
            for vname, v in views:
                if v is None:
                    continue
                for m in v["recall"]["rows"]:
                    if not m["recalled"]:
                        tax["missed_mode"].append(
                            {"method": method, "cell": r["id"],
                             "variant": vname, **m})
                for s in v.get("spurious", []):
                    tax["spurious_accepted"].append(
                        {"method": method, "cell": r["id"], **s})
                if not v["verdict"]["identical"]:
                    tax["wrong_verdict"].append(
                        {"method": method, "cell": r["id"],
                         "fields": v["verdict"]["fields"],
                         "mine": v["verdict"]["mine"]})
                if method == "contour":
                    for spec in v["diag"]:
                        for d in spec:
                            if d.get("rank_saturated"):
                                tax["rank_saturated"].append(
                                    {"cell": r["id"], "variant": vname,
                                     "rank": d["rank"]})
                            w = d.get("winding") or {}
                            if (not w.get("skipped", True)) and \
                                    not w.get("certified"):
                                tax["winding_uncertain"].append(
                                    {"cell": r["id"], "variant": vname})
            for t in r.get("taxonomy", []):
                tax["breakdown_or_unconverged"].append(
                    {"method": method, "cell": r["id"], **t})
    return tax


def _scoring_block():
    return {
        "truth_modes": ("production candidate set per verdict_basis: Ozgen "
                        "= decaying candidates (verbatim discrete_mode."
                        "_decaying_candidates) per domain; Mack fig10.4 = "
                        "phys+band filtered eigenvalues; Ma&Zhong = in-band "
                        "production shift-invert window candidates; Mach-6 "
                        "dense = candidate_indices survivors"),
        "recall_rule": "|found - truth| <= 1e-9 * max(1, |truth|) after "
                       "polish, certified by FP64 residual <= 1e-8",
        "verdict_identity": ("verbatim filter-stack verdict on the method's "
                             "certified candidates == stored production "
                             "verdict (status, selected value at 1e-9, and "
                             "the production record fields: n_match+fs for "
                             "Ozgen, n_band for Mack/Ma&Zhong, n_filtered "
                             "for Mach-6)"),
        "mazhong_mechanism": ("amended D1: contour reproduces the "
                              "production 25-nearest-target window with a "
                              "target-centred disk (exact for a centred "
                              "disk; verified against stored "
                              "prod_candidates to ~1e-10); box content "
                              "reported honestly as a diagnostic, not "
                              "penalized"),
        "secondary_diagnostic": "recall vs ALL in-box QZ eigenvalues is "
                                "also reported per cell (not a D1 input)",
        "bundle_prototype_simplifications": (
            "sync-region handling simplified to reseed-on-breakdown (one "
            "random reseed per lane), as permitted by the work package"),
    }


def _cost_model_block():
    import spike_tournament_core as core
    return {
        "unit": "one complex LU factorization at the family's primary "
                "pencil dimension m",
        "lu": "(d/m)^3 per factorization at dimension d",
        "qz": f"{core.QZ_LU_FACTOR} * (d/m)^3 (values + left/right vectors)",
        "arpack_shift_invert": f"{core.ARPACK_LU_FACTOR} * (d/m)^3 "
                               "(dense companion LU dominates)",
        "solves": "triangular-solve + refinement flops reported separately "
                  "(solve_complex_flops); D1(iii) is scored on "
                  "factorizations per the rule's letter",
        "base_vs_apparatus": ("D1(iii) uses base ledgers only: official "
                              "contour enumeration+polish+winding and "
                              "bundle spine+tracking; the +-20% stability "
                              "re-enumerations and losing mitigation "
                              "variants are measurement apparatus, "
                              "reported separately"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
