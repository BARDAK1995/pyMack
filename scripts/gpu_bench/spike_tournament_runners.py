"""Per-cell method runners for the slice-02 tournament.

Each public function is a picklable worker entry that loads the (checksum-
verified) truth NPZ, assembles the cell's operators through the committed
production assemblers, runs one contender, and returns a JSON-safe row.

Bundle-RQI seeding rule (pre-registered): every corpus cell is seeded from a
full QZ "spine" at a NEARBY parameter point (one production-style grid step,
+2% in the family's sweep parameter), except the Mack fig10.4 alpha strip,
which is a genuine neighbor chain in the corpus: one anchor spine below the
strip, then cell-to-cell continuation.  Spine cost is counted honestly.
Sync-region handling is simplified to reseed-on-breakdown for this prototype
(one random reseed per lane), as permitted by the work package -- stated in
the report.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.linalg as sla

import build_truth_set as bts
import spike_tournament_core as core
from spike_tournament_families import (
    CFG,
    PencilOps,
    QepOps,
    _cpair,
    _seed_for,
    assemble_mack,
    assemble_mazhong,
    assemble_mach6,
    assemble_ozgen,
    assembly_certificate,
    candidate_indices,
    contour_mazhong_disk,
    contour_rect_spectrum,
    mack_band_mask,
    mazhong_band_mask,
    omega_from_frequency,
    ozgen_decaying,
    score_recall,
    spurious_check,
    truth_sets,
    verdict_from_candidates,
    verdict_identity,
    _mach6_case,
    solve_spatial_evp,
)

ASSEMBLY_CERT_TOL = 1e-10


# ---------------------------------------------------------------------------
# Shared loading / assembly
# ---------------------------------------------------------------------------
def _load(entry, corpus_dir):
    path = Path(corpus_dir) / entry["npz"]
    _cell, spectra, _verdict, _kappa, _cross = bts.load_npz(path)
    return spectra


def _cell_ops(entry):
    """Assemble the cell's operators (one ops per stored spectrum)."""
    kind, p = entry["kind"], entry["params"]
    ops_list, ys = [], []
    if kind == "ozgen_pair":
        for ymf in p["ymf_pair"]:
            A, B, y = assemble_ozgen(p, ymf, p["alpha"])
            ops_list.append(PencilOps(A, B))
            ys.append(y)
    elif kind == "mack_3d":
        A, B, y = assemble_mack(p, p["alpha"], p["beta"])
        ops_list.append(PencilOps(A, B))
        ys.append(y)
    elif kind == "mazhong_spatial":
        C0, C1, C2, y = assemble_mazhong(p, p["omega"])
        ops_list.append(QepOps(C0, C1, C2))
        ys.append(y)
    elif kind == "mach6_dense":
        A0, A1, A2, y = assemble_mach6(p, p["omega_L"], p["R_L"])
        ops_list.append(QepOps(A0, A1, A2))
        ys.append(y)
    else:
        raise ValueError(kind)
    return ops_list, ys


def _refine_truth(ops, vals, vecs):
    """FP64-refine stored QZ truth pairs in place (scoring apparatus, not
    method cost).  Measured motivation: QZ values carry ~2e-9 rounding on
    cluster-adjacent nonnormal modes (|dz| ~ kappa_eig * eps64 * ||A||),
    which a 1e-9 match tolerance would misread as a missed mode for BOTH
    contenders.  Raw distances are still recorded per truth mode."""
    scratch = core.CostLedger()
    refined, shifts = [], []
    for j in range(len(vals)):
        v = complex(vals[j])
        x = np.asarray(vecs[:, j])
        if not np.any(x):
            refined.append(v)
            shifts.append(None)
            continue
        if ops.kind == "pencil":
            r = core.two_sided_rqi(ops.A, ops.B, v, x, None, scratch,
                                   ops.normA, ops.normB, max_iter=3,
                                   use_fp64=True)
        else:
            r = core.bordered_newton(ops.C0, ops.C1, ops.C2, v, x, scratch,
                                     ops.norms, max_iter=3, use_fp64=True)
        shift = abs(complex(r["value"]) - v)
        if r["breakdown"] or shift > 1e-6 * max(1.0, abs(v)):
            refined.append(v)
            shifts.append(None)
        else:
            refined.append(complex(r["value"]))
            shifts.append(float(shift))
    return np.asarray(refined, dtype=complex), shifts


def _base_row(entry, spectra, ops_list):
    cert = assembly_certificate(entry, spectra, ops_list)
    if cert > ASSEMBLY_CERT_TOL:
        raise RuntimeError(
            f"{entry['id']}: assembled operators do not reproduce stored "
            f"truth pairs (relres {cert:.3e}) -- assembly binding broken")
    prod_raw, prod_vecs, inbox = truth_sets(entry, spectra)
    prod, refine_info = [], []
    for i, vals in enumerate(prod_raw):
        ops = ops_list[i if entry["kind"] == "ozgen_pair" else 0]
        ref, shifts = _refine_truth(ops, vals, prod_vecs[i])
        prod.append(ref)
        refine_info.append({
            "raw": [_cpair(v) for v in vals],
            "shifts": shifts,
            "max_shift": max((s for s in shifts if s is not None),
                             default=0.0)})
    return cert, prod, inbox, refine_info


def _sets_match(base_set, other_set, tol):
    if len(base_set) != len(other_set):
        return False
    other = np.asarray(other_set, dtype=complex)
    for z in base_set:
        if other.size == 0:
            return False
        if float(np.min(np.abs(other - z))) > tol * max(1.0, abs(z)):
            return False
    return True


# ---------------------------------------------------------------------------
# CONTOUR runner
# ---------------------------------------------------------------------------
def run_contour_cell(entry, corpus_dir):
    spectra = _load(entry, corpus_dir)
    ops_list, ys = _cell_ops(entry)
    cert, truth_prod, truth_inbox, refine_info = _base_row(entry, spectra,
                                                           ops_list)
    kind = entry["kind"]
    fam = entry["family"]
    row = {"id": entry["id"], "family": fam, "stratum": entry["stratum"],
           "method": "contour", "dim": ops_list[0].m,
           "assembly_certificate": cert,
           "truth_counts": {"prod_basis": [len(t) for t in truth_prod],
                            "in_box": [len(t) for t in truth_inbox]},
           "truth_refinement": refine_info,
           "variants": {}}
    if kind == "mazhong_spatial":
        row["variants"]["window"] = _contour_mazhong_cell(
            entry, ops_list[0], truth_prod, truth_inbox)
        return row
    fam_cfg = CFG["families"][fam]["contour"]
    for vname, v in fam_cfg["variants"].items():
        vrow = {"perturb": {}, "stability": None}
        runs = {}
        for perturb in CFG["perturbations"]:
            per_spec = []
            for i_s, ops in enumerate(ops_list):
                seed = _seed_for(entry["id"], i_s, vname)
                per_spec.append(contour_rect_spectrum(
                    entry, ops, fam_cfg, vname, v, perturb, seed,
                    want_winding=(abs(perturb - 1.0) < 1e-12)))
            runs[perturb] = per_spec
        base = runs[1.0]
        cands = [([c["value"] for c in r["certified"]],
                  (np.column_stack([c["vector"] for c in r["certified"]])
                   if r["certified"] else
                   np.zeros((ops_list[i].m, 0), dtype=complex)))
                 for i, r in enumerate(base)]
        verdict = verdict_from_candidates(entry, cands, ys)
        vid = verdict_identity(entry, verdict, CFG["recall_match_tol"])
        found = [vals for vals, _ in cands]
        recall = score_recall(found, truth_prod, CFG["recall_match_tol"])
        recall_inbox = score_recall(found, truth_inbox,
                                    CFG["recall_match_tol"])
        spurious = spurious_check(entry, found, truth_inbox,
                                  CFG["spurious_tol"])
        stab_specs = []
        for i_s in range(len(ops_list)):
            counts = {str(pf): runs[pf][i_s]["interior_count"]
                      for pf in CFG["perturbations"]}
            base_set = runs[1.0][i_s]["interior_set"]
            ok = all(_sets_match(base_set, runs[pf][i_s]["interior_set"],
                                 CFG["spurious_tol"])
                     for pf in CFG["perturbations"])
            stab_specs.append({"counts": counts, "sets_match": bool(ok),
                               "pass": bool(ok and len(set(
                                   counts.values())) == 1)})
        base_ledger = core.CostLedger()
        stab_ledger = core.CostLedger()
        for pf in CFG["perturbations"]:
            for r in runs[pf]:
                (base_ledger if abs(pf - 1.0) < 1e-12
                 else stab_ledger).merge(r["ledger"])
        vrow.update({
            "verdict": vid,
            "recall": recall,
            "recall_inbox_secondary": {
                "n_truth": recall_inbox["n_truth"],
                "n_recalled": recall_inbox["n_recalled"],
                "recall": recall_inbox["recall"]},
            "spurious": spurious,
            "certified_per_spectrum": [
                [_cpair(c["value"]) for c in r["certified"]] for r in base],
            "certified_residual_max": max(
                [c["residual"] for r in base for c in r["certified"]],
                default=None),
            "polish_stats": [r["polish_stats"] for r in base],
            "diag": [r["diag"] for r in base],
            "stability": {"per_spectrum": stab_specs,
                          "pass": bool(all(s["pass"] for s in stab_specs))},
            "ledger_base": base_ledger.summary(ops_list[0].m),
            "ledger_stability": stab_ledger.summary(ops_list[0].m),
        })
        row["variants"][vname] = vrow
    return row


def _contour_mazhong_cell(entry, ops, truth_prod, truth_inbox):
    fam_cfg = CFG["families"][entry["family"]]["contour"]
    seed = _seed_for(entry["id"], 0, "window")
    base = contour_mazhong_disk(entry, ops, fam_cfg, 1.0, seed)
    window_vals = [c["value"] for c in base["window"]]
    verdict = bts.derive_mazhong_verdict(
        np.asarray(window_vals, dtype=complex), entry["params"])
    vid = verdict_identity(entry, verdict, CFG["recall_match_tol"])
    found = [[c["value"] for c in base["certified"]]]
    recall = score_recall(found, truth_prod, CFG["recall_match_tol"])
    recall_inbox = score_recall(found, truth_inbox, CFG["recall_match_tol"])
    spurious = spurious_check(entry, found, truth_inbox, CFG["spurious_tol"])
    # Criterion (iv) for the shift-invert WINDOW mechanism.
    #
    # The production quantity is the 25 eigenvalues NEAREST the target -- a
    # DISTANCE-RANKED set, invariant by construction under any disk radius
    # that fully contains it.  So a +-20% radius perturbation cannot move the
    # window; it can only fail to ENUMERATE it (a fixed-L Beyn disk clips when
    # a +20% radius crosses the continuous-spectrum arc that sits just past
    # the window).  Hence the honest (iv) analog is enumeration completeness:
    # window_complete == the full 25-nearest were certified, so the window,
    # its band members, and the verdict are radius-stable.
    p = entry["params"]
    a0 = complex(float(p["omega"]) / float(p["c_guess"]))
    sorted_cert = sorted(base["certified"],
                         key=lambda c: abs(c["value"] - a0))
    n_base = int(fam_cfg["window_n"])
    # Diagnostic (reported, NOT the (iv) pass gate): band-member count under
    # the production mechanism's OWN parameter n_modes +-20% (20/25/30).  For
    # verdict-empty shift-invert cells this exposes the production basis's
    # inherent cutoff fragility (e.g. hc_043 gains a candidate at n=30) --
    # a property of the production basis, flagged in slice 01 as
    # box_content_contradicts_verdict, NOT a contour-method artifact.
    nmodes_counts = {}
    for pf in CFG["perturbations"]:
        n = max(1, int(round(n_base * pf)))
        wv = np.asarray([c["value"] for c in sorted_cert[:n]], dtype=complex)
        band = (mazhong_band_mask(wv, p) if wv.size
                else np.zeros(0, dtype=bool))
        nmodes_counts[str(pf)] = int(np.count_nonzero(band))
    stab = {
        "quantity": "window_complete (25-nearest fully enumerated; the "
                    "window is distance-ranked hence radius-invariant)",
        "window_complete": bool(base["window_complete"]),
        "n_certified": len(base["certified"]),
        "nmodes_pm20_band_counts": nmodes_counts,
        "nmodes_pm20_note": ("production-basis cutoff sensitivity "
                             "(diagnostic, not a (iv) input)"),
        "pass": bool(base["window_complete"]),
    }
    base_ledger = core.CostLedger()
    base_ledger.merge(base["ledger"])
    stab_ledger = core.CostLedger()  # window-size diagnostic is free
    in_box_cert = [c["value"] for c in base["certified"]
                   if bts.in_box_mask(entry, np.asarray(
                       [c["value"]]))[0]]
    return {
        "verdict": vid,
        "recall": recall,
        "recall_inbox_secondary": {"n_truth": recall_inbox["n_truth"],
                                   "n_recalled": recall_inbox["n_recalled"],
                                   "recall": recall_inbox["recall"]},
        "spurious": spurious,
        "window_values": [_cpair(z) for z in window_vals],
        "window_complete": bool(base["window_complete"]),
        "radius": base["radius"],
        "radius_trials": base["trials"],
        "box_content_certified": len(in_box_cert),
        "certified_per_spectrum": [[_cpair(c["value"])
                                    for c in base["certified"]]],
        "certified_residual_max": max(
            [c["residual"] for c in base["certified"]], default=None),
        "polish_stats": [base["polish_stats"]],
        "diag": [base["diag"]],
        "stability": {"per_spectrum": [stab], "pass": stab["pass"]},
        "ledger_base": base_ledger.summary(ops.m),
        "ledger_stability": stab_ledger.summary(ops.m),
    }


# ---------------------------------------------------------------------------
# BUNDLE-RQI runner (temporal: two-sided RQI; spatial: bordered Newton)
# ---------------------------------------------------------------------------
def _track_seed(ops, seed, ledger, rng, cfg=CFG):
    z0, x0, w0 = seed

    def run(z, x, w, use_fp64=False, steps=None):
        steps = cfg["track_max_iter"] if steps is None else steps
        if ops.kind == "pencil":
            return core.two_sided_rqi(ops.A, ops.B, z, x, w, ledger,
                                      ops.normA, ops.normB, max_iter=steps,
                                      cert_tol=cfg["cert_residual"],
                                      use_fp64=use_fp64)
        return core.bordered_newton(ops.C0, ops.C1, ops.C2, z, x, ledger,
                                    ops.norms, max_iter=steps,
                                    cert_tol=cfg["cert_residual"],
                                    use_fp64=use_fp64)

    r = run(z0, x0, w0)
    r["reseeded"] = False
    r["promoted_fp64"] = False
    if r["breakdown"] or not r["converged"]:
        x1 = (rng.standard_normal(ops.m) + 1j * rng.standard_normal(ops.m))
        w1 = (rng.standard_normal(ops.m) + 1j * rng.standard_normal(ops.m))
        r2 = run(z0, x1, w1)
        if r2["converged"] and not r2["breakdown"]:
            r2["reseeded"] = True
            r2["promoted_fp64"] = False
            r = r2
    if not r["breakdown"] and (
            r["residual"] > cfg["polish_residual_target"]
            or not r["converged"]):
        # design-E promotion rung (identical policy to the contour polish)
        r3 = run(r["value"], r["vector"], r.get("left"), use_fp64=True,
                 steps=3)
        r3["reseeded"] = r.get("reseeded", False)
        r3["promoted_fp64"] = True
        if r3["converged"] or not r["converged"]:
            r = r3
    return r


def _dedupe_results(results, cfg=CFG):
    ok = [r for r in results if r["converged"] and not r["breakdown"]]
    if not ok:
        return []
    vals = np.asarray([r["value"] for r in ok])
    keep = core.dedupe(vals, None, [r["residual"] for r in ok],
                       tol=cfg["dedupe_tol"])
    return [ok[i] for i in keep]


def _bundle_variant_summary(entry, converged_per_spec, ys, truth_prod,
                            truth_inbox, ledger, base_dim, extras=None):
    cands = []
    for i, conv in enumerate(converged_per_spec):
        vals = [r["value"] for r in conv]
        vecs = (np.column_stack([r["vector"] for r in conv]) if conv
                else np.zeros((base_dim, 0), dtype=complex))
        cands.append((vals, vecs))
    verdict = verdict_from_candidates(entry, cands, ys)
    vid = verdict_identity(entry, verdict, CFG["recall_match_tol"])
    found = [vals for vals, _ in cands]
    recall = score_recall(found, truth_prod, CFG["recall_match_tol"])
    recall_inbox = score_recall(found, truth_inbox,
                                CFG["recall_match_tol"])
    spurious = spurious_check(entry, found, truth_inbox,
                              CFG["spurious_tol"])
    out = {
        "verdict": vid,
        "recall": recall,
        "recall_inbox_secondary": {"n_truth": recall_inbox["n_truth"],
                                   "n_recalled": recall_inbox["n_recalled"],
                                   "recall": recall_inbox["recall"]},
        "spurious": spurious,
        "converged_per_spectrum": [[_cpair(r["value"]) for r in conv]
                                   for conv in converged_per_spec],
        "ledger_base": ledger.summary(base_dim),
    }
    if extras:
        out.update(extras)
    return out


def run_bundle_cell(entry, corpus_dir):
    """Bundle-RQI for all families except the Mack strip (chained task)."""
    spectra = _load(entry, corpus_dir)
    ops_list, ys = _cell_ops(entry)
    cert, truth_prod, truth_inbox, refine_info = _base_row(entry, spectra,
                                                           ops_list)
    kind, p = entry["kind"], entry["params"]
    rng = np.random.default_rng(_seed_for(entry["id"], "bundle"))
    cap = CFG["bundle_cap"]
    step = CFG["spine_param_step"]

    spine_ledger = core.CostLedger()
    seeds_per_spec = []       # full (uncapped) seed lists, verdict-ordered
    spine_info = []
    if kind == "ozgen_pair":
        for ymf in p["ymf_pair"]:
            A2, B2, y2 = assemble_ozgen(p, ymf, float(p["alpha"]) *
                                        (1.0 + step))
            w, vl, vr = sla.eig(A2, B2, left=True, right=True)
            spine_ledger.add_fact("qz", A2.shape[0])
            fin = np.isfinite(w)
            phys = (fin & (w.real > -0.5) & (w.real < 1.5)
                    & (np.abs(w.imag) < 0.5))
            idx = np.flatnonzero(phys)
            keep, _fs = ozgen_decaying(w[idx], vr[:, idx], y2, p)
            sel = idx[keep]
            order = np.argsort(-w[sel].imag)
            sel = sel[order]
            seeds_per_spec.append([(complex(w[k]), vr[:, k], vl[:, k])
                                   for k in sel])
            spine_info.append({"param": "alpha",
                               "value": float(p["alpha"]) * (1.0 + step),
                               "n_seed_candidates": int(sel.size)})
    elif kind == "mazhong_spatial":
        om_spine = float(p["omega"]) * (1.0 + step)
        if "profile" not in bts._MAZHONG_PROFILE_CACHE:
            bts._MAZHONG_PROFILE_CACHE["profile"] = bts.mzc.build_profile()
        np.random.seed(_seed_for(entry["id"], "arpack"))
        alphas, modes, _y0 = bts.solve_spatial(
            bts._MAZHONG_PROFILE_CACHE["profile"], om_spine, float(p["R"]),
            float(p["Ma"]), float(p["Pr"]), float(p["gamma"]),
            N=int(p["N"]), y_max=float(p["y_max"]),
            wall_bc=str(p["wall_bc"]),
            target_alpha=om_spine / float(p["c_guess"]) + 0j,
            n_modes=int(p["n_modes"]), length_scale="L_star",
            lambda_mu_ratio=float(p["lambda_mu_ratio"]))
        spine_ledger.add_fact("arpack", 2 * ops_list[0].m)
        alphas = np.asarray(alphas, dtype=complex)
        band = mazhong_band_mask(alphas, p, omega=om_spine)
        sel = np.flatnonzero(band)
        order = np.argsort(alphas[sel].imag)
        sel = sel[order]
        seeds_per_spec.append([(complex(alphas[k]),
                                np.asarray(modes[:, k], dtype=complex),
                                None) for k in sel])
        spine_info.append({"param": "omega", "value": om_spine,
                           "n_seed_candidates": int(sel.size),
                           "n_window": int(alphas.size)})
    elif kind == "mach6_dense":
        r_spine = float(p["R_L"]) * (1.0 + step)
        om_spine = omega_from_frequency(float(p["freq_parameter"]), r_spine,
                                        "mack")
        gas, lst_cfg, y, D, base_grid = _mach6_case(p)
        vals, vecs = solve_spatial_evp(om_spine, r_spine, y, D, base_grid,
                                       gas, lst_cfg)
        spine_ledger.add_fact("qz", 2 * ops_list[0].m)
        vals = np.asarray(vals, dtype=complex)
        idx = candidate_indices(vals, om_spine, lst_cfg)
        growth_order = np.argsort(vals[idx].imag)   # verdict = argmin a_i
        idx = idx[growth_order]
        seeds_per_spec.append([(complex(vals[k]),
                                np.asarray(vecs[:, k], dtype=complex), None)
                               for k in idx])
        spine_info.append({"param": "R_L", "value": r_spine,
                           "omega": om_spine,
                           "n_seed_candidates": int(idx.size)})
    else:
        raise ValueError(f"run_bundle_cell does not handle {kind}")

    # Track every seed once; capped/degraded = ledger + result subsets.
    cap_ledger = core.CostLedger()
    cap_ledger.merge(spine_ledger)
    all_ledger = core.CostLedger()
    all_ledger.merge(spine_ledger)
    conv_cap, conv_all, taxonomy = [], [], []
    for i_s, seeds in enumerate(seeds_per_spec):
        ops = ops_list[i_s if kind == "ozgen_pair" else 0]
        results = []
        for j, seed in enumerate(seeds):
            led = core.CostLedger()
            r = _track_seed(ops, seed, led, rng)
            if j < cap:
                cap_ledger.merge(led)
            all_ledger.merge(led)
            results.append(r)
            if r["breakdown"] or not r["converged"]:
                taxonomy.append({"spectrum": i_s, "seed": _cpair(seed[0]),
                                 "capped_lane": bool(j < cap),
                                 "type": ("breakdown" if r["breakdown"]
                                          else "unconverged"),
                                 "residual": float(r["residual"])})
        conv_cap.append(_dedupe_results(results[:cap]))
        conv_all.append(_dedupe_results(results))

    base_dim = ops_list[0].m
    row = {"id": entry["id"], "family": entry["family"],
           "stratum": entry["stratum"], "method": "bundle_rqi",
           "dim": base_dim, "assembly_certificate": cert,
           "truth_counts": {"prod_basis": [len(t) for t in truth_prod],
                            "in_box": [len(t) for t in truth_inbox]},
           "truth_refinement": refine_info,
           "spine": {"rule": f"+{step:.0%} nearby-parameter full spine",
                     "shared_between_capped_and_degraded": True,
                     "info": spine_info},
           "taxonomy": taxonomy}
    row["capped"] = _bundle_variant_summary(
        entry, conv_cap, ys, truth_prod, truth_inbox, cap_ledger, base_dim,
        extras={"bundle_cap": cap,
                "n_seeds": [min(len(s), cap) for s in seeds_per_spec]})
    row["degraded_uncapped"] = _bundle_variant_summary(
        entry, conv_all, ys, truth_prod, truth_inbox, all_ledger, base_dim,
        extras={"n_seeds": [len(s) for s in seeds_per_spec]})
    return row


def run_mack_chain(entries, corpus_dir):
    """Bundle-RQI over the Mack fig10.4 alpha strip as a neighbor chain:
    one anchor spine below the strip, then cell-to-cell continuation with a
    3-8 mode bundle (capped variant) and an uncapped diagnostic chain."""
    entries = sorted(entries, key=lambda e: float(e["params"]["alpha"]))
    p0 = entries[0]["params"]
    alpha_anchor = float(p0["alpha"]) * (1.0 - CFG["spine_param_step"])
    beta_anchor = alpha_anchor * math.tan(math.radians(float(
        p0["psi_deg"])))
    A, B, _y = assemble_mack(p0, alpha_anchor, beta_anchor)
    spine_ledger = core.CostLedger()
    w, vl, vr = sla.eig(A, B, left=True, right=True)
    spine_ledger.add_fact("qz", A.shape[0])
    band = mack_band_mask(w, p0)
    sel = np.flatnonzero(band)
    sel = sel[np.argsort(-w[sel].imag)]
    seeds_all = [(complex(w[k]), vr[:, k], vl[:, k]) for k in sel]
    cap = CFG["bundle_cap"]
    anchor_info = {"alpha": alpha_anchor, "beta": beta_anchor,
                   "n_band_seeds": len(seeds_all)}

    states = {"capped": seeds_all[:cap], "degraded_uncapped": list(seeds_all)}
    ledgers = {"capped": core.CostLedger(),
               "degraded_uncapped": core.CostLedger()}
    for led in ledgers.values():
        led.merge(spine_ledger)   # anchor shared; noted in the report
    rows = {}
    for entry in entries:
        spectra = _load(entry, corpus_dir)
        ops_list, ys = _cell_ops(entry)
        cert, truth_prod, truth_inbox, refine_info = _base_row(
            entry, spectra, ops_list)
        ops = ops_list[0]
        rng = np.random.default_rng(_seed_for(entry["id"], "chain"))
        row = {"id": entry["id"], "family": entry["family"],
               "stratum": entry["stratum"], "method": "bundle_rqi",
               "dim": ops.m, "assembly_certificate": cert,
               "truth_counts": {"prod_basis": [len(t) for t in truth_prod],
                                "in_box": [len(t) for t in truth_inbox]},
               "truth_refinement": refine_info,
               "spine": {"rule": "anchor QZ below strip + neighbor chain",
                         "info": [anchor_info]},
               "taxonomy": []}
        for variant in ("capped", "degraded_uncapped"):
            led = core.CostLedger()
            results = []
            respined = False
            if not states[variant]:
                # emergency re-spine at this cell (counted, loud)
                w2, vl2, vr2 = sla.eig(ops.A, ops.B, left=True, right=True)
                led.add_fact("qz", ops.m)
                band2 = mack_band_mask(w2, entry["params"])
                sel2 = np.flatnonzero(band2)
                sel2 = sel2[np.argsort(-w2[sel2].imag)]
                states[variant] = [(complex(w2[k]), vr2[:, k], vl2[:, k])
                                   for k in sel2]
                if variant == "capped":
                    states[variant] = states[variant][:cap]
                respined = True
            for seed in states[variant]:
                r = _track_seed(ops, seed, led, rng)
                results.append(r)
                if r["breakdown"] or not r["converged"]:
                    row["taxonomy"].append(
                        {"variant": variant, "seed": _cpair(seed[0]),
                         "type": ("breakdown" if r["breakdown"]
                                  else "unconverged"),
                         "residual": float(r["residual"])})
            conv = _dedupe_results(results)
            ledgers[variant].merge(led)
            summary = _bundle_variant_summary(
                entry, [conv], ys, truth_prod, truth_inbox, led, ops.m,
                extras={"n_seeds": [len(states[variant])],
                        "respined": respined,
                        **({"bundle_cap": cap} if variant == "capped"
                           else {})})
            row[variant] = summary
            # next state: converged modes still in this cell's band,
            # verdict-ordered
            vals = np.asarray([r["value"] for r in conv], dtype=complex)
            keep = np.flatnonzero(mack_band_mask(vals, entry["params"])) \
                if vals.size else np.zeros(0, dtype=int)
            keep = keep[np.argsort(-vals[keep].imag)] if keep.size else keep
            nxt = [(complex(vals[k]), conv[k]["vector"], conv[k]["left"])
                   for k in keep]
            states[variant] = nxt[:cap] if variant == "capped" else nxt
        rows[entry["id"]] = row
    # attach chain-total ledgers to the last row for family accounting
    last_id = entries[-1]["id"]
    for variant in ("capped", "degraded_uncapped"):
        rows[last_id][variant]["chain_total_ledger"] = \
            ledgers[variant].summary(rows[last_id]["dim"])
    return rows
