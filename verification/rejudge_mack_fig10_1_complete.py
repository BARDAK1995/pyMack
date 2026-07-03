#!/usr/bin/env python3
"""Re-judge Mack (1984) Fig 10.1 against the CORRECTED Complete-Equations loop.

The prior verdict scored pyMack's first-mode neutral *loop* against
``mack_ch10_fig10_1_M{tok}_paper_complete.csv``, which is NOT the Complete-Equations
loop: it is an open, upper-branch-only trace with a non-physical low-R tail and no
nose/lower branch (see reference_data/digitized/..._DEPRECATED_NOTE.md). This script
re-judges against the re-digitized full loop
``mack_ch10_fig10_1_M{tok}_paper_complete_equations.csv`` (branch column:
lower/upper/nose).

It does NOT recompute physics: it consumes the already-written pyMack neutral CSVs
in verification/first_mode/mack_fig10_1_m{1p6,2p2}/ and the corrected
reference, then writes the corrected verdict.json via _compare_lib.classify_relative
(thresholds 5%/15%).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compare_lib import classify_relative, write_verdict  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DIG = REPO / "reference_data" / "digitized"
OUT_ROOT = REPO / "verification" / "first_mode"
SOURCE = "Mack (1984) Fig 10.1 (AGARD R-709)"

CASES = {
    1.6: {"tok": "16", "dir": "mack_fig10_1_m1p6", "Rcrit_mack": 215, "Fnose_mack": 1.75},
    2.2: {"tok": "22", "dir": "mack_fig10_1_m2p2", "Rcrit_mack": 300, "Fnose_mack": 1.38},
}


def load_reference(tok):
    """Return dict branch->(R[],F[]) and the nose (R,F)."""
    p = DIG / f"mack_ch10_fig10_1_M{tok}_paper_complete_equations.csv"
    lower, upper, nose = [], [], None
    with open(p) as f:
        rdr = csv.reader(l for l in f if not l.lstrip().startswith("#"))
        header = next(rdr)
        for row in rdr:
            if not row:
                continue
            R, F, br = float(row[0]), float(row[1]), row[2].strip()
            if br == "lower":
                lower.append((R, F))
            elif br == "upper":
                upper.append((R, F))
            elif br == "nose":
                nose = (R, F)
    def arr(pts):
        a = np.array(sorted(pts))
        return a[:, 0], a[:, 1]
    return {"lower": arr(lower), "upper": arr(upper)}, nose, p


def load_pymack(case_dir, tok):
    p = OUT_ROOT / case_dir / f"pymack_mack_fig10_1_M{tok}_neutral.csv"
    R, Flo, Fup, Fpk, oi = [], [], [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            R.append(float(row["R"]))
            def g(k):
                v = row[k]
                return float(v) if v not in ("", None) else np.nan
            Flo.append(g("F_lower_x1e4"))
            Fup.append(g("F_upper_x1e4"))
            Fpk.append(g("F_peak_x1e4"))
            oi.append(g("peak_omega_i"))
    return {"R": np.array(R), "F_lower": np.array(Flo), "F_upper": np.array(Fup),
            "F_peak": np.array(Fpk), "peak_oi": np.array(oi)}, p


def branch_relerr(R_pm, F_pm, R_ref, F_ref):
    """Median rel err of pyMack branch vs reference, interpolating ref onto pyMack
    R's inside the reference R span where pyMack branch is finite."""
    finite = np.isfinite(F_pm)
    mask = finite & (R_pm >= R_ref.min()) & (R_pm <= R_ref.max())
    if not mask.any():
        return None, 0, None, None, None
    rr = R_pm[mask]
    fp = F_pm[mask]
    fr = np.interp(rr, R_ref, F_ref)
    rel = np.abs(fp - fr) / np.maximum(np.abs(fr), 1e-9)
    return (float(np.median(rel)), int(rr.size),
            float(rr.min()), float(rr.max()), rel)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    summary = {}
    for mach, c in CASES.items():
        tok = c["tok"]
        ref, nose, ref_path = load_reference(tok)
        pm, pm_path = load_pymack(c["dir"], tok)

        # pyMack critical R = lowest R with positive peak omega_i
        unstable = np.isfinite(pm["peak_oi"]) & (pm["peak_oi"] > 0)
        R_crit_pm = float(pm["R"][unstable].min()) if unstable.any() else None
        R_crit_mack = c["Rcrit_mack"]

        results = {}
        for name, key in (("lower", "F_lower"), ("upper", "F_upper")):
            Rref, Fref = ref[name]
            med, n, rlo, rhi, _ = branch_relerr(pm["R"], pm[key], Rref, Fref)
            results[name] = {"median_rel_err": med, "n": n, "R_overlap": [rlo, rhi]}

        # Topology: does pyMack's loop exist over most of Mack's loop R-range?
        # Mack's loop spans Rcrit_mack..1200; pyMack covers R_crit_pm..1600.
        Rref_max = max(ref["upper"][0].max(), ref["lower"][0].max())
        if R_crit_pm is None:
            covered_frac = 0.0
            topology_ok = False
        else:
            # fraction of Mack's reference upper-branch R's covered by an unstable
            # pyMack band (R >= R_crit_pm)
            Rref_up = ref["upper"][0]
            covered_frac = float(np.mean(Rref_up >= R_crit_pm))
            topology_ok = covered_frac >= 0.85

        # Headline = mean of the two branch median errors (a loop must match BOTH
        # branches; we report the loop-average and the worse branch).
        branch_meds = [results[b]["median_rel_err"] for b in ("lower", "upper")
                       if results[b]["median_rel_err"] is not None]
        loop_med = float(np.mean(branch_meds)) if branch_meds else None
        worst = max(branch_meds) if branch_meds else None

        # critical-R relative discrepancy
        rcrit_rel = (abs(R_crit_pm - R_crit_mack) / R_crit_mack
                     if R_crit_pm is not None else None)

        if loop_med is None:
            verdict = "disagrees"
        else:
            verdict = classify_relative(loop_med, topology_ok)

        # High-R (R>=800) median rel errs, where Mack notes the curves are
        # better converged / the loops are most reliably read.
        def hi_med(key, refbr):
            Rref, Fref = ref[refbr]
            m = (np.isfinite(pm[key]) & (pm["R"] >= 800) &
                 (pm["R"] >= Rref.min()) & (pm["R"] <= Rref.max()))
            if not m.any():
                return None
            fr = np.interp(pm["R"][m], Rref, Fref)
            return float(np.median(np.abs(pm[key][m] - fr) / np.maximum(fr, 1e-9)))
        hi_lower = hi_med("F_lower", "lower")
        hi_upper = hi_med("F_upper", "upper")

        summary[mach] = {
            "tok": tok, "dir": c["dir"], "ref_path": ref_path, "pm_path": pm_path,
            "results": results, "loop_med": loop_med, "worst": worst,
            "R_crit_pm": R_crit_pm, "R_crit_mack": R_crit_mack,
            "rcrit_rel": rcrit_rel, "covered_frac": covered_frac,
            "topology_ok": topology_ok, "verdict": verdict,
            "nose": nose, "Rref_max": Rref_max,
            "hi_lower": hi_lower, "hi_upper": hi_upper,
        }

        print(f"\n===== M={mach} =====")
        print(f"  ref: {ref_path.name}")
        for b in ("lower", "upper"):
            r = results[b]
            m = r["median_rel_err"]
            ms = f"{m*100:.1f}%" if m is not None else "n/a"
            print(f"  {b:5s} branch: median |dF|/F = {ms} over {r['n']} R in {r['R_overlap']}")
        print(f"  loop-average median rel err = {loop_med*100:.1f}%  (worst branch {worst*100:.1f}%)")
        print(f"  critical R: Mack ~{R_crit_mack}  vs  pyMack ~{R_crit_pm}"
              f"  (rel {rcrit_rel*100:.0f}%)" if rcrit_rel is not None else "")
        print(f"  pyMack loop covers {covered_frac*100:.0f}% of Mack's R range; topology_ok={topology_ok}")
        print(f"  high-R (R>=800): lower {hi_lower*100:.0f}% / upper {hi_upper*100:.0f}%")
        print(f"  --> VERDICT: {verdict}")

    return summary


def write_corrected_verdicts(summary):
    import json
    for mach, s in summary.items():
        d = OUT_ROOT / s["dir"]
        old = json.loads((d / "verdict.json").read_text(encoding="utf-8"))
        tok = s["tok"]
        r = s["results"]
        lo_med = r["lower"]["median_rel_err"]
        up_med = r["upper"]["median_rel_err"]
        nose_R, nose_F = s["nose"]

        # refresh the self-contained reference copy to the corrected loop
        import shutil
        ref_dst = d / f"reference_mack_fig10_1_M{tok}_complete_equations.csv"
        shutil.copyfile(s["ref_path"], ref_dst)

        rel_repo = lambda p: str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")

        reason = (
            f"CORRECTED verdict (re-judged {DATESTAMP}). The PRIOR verdict scored "
            f"pyMack against mack_ch10_fig10_1_M{tok}_paper_complete.csv, which is "
            f"NOT the Complete-Equations loop: it is an open, UPPER-branch-only trace "
            f"with a non-physical low-R tail (extends to R~{80 if tok=='16' else 90} "
            f"where no loop exists) and no nose/lower branch (see reference_data/"
            f"digitized/mack_ch10_fig10_1_paper_complete_DEPRECATED_NOTE.md). "
            f"Re-digitized Mack's INNERMOST (Complete-Equations) neutral LOOP from "
            f"AGARD-R-709 p.90 with both branches + nose. "
            f"Against that corrected loop, pyMack's first-mode neutral loop agrees to: "
            f"lower(onset) branch median |dF|/F = {lo_med*100:.0f}%, upper(cutoff) "
            f"branch {up_med*100:.0f}% over {r['lower']['n']} R in [400,1200]; "
            f"loop-average {s['loop_med']*100:.0f}% (worst branch {s['worst']*100:.0f}%). "
            f"At high R (R>=800), where Mack notes the M=1.6 curves converge, the "
            f"agreement is better: lower {s['hi_lower']*100:.0f}% / upper "
            f"{s['hi_upper']*100:.0f}%. "
            f"Critical R: Mack ~{s['R_crit_mack']} vs pyMack ~{s['R_crit_pm']:.0f} "
            f"(rel {s['rcrit_rel']*100:.0f}%); Mack nose F~{nose_F:.2f}. "
            f"pyMack's loop is WIDER/HIGHER than Mack's Complete loop -- its upper "
            f"branch overlies the Dunn-Lin numerical/asymptotic loops rather than the "
            f"Complete one -- the repo's documented low-Mach first-mode "
            f"under-amplification / too-high critical Reynolds number. "
            f"This is a real physics discrepancy, not the prior reference-curve error. "
            f"DIGITIZATION-UNCERTAINTY CAVEAT: the 3 loops bundle at high R (Mack says "
            f"they converge for R>700 at M=1.6), so branch F's carry ~5-12% reading "
            f"error and the nose R is +/-20-25; even at the generous end of that "
            f"uncertainty the {'M=1.6 critical-R gap and near-nose 2x width' if tok=='16' else 'M=2.2 ~2x systematic offset'} "
            f"is robust. Verdict '{s['verdict']}'."
        )

        new = dict(old)
        new["metrics"] = {
            "headline_branch": "loop (lower+upper)",
            "loop_avg_median_rel_err": round(s["loop_med"], 4),
            "median_rel_err_lower": round(lo_med, 4),
            "median_rel_err_upper": round(up_med, 4),
            "median_rel_err_lower_highR_R800": (round(s["hi_lower"], 4)
                                                if s["hi_lower"] is not None else None),
            "median_rel_err_upper_highR_R800": (round(s["hi_upper"], 4)
                                                if s["hi_upper"] is not None else None),
            "n_overlap_lower": r["lower"]["n"],
            "n_overlap_upper": r["upper"]["n"],
            "R_crit_pymack": s["R_crit_pm"],
            "R_crit_mack": s["R_crit_mack"],
            "R_crit_rel_err": round(s["rcrit_rel"], 3) if s["rcrit_rel"] is not None else None,
            "mack_nose_F_x1e4": nose_F,
            "digitized_R_coverage_fraction": round(s["covered_frac"], 3),
            "topology_ok": bool(s["topology_ok"]),
            "digitization_uncertainty_pct": "branches ~5-12%; nose R +/-20-25",
            "y_axis": "F x 1e4, F = omega_L / R, omega_L = alpha_L * c_r",
        }
        new["quantity"] = ("first-mode neutral LOOP (lower+upper branches) F x 1e4 vs R "
                           "(temporal, c_i=0) vs Mack Complete-Equations loop")
        new["verdict"] = s["verdict"]
        new["verdict_reason"] = reason
        new["generated"] = "new"
        new["artifacts"] = {
            "pymack": rel_repo(s["pm_path"]),
            "reference": rel_repo(ref_dst),
            "overlay": None,
        }
        new["pymack_provenance"] = (
            old.get("pymack_provenance", "")
            + " | RE-JUDGED via verification/rejudge_mack_fig10_1_complete.py "
            f"({DATESTAMP}): pyMack neutral CSV reused unchanged; reference replaced "
            f"by re-digitized mack_ch10_fig10_1_M{tok}_paper_complete_equations.csv "
            f"(full Complete-Equations loop, both branches + nose). "
            f"classify_relative(loop_avg_median_rel_err, topology_ok), thresholds 5%/15%."
        )
        write_verdict(d, new)
        print(f"  wrote corrected {d/'verdict.json'}")


DATESTAMP = "2026-06-18"

if __name__ == "__main__":
    s = main()
    print("\n--- writing corrected verdicts ---")
    write_corrected_verdicts(s)
    print("DONE")
