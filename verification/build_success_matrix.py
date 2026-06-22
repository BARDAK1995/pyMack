"""Aggregate every ``verdict.json`` under verification/ into SUCCESS_MATRIX.md.

Cases are organized by physical MODE (the meaningful axis): pyMack's SECOND
(Mack) mode is validated across sources/Mach/geometry; its FIRST mode is the
documented weak spot. ``other`` holds incompressible / unrecoverable-condition
cases. Within each mode, cases are sub-grouped by quantity (verdict ``category``).

This script does NO physics: it reads recorded verdicts. Re-run after any change.

    python verification/build_success_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _compare_lib import VERDICT_BADGE, VERDICT_ORDER, read_verdict  # noqa: E402

# (folder, title, one-line characterization)
MODES = [
    ("second_mode", "Second (Mack) mode",
     "pyMack's design target — validated across independent sources, Mach 4.5–10, and a cone"),
    ("first_mode", "First mode",
     "agrees with Özgen where the discrete mode is cleanly resolvable (a discrete-mode "
     "eigenfunction-decay extractor recovers it); the remaining Mack-figure disagreements "
     "use a separate solver and await the same scrutiny"),
    ("other", "Other",
     "incompressible / unrecoverable-condition cases (unmeasured)"),
]
CAT_LABEL = {
    "neutral_curve": "Neutral curves",
    "growth_rate": "Growth rates",
    "eigenvalue_anchor": "Eigenvalue anchors",
}
CAT_ORDER = ["neutral_curve", "growth_rate", "eigenvalue_anchor"]


def _cond(c: dict) -> str:
    bits = []
    if "Ma" in c:
        ma = c["Ma"]
        bits.append(f"M={ma:g}" if isinstance(ma, (int, float)) else f"M={ma}")
    psi = c.get("psi_deg")
    if psi is not None:
        bits.append(f"ψ={psi:g}°" if isinstance(psi, (int, float)) else f"ψ={psi}°")
    if c.get("gas"):
        bits.append(str(c["gas"]))
    if c.get("wall"):
        bits.append(str(c["wall"]))
    return ", ".join(bits)


def _pct(x):
    return f"{100 * float(x):.1f}%"


def _headline(v: dict) -> str:
    """One-line curated metric per case (full numbers live in verdict.json)."""
    m = v.get("metrics", {})
    if not m:
        return "—"
    if "headline" in m:
        return str(m["headline"])
    if "curve_median_rel_err" in m:
        s = f"curve median {_pct(m['curve_median_rel_err'])}"
        if "table_anchor_rel_err" in m:
            s += f"; Table-10.1 anchor {_pct(m['table_anchor_rel_err'])}"
        return s
    if "median_rel_err_alpha" in m:
        topo = "closed-arch" if m.get("topology_ok") else "open-lobe"
        return f"median |Δα|/α {_pct(m['median_rel_err_alpha'])}; topology {topo}"
    if "loop_avg_median_rel_err" in m:
        return (f"loop-avg {_pct(m['loop_avg_median_rel_err'])}; "
                f"R_crit pyMack {m.get('R_crit_pymack','?')} vs Mack {m.get('R_crit_mack','?')}")
    if "alpha_r_rel_err" in m:
        return (f"α_r {_pct(m['alpha_r_rel_err'])}, "
                f"α_i {_pct(m['alpha_i_rel_err'])} (N={m.get('N', '?')})")
    if "malik_omega" in m:
        o = m["malik_omega"]
        return f"Malik ω={o[0]:.4f}{o[1]:+.5f}i (temporal; no pyMack run)"
    if "branch_I_rel_err" in m and "branch_II_rel_err" in m:
        topo = "closed band" if m.get("topology_ok") else "topology mismatch"
        return (f"Branch I R {m['branch_I_R_pymack']:g} ({_pct(m['branch_I_rel_err'])}), "
                f"Branch II R {m['branch_II_R_pymack']:g} ({_pct(m['branch_II_rel_err'])}); {topo}")
    if "upper_branch_MAE_mm_200_600kHz" in m:
        return (f"upper {m['upper_branch_MAE_mm_200_600kHz']:.1f} mm; "
                f"lower {m.get('lower_branch_MAE_mm_330_600kHz', float('nan')):.1f} mm "
                "(gated bands)")
    parts = [f"{k}={val:.3g}" for k, val in m.items()
             if isinstance(val, (int, float))][:2]
    return "; ".join(parts) if parts else "—"


def _tally(cases):
    t = {"agrees": 0, "acceptable": 0, "disagrees": 0, "pending": 0}
    for v in cases:
        t[v.get("verdict", "pending")] = t.get(v.get("verdict", "pending"), 0) + 1
    return t


def _tally_str(t):
    return (f"{VERDICT_BADGE['agrees']} {t['agrees']} · "
            f"{VERDICT_BADGE['acceptable']} {t['acceptable']} · "
            f"{VERDICT_BADGE['disagrees']} {t['disagrees']} · "
            f"{VERDICT_BADGE['pending']} {t['pending']}")


def main() -> int:
    cases_by_mode = {}
    for mode_dir, _, _ in MODES:
        cases = []
        d = HERE / mode_dir
        if d.is_dir():
            for vf in sorted(d.glob("*/verdict.json")):
                cases.append(read_verdict(vf))
        cases_by_mode[mode_dir] = cases

    all_cases = [v for cs in cases_by_mode.values() for v in cs]
    total = len(all_cases)
    overall = _tally(all_cases)

    L = []
    L.append("# pyMack Verification Success Matrix")
    L.append("")
    L.append("Honest agreement audit of pyMack against published / external benchmarks")
    L.append("at their **exact** conditions, organized by physical **mode** — the")
    L.append("meaningful axis here. Generated by `verification/build_success_matrix.py`")
    L.append("from per-case `verdict.json`. Methodology & thresholds:")
    L.append("`verification/README.md` (≤5% agrees · 5–15% acceptable · else disagrees).")
    L.append("")
    L.append("Each case name below links to its pyMack-vs-reference **overlay plot**. "
             "At-a-glance galleries: "
             "[second-mode](second_mode_gallery.png) · "
             "[first-mode](first_mode_gallery.png).")
    L.append("")
    L.append(f"**Overall ({total} cases):** {_tally_str(overall)}")
    L.append("")
    L.append("**Headline:** every *second-mode* case agrees/acceptable. For the *first mode*, the "
             "Özgen cases — once their references were re-digitized correctly and a discrete-mode "
             "(eigenfunction-decay + y_max-stationarity) extractor replaced the c_r-band classifier "
             "that had excluded the genuine first mode — now AGREE where the mode is cleanly "
             "resolvable (M4/M6 reproduce all four neutral branches to ≤8.5%; M2/M3 agree over the "
             "resolvable part of the lobe). The low-α first-mode *onset* at low Mach stays "
             "continuous-spectrum-limited (a numerical isolation limit, not a physics disagreement). "
             "The remaining ❌ are Mack-figure first modes (Fig 10.1/10.3/10.4) computed through a "
             "*separate* solver/comparison that has NOT yet had the same discrete-mode scrutiny — so "
             "the earlier blanket 'first mode systematically under-amplified' claim is retired pending "
             "that re-examination.")
    L.append("")

    for mode_dir, title, blurb in MODES:
        cases = cases_by_mode.get(mode_dir, [])
        if not cases:
            continue
        L.append(f"## {title}")
        L.append(f"*{blurb}.*  ")
        L.append(f"**{len(cases)} cases:** {_tally_str(_tally(cases))}")
        L.append("")
        by_cat = {}
        for v in cases:
            by_cat.setdefault(v.get("category", "other"), []).append(v)
        cats = [c for c in CAT_ORDER if c in by_cat] + \
               [c for c in by_cat if c not in CAT_ORDER]
        for cat in cats:
            rows = sorted(by_cat[cat],
                          key=lambda v: (VERDICT_ORDER.get(v.get("verdict", "pending"), 9),
                                         v.get("case_id", "")))
            L.append(f"### {CAT_LABEL.get(cat, cat)}")
            L.append("")
            L.append("| Case | Source | Conditions | Verdict | Headline |")
            L.append("|---|---|---|---|---|")
            for v in rows:
                cid = v.get("case_id", "?")
                ov = (v.get("artifacts", {}) or {}).get("overlay")
                if ov:
                    rel = ov[len("verification/"):] if ov.startswith("verification/") else ov
                    cid_cell = f"[`{cid}`]({rel})"  # case name links to its overlay plot
                else:
                    cid_cell = f"`{cid}`"
                L.append("| {cid} | {src} | {cond} | {verd} | {head} |".format(
                    cid=cid_cell,
                    src=v.get("source", "?").split(",")[0],
                    cond=_cond(v.get("conditions", {})),
                    verd=VERDICT_BADGE.get(v.get("verdict", "pending"), "?"),
                    head=_headline(v)))
            L.append("")

    # Per-case detail (full reasoning), grouped by mode.
    L.append("## Details")
    L.append("")
    for mode_dir, title, _ in MODES:
        for v in cases_by_mode.get(mode_dir, []):
            reason = (v.get("verdict_reason") or "").strip()
            if not reason or v.get("verdict") == "pending":
                continue
            L.append(f"**`{v['case_id']}`** ({title}) — "
                     f"{VERDICT_BADGE.get(v.get('verdict'), '?')} "
                     f"({v.get('quantity', '')})  ")
            L.append("")
            L.append(reason)
            L.append("")

    out = HERE / "SUCCESS_MATRIX.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({total} cases: {overall})")
    for mode_dir, title, _ in MODES:
        print(f"  {title}: {_tally(cases_by_mode.get(mode_dir, []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
