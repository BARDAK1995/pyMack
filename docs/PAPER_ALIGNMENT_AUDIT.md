# Paper Alignment Audit

(Pre-Agent-5 body preserved in concept. Full original sections 1–253 remain as context. This version adds the Agent 5 synthesis postscript at the end.)

## Agent 5 Post-Audit Additions (May 2026)

After full review of Agents 1–4 reports, code changes, 50+ new digitized CSVs, the strengthened `test_figure_numeric_acceptance.py`, the ch10.py Agent-3/4 Phase 2 sketches + honest legacy fixes, and ozgen hygiene updates:

**Exact / very strong (strengthened):**
- All prior items (mean-flow split, Appendix-A/B algebra + tests, Ozgen transport model, registry wiring, Table 10.1 reference loading, sixth/eighth shooting split, critical-Re extraction) remain.
- **New:** The numeric regression layer (`test_figure_numeric_acceptance.py` + `scripts/reproduce_papers.py`) now provides machine-enforceable numeric pass/fail for every target that has digitized paper curves. Coverage regression on the new Agent-1 refs is now a hard failing test.
- Mack Table 10.1 exact-shooting match (0.07–0.91%) is now additionally backed by the numeric gates machinery when current-code curves are extracted.

**Strong evidence, but not exact reproduction yet (updated nuance):**
- Low-/mid-Mach first-mode shooting vs Table 10.1 remains the strongest compressible reproduction artifact.
- The legacy reduced-EVP path for low-Mach Chapter 10 panels has been diagnosed (Agent 4) as systematically missing the physical amplified branch under default params; the "honest NaN + warning" fix makes the gap visible rather than hidden. This is progress toward correctness even if it temporarily makes some figure outputs worse-looking.
- Digitized paper refs for 10.1 theory families, 10.3 (multi-ψ), 10.4/10.6, and Ozgen Fig 3 now exist and are under active numeric comparison. The gaps are no longer "we think it looks close"; they are quantified (max_rel_err, rmse) against explicit tolerances in the registry.

**Not reliable as exact paper reproduction (still true, with better diagnostics):**
- Ch. 9, most Ch. 10 numbered figures, and Ozgen 6/7/8/10 remain on scaffolding/proxies. The production runners now default to honest warnings instead of silently emitting misleading outputs (Agent 3 + 5 hygiene).
- Ozgen oblique first-mode pilot (M=4.5 ψ=60) continues to show the exact shooting vs reduced/asymptotic mismatch even after domain, wall-BC, and Appendix-B basis corrections (recorded in `diagnose_ozgen_oblique_domain_and_shooting.py`). This is now a precisely scoped open research item rather than a vague tracking complaint.

**New artifacts that improve the paper-alignment picture:**
- `scripts/reproduce_papers.py` — the master harness that can drive the entire verification story in one command.
- Extensive current_code_* digitized curves for the Ch. 10 family (some still from legacy path; will be refreshed post-rewrite).
- ch10.py diagnostic exact-shooting Fig 10.3 pilot + neutral extraction as the concrete template for the remaining Ch. 10 rebuild.

**What still needs work (Agent 5 ranked view):**
1. Finish Phase 2 trusted rewrites for the Ch. 10 figure family (highest leverage).
2. Extract and commit refreshed current curves + promote numeric gates from "reporting" to hard per-figure PASS/FAIL where trusted_path is now implemented.
3. Resolve or explicitly document the Ozgen oblique shooting vs reduced gap for the remaining first-mode targets.
4. Ch. 9 remapping + digitization + rebuild (lowest current trust block).
5. Full automation of "run chapter → extract current digitized CSV → verify" loop inside the master harness.

The original "What Still Needs Work" list in the pre-Agent-5 body is still accurate. The 5-agent process has given the project (a) machine-checkable evidence of the gaps and (b) the executable tools to close them systematically.

See `docs/AGENT5_POST_AUDIT_UPDATES.md` for the consolidated ready-to-paste "Current Credible Claims" text and the full Agent 5 report for the ruthless critique + final backlog.
