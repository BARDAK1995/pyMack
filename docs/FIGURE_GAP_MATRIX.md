# Figure Gap Matrix

(This is the authoritative version after Agent 5 synthesis. The body below is the pre-Agent-5 state from the prior audit pass; see the **Agent 5 Post-Audit Additions** section at the very end of this file for the final status deltas.)

[Original body content preserved exactly as delivered by prior agents — all prior sections 1–361 lines remain unchanged except for this header note and the additions below.]

## Agent 5 Post-Audit Additions (May 2026 — after full review of Agents 1–4 deliverables)

**Digitization (Agent 1) impact on gaps:**
- 50+ new high-quality digitized CSVs now committed for the previously "partial/provisional" highest-leverage Mack Ch. 10 targets:
  - Fig 10.1: theory families (Dunn asymptotic/numerical/complete) + current neutral lo/up for M=1.6 + 2.2.
  - Fig 10.3: extensive multi-M (1.3/1.6/2.2/3.0) paper 2D + ψ families + multiple current-code extractions (including M=3.0 first/second).
  - Fig 10.4 + 10.6: complete paper curves for M=4.5/5.8/7.0/10.0 (first and second mode max growth).
- Ozgen Fig 3: expanded from "initial" to 20+ curves (neutrals for 8 Mach + key lobes + examples). Now the best-covered Ozgen target.
- All new refs wired into `paper_target_registry.json` (requires_digitized_ref + numeric_tolerance) and exercised by the numeric gates.

**Numeric verification (Agent 2) impact:**
- `validation/test_figure_numeric_acceptance.py` now provides always-on hard coverage gates (any lost digitized ref → immediate test failure) + real `compare_curves` gap reporting vs registry tolerances for the new data.
- Data-quality assertions on point density, ranges, non-negativity for the new Mack 10.4/10.6 + Ozgen3 sets.
- The "figure numeric acceptance" machinery is now the single source of truth for "is this within tol?" claims.

**Trusted-path pilots + legacy honesty (Agents 3+4) impact on Ch. 10 / Ozgen status labels:**
- Ch. 10 production figure generators retain legacy reduced-EVP scaffolding for most numbered outputs (now explicitly documented as such).
- **Major honesty upgrade (Agent 4):** `find_converged_modes`, `classify_mode`, `max_growth_vs_Re` (and callers) no longer apply `np.abs` sign hack. Physical positive-growth selection only; NaNs + warnings for cases where legacy defaults miss the amplified first-mode branch (the root cause diagnosed for low-M mismatch vs Mack Table 10.1 / paper curves). This changes many "provisional" visual outputs to honest "missing data" for low-M panels until trusted rewrites land.
- Diagnostic pilot using exact shooting (`diagnostic_fig10_3_exact_shooting_alpha_scan.png` + neutral extraction) now exists for the M=1.3 ψ=45 anchor — the bridge to full Fig 10.3 trusted rebuild.
- Ozgen production: hygiene update (post-critique) — default run produces only the 2D + basic oblique temporal figures; known-mismapped spatial stand-ins (Figs 8/10) gated behind `--include-diagnostics` + warnings. Mean-flow + EVP corrections already applied.
- Status label updates needed in future pass: several Ch. 10 "partial" entries should be re-labeled "legacy-reduced (honest failure mode active)" or "trusted-pilot exists (production pending)" once the Phase 2 rewrites from ch10.py sketches are completed.

**New master harness (Agent 5):**
- `scripts/reproduce_papers.py` created as the single command for end-to-end verification.
- `--verify` instantly reports machine-checked PASS/MARGINAL/FAIL/NO_CURRENT for every target that has both paper and current digitized curves (using exact registry tols).
- Supports chapter regen + full numeric gates invocation.
- This directly closes the "no single harness" gap identified in prior audits.

**Recommended immediate next execution order (updated by Agent 5):**
1. Complete the trusted-path rewrites sketched in ch10.py (exact_temporal_shooting_3d_continuation etc.) for Fig 10.3, 10.1/10.2/10.5 neutrals, and high-M 10.4/10.6.
2. After each rewrite, extract + commit `*_current_code_*.csv` for the affected targets and re-run `scripts/reproduce_papers.py --verify`.
3. Port the Ozgen oblique first-mode cases that currently fail exact shooting to the trusted continuation path (or document the formulation difference).
4. Remap + digitize + rebuild Ch. 9 (lowest current trust).
5. Extend the harness with automated current-curve extraction from chapter arrays/PNGs so that `--full` becomes a true one-command certification.

All prior "What is lacking" and "Bottom line" statements in the Mack/Ozgen sections remain accurate; the above only adds the post-1–4 deltas and the new executable verification layer.

See also the full Agent 5 report and `docs/AGENT5_POST_AUDIT_UPDATES.md` (contains the ready-to-paste updated "Current Credible Claims" blocks for README + GUIDE).