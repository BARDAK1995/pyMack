# pyMack session log — 2026-07-02 (working note, UNCOMMITTED)

Handoff/resume note for the big validation-data + verification-directory work done this
session. **Nothing below is committed** (branch `verification`, HEAD `7f664e2`). Working
tree: ~82 modified, ~74 deleted (mostly Özgen dir moves git sees as delete+add), ~29 untracked.

---

## What this session accomplished (in order)

### 1. refPapers/ — source-paper figure extraction (local-only, gitignored)
- Built high-DPI (400 dpi) figure crops from the two source PDFs and wired all **85 figures**
  into the LaTeX transcriptions under `refPapers/latex_papers/` (Mack_1984.tex, Ozgen_Kircali_2008.tex).
- Source crops now live at `refPapers/latex_papers/figures/fig10_1.png … fig10_6.png`, `ozgen_fig*.png`, etc.
- Tooling: `refPapers/_tools/pdf_tools.py` (render/crop/search), `_cache/` page renders.
- **Everything under `refPapers/` is gitignored** (copyright — never commit paper scans/renders).

### 2. Verified digitizations against the real source figures
- **Özgen Fig 3 (all 7 Mach): digitization was CORRECT** — no changes needed. Proof overlays
  `mixed_mode/ozgen_fig3/_refdigitize/_verify2_M*.png` (gitignored; show digitized points on the scanned panel).
- **Mack Ch.10 (10.1/10.3/10.4/10.6): digitizations were WRONG** — offsets, wrong-curve
  conflation, ~2× tail error (10.4 M10), fabricated out-of-range points. Found via pixel-trace overlays.

### 3. Re-digitized the 20 flawed Mack Ch.10 reference CSVs
- Old versions archived to `reference_data/digitized/_archive_pre_redigitize_2026-07-02/`.
- New CSVs traced from the 400-dpi crops, cross-checked against pyMack's own computed curves.
- Fixed a real bug: `mack_ch10_fig10_6_M58_paper.csv` rows R=1100–1800 were byte-identical
  duplicates of M45 (tracer locked onto wrong curve) — replaced with re-traced values.

### 4. Regenerated all Mack Ch.10 verdicts + overlays + matrix + validation.tex
- **7 verdict categories FLIPPED** after correction (old digitization was the culprit, not pyMack):
  - 10.1 M1.6: disagrees 36.6% → **acceptable 7.7%**;  M2.2: disagrees 128% → disagrees 54.5%
  - 10.3 M1.3: agrees 3.8% → agrees **1.3%**;  M2.2: unchanged 32.2% (CSV was byte-identical);  M3.0: disagrees 27.7% → **agrees 3.4%**
  - 10.4 M4.5: acceptable 6.9% → **agrees 1.4%**;  M5.8: disagrees 16.7% → **agrees 2.8%**;  M7.0: disagrees 31.6% → **agrees 1.7%**;  M10.0: disagrees 60.9% → **acceptable 6.1%**
  - 10.6 M4.5: agrees 1.0% → acceptable 6.0%;  M5.8: acceptable 6.7% → acceptable 8.9%;  M7/M10 untouched (agrees 2.8/4.0%)
- **Headline:** the old "first mode systematically under-amplified at high Mach" narrative was
  mostly a digitization artifact; genuine disagreement is now isolated to the **M≈2.2** cases.
- `docs/validation.tex` updated throughout (percentages, verdict words, summary table, abstract,
  conclusion tally) + 6 embedded overlays regenerated in place. Recompiles clean (20 pp, 0 errors).
- Patched pre-existing stale-path bugs in the Mack compute/compare scripts (neutralCurve_verification
  / growthRate_verification → first_mode/second_mode).

### 5. Full read-only audit of verification/ (4 parallel agents + synthesis)
Found the real problems the user intuited: stale/wrong "production" figures + Özgen mode mislabel.

### 6. Reorganized verification/ (this is the most recent work — see state below)
- **Özgen → new `verification/mixed_mode/ozgen_fig3/` family** (chosen over leaving in first_mode/),
  because one Özgen panel spans BOTH first and second mode for M≥4. Layout:
  `mixed_mode/ozgen_fig3/{M2,M3,M4,M6,M7,M8,M10,lobes,_compute,_refdigitize}`.
  verdict.json `mode` field fixed: `first` for M2/M3, `first+second` for M4–M10 + lobes.
- **Rebuilt all 8 Özgen overlays CORRECTLY**: pyMack neutral curves over the corrected
  multi-branch digitized POINTS (no scanned panel — copyright), verdict-driven titles read from
  verdict.json. The old wrong "closed-arch / disagrees 23.9%" figures are gone. **Verified by eye.**
- **Fixed Fig 10.1 hardcoded overlay titles** in `_make_overlays_audit.py` (were "~37%"/"128%",
  now read live verdict values); removed the old broken Özgen plotting from that script.
- **Archived dead scripts** → `verification/_archive/` (+ README): `make_mack_fig10_4_overlay.py`
  (dup), `_migrate_by_mode.py` + `_seed_pending.py` (pre-migration one-shots), `_probe_m10.py`
  (throwaway), `compare_ozgen_fig3.py` (superseded — read old non-v2 CSV, stale output path;
  real producer is the `_refdigitize/finalize_*` + `discrete_mode.py` pipeline using `_v2` CSVs).
- **Deleted ~17 MB junk scratch** (kept all .py pipeline scripts + panel renders + proofs).
- **Taught `build_success_matrix.py` + `build_galleries.py` about `mixed_mode`.** Rebuilt
  SUCCESS_MATRIX.md (now has "Mixed first + second mode" section, 8 Özgen cases) and **3 galleries**
  (first_mode, second_mode, **mixed_mode** — all fresh 10:43).
- Fixed cosmetic verdict artifact paths (fig10.4 overlay=null → set; fig10.6 pymack path → in-dir).
- Fixed 3 more latent stale-path scripts uniformly → second_mode: `compare_egorov2006_m6.py`,
  `compare_sean_m5p35.py`, `verify_mack_fig10_6.py`, `write_verdict_mazhong.py`,
  `compute_balakumar_malik1992_branches.py`.

---

## Current verification/ layout (final)
```
verification/
  _archive/            (5 retired scripts + README)
  _compare_lib.py      (LIVE shared verdict lib — imported widely, do NOT move)
  build_success_matrix.py  build_galleries.py   (canonical builders; know about mixed_mode now)
  _make_overlays.py  _make_overlays_audit.py  _make_overlays_mack_103_104.py  _make_overlays_spectrum.py
  SUCCESS_MATRIX.md  README.md  TARGETS.md
  first_mode/  (11 cases: mack_fig10_1 ×2, mack_fig10_3 ×3, mack_fig10_4 ×4 +family, malik_case3)
  second_mode/ (16 cases: mack_fig10_6 ×4 +family, malik_case4/5/6, balakumar ×2, egorov, cone,
                mazhong, sean, malik_tableX, malik_fig4_eigenfunction)
  mixed_mode/ozgen_fig3/ (M2,M3,M4,M6,M7,M8,M10,lobes)   ← NEW
  other/       (malik_case1, orszag_spectrum)
  first_mode_gallery.png  second_mode_gallery.png  mixed_mode_gallery.png
```

## Verified-correct end state (checked this session, post-interruption)
- No Özgen left under first_mode/. mixed_mode overlays correct (multi-branch, points-not-panel,
  verdict-driven titles) — M2 & M6 viewed and confirmed. Matrix mixed-mode section consistent.
  3 galleries fresh. `_make_overlays_audit.py` has no hardcoded numbers / no Özgen code. Cosmetics fixed.

---

## REMAINING / OPEN ITEMS
1. **`verification/compare_malik1990_anchors.py`** still has `OUT = HERE / "eigenvalueAnchor_verification"`
   (a folder that no longer exists). It writes to MULTIPLE mode dirs (malik_case6 + via_xirenfu →
   second_mode/, malik_case1 → other/), so it needs a **per-case mode map**, not a token swap.
   LATENT ONLY — the committed verdicts are all correct on disk; this only bites on a future rerun.
2. **NOTHING is committed.** Huge uncommitted pile spanning: digitization CSVs, regenerated
   verdicts/overlays, validation.tex, the whole verification reorg (dir moves, archives, deletions),
   and earlier refPapers .gitignore. Needs a thoughtful commit (or several) when the user is ready.
   Özgen dir moves will show as delete+add until committed (git detects renames on commit).
3. The interrupted workflow's own two agent reports (build + independent-verify) were LOST (process
   exited with no transcript). I re-verified the end state myself instead — it's sound. If deeper
   re-verification is wanted, re-run a read-only check pass.
4. Özgen case_id inside verdict.json is still `ozgen_m4` etc. (folder is `M4`). Cosmetic mismatch;
   matrix links work. Could rename case_id → match folder if desired.
5. Minor: some Özgen overlay titles are wide (M6 runs past plot). Cosmetic only.

## Key conventions / facts to remember
- Verdict scale: agrees ≤5%, acceptable 5–15%, disagrees otherwise.
- Production overlays = pyMack curves over digitized POINTS (NEVER embed scanned paper figures).
- refPapers/ is gitignored (copyright). validation.tex embeds only pyMack's own overlays.
- Do NOT re-run compute_*/verify_*/rejudge_* or _refdigitize finalize scripts casually — some take
  90+ min. Current verdict.json numbers are all correct; only cheap replot/matrix/gallery regen needed.
- pyMack 0.1.0 refactor dropped `find_temporal_mode_anchor_3d_shooting` etc. from the public
  `pymack/__init__` facade (still in `pymack/analysis.py`) — blocks Özgen-Fig10.3-M1.6 pyMack curve
  and needed a direct-import workaround in scripts/make_mack_fig10_3_overlay.py.
- NEVER add Co-Authored-By: Claude trailers to commits in this repo (history was rewritten to strip them).
```
