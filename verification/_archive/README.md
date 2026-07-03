# verification/_archive

Retired verification scripts, kept for provenance (archived, not deleted). None
of these is on any live producer path; they were superseded during the by-mode
reorganisation (`neutralCurve_verification`/`growthRate_verification` -> mode
folders, and the Özgen cases -> `mixed_mode/ozgen_fig3/`).

| File | What it was | Why retired |
|---|---|---|
| `compare_ozgen_fig3.py` | Özgen Fig. 3 NEUTRAL-curve verification engine (nearest-crossing on a single closed arch). Wrote `neutralCurve_verification/ozgen_m{N}/verdict.json`. | **Superseded.** Nothing imports it (only doc-comment mentions elsewhere); its output root `neutralCurve_verification/` no longer exists; it reads the OLD mis-digitized `ozgen_fig3_M{N}_neutral.csv` (single arch), not the corrected `*_neutral_v2.csv`; and its verdict schema (`median_rel_err_alpha`, closed-arch-vs-open-lobe) is not what the current Özgen verdicts carry. The live Özgen verdicts were produced by the `_refdigitize` discrete-mode finalize pipeline (`per_branch_nearest_rel_err`, full c_i=0 contour, first+second mode). Overlays are now made by `verification/make_ozgen_overlays.py`. |
| `make_mack_fig10_4_overlay.py` | Standalone producer of the Mack Fig. 10.4 overlay PNGs (M4.5/5.8/7/10). | **Superseded duplicate.** The Fig. 10.4 overlays are produced by `make_fig10_4()` in `verification/_make_overlays_mack_103_104.py` (the live producer, which also handles Fig. 10.3). Nothing imports this file. |
| `_migrate_by_mode.py` | One-shot migration that moved case folders from the old `neutralCurve_verification/` + `growthRate_verification/` layout into the by-mode `first_mode/`/`second_mode/` folders. | **Done.** Single-use migration; the source layout no longer exists. |
| `_seed_pending.py` | Pre-migration scaffold that seeded skeleton `verdict.json` (`pending`) files under the old `neutralCurve_verification/`/`growthRate_verification/` folder names. | **Superseded.** Uses the old folder names and the pending-seed workflow that predates the finalized verdicts. |
| `_probe_m10.py` | Throwaway diagnostic probe for the M=10 Özgen panel. | **Throwaway.** Ad-hoc diagnostic, not part of any pipeline. |

## Still live (NOT archived)
- `_compare_lib.py` — shared verdict/classify library, imported widely.
- `compare_ozgen_lobes.py` — live producer of the `mixed_mode/ozgen_fig3/lobes` growth-rate-contour verdict.
- All `compute_*.py` / `verify_*.py` / the remaining `compare_*.py` — live physics producers.
- `make_ozgen_overlays.py` — the single canonical Özgen Fig. 3 neutral-curve overlay generator.
