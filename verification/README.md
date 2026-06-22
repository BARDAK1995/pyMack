# pyMack Verification Audit

A rigorous, honest agreement audit of pyMack against published and external
benchmarks **at their exact conditions** — neutral curves and growth rates from
Mack (1984), Özgen & Kırcalı (2008), and the collaborator (Sean) Mach 5.35 N₂
case. The deliverable is [`SUCCESS_MATRIX.md`](SUCCESS_MATRIX.md): a 3-tier
agreement table built mechanically from per-case `verdict.json` files.

This is deliberately separate from `validation/` (the CI gates): here we run
*their* conditions, measure agreement, and report it — including where pyMack
**does not** agree. The goal is an honest success matrix, not a pass.

## Layout

Cases are organized by physical **mode** — the meaningful axis (pyMack's second
mode is validated; its first mode is the documented weak spot). Each case's
`verdict.json` carries `mode` ("second"/"first"/"other") and `category`
("neutral_curve"/"growth_rate"/"eigenvalue_anchor"); the matrix groups by mode,
then category.

```
verification/
  README.md                     this file
  TARGETS.md                    exact conditions per case
  _compare_lib.py               shared metrics + the 3-tier classifier
  build_success_matrix.py       verdict.json -> SUCCESS_MATRIX.md (mode-grouped)
  _migrate_by_mode.py           one-shot: the authoritative case->mode mapping
  compare_*.py / compute_*.py / verify_*.py   per-source engines (see note below)
  SUCCESS_MATRIX.md             the deliverable
  second_mode/<case>/{pymack..., reference..., overlay.png, verdict.json}
  first_mode/<case>/{...}       (+ first_mode/_ozgen_compute/ shared Özgen grids)
  other/<case>/{...}            incompressible / unrecoverable-condition cases
  comparisons/                  cross-case overlays (not single-case verdicts)
```

`comparisons/ozgenM6_vs_pymack_M5p85_M6_neutral.png` overlays Özgen's M=6
*first-mode* neutral curve with pyMack's M=5.85 N₂ and M=6 air *second-mode*
neutral curves in a common (R, F) plane — a where-do-they-sit view (different
instabilities), not a like-for-like verdict.

> **Note on the engines.** `compare_*.py`, `compute_*.py`, and `verify_*.py` were
> authored against the original *by-quantity* layout
> (`neutralCurve_/growthRate_/eigenvalueAnchor_verification/`) and were used to
> *produce* the verdicts; the results were then reorganized **by mode**
> (`_migrate_by_mode.py` holds the exact case→mode map). They are kept as run
> provenance. Re-running one writes to the legacy by-quantity path; move its
> output to the matching `mode/` folder (or repoint the script's output constant)
> before rebuilding the matrix.

## Verdict tiers

Hand-digitized literature curves carry ~2–5 % reading error, so a median
relative error at/below the **5 %** digitization noise floor is genuine perfect
agreement.

| Badge | Meaning | Criterion |
|---|---|---|
| ✅ agrees | at digitization noise floor | median rel-err ≤ 5 % **and** matching topology |
| 🟡 acceptable | correct physics, bounded offset | rel-err 5–15 %, or sub-domain match with a documented localized discrepancy |
| ❌ disagrees | real disagreement | rel-err > 15 %, wrong magnitude, wrong topology, or missing/extra mode |
| ⬜ pending | reference ready, pyMack run not generated | machinery/compute gap |

Dimensional curves (Sean) use MAE as a fraction of the curve's span with the
same thresholds.

## Reproducing

```bash
# per-source comparison engines (re-read raw data, recompute, rewrite verdicts)
python verification/compare_sean_m5p35.py
python verification/compare_mack_fig10_3.py
python verification/compare_ozgen_fig3.py
# aggregate
python verification/build_success_matrix.py
```

Each `compare_*.py` records its pyMack provenance (which run, which grid,
which conditions) in the case's `verdict.json`. No tuning to pass: thresholds
are fixed in `_compare_lib.py` and applied uniformly.
