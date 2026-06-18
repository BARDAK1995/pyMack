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

```
verification/
  README.md                     this file
  TARGETS.md                    exact conditions per case (from the registry)
  _compare_lib.py               shared metrics + the 3-tier classifier
  build_success_matrix.py       verdict.json -> SUCCESS_MATRIX.md
  compare_*.py                  per-source comparison engines (rigor lives here)
  SUCCESS_MATRIX.md             the deliverable
  neutralCurve_verification/<case>/{pymack..., reference..., overlay.png, verdict.json}
  growthRate_verification/<case>/{...}
```

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
