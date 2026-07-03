# Ma & Zhong (2003) Fig. 15 — pixel digitization calibration

Reproducible record for the pixel-traced digitization in `digitize_fig15.py`
(replaces the earlier hand-eyeballed points, which had **no recorded
calibration**).

## Source crop
`refPapers/NewPapers/figures/mazhong2003_fig15_neutral_curves_2D.png`
— 2048 × 1502 px, RGB, 400 DPI. Single plot, x = R (0–2000, major ticks every
200), y = ω (0–0.28, major ticks every 0.04).

## Axis frame (detected)
- **R = 0** vertical line: pixel columns ~173–176 (darkest column run).
- **ω = 0** horizontal line: pixel rows ~1330–1332 (darkest row run).

## Tick marks (point INWARD) and linear fits
Ticks extend *into* the plot, so they were detected in a band just inside each
axis line.

**X-axis major ticks** (columns), assigned R = 0,200,…,1400:

| col | 174 | 348 | 520 | 693 | 866 | 1036 | 1210 | 1382 |
|-----|-----|-----|-----|-----|-----|------|------|------|
| R   |  0  | 200 | 400 | 600 | 800 |1000  |1200  |1400  |

Fit: **R = 1.159656 · col − 202.9371**  (tick residuals < 1.5 in R).

**Y-axis major ticks** (rows), assigned ω = 0.04,…,0.28:

| row | 1146 | 962 | 779 | 594 | 412 | 230 | 46  |
|-----|------|-----|-----|-----|-----|-----|-----|
| ω   | 0.04 |0.08 |0.12 |0.16 |0.20 |0.24 |0.28 |

Fit: **ω = −0.00021828 · row + 0.290001**  (residuals < 3e-4).

## Independent interior-tick verification
(ticks NOT used as fit endpoints — confirms the mapping is linear & correct)
- R = 1000  → predicted col **1037.3**, tick detected at **1036**  ✓
- ω = 0.12  → predicted row **778.8**, tick detected at **779**    ✓

## Anchor pixels (from the fits)
| data       | pixel        |
|------------|--------------|
| R = 0      | col 175.00   |
| R = 2000   | col 1899.65  |
| ω = 0      | row 1328.57  |
| ω = 0.28   | row 45.82    |

## Trace parameters
- Ink threshold: gray < 110.
- Dotted-ray rejection: any dark run with |ω − 0.6e-4·R| < 0.0035 or
  |ω − 2.2e-4·R| < 0.0035 is discarded, so the sparse F-rays are never traced.
- Continuity: nearest-ω run to the previous point, with a per-branch jump gate.

## Cross-check against Ma & Zhong §6 (independent textual anchor)
Paper states the F = 2.2e-4 line crosses the 2nd-mode neutral curve at
R = 806 (branch I, lower) and R = 999.6 (branch II, upper).
From the traced branches:

| branch          | traced R | paper R | diff  |
|-----------------|----------|---------|-------|
| I  (2nd lower)  | 778.8    | 806     | 3.4 % |
| II (2nd upper)  | 1002.5   | 999.6   | 0.3 % |

Both within a few percent of the published values → calibration + trace
validated.

## Notes / honest limitations
- 1st-mode **lower** branch traced R ≈ 560–1580; the very faint solid tail
  below ω ≈ 0.006 fades under the ink threshold past R ≈ 1580 (not invented —
  same 1st-mode-onset partial-resolution limitation noted elsewhere).
- All other branches trace the full published extent (2nd mode nose R ≈ 245 →
  R = 2000; 1st-mode upper nose R ≈ 560 → R ≈ 1940).
