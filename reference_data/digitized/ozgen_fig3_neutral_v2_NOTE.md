# Özgen & Kırcalı (2008) Fig 3 — corrected c_i=0 neutral-curve digitization (v2)

**Source:** Özgen, S. & Kırcalı, S.A. (2008), *Linear stability analysis in
compressible, flat-plate boundary layers*. PDF at
`reference_data/ozgen/Özgen and Kırcalı - 2008 - Linear stability analysis in compressible, flat-pl.pdf`,
Fig 3 on PDF page index 11 (`doc[11]`; owner-password protected, authenticate
with empty string).

Fig 3 is an 8-panel grid of stability diagrams in the (Re, α) plane showing
contours of the temporal growth rate c_i. We digitize only the **outermost
c_i = 0 (neutral) curve** for M = 2, 3, 4, 6.

## CORRECTED panel mapping (this was wrong in the prior reference)

Subfigure letters sit at the **lower-left, BELOW** each panel:

| label | Mach | (the letter above a panel belongs to the panel above it) |
|-------|------|----------------------------------------------------------|
| a | M0  |
| **b** | **M2** |
| **c** | **M3** |
| **d** | **M4** |
| **e** | **M6** |
| f | M7  |
| g | M8  |
| h | M10 |

The label self-check was performed by rendering each panel clip extended
downward by +20 pt to include the label, and confirming the visible letter:
b→M2, c→M3, d→M4, e→M6. **All four passed.**

## Verified panel clips (fitz, `Matrix(6,6)`) and PER-PANEL axis ranges

| Mach | clip Rect (x0,y0,x1,y1) | Re range | α range |
|------|-------------------------|----------|---------|
| M2 (b) | (300, 73, 545, 213)  | 0 .. 5000 | **0 .. 0.08** |
| M3 (c) | (63, 220, 298, 358)  | 0 .. 5000 | **0 .. 0.08** |
| M4 (d) | (300, 220, 545, 358) | 0 .. 5000 | **0 .. 0.4**  |
| M6 (e) | (63, 366, 298, 504)  | 0 .. 5000 | **0 .. 0.4**  |

**The α axis is 0..0.08 for M2/M3 but 0..0.4 for M4/M6.** Calibration is done
PER PANEL from the auto-detected axis-frame pixels (the black box edges in the
6× render):

| Mach | left px | right px | top px | bottom px | α_max |
|------|---------|----------|--------|-----------|-------|
| M2 | 180.5 | 1254.5 | 140.5 | 757.5 | 0.08 |
| M3 | 280.5 | 1353.5 | 128.5 | 744.0 | 0.08 |
| M4 | 179.5 | 1255.5 | 128.5 | 744.0 | 0.40 |
| M6 | 278.0 | 1355.5 | 120.5 | 737.5 | 0.40 |

(left→Re=0, right→Re=5000, bottom→α=0, top→α=α_max.)

## What the prior reference got WRONG

The previous attempt (do NOT trust any pre-existing `*_v2` files predating this
run, nor the old `ozgen_fig3_M{2,3,4,6}_neutral.csv`):

1. **Panels off by one row** — it read the panel one row up/down from the
   correct one, i.e. it mis-identified which subfigure corresponds to which
   Mach number (the labels sit *below* their panel, which is easy to misread).
2. **Wrong α axis range** — it assumed α = 0 .. 0.4 for **all** panels. In fact
   M2 and M3 use α = 0 .. 0.08. Using 0.4 for M2/M3 compresses the curve by 5×
   and places it far too low.

## Digitized topology (this v2, all verified by overlay read-back)

CSV columns: `lobe, Re, alpha, mode`.
- `lobe` ∈ {upper, lower} = the upper/lower **branch** of a curve.
- `mode` identifies the instability mode AND (for M4) which lobe:
  - M2, M3: `first` (first mode only).
  - M4: two well-separated open lobes. `mode=first` rows = the **lower
    (first-mode) lobe**; `mode=second` rows = the **upper (second-mode) lobe**.
    So (lobe=upper/lower) × (mode=first/second) gives all four branches.
  - M6: `mode=second` = upper second-mode branch; `mode=first` = lower
    first-mode branch.

Branch ranges (Re → Re ; α → α):

- **M2** — single open first-mode band. upper: Re 375→4975, α 0.0705→0.0222
  (peaks at the nose ~Re 375 and decreases rightward). lower: Re 375→4975,
  α 0.048→0.0032. Both open to Re=5000.
- **M3** — first mode only, very weak. The outermost c_i=0 curve has a **left
  closed lobe** (Re~700–2000, α top 0.027→0.019, bottom 0.025→0.011) plus a
  **right open band with a notch/kink near Re~2400**. upper (right band):
  Re 2300→4900, α ~0.033→0.0467. lower (envelope): Re 700→4950, α 0.025→0.0025.
- **M4** — TWO well-separated open lobes (large stable gap). LOWER first-mode
  lobe: nose ~(Re 1250, α 0.043), upper branch →0.080, lower branch →0.017 at
  Re~4950. UPPER second-mode lobe: nose ~(Re 1200, α 0.325–0.337), upper branch
  →0.375, lower branch →0.306 at Re~4950.
- **M6** — connected open band. UPPER second-mode branch rises from the nose
  tip (~Re 180, α 0.155) to ~0.205 at Re=5000. LOWER first-mode branch from the
  notch ~(Re 850, α 0.048) descending to ~0.018 at Re=5000.

Inner growth contours (0.001, 0.002, 0.00015, …) and all text labels /
leader-lines were excluded. Leader lines from the "c_i = 0" / "0.00015" text
intrude as spurious topmost crossings; they were rejected via continuity
tracing of the upper branches.

## Verification (Task B)

For each Mach the digitized points were overlaid on the rendered panel
(`imshow` with extent calibrated to the axis box) and read back:
`verification/first_mode/_ozgen_refdigitize/_verify2_M{2,3,4,6}.png` (+ zoom
crops `_zoom_M3_notch.png`, `_zoom_M4_lobenose.png`). Every point sits on the
c_i=0 outermost contour.

Pipeline scripts: `_run_digitize_v2.py`, `_digitize_all_v2.py`, `_verify_v2.py`
under `verification/first_mode/_ozgen_refdigitize/`.
