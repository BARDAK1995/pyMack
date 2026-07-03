# Malik (1990) Fig. 4 — pixel→data calibration

Source crop (local-only, gitignored):
`refPapers/NewPapers/figures/malik1990_fig4_eigenfunctions_M10.png`
Image size: 912 (W) x 857 (H) pixels.

Axis frame lines detected from dark-pixel column/row sums:
- vertical (y-axis) line at pixel column ≈ 84
- horizontal (x-axis, value=-1.2) line at pixel row ≈ 766

## X axis (wall-normal coordinate `y`, range 0..30)
Inward tick marks detected above the x-axis line (rows 752..764).
Clean tick pixel-columns vs data value:

| y value | pixel col |
|--------:|----------:|
| 0       | 83.0      |
| 5       | 208.0     |
| 10      | 333.0     |
| 15      | 459.5     |
| 20      | 585.0     |
| 25      | 709.5     |
| 30      | 834.0     |

Linear fit:  `y = 0.0399086 * col - 3.31234`  (≈ 25.06 px per unit y).
Max residual over the 7 ticks: 0.034 in y-units.

## Value axis (eigenfunction amplitude, range -1.2..+1.2)
Inward tick marks detected right of the y-axis line (cols 87..97).
Clean tick pixel-rows vs data value:

| value | pixel row |
|------:|----------:|
| +1.2  | 14.0      |
| +0.8  | 139.5     |
| +0.4  | 268.0     |
|  0.0  | 395.0     |
| -0.4  | 518.5 (predicted 517.2) |
| -0.8  | 641.0     |
| -1.2  | 767.5     |

Linear fit:  `value = -0.00318828 * row + 1.24899`  (≈ 125.46 px per 0.4 in value).
Max residual over the clean ticks: 0.010 in value units.

## Verification checks
- Interior tick y=15 lands at col 459.5 -> fit gives y = 15.026 (OK).
- Interior tick value=0.4 lands at row 268 -> fit gives value = 0.396 (OK).
- Predicted value=0 row 395 and value=-0.4 row 517.2 match detected ticks.
- Digitized T_hat_r peak lands at (y≈13, value≈+1.0) as expected (Malik normalises T_hat_r peak to +1.0).
