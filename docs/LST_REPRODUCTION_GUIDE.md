# LST Reproduction Guide

This repository is trying to do two things at once:

1. implement a reusable linear-stability toolkit in [`lst/`](../lst)
2. reproduce figures and trends from Mack's AGARD report and Ozgen & Kircali (2008)

The important distinction is that "solver correctness" and "paper reproduction correctness" are not the same problem. The shared solver may be numerically reasonable while a chapter script still reproduces the wrong quantity, the wrong normalization, or a proxy problem.

For the current paper-by-paper trust status, see
[`docs/PAPER_ALIGNMENT_AUDIT.md`](./PAPER_ALIGNMENT_AUDIT.md).

For the current figure-by-figure status and remaining gap list, see
[`docs/FIGURE_GAP_MATRIX.md`](./FIGURE_GAP_MATRIX.md).

For the shared workflow/API surface added during the flat-plate unification
work, see [`docs/LST_API_CHEATSHEET.md`](./LST_API_CHEATSHEET.md).

## 1. What the Main Stability Quantities Mean

For a normal mode

`q'(x, y, z, t) = q_hat(y) exp(i(alpha x + beta z - omega t))`

the standard quantities are:

- `alpha = alpha_r + i alpha_i`: streamwise wavenumber
- `beta`: spanwise wavenumber
- `omega = omega_r + i omega_i`: frequency
- `c = omega / alpha = c_r + i c_i`: phase speed
- `sigma = -alpha_i`: spatial growth rate

Two formulations matter.

### Temporal formulation

Hold `alpha` and `beta` real, solve for complex `omega` or `c`.

- unstable if `omega_i > 0`
- equivalently unstable if `c_i > 0` for `alpha_r > 0`
- temporal growth curve at fixed `Re`: `omega_i(alpha)` or `c_i(alpha)`

This is the main robust path currently implemented in the repo.

For compressible oblique first-mode work, the best current branch-tracking path
is the exact first-order shooting workflow in [`lst/analysis.py`](../lst/analysis.py):

- `search_temporal_roots_3d_shooting`
- `temporal_growth_scan_3d_shooting`
- `temporal_growth_scan_3d_shooting_from_anchor`
- `temporal_neutral_points_from_scan`

### Spatial formulation

Hold `omega` and `beta` real, solve for complex `alpha`.

- unstable if `alpha_i < 0`
- spatial growth rate is `sigma = -alpha_i > 0`
- spatial growth curve at fixed `Re`: `sigma(omega)` or `sigma(alpha_r)`

This is what transition calculations and many neutral curves ultimately want.

## 2. What a Neutral Curve Actually Is

A neutral curve is the zero-growth boundary of an unstable pocket.

### Temporal neutral curve

At fixed `Re`, sweep `alpha` or `omega` and find where `omega_i = 0`.

For a given `Re`, there are usually two roots:

- lower branch
- upper branch

The unstable band lies between them.

### Spatial neutral curve

At fixed `Re`, sweep `omega` or `alpha_r` and find where `alpha_i = 0`.

Again, two roots usually appear because the unstable region is bounded.

So when you say "at a given boundary layer and a given x location the growth rate is zero and there are two solutions", that is exactly the lower/upper neutral branches.

## 3. How to Compute the Curves in Practice

### To get a temporal growth curve

1. choose the base flow at the station of interest
2. choose `Re`, `Ma`, wall condition, and wave angle
3. sweep `alpha`
4. solve the temporal EVP for each `alpha`
5. isolate the physical discrete mode from the continuous spectrum
6. plot `omega_i = alpha c_i` or `c_i`

### To get a temporal neutral curve

1. repeat the temporal sweep over a grid of `(Re, alpha)` or `(Re, omega)`
2. evaluate `omega_i`
3. find the contour `omega_i = 0`

### To get a spatial growth curve

1. choose the base flow and `Re`
2. sweep `omega`
3. solve for complex `alpha`
4. plot `sigma = -alpha_i`

### To get a spatial neutral curve

1. repeat the spatial solve over `(Re, omega)` or `(Re, alpha_r)`
2. extract `alpha_i`
3. find the contour `alpha_i = 0`

## 4. What the Repo Solves Well Right Now

### Strongest block

- incompressible Orr-Sommerfeld in [`lst/solver.py`](../lst/solver.py)
- validated against Poiseuille and Blasius in [`validation/test_orr_sommerfeld.py`](../validation/test_orr_sommerfeld.py)

### Usable compressible block

- 2D temporal compressible EVP in [`lst/solver.py`](../lst/solver.py)
- explicit `delta* <-> L*` conversion helpers in [`lst/scales.py`](../lst/scales.py)
- mode filtering by convergence across resolutions
- refined spatial growth path in [`lst/analysis.py`](../lst/analysis.py)
  - temporal solve
  - Gaster-style initial guess
  - complex-`alpha` refinement
- oblique-wave temporal solver in [`lst/solver.py`](../lst/solver.py)
  - full 8th-order path
  - Mack-style 6th-order switch obtained by dropping the single spanwise dissipation feedback term
- Appendix-B freestream decay basis / leakage tools in [`lst/asymptotic.py`](../lst/asymptotic.py)
- experimental Appendix-A/B bounded shooting tools in [`lst/mack_shooting.py`](../lst/mack_shooting.py)

This refined spatial route is now the default in the shared analysis utilities because the old companion-form quadratic EVP was returning near-neutral or obviously less physical branches in some cases.

## 5. Where the Current Reproduction Still Breaks

### Mean-flow and condition mismatch

The shared compressible mean flow in [`lst/baseflow.py`](../lst/baseflow.py)
now solves the coupled velocity/temperature boundary-value problem and supports
Mack's Appendix-A air transport law, but paper reproduction still fails if the
wrong external temperature schedule is used.

Two Mack condition sets are now separated explicitly in
[`lst/mack_conditions.py`](../lst/mack_conditions.py):

- `table_11_1`: the inferred schedule that reproduces Table 11.1 thicknesses
- `wind_tunnel`: the figure-caption setup used through Chapters 9-11

This distinction matters because the Chapter 6 thickness table and the Chapter
9-11 figure captions are not using the same `T_1^*` values.

The first-order Eq. 8.9 / Appendix-A reduction is now exact in the
`mack_shooting` path. In particular, the pressure-row coefficients `a43` and
`a46` had to be corrected to the forms recovered by direct reduction from
Eq. 8.9b; the printed Appendix-A transcription does not match the reduced
operator in those two entries. The remaining mismatch is no longer at the
first-order algebra level. It is in the production temporal solver and the
resulting Chapter 10 figure / Table 10.1 reproduction.

### Figure-layer mismatch

Several chapter scripts do not reproduce the literal paper figure even when the computed trend is physically plausible.

Typical failure modes:

- wrong axis variable
- wrong normalization (`delta*` scale vs `L*` scale)
- temporal quantity shown where the paper uses spatial quantity
- 3D result replaced by a proxy transform
- stale generated images left beside current ones

## 6. The Key Scaling That Keeps Causing Confusion

Mack's chapter figures are commonly expressed using the Falkner-Skan length scale

`L* = sqrt(nu_e x / U_e)`

and Reynolds number

`R = U_e L* / nu_e = sqrt(Re_x)`.

Many internal repo calculations, however, are carried out on `y / delta*` grids because that is convenient for the collocation solver.

That means every paper-comparison script must be explicit about which scale it is using:

- `alpha` based on `delta*`
- `alpha` based on `L*`
- `R` based on `L*`
- wall-normal coordinate based on `delta*`, `delta`, or `L*`

If that conversion is not explicit, the figure may look plausible but still be quantitatively wrong.

## 7. How to Think About Mach 6 Specifically

For a Mach-6 adiabatic or weakly heated flat-plate boundary layer:

- the second mode is usually the dominant 2D instability
- a temporal scan should show a discrete high-`c_r` mode with `c_r` close to 1
- the unstable band is bounded by two neutral points
- a spatial growth curve should be reported as `sigma = -alpha_i`

For the first mode:

- at lower Mach numbers it is often most unstable for oblique waves
- at high Mach numbers it can still exist, but the second mode usually dominates 2D amplification

## 8. Current Repo Workflow by Area

### Shared library

- [`lst/baseflow.py`](../lst/baseflow.py): mean-flow profiles
- [`lst/baseflow.py`](../lst/baseflow.py): now contains both the shared Mack and shared Ozgen flat-plate builders
- [`lst/mack_conditions.py`](../lst/mack_conditions.py): Mack Table 11.1 vs figure-caption condition helpers
- [`lst/equations.py`](../lst/equations.py): compressible matrix assembly
- [`lst/solver.py`](../lst/solver.py): temporal, spatial, and mode-tracking solvers
- [`lst/asymptotic.py`](../lst/asymptotic.py): Appendix-B freestream basis and leakage residuals
- [`lst/mack_shooting.py`](../lst/mack_shooting.py): experimental first-order Appendix-A/B shooting path
- [`lst/analysis.py`](../lst/analysis.py): sweeps, neutral maps, N-factors, exact-shooting helpers, and the shared high-level growth / neutral / critical-Re / wave-angle workflows
- [`lst/reference_data.py`](../lst/reference_data.py): loaders for the shared paper target registry and numeric reference tables
- [`reference_data/`](../reference_data): shared Mack/Ozgen target metadata and numeric reference CSVs
- [`validation/diagnose_mack_table_10_1.py`](../validation/diagnose_mack_table_10_1.py): current 6th-/8th-order oblique-wave diagnostic against Mack Table 10.1
- [`validation/diagnose_oblique_mode_selection.py`](../validation/diagnose_oblique_mode_selection.py): compares reduced-EVP candidates against Appendix-B leakage and QR-stabilized shooting residuals for hard Chapter 10 cases
- [`validation/diagnose_oblique_continuation.py`](../validation/diagnose_oblique_continuation.py): tracks representative first-mode families through alpha and beta continuation to separate mode-selection errors from deeper operator mismatches
- [`validation/diagnose_low_mach_shooting_root.py`](../validation/diagnose_low_mach_shooting_root.py): shows that the exact first-order shooting system has a low-Mach amplified root absent from the reduced-EVP spectrum
- [`validation/diagnose_low_mid_table_10_1_shooting.py`](../validation/diagnose_low_mid_table_10_1_shooting.py): compares the exact first-order shooting branch against the low/mid-Mach `M=1.3`, `1.6`, and `2.2` Table 10.1 families
- [`validation/diagnose_low_mach_shooting_growth_scan.py`](../validation/diagnose_low_mach_shooting_growth_scan.py): continues an exact low-Mach first-mode branch away from an interior anchor and extracts temporal neutral points from the resulting `omega_i(alpha)` scan
- [`validation/test_appendix_a_reduction.py`](../validation/test_appendix_a_reduction.py): reconstructs the Eq. 8.9 / Appendix-A first-order system from the reduced collocation equations and verifies exact agreement
- [`validation/test_asymptotic.py`](../validation/test_asymptotic.py): sanity checks for Appendix-A/B helper algebra

### Mack chapter scripts

- [`chapters/ch06_compressible_formulation/ch06.py`](../chapters/ch06_compressible_formulation/ch06.py): mean-flow scaling diagnostics
- [`chapters/ch09_compressible_inviscid/ch09.py`](../chapters/ch09_compressible_inviscid/ch09.py): inviscid-mode exploration
- [`chapters/ch10_compressible_viscous/ch10.py`](../chapters/ch10_compressible_viscous/ch10.py): viscous compressible figure scaffolding

### Ozgen reproduction

- [`chapters/ozgen_kircali_2008/ozgen.py`](../chapters/ozgen_kircali_2008/ozgen.py)

This path is the closest thing in the repo to a paper-reproduction workflow, but it still mixes faithful 2D work with proxy 3D transformations.

## 9. What to Trust and What Not to Trust

Trust more:

- incompressible validation
- temporal discrete-mode detection logic
- refined spatial growth scans from the shared analysis layer
- Appendix-B decay-basis algebra and the corrected `a66` conductivity-gradient factor

Trust less:

- exact Chapter 6 scaling against Mack
- any figure whose script is not computing the same quantity as the caption
- Ozgen 3D results generated through `cos(psi)` transformations instead of a true 3D compressible EVP
- exact Mack Chapter 9 oblique-wave reproduction from the current Chapter 9 script
- exact Mack Table 10.1 or Chapter 10 oblique-wave amplitudes from the current full 3D viscous solver
- the experimental Appendix-A/B shooting residual as a production eigenvalue solver; it can now expose low-Mach roots missing from the reduced EVP, but it is still not a robust general-purpose replacement solver
- low-/mid-Mach branch selection from the reduced 3D EVP when the QR-stabilized shooting residual prefers a different candidate family

At low Mach numbers, the current evidence is stronger than a branch-selection
warning. The exact first-order shooting system has a positive-growth root that
moves toward Mack's Table 10.1 value as `y_max` increases, while the reduced
EVP and the reduced asymptotic refinement both return to a much larger
high-growth branch even when seeded with that exact shooting root. That means
the reduced temporal formulation is still missing a physically relevant mode
family in the low-Mach regime.

## 10. What "Done" Looks Like for This Repo

To claim real reproduction rather than exploratory agreement, the repo needs:

1. one authoritative mean-flow implementation for each paper model
2. explicit and correct `delta* <-> L*` conversions
3. mode tracking that follows the same branch across parameter sweeps
4. figure scripts that compute the exact quantity named by the paper caption
5. no stale generated outputs mixed into the active folders

That is the standard to use when deciding whether a result here is "understood" versus only "looks similar".
