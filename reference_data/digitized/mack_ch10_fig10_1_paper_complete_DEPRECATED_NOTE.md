# Deprecation note: `mack_ch10_fig10_1_M{16,22}_paper_complete.csv`

**Date:** 2026-06-18

## What was wrong

`mack_ch10_fig10_1_M16_paper_complete.csv` and `..._M22_paper_complete.csv` were
used as the "Complete Equations" reference for the Mack (1984) Fig 10.1 first-mode
neutral-curve verification. Re-inspection of the source figure (AGARD-R-709, p.90)
shows that these files **do not faithfully trace the Complete-Equations neutral
loop**:

1. **They are open single-valued branches, not closed loops.** Fig 10.1 plots a
   *neutral loop* per theory: at each R above the critical R there are TWO neutral
   frequencies (a lower/onset and an upper/cutoff). The digitized `_complete` (and
   sibling `_dunn_numerical`, `_dunn_asymptotic`) files contain only one
   monotonically-decreasing branch each — the loops' **upper/cutoff branches** at
   high R — with **no lower/onset branch and no nose**.

2. **Their low-R tails follow no real curve.** Each file extends to R ~= 50-90
   (e.g. M16 `_complete` starts at R=80, F=1.92), but at those Reynolds numbers the
   Complete-Equations loop does not exist — its nose (critical R) is at R ~= 215
   (M=1.6) / R ~= 300 (M=2.2). The low-R points lie in empty figure space; the
   digitizer extrapolated the upper branch instead of turning around the nose.

3. **Net effect on the verdict.** pyMack correctly produces a *closed loop*
   (F_lower, F_upper per R) with a nose at R ~= 215 / 300. Scoring that loop against
   a mislabeled open branch with a spurious R~80 tail produced an artificial
   "topology gap / critical-R mismatch" and inflated relative errors (the prior
   verdict reported 30-166% disagreement). Against the correctly re-digitized
   Complete-Equations loop the agreement is far better (see corrected verdicts).

### Diagnosis nuance vs. the prior investigation

The prior note said the `_complete` file "actually traces the Dunn-Lin asymptotic
loop." That is **not** what the overlay shows. At high R the `_complete` file tracks
the **innermost (Complete-Equations) UPPER branch** correctly (it is the lowest of
the three upper branches, as Complete should be). The real defects are (a) it omits
the lower/onset branch entirely and (b) its low-R tail diverges into empty space
past the nose. So `_complete` is best described as *an incomplete (upper-branch-only)
trace of the Complete-Equations loop with a non-physical low-R tail* — not the
asymptotic curve. The three files' relative ordering (asymptotic highest, numerical
middle, complete lowest) is correct; their topology is not.

## Replacement

Use the re-digitized full loops:

- `mack_ch10_fig10_1_M16_paper_complete_equations.csv`
- `mack_ch10_fig10_1_M22_paper_complete_equations.csv`

These contain both branches (`branch` column: lower / upper / nose) plus the nose
(critical R). Pixel calibration and per-point digitization uncertainty are recorded
in each file's header.

The old `_paper_complete.csv` files are **retained unchanged** (not overwritten) for
provenance, but should no longer be used as the Complete-Equations reference.
