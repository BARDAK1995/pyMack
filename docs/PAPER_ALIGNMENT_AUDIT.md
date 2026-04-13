# Paper Alignment Audit

This document is the current paper-by-paper status check for the repository.
It is intentionally narrower than the README: it records what is actually
validated against Mack's AGARD report and Ozgen & Kircali (2008), what is only
approximate, and what remains unresolved.

For the authoritative figure-by-figure audit and cleanup status, see
[`docs/FIGURE_GAP_MATRIX.md`](./FIGURE_GAP_MATRIX.md). This file remains the
short paper-family summary; the gap matrix is the source of truth for which
generated outputs are validated, provisional, diagnostic, mismapped, proxy, or
still missing.

## Scope

The repo contains three different layers of work:

1. shared solver infrastructure in [`lst/`](../lst)
2. validation/diagnostic scripts in [`validation/`](../validation)
3. chapter-style reproduction scripts in [`chapters/`](../chapters)

Those layers do not all have the same trust level. A mathematically correct
helper can coexist with a chapter script that still plots the wrong quantity or
tracks the wrong branch.

## Status Summary

### Exact or very strong

- Mack mean-flow condition split:
  [`lst/mack_conditions.py`](../lst/mack_conditions.py) now separates the
  Table 11.1 edge-temperature schedule from the Chapter 9-11 wind-tunnel
  figure-caption conditions.
- Mack-style compressible mean flow:
  [`validation/test_mack_mean_flow.py`](../validation/test_mack_mean_flow.py)
  confirms the current base-flow reconstruction against the intended Chapter 6
  thickness/shape targets.
- Appendix-B freestream algebra:
  [`validation/test_asymptotic.py`](../validation/test_asymptotic.py) verifies
  the corrected conductivity-gradient factor and now also verifies that the
  freestream decay basis is an exact eigensubspace of the uniform first-order
  matrix. The earlier closed-form `lambda_1` branch was not consistent with the
  exact uniform matrix and has been replaced by a basis derived directly from
  that exact freestream operator.
- Eq. 8.9 to Appendix-A first-order reduction:
  [`validation/test_appendix_a_reduction.py`](../validation/test_appendix_a_reduction.py)
  now shows exact agreement between the reduced collocation operator and the
  first-order shooting matrix assembled in [`lst/mack_shooting.py`](../lst/mack_shooting.py).

### Strong evidence, but not exact reproduction yet

- Low-/mid-Mach first-mode growth rates from the exact first-order shooting
  system are substantially closer to Mack Table 10.1 than the reduced 3D EVP.
- [`validation/diagnose_low_mach_shooting_root.py`](../validation/diagnose_low_mach_shooting_root.py)
  shows that the exact shooting system contains a low-Mach amplified branch that
  the reduced finite-domain EVP does not recover.
- [`validation/diagnose_low_mid_table_10_1_shooting.py`](../validation/diagnose_low_mid_table_10_1_shooting.py)
  extends that comparison across the `M=1.3`, `1.6`, and `2.2` families in
  Mack Table 10.1. The exact shooting branch still underpredicts the eighth-
  order table values by roughly 10-25%, but it is clearly on the physically
  relevant family.

### Not reliable as exact paper reproduction

- The reduced oblique temporal EVP in [`lst/solver.py`](../lst/solver.py) is
  still not reliable for low-Mach Chapter 10 first-mode data.
- [`chapters/ch10_compressible_viscous/ch10.py`](../chapters/ch10_compressible_viscous/ch10.py)
  still uses the reduced collocation path and should be treated as figure
  scaffolding, not as a validated Mack Chapter 10 reproduction.
- [`chapters/ozgen_kircali_2008/ozgen.py`](../chapters/ozgen_kircali_2008/ozgen.py)
  now uses a true 3D reduced-EVP path for the compressible oblique figures,
  but those outputs are still provisional and should not yet be read as exact
  Ozgen reproductions.
- The shared temporal solvers and exact first-order shooting path now enforce
  the adiabatic thermal wall condition consistently for Mack/Ozgen flat-plate
  cases. Older diagnostic conclusions drawn from mixed adiabatic-baseflow /
  isothermal-perturbation runs should be treated as superseded.
- In particular, Ozgen Fig. 8 and Fig. 10 are now understood to be mismapped
  in the current chapter script: Fig. 8 is a spatial-style `(Re_delta, sigma_i)`
  target with constant-`omega_r` contours, while Fig. 10 is a neutral-curve
  comparison rather than a temporal contour plot.
- The first generated Ozgen Fig. 6 / Fig. 7 oblique outputs were also removed:
  after correcting the shared `L* <-> delta*` conversions, it became clear that
  the old `y_max=10/12` values on `L*` scale truncated the domain inside the
  boundary layer for moderate and high Mach number cases.
- [`chapters/ch09_compressible_inviscid/ch09.py`](../chapters/ch09_compressible_inviscid/ch09.py)
  contains useful diagnostics, but not every generated figure is a literal Mack
  caption match.

## Mack-Specific Findings

### Mean flow

The most important correction was separating Mack's condition sets. The Table
11.1 thickness comparisons and the Chapter 9-11 figure captions are not using
the same freestream temperature schedule. Treating them as a single condition
set contaminates every downstream comparison.

### Appendix-A pressure row

The current shooting matrix uses the pressure-row coefficients recovered by
direct reduction of Mack Eq. 8.9b. The printed Appendix-A transcription appears
to miss one factor of `DT` in `a43` and one factor of `1/T` in `a46`.

That statement is not speculative in this repo anymore: the reduction test now
reconstructs the first-order matrix from the reduced operator and matches the
implemented coefficients exactly.

### Chapter 10 first-mode mismatch

The remaining low-Mach Chapter 10 mismatch is now localized much more tightly:

- it is not a mean-flow schedule error for the tested Table 10.1 cases
- it is not just branch selection inside the reduced spectrum
- it is not the Appendix-A first-order algebra

The remaining discrepancy sits in the eigenvalue formulation and finite-domain
boundary treatment used by the production temporal solver, or in what is still
missing between the exact first-order shooting formulation and Mack's full
eighth-order table values.

## New Workflow That Is Worth Trusting

For temporal growth curves and neutral-point work on the first mode, the best
current workflow is:

1. use [`lst.analysis.search_temporal_roots_3d_shooting`](../lst/analysis.py)
   to search exact first-order roots from multiple complex seeds at one anchor
   point
2. continue the chosen branch in `alpha` with
   [`lst.analysis.temporal_growth_scan_3d_shooting`](../lst/analysis.py)
   or continue away from an interior amplified point with
   [`lst.analysis.temporal_growth_scan_3d_shooting_from_anchor`](../lst/analysis.py)
3. extract the lower and upper temporal neutral points from the resulting
   `omega_i(alpha)` scan with
   [`lst.analysis.temporal_neutral_points_from_scan`](../lst/analysis.py)

This does not yet make the repo "finished," but it does give a workflow that
tracks the physically relevant low-Mach branch better than the reduced EVP.

The workflow now has one concrete validated example:
[`validation/diagnose_low_mach_shooting_growth_scan.py`](../validation/diagnose_low_mach_shooting_growth_scan.py)
tracks the `M=1.3`, `R=1500`, `psi=45` first-mode branch and finds temporal
neutral points at approximately `alpha = 0.021139` and `alpha = 0.085670`
on Mack's `L*` scale.

That workflow is now exposed in the shared API through
[`lst.find_temporal_mode_anchor_3d_shooting`](../lst/analysis.py) and
[`lst.trace_temporal_neutral_curve_shooting`](../lst/analysis.py), with a
regression in
[`validation/test_analysis_workflows.py`](../validation/test_analysis_workflows.py).

For Ozgen's oblique `M=4.5`, `psi=60` pilot case, the repo now has a focused
diagnostic in
[`validation/diagnose_ozgen_oblique_domain_and_shooting.py`](../validation/diagnose_ozgen_oblique_domain_and_shooting.py).
That script shows:

1. the legacy `L*` domain height was under-truncated
2. correcting the far-field height moves the reduced oblique root materially
3. enforcing the adiabatic wall condition consistently does not remove the gap
4. replacing the closed-form Appendix-B `lambda_1` branch with the exact
   uniform-matrix decay basis also does not remove the gap
5. the asymptotic-BC refinement still agrees with the unstable reduced root
6. the exact Appendix-A/B shooting solve instead returns only stable roots

So the current Ozgen oblique blocker is no longer a vague "needs better
tracking" issue; it is a concrete mismatch between the reduced/asymptotic
oblique formulation and the exact shooting boundary-value solve.

## What Still Needs Work

1. A production-quality nonlinear eigenvalue solve around the exact first-order
   shooting boundary condition, or an equivalent finite-domain formulation that
   recovers the same branch without manual seed management.
2. Rebuilt Chapter 10 figure scripts that use the trusted branch-tracking path
   rather than the reduced leading eigenvalue alone.
3. A true 3D spatial workflow for Ozgen-style oblique `(Re_delta, sigma_i)`
   maps plus an oblique neutral-curve path so Fig. 8 and Fig. 10 stop relying
   on temporal stand-ins.
4. Cleanup of stale and superseded generated figures so the chapter folders do
   not overstate what has been reproduced.
