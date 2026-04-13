# Figure Gap Matrix

This is the current authoritative figure/result audit for the flat-plate LST
work in this repository. It replaces ad hoc status judgments based only on
whether a PNG exists in a chapter folder.

Status labels used here:

- `validated`: numeric benchmark or regression target exists and the current
  implementation passes it
- `provisional`: topology or trend is useful, but exact paper-level validation
  is still missing
- `diagnostic`: intentionally useful non-paper output
- `mismapped`: current file is not the paper quantity/caption it is being
  compared to
- `proxy`: paper quantity is being approximated by a different problem
- `missing`: no current faithful implementation

## Current Confidence

- The strongest blocks are the incompressible Orr-Sommerfeld core, the shared
  Mack mean-flow reconstruction, the Appendix-A/B algebra checks, and the new
  shared analysis workflow layer in [`lst/analysis.py`](../lst/analysis.py).
- The weakest blocks are the Mack Chapter 9 figure layer, the Mack Chapter 10
  production figure layer, and the Ozgen oblique-wave figure family.
- The repo is not yet at "full mastery" because several production figures are
  still either diagnostics, partial reproductions, or proxy solves rather than
  exact paper quantities.

## Cross-Cutting Findings

- Mack Table 11.1 mean-flow thicknesses are now numerically reproduced by
  [`validation/test_mack_mean_flow.py`](../validation/test_mack_mean_flow.py)
  with worst relative error below 0.5%. The old claim that Mack scaling is
  fundamentally broken is no longer current.
- Shared scale conversions among `eta`, `L*`, `delta*`, and physical `y` now
  live in [`lst/scales.py`](../lst/scales.py). Chapter scripts still need to
  consume those helpers consistently instead of recomputing local conversions.
- The exact first-order oblique shooting path is the current trusted path for
  low-/mid-Mach Mack Chapter 10 first-mode work. The reduced finite-domain 3D
  temporal EVP is still not reliable enough to serve as the production solver
  for those cases.
- Ozgen's shared mean-flow builder is now in
  [`lst/baseflow.py`](../lst/baseflow.py), but its transport closure still
  matches the legacy constant-Pr chapter behavior. Viscosity is on the shared
  Sutherland path; conductivity and heat-capacity laws are not yet implemented
  as the full paper-specific production model.
- Any oblique-wave figure that still uses `alpha/cos(psi)` and
  `Re cos(psi)` should be treated as `proxy`, not paper-faithful 3D LST.

## Mack Audit

### Chapter 2-3

- `Table 3.1 inviscid eigenvalues`: `provisional`
  - Current Rayleigh/eigenvalue path does not yet have digitized numeric
    references in the repo.
  - Previous audit evidence indicated branch/sign mismatch. That needs to be
    rechecked against the actual table data, not by visual similarity.
- [`chapters/ch02_03_formulation/fig3_3_inviscid_damping.png`](../chapters/ch02_03_formulation/fig3_3_inviscid_damping.png): `provisional`
  - The file is active, but its paper numbering and exact table/figure mapping
    still need cleanup.
- [`chapters/ch02_03_formulation/fig_inflection_analysis.png`](../chapters/ch02_03_formulation/fig_inflection_analysis.png): `diagnostic`
- [`chapters/ch02_03_formulation/fig_falkner_skan_inflection.png`](../chapters/ch02_03_formulation/fig_falkner_skan_inflection.png): `diagnostic`

What is lacking:

- digitized Table 3.1 references
- explicit branch locking for the intended inviscid mode family
- consistent figure numbering against the paper/LaTeX source

### Chapter 5

- [`chapters/ch05_incompressible_viscous/fig5_1_blasius_neutral.png`](../chapters/ch05_incompressible_viscous/fig5_1_blasius_neutral.png): `validated`
- [`chapters/ch05_incompressible_viscous/fig5_3_neutral_RF.png`](../chapters/ch05_incompressible_viscous/fig5_3_neutral_RF.png): `validated`
- [`chapters/ch05_incompressible_viscous/fig5_7_spatial_amplification.png`](../chapters/ch05_incompressible_viscous/fig5_7_spatial_amplification.png): `validated_core`
- [`chapters/ch05_incompressible_viscous/fig5_8_max_amplification.png`](../chapters/ch05_incompressible_viscous/fig5_8_max_amplification.png): `validated_core`

What is lacking:

- shared digitized paper overlays so acceptance is numeric rather than visual
- cleanup of chapter-number mapping against Mack's split LaTeX numbering

### Chapter 6 / Table 11.1

- `Table 11.1 displacement thickness`: `validated`
  - Reference data live in
    [`reference_data/mack/table_11_1_delta_star_over_lstar.csv`](../reference_data/mack/table_11_1_delta_star_over_lstar.csv).
  - Validation lives in
    [`validation/test_mack_mean_flow.py`](../validation/test_mack_mean_flow.py).
- [`chapters/ch06_compressible_formulation/fig_mean_flow_mach.png`](../chapters/ch06_compressible_formulation/fig_mean_flow_mach.png): `diagnostic`
- [`chapters/ch06_compressible_formulation/fig_wall_properties.png`](../chapters/ch06_compressible_formulation/fig_wall_properties.png): `diagnostic`

What is lacking:

- explicit regression targets for the generated diagnostic figures
- clear separation in documentation between "validated table" and
  "supporting mean-flow plots"

### Chapter 9

- [`chapters/ch09_compressible_inviscid/fig9_1_generalized_inflection.png`](../chapters/ch09_compressible_inviscid/fig9_1_generalized_inflection.png): `mismapped`
  - Current output is a generalized-inflection diagnostic.
  - Paper Fig. 9.1 is a phase-velocity figure, not this quantity.
- [`chapters/ch09_compressible_inviscid/fig9_6_max_amp_vs_mach.png`](../chapters/ch09_compressible_inviscid/fig9_6_max_amp_vs_mach.png): `mismapped`
  - Current script tracks `c_i`, not the paper quantity `omega_i`, and only a
    reduced subset of the mode family.
- [`chapters/ch09_compressible_inviscid/fig_relative_mach.png`](../chapters/ch09_compressible_inviscid/fig_relative_mach.png): `diagnostic`
- [`chapters/ch09_compressible_inviscid/fig_oblique_wave_angle_M45_experimental.png`](../chapters/ch09_compressible_inviscid/fig_oblique_wave_angle_M45_experimental.png): `diagnostic`
- [`chapters/ch09_compressible_inviscid/fig_mode_branches_M45_diagnostic.png`](../chapters/ch09_compressible_inviscid/fig_mode_branches_M45_diagnostic.png): `diagnostic`
- [`chapters/ch09_compressible_inviscid/fig_wall_cooling_diagnostic.png`](../chapters/ch09_compressible_inviscid/fig_wall_cooling_diagnostic.png): `diagnostic`

What is lacking:

- true paper remap for Figs. 9.1, 9.3, 9.4, 9.5, 9.8, 9.12
- a clear distinction between inviscid paper figures and viscous diagnostic
  plots generated during solver exploration
- digitized reference data for the Chapter 9 figure family

Bottom line:

- Chapter 9 is the least trustworthy Mack chapter at the figure layer.
- The current chapter outputs are still useful for understanding the mode
  structure, but not yet as literal paper reproductions.

### Chapter 10

- [`chapters/ch10_compressible_viscous/fig10_1_neutral_frequency.png`](../chapters/ch10_compressible_viscous/fig10_1_neutral_frequency.png): `partial`
  - Missing the theory-comparison families discussed in the paper.
- [`chapters/ch10_compressible_viscous/fig10_2_neutral_wavenumber.png`](../chapters/ch10_compressible_viscous/fig10_2_neutral_wavenumber.png): `partial`
  - Current script uses `R` on the abscissa; the paper treatment emphasizes
    the reciprocal-Reynolds-number view for this comparison.
- [`chapters/ch10_compressible_viscous/fig10_3_max_growth_vs_R.png`](../chapters/ch10_compressible_viscous/fig10_3_max_growth_vs_R.png): `partial`
  - Paper coverage includes 3D content; current script is explicitly 2D only.
- [`chapters/ch10_compressible_viscous/fig10_4_first_mode_growth.png`](../chapters/ch10_compressible_viscous/fig10_4_first_mode_growth.png): `partial`
  - Needs true wave-angle optimization / continuation rather than 2D scans.
- [`chapters/ch10_compressible_viscous/fig10_5_neutral_two_modes.png`](../chapters/ch10_compressible_viscous/fig10_5_neutral_two_modes.png): `provisional`
  - Structurally close, but still tied to the reduced collocation production
    path instead of the trusted exact branch-tracked path.
- [`chapters/ch10_compressible_viscous/fig10_6_second_mode_growth.png`](../chapters/ch10_compressible_viscous/fig10_6_second_mode_growth.png): `provisional`
- [`chapters/ch10_compressible_viscous/fig10_9_wall_cooling.png`](../chapters/ch10_compressible_viscous/fig10_9_wall_cooling.png): `mismapped`
  - Current layout is a diagnostic split by mode, not the paper figure
    structure.
- [`chapters/ch10_compressible_viscous/fig10_10_mach_effect.png`](../chapters/ch10_compressible_viscous/fig10_10_mach_effect.png): `provisional`
- [`chapters/ch10_compressible_viscous/fig10_11_spatial_mach.png`](../chapters/ch10_compressible_viscous/fig10_11_spatial_mach.png): `proxy`
  - Computed through a Gaster relation, not an actual spatial solve.

What is lacking:

- replacement of the Chapter 10 production layer with the trusted exact
  branch-tracked temporal workflow for low-/mid-Mach first-mode work
- direct spatial solves for the figures that are spatial in the paper
- digitized figure targets and acceptance metrics for the chapter family
- explicit 3D branch continuation where the paper is not 2D

Bottom line:

- Chapter 10 is the highest-value remaining Mack rebuild.
- The algebra is now in much better shape than the figure layer.

## Ozgen Audit

### Figure 1

- [`chapters/ozgen_kircali_2008/fig1_profiles.png`](../chapters/ozgen_kircali_2008/fig1_profiles.png): `provisional`
  - Now runs through the shared `make_ozgen_profile(...)` builder.
  - Still needs digitized paper overlays and explicit validation of the exact
    transport-law interpretation used in the paper.

### Figure 2

- `Figure 2 literature profile comparison`: `missing`
  - No current script or generated output.

### Figure 3

- [`chapters/ozgen_kircali_2008/fig3_stability_diagrams.png`](../chapters/ozgen_kircali_2008/fig3_stability_diagrams.png): `provisional`
  - Main path is still the coarse contour workflow.
- [`chapters/ozgen_kircali_2008/fig3_neutral_curves.png`](../chapters/ozgen_kircali_2008/fig3_neutral_curves.png): `diagnostic`
  - This is closer to the right neutral-tracing method, but it is not yet the
    main production route for the paper figure family.

What is lacking:

- integrate direct neutral tracing into the production path
- validate branch count, lower/upper branches, and nose location numerically

### Figure 4

- [`chapters/ozgen_kircali_2008/fig4_critical_Re.png`](../chapters/ozgen_kircali_2008/fig4_critical_Re.png): `provisional`
  - Current implementation finds first unstable points on a coarse scan rather
    than solving the actual neutral nose accurately.

### Figure 5

- `Figure 5 M=8 comparison`: `missing`
  - No current script or generated output.

### Figures 6, 7, 8, 10

- `Ozgen Fig. 6 paper target`: `missing`
- `Ozgen Fig. 7 paper target`: `missing`
- `Ozgen Fig. 8 paper target`: `missing`
- `Ozgen Fig. 10 paper target`: `missing`
- `latest Fig. 8 / Fig. 10 temporal diagnostic PNGs`: `removed as stale`

What changed:

- the production chapter path for the compressible Mach-number cases no longer
  uses the `cos(psi)` effective-parameter proxy
- the current implementation now evaluates the true 3D temporal solver with
  `beta = alpha * tan(psi)` and mode-family filtering through wave-speed
  windows plus freestream-leakage screening
- `lst.analysis` now exposes `spatial_growth_map(...)`, which is the shared
  map-level API needed for the spatial figure family
- the shared scale conversions and the Ozgen oblique domain-height helper were
  corrected so `L*`-scaled solves now reach the intended `10-12 delta*`
  far-field height instead of truncating inside the boundary layer
- the adiabatic thermal wall condition is now enforced consistently in the
  reduced EVP and exact-shooting paths for flat-plate Mack/Ozgen cases
- the Appendix-B freestream decay basis is now built from the exact uniform
  first-order matrix, fixing the earlier closed-form `lambda_1` inconsistency

Why they are still not done:

- the previously generated Fig. 6 and Fig. 7 outputs were removed because they
  used under-scaled `L*` domains (`y_max=10/12` in `L*`, not `10-12 delta*`)
  and therefore did not reach the freestream at moderate and high Mach number
- figure-level numeric regression against digitized Ozgen curves is still
  missing
- the current branch family is selected from reduced-EVP mode windows, not yet
  by an exact first-order shooting continuation for every plotted point
- Figure 7 still extracts critical Reynolds number from a coarse growth-onset
  series, so the wave-angle optimum is only provisional
- Figure 8 is not a temporal `(Re, alpha, c_i)` target in the paper; it
  requires a spatial oblique workflow in `(Re_delta, sigma_i)` with
  constant-`omega_r` contours
- Figure 10 is a neutral-curve comparison target, not a temporal contour map,
  so the current 3D output is only a diagnostic stand-in
- the new exact-shooting neutral-tracing API is validated on Mack low-Mach
  first-mode work, but a first Ozgen `M=4.5`, `psi=60` pilot still converges
  to stable exact roots from representative reduced-EVP seeds, so the Ozgen
  oblique first-mode branch is not yet ported to the trusted shooting path
- [`validation/diagnose_ozgen_oblique_domain_and_shooting.py`](../validation/diagnose_ozgen_oblique_domain_and_shooting.py)
  now records the corrected-domain mismatch explicitly: even after the
  adiabatic wall-condition fix and the Appendix-B freestream-basis fix, the
  unstable reduced / asymptotic-root candidate still has large shooting
  residual, while the exact shooting search returns only stable roots for the
  pilot Ozgen case
- the `psi = 90` limit remains a degenerate `alpha -> 0` panel rather than a
  direct solve at exactly zero streamwise wavenumber

### Figure 9

- [`chapters/ozgen_kircali_2008/fig9_Tref_M4p8.png`](../chapters/ozgen_kircali_2008/fig9_Tref_M4p8.png): `provisional`
  - Current output covers the repo-generated temperature cases.
  - The literature comparison curves cited in the paper are not yet loaded and
    overlaid.

## Equation / Model Gaps Still Open

- Mack Eq. 8.9 to Appendix A/B algebra: substantially closed in the repo.
  Remaining work is production solver fidelity and branch continuation, not the
  reduced first-order derivation itself.
- Mack low-/mid-Mach oblique first-mode production solve: still open.
  The trusted exact shooting branch exists, but the chapter production layer is
  not yet rebuilt on it.
- Ozgen transport model: still open.
  The shared builder currently preserves the legacy chapter behavior with a
  constant-Pr conductivity path. If the paper requires a stricter Eq. 2.37 /
  Eq. 2.38 interpretation, that has to be implemented and revalidated.
- Ozgen true 3D oblique stability: still open.
  No production figure in this family should be called reproduced until the
  `cos(psi)` proxy path is replaced.

## Cleanup Performed In This Audit

Removed dead or stale workflow artifacts:

- `_mack_neutral_worker.py`
- `_neutral_worker.py`
- `automode_log.md`
- `chapters/ozgen_kircali_2008/automode_log.md`
- `chapters/ozgen_kircali_2008/neutral_data/`
- `chapters/ozgen_kircali_2008/fig8_mach_independence_temporal_diagnostic.png`
- `chapters/ozgen_kircali_2008/fig10_Tref_M4p5_3D_temporal_diagnostic.png`
- stale Chapter 10 neutral/cache logs under
  `chapters/ch10_compressible_viscous/`

These paths were not referenced by the active chapter scripts or shared `lst`
APIs and were remnants of older worker-based batch runs.

## Recommended Execution Order From Here

1. Finish figure-level numeric references and regression targets for every
   Mack/Ozgen target already listed in
   [`reference_data/paper_target_registry.json`](../reference_data/paper_target_registry.json).
2. Rebuild Ozgen's transport model and then replace the `cos(psi)` path with a
   true 3D compressible solve plus branch continuation.
3. Rebuild Mack Chapter 10 on top of the trusted exact first-order branch
   tracker and the shared high-level analysis APIs.
4. Remap and rebuild Mack Chapter 9 from the actual paper quantities instead of
   the current diagnostic substitutes.
5. Promote direct neutral-curve and critical-Re extraction into the production
   Ozgen 2D figure path.
6. Add figure-family acceptance tests so a regenerated PNG cannot silently
   drift away from the paper quantity it claims to represent.
