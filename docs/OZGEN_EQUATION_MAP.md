# Ozgen Equation Map

This is the short map from Ozgen & Kircali (2008) to the shared code.

## Mean Flow

Paper equations:

- momentum: Eq. 2.32
- energy: Eq. 2.33
- property laws: Eqs. 2.36, 2.37, 2.38

Shared builder:

- `pymack.make_ozgen_profile(...)`
- backing class: `pymack.baseflow.OzgenFlatPlateProfile`

The current shared implementation solves the coupled Ozgen mean-flow BVP rather
than using the earlier Crocco/Walz-style temperature closure. The paper's
streamfunction identity `F' = rho U`, the `mu/sigma` thermal diffusivity in
Eq. 2.33, and the factor-of-two coefficients in Eqs. 2.32-2.33 are retained
explicitly. The residuals and physical `delta*/L*` integral are covered by
`validation/test_ozgen_mean_flow.py`.

Implemented property model:

- Eq. 2.36 is used for `mu/mu_e`
- Eq. 2.37 is used for `kappa/kappa_e`
- Eq. 2.38 is used for `Cp/Cp_e`
- the profile exposes local `Pr_local = Pr_e (mu/mu_e)(Cp/Cp_e)/(kappa/kappa_e)`
- the solver-facing `kappa` is normalized as `kappa*/(Cp_e mu_e)`, so
  `kappa(y=inf) = 1 / Pr_e`, matching the existing Mack transport convention

## Scales

Use the shared helpers in `pymack.scales` for:

- `delta* / L*`
- `theta / L*`
- `eta <-> y/L*`
- `eta <-> y/delta*`

Do not recompute these conversions locally in chapter code.

## Stability Workflows

- 2D temporal growth / neutral work for Ozgen Fig. 3 should use
  `pymack.ozgen_solver.solve_temporal_ozgen_2d(...)`, which assembles Ozgen's
  temperature equation directly from Eq. 2.15 instead of the Mack enthalpy row
  in `pymack.solver.solve_temporal_compressible(...)`
- paper-agnostic temporal growth / neutral work should use `pymack.analysis`
- 3D oblique work should use the true 3D solver paths in `pymack.analysis` and
  `pymack.solver`
- Ozgen's perturbation equations use Stokes-hypothesis viscous coefficients;
  chapter stability calls therefore set `lambda_mu_ratio = 0.0` rather than
  Mack's default `1.2`
- Ozgen Eq. 19 sets the perturbation thermal wall condition as
  `T_tilde(0) = 0`; the mean flow is adiabatic, but the eigenfunction wall
  condition used by the chapter stability solves is `wall_bc='isothermal'`
- the shared temporal APIs now support phase-speed windowing and full
  Reynolds-wavenumber growth maps, which is what the rebuilt Ozgen oblique
  chapter path uses instead of the old `cos(psi)` effective-parameter proxy
- the shared spatial APIs now include `spatial_growth_map(...)`, which is the
  correct map-level abstraction for Ozgen figures that are plotted in
  `(Re_delta, sigma_i)` rather than `(Re, alpha)`
- the Ozgen chapter helper for oblique-domain height now converts the intended
  `10-12 delta*` far-field extent onto `L*` scale explicitly; the earlier fixed
  `y_max=10/12` values on `L*` scale were under-truncated for moderate and high
  Mach number cases

## Numeric Reference Data

- `reference_data/ozgen/critical_reynolds_points.csv`
- `reference_data/ozgen/most_unstable_wave_angles.csv`
- `reference_data/ozgen/reference_temperature_cases.csv`
- target registry entry file: `reference_data/paper_target_registry.json`

## Current Interpretation

- The mean-flow construction is now shared with the library instead of being trapped inside `chapters/ozgen_kircali_2008/ozgen.py`.
- The Ozgen mean-flow construction now solves the coupled printed momentum and
  energy equations. This replaced the old migrated Crocco closure and then the
  later incorrect incompressible-streamfunction interpretation. For `M=6`, the
  current paper-coordinate value is `delta*/L* = 12.816816`.
- The oblique chapter figures have now been moved off the `cos(psi)` proxy and
  onto true 3D reduced-EVP maps with wave-speed filtering and leakage
  screening.
- The old focused Mach 6 second-mode diagnostic was removed because it was
  generated with the wrong streamfunction interpretation. The corrected
  mean-flow scale puts quick-check neutral crossings inside Ozgen's plotted
  alpha range, but the production diagnostic still needs upper-lobe
  continuation and digitized paper-data acceptance before being called a
  reproduction.
- The Appendix-B freestream decay basis used by the exact-shooting path is now
  built from the exact uniform first-order matrix rather than the earlier
  closed-form branch formulas, because the old `lambda_1` branch was not
  algebraically consistent with the first-order operator.
- Figures 6 and 7 are still provisional because the branch families and
  critical-Re extractions still need direct numeric validation against the
  paper curves.
- Figures 8 and 10 are now explicitly treated as mismapped diagnostics in the
  current chapter script: Fig. 8 needs a 3D spatial `(Re_delta, sigma_i)`
  workflow with constant-`omega_r` contours, and Fig. 10 needs an oblique
  neutral-curve comparison rather than a temporal contour map.
