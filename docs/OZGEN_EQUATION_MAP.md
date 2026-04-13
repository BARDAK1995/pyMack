# Ozgen Equation Map

This is the short map from Ozgen & Kircali (2008) to the shared code.

## Mean Flow

Paper equations:

- momentum: Eq. 2.32
- energy: Eq. 2.33
- property laws: Eqs. 2.36, 2.37, 2.38

Shared builder:

- `lst.make_ozgen_profile(...)`
- backing class: `lst.baseflow.OzgenFlatPlateProfile`

The current shared implementation migrates the Sutherland-law flat-plate builder
out of the old Ozgen chapter script and exposes the same profile information
through the common `lst` interface.

Important limitation:

- the current shared path preserves the legacy chapter behavior for thermal
  transport by using a constant-Pr conductivity closure, effectively
  `kappa ~ mu / Pr`
- that means Eq. 2.36 viscosity is on the shared path, but Eq. 2.37 / Eq. 2.38
  are not yet implemented as a stricter paper-specific production model

## Scales

Use the shared helpers in `lst.scales` for:

- `delta* / L*`
- `theta / L*`
- `eta <-> y/L*`
- `eta <-> y/delta*`

Do not recompute these conversions locally in chapter code.

## Stability Workflows

- 2D temporal growth / neutral work should use `lst.analysis`
- 3D oblique work should use the true 3D solver paths in `lst.analysis` and
  `lst.solver`
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
- The oblique chapter figures have now been moved off the `cos(psi)` proxy and
  onto true 3D reduced-EVP maps with wave-speed filtering and leakage
  screening.
- The shared flat-plate stability workflows now enforce the adiabatic thermal
  wall condition consistently; earlier mixed adiabatic/isothermal diagnostic
  runs should not be treated as current evidence.
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
