# LST API Cheat Sheet

This document is the short operational guide for the shared flat-plate LST APIs.

## Base-Flow Builders

- `lst.make_mack_profile(...)`
  - Mack flat-plate builder
  - uses the shared compressible BVP mean-flow solver
  - supports the `table_11_1` and `wind_tunnel` condition splits through `lst.mack_conditions`

- `lst.make_ozgen_profile(...)`
  - Ozgen flat-plate builder
  - uses the shared Sutherland-law shoot-search mean-flow path migrated out of the Ozgen chapter script
  - current shared implementation matches the repo's migrated Ozgen reproduction path

## Scale Conversions

- `lst.delta_star_over_lstar(profile)`
  - returns `delta* / L*`

- `lst.momentum_thickness_over_lstar(profile)`
  - returns `theta / L*`

- `lst.eta_to_lstar(profile, eta)`
- `lst.lstar_to_eta(profile, y_over_lstar)`
- `lst.eta_to_delta_star(profile, eta)`
- `lst.delta_star_to_eta(profile, y_over_delta_star)`

Use these helpers instead of embedding scale factors inside chapter scripts.

## Temporal Workflows

- `lst.temporal_growth_curve(...)`
  - high-level entry point for temporal growth scans
  - incompressible: returns `omega_i(alpha)` from Orr-Sommerfeld
  - compressible reduced mode: returns selected `omega_i(alpha)` and `c(alpha)` from the 3D temporal EVP
  - compressible shooting modes: uses the exact first-order branch tracker when `method='shooting'` or `method='shooting_anchor'`
  - supports physical mode-family filtering through `phase_speed_bounds`,
    `phase_speed_metric`, and `freestream_leakage_tol`

- `lst.temporal_growth_map(...)`
  - shared `(Re, alpha)` temporal map builder for chapter figures and
    validation sweeps
  - returns `omega_i`, `c`, `c_i`, `leakage`, and rowwise neutral estimates

- `lst.trace_temporal_neutral_curve(...)`
  - traces lower and upper temporal neutral branches across `Re`
  - output contains `lower_alpha`, `upper_alpha`, and per-`Re` scan records

- `lst.find_temporal_mode_anchor_3d_shooting(...)`
  - searches exact 3D shooting roots from a seed list and selects one physical
    anchor candidate using optional phase-speed filtering

- `lst.trace_temporal_neutral_curve_shooting(...)`
  - traces temporal lower and upper neutral branches from an exact-shooting
    anchor root and continues in `Re` in both directions from the anchor row
  - intended for first-mode work when the reduced EVP branch selection is not
    trustworthy enough

- `lst.critical_reynolds_curve(...)`
  - derives an approximate critical point from the traced temporal neutral branches
  - output includes `Re_crit` and `alpha_crit`

- `lst.most_unstable_wave_angle(...)`
  - scans `psi` and picks the smallest temporal critical Reynolds number

- `lst.critical_reynolds_from_growth_series(...)`
  - converts a coarse maximum-growth series into an estimated critical Reynolds
    number by locating the first zero crossing

## Spatial Workflows

- `lst.spatial_growth_curve(...)`
  - high-level `sigma(omega)` wrapper around the shared spatial solve
  - returns `sigma`, `alpha_r`, and complex `alpha = alpha_r - i sigma`

- `lst.spatial_growth_map(...)`
  - shared `(Re, omega)` spatial map builder for validation sweeps and future
    paper figures
  - returns `sigma`, `alpha_r`, complex `alpha`, and rowwise neutral estimates

- `lst.trace_spatial_neutral_curve(...)`
  - locates spatial neutral branches by extracting `sigma = 0`

## Exact Oblique First-Mode Tools

These remain the advanced low-level interfaces backing the production wrappers:

- `lst.search_temporal_roots_3d_shooting(...)`
- `lst.temporal_growth_scan_3d_shooting(...)`
- `lst.temporal_growth_scan_3d_shooting_from_anchor(...)`
- `lst.temporal_neutral_points_from_scan(...)`

Use these when a hard low-/mid-Mach oblique branch needs exact first-order continuation rather than the reduced EVP.

## Shared Reference Data

- `lst.load_paper_target_registry()`
- `lst.load_reference_csv(relative_path)`
- `lst.find_paper_target(target_id)`
- `lst.reference_data_root()`

All paper targets and numeric reference tables now live under `reference_data/`.
