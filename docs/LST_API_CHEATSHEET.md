# LST API Cheat Sheet

This document is the short operational guide for the shared flat-plate LST APIs.

## Base-Flow Builders

- `pymack.make_mack_profile(...)`
  - Mack flat-plate builder
  - uses the shared compressible BVP mean-flow solver
  - supports the `table_11_1`, legacy sensitivity `table_10_1`, and
    `wind_tunnel` condition splits through `pymack.mack_conditions`
  - Mack Table 10.1 exact-shooting diagnostics currently use
    `condition='table_11_1'` with isothermal disturbance wall conditions

- `pymack.make_ozgen_profile(...)`
  - Ozgen flat-plate builder
  - solves the coupled Ozgen Eq. 2.32-2.33 mean-flow BVP
  - exposes Eq. 2.36 viscosity, Eq. 2.37 conductivity, Eq. 2.38 `Cp`, and variable `Pr_local`

## Scale Conversions

- `pymack.delta_star_over_lstar(profile)`
  - returns `delta* / L*`

- `pymack.momentum_thickness_over_lstar(profile)`
  - returns `theta / L*`

- `pymack.eta_to_lstar(profile, eta)`
- `pymack.lstar_to_eta(profile, y_over_lstar)`
- `pymack.eta_to_delta_star(profile, eta)`
- `pymack.delta_star_to_eta(profile, y_over_delta_star)`

Use these helpers instead of embedding scale factors inside chapter scripts.

## Temporal Workflows

- `pymack.temporal_growth_curve(...)`
  - high-level entry point for temporal growth scans
  - incompressible: returns `omega_i(alpha)` from Orr-Sommerfeld
  - compressible reduced mode: returns selected `omega_i(alpha)` and `c(alpha)` from the 3D temporal EVP
  - compressible shooting modes: uses the exact first-order branch tracker when `method='shooting'` or `method='shooting_anchor'`
  - supports physical mode-family filtering through `phase_speed_bounds`,
    `phase_speed_metric`, and `freestream_leakage_tol`

- `pymack.temporal_growth_map(...)`
  - shared `(Re, alpha)` temporal map builder for chapter figures and
    validation sweeps
  - returns `omega_i`, `c`, `c_i`, `leakage`, and rowwise neutral estimates

- `pymack.track_complex_branch(...)`
  - tracks one complex eigenvalue branch through a 1D candidate spectrum series
  - use this when a paper mode must remain continuous even if another
    eigenvalue has larger local growth
  - the Ozgen Mach 6 second-mode diagnostic uses it to avoid jumping from the
    high phase-speed Mack branch to lower phase-speed spectral/acoustic families

- `pymack.solve_temporal_ozgen_2d(...)`
  - paper-specific 2D temporal EVP for Ozgen Fig. 3 style work
  - assembles Ozgen's Eq. 2.15 temperature equation directly, rather than the
    Mack enthalpy/pressure energy row used by `solve_temporal_compressible`
  - returns complex phase speeds `c`, eigenvectors, and the collocation grid

- `pymack.trace_temporal_neutral_curve(...)`
  - traces lower and upper temporal neutral branches across `Re`
  - output contains `lower_alpha`, `upper_alpha`, and per-`Re` scan records
  - `refine_neutral=True` refines sign-change brackets with a scalar root solve for reduced/OS scans

- `pymack.find_temporal_mode_anchor_3d_shooting(...)`
  - searches exact 3D shooting roots from a seed list and selects one physical
    anchor candidate using optional phase-speed filtering
  - `include_spanwise_dissipation_coupling=False` switches the exact-shooting
    `8x8` operator to the `a68=0` algebra-check form. Use
    `pymack.search_temporal_roots_6_shooting(...)` for the true primary `6x6`
    Mack sixth-order determinant.

- `pymack.trace_temporal_neutral_curve_shooting(...)`
  - traces temporal lower and upper neutral branches from an exact-shooting
    anchor root and continues in `Re` in both directions from the anchor row
  - intended for first-mode work when the reduced EVP branch selection is not
    trustworthy enough
  - forwards the same sixth/eighth-order coupling switch used by the low-level
    shooting routines

- `pymack.critical_reynolds_curve(...)`
  - derives an approximate critical point from the traced temporal neutral branches
  - output includes `Re_crit` and `alpha_crit`
  - forwards `refine_neutral` and `neutral_xtol` to the temporal neutral tracer

- `pymack.critical_reynolds_by_max_growth(...)`
  - solves `max_alpha growth(Re, alpha) = 0` with scalar optimization over
    wavenumber and Brent refinement in Reynolds number
  - use this for paper critical-Re extraction when a neutral nose is needed,
    rather than taking the first positive value on a coarse grid

- `pymack.maximize_growth_over_parameter(...)`
  - samples a bounded interval, ignores isolated non-finite values, and refines
    the maximum with bounded scalar optimization

- `pymack.most_unstable_wave_angle(...)`
  - scans `psi` and picks the smallest temporal critical Reynolds number

- `pymack.critical_reynolds_from_growth_series(...)`
  - converts a coarse maximum-growth series into an estimated critical Reynolds
    number by locating the first zero crossing

## Spatial Workflows

- `pymack.spatial_growth_curve(...)`
  - high-level `sigma(omega)` wrapper around the shared spatial solve
  - returns `sigma`, `alpha_r`, and complex `alpha = alpha_r - i sigma`

- `pymack.spatial_growth_map(...)`
  - shared `(Re, omega)` spatial map builder for validation sweeps and future
    paper figures
  - returns `sigma`, `alpha_r`, complex `alpha`, and rowwise neutral estimates

- `pymack.trace_spatial_neutral_curve(...)`
  - locates spatial neutral branches by extracting `sigma = 0`

## N-Factor Integration

- `pymack.integrate_n_factor(spatial_growths, x_or_Re=None, ...)`
  - strict N-factor primitive for already-computed spatial growth samples
  - accepts either arrays or dicts containing `sigma`/`growth` plus `x` or `Re`
  - integrates by the trapezoid rule along the supplied path variable
  - defaults to the transition-envelope convention `N = integral max(sigma, 0) dx`
  - set `clip_negative=False` only when a signed amplification integral is intended
  - rejects missing, mismatched, non-finite, or non-monotone path coordinates
  - physical N-factor values require `sigma` units reciprocal to the supplied
    path coordinate; integrating over `Re` is a Reynolds-coordinate diagnostic,
    not a dimensional transition prediction

- `pymack.compute_n_factor(...)`
  - backward-compatible wrapper returning `(path, N, sigma)`
  - delegates to `integrate_n_factor` and therefore has the same validation rules

- `pymack.nfactor(...)`
  - legacy spatial-solve-plus-integration wrapper over `Re_range`
  - useful for diagnostics, but production transition work should call the
    spatial-growth solver first and integrate with the explicit physical path

## Exact Oblique First-Mode Tools

These remain the advanced low-level interfaces backing the production wrappers:

- `pymack.search_temporal_roots_3d_shooting(...)`
- `pymack.search_temporal_roots_6_shooting(...)`
- `pymack.continue_temporal_mode_3d_shooting_sigma_min(...)`
- `pymack.continue_temporal_mode_6_shooting_sigma_min(...)`
- `pymack.temporal_growth_scan_3d_shooting(...)`
- `pymack.temporal_growth_scan_3d_shooting_from_anchor(...)`
- `pymack.temporal_neutral_points_from_scan(...)`
  - accepts optional `refine_func` for bracketed Brent refinement of neutral points

Use the `3d`/`8x8` tools for Mack's full oblique system and the `6` tools for
the primary sixth-order approximation. The `a68=0` `8x8` switch is an algebra
diagnostic, not the production sixth-order determinant.

## Shared Reference Data

- `pymack.load_paper_target_registry()`
- `pymack.load_reference_csv(relative_path)`
- `pymack.find_paper_target(target_id)`
- `pymack.reference_data_root()`
- `pymack.load_mack_table_10_1_cases()`
  - returns typed Mack Table 10.1 oblique growth records with `Ma`, `Re_L`,
    `alpha_L`, `psi_deg`, `beta_L`, `omega_i_6th`, and `omega_i_8th`
- `pymack.select_mack_table_10_1_cases(...)`
  - filters the canonical Table 10.1 CSV by Mach number, Reynolds number, or
    wave angle so diagnostics do not duplicate paper data
- `pymack.evaluate_table_10_1_exact_shooting(...)`
  - evaluates the validated exact Appendix-A/B shooting reproduction for the
    low-/mid-Mach Table 10.1 rows
  - default setup is `condition='table_11_1'` and
    `wall_bc='isothermal'`

All paper targets and numeric reference tables now live under `reference_data/`.

## Dimensional Units / Plots (physical kHz, mm, 1/m)

The solver is fully nondimensional. To turn results into physical units — for
dimensional plots (e.g. frequency in kHz vs streamwise distance in mm, or growth
rate in 1/m) — supply a `DimensionalEdgeState` and use the converters in
`pymack.scales`. The eigenvalue problem itself stays nondimensional; this only
applies the flat-plate similarity scale `L* = nu_e * R_L / U_e` (and
`x = nu_e * R_L**2 / U_e`).

```python
from pymack import (
    DimensionalEdgeState,
    F_to_frequency_khz, frequency_khz_to_F,
    R_L_to_x_mm, x_mm_to_R_L,
    sigma_L_to_per_m, alpha_L_to_per_m, wavelength_L_to_mm,
)

# Physical edge state (illustrative Mach 6 air values)
edge = DimensionalEdgeState(U_e=2080.0, nu_e=1.0e-4, T_e=300.0, M_e=6.0, gas="air")

f_khz = F_to_frequency_khz(F, edge)             # F            -> frequency [kHz]
x_mm  = R_L_to_x_mm(R_L, edge)                  # R_L=sqrt(Re_x) -> x [mm]
g_per_m = sigma_L_to_per_m(sigma_L, R_L, edge)  # spatial growth -> [1/m]
a_per_m = alpha_L_to_per_m(alpha_L, R_L, edge)  # wavenumber     -> [1/m]
# wavelength_L_to_mm(wavelength_L, R_L, edge)   # wavelength     -> [mm]
```

Conventions:

- **Frequency / position** converters take `(value, edge_state)`:
  `F_to_frequency_khz`, `frequency_khz_to_F`, `R_L_to_x_m`, `R_L_to_x_mm`,
  `x_mm_to_R_L`, `lstar_m_from_R_L`.
- **Growth / wavenumber / wavelength** converters take `(value, R_L, edge_state)`
  — because `L*` grows with `R_L`: `sigma_L_to_per_m`, `sigma_L_to_per_mm`,
  `alpha_L_to_per_m`, `alpha_L_to_per_mm`, `wavelength_L_to_mm`.
- Optionally pass `unit_reynolds_per_m` to `DimensionalEdgeState`; then
  `edge.unit_reynolds_consistency_error` reports the mismatch vs the implied
  `U_e / nu_e`.

So a dimensional plot is just: solve nondimensionally, then map the axes with
these converters.
