# Mack Equation Map

This is the short map from Mack's flat-plate theory to the shared code.

## Mean Flow

- Shared builder: `lst.make_mack_profile(...)`
- Condition split: `lst.mack_conditions`
  - `table_11_1` for the thickness table and the current low-/mid-Mach Table
    10.1 exact-shooting reproduction
  - `table_10_1` for the older colder sensitivity schedule
  - `wind_tunnel` for the Chapter 9-11 figure conditions
- Validation: `validation/test_mack_mean_flow.py`

## Temporal / Spatial Solvers

- Reduced temporal 2D / 3D EVP: `lst.solver`
- Spatial EVP plus refinement: `lst.solver` and `lst.analysis`
- Shared workflow APIs: `lst.analysis`

## Exact First-Order Oblique Path

- Full eighth-order system: `lst.mack_shooting.mack_first_order_matrix_3d(...)`
- Primary sixth-order system: `lst.mack_shooting.mack_first_order_matrix_6(...)`
- Exact bounded-shooting continuation: `lst.mack_shooting`
- Branch search / growth scans / neutral extraction: `lst.analysis`
- Mack sixth-order production path: use
  `continue_temporal_mode_6_shooting_sigma_min(...)` or
  `search_temporal_roots_6_shooting(...)`. This solves the true primary `6x6`
  determinant rather than a passive `8x8` surrogate.
- Mack sixth/eighth algebra switch: setting
  `include_spanwise_dissipation_coupling=False` in the `8x8` operator drops
  only Appendix-A `a68`, the single energy-equation coupling term discussed in
  Sec. 10.4. That path remains useful for algebra checks against the primary
  `6x6` block.

## Algebra Checks

- Appendix A / Eq. 8.9 reduction check:
  - `validation/test_appendix_a_reduction.py`
- Appendix B decay-basis check:
  - `validation/test_asymptotic.py`

## Numeric Reference Data

- Table 11.1 thickness targets:
  - `reference_data/mack/table_11_1_delta_star_over_lstar.csv`
- Table 10.1 oblique temporal growth targets:
  - `reference_data/mack/table_10_1_oblique_growth.csv`
  - shared loader: `lst.load_mack_table_10_1_cases()`
  - filtered selectors: `lst.select_mack_table_10_1_cases(...)`
  - reduced-EVP diagnostic: `validation/diagnose_mack_table_10_1.py`
  - exact-shooting diagnostic:
    `validation/diagnose_low_mid_table_10_1_shooting.py --order both`

## Current Interpretation

- Mean flow and first-order algebra are on firm ground.
- The exact first-order shooting branch is the current authoritative path for low-/mid-Mach oblique first-mode work.
- The reduced finite-domain 3D temporal EVP still misses the physically
  relevant low-Mach branch in some Chapter 10 cases. The data-driven
  diagnostics now report relative errors directly against the canonical Table
  10.1 CSV rather than repeating hardcoded table rows.
- Low-/mid-Mach Table 10.1 now matches quantitatively when the exact shooting
  path uses `condition=table_11_1` and `wall_bc=isothermal`. A sweep of all
  `M=1.3`, `1.6`, and `2.2` rows gives sixth/eighth relative errors between
  about `0.07%` and `0.91%` with `--n-steps 300`.
- The earlier `condition=table_10_1` plus adiabatic-wall diagnostic was a
  sensitivity result, not the paper-faithful Table 10.1 setup.
