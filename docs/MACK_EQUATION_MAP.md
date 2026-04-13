# Mack Equation Map

This is the short map from Mack's flat-plate theory to the shared code.

## Mean Flow

- Shared builder: `lst.make_mack_profile(...)`
- Condition split: `lst.mack_conditions`
  - `table_11_1` for the thickness table
  - `wind_tunnel` for the Chapter 9-11 figure conditions
- Validation: `validation/test_mack_mean_flow.py`

## Temporal / Spatial Solvers

- Reduced temporal 2D / 3D EVP: `lst.solver`
- Spatial EVP plus refinement: `lst.solver` and `lst.analysis`
- Shared workflow APIs: `lst.analysis`

## Exact First-Order Oblique Path

- First-order system: `lst.mack_shooting.mack_first_order_matrix_3d(...)`
- Exact bounded-shooting continuation: `lst.mack_shooting`
- Branch search / growth scans / neutral extraction: `lst.analysis`

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

## Current Interpretation

- Mean flow and first-order algebra are on firm ground.
- The exact first-order shooting branch is the current authoritative path for low-/mid-Mach oblique first-mode work.
- The reduced finite-domain 3D temporal EVP still misses the physically relevant low-Mach branch in some Chapter 10 cases.
