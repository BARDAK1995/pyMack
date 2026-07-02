# pyMack architecture

This document explains how the library is put together and why — the map to
keep in your head when reading, using, or extending the code.

## Design tenets

1. **Correctness first, and visibly so.** The numerical kernels are verified
   line-by-line against the published formulations (Özgen & Kırcalı 2008;
   Mack 1984) and against each other; the derivations live in
   `docs/*.tex` and the agreement audit in `verification/SUCCESS_MATRIX.md`.
   Anything that changes a kernel must keep `validation/` green.
2. **Functional core, thin facade.** The engines are plain functions over
   NumPy arrays — easy to test, compose, and reason about. A small facade
   (`pymack.api`) provides the one-obvious-way entry points and rich result
   objects; it *wraps* the kernels, never hides them.
3. **Importing pyMack does nothing.** No banner, no global matplotlib state,
   no wrapped functions. A scientific library must be inert until called.
4. **Results carry their provenance.** A `ModeResult` records every parameter
   that produced it, so any number in a paper can be traced and reproduced.
5. **Independent cross-checks are a feature, not duplication.** Deliberately
   redundant formulations (spectral vs. shooting; temperature-form vs.
   enthalpy-form energy equation) let one method verify another. Where code
   was *accidentally* duplicated, it has been unified.

## Layers

```
                        ┌──────────────────────────────┐
   user-facing          │  pymack.api  (facade)        │  flat_plate,
                        │  ModeResult                  │  temporal_mode,
                        └──────────────┬───────────────┘  spatial_mode
                                       │
                        ┌──────────────▼───────────────┐  sweeps, neutral
   workflows            │  pymack.analysis             │  curves, growth maps,
                        │                              │  N-factors
                        └──────────────┬───────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              │                        │                         │
   eigenvalue │ temporal_solver        │ solver                  │ mack_shooting
   engines    │  (2-D temporal EVP,    │  (spatial QEP drivers,  │ temporal_shooting
              │   Özgen temperature    │   temporal 3-D EVP,     │  (exact first-order
              │   arrangement)         │   O-S, local root       │   marching; Mack
              │                        │   finders)              │   Appendix A)
              │ dense (config-driven dense workflow)             │
              └────────────────────────┬─────────────────────────┘
                                       │
                        ┌──────────────▼───────────────┐  Chebyshev matrices,
   operators            │  spectral      equations     │  domain mapping,
                        │                              │  QEP block assembly
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐  self-similar profiles,
   base flows           │  baseflow    boundary_layer  │  transport laws,
                        │                              │  standalone generator
                        └──────────────────────────────┘

   cross-cutting:  scales (nondimensionalization, unit conversions,
                   sample_baseflow — the single base-flow resampling helper)
                   cone (Mangler mapping)          plotting (style on demand)
   reference:      reference_data, mack_conditions, mack_table_10_1, asymptotic
```

Higher layers may import lower ones, never the reverse. `scales` is the one
cross-cutting module everything may use.

## The two problems and the four ways to solve them

Everything reduces to: *given a base flow, which small waves grow?*

| Route | Module | Discretization | Returns | Use it for |
|---|---|---|---|---|
| Temporal spectral | `temporal_solver` | Chebyshev collocation, QZ | all eigenvalues | surveys, neutral curves |
| Spatial spectral | `solver` (+`equations`) | companion QEP, shift-invert | modes near target | growth curves, N-factors |
| Exact shooting | `mack_shooting`, `temporal_shooting` | RK4 march + wall matrix | one refined mode | anchors, cross-checks |
| Dense workflow | `dense` | config-driven spectral | branch objects | scripted production runs |

The spectral and shooting routes are *independent discretizations of the same
physics*; their agreement on an eigenvalue (see
`validation/test_temporal_shooting_cross_check.py`) is a verification
statement, which is why both are kept first-class.

Two *energy-equation arrangements* coexist by design: `temporal_solver`
follows Özgen's temperature form (Stokes closure), `solver`/`equations` follow
Mack's enthalpy form (λ/μ = 1.2 closure). They bracket the closure choice and
agree on the physical mode to discretization accuracy.

## Mode acceptance: what "the" eigenvalue means

A collocated operator returns 4n eigenvalues; most belong to the discretized
continuous spectrum. A **discrete boundary-layer mode** must:

1. **decay toward the freestream** (edge-to-peak amplitude below ~0.1), and
2. **stay put when the domain grows** (a box-height change moves continuum
   artifacts, not physical modes).

The facade applies test 1 always and test 2 automatically whenever no user
guess anchors the branch (`check_stationarity='auto'`). Lower-level callers
are expected to apply their own acceptance (see `verification/` for the
full machinery).

## Conventions

* **Node ordering**: index 0 = freestream, index n−1 = wall (Chebyshev
  ξ = +1 → y_max). Getting this backwards produces plausible-looking wrong
  answers; boundary-condition rows depend on it.
* **Length scales**: `length_scale ∈ {'delta_star', 'L_star'}` everywhere;
  `R = √Re_x` is the L*-scaled Reynolds number. Conversions live in
  `scales`, never inline.
* **Parameter vocabulary** (same names, same meaning, everywhere):
  `alpha, beta, omega, c, Re, Ma, Pr, gamma, N, y_max, L, wall_bc,
  length_scale, lambda_mu_ratio`.
* **Wall BCs**: `wall_bc ∈ {'isothermal', 'adiabatic'}` refers to the
  *disturbance* temperature condition; the mean-flow wall state is a
  property of the profile.
* **Growth-rate signs**: temporal `omega_i = alpha·Im(c) > 0` unstable;
  spatial `sigma = −Im(alpha) > 0` unstable.

## Public API policy

* The curated namespace is `pymack.__all__` (grouped by layer in
  `__init__.py`); submodules are also public API for power users.
* Private names (leading `_`) may change without notice; nothing outside a
  module may import them (the test suite imports only public names).
* Deprecations go through `pymack.__getattr__` with a `DeprecationWarning`
  and a pointer to the replacement; pre-1.0 they are removed after one
  minor release. Current table: `pymack._DEPRECATED`.
* Version is single-sourced in `pymack/__init__.py` (`hatch` reads it).

## Extending pyMack

* **New base flow** (e.g. a CFD-imported profile): implement the profile
  protocol — a callable returning the base-flow dict (`U,dU,d2U,T,dT,d2T,
  rho,mu,dmu`, plus transport derivatives) on a `y/δ*` grid, with a
  `delta_star_over_lstar`-compatible scale — and every solver accepts it.
* **New geometry**: follow `cone.py` — a station mapping onto the flat-plate
  solution plus unit converters; keep the solvers geometry-agnostic.
* **New solver/closure**: add it as a sibling engine; wire a cross-check
  test against an existing route before exposing it in the facade.
* **Tests**: fast physics checks go in `validation/` (each file < ~30 s);
  long benchmarks get `@pytest.mark.slow`; literature reproductions live in
  `verification/` with a `verdict.json`.
