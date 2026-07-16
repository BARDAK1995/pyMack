# Changelog

All notable changes to pyMack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 0.2.0 (PROPOSED; not tagged)

#### Added

- CPU sweep productization: Windows-safe process pools, opt-in BLAS thread
  pinning, an eigenvalues-only 2-D route, and committed floor/point-budget
  measurements with identity checks.
- The complete 37-case validation record, including the provenance census,
  Mach 4.5 and Mach 5.8 reference repairs, amendments, and archived historical
  records needed to audit those corrections.
- A committed 2,880-node Ozgen capability demonstration (4x the original
  sampling), its full-QZ and eigenvalues-only artifacts, a deterministic
  spot-check, and figure provenance.
- JOSS paper sources, CPU reproducibility instructions, refreshed citation
  metadata, and committed benchmark evidence for every performance number in
  the manuscript.

#### Changed

- The public package scope is CPU-only and matches the JOSS service
  contribution: NumPy/SciPy solvers, CPU batch sweeps, validation evidence,
  user documentation, and the paper.
- An explicit `backend='gpu'` request now raises a clean
  `NotImplementedError` explaining that the public build is CPU-only.
- Base-flow profiles now serialize their sampled numerical state instead of
  SciPy's runtime spline objects, preserving Windows process-pool sweeps with
  current SciPy releases; the development extra includes `threadpoolctl` for
  the BLAS-pinning verification gate.

#### Release decision

- Version 0.2.0 is proposed only. No version field, release date, archive DOI,
  or tag is applied until the owner performs the release step.

## [0.1.0] - 2026-07-01

First design release: the library grew a curated public API, a high-level
facade, and packaging polish, on top of the verified numerical kernels.

### Added
- **High-level facade** (`pymack.api`, re-exported at top level):
  `flat_plate(...)`, `temporal_mode(...)`, `spatial_mode(...)` returning a
  frozen **`ModeResult`** (eigenvalue, growth rate, phase speed,
  eigenfunctions, grid, and the full parameter provenance). Mode selection
  applies the two discrete-mode acceptance tests from the docs — freestream
  decay always, and domain-height stationarity automatically whenever no
  user guess anchors the branch (`check_stationarity='auto'`).
- **`examples/`** — three runnable introductions: first Mack mode,
  fixed-frequency growth curve + N-factor (seed-then-continue pattern),
  dimensional-unit conversions.
- **`docs/ARCHITECTURE.md`** — the layer map, design tenets, conventions,
  and extension guide.
- `pymack.scales.sample_baseflow` — the single canonical base-flow
  resampling helper (replaces four private copies across the solvers).
- `pymack.plotting.apply_plot_style()` — the house style, applied on demand.
- `pymack.analysis.n_factor_curve` — renamed from `nfactor`.
- Root `conftest.py` so a plain checkout is importable by pytest without
  per-file `sys.path` hacks; `[tool.pytest.ini_options]` with a `slow`
  marker; ruff configuration; dynamic version via hatch
  (single-sourced in `pymack/__init__.py`).
- Facade test suite `validation/test_api_facade.py`, including a regression
  for domain-artifact rejection in unguided spatial selection.

### Changed
- `pymack/__init__.py` is now a curated, layer-grouped namespace with
  `__all__` (~80 names instead of an uncurated ~130) and **no import-time
  side effects**.
- Previously private helpers that external code needed are now public API:
  `solver.assemble_temporal_compressible_3d_evp`, `solver.apply_wall_bc_3d`,
  `solver.apply_dirichlet_freestream_bc_3d`,
  `solver.temperature_wall_operator`, `mack_shooting.sample_scaled_baseflow`,
  `mack_shooting.wall_condition_rows_3d/_6` (old underscore names remain as
  aliases).
- `pymack.plotting` no longer mutates global matplotlib state (or forces the
  Agg backend) at import.
- All in-repo callers migrated off deprecated names; `validation/` tests
  import pymack like any user would.
- Version bumped to 0.1.0.

### Deprecated
- `pymack.pymack_dense` → `pymack.dense` (module renamed; shim warns).
- `pymack.make_ozgen_profile` → `make_flatplate_profile`;
  `OzgenFlatPlateProfile` → `FlatPlateProfile`;
  `solve_temporal_ozgen_2d` → `solve_temporal_2d`;
  `nfactor` → `n_factor_curve`. All forwarded lazily with
  `DeprecationWarning` via `pymack.__getattr__`.

### Removed
- The import-time citation banner and the `globals()`-level citation-nudge
  wrappers (they changed function identity); `mark_cited` /
  `CITED_IN_PAPER`. `pymack.cite()` remains the way to get the reference.
- (The `ozgen_solver` / `ozgen_shooting` module paths now emit a
  `DeprecationWarning` and forward to `temporal_solver` /
  `temporal_shooting`; they will be removed after one minor release.)

### Fixed
- **Özgen 2-D shooting matrix** (`pymack.temporal_shooting.
  ozgen_first_order_matrix_2d`) was a faulty transcription of Özgen & Kırcalı
  (2008) Eqs. 2.21–2.28: several coefficients dropped 1/μ, T′, or ⅓ factors
  and mixed X₁/X₂ couplings, so the wall matrix never went singular
  (σ_min ≈ 0.116, flat, at the true eigenvalue). It now delegates to the
  validated Appendix-A operator (`mack_first_order_matrix_6` at β=0,
  λ/μ=0 Stokes), which is algebraically identical to Özgen's printed system —
  with one deliberate correction to the printed Eq. (2.24) X₁ term
  (sign and μ-placement, recovered by re-derivation). After the fix,
  σ_min = 4×10⁻⁶ at the spectral solver's worked-example eigenvalue
  c = 0.9301 + 0.0200i (M6, R=5500, α_L=0.174) and rises two orders of
  magnitude within |Δc| ~ 2×10⁻³. New regression test:
  `validation/test_temporal_shooting_cross_check.py`. The module was not used
  by any verification case (those use `mack_shooting`), so no published
  result changes; docs/numerical_methods.tex Part B updated to match.
- **Momentum thickness θ/L\*** in `CompressibleBlasiusProfile` integrated the
  wrong quantity (`U(T−U)` instead of `U(1−U)`; in Levy–Lees variables the T
  factors cancel in θ's integrand). The bug vanished in the incompressible
  limit (θ → Blasius 0.6641, which is why it survived) but inflated θ at high
  Mach (~7× at Ma=4.5 adiabatic). δ\*/L\* was always correct. Found by
  adversarial review of the new generator; corrected value verified against
  the Blasius limit and Malik (1990) Case-6 δ\* (9.3946 vs printed 9.3992).

### Added
- **Standalone compressible boundary-layer generator**
  (`pymack.generate_boundary_layer`, `scripts/generate_boundary_layer.py`):
  adiabatic/isothermal walls (`Tw_over_Te`/`Tw_over_Taw`/`T_wall_K`),
  Sutherland/power-law/Mack transport, gas presets, automatic adiabatic-seeded
  continuation for difficult cold walls (now the single source of the recipe —
  `make_mack_profile` delegates to it), CSV + SI export, and
  `as_stability_profile()` for direct solver input. 12 new validation tests.
- **Sharp-cone (Mangler) support** (`pymack.cone`, runner `--geometry cone`):
  station mapping `R_eq = sqrt(Re_s/3)` (half-angle cancels exactly), cone
  N-factor path integral `N = ∫ 6 σ_L dR_eq = 3 × N_plate`, `cone_*` twins of
  the dimensional converters, manifest geometry block, and
  `docs/CONE_WORKFLOW.md` with the honest Mangler-only scope statement
  (transverse curvature deferred; edge state must be the post-shock cone
  edge). The √3 derivation was independently re-verified in adversarial
  review; flat-plate runner behavior is bit-identical without the flag.
- **Layer-4b validation gate — Malik (1990) anchor**
  (`validation/test_malik1990_case6_anchor.py`): pyMack reproduces the
  canonical tabulated compressible spatial eigenvalue (Malik JCP 86, Table IX,
  Test Case 6: M=4.5, R=1500, ω=0.23, insulated wall) to ~5×10⁻⁶ at N=120 —
  inside the published literature spread. Source digits verified against the
  original archived paper and two independent citing papers. The match also
  establishes that Malik's formulation corresponds to `lambda_mu_ratio=1.2`
  (the package default), settling the second-viscosity convention question.
- **Layer-4a validation gates — cross-method spatial consistency**
  (`validation/test_spatial_cross_method_consistency.py`): six CI gates
  comparing the two independent spatial operators and all solution routes
  within the main family, with evidence-based tolerances; pins the
  Stokes-vs-1.2 λ/μ systematic and documents dense-grid limits.
- **Validation strategy** (`docs/VALIDATION_STRATEGY.md`): layered validation
  pyramid replacing figure-by-figure paper replication — tables and independent
  codes as gates, figures as qualitative demonstrations.
- **Layer-5 independent-code validation gate**: pyMack production sweep at the
  collaborator Mach 5.35 N₂ benchmark conditions, with committed artifacts
  (`validation/data/collaborator_mach5p35/`) and CI-gated tolerances
  (`validation/test_collaborator_mach5p35_benchmark.py`). Upper neutral branch
  agrees to MAE 3.2 mm (200–600 kHz); lower branch to MAE 1.3 mm (330–600 kHz).
  The low-frequency lower-branch difference is documented as an open
  mode-family investigation; the wide-phase-window alternative was tested and
  rejected.
- Added `scripts/run_mach6_spatial_neutral_case.py`, the canonical one-command
  Mach 6 second-mode spatial workflow for growth curves, neutral branches,
  N/amplification, and a manifest with explicit no-stitch/no-smoothing policy.
- Added validation coverage for the Mach 6 runner dry-run contract so the
  production command path cannot silently drift.
- Added `docs/MACH6_SPATIAL_NEUTRAL_WORKFLOW.md` to document the canonical
  Mach 6 case assumptions, expected endpoints, and artifact policy.

### Changed
- Made citation reminders significantly lighter and less intrusive:
  - Import banner is now two short friendly lines (instead of a large boxed message).
  - Nudges are now once-per-session only (no more repeated 30s reminders during long runs).
  - Only high-level analysis functions that produce publishable results trigger nudges (`nfactor`, `integrate_n_factor`, `neutral_curve`, etc.).
  - Removed heavy "AI assistant" language from runtime output.
  - `mark_cited()` now gives a friendly thank-you message.
- Updated README Citing section to be warmer and mention the silence option.
- Updated Mach 6 workflow docs to prefer the canonical single-sweep runner over
  manual ad hoc command sequences.

## [0.1.0] - 2026-06-06

### Added
- Initial public release of pyMack.
- Core Chebyshev spectral collocation solver for compressible boundary layer stability (temporal and spatial).
- Support for Mack and Özgen mean-flow models.
- `pymack_dense
