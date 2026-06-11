# Changelog

All notable changes to pyMack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
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
- `pymack_dense` backend for reliable Mach-6 second-mode spatial growth and N-factor calculations.
- Exact first-order shooting methods for improved low/mid-Mach oblique mode tracking.
- Comprehensive validation suite against Orszag (1971), Mack (1984) Table 10.1 / 11.1, and other benchmarks.
- Digitized reference data from classic papers + numeric comparison machinery.
- `CITATION.cff` and in-code citation helpers (`pymack.cite()`, `pymack.mark_cited()`).
- MIT license and `pyproject.toml` for `pip install pymack`.

### Changed
- Package renamed from `lst` to `pymack`.
- Major README rewrite with honest status of compressible validation.
- Many internal reproduction scripts and partial results moved to private to keep the public repo focused on validated content.
- Improved honesty around limitations in the legacy reduced-EVP paths.

### Fixed
- Various bugs in spatial neutral curve extraction and branch tracking during the dense backend development.

[Unreleased]: https://github.com/BARDAK1995/pyMack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BARDAK1995/pyMack/releases/tag/v0.1.0
