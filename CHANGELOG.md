# Changelog

All notable changes to pyMack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
