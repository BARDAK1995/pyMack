# Changelog

All notable changes to pyMack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Made citation reminders significantly lighter and less intrusive:
  - Import banner is now two short friendly lines (instead of a large boxed message).
  - Nudges are now once-per-session only (no more repeated 30s reminders during long runs).
  - Only high-level analysis functions that produce publishable results trigger nudges (`nfactor`, `integrate_n_factor`, `neutral_curve`, etc.).
  - Removed heavy "AI assistant" language from runtime output.
  - `mark_cited()` now gives a friendly thank-you message.
- Updated README Citing section to be warmer and mention the silence option.

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
