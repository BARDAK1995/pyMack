# pyMack Roadmap

This document outlines the high-level development direction for pyMack. It is intentionally minimal and will be updated as work progresses.

## Near term (next few months)

- **Mach 6 validation & results**
  - Add validated Mach-6 second-mode spatial neutral curves, growth rates, and N-factor results.
  - Reproduce key results from Mack (1984) and Özgen & Kırcalı (2008) for flat-plate cases.
  - Provide clear comparison tables and data files against published literature.

- **Standalone compressible boundary layer support**
  - Add a reusable tool for generating compressible flat-plate (Blasius-like) boundary layer profiles.
  - Support adiabatic and isothermal wall conditions, variable wall temperature ratios, and realistic gas properties.
  - Designed to be useful both as input to the stability solver and as a standalone component.

- **Code quality & portability**
  - Improve installation experience and documentation.
  - Increase test coverage for core functionality.
  - Refactor for better modularity and easier extension (e.g., additional base flows).

## Medium term

- Axisymmetric flow support (e.g., cones, cylinders).
- Additional base-flow generators and improved file-based profile ingestion.
- Better support for oblique waves and wave-angle sweeps in N-factor calculations.
- Expanded validation against more literature cases.

## Longer term / JOSS milestones

- Feature-complete v1 with documented neutral-curve and N-factor workflows.
- Publication of research results that use the library (planned ~3 months and ~5 months after public repo launch).
- JOSS submission targeted at the 6-month mark from public availability.
- Zenodo-archived releases with clear versioning.

## Contributing

See `CONTRIBUTING.md` for how to get involved. Priority is given to well-tested, documented additions that align with the validation and usability goals above.

---

*Last updated: June 2026*