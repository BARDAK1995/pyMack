# pyMack

**pyMack: local linear stability solver for compressible and hypersonic
boundary layers.** A from-scratch Chebyshev spectral-collocation solver for
disturbance **growth rates, eigenvalue spectra, neutral curves, and N-factors**
— focused on **Mack modes**.

[![CI](https://github.com/BARDAK1995/pyMack/actions/workflows/ci.yml/badge.svg)](https://github.com/BARDAK1995/pyMack/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20588214.svg)](https://doi.org/10.5281/zenodo.20588214)

Named after Leslie M. Mack, author of *Boundary-Layer Linear Stability Theory*
(AGARD-R-709, 1984).

> To our knowledge, no open-source Python compressible LST solver existed before
> this — pyMack fills that gap.

---

## What it does

- **Temporal and spatial** stability of compressible boundary layers via
  Chebyshev spectral collocation (full spectrum — the second mode is found
  without an initial guess).
- **Growth rates, neutral curves, and N-factor** ($e^N$) amplification.
- **First (Tollmien–Schlichting) and second (Mack) modes**; 2D and oblique waves.
- Built-in base flows: incompressible Blasius, compressible self-similar
  (adiabatic / isothermal walls), and Özgen flat-plate profiles.
- Pure Python — depends only on `numpy`, `scipy`, `matplotlib`.

## Mach 6 results

At Mach 6 the boundary layer is dominated by the second (Mack) mode. pyMack
resolves its neutral curve and amplification directly:

![Mach 6 second-mode spatial neutral curve](docs/figures/mach6_spatial_neutral_curve.png)

*Spatial neutral curve for the Mach 6 second mode: filled contours of spatial
growth rate $\sigma_L=-\mathrm{Im}(\alpha_L)$ in the
$(R_L=\sqrt{Re_x},\ F=\omega_L/R_L)$ plane, with the lower and upper neutral
branches (purple / orange) enclosing the unstable ($\sigma_L>0$) region.*

![Mach 6 growth rate and N-factor](docs/figures/mach6_growth_nfactor.png)

*Fixed-frequency spatial results for three frequencies: spatial growth rate
$\sigma_i$ (top), N-factor measured from the lower neutral point (middle), and
amplitude ratio (bottom).*

## Background

At hypersonic speeds (Mach ≳ 4), transition is driven not by the viscous
Tollmien–Schlichting (first) mode but by **Mack's second mode** — a
trapped-acoustic instability that resonates between the wall and the relative
sonic line. Linear stability theory gives its growth rate as a function of
frequency and Reynolds number; integrating that growth yields the N-factor used
for transition prediction. pyMack solves the compressible stability eigenvalue
problem with a spectral method, returning the complete spectrum at each station.

## Validation

The **incompressible core** is validated by the test suite — the
**Orszag (1971)** plane-Poiseuille eigenvalue is reproduced to 5+ significant
figures ($c = 0.23753 + 0.00374\,i$ at $Re=10^4$), and the Blasius
Tollmien–Schlichting neutral curve is recovered.

> ⚠️ **The compressible / hypersonic validation against published benchmarks is
> a work in progress**, developed separately and **not yet included here**. The
> Mach 6 results shown above are pyMack's own computations, cross-checked
> internally — not yet claimed as 1:1 reproductions of the figures in
> Mack (1984) or Özgen & Kırcalı (2008).

## Install

```bash
git clone https://github.com/BARDAK1995/pymack
cd pymack
pip install -e .
```

Requires Python ≥ 3.9.

## Testing

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run the validation test suite
pytest validation/ -q
```

The `validation/` directory contains the main benchmark tests (Orr-Sommerfeld, Mack mean flow, Table 10.1 oblique growth, spatial amplification guardrails, etc.).

## Usage

Runnable scripts and examples:

- `scripts/compute_spatial_neutral_curve.py` — spatial neutral curves.
- `scripts/compute_spatial_fixed_frequency_curves.py` — fixed-frequency growth and N-factor.
- `scripts/compute_mach6_growth_nfactor.py` — the Mach 6 second-mode workflow.
- `validation/` — benchmark tests (incompressible core + compressible diagnostics).

## Citing

If pyMack contributes to your work, citing it helps others discover the tool
(and makes the author happy). Citation metadata lives in
[`CITATION.cff`](CITATION.cff) — GitHub also shows a *"Cite this repository"*
button — or just run `pymack.cite()` from Python.

> Mert Senkardesler, *pyMack: local linear stability solver for compressible
> and hypersonic boundary layers* (2026). DOI: 10.5281/zenodo.20588214.
> https://github.com/BARDAK1995/pyMack

The archived release is on Zenodo — DOI
[10.5281/zenodo.20588214](https://doi.org/10.5281/zenodo.20588214). A JOSS paper
is planned.

*(If an AI assistant helped you run pyMack, just double-check that the citation
actually ends up in the final write-up. :))*

Quick silence: set the environment variable `PYMACK_NO_BANNER=1` if the little
reminders ever get in your way.

## References

1. L. M. Mack, *Boundary-Layer Linear Stability Theory*, AGARD Report 709 (1984).
2. M. R. Malik, *Numerical methods for hypersonic boundary layer stability*,
   J. Comput. Phys. **86**, 376–413 (1990).
3. S. A. Orszag, *Accurate solution of the Orr–Sommerfeld stability equation*,
   J. Fluid Mech. **50**, 689–703 (1971).
4. S. Özgen & S. A. Kırcalı, *Linear stability analysis in compressible,
   flat-plate boundary-layers*, Theor. Comput. Fluid Dyn. **22**, 1–20 (2008).
5. P. J. Schmid & D. S. Henningson, *Stability and Transition in Shear Flows*,
   Springer (2001).

## Development & AI usage

pyMack started out as an implementation of old course notes on boundary-layer
stability theory. Its incompressible core is checked against published
benchmarks (e.g. the Orszag 1971 plane-Poiseuille eigenvalue); validation of the
compressible results against the published figures of Mack (1984) and
Özgen & Kırcalı (2008) is ongoing. The code was debugged substantially with the
help of AI tools, then reorganized and refactored into a reusable package. The
author made the core modeling decisions and is responsible for the correctness
of the code and results.

## License

[MIT](LICENSE) © 2026 Mert Senkardesler.
