# pyMack Validation Strategy

*Why pyMack is validated the way it is — and deliberately **not** by replicating
every figure of the source papers.*

## The question

The obvious validation plan for an LST solver is: reproduce every numbered
figure of Mack (1984) and Özgen & Kırcalı (2008), 1:1. An earlier phase of this
project attempted exactly that (50+ digitized curves, per-figure tolerance
gates, a master verification harness). The result, recorded in
[`PAPER_ALIGNMENT_AUDIT.md`](PAPER_ALIGNMENT_AUDIT.md) and
[`FIGURE_GAP_MATRIX.md`](FIGURE_GAP_MATRIX.md): after substantial effort, the
strongest results were the **table-based** comparisons, while most **figure**
comparisons remained partial — and it was rarely attributable whether a gap came
from the solver, from guessing the figure's exact flow conditions, or from
digitization error.

That experience motivates a first-principles redesign.

## Why figure-by-figure replication is the wrong primary strategy

1. **Digitized figures are low-precision data.** Reading curves off 1984 scanned
   plots is good to a few percent at best (line width, scan skew, axis
   interpolation). Comparing a spectral solver that converges to 10 digits
   against pixel-read curves inverts the precision hierarchy. **Tables beat
   figures** — and the project's own two strongest validations (Mack Table 10.1
   growth rates to 0.07–0.91%; Table 11.1 mean-flow thicknesses to <0.5%) are
   both tables.
2. **Every figure carries hidden condition ambiguity.** Mack's figures mix
   Table 11.1 conditions, figure-caption wind-tunnel conditions, and unstated
   defaults. A mismatch proves nothing about the solver; diagnosing it is
   archaeology, not validation.
3. **Figures are massively redundant.** All of Mack's Ch. 10 figures are
   outputs of the *same* operator, eigenvalue solver, and mode tracker. If
   those are verified at a handful of high-precision points, the figures follow;
   the N-th figure adds almost no independent information about correctness.
4. **A wall of partial gates reads worse than a few decisive ones.** A
   validation dashboard with many FAIL/NO_CURRENT entries (mostly digitization
   noise and condition guesses) undermines exactly the trust it is meant to
   build.

Figure overlays remain useful as *qualitative demonstrations* — shapes, mode
topology, trends — so we keep a small number of them, clearly labeled, outside
the pass/fail machinery.

## The strategy: a layered validation pyramid

Each layer is verified independently, at points where the truth is known with
the highest available precision. If every layer holds, the end products
(neutral curves, N-factors) are validated by construction — and when something
breaks, the failing layer localizes it.

| Layer | What it verifies | Benchmark (type) | Status |
|---|---|---|---|
| 0 | Spectral machinery (differentiation, mapping) | Analytic functions (exact) | ✅ machine precision |
| 1 | Compressible mean flow | Mack Table 11.1 thicknesses (**table**) | ✅ < 0.5% |
| 2 | Incompressible eigenvalue problem | Orszag (1971) Poiseuille eigenvalue (**table**) | ✅ 5+ significant figures |
| 3 | Compressible temporal LST, incl. oblique 3D | Mack Table 10.1, 6th & 8th order systems (**table**) | ✅ 0.07–0.91% (exact shooting) |
| 4a | Spatial path, cross-method | Internal: dense QEP vs `solve_spatial` vs Newton vs Gaster-pipeline vs Muller at common points (**internal redundancy**) | ✅ gated (`test_spatial_cross_method_consistency.py`): independent operators agree to \|Δα_r\|≤2×10⁻⁴, \|Δσ\|≤1×10⁻⁴ at the canonical M6 point; within-family ≤10⁻⁶ |
| 4b | Spatial path, external anchor | Malik (1990) Test Case 6 tabulated eigenvalue (**table**) | ✅ gated (`test_malik1990_case6_anchor.py`): α matches Malik's printed value to ~5×10⁻⁶ at N=120 — inside the published literature spread for this case |
| 5 | End-to-end dimensional product (base flow → spatial sweep → neutral branches → physical units) | **Independent-code benchmark:** collaborator Mach 5.35 N₂ flat-plate neutral curve (dimensional) | ✅ gated (upper branch 200–600 kHz, lower branch 330–600 kHz); low-frequency lower branch = documented open investigation |
| 6 | Qualitative literature agreement | Özgen Fig 3 overlay (M=2, M=4) ✅; Mack Fig 10.3 overlay (M=1.3, ψ=45°) ⏳ — **demonstrations, not gates** | ✅ partially complete (see below) |

Design principles:

- **Tables and independent codes are gates; figures are illustrations.**
- **Internal cross-method redundancy is free validation.** Two formulations
  (dense QEP, shooting) agreeing at roundoff catches sign/scaling/BC bugs that
  no single-method benchmark can.
- **One decisive benchmark per layer.** Additional benchmarks of the same layer
  are nice-to-have, not required.
- **The end-to-end layer (5) is the one users actually consume** — it exercises
  the dimensional-units machinery, branch extraction, and the full pipeline in
  the regime pyMack targets (hypersonic second mode).

## Disposition of the figure-replication assets

- The 50+ digitized reference CSVs and the target registry stay in
  `reference_data/` — they are useful data and feed the Layer-6 overlays.
- The chapter-by-chapter reproduction scripts remain in the private workspace;
  individual reproductions can be promoted later *as demonstrations* once they
  match qualitatively. They are **no longer validation requirements.**
- The known Özgen oblique first-mode discrepancy (exact shooting vs reduced
  formulations) is reclassified from "validation blocker" to a **documented
  formulation-difference investigation**: the oblique temporal path is already
  table-validated against Mack Table 10.1 at Layer 3.

## Layer-4a result (June 2026): cross-method consistency — and a pinned systematic

Running all six spatial solution paths at shared Mach 6 points showed the two
independently-implemented operators (dense companion QEP with its own
Lees–Dorodnitsyn base flow vs the main `solve_spatial` family on
`CompressibleBlasiusProfile`) agree to |Δα_r| ≈ 2×10⁻⁴, |Δσ| ≈ 10⁻⁴–10⁻⁵ (L*
units), and routes within the main family (shift-invert QEP, full spectrum,
Newton-on-temporal-EVP, Gaster-seeded pipeline, Muller) agree to ~10⁻⁶.

The study also surfaced a **real systematic, now pinned by the test suite**:
`pymack_dense` hardcodes Stokes second viscosity (λ/μ = 0) while the
`pymack.solver` family defaults to `lambda_mu_ratio = 1.2` (Mack's choice), and
`solve_spatial_muller` has no such parameter (fixed at 1.2). At the canonical
point this shifts σ_L by ~9%. Cross-operator gates therefore compare at
λ/μ = 0, the Muller gate compares against the QEP at 1.2, and one assertion
verifies the 0-vs-1.2 offset *remains finite and visible* so a silent default
change cannot slip through. Choosing/unifying the package-wide default is an
open physics-documentation item.

Also documented honestly: the dense backend's default ny=31 grid is
under-resolved beyond R_L ≈ 2000 (observed Δσ = 2.5×10⁻³ at R_L=2500 vs the
N=80 QEP) — gate points are kept below that, and production use at large R_L
should raise `ny`.

## Layer-4b result (June 2026): Malik (1990) Test Case 6 anchor

The canonical compressible spatial LST benchmark — Malik (1990), J. Comput.
Phys. 86, Table IX, Test Case 6 (M=4.5, R=1500, ω=0.23, insulated wall,
T₀=611.11 K, Sutherland, Pr=0.7): **α = 0.2534048 − 0.0024921 i**. Source
digits were verified against the original archived scan and confirmed
digit-for-digit in two independent citing papers (arXiv:2006.05970 Table 3;
arXiv:1712.08239 Table I).

pyMack (companion QEP, `lambda_mu_ratio=1.2`, isothermal perturbation BC on the
adiabatic mean flow) converges to Malik's printed value:

| N | pyMack α | Δα_r | Δα_i |
|---|---|---|---|
| 100 | 0.2533856 − 0.0025138i | −1.9×10⁻⁵ | −2.2×10⁻⁵ |
| 120 | 0.2533998 − 0.0024898i | −5.0×10⁻⁶ | +2.3×10⁻⁶ |
| 150 | 0.2534010 − 0.0024935i | −3.8×10⁻⁶ | −1.4×10⁻⁶ |

For calibration: Tumin (2007) recomputed this case with a different
formulation and obtained α_i = −0.0027738 (~11% from Malik) — pyMack's
agreement with Malik's printed value is far inside the literature spread.

The formulation mapping itself is a validation by-product: matching Malik
requires `lambda_mu_ratio = 1.2` (the package default; Mack's second-viscosity
convention), **not** Stokes — empirically settling which convention the
package default should keep. The probe matrix (λ/μ ∈ {0, 1.2} × perturbation
wall BC ∈ {isothermal, adiabatic}) showed every other combination misses the
anchor by 10–100×.

## Layer-5 result (June 2026): Mach 5.35 independent-code benchmark

A production pyMack sweep at the benchmark's recorded conditions (M=5.35, N₂,
T_e=64 K, T_w=370 K ≈ adiabatic recovery, Re′=1.176×10⁷/m, 100–600 kHz,
81 frequencies × 177 R-points, single-sweep second-mode window c∈[0.90, 0.97])
against the collaborator's dimensional neutral curve gives:

| Region | n | MAE | max abs | Verdict |
|---|---|---|---|---|
| Upper branch, 200–600 kHz | 66 | 3.2 mm | ~8.6 mm | ✅ gated in CI |
| Lower branch, 330–600 kHz | 44 | 1.3 mm | 3.5 mm | ✅ gated in CI |
| Lower branch, 100–330 kHz | — | up to ~31 mm | — | ⚠ open investigation |

![Mach 5.35 benchmark comparison](figures/mach5p35_collaborator_benchmark.png)

The low-frequency lower-branch difference is structural, not noise: the
benchmark's lower branch stays near R≈520–860 while pyMack's clean
second-mode-window sweep places onset further downstream. Widening the
phase-speed tracking window to c∈[0.80, 1.02] was tested and *rejected* — it
degrades tracking (the continuation latches onto a different branch family)
rather than reconciling the curves. Leading hypothesis: a mode-family /
envelope-definition difference where the first- and second-mode bands interact;
note also the benchmark metadata records two unit-Reynolds values differing by
1.5%. Resolving this (robust S-mode continuation from low phase speed, and/or
clarifying the benchmark's branch bookkeeping with the collaborator) is an open
work item — and exactly the kind of physics question Layer 5 exists to surface.

Artifacts: committed under `validation/data/collaborator_mach5p35/` (pyMack
envelope, run manifest, comparison summary/errors); gates in
`validation/test_collaborator_mach5p35_benchmark.py`.

## Layer-6 result (June 2026): qualitative literature overlays

**Özgen & Kırcalı (2008) Fig 3** — pyMack's 2D temporal stability maps at M=2
and M=4 (1,440 eigensolves at the paper-baseline conditions, which are exactly
`make_ozgen_profile`'s defaults) overlaid on the digitized paper curves:

![Özgen Fig 3 overlay](figures/ozgen_fig3_overlay.png)

- **M=2:** good qualitative agreement — pyMack's c_i = 0.004 contour threads
  the digitized 0.004 markers; the neutral boundary tracks the paper's arch.
- **M=4:** partial topology match — pyMack reproduces the lower-α flank, while
  in the paper's upper-lobe region the genuine discrete mode in pyMack's
  formulation is *marginally damped* (c_r ≈ 0.54–0.62, c_i ≈ −10⁻⁵…−4×10⁻⁴,
  grid-converged). This is the precisely-localized first-mode formulation
  discrepancy, not a plotting artifact: probes confirmed that admitting that
  phase-speed band floods the map with continuous-spectrum junk instead of
  recovering the lobe.
- Selection details (mode-family classification by phase-speed band, with the
  Mack band capped below the free-stream acoustic cluster) are recorded in the
  figure's JSON metadata; the c_i grid is committed alongside for
  reproducibility without recompute.

**Mack (1984) Fig 10.3 (M=1.3, ψ=45°)** — deferred. The overlay script's smoke
path reproduces the Layer-3-validated Table 10.1 points to +0.17…+0.91%, but
the production sweep's bracket/continuation path returned non-converged
shooting artifacts and the script's **built-in table cross-check gate refused
to emit the figure** (exit 1) — the honest-gate design working as intended.
The production-path bug is tracked; the figure will be added once the
generator passes its own gate. (Fig 10.6 was evaluated and explicitly rejected
as an overlay target: a live grid-converged probe showed a ~6× magnitude gap
vs the digitized curve under the repo's condition mapping — exactly the
condition-archaeology this strategy avoids.)

## Execution order

1. **Collaborator Mach 5.35 benchmark (Layer 5)** — matched-condition pyMack
   sweep, quantified branch-location error, tolerance-gated test, comparison
   figure. *(Best credibility-per-effort: validates the whole product path.)*
2. **Internal cross-method consistency gates (Layer 4a)** — a few (M, Re, F)
   points asserted in CI. No external data needed.
3. **Malik (1990) spatial anchor (Layer 4b)** — source the tabulated Mach 4.5
   eigenvalues from the paper, map the nondimensionalization, gate.
4. **Two qualitative overlays (Layer 6)** + this document linked from README.
5. A short `VALIDATION` summary table in the README pointing here.

Each step lands as its own public commit with tests.

---
*Adopted June 2026, replacing the figure-by-figure replication plan. The audit
trail of the earlier approach is preserved in `PAPER_ALIGNMENT_AUDIT.md` /
`FIGURE_GAP_MATRIX.md` and the private development archive.*
