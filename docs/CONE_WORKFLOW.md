# Sharp-Cone (Mangler) Workflow

pyMack supports sharp cones at zero incidence through the **Mangler
transformation only**. No solver, operator, or profile changes are involved:
the local-parallel LST problem at cone surface station `s` is *exactly* the
flat-plate problem at the equivalent Reynolds number `R_eq`. Everything
cone-specific lives in station bookkeeping (`pymack/cone.py`) and one
N-factor multiplier.

## Honest scope statement

- **Mangler only.** Transverse-curvature terms in the stability operator are
  **omitted**. This is valid only while the boundary layer is thin relative to
  the local body radius, `delta << r_local = s*sin(theta_c)`. Check this at
  your *lowest* station: near the tip it always fails, and for slender cones
  at low unit Reynolds number transverse curvature materially shifts
  second-mode growth.
- **Sharp tip, zero incidence only.** Nose bluntness (entropy layer) or angle
  of attack breaks the plate/cone similarity entirely; neither is supported.
- **The edge state is your input.** You must supply the **post-shock cone
  edge** conditions (`U_e`, `nu_e`, `T_e`, `M_e`), e.g. from a Taylor–Maccoll
  solution — *not* the freestream. Nothing in the code can validate this.
- **The half-angle is metadata.** It cancels exactly in the Mangler mapping
  (see below), so `--cone-half-angle-deg` never changes results. The
  half-angle enters the physics only through the edge state you supply.

## The factor-3 mapping (derivation sketch)

Mangler: `x_eq = (1/L^2) * int r_w^2 ds'` with `r_w(s) = s*sin(theta_c)` for a
sharp cone gives `x_eq = s^3 sin^2(theta_c) / (3 L^2)`. Choosing the local
radius as reference length (`L = r_w`) cancels `sin(theta_c)` exactly:

```
x_eq = s / 3                          (half-angle independent)
Re_x_eq = Re_s / 3                    =>  R_eq = sqrt(Re_s/3) = R_s/sqrt(3)
L*_eq = nu_e * R_eq / U_e             (same formula as the plate)
s     = 3 * nu_e * R_eq^2 / U_e       =>  ds = 6 * L*_eq * dR_eq
N(s)  = int sigma_phys ds = int 6*sigma_L dR_eq = 3 * N_plate  (same R_eq window)
```

Classic consequences (White, *Viscous Fluid Flow*; Schlichting):
`delta_cone(s) = delta_plate(s)/sqrt(3)`, and at a fixed physical station the
second-mode frequency is `sqrt(3) ~ 1.732` times higher on the cone.
`F = 2*pi*f*nu_e/U_e^2` is geometry-independent, so all kHz<->F conversions
are unchanged.

## Runner usage

```bash
# Surface-distance window s = 10..120 mm along the cone ray; edge state is the
# post-shock cone edge (here the APS preset values):
python scripts/run_mach6_spatial_neutral_case.py \
    --preset aps-paper-baseline \
    --geometry cone --cone-half-angle-deg 7 \
    --quality smoke --dry-run
```

In cone mode:

- `--x-min-mm/--x-max-mm` mean **surface distance s** along the cone ray; the
  runner maps them to the solver grid via `R_eq = sqrt(Re_s/3)`. The
  eigensolver chain is geometry-blind — its R grid *is* the `R_eq` grid.
- The postprocess step automatically receives
  `--r-convention custom --dx-over-lstar-per-dr 6.0`
  (= 3 x the plate's 2.0 under the repo's `R = sqrt(Re_x)` convention).
- In dimensional CSVs the `x_mm` column means **s** (so every plot helper
  keyed on `x_mm` works unchanged) and an extra `x_eq_mm = s/3` column records
  the equivalent plate station. The manifest and metadata carry a `geometry`
  block stating all of this (`mangler_factor`, `R_L_meaning`,
  `transverse_curvature_terms: omitted`, ...).
- Case slugs gain a `CONE{angle}deg_` prefix so cone runs never clobber plate
  output directories.
- Nondimensional cone runs work too: `--r-min/--r-max` are interpreted as
  `R_eq`, and the N multiplier is still 6.
- Default behavior without `--geometry` is the flat plate, bit-identical to
  before.

**Comparison trap:** in cone outputs `R_L` means `R_eq`. Comparing cone and
plate runs "at the same R_L" silently compares different physical stations
(factor 3 in distance).

## Library usage

```python
from pymack.scales import DimensionalEdgeState
from pymack.cone import (
    cone_s_mm_to_R_eq, cone_R_eq_to_s_mm,        # station maps (math changes)
    cone_n_factor, cone_n_factor_multiplier,     # N = int 6*sigma_L dR_eq
    R_eq_from_R_s, R_s_from_R_eq,                # sqrt(3) Reynolds maps
)

edge = DimensionalEdgeState(U_e=858.0, nu_e=7.313e-5, T_e=52.0, M_e=5.85,
                            gamma=1.4, gas="nitrogen")  # POST-SHOCK cone edge!
R_eq = cone_s_mm_to_R_eq(91.83, edge)   # solve the plate problem at this R
```

Reuse contract: `lstar_m_from_R_L`, `sigma_L_to_per_m/_mm`,
`alpha_L_to_per_m/_mm`, `wavelength_L_to_mm`, `frequency_khz_to_F`,
`F_to_frequency_khz` from `pymack.scales` are correct for the cone
**unchanged** when called with `R = R_eq` (thin `cone_*` wrappers exist in
`pymack.cone` for readability; they delegate verbatim — never add cone factors
to them).

Standalone postprocessing of an existing cone growth CSV:

```bash
python scripts/postprocess_spatial_amplification.py growth.csv \
    --output-dir out --r-convention custom --dx-over-lstar-per-dr 6.0
```

If you work in the `R = sqrt(2 Re_x)` convention instead (plate multiplier
1.0), the cone multiplier is `cone_n_factor_multiplier(1.0) = 3.0` — it is
always 3 x the plate multiplier, never a bare 6.

## Validation

`validation/test_cone_mangler.py` (fast, no eigenvalue solves) pins:
exact `sqrt(3)` station-map consistency against `pymack.scales`, the
`L*`/frequency relations at fixed physical station, the analytic factor-3
N-factor (constant and linear `sigma_L`, plus a physical arc-length
cross-check), the runner's cone dry-run contract, and that the default plate
path emits no cone artifacts.
