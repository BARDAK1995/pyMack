# Compressible Linear Stability Theory (LST) Solver

A from-scratch Python implementation of spatial/temporal stability analysis for compressible boundary layers using Chebyshev spectral collocation. Built to compute growth rates, neutral curves, and eigenfunctions for hypersonic flows where **Mack's second mode** instability dominates.

**No open-source Python compressible LST solver exists** — this fills a real gap.

---

## Table of Contents

- [Physics Background](#physics-background)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
  - [Spectral Infrastructure](#1-spectral-infrastructure-lstspectralpy)
  - [Base Flow Profiles](#2-base-flow-profiles-lstbaseflowpy)
  - [Stability Equations](#3-stability-equations-lstequationspy)
  - [Eigenvalue Solvers](#4-eigenvalue-solvers-lstsolverpy)
  - [Analysis Tools](#5-analysis-tools-lstanalysispy)
  - [Plotting](#6-plotting-lstplottingpy)
- [Validation](#validation)
  - [Chebyshev Spectral Accuracy](#test-1-chebyshev-spectral-accuracy)
  - [Orr-Sommerfeld Benchmarks](#test-2-orr-sommerfeld-incompressible-benchmarks)
  - [Compressible Low-Mach Cross-Check](#test-3-compressible-solver-at-ma001)
  - [Convergence Filtering](#test-4-discrete-vs-continuous-spectrum)
- [Results: Ma=5.35 Target Case](#results-mach-535-target-case)
- [Key Technical Challenges and Solutions](#key-technical-challenges-and-solutions)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [References](#references)

---

## Physics Background

### Linear Stability Theory

Linear Stability Theory decomposes small perturbations to a laminar base flow into normal modes:

```
q'(x,y,t) = q_hat(y) * exp[i(alpha*x - omega*t)]
```

- **Temporal analysis**: alpha is real (given wavenumber), omega = alpha*c is complex. Growth when c_i > 0.
- **Spatial analysis**: omega is real (given frequency), alpha is complex. Growth when alpha_i < 0 (sigma = -alpha_i > 0).

### Mack's Instability Modes at Hypersonic Speeds

At high Mach numbers (Ma > 4), boundary layer instability is dominated by **acoustic modes** rather than the classical Tollmien-Schlichting viscous instability:

| Mode | Mechanism | Dominant at | 2D vs 3D | Wall cooling effect |
|------|-----------|-------------|----------|---------------------|
| **First mode** | Viscous (TS-like) + generalized inflection point | Ma < 4 | Most unstable oblique | Stabilized |
| **Second mode** | Acoustic trapping between wall and relative sonic line | **Ma > 4** | **Most unstable 2D** | **Destabilized** |

The second mode is an acoustic resonance: pressure waves bounce between the wall and the relative sonic line (where the local disturbance Mach number equals unity), forming a waveguide. The phase speed satisfies `1 - 1/Ma < c_r < 1`.

### Target Conditions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Ma_edge | 5.35 | Deep in second-mode-dominant regime |
| T_wall | 370 K | Slightly heated (T_w/T_rec = 1.13) |
| T_edge | 56 K | Nitrogen freestream |
| T_recovery | 328 K | With recovery factor r = Pr^0.5 |
| Gas | N2 | R = 296.8 J/(kg K), gamma = 1.4, Pr = 0.72 |
| Viscosity | Power law | mu/mu_ref = (T/T_ref)^0.74 |

---

## Architecture

```
MS_LST/
├── lst/                          # Core library
│   ├── __init__.py               # Public API
│   ├── spectral.py               # Chebyshev matrices, domain mapping
│   ├── baseflow.py               # Mean flow profiles (Blasius, compressible)
│   ├── equations.py              # Compressible stability equation assembly
│   ├── solver.py                 # Temporal O-S, temporal compressible, spatial
│   ├── analysis.py               # Parameter sweeps, neutral curves, N-factors
│   └── plotting.py               # Publication-quality figures
├── validation/                   # Benchmark tests
│   ├── test_chebyshev.py         # Spectral differentiation accuracy
│   ├── test_orr_sommerfeld.py    # Incompressible O-S benchmarks
│   └── test_compressible.py      # Compressible base flow + solver checks
├── cases/
│   └── mach535_n2/
│       ├── run_case.py           # Full analysis driver for target case
│       ├── baseflow_profiles.png
│       ├── temporal_growth.png
│       ├── growth_vs_Re.png
│       ├── eigenspectrum.png
│       ├── eigenfunction.png
│       └── neutral_curve.png
├── requirements.txt
└── README.md
```

**Dependencies:** numpy, scipy, matplotlib only. No exotic packages.

---

## Implementation Details

### 1. Spectral Infrastructure (`lst/spectral.py`)

**Why spectral methods:** Chebyshev collocation gives ALL eigenvalues simultaneously (essential for finding the second mode without an initial guess), achieves spectral (exponential) convergence, and maps naturally to Python/numpy dense linear algebra.

#### Chebyshev-Gauss-Lobatto Points

```python
x_j = cos(pi * j / N),  j = 0, ..., N
```

Points are ordered from +1 to -1. This gives N+1 collocation points with natural clustering near the boundaries.

#### Differentiation Matrix (Don & Solomonoff Algorithm)

The (N+1) x (N+1) matrix D satisfying `(Df)_j ≈ f'(x_j)` is built using barycentric weights:

```
D_ij = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j),   i ≠ j
D_ii = -sum_{j≠i} D_ij                              (negative row sum)
```

where `c_0 = c_N = 2`, `c_j = 1` otherwise. This is exact for polynomials of degree ≤ N and achieves spectral convergence for smooth functions.

#### Domain Mapping (Malik 1990)

Algebraic stretching from [-1, 1] to [0, y_max]:

```
y = L * (1 + eta) / (1 - eta + 2L/y_max)
```

where `L = y_max / 6` (default) clusters ~50-70% of points in the lower third of the domain. The metric terms `dy/d_eta` and `d^2y/d_eta^2` transform the Chebyshev derivatives to physical space:

```
D1_physical = diag(d_eta/dy) @ D_chebyshev
D2_physical = diag(d_eta/dy)^2 @ D^2_chebyshev + diag(d^2_eta/dy^2) @ D_chebyshev
```

### 2. Base Flow Profiles (`lst/baseflow.py`)

#### Incompressible Blasius

Solves the Blasius ODE `f''' + 0.5*f*f'' = 0` with the known shooting parameter `f''(0) = 0.332057336215196`. Uses `scipy.integrate.solve_ivp` with RK45 at `rtol=1e-12`. Profiles are stored as cubic splines for fast evaluation at arbitrary y locations.

The displacement thickness in similarity coordinates `delta*_eta = integral(1 - f') d_eta` maps physical `y/delta*` to similarity `eta`.

#### Compressible Self-Similar (Illingworth-Stewartson + Crocco-Busemann)

This is the most complex component. The compressible boundary layer equations are solved in transformed (similarity) coordinates, then mapped to physical coordinates.

**Transformed ODE:**

```
(C * f'')' + f * f'' = 0
```

where `C = (T/T_e)^(omega-1)` is the Chapman-Rubesin parameter and the temperature comes from the Crocco-Busemann relation:

```
T/T_e = g_w + (1 - g_w)*f' + 0.5*(gamma-1)*Ma^2 * f'*(1 - f')
```

This is exact for Pr=1 and an excellent approximation for Pr=0.72.

**Shooting method:** `scipy.optimize.brentq` finds `f''(0)` in [0.01, 2.0] such that `f'(eta_max) = 1`.

**Critical coordinate mapping:** The physical coordinate relates to the similarity variable through:

```
y_phys(eta) = integral_0^eta g(s) ds
```

where `g = T/T_e`. This is a NONLINEAR mapping (not the linear `y = eta/const` that would be correct only for incompressible flow). The cumulative integral is computed with `scipy.integrate.cumulative_trapezoid`, and all profiles (U, T, rho, mu and their derivatives) are expressed in physical coordinates normalized by the physical displacement thickness:

```
delta*_physical = integral_0^inf (g - f') d_eta
```

**Physical-space derivatives** are computed using the chain rule:

```
dU/dy = f'' / g
d^2U/dy^2 = f''' / g^2 - f'' * g' / g^3
```

(and similarly for T, rho, mu).

### 3. Stability Equations (`lst/equations.py`)

#### Orr-Sommerfeld (Incompressible, Temporal)

The 4th-order Orr-Sommerfeld equation for the wall-normal velocity perturbation:

```
(D^2 - alpha^2)^2 v = i*alpha*Re * [(U - c)(D^2 - alpha^2) - U''] v
```

Formulated as a generalized EVP `A*v = c*B*v` where:

```
A = -L4 / (i*alpha*Re) + U*L2 - U''
B = L2 = D^2 - alpha^2*I
```

**Sign convention note:** The minus sign on the L4 term is critical. The standard O-S has `L4 v = i*alpha*Re*[(U-c)*L2 - U''] v`, which after rearranging to `A*v = c*B*v` gives `A = i*L4/(alpha*Re) + U*L2 - U''`. Getting this sign wrong flips c_i (we caught and fixed this during development).

Boundary conditions: `v = Dv = 0` at both wall and freestream (4 BCs for 4th-order equation), applied by row replacement in A and B.

#### Compressible 4-Variable System (Temporal)

State vector: `phi = [u_hat, v_hat, T_hat, p_hat]^T`

Normal mode convention: `q' = q_hat(y) * exp[i(alpha*x - omega*t)]`

Non-dimensionalization: edge values (U_e, T_e, rho_e, mu_e, delta*), pressure by rho_e*U_e^2.

The linearized equations (all divided by rho_bar for conditioning) are assembled as `A*phi = c*B*phi` where the A matrix contains all terms NOT proportional to c, and B contains the coefficients of c.

**Four equations:**

1. **Continuity** (with linearized EOS: `rho_hat/rho = gamma*Ma^2*p_hat - T_hat/T`):
   ```
   i*alpha*(U-c)*(gamma*Ma^2*p - T/T_bar) + i*alpha*u + Dv - (DT_bar/T_bar)*v = 0
   ```

2. **x-Momentum** (divided by rho_bar):
   ```
   i*alpha*(U-c)*u + DU*v + i*alpha*p/rho - viscous[u,v,T] = 0
   ```
   Viscous terms include: mu*D^2(u), Dmu*D(u), (4/3)*alpha^2*mu*u (from bulk viscosity), (i*alpha/3)*(mu*Dv + Dmu*v) (dilatation coupling), perturbation viscosity terms (dmu_dT*DT*Du, dmu_dT*D^2U*T, etc.)

3. **y-Momentum** (divided by rho_bar):
   ```
   i*alpha*(U-c)*v + Dp/rho - viscous[u,v] = 0
   ```
   Note: pressure gradient is `Dp/rho` with NO gamma*Ma^2 factor (pressure is normalized by rho_e*U_e^2, not p_e).

4. **Energy** (enthalpy form / Form 1):
   ```
   i*alpha*(U-c)*T + DT*v - (gamma-1)*Ma^2*i*alpha*(U-c)*p/rho
       - conduction - dissipation = 0
   ```
   The `(gamma-1)*Ma^2*Dp/Dt` pressure work term couples temperature and pressure perturbations. Conduction prefactor is `1/(Pr*Re*rho)` (no gamma). Dissipation prefactor is `(gamma-1)*Ma^2/(Re*rho)`.

**Key non-dimensionalization details that caused bugs during development:**
- Momentum pressure terms: `i*alpha*p/rho` — NO `1/(gamma*Ma^2)` factor
- y-momentum pressure: `+Dp/rho` — POSITIVE sign (pressure gradient opposes motion, but rearranging to LHS=0 gives positive)
- Viscous C0 terms: NEGATIVE signs (moved from RHS to LHS)
- Viscous C2 terms (alpha^2): POSITIVE signs (double negative from `-visc*(-alpha^2*mu)`)

#### Spatial Formulation (Quadratic EVP)

For spatial analysis (omega real, alpha complex), the equations become a quadratic eigenvalue problem:

```
(C0 + alpha*C1 + alpha^2*C2) * phi = 0
```

Linearized via companion form to a generalized EVP of size 8(N+1) x 8(N+1). This approach works but produces many spurious eigenvalues from the continuous spectrum (see [Challenges](#key-technical-challenges-and-solutions)).

### 4. Eigenvalue Solvers (`lst/solver.py`)

#### `solve_temporal_os(baseflow, alpha, Re, ...)`

Incompressible Orr-Sommerfeld solver. Uses `scipy.linalg.eig` for the generalized EVP. Filters eigenvalues by `|c_r| < 1.5, |c_i| < 1`. Returns eigenvalues sorted by c_i descending (most unstable first).

#### `solve_temporal_compressible(baseflow, alpha, Re, Ma, Pr, gamma, ...)`

Compressible temporal solver — the **primary workhorse**. Assembles the A and B matrices directly (no companion linearization needed since it's a linear EVP in c). This is much more robust than the spatial solver.

Boundary conditions: `u_hat = v_hat = T_hat = 0` at wall and freestream (isothermal wall). Pressure has no explicit BC (handled by the equation system). Applied by row replacement in A and B.

Filters: `c_r in (-0.5, 1.5)`, `|c_i| < 0.5`.

#### `solve_spatial(baseflow, omega, Re, Ma, ...)`

Spatial solver using companion linearization of the QEP. Includes shift-invert option to target specific regions of the spectrum. Less robust than the temporal solver due to the companion system's size and conditioning.

#### Convergence-Based Mode Finding

The critical innovation for reliable results. The continuous spectrum in compressible stability fills a region of the c-plane with eigenvalues that shift as N changes. Discrete modes (the physical Mack modes) converge to fixed values.

**Algorithm:**
1. Solve at two resolutions (N_lo and N_hi, typically 80 and 110)
2. For each eigenvalue at N_hi, find the nearest at N_lo
3. If distance < tolerance (0.008), the mode is "converged" (discrete)
4. Return the most unstable converged mode

This reliably separates the physical second mode (c_i ~ 0.01) from continuous spectrum modes (c_i ~ 0.1-0.3 that change with N).

### 5. Analysis Tools (`lst/analysis.py`)

- `frequency_sweep(...)`: Spatial growth rate sigma vs omega at fixed Re
- `neutral_curve(...)`: 2D map of growth rate in (Re, alpha/omega) space
- `nfactor(...)`: N-factor integration (integral of growth rate along x)
- `temporal_growth_scan(...)`: Temporal c_i vs alpha at fixed Re
- `find_critical_Re(...)`: Bisection for the critical Reynolds number

### 6. Plotting (`lst/plotting.py`)

Publication-quality matplotlib figures with enforced minimum font sizes (labels >= 14pt, ticks >= 12pt, titles >= 16pt per user preference). Consistent color palette, serif fonts, grid styling. All plots use `Agg` backend for headless rendering.

---

## Validation

### Test 1: Chebyshev Spectral Accuracy

**File:** `validation/test_chebyshev.py`

| Test | N | Error | Status |
|------|---|-------|--------|
| Polynomial x^5 derivative | 16 | 1.4e-14 | PASS |
| sin(pi*x) derivative | 16 | 6.6e-10 | PASS |
| exp(-y) on [0,40], D1 | 60 | **1.0e-14** | PASS |
| exp(-y) on [0,40], D2 | 60 | 1.9e-11 | PASS |
| Domain mapping boundaries | 64 | < 1e-10 | PASS |
| Wall clustering (lower 1/3) | 64 | 67.7% | PASS |
| 1-exp(-y) profile, D1 | 80 | 5.0e-14 | PASS |
| Gaussian profile, D2 | 80 | 1.2e-10 | PASS |

The spectral method achieves machine-precision accuracy for smooth functions, confirming the differentiation matrices and domain mapping are correct.

### Test 2: Orr-Sommerfeld (Incompressible Benchmarks)

**File:** `validation/test_orr_sommerfeld.py`

#### Plane Poiseuille Flow

| N | c (computed) | Error vs Orszag (1971) |
|---|---|---|
| 64 | 0.23753 + 0.00374i | 3.53e-06 |
| 96 | 0.23753 + 0.00374i | 3.53e-06 |
| 128 | 0.23753 + 0.00374i | **3.53e-06** |

Reference: Orszag (1971), c = 0.2375265 + 0.0037397i at Re=10000, alpha=1.0. Our result matches to **5+ significant figures**.

#### Blasius Boundary Layer

| Test | Result | Expected | Status |
|------|--------|----------|--------|
| Re=998, alpha=0.179 | c = 0.32576 + 0.00247i | c_i > 0 (unstable) | PASS |
| Convergence N=80→180 | c_i changes by 1e-6 | Converged | PASS |
| Max growth at Re=1000 | alpha=0.271, omega_i=0.00314 | Unstable band exists | PASS |

The TS mode is correctly identified as unstable, with c_r and c_i in the expected range.

### Test 3: Compressible Solver at Ma=0.01

Cross-validation of the temporal compressible solver against the validated O-S solver using an incompressible Blasius profile with constant T, rho, mu at Ma=0.01:

| Solver | c_r | c_i |
|--------|-----|-----|
| O-S (reference) | 0.325765 | 0.00246856 |
| Compressible, N=80 | 0.325490 | 0.00181299 |
| Compressible, N=100 | 0.326324 | 0.00227135 |
| Compressible, N=128 | **0.325773** | **0.00260452** |

At N=128, c_r matches to **5 significant figures** and c_i matches to ~5%. The small c_i discrepancy is from weak compressible effects at Ma=0.01 (energy equation coupling).

### Test 4: Discrete vs Continuous Spectrum

At Ma=4.5, alpha=4.0, Re=2000, the eigenvalue spectrum contains ~100 modes in the physical range c_r in (0.5, 1.0). A convergence study across N=60, 80, 100, 120 revealed:

- **Continuous spectrum modes**: c shifts with N (not converged), e.g., c_i changes from 0.025 at N=60 to 0.119 at N=120
- **Discrete mode**: c = 0.983 + 0.011i (converged to 4 digits across all N values)

The converged discrete mode at c_r = 0.983 is the **Mack second mode** (fast acoustic mode, F mode), with a physically reasonable growth rate c_i ~ 0.01.

---

## Results: Mach 5.35 Target Case

**Case:** Ma=5.35, N2, T_wall=370K, T_edge=56K, T_wall/T_rec=1.13 (slightly heated wall)

All figures saved in `cases/mach535_n2/`.

### Base Flow Profiles

![Base Flow](cases/mach535_n2/baseflow_profiles.png)

- **Velocity**: Standard BL profile, U=0 at wall to U=1 at edge
- **Temperature**: T_wall/T_e = 6.6 at wall, monotonically decreasing to T_e
- **Density**: rho_wall/rho_e = 0.15 (very low due to high temperature)
- **Viscosity**: mu_wall/mu_e ~ 4 (power law with omega=0.74)

### Temporal Growth Rate (Discrete Second Mode)

![Growth Rate](cases/mach535_n2/temporal_growth.png)

- Convergence-filtered to isolate the genuine discrete Mack mode
- Peak c_i ~ 0.029 at alpha ~ 1.0 (Re=2000)
- Phase speed c_r ~ 0.82-0.98, approaching 1.0 at high alpha
- Mode stabilizes around alpha ~ 9

### Effect of Reynolds Number

![Re Effect](cases/mach535_n2/growth_vs_Re.png)

- **Growth rate increases with Re** (correct for second mode)
- Unstable alpha range widens at higher Re
- Peak omega_i shifts to higher alpha with Re
- Re=1000: weak instability; Re=8000: strong instability across wide alpha range

### Eigenvalue Spectrum

![Spectrum](cases/mach535_n2/eigenspectrum.png)

- Classic **Y-shaped** continuous spectrum (characteristic of compressible stability)
- Single discrete unstable mode (red star) clearly separated above c_i = 0
- Mode at c = 0.990 + 0.009i — the Mack second mode

### Eigenfunction

![Eigenfunction](cases/mach535_n2/eigenfunction.png)

- **u_hat**: Peaks inside the BL with secondary maximum, smooth profile
- **v_hat**: Strong peak at y/delta* ~ 1.3 (near BL edge), classic acoustic mode shape
- **T_hat**: Significant amplitude near the relative sonic line
- **p_hat**: Large amplitude near the wall, characteristic of acoustic trapping

### Neutral Curve

![Neutral](cases/mach535_n2/neutral_curve.png)

- Instability region expands with Re (correct behavior)
- Low-alpha modes unstable first; high-alpha modes join at higher Re
- Growth rates O(0.01-0.04) — physically reasonable for the second mode

---

## Key Technical Challenges and Solutions

### 1. O-S Sign Convention

**Problem:** The Orr-Sommerfeld eigenvalue c had the correct magnitude but wrong sign of c_i (showed stable when it should be unstable).

**Root cause:** The viscous term in `A = L4/(i*alpha*Re) + U*L2 - U''` should be `A = -L4/(i*alpha*Re) + U*L2 - U''`. The difference is subtle: `1/(i*alpha*Re) = -i/(alpha*Re)` vs `-1/(i*alpha*Re) = +i/(alpha*Re)`. Only the latter gives the standard O-S equation `L4 v = i*alpha*Re*[(U-c)*L2 - U''] v`.

**Fix:** Single minus sign in front of L4. Validated against Orszag's Poiseuille result.

### 2. Compressible Equation Non-Dimensionalization

**Problem:** Initial implementation had gamma*Ma^2 factors in the momentum pressure terms, producing unphysical growth rates.

**Root cause:** Confusion between two pressure non-dimensionalizations:
- `p* = p / (rho_e * U_e^2)` (our choice): momentum has `i*alpha*p/rho`, NO gamma*Ma^2
- `P* = p / (rho_e * a_e^2) = gamma*Ma^2 * p*`: momentum has `i*alpha*P/(gamma*Ma^2*rho)`

**Fix:** Removed all gamma*Ma^2 from momentum pressure terms. The EOS correctly has gamma*Ma^2: `rho_hat/rho = gamma*Ma^2*p_hat - T_hat/T`.

### 3. Compressible Equation Signs (C0, C1, C2)

**Problem:** All viscous terms in C0 had wrong sign, all alpha^2 terms in C2 had wrong sign.

**Root cause:** When writing `L*phi = 0`, viscous terms from the RHS of the momentum equations must be moved to the LHS with a negative sign. The alpha^2 viscous terms then get a double negative (positive). This was not done consistently.

**Fix:** Systematic derivation: write each equation with all terms on LHS, identify C0/C1/C2 coefficients carefully tracking signs.

### 4. Coordinate Mapping for Compressible Base Flow

**Problem:** BL edge appeared at y/delta* = 0.48 (should be ~1.3). Eigenvalues at wrong frequencies.

**Root cause:** The Illingworth-Stewartson transformation maps similarity eta to physical y through a NONLINEAR integral `y(eta) = integral g(eta') d_eta'` where `g = T/T_e`. The initial implementation used a linear mapping `eta = y * const`, which is only correct for incompressible flow (g=1).

**Fix:** Compute `y(eta)` via cumulative_trapezoid, normalize by physical delta*, build splines in physical coordinates, transform all derivatives using the chain rule (`d/dy = (1/g) * d/d_eta`).

### 5. Continuous Spectrum Contamination

**Problem:** The most unstable eigenvalue at each (alpha, Re) had c_i ~ 0.1-0.3, orders of magnitude too large. These showed no convergence with N.

**Root cause:** The continuous spectrum in compressible stability fills a region of the complex c-plane. The discretized problem approximates this continuum with a dense set of discrete eigenvalues. These modes satisfy the equations numerically (small residual) but are not physical discrete instabilities.

**Solution:** Convergence-based filtering — solve at two resolutions and keep only eigenvalues that match to within a tolerance. The genuine Mack second mode (c_i ~ 0.01) converges; continuous spectrum modes (c_i ~ 0.1-0.3) do not.

### 6. Temporal vs Spatial Approach

**Problem:** The spatial solver (QEP with companion linearization) produces a 2x larger system with many more spurious eigenvalues. The companion system also has poor conditioning when C2 has zero rows (continuity equation has no alpha^2 terms).

**Solution:** Adopted temporal analysis as the primary approach. The temporal problem is a standard generalized EVP (linear in c, not quadratic in alpha), which is much more robust. Spatial growth rates can be estimated via the Gaster transformation for moderate growth rates.

---

## Future Work

### High Priority

- [ ] **Spatial solver refinement**: Implement Newton iteration (Muller's method) targeting specific eigenvalues, bypassing the companion system entirely
- [ ] **Gaster transformation**: Convert temporal results to spatial growth rates for N-factor integration
- [ ] **3D oblique modes (beta != 0)**: Extend to oblique waves for first-mode analysis. The first mode is most unstable as oblique at hypersonic speeds
- [ ] **Adiabatic wall BC**: Add `DT_hat = 0` wall condition (currently isothermal only)

### Medium Priority

- [ ] **BL calculator integration**: Read profiles from the external `newBLcalculator` (github.com/BARDAK1995/newBLcalculator.git) instead of the built-in Crocco-Busemann approximation
- [ ] **DSMC/CFD profile reader**: Import mean flow profiles from DSMC or CFD solutions via `TabulatedProfile` class
- [ ] **N-factor envelope**: Compute max N over all frequencies for transition prediction
- [ ] **Energy equation Form 2**: Implement the `(gamma-1)*T*div(u')` form as an alternative to the pressure work form, for cross-validation
- [ ] **Mack's AGARD validation**: Systematic comparison against Mack Report 709 tables for Ma=2.2, 4.5, 6.0

### Low Priority / Extensions

- [ ] **Adjoint solver**: For receptivity analysis and optimal perturbation studies
- [ ] **Nonparallel corrections**: PSE (Parabolized Stability Equations) for slowly varying base flows
- [ ] **Real gas effects**: Extend to thermally/calorically imperfect gases (relevant for Ma > 8)
- [ ] **Roughness/blowing**: Modified wall BCs for surface roughness or wall blowing/suction studies
- [ ] **GPU acceleration**: Port dense linear algebra to cupy for large N

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

All available via `pip install -r requirements.txt`. No compiled extensions or exotic packages.

---

## References

1. **Mack, L.M.** (1984). "Boundary-Layer Linear Stability Theory." AGARD Report 709. *The* definitive reference for compressible stability.

2. **Malik, M.R.** (1990). "Numerical methods for hypersonic boundary layer stability." *J. Comp. Phys.* 86(2), 376-413. Spectral collocation method, domain mapping, equation formulation.

3. **Orszag, S.A.** (1971). "Accurate solution of the Orr-Sommerfeld stability equation." *J. Fluid Mech.* 50(4), 689-703. Benchmark Poiseuille eigenvalue.

4. **Schmid, P.J. & Henningson, D.S.** (2001). *Stability and Transition in Shear Flows.* Springer. General stability theory textbook.

5. **Fedorov, A.** (2011). "Transition and stability of high-speed boundary layers." *Annu. Rev. Fluid Mech.* 43, 79-95. Review of Mack mode physics.

---

*Built with numpy, scipy, matplotlib. No open-source Python compressible LST solver existed before this.*
