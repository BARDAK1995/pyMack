# pyMack-GPU — Adversarial (Red-Team) Review

*Produced by the design fan-out (2026-07-02); all numerical claims were measured on the real pyMack matrices (read-only). Companion to PLAN.md.*

## Summary

Red-team review of the pyMack GPU design, grounded in code reading plus live numerical experiments on the actual solver matrices. The three core ideas largely SURVIVE, with two important corrections and one honest-marketing downgrade. (1) Affinity: the 2D spatial QEP and both 2D temporal formulations are EXACTLY affine (probe residuals 1e-16) with constant BC rows, and every production sweep uses a per-Mach-fixed profile/grid â€” but the 3D wave-aligned operator is NOT polynomial in (alpha,beta) (measured probe residual 6.1e-4, because of k=sqrt(alpha^2+beta^2) and alpha/k, beta/k terms at solver.py:632,679-684); it IS exactly affine in an 8-term (k,cos psi,sin psi) basis (9e-14), so the Vandermonde probe must use those variables or fig10.4 silently forfeits the GPU path. (2) Precision: raw kappa(A) is 6e8-2e10, so bare complex64 LU is dead, but max row/column equilibration reduces it to 1.2e5-1.1e7, and an end-to-end mixed-precision two-sided RQI (c64 equilibrated LU + FP64 Rayleigh quotient) I implemented against the real matrices reproduces FP64 QZ eigenvalues to 1.3e-11 (N=128), 1.4e-11 (the feared M10 dim-1005 case, 4 iterations, 0.29 s vs 7.3 s QZ) and 3.4e-10 at the marginal N=256 delta*-scale case; a bordered-Newton spatial solve hits 1.2e-12 in 6-7 iterations with a >=2% seed basin. (3) Mode tracking: the real risk is not continuation per se but candidate-SET semantics â€” Ozgen's discrete-mode extractor requires two-domain matching over enumerated decaying candidates, and the M10 first mode literally collapses into the continuous spectrum mid-grid (documented at compute_mack_fig10_4.py:112-116) â€” so the tracker must be a small mode BUNDLE with device-side decay/band filters and explicit mode-loss verdicts, with QZ fallback fraction >3-5% halving the speedup. The biggest paper-level threat is baseline fairness: the RQI reformulation alone gives 10-30x on CPU, so the GPU-vs-strongest-CPU factor is ~5-15x, and honest claims are 100-1000x versus the deployed pipeline decomposed into algorithm x hardware. Realistic targets: Ozgen 1600s -> 15-30s, M10 station 950s -> 10-20s, eN compute sub-second on GPU but seconds end-to-end. One ground-truth error found: B is SINGULAR after BC row replacement (8 zeroed rows), harmless for (A-cB) LU but it forbids any B^{-1}A formulation.

## Key decisions

- Probe the 3D wave-aligned operator in the (k, cos psi, sin psi) x {1, 1/Re} basis (8-16 terms), never in (alpha, beta) monomials - verified exact to 9e-14 vs 6e-4 failure; keep the held-out self-verification as the shipping gate
- Make max row+column equilibration a mandatory, fused part of every complex64 factorization (reduces measured kappa from 6e8-2e10 to 1.2e5-1.1e7); cap c64 families at N<=256 delta*-scale and provide an FP64 zgetrfBatched polish path for marginal cases
- Adopt two-sided RQI with c64 equilibrated LU + FP64 Rayleigh quotient as the temporal engine and bordered/implicit-determinant Newton (size 4n+1, analytic T') as the spatial engine - both verified to 1e-11/1e-12 against FP64 QZ on the real matrices, including the M10 dim-1005 case in 4 iterations
- Track a BUNDLE of 3-8 modes per grid point with device-side c_r-band, freestream-decay, and two-domain-stationarity filters, and emit explicit 'no discrete mode' verdicts - single-mode wavefront continuation cannot reproduce the validated Ozgen/fig10.4 semantics
- Treat the 8 asymptotic freestream BC rows (the only non-affine, parameter-dependent rows) as a per-point device row-patch after tensor-contraction assembly, or keep that refinement stage on CPU
- Report speedups decomposed as algorithm x hardware x precision against a strong baseline (same RQI algorithm on 12-16 CPU cores), with verdict identity on the 37-case matrix and fallback fraction as first-class metrics
- Run a 1-day CuPy-on-Windows spike (getrfBatched throughput at n=645-1285, batch 64-256, c64/z128; sysmem-fallback disabled) before committing to the performance model
- Never rely on B being invertible: B has 8 zeroed BC rows (singular by construction) - all formulations must go through (A - cB) or bordered T(alpha) factorizations

## Risks (ranked)

- Baseline-fairness rejection (highest paper risk): the algorithmic reformulation alone gives 10-30x on CPU, so GPU-vs-strong-CPU is only ~5-15x; mitigate by claiming the decomposed matrix (algorithm 10-30x, hardware 5-15x, combined 100-1000x vs deployed) and by demonstrating a new capability (4x-denser Ozgen map in less time)
- Silent forfeiture of the flagship 3D workload if the affinity probe uses (alpha,beta) monomials - the auto-fallback would quietly route fig10.4 to CPU assembly; mitigate with the (k,psi) basis and a CI check that every production family passes the 1e-14 probe residual
- Mode-tracking failures concentrated at physically critical locations (synchronization points, neutral curves, M10 band collapse) where RQI basins shrink with the eigenvalue gap; if CPU-QZ fallback exceeds ~3-5% of points the Ozgen speedup halves (>20% kills it); mitigate with bundle tracking, consensus re-seeding, async CPU/GPU overlap, and fallback-fraction reporting
- FP32 conditioning frontier: delta*-scaled N=256 has kappa_equil~1.1e7 (refinement rate ~0.7, 12 iterations); any future case at N>=300 or stiffer wall conditions breaks c64 - keep the FP64 polish path and a per-family kappa estimate in the certification gate
- Performance-model uncertainty (+-3x): cuBLAS getrfBatched is documented for small n and unproven at n~1000 on a WDDM Windows consumer card; Windows driver sysmem-fallback can silently corrupt benchmarks by spilling VRAM to host RAM - spike first, pin driver settings
- Semantic drift vs the validated pipeline: the GPU bundle-tracker replaces full-spectrum argmax-over-filtered-candidates; edge cases (None cells, two-domain match tolerance 4e-3, fs threshold 0.06) must reproduce CPU verdicts exactly on all 37 cases or the certification gate fails - budget iteration time for filter-threshold parity
- Realistic wall-time targets are softer than hoped: Ozgen 1600s -> 15-30s (not 'seconds'), full validation -> minutes (baseflow builds, seeds, fallbacks, I/O dominate once EVP time collapses) - set expectations accordingly in the paper's abstract

---

# Adversarial Review: pyMack GPU Design (Red Team Report)

All code claims verified against `C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST`. All numerical claims below were **measured on the actual pyMack matrices** in this session (read-only Python runs, no files written). Local GPU confirmed: **RTX 4070 Ti, 12 GB, Ada (FP64 = 1/64 rate)**; CuPy not yet installed.

---

## 1. AFFINITY CENSUS â€” idea (1) survives with one mandatory correction

### 1a. Operator entries: measured polynomial structure

**2D spatial QEP** (`pymack/equations.py:90-264`, BCs at `pymack/solver.py:150-171`): EXACTLY affine.
- C0 âˆˆ span{1, Ï‰, 1/Re} (held-out probe residual **2.5e-16**; the Ï‰/Re cross-term coefficient is numerically zero, â€–K_{Ï‰/Re}â€–/â€–K_1â€– = 2e-11)
- C1 âˆˆ span{1, 1/Re} (residual **1.0e-16**), C2 âˆˆ span{1, 1/Re} (residual **3.2e-16**)
- Ï‰ enters only via `iw = 1j*omega` linearly (equations.py:165); Re only via `visc/cond/diss âˆ 1/Re` (equations.py:168-170). **6 constant K_m matrices per family. No sqrt/abs/conditionals anywhere in the assembly.**
- **BC rows are parameter-independent** (identity rows + a constant D1 row for adiabatic; `temperature_wall_operator` solver.py:103-113, applied solver.py:150-171). They simply land in the constant K_1 term â€” probing the post-BC matrices works directly (verified).

**2D temporal, both formulations** (`solver.py:414-608` Mack-style; `temporal_solver.py:37-207` Ozgen-style used by the Ozgen grids): affine in {1, Î±, Î±Â²}Ã—{1, 1/Re}; B is affine in Î± (measured **1.9e-16**). Note there are TWO distinct 2D temporal formulations in production â€” the GPU engine must support both K-extractions (probing handles this for free).

**3D oblique wave-aligned** (`solver.py:611-783`): **NOT polynomial in (Î±, Î²)**. Measured: 6-term (Î±,Î²)-monomial Vandermonde fit leaves residual **6.1e-4** â€” the operator contains `k = hypot(alpha,beta)` (line 632), `ik`, `kÂ²`, and the direction cosines `alpha_over_k`, `beta_over_k` (lines 679-684). However it **IS exactly affine in (k, cosÏˆ, sinÏˆ)**: the 8-term basis {1, cosÏˆ, sinÏˆ, k, kÂ·cosÏˆ, kÂ·sinÏˆ, kÂ², kÂ²Â·cosÏˆ} fits to **9.1e-14**, and along a fixed-Ïˆ ray {1, k, kÂ²} fits to **7.2e-16**. With 1/Re included, â‰¤ ~16 K_m. B is affine in Î± alone (1.9e-16).
- **Consequence:** the design's "probe in monomials of (a, w, 1/R), fall back if non-affine" would *silently* dump the entire fig10.4 M10 workload (the flagship 8 s/solve case) onto the CPU-assembly fallback. **Mandatory fix: the probe basis for the 3D solver must be (k, cosÏˆ, sinÏˆ, 1/Re).** The self-verification residual check is what saves you from shipping this bug â€” keep it.

**Parameter-dependent BC rows exist in exactly one place:** `_apply_asymptotic_freestream_bc_3d` (solver.py:812-869), used only by `refine_temporal_compressible_3d_asymptotic` (solver.py:1381) inside `continue_temporal_mode_3d` when `use_asymptotic_refinement=True`. Those 8 rows depend on (Î±, Î², c_ref, Re) through the sqrt-laden Mack Appendix-B decay basis â€” **not affine**. Mitigation: it is only 8 rows out of 5n â€” after the batched tensor-contraction assembly, overwrite 8 rows per matrix on device (rank-8 per-point patch, CPU-computed rows, trivial transfer), or keep that refinement stage on CPU. The default production paths (fig10.1, fig10.4) use the Dirichlet freestream BC (solver.py:962, compute_mack_fig10_4.py:171), which is constant.

### 1b. Per-workload grid/profile fixity (does one K-set cover the sweep?)

| Workload | Profile | N, y_max | Affine granularity |
|---|---|---|---|
| Mack fig10.4 (Î±,Ïˆ) optimization | 1 self-similar/Mach (`make_mack_profile`, table_11_1) | **fixed per Mach**, `Y_MAX_BY_MACH`/`N_BY_MACH` at compute_mack_fig10_4.py:131-132 â€” y_max does NOT vary with R | 1 family per Mach covers the whole (R, Î±, Ïˆ) sweep â€” 119Ã—9 solves at M10 |
| Mack fig10.1 neutral loops | 1/Mach | fixed per Mach | 1 family per Mach |
| Ozgen c_i grids (720 nodes) | 1/Mach (`make_flatplate_profile`) | N=200 fixed; **TWO y_max per node** (short/tall domain-stationarity pair, build_firstmode_grid.py:46-51, discrete_mode.py:83-91) | **2 families per Mach** â€” the two-domain filter is intrinsic to the physics verdict; everything doubles |
| Mach-6 eN pipeline | 1 profile | **TWO N values per node** (convergence check at N and Nâˆ’Î”, compute_mach6_growth_nfactor.py:86-112); spatial neutral case N=31 (run_mach6_spatial_neutral_case.py:1092-1093) | 2 families; tiny matrices (4n=128) |
| Ma&Zhong spatial traces | 1 profile | fixed N, Y_MAX (trace_mazhong_curves.py:19-31) | 1 family, 704-node (R, Ï‰) grid |
| Cone SF2015 | **Mangler-reduced single self-similar profile** â€” the feared "per-station profiles" do not exist here; stations differ only in R_eq (compute_cone_sf2015.py:59-76, N=90, Y_MAX=40 fixed) | fixed | 1 family across all 28 stations Ã— frequencies |

`sample_baseflow` (scales.py:236-267) is R-independent for self-similar profiles; `length_scale='L_star'` only rescales derivatives by the per-profile constant Î´*/L*. `wall_bc` and `lambda_mu_ratio` are constant *within* each sweep (0.0 for Mack figs, 1.2 for Ma&Zhong/cone) â€” each defines a family, never a per-point switch.

**Verdict: affinity holds at "one K-set per (profile, N, y_max, L, wall_bc, Î»-ratio, formulation) family" granularity, and every current production sweep decomposes into 1-2 families per Mach.** The genuine future breaker is non-self-similar station-wise mean flows (PSE/DNS baseflows) â€” then affinity holds only per station with batching across Ï‰, which the design should treat as the standard granularity from day one.

---

## 2. PRECISION ATTACK â€” idea (3) survives, measured, with one marginal frontier

Measured 2-norm condition numbers of the assembled, BC-applied matrices:

| Case | dim | Îº(A) raw | Îº after max-row+col equilibration | Îº_eqÂ·u32 |
|---|---|---|---|---|
| M4.5 2D, N=128, Î´* scale | 645 | 5.9e8 | **5.4e5** | 0.03 |
| M4.5 2D, N=256, Î´* scale | 1285 | 1.9e10 | **1.1e7** | 0.66 â† marginal |
| M10 3D, N=200, y_max=150 L* (the fig10.4 flagship) | 1005 | 2.0e10 | **1.2e5** | 0.007 â† best! |

- Bare complex64 LU is dead (ÎºÂ·u32 â‰¥ 35 raw). **Diagonal equilibration is not optional â€” it is the load-bearing component of the mixed-precision claim** and costs O(nÂ²) per matrix, fusable into assembly.
- **End-to-end experiment (the decisive one):** two-sided RQI with per-iteration equilibrated **complex64 LU** + **FP64 Rayleigh quotient** from FP64 master matrices, seeded with a 1e-3 to 1e-2 perturbed eigenvalue (simulating a wavefront neighbor seed), versus FP64 QZ reference:
  - M4.5 N=128: relative eigenvalue error **1.3e-11**, 4-6 iterations, 0.17-0.25 s CPU (QZ: 1.7 s)
  - M4.5 N=256: **3.4e-10**, 12 iterations (slow refinement, Îº_eqÂ·u32â‰ˆ0.66 â€” the FP32 failure frontier is Î´*-scaled Nâ‰³256; Nâ‰³300 will break)
  - **M10 3D N=200 (dim 1005, Î´*/L*=37): 1.4e-11, 4 iterations, 0.29 s vs 7.3 s QZ.** The feared case is actually the *best*-conditioned after equilibration.
- Spatial engine: **bordered/implicit-determinant Newton** on T(Î±)Ï†=0 directly (size 4n+1, analytic Tâ€²=C1+2Î±C2, no 8n companion): error **1.2e-12**, 6-7 iterations, seed basin â‰¥ 2% relative, 0.2 s vs 1.2 s shift-invert at N=128.
- Certification target of 1e-5 relative is beaten by 5+ orders of margin. **Recommendation:** cap c64 families at Nâ‰¤256 (Î´* scale); add an optional final FP64 `zgetrfBatched` polish (1/64 rate: batch-256 n=1285 â‰ˆ 1.5e12 FLOP â‰ˆ 2-3 s on the 4070 Ti â€” acceptable as an exception path, and free on A100 later).

---

## 3. MODE-TRACKING ATTACK â€” the real risk is candidate-SET semantics, not continuation

- **Documented in-repo evidence that local iteration lands on wrong branches:** `solve_spatial_full_spectrum` exists precisely "when a branch tracker must not miss the Mack/S root because the local shift landed on a nearby acoustic or spurious branch" (solver.py:313-319). Any pure-continuation engine re-imports this failure mode.
- **Mode death mid-grid:** at M10, "by alpha~0.040 the discrete band-mode has collapsed... only the decaying continuous spectrum remains" (compute_mack_fig10_4.py:112-116). RQI *will* converge to a continuous-spectrum discretization mode there and report plausible garbage. Required: device-side mode-identity tests per converged point â€” c_r band + the freestream-decay `fs` amplitude test (top-10%-of-domain amplitude ratio, discrete_mode.py:44-63 â€” needs only the RQI eigenvector, trivially batched) + the two-domain stationarity match. The GPU sweep must be able to output "**no discrete mode here**" exactly as the CPU pipeline does (None cells in fig10.4's `maximize_growth`, unmatched candidates in `discrete_mode`).
- **Ozgen semantics:** the c_i map value is "most-unstable *discrete* mode among decaying candidates matched across two domain heights" â€” an argmax over a SET whose winner switches family (first mode vs Mack modes; cr_band 0.05-0.97 spans both). A single-mode wavefront tracks the wrong family after crossings. **Fix: track a bundle of ~3-8 modes per grid point** (batched RQI is trivially parallel over the bundle), seeded from full QZ at a sparse spine (e.g., one Re column per Mach), then apply the same filters + argmax. This preserves the validated verdict semantics; the certification gate (37-case `verification/build_success_matrix.py` + per-case `verdict.json`, confirmed present) is the arbiter.
- **fig10.4 optimization loops: the wavefront idea DOES apply** â€” affine assembly evaluates arbitrary batched (k, Ïˆ) tuples, so the coarse grid sweeps as a frontier and the fine-refine points (9 per station) batch across all R stations, seeded from each station's coarse winner.
- **Synchronization/branch-cut points:** measured basins â‰¥1e-2 at benign points, but basins shrink like the local eigenvalue gap near mode-S/mode-F synchronization and near the neutral-curve continuous-spectrum approach. Expect fallback density concentrated exactly along the physically interesting boundaries.
- **Amdahl bound:** QZ fallback costs 1.7 s (dim 645) / 3-8 s (dim 804-1005) / 17 s (dim 1285) per point. For the Ozgen grid, a 5% fallback rate â‰ˆ 36 pts Ã— 2 domains Ã— ~3.3 s â‰ˆ 4 CPU-min serial â‰ˆ 20-30 s on 12 cores â€” comparable to the entire GPU phase (~5-15 s). **>3-5% fallback halves the speedup; >20% kills it.** Mitigations, in order: (i) consensus re-seed from 2-3 converged neighbors before declaring failure; (ii) run CPU-QZ fallbacks asynchronously (ProcessPool) overlapped with the advancing GPU frontier; (iii) report the fallback fraction per case as a first-class paper metric.

---

## 4. LIBRARY-REALITY ATTACK (uncertainty flagged where it exists)

- **Hardware (verified):** RTX 4070 Ti, 12 GB, Ada â†’ FP32 â‰ˆ 40 TFLOPS, FP64 â‰ˆ 0.6 TFLOPS (1/64). 12 GB comfortably fits all workloads: largest working set â‰ˆ batch-128 of 1005Â² c64 (â‰ˆ1.0 GB) + FP64 masters (â‰¤16 K_m Ã— 8-16 MB â‰ˆ 130-260 MB) + LU workspace.
- **cuBLAS batched LU:** `cgetrfBatched`/`zgetrfBatched`/`getrsBatched` exist for all four types; CuPy exposes low-level bindings (`cupy.cuda.cublas.*`) and `cupy.linalg.solve` accepts stacked (..., n, n) arrays. **Caveat:** NVIDIA documents getrfBatched as optimized for small n (â‰²100); at n = 645-1285 it works but at reduced efficiency â€” plan for effective 1-4 TFLOPS FP32, and A/B-test against looped cuSOLVER `getrf` inside CUDA graphs. Exact public-API surface varies across CuPy 12/13 â€” needs a 1-day spike, flagged as uncertainty.
- **No GPU nonsymmetric eig anywhere** (`cupy.linalg.eig` does not exist; cuSOLVER lacks ggev) â€” the design premise holds; QZ stays CPU.
- **RawKernel complex support:** yes, via `<cupy/complex.cuh>` (`cupy::complex<float/double>`).
- **Windows pitfalls:** (i) WDDM launch latency (5-20 Âµs, KMD batching) punishes many-small-kernel designs â€” the batched-LU design is compute-dominated (0.1-0.5 s per frontier step at batch 256), so fine, but per-point looping would be latency-bound; enable Hardware-Accelerated GPU Scheduling; use CUDA graphs for the RQI inner loop. (ii) **GeForce driver â‰¥536 "CUDA sysmem fallback" can silently spill VRAM over-allocation to host RAM (10-50Ã— slowdown) instead of raising OutOfMemoryError** â€” set "Prefer No Sysmem Fallback" in NVCP for all benchmarks or the paper's timings are corrupted. (iii) Display shares the 12 GB.
- **Corrected performance model:** per RQI iteration per point â‰ˆ one batched c64 LU (8/3Â·nÂ³ real FLOP: 1.4e9 at n=804; 2.7e9 at n=1005) + 2 batched triangular solves + FP64 GEMV residuals (memory-bound, negligible). Ozgen: 720Ã—2 domainsÃ—~5 iters â‰ˆ 1e13 FLOP â‰ˆ **3-6 s GPU** + sparse QZ seeds (CPU, ~10 s parallel) + fallbacks â‡’ **15-30 s wall** (not "seconds"). fig10.4 M10: 1071 solves â‡’ **7-15 s + seeds** (vs 7,800 s deployed). eN N=31: GPU compute < 1 s; pipeline overhead dominates. Full 37-case matrix: **minutes, not tens of seconds** (baseflow solves ~2 s each, seeds, fallbacks, I/O).

---

## 5. BASELINE-FAIRNESS ATTACK â€” the biggest threat to the paper

Measured on this machine: the mixed-precision RQI/bordered-Newton is **10-30Ã— faster than QZ per point on the CPU alone** (0.17-0.43 s vs 1.7-7.3 s). A referee will run idea (2) on a 16-core workstation and get within ~5-15Ã— of the GPU. Honest decomposition (all measured or measurable):
- Algorithmic (mode-following + affine assembly vs full-spectrum QZ): **10-30Ã—**, hardware-independent.
- Hardware (GPU batched c64 LU vs 12-16-core CPU running the *same* algorithm): **~5-15Ã—** (CPU batched c64 LU â‰ˆ 100-300 GFLOPS effective vs 1-4 TFLOPS GPU), growing with n.
- Combined vs the deployed pipeline: **100-1000Ã—** â€” this is the claimable headline *only if decomposed*.

**Honest benchmark matrix for the paper:** algorithms {full-QZ (deployed), shift-invert companion, RQI/bordered continuation} Ã— hardware {1 core, 12-16 cores MKL, RTX 4070 Ti c64+FP64-refine, A100 FP64 (later)} Ã— workloads {Ozgen 720-node, fig10.4 M10 station, eN 3649-solve, full 37-case matrix}, each reporting wall time, eigenvalue agreement vs FP64 QZ, **verdict agreement (37/37 required)**, and **fallback fraction**. The novelty framing must be "a full-spectrum-free, mixed-precision, batched formulation of Malik-1990-class LST *that makes consumer GPUs usable*" â€” not raw Ã—-factors.

---

## 6. MINIMUM VIABLE PAPER

If wavefront tracking proves fragile near synchronization points, the surviving core is still publishable: **affine operator decomposition with probe-based extraction and self-verification (zero physics duplication) + equilibrated complex64 batched LU with FP64 Rayleigh/Newton correction, certified verdict-identical against a 37-case validated LST matrix on consumer hardware**. Degraded-mode engine: CPU-QZ coarse seeding + GPU batch-refinement of *all* candidates (no continuation), still 10-50Ã— end-to-end.

Strong-paper target figures/tables: (T1) 37-case CPU-vs-GPU verdict identity + eigenvalue-error table (~1e-11); (F1) time-to-solution across the benchmark matrix (algorithm Ã— hardware, log scale); (F2) eigenvalue error vs RQI iteration at c64, the three Îº regimes; (F3) Ozgen c_i map recomputed at 4Ã— grid density in less time than the deployed 1Ã— run â€” a *new physics capability*, the strongest possible reviewer argument; (F4) map of fallback locations showing they coincide with physical branch interactions (turns the weakness into physics insight).

---

## Ranked findings

1. **HIGH â€” Baseline fairness:** algorithmic speedup (10-30Ã—) is separable from hardware speedup (5-15Ã—); claim them decomposed or the paper dies in review.
2. **HIGH â€” 3D operator is not affine in (Î±,Î²)** (measured 6.1e-4 probe residual; solver.py:632,679-684): probe must use the (k, cosÏˆ, sinÏˆ)Ã—{1,1/Re} basis (8-16 terms, verified 9.1e-14) or fig10.4 silently forfeits the GPU path via the auto-fallback.
3. **HIGH â€” Candidate-set semantics and mode death:** Ozgen two-domain matched argmax (discrete_mode.py:83-107) and M10 band collapse (compute_mack_fig10_4.py:112-116) require a bundle tracker (3-8 modes) with device-side decay/band/stationarity filters and explicit "no discrete mode" verdicts â€” a single-mode wavefront produces wrong physics.
4. **MED â€” FP32 frontier at Nâ‰³256 Î´*-scale:** Îº_equil â‰ˆ 1.1e7 â†’ refinement rate ~0.7, 12 iterations, still 3.4e-10, but Nâ‰³300 breaks; require equilibration always, and an FP64 zgetrfBatched polish path (2-3 s/batch on Ada, free on A100).
5. **MED â€” Fallback Amdahl:** >3-5% CPU-QZ fallback halves the Ozgen speedup; mandate consensus re-seeding, async CPU/GPU overlap, and fallback-fraction reporting.
6. **MED â€” Library/Windows uncertainty:** getrfBatched efficiency at nâ‰ˆ1000 unproven (Â±3Ã— model risk â€” run a 1-day CuPy spike first); driver sysmem-fallback silently corrupts benchmarks; WDDM launch latency constrains design to batched kernels/graphs.
7. **LOW â€” Ground-truth correction: B is SINGULAR after BC row replacement** (8 zeroed rows at solver.py:581,588,591 and temporal_solver.py:180-193), giving 8 infinite eigenvalues. Harmless for (Aâˆ’cB) LU-based RQI (verified working), but any Bâ»Â¹A/standard-EVP reformulation is forbidden.
8. **LOW â€” Two 2D temporal formulations coexist** (Mack-style solver.py:414-608 vs Ozgen-style temporal_solver.py:37-207) and the asymptotic freestream BC rows (solver.py:812-869) are parameter-dependent (sqrt) â€” handle as per-family probing plus an 8-row per-point device patch (or CPU-side refinement stage).

### Critical Files for Implementation
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/solver.py (3D assembly lines 611-783, BCs 786-869, continuation 1121-1288, spatial local iterations 1536-1764)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/equations.py (2D spatial QEP assembly, the probe target)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/temporal_solver.py (Ozgen-formulation 2D temporal EVP used by the c_i grids)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/verification/mixed_mode/ozgen_fig3/_refdigitize/discrete_mode.py (the candidate-set/two-domain filter semantics the GPU tracker must reproduce)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/verification/compute_mack_fig10_4.py (flagship 3D workload: fixed per-Mach grids, band selection, mode-collapse documentation)
