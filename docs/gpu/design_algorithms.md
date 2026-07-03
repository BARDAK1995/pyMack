# pyMack-GPU — Numerical Algorithms Specification

*Produced by the design fan-out (2026-07-02); all numerical claims were measured on the real pyMack matrices (read-only). Companion to PLAN.md.*

## Summary

I designed the algorithm-level spec for pyMack's batched dense GPU sweep engine, grounded in direct reading of solver.py, equations.py, temporal_solver.py, spectral.py, and analysis.py, plus two measured anchors on the real code: assembly+QZ is 2.0 s at n=516, and the assembled temporal matrix has raw kappa_2(A)=1.06e8 that collapses to 3.8e4 after two-sided power-of-2 equilibration â€” this single measurement certifies the FP32-factorize/FP64-refine strategy. I verified the exact affine parameter structure of every operator: the spatial QEP needs only 6 constant matrices (T = K1 + omega*K2 + R^-1*K3 + alpha*(K4 + R^-1*K5) + alpha^2*R^-1*K6), the temporal 2D pencil needs <=6 for A and exactly 1 for B (B = alpha*B1 with 8 zero BC rows â€” I corrected the shared-context claim that B is nonsingular; it is singular and the design never inverts it). The temporal kernel is two-sided RQI (cubic for nonnormal pencils; one-sided is only quadratic here) with one cf32 batched LU per iteration shared by the right and transposed left solves, FP64 Rayleigh quotients and residuals evaluated matrix-free through the affine basis, and exact BC preservation for free because B's BC rows are zero. The spatial kernel is bordered implicit-determinant Newton on T(alpha) directly (size 4n+1, analytic T' = C1 + 2*alpha*C2), with Muller on the same bordered scalar g(alpha) as the derivative-free fallback â€” no companion blow-up, no batched QZ needed anywhere in the sweep. Wavefront continuation over structured grids uses anti-chain frontiers batched across independent workloads, a cheap on-device mode-identity metric (phase-speed continuity + neighbor eigenvector correlation), and a 2-vector two-sided subspace tracker that solves an analytic 2x2 pencil through the first/second-mode synchronization region instead of letting RQI jump branches. The performance model predicts Ozgen 720-node grid 1600 s -> ~2-5 s, an M10 station 950 s -> ~0.5-2 s, the Mach-6 eN database compute to sub-second (launch-bound, needs frontier fusion), and the full validation sweep to tens of seconds, gated by verdict-equality against the existing 37-case harness.

## Key decisions

- Two-sided (not one-sided) Rayleigh-quotient iteration for the temporal kernel: cubic convergence on these strongly nonnormal pencils, at the cost of one extra transposed triangular solve reusing the same batched cf32 LU.
- Never invert or reduce by B: B has 8 identically-zero BC rows (verified in solver.py:574-592, correcting the shared context) â€” all iterations solve (A - rho*B)x = Bx, which also enforces the discrete BCs exactly at every iteration for free.
- Spatial kernel = bordered implicit-determinant Newton (Spence-Poulton) on T(alpha) directly at size 4n+1 with analytic T' = C1 + 2*alpha*C2; fallback = Muller on the same bordered scalar g(alpha) (1 LU + 1 solve per eval, no eigensolve) â€” batched Muller on sigma_min is rejected because it needs a dense eigensolve/SVD per step.
- Affine operator extraction by probing the existing validated CPU assembly at ~8-14 parameter tuples + least-squares Vandermonde, with a hard 1e-13 held-out self-verification gate and automatic fallback to per-point CPU assembly + batched device solve; exact bases verified from the code: spatial T needs 6 constant matrices, temporal 2D A needs <=6 and B exactly 1 (B = alpha*B1).
- Mandatory two-sided power-of-2 equilibration baked into the device basis matrices: measured on the real assembled operator, it reduces kappa_2 from 1.06e8 to 3.8e4 at n=516, making cf32 LU + FP64 matrix-free residual/refinement safely convergent (contraction ~4.5e-3/step); per-lane promotion to zgetrfBatched only on stall (<5% of lanes expected).
- Wavefront continuation with heterogeneous (mixed-parameter) batches, speculative frontiers, and batching across independent workload panels to reach saturating batch sizes (128-512 at m~500-1000, >=1024 at m~128); mode identity guarded by phase-speed continuity + a one-dot-product p/T-weighted eigenvector correlation against converged neighbors.
- Synchronization-region handling by construction, not retry: switch flagged lanes to a 2-vector two-sided subspace iteration (same batched LU, 4 RHS) with a closed-form 2x2 projected pencil, continue the eigenvalue pair as an unordered set, and assign first/second-mode labels only after exiting the region; async CPU QZ remains the last-resort reseed.
- Mode selection and filtering semantics are inherited, not reimplemented: c-band and residual filters run on device, but leakage/domain-stationarity and final verdicts run the existing CPU filter code on the GPU-found modes, and certification requires verdict equality on the full 37-case harness.
- Growth maximization (Mack fig 10.3/10.4) is implemented as a batched optimizer: coarse (alpha,psi) grids warm-started station-to-station, then batched parabolic/golden-section refinement in alpha across all ~45 (M,R) stations simultaneously, one batched RQI generation per optimizer step.
- Neutral curves reuse analysis.neutral_points_from_growth_map with a batched secant refine_func plugged into its existing hook; Mach-6 N-factor integration is a single device scan matching integrate_n_factor's clip_negative convention.

## Risks (ranked)

- First/second-mode synchronization region: RQI/Newton can jump branches or break down (|w^H B x| -> 0) where modes nearly coalesce. Mitigation: automatic detection (pair distance, bi-orthogonality collapse, correlation ambiguity), 2-vector subspace continuation with closed-form 2x2 pencil, pair-labeling deferred to region exit, async CPU QZ reseed as last resort; per-point kappa_c reported so ill-conditioned answers are flagged, not silently wrong.
- FP32 factorization stalls at high resolution (N=256, y_max~6: max|D2| = 2.5e9 measured, expected equilibrated kappa ~1e6): refinement contraction degrades to ~0.1/step. Mitigation: per-sweep kappa estimation sets the refinement budget; per-lane promotion to batched complex128 LU; worst case the sweep runs FP64-factorized and is still ~10x faster than CPU QZ.
- Kernel-launch/Python/WDDM overhead dominates small-matrix workloads (Mach-6 eN, m~128) and narrow frontiers. Mitigation: speculative multi-frontier batching, batching across independent panels, fused RawKernel assembly+LU for small m, CUDA graphs; the long-term C++/CUDA core removes this class and the design keeps all state in flat device slabs so nothing precludes it.
- Hidden non-affinity or basis drift (e.g., a future upstream change makes a coefficient depend nonpolynomially on a swept parameter): the 1e-13 held-out probe verification gate catches it deterministically and triggers the per-point CPU-assembly fallback, so results stay correct (only slower); the gate must run at the start of every sweep, not once.
- Mode-selection (verdict) mismatch rather than eigenvalue mismatch: GPU finds correct eigenvalues but a different mode gets selected than the CPU full-spectrum path picks. Mitigation: run the existing CPU filter/selection code verbatim on GPU-found candidates, seed from CPU QZ anchors that fix branch identity, and gate on verdict equality across all 37 cases before any workload is switched over.
- cuBLAS getrfBatched throughput at m~1000 may fall below the assumed ~15% efficiency (pivot row-swap traffic), stretching M10-class wall-clock estimates by 2-5x. Mitigation: benchmark early on the actual RTX card, switch to cuSOLVER batched or a tiled RawKernel LU if needed; even a 5x miss still leaves >50x speedup vs CPU.
- VRAM pressure on 8 GB cards at m=1028 with batch 512 (4.3 GB of lane slabs plus workspaces): batch size is an engine parameter (256 on 8 GB), with lane compaction keeping slabs dense; overflow degrades throughput, not correctness.

---

# Batched Dense Sweep Engine â€” Numerical Algorithm Specification
**pyMack GPU engine #1 (CuPy + RawKernels, FP32-first with FP64 certification)**

All file references are to `C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST`. All claims marked **[verified]** were checked against the code or measured on this machine during design.

---

## 0. Ground truth verified from the code (read this first â€” two corrections to shared context)

1. **[verified] B is SINGULAR, not nonsingular.** In `pymack/solver.py:574-592` (2D temporal) and `apply_wall_bc_3d` / `apply_dirichlet_freestream_bc_3d` (`solver.py:786-809`), the 8 BC rows of B are zeroed (`B[row,:]=0`) while A's BC rows become unit/operator rows. The pencil (A,B) is regular with 8 infinite eigenvalues. Consequence: **nothing in this engine may form B^-1 A**; all iterations solve shifted systems `(A - rho*B) x = B x_prev`, which is well-posed. Bonus (Section 1.4): the zero BC rows make BC enforcement in RQI automatic and exact.
2. **[verified] Exact affine parameter structure.** From `pymack/equations.py:90-264` (`assemble_compressible_matrices`): omega enters C0 only and linearly (all `iw` terms); 1/Re enters via `visc/cond/diss` prefactors; C2 is *purely* (1/Re)-proportional; C1 has {1, 1/Re} parts and **no omega**. From `solver.py:414-608` and `temporal_solver.py:37-207`: A(alpha,Re) spans monomials {1, alpha, alpha^2} x {1, 1/Re} (6 matrices max), and **every entry of B carries the factor i*alpha** â€” B = alpha * B1 with one constant matrix. From `solver.py:611-783` (3D oblique, parametrized by k=|(alpha,beta)| and psi with alpha=k cos psi): scalars appearing are exactly {1, k, k^2, k cos psi, cos psi, sin psi, k sin psi} x {1, 1/Re}, and B = (k cos psi) * B1. Base flow sampling (`scales.py:236-267`) depends only on (grid, length_scale), never on (alpha, omega, Re) â€” so for a fixed profile+grid the decomposition is **exact**, not approximate.
3. **[measured] Conditioning at n=516** (M4.5 table_11_1 profile, R=1500, alpha=0.06, N=128, y_max=42, L_star scale): `max|A| = 5.33e4` (viscous D2/Re blocks vs O(1) BC rows), raw `kappa_2(A) = 1.06e8`, and after two-sided inf-norm equilibration `kappa_2 = 3.82e4`. `max|D2|` from `spectral.py`: 1.55e8 (N=128, y_max=6), 1.48e6 (N=200, y_max=150), 2.48e9 (N=256, y_max=6), 8.6e4 (N=31, y_max=15). Assembly+QZ measured at 2.0 s for dim 516 on this machine.

**Notation.** n = N+1 points; temporal-2D pencil size m = 4n (516 at N=128); temporal-3D m = 5n (1005 at N=200, M10); spatial QEP size m = 4n (~124-128 for the Mach-6 eN cases). "Lane" = one parameter point in a batch. eps32 = 1.19e-7, eps64 = 2.2e-16.

---

## A. Affine operator extraction by probing (zero physics duplication)

**Per sweep** (fixed profile, grid, Ma, Pr, gamma, wall_bc, length_scale, lambda_mu_ratio):

1. **Basis declaration.** Temporal 2D: `S_A = {1, a, a^2, 1/R, a/R, a^2/R}`, `S_B = {a}`. Spatial: `S_T = {1, w, 1/R, a, a/R, a^2/R}` (6 matrices; C2 is pure 1/R **[verified]**). Temporal 3D at fixed psi: `S_A = {1, k, k^2} x {1, 1/R}` pruned, `S_B = {k}`. Temporal 3D over (k, psi): `S_A = {1, kc, k, k^2, c, s, ks} x {1, 1/R}` pruned (c=cos psi, s=sin psi), `S_B = {kc}`.
2. **Probing.** Choose M = |S| + 2 parameter tuples (Chebyshev-spaced in each parameter over the sweep's range, plus 2 random). Call the *existing validated CPU assembly* at each tuple: `assemble_temporal_compressible_3d_evp` (returns A,B directly, `solver.py:611`), `_assemble_spatial_qep` (returns C0,C1,C2 post-BC, `solver.py:116-173`). For `solve_temporal_2d` (`temporal_solver.py`) the assembly is inline before `linalg.eig`; **preferred**: a tiny upstream refactor extracting `assemble_temporal_2d(...) -> (A, B, y, D1)` (pure code motion, CPU path byte-identical); interim: capture (A,B) by intercepting `scipy.linalg.eig` (**[verified]** this interception works â€” used for the conditioning measurement).
3. **Least-squares Vandermonde inversion, whole-matrix at once.** With V[k,m] = s_m(p_k), compute K_m = sum_k (V^+)_{m,k} A(p_k) â€” M matrix-linear-combinations, done once on CPU in FP64. Cost: M x 2-5 ms assembly + trivial algebra.
4. **Self-verification (hard gate).** Assemble at 2 held-out random tuples; require `||A_cpu - sum_m s_m K_m||_F / ||A_cpu||_F < 1e-13` (exact affinity â‡’ only roundoff survives). On failure: log the offending basis, and **fall back automatically** to per-point CPU assembly + async H2D transfer with the batched solve unchanged (engine API identical, just slower assembly).
5. **BC rows.** Wall + Dirichlet-freestream BC rows are parameter-independent **[verified]** (`temperature_wall_operator` uses constant D1) and come out of probing correctly since probing runs post-BC. The Appendix-B asymptotic freestream BC (`_apply_asymptotic_freestream_bc_3d`, `solver.py:812-869`) depends on (alpha, beta, c_ref): when a workload needs it (the `refine_temporal_compressible_3d_asymptotic` outer loop), treat it as a **per-lane rank-4 row replacement**: the 4 bc rows are built from an 8x4 null-space problem â€” microseconds per lane on CPU â€” uploaded and scattered into the lane's assembled matrix each outer iteration. Default sweep mode = Dirichlet BC + leakage post-filter, matching the production continuation path (`compute_mack_fig10_4.py:166-185` uses exactly this **[verified]**).
6. **Upload once per sweep:** K_m in complex128 (residual path) and complex64 pre-scaled copies (factorization path). VRAM: at m=1005, 12 matrices x 16 MB (cf128) + 8 MB (cf64-scaled-cf32) â‰ˆ 300 MB worst case â€” negligible.

**Equilibration (part of extraction).** At a sweep-central parameter point, assemble A in FP64, compute two-sided inf-norm scalings r_i, c_j, **round to powers of 2** (exact in binary FP, so CPU/GPU results remain bitwise comparable through scaling), and bake D_r K_m D_c into the device copies. Measured effect: kappa 1.06e8 -> 3.8e4 at n=516. Eigenvalues are invariant under two-sided diagonal scaling of the *pencil* (both A and B scaled identically); eigenvectors are unscaled by D_c^-1 on output.

---

## B. Batched TEMPORAL kernel â€” two-sided Rayleigh-quotient iteration on (A - cB)

**Choice: two-sided, not one-sided.** These pencils are strongly nonnormal (boundary-layer mode nonorthogonality; measured kappa above). One-sided RQI (rho = x^H A x / x^H B x) is only quadratically convergent for nonnormal problems and its stationary values are biased by nonnormality. Two-sided RQI (Ostrowski; Parlett 1974 generalization) with rho = (w^H A x)/(w^H B x) is **locally cubically convergent for simple eigenvalues of nonnormal pencils** â€” the eigenvalue error is O(|e_x||e_w|) per step and both vector errors contract by the eigen-gap ratio, giving the cubic composition. The extra cost is one transposed triangular solve reusing the **same LU factorization** â€” essentially free.

**Per-iteration algorithm (batch of L lanes, per lane):**
1. **Assemble** M = sum_m s_m(p) K_m^(32) âˆ’ rho_k * s_B(p) K_B^(32) in complex64, directly in the lane's slab (custom RawKernel: one fused pass over the 6-13 basis matrices; K's are batch-shared so they stay hot in L2). Traffic â‰ˆ 14 x m^2 x 8 B â‰ˆ 15 MB/lane at m=516 â†’ ~15 us/lane at 1 TB/s.
2. **Factor**: `cublas cgetrfBatched` on the active-lane slab.
3. **Solve** (same factorization, `cgetrsBatched`): `M x~ = B x_k` (no-trans) and `M^H w~ = B^H w_k` (trans='C'). B x_k is formed matrix-free: B = alpha*B1 â‡’ one batched GEMV against K_B + scalar scale.
4. **Inexact-solve refinement (conditional):** one step `x~ <- x~ + M32^{-1}(Bx_k âˆ’ M64 x~)` with the FP64 residual evaluated matrix-free through the cf128 K's. Contraction factor â‰ˆ kappa_eq*eps32 â‰ˆ 4.5e-3 at n=516 â€” one step suffices. Skipped when the FP64 eigen-residual (step 6) is already decreasing cubically; inexact-RQI theory (Freitag & Spence) shows solve accuracy proportional to the current eigen-residual preserves the convergence rate, so most iterations need zero refinement steps.
5. **Normalize**: x_{k+1} = x~/||x~||_2 with phase fixed by rotating the component that was largest at seed time to be real-positive (prevents phase drift in neighbor-correlation metrics); same for w.
6. **FP64 Rayleigh quotient + residual**: rho_{k+1} = (w^H A x)/(w^H B x) and r = A x âˆ’ rho B x, both evaluated matrix-free: stack lane vectors into X (m x L), compute K_m X with ~12 zgemms shared across the whole batch, then combine with per-lane scalars s_m(p). FP64 GEMM cost â‰ˆ 12 x 8 m^2 per lane per pass (~0.05 GFLOP at m=516) â€” fits the RTX FP64 budget because it is O(m^2), not O(m^3).
7. **Breakdown guard**: if |w^H B x| < 1e-6 ||w|| ||Bx|| â€” modes coalescing or wrong pairing â€” divert the lane to the sync-region subspace tracker (Section D.4).

**BC-row interaction [verified structural fact]:** because B's 8 BC rows are identically zero, the RHS B x has exact zeros there, and M = A âˆ’ rho B retains A's BC rows unchanged (unit rows / adiabatic D1 row). Hence the solve yields x satisfying the discrete BCs **exactly at every iteration** â€” no projection, no BC drift, regardless of precision. The left vector w acquires components in BC rows; harmless (it never multiplies a BC-violating direction in the RQ because Bx is zero there).

**Stopping:** converged when `||r||_inf / (||A||_eq ||x||_2) < 5e-13` (FP64-certified) **and** |Delta rho| < 1e-10 max(1,|rho|); accept-provisional at 5e-7 for interior sweep points that only feed the predictor. Max 6 iterations; with neighbor seeds |c0 âˆ’ c| ~ 1e-3-1e-2, cubic convergence gives **2-4 iterations typical** (1e-3 -> 1e-9 -> eps in two steps).

**Deflation / locking = lane compaction.** Lanes are independent parameter points chasing one mode each, so there is no spectral deflation; converged lanes are removed by index compaction each iteration so `getrfBatched` always runs on active lanes only. When two lanes at the *same* parameter point track different mode families (e.g., first and second mode layers of the same grid), collision is detected by |w_i^H B x_j| growing toward |w_i^H B x_i| (bi-orthogonality collapse) â†’ sync-region handling.

**Attainable eigenvalue accuracy:** backward error is driven by the FP64 residual (eps64-level, since RQ and r are FP64); forward error |delta c| â‰ˆ kappa_c * eps64 * ||(A,B)||_eq with kappa_c = ||x|| ||w|| / |w^H B x|, reported per point for free. Typical kappa_c ~ 1e2-1e4 â‡’ |delta c| ~ 1e-12-1e-10 â€” orders of magnitude inside the 1e-5 certification target, except at synchronization where kappa_c blows up and the subspace method takes over.

---

## C. Batched SPATIAL kernel â€” bordered implicit-determinant Newton on T(alpha)

Problem: find alpha with T(alpha) phi = 0, T(alpha) = C0 + alpha C1 + alpha^2 C2, omega and R fixed per lane. **No companion linearization anywhere in the sweep** (the 8n x 8n companion + shift-invert in `solver.py:215-358` is used only for CPU seeding).

**Comparison and choice:**
- **(a) Bordered / implicit-determinant Newton (Spence-Poulton) â€” PRIMARY.** Solve `[[T(alpha), b],[c^H, 0]] [x; g] = [0; 1]`. g(alpha) is a smooth scalar proportional to det T(alpha)/det(bordered), with simple zeros exactly at the simple eigenvalues; the bordered matrix stays **well-conditioned at the root** (unlike T itself, which is exactly singular there â€” this is why plain inverse-iteration Newton frays in FP32). Analytic derivative: g'(alpha) from a second solve with the same factorization, RHS `[-T'(alpha) x; 0]`, where **T'(alpha) = C1 + 2 alpha C2 analytically** â€” assembled from 3 basis matrices (K4 + R^-1 K5 + 2 alpha R^-1 K6). Newton: alpha+ = alpha âˆ’ g/g'. Per iteration: 1 cf32 batched LU of size (4n+1), 2 batched solves (can be one `getrs` call with 2 RHS), 1 FP64 residual pass. Border vectors: c = seed eigenvector from neighbor (normalization functional), b = T'(alpha_seed) x_seed; frozen during one grid point's Newton, refreshed at convergence. Guard: |g'| small or bordered LU growth large â‡’ refresh borders from current x and a left-vector estimate.
- **(b) Inverse-iteration Newton (nonlinear inverse iteration / Jacobi-Davidson-like):** x+ = T^{-1} T' x, alpha+ = alpha âˆ’ (c^H x)/(c^H x+). Same per-iteration linear algebra but factors T itself, which becomes numerically singular at convergence â€” in cf32 the factorization quality collapses exactly when you need it; rejected as primary, retained as a mathematical cross-check in FP64 tests.
- **(c) Batched Muller on sigma_min:** the CPU version (`solve_spatial_muller`, `solver.py:1582`) evaluates the eigenvalue-nearest-zero of L(alpha) via full `eigvals` per step â€” a dense eigensolve per iteration, precisely the primitive the GPU lacks; sigma_min needs an SVD, also unbatched. Rejected **as stated**. However, Muller **on the bordered scalar g(alpha)** needs only 1 bordered LU + 1 solve per evaluation, no derivative, and inherits g's smooth deflated-determinant behavior. **FALLBACK = Muller on g(alpha)**: used when Newton's basin fails (3 non-decreasing steps) â€” notably near the first/second-mode synchronization where two roots are close and Muller's quadratic model natively represents a root pair.

**Eigenvector recovery:** at convergence x from the bordered solve is (up to normalization) the null vector; certify with the FP64 QEP residual `||(C0 + a C1 + a^2 C2) x|| / (scale ||x||)` evaluated matrix-free through the 6 K's â€” the same residual filter as `_filter_with_residual` (`solver.py:361`), so the CPU filter semantics carry over verbatim.

**Precision:** identical scheme to the temporal kernel â€” cf32 bordered LU (the bordered matrix is well-conditioned near the root, so plain LU-based iterative refinement on the bordered solves converges with contraction kappa_eq*eps32), FP64 g, g', and residuals. Newton with FP64 function values and cf32-preconditioned solves retains full quadratic convergence to FP64-limited roots.

**Iteration count:** neighbor-continued seeds |Delta alpha| ~ 1e-3: 3-4 Newton steps to |g| < 1e-12Â·scale.

---

## D. Seeding + wavefront continuation over structured grids

**D.1 Seeds.** Full-spectrum QZ on CPU (existing scipy path with the validated filter stack â€” c-band, leakage, QEP residual) at a small set of anchor points: for each mode family, one anchor column (3-5 QZ solves along alpha or omega at a mid-grid R) to establish branch identity and the local c-band. Seeds run on CPU threads concurrently with K_m extraction/upload. Spatial anchors use the existing shift-invert companion (`solve_spatial`) with the workload's own band selection (e.g., `trace_mazhong_curves.py:34-53` semantics).

**D.2 Frontier ordering and iteration batches.** The grid is a DAG: point q is *ready* when >=1 (prefer >=2) of its already-processed neighbors converged. Process **anti-chain frontiers** (diagonal wavefronts from the seed column). Predictor: first-order extrapolation `c_pred = c_nb + (dc/dp)|_nb Â· Delta p` using converged-neighbor differences (this is the grid-parallel generalization of `continue_temporal_mode_3d`'s nearest-neighbor target, `solver.py:1121-1288`); eigenvector seed = converged neighbor's (x, w). An **iteration batch is heterogeneous by construction**: lanes carry distinct (R, alpha/omega[, psi]) tuples, per-lane scalars s_m(p), per-lane shifts rho_k â€” the batched assembly is exactly the tensor contraction over shared K_m, so mixed-parameter batches cost the same as uniform ones. To reach saturating batch sizes despite frontier widths of 16-64: (i) run **all independent workload panels simultaneously** (e.g., 4 Mach panels x all R stations; all eN frequencies), (ii) allow **speculative frontiers**: launch generation g+1 with predictors from provisionally-converged generation g, verify afterward and re-run the (rare) invalidated lanes.

**D.3 Mode-identity safeguards (on device, cheap).** Accept a converged lane only if:
1. **Phase-speed continuity:** |c âˆ’ c_pred| < max(band_tol, 5 x median neighbor step); plus the workload's absolute c-band (e.g., CR in [0.40, 0.95], c_i < 0.12 for fig10.4 first mode **[verified constants]**).
2. **Eigenvector correlation:** rho_x = |x_nb^H D x| / (||D^(1/2)x_nb|| ||D^(1/2)x||) with D a diagonal weight selecting the pressure+temperature blocks (these carry the sharpest first-vs-second-mode signature: the second mode's p-eigenfunction phase structure). All lanes share one grid, so this is a single weighted dot product per lane (O(m) flops). Thresholds: >0.9 accept; 0.6-0.9 â†’ insert a midpoint (halve the continuation step for that lane and re-run); <0.6 â†’ mode swap: reseed from neighbor consensus (majority vote of adjacent converged lanes' predictors) or enqueue async CPU QZ fallback.
3. **FP64 residual + (final pass only) the CPU filter stack** â€” leakage (`temporal_freestream_leakage_3d`) and domain-stationarity are computed on the *selected converged modes only* (hundreds, not the full spectrum), on CPU, so verdict logic is character-identical to the validated path.

**D.4 Synchronization region (first/second-mode near-coalescence â€” the known hard case).** Detection: |c_F âˆ’ c_S| < delta_sync between the two tracked layers, RQI breakdown guard |w^H B x| collapse, or correlation ambiguity (rho_x similar against both anchors). Response â€” **2-vector two-sided subspace continuation** instead of scalar RQI: iterate `M X~ = B X`, `M^H W~ = B^H W` with X, W in C^{m x 2} (same batched LU, 4 RHS), bi-orthonormalize the 2-blocks (tiny 2x2), form the projected pencil (W^H A X, W^H B X) in FP64 and solve the 2x2 generalized eigenproblem **in closed form** on device. The eigenvalue *pair* is continued as an unordered set through the region; branch labels are assigned only after exit, by eigenvector correlation against outside-region anchors on both sides. This removes the branch-jumping failure mode by construction rather than by retry, and is a publishable element of the method. Spatial analogue: Muller-on-g with pair-aware quadratic model, or the same 2-vector idea on the bordered formulation. Last resort per flagged point: async CPU QZ (full spectrum + filters) requeues the point â€” the engine's answer is then still certified, just slower for that point.

**D.5 Neutral-curve extraction.** From the completed growth grid (c_i(R,alpha) temporal; âˆ’alpha_i(R,omega) spatial): reuse `analysis.neutral_points_from_growth_map` (`pymack/analysis.py:682`) on CPU for bracketing (grid is tiny), and pass a **batched refine_func**: all sign-change brackets across all rows/branches become one batch of scalar secant iterations on c_i(alpha) (each secant step = one batched RQI evaluation at the trial abscissae, warm-started from the bracket endpoints' eigenvectors). Neutral points to 1e-6 in 3-4 batched steps. This plugs into the existing hook signature **[verified]**.

---

## E. Mixed-precision scheme and certification

**Placement:** complex64 = batched factorizations (`cgetrfBatched`) and triangular solves; complex128 = affine residual/RQ evaluation (matrix-free through K_m^(64)), eigenvalue updates, refinement residuals, N-factor integration. Nothing is ever computed in FP64 dense O(m^3).

**Why it works here (measured):** two-sided power-of-2 equilibration reduces kappa_2 from 1.06e8 to **3.8e4 at n=516** [measured]. LU-based iterative refinement converges when kappa_eq * eps32 < 1; here the contraction factor is ~4.5e-3 per refinement step â€” one step recovers ~2.3 digits beyond FP32, and the RQI/Newton outer loop itself acts as eigenvalue-level refinement. At N=256, y_max=6, max|D2| = 2.48e9 [measured] â‡’ expected kappa_eq ~ 1e6, contraction ~0.1 â€” still convergent with 1-2 refinement steps. The engine measures kappa_eq once per sweep (LU-based condition estimate at the central point) and sets the refinement-step budget automatically.

**Refinement loop (per solve, conditional):** r64 = b âˆ’ M64 x (matrix-free); x <- x + M32^{-1} r64; repeat while ||r64|| decreases by >10x, max 2 steps. Applied only when the outer eigen-residual stalls (inexact-RQI tolerance theory makes most inner solves refinement-free).

**Promotion policy (per lane):** promote to a complex128 factorization sub-batch (`zgetrfBatched`) when any of: (i) eigen-residual reduction < 10x per outer iteration after iteration 3; (ii) refinement contraction > 0.5; (iii) LU pivot-growth flag or `info != 0`; (iv) |w^H B x| < 1e-6 (after the sync-region check). Expected promotion rate: <2-5% of lanes; FP64 batched LU at m~1000 costs ~64x the flops at 1/64 rate â‡’ same order wall time as the cf32 batch for that small sub-batch â€” acceptable. If FP64 also stalls â†’ CPU QZ fallback queue.

**Certification gate:** rerun the 37-case verification matrix (`verification/` + `verdict.json` + `build_success_matrix.py`) with the GPU engine substituted; require (i) verdict equality case-by-case, (ii) eigenvalue metrics within 1e-5 relative (expected achieved: <=1e-9), (iii) identical mode *selection* â€” guaranteed by running the CPU filter stack on GPU-found modes rather than reimplementing filters.

---

## F. Per-workload mapping

1. **Ozgen temporal c_i grid (720 nodes, `temporal_solver.solve_temporal_2d`, m=516).** Basis {1,a,a^2}x{1,1/R} (+ B = a*B1). 3 CPU QZ seed points; wavefront over (R, alpha); ~4 RQI iters/point. Filters: c-band (`solver` physical band) + FP64 residual; final CPU cross-check on 5% random sample.
2. **Mack fig 10.3/10.4 â€” growth *maximization* per (M,R), 3D oblique m=5n (655-1005).** This is an optimization, not a fixed grid: **phase 1** â€” the existing coarse (alpha,psi) grid (`ALPHA_GRID`/`PSI_GRID`, 119-136 points/station **[verified]**) becomes one batch, warm-started station-to-station by wavefront in R (the whole (alpha,psi) field of station R_{i} seeds station R_{i+1}); band selection replicates `first_mode_growth` (CR 0.40-0.95, c_i<0.12, Dirichlet BC â€” the validated fast path). **Phase 2** â€” batched ridge refinement: fit a 2D quadratic to the top-k grid values per station, then run **batched safeguarded parabolic/golden-section in alpha at the best psi across all ~45 (M,R) stations simultaneously** â€” each optimizer step evaluates one batched RQI solve at the trial alphas (seeded from the nearest converged grid mode), so every station shares the same generation of factorizations. Termination: |Delta alpha| < da/16 (matches the CPU refine granularity `compute_mack_fig10_4.py:231-239`). fig10.3-style 2D maximizations are the psi=0 special case of the same optimizer.
3. **Ma & Zhong spatial neutral traces (2 modes x 16 R x 22 omega).** Spatial bordered Newton on (R, omega) grids per mode family with the c-band from `trace_mazhong_curves.py` applied as the mode-identity band; seeds via CPU `solve_spatial` shift-invert at 2-3 anchors per family; neutral contour via Section D.5. Output schema identical to `mazhong_curve_grid.csv`.
4. **Mach-6 eN pipeline (3,649 spatial solves at mâ‰ˆ124-128 + N-factor).** All fixed-frequency lanes batched together: frontier along R, batch = all frequencies (~30-100 lanes) x speculative next-columns to keep >=1024 lanes in flight (these matrices are tiny; Section G says this workload is launch-bound, so lanes are nearly free). sigma(R)=-alpha_i per frequency; **N-factor on device**: cumulative trapezoid of max(sigma,0) over R per frequency (one scan kernel), envelope + transition-R on CPU via `integrate_n_factor` conventions (clip_negative=True **[verified]** `analysis.py:2248-2280`). The whole database refresh becomes a single engine call.
5. **Full validation sweep (~7,000 solves):** the union of the above through the engine API, gated by the Section E certification.

---

## G. Complexity and performance model

**Per-iteration FLOPs per lane** (complex LU = (8/3)m^3 real flops; solve = 8m^2/RHS; residual pass = ~12 basis GEMVs x 2 evaluations):

| m | assemble (cf32) | LU (cf32) | 4 solves | FP64 residual+RQ | total/iter |
|------|------|------|------|------|------|
| 516 (temporal 2D) | 0.03 GF | 0.37 GF | 0.009 GF | 0.10 GF (fp64) | ~0.5 GF |
| 1005 (temporal 3D M10) | 0.12 GF | 2.71 GF | 0.032 GF | 0.19 GF (fp64) | ~3.1 GF |
| 128 (spatial eN, bordered 4n+1) | 0.002 GF | 0.006 GF | â€” | 0.003 GF | ~0.01 GF |

**Device model (RTX 4090-class; scale linearly for smaller cards):** FP32 peak 82 TF, assume 15% efficiency for `getrfBatched` at m~500-1000 â‡’ ~12 TF/s effective; FP64 1.3 TF at ~60% GEMM efficiency; 1 TB/s bandwidth. Batch sizes to saturate: **>=128 lanes at m~500-1000** (each LU has limited intra-matrix parallelism; 128-512 recommended), **>=1024 lanes at m~128**. VRAM: lane slabs dominate â€” batch 512 at m=1028 cf32 = 4.3 GB (use 256 on 8 GB cards; engine parameter).

**Wall-clock predictions:**
- **Ozgen 720-node grid:** 720 x 4 iters x 0.5 GF â‰ˆ 1.4 TF-mixed â‡’ ~0.3 s compute; + 3 CPU QZ seeds (~6 s, overlapped with extraction) + probing (~10 x 5 ms) + frontier overheads â‡’ **~2-5 s total vs 1,600 s CPU (~400x)**.
- **Mack fig 10.4, all four panels:** M10 dominates: 9 stations x ~140 points x 4 iters x 3.1 GF â‰ˆ 15.6 TF â‡’ ~2-4 s; M4.5-7 panels (m=655-855) add a few seconds â‡’ **~10-20 s for the entire figure vs multiple hours CPU**; single M10 station 950 s -> **~0.5-2 s**.
- **Ma & Zhong traces:** 704 points x 4 x ~1.3 GF â‰ˆ 3.7 TF â‡’ **~1-2 s + seeds**.
- **Mach-6 eN:** compute ~0.1 s; wall-clock is **launch-bound** (see below) â‡’ **sub-second compute, <2-3 s end-to-end** including N-factor.
- **Full validation matrix:** dominated by CPU seeds and base-flow construction (cached; ~2 s/profile) â‡’ **tens of seconds**, matching the target.

**Where the model bites (ranked):**
1. **Kernel-launch + Python overhead per frontier generation** â€” each generation issues ~6-10 kernels; at m=128 this dominates entirely. Mitigations: speculative multi-frontier batching, fused assembly+small-LU RawKernel (one CTA per lane; m=124 cf32 matrix = 123 KB â€” marginally above the 99 KB shared-memory limit, so use a register-blocked panel LU or accept cuBLAS), CUDA graphs on the steady-state iteration loop (roadmap: the C++/CUDA core removes this class entirely â€” nothing in this design precludes it since all state lives in flat device slabs).
2. **Windows/WDDM submission latency** (local dev is Windows 10): enable hardware-accelerated GPU scheduling; batch enough work per submission; this disappears on Linux/A100.
3. **FP64 residual passes** â€” O(m^2) x 12 basis matrices; at m=1005 they are ~7% of iteration cost on a 4090 but grow on cards with weaker FP64; mitigation: evaluate residuals every other iteration once contraction is verified cubic.
4. **`getrfBatched` pivoting at m~1000** â€” row-swap traffic is memory-bound; if measured efficiency falls below ~10%, switch to `cusolver` batched or tile-blocked RawKernel LU (roadmap).
5. **H2D transfer:** only K_m (tens of MB once per sweep) + seed vectors + per-lane scalars â€” never on the critical path.

---

## H. Module map (implementation skeleton, no physics duplicated)

```
pymack_cuda/
  affine.py        # probing, Vandermonde LSQ, self-verification, equilibration, upload
  engine_temporal.py  # batched two-sided RQI, lane compaction, sync-region 2x2 subspace tracker
  engine_spatial.py   # bordered implicit-determinant Newton + Muller-on-g fallback
  wavefront.py     # grid DAG, frontiers, predictors, speculative batching, reseed queue
  precision.py     # cf32/cf64 slabs, refinement loop, promotion policy, kappa estimation
  fallback.py      # async CPU QZ worker pool (reuses scipy paths + existing filters verbatim)
  workloads.py     # Ozgen grid, fig10.3/10.4 batched optimizer, Ma&Zhong traces, eN pipeline
  certify.py       # 37-case gate: GPU vs CPU verdict equality + metric deltas
```

### Critical Files for Implementation
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/solver.py (temporal 2D/3D assembly + BC structure, spatial QEP/companion seeding, existing local iterations and continuation to generalize)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/equations.py (assemble_compressible_matrices â€” the affine C0/C1/C2 structure the probing extracts)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/temporal_solver.py (Ozgen-arrangement 2D pencil; needs the assemble-return refactor)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/analysis.py (neutral_points_from_growth_map refine hook, integrate_n_factor conventions)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/verification/compute_mack_fig10_4.py (the growth-maximization workload contract: bands, grids, fast-path BC choice)
