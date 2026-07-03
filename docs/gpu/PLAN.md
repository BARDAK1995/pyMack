# pyMack-GPU: a GPU-native batched LST engine (APPROVED PLAN)

*Approved 2026-07-02. Companion design documents: [design_algorithms.md](design_algorithms.md),
[design_architecture.md](design_architecture.md), [design_redteam.md](design_redteam.md).*

## Context

pyMack is a validated compressible LST solver — 37-case validation matrix vs
Mack/Malik/Ma&Zhong/Özgen. Production sweeps are slow: Özgen c_i grid 720 nodes = 1,600 s;
Mack fig10.4 M10 ≈ 8 s/solve × 1,071 solves ≈ 2.2 h; full validation ≈ 7,000 EVP solves.
Everything bottlenecks on `scipy.linalg.eig` (QZ, full spectrum, per point) — and cuSOLVER has
**no** GPU `zggev`, which is why no GPU LST solver exists.

**Goal ordering (explicit): build something GREAT first** — a GPU-native LST engine nobody has,
drastically faster, that scales — publication (JCP/CPC methods paper) is a byproduct.
Decisions locked: **RTX 4070 Ti (12 GB, Ada, FP64=1/64) first**, A100 later; **CuPy + RawKernels**
first, C++/CUDA core on the roadmap; **batched dense sweep engine** first (shooting engine #2 later).

Planning was grounded by a 3-agent code survey + a 3-agent design fan-out (algorithms /
architecture / adversarial red-team) that **measured on the real matrices** (read-only):
condition numbers, affine-basis exactness, and end-to-end kernel prototypes vs FP64 QZ.

## The scheme (4 pillars, with the red-team's corrections baked in)

**P-A. Affine operator decomposition — assembly becomes a tensor contraction.**
Every assembled operator is *exactly* polynomial in the sweep scalars (measured residuals 1e-16):
- Spatial QEP: `C0 ∈ span{1, ω, 1/Re}`, `C1 ∈ span{1, 1/Re}`, `C2 ∈ span{1/Re}` → 6 constant matrices.
- Temporal 2D (both Mack & Özgen forms): `A ∈ {1,α,α²}×{1,1/Re}` (≤6), `B = α·B̃` (1).
- Temporal 3D: **NOT polynomial in (α,β)** (measured 6.1e-4) — MUST probe in
  `(k, cos ψ, sin ψ)×{1,1/Re}` (≤16 terms, measured 9.1e-14). `pymack/solver.py:632,679-684`.
Extract K_m by **probing the existing validated CPU assemblers** at ~8–14 parameter tuples +
one lstsq Vandermonde (zero physics duplication), with a **held-out self-verification gate
(rtol 1e-12)** and automatic fallback to per-point CPU assembly + batched device solve.
Affinity census (verified per workload): every production sweep = 1–2 families per Mach
(Özgen needs 2: the two-domain stationarity pair; eN needs 2: the N-convergence pair; cone is
1 family — Mangler profile is self-similar across stations). Analytic ∂A/∂α, ∂T/∂α for free.
Only non-affine rows: the asymptotic freestream BC (`solver.py:812-869`) → 8-row per-point
device patch (or keep that refinement stage on CPU).

**P-B. Full-spectrum-free sweeps — batched local iteration + wavefront continuation.**
CPU QZ only SEEDS (sparse spine, e.g. one Re column per family). Then:
- Temporal: **batched two-sided RQI** on (A−ρB) — cubic convergence; one equilibrated
  complex64 batched LU per iteration (cuBLAS `cgetrfBatched`, which EXISTS) shared by right +
  transposed-left solves; FP64 Rayleigh quotients/residuals matrix-free via the affine form.
  **B is SINGULAR** (8 zeroed BC rows — corrected ground truth): never invert B; the zero rows
  make BC enforcement in RQI exact and free. Measured: M4.5 n=645 → 1.3e-11 in 4–6 iters;
  **M10 dim-1005 → 1.4e-11 in 4 iters** (0.29 s CPU proto vs 7.3 s QZ).
- Spatial: **bordered implicit-determinant Newton** on T(α)φ=0 directly (size 4n+1, analytic
  T′=C1+2αC2 — no 8n companion blow-up); Muller on the bordered scalar as derivative-free
  fallback. Measured: 1.2e-12 in 6–7 iters, seed basin ≥2%.
- **Bundle tracking (red-team mandatory):** track 3–8 modes per grid point, not one — the
  validated semantics are an argmax over a filtered candidate SET (`discrete_mode.py:83-107`),
  and modes DIE mid-grid (M10 band collapse, `compute_mack_fig10_4.py:112-116`). Device-side
  filters (c_r band, freestream-decay ratio, two-domain stationarity) + explicit
  "no discrete mode" verdicts. Final selection/leakage filters run the EXISTING CPU code
  verbatim on GPU-found candidates.
- Wavefront continuation over structured grids: frontier batches seeded from converged
  neighbors; synchronization regions (first/second-mode coalescence) handled by a 2-vector
  subspace tracker with a closed-form 2×2 projected pencil (labels assigned on region exit);
  consensus re-seed, then **async CPU-QZ fallback overlapped with the GPU frontier**
  (fallback fraction is a first-class reported metric; >3–5% halves the speedup).
- fig10.3/10.4 growth MAXIMIZATION: batched optimizer — coarse (α,ψ) frontier + batched
  parabolic refinement across all ~45 (M,R) stations simultaneously.

**P-C. Mixed precision with certification — the consumer-GPU enabler.**
Two-sided power-of-2 equilibration is MANDATORY and load-bearing (measured: κ 5.9e8→5.4e5 at
n=645; 2.0e10→**1.2e5** at M10 n=1005 — the flagship is the best-conditioned case!).
complex64 LU + FP64 matrix-free refinement reaches 1e-10..1e-11 eigenvalue accuracy.
FP32 frontier: N≳256 δ*-scale (κ_eq~1.1e7, slow refinement) → per-lane promotion to
`zgetrfBatched` FP64 polish path (2–3 s/batch on Ada; free on A100). Cap c64 at N≤256.

**P-D. GPU-native end-to-end.** The whole sweep lives on device: contraction-assembly →
batched LU/RQI/Newton → filters → neutral-contour + N-factor scan. CPU only: base flow
(solve_bvp, ~2 s, amortized), QZ seeds, final verdict code. Nothing crosses PCIe per point.
CUDA graphs for the RQI inner loop (Windows WDDM launch latency); pinned staging; VRAM-adaptive
tiles (12 GB fits batch-128 of 1005² c64 ≈ 1 GB + K-tensors ≈ 0.3 GB comfortably).

## Architecture (from the design fan-out; CPU path never behaviorally changed)

```
pymack/
  sweep.py               # NEW numpy-safe facade: temporal_sweep()/spatial_sweep(), backend='auto'|'gpu'|'cpu'
  gpu/                   # pymack[gpu] extra (cupy-cuda12x); import-guarded
    affine.py            # AffineOperatorCache + ParameterBasis (PURE NUMPY — testable GPU-less, keeps JAX fork open)
    assemblers.py        # probe adapters binding affine.py to the 4 existing CPU assemblers
    backend.py, batch.py # device policy; TileScheduler (VRAM-adaptive, streams, determinism contract)
    temporal.py          # batched two-sided RQI engine
    spatial.py           # batched bordered-Newton engine
    wavefront.py         # seeding, frontiers, bundle tracker, consensus re-seed, async CPU-QZ fallback queue
    refine.py            # c64→c128 iterative refinement + residual certification
    diagnostics.py       # batched filters: c-band, edge-decay, stationarity, residuals
    api.py               # SweepResult dataclasses (converged/residual/seed_map/meta provenance)
    kernels/cupy_ops.py  # BatchedLinalgOps protocol impl (cuBLAS getrf/getrsBatched, contraction-GEMM)
    kernels/raw/*.cu     # RawKernels: bordered-solve fuse, residuals, equilibration scaling
```
- **New batch API, not a backend flag** on existing functions (different semantics: tracked
  modes + provenance, not full spectra). `pymack.sweep` also ships a CPU ProcessPool backend so
  drivers adopt it unconditionally; `backend='auto'` upgrades to GPU.
- **Exactly ONE CPU-side change**: pure code-motion extraction of the two inline 2D temporal
  assemblies (`solver.py:414`, `temporal_solver.py:37`) into `_assemble_*_evp()` functions,
  certified by **bitwise** (A,B) regression fixtures captured pre-refactor.
- Frozen protocols for the roadmap: `BatchedLinalgOps` (CuPy now → C++/CUDA extension later)
  and `SweepEngine` (dense now → batched shooting engine #2 later).
- Determinism contract: bitwise-stable at fixed (GPU, driver, CuPy, precision, tile_size);
  tile_size + environment recorded in result meta / CSV headers (resumable-grid compatible).

## Milestones (each gated by a real wall-clock number on the RTX 4070 Ti)

- **M0 — CuPy spike (~1 day, do first).** Install `cupy-cuda12x`; benchmark
  `cgetrfBatched`/`getrsBatched` c64/z128 at n∈{516, 645, 1005, 1285}, batch∈{64, 256};
  set NVCP "Prefer No Sysmem Fallback" (driver ≥536 silently spills VRAM→RAM and corrupts
  timings); enable HW-accelerated GPU scheduling. Output: measured GFLOPS → recalibrate the
  performance model (±3× uncertainty today). GATE: batched LU ≥ 1 TFLOPS effective at n≈645.
- **M1 — AffineOperatorCache (CPU-only, no GPU needed).** The code-motion refactor + bitwise
  fixtures; probe adapters for all 4 assemblers; bases incl. the 3D (k,cosψ,sinψ) basis.
  GATE: held-out residual ≤1e-12 on EVERY production family (CI check).
- **M2 — Batched temporal RQI engine.** Equilibrated c64 LU + FP64 refinement, bundle-of-modes,
  TileScheduler. GATE: eigenvalues ≤1e-9 vs CPU QZ on Özgen-M4 + fig10.6-style growth curves;
  single panel end-to-end < 60 s.
- **M3 — Wavefront + seeding + sync-region subspace tracker + async fallback + device filters.**
  GATE: full Özgen family verdict-identical; fallback fraction <5%; **720-node grid ≤ 30 s wall
  (vs 1,600 s deployed)**.
- **M4 — Spatial bordered-Newton engine + Ma&Zhong traces + eN pipeline on device.**
  GATE: Ma&Zhong 704-node trace verdict-identical in seconds; Mach-6 eN database GPU compute
  ≤ ~1 s ("interactive eN").
- **M5 — 3D (k,ψ) batched optimizer → the flagship.** GATE: **fig10.4 M10 station family
  (1,071 solves) ≤ 15 s vs ~7,800 s deployed (~500×)**, verdict-identical.
- **M6 — Full certification + benchmark matrix.** Re-run all 37 cases with
  `PYMACK_SWEEP_BACKEND=gpu` through UNCHANGED judges → verdict identity 37/37, metric drift
  ≤1e-5 (eigenvalue anchors) / ≤0.1% (branch locations). Benchmark matrix: algorithms
  {deployed QZ, shift-invert, RQI/bordered continuation} × hardware {1 core, 12–16 cores,
  4070 Ti mixed, (A100 later)} × 4 workloads; report wall time, accuracy, verdict agreement,
  fallback fraction. Honest decomposition: **algorithmic 10–30× (hardware-independent) ×
  GPU 5–15× ⇒ 100–1000× vs the deployed pipeline** — plus new-capability demos (4×-denser
  Özgen map in less time than today's 1×). Paper drafting starts only after this exists.

**Roadmap (post-v1):** C++/CUDA extension core (swap behind `BatchedLinalgOps`); batched
QR-shooting engine #2 (independent physics cross-check, one thread-block per point);
A100/multi-GPU; station-batched non-self-similar profiles (PSE/DNS mean flows —
affine-per-station, batch across ω); JAX/differentiable-LST fork.

## Top risks (red-team ranked, with mitigations)

1. **Baseline fairness** — algorithmic vs hardware speedup must be claimed decomposed (M6 design).
2. **3D basis trap** — (α,β) probing silently forfeits fig10.4 to CPU fallback → (k,ψ) basis + CI probe-residual check per family.
3. **Candidate-set semantics** — single-mode tracking = wrong physics → bundle tracker + device filters + "no discrete mode" verdicts + verdict-equality gate.
4. **FP32 frontier at N≳256 δ*-scale** — mandatory equilibration + per-lane FP64 promotion + per-family κ estimate in certification.
5. **Fallback Amdahl** (>3–5% halves speedup) — consensus re-seed, async CPU/GPU overlap, fallback-fraction reporting (also a physics map: failures cluster at branch interactions).
6. **Library/Windows reality** (getrfBatched at n≈1000 unproven on WDDM; sysmem fallback) — M0 spike de-risks before any engine code.

## Verification

- **Unit:** affine cache exactness (held-out 1e-12, all families); RQI/bordered vs `scipy.linalg.eig`
  single-point (≤1e-9); equilibration idempotence; determinism check at fixed tile.
- **Certification (the acceptance gate):** 37-case harness re-run with the GPU backend through
  unchanged `verification/` judges — verdict classes identical, metric drift thresholds above.
- **Benchmarks:** the M6 matrix, with pinned driver settings recorded in result metadata.
- CI on GPU-less runners: `affine.py` tests run pure-numpy; GPU tests behind skip markers.

## Critical files

- `pymack/solver.py` (3D assembly 611–783, BCs 786–869, continuation 1121–1288, spatial local iterations 1536–1764)
- `pymack/equations.py` (spatial QEP assembly — probe target), `pymack/temporal_solver.py` (Özgen 2D form)
- `pymack/spectral.py`, `pymack/scales.py:236-267` (R-independent base-flow sampling — affinity linchpin)
- `verification/mixed_mode/ozgen_fig3/_refdigitize/discrete_mode.py` (candidate-set semantics to reproduce)
- `verification/compute_mack_fig10_4.py` (flagship 3D workload + mode-collapse documentation)
- NEW: `pymack/sweep.py`, `pymack/gpu/**` as laid out above
