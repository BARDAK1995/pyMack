# pyMack-GPU — Software Integration & Architecture

*Produced by the design fan-out (2026-07-02); all numerical claims were measured on the real pyMack matrices (read-only). Companion to PLAN.md.*

## Summary

Designed pymack/gpu/ as an optional, import-guarded subpackage that adds a new batch/sweep API rather than a backend flag on existing functions, leaving the validated CPU path byte-identical (the only CPU-side change is a behavior-preserving extraction of the two inline 2D assembly blocks into probe-able functions, gated by exact-equality regression tests). The cornerstone AffineOperatorCache extracts constant coefficient matrices K_m by probing the EXISTING assembly functions (pymack.solver._assemble_spatial_qep, assemble_temporal_compressible_3d_evp + BC appliers, and the newly extracted 2D assemblers) at a few well-conditioned parameter tuples and solving one dense Vandermonde least-squares across all matrix entries, with held-out self-verification at ~1e-12 and automatic fallback to per-point CPU assembly + pinned transfer. On top of it sit a VRAM-adaptive tile scheduler (deterministic at fixed recorded tile size), a batched two-sided-RQI temporal engine and a bordered-Newton (implicit-determinant) spatial engine on T(a) directly (4n+1 solves, no 8n companion), wavefront continuation seeded by a few CPU QZ solves, and complex64 LU + complex128 residual/refinement. Certification re-runs the existing verdict.json verification harness with PYMACK_SWEEP_BACKEND=gpu and diffs verdict classes and metrics; CI on GPU-less runners still exercises the affine cache (pure numpy) and all scheduling logic via a mock engine. Kernel access goes through a BatchedLinalgOps protocol so CuPy RawKernels can later be swapped for a compiled C++/CUDA core, and the SweepEngine protocol reserves slots for the batched-shooting engine #2 and a JAX fork. Drivers adopt the new API with ~10-line diffs (before/after shown for the Ozgen c_i grid and the Mazhong trace loop).

## Key decisions

- Put the GPU code inside the package as pymack/gpu/ with a pymack[gpu] extra and hard import guards (only pymack/gpu/affine.py is importable without CuPy), not a separate pymack_gpu distribution
- Expose a NEW batch API (pymack.sweep.temporal_sweep / spatial_sweep with structured SweepResult + CBand mode families) instead of a backend= flag on existing single-point functions; pymack.sweep also ships a CPU ProcessPool backend so drivers adopt it unconditionally and backend='auto' upgrades to GPU
- Extract operator coefficients by PROBING the existing CPU assemblers (pymack.solver._assemble_spatial_qep, assemble_temporal_compressible_3d_evp + BC appliers, and two newly extracted 2D assembly functions) at ~8-14 Chebyshev-placed parameter tuples, solving one dense Vandermonde lstsq across all matrix entries, with held-out verification at rtol 1e-12 and automatic fallback to per-point CPU assembly + pinned transfer
- Permit exactly ONE CPU-side change: a pure code-motion extraction of the inline 2D temporal assemblies (solver.py:414, temporal_solver.py:37) into _assemble_*_evp functions, certified by bitwise-equality regression fixtures recorded before the refactor
- Spatial engine solves T(a)phi=0 directly via bordered-system (Spence-Poulton implicit-determinant) Newton at size 4n with analytic T'(a)=C1+2aC2 from the affine cache â€” no 8n companion linearization on the GPU; temporal engine is batched two-sided RQI with cuBLAS getrfBatched/getrsBatched
- Mixed precision as a first-class scheme: complex64 equilibrated LU + complex128 affine-form residuals and eigenvalue updates, with per-point automatic escalation to full fp64 when fp32 factorization degrades
- Determinism contract: bitwise-stable results at fixed (GPU, driver, CuPy version, precision, tile_size); tile_size resolved once per sweep and recorded in result metadata / CSV headers so resumable-grid workflows stay consistent
- Certification gate = re-run the existing 37-case verdict.json harness with PYMACK_SWEEP_BACKEND=gpu through the UNCHANGED compare_* judges; verdict classes must match and metric drift thresholds (1e-5 relative on eigenvalue anchors, 0.1% on branch locations) must hold
- Future-proofing via two frozen protocols: BatchedLinalgOps (CuPy ops now, compiled C++/CUDA extension later via entry point) and SweepEngine (dense engines now, batched shooting engine #2 later); affine.py stays pure numpy to keep the JAX/differentiable fork open

## Risks (ranked)

- FP32 LU accuracy/breakdown at high N (Chebyshev D2 conditioning grows ~N^4; N=256 tiles may exceed complex64 headroom) â€” mitigated by pre-LU equilibration, fp64 iterative refinement with residual certification, and automatic per-point escalation to full-fp64 solves; certify at the N values the 37 cases actually use (31-130)
- Wavefront mode-swapping near branch interactions (first/second-mode synchronization regions, e.g. Mazhong c-band edges) could silently track the wrong family â€” mitigated by per-point c-band + edge-decay + certified-residual acceptance, neighbor-consensus re-seeding, async CPU QZ fallback, and explicit converged=False reason codes instead of silent values; certification diffs the resulting neutral curves against CPU verdicts
- The affine premise can break for drivers that vary structural parameters per point (y_max, N, wall_bc, or non-self-similar per-station profiles as in the Mach-6 eN pipeline where each station has its own profile) â€” the cache key detects this and the design degrades gracefully to per-point CPU assembly + batched device solve, but the headline speedup then depends on assembly-transfer overlap; measure this path in benchmarks before promising eN sub-second numbers
- Determinism promise is conditional on cuBLAS behavior (kernel selection can vary with batch size and library version) â€” pinned tile_size + recorded environment metadata + a determinism_check() in certification make the contract testable, but cross-machine bitwise stability is explicitly NOT promised; the resumable-CSV pattern must rely on the fixed-tile guarantee only
- Windows-first CuPy development friction (pinned-memory pools, RawKernel compilation via NVRTC on Win10, OneDrive-synced repo slowing compilation caches) â€” keep kernels few and simple in P2, cache compiled kernels under %LOCALAPPDATA%, and treat the Linux/A100 port as a certification re-run rather than new code
- Scope creep into the validated CPU path during the 2D assembly extraction â€” bounded by bitwise regression fixtures captured pre-refactor and by making P1 a standalone, revertible PR
- Seeding cost could erode small-sweep speedups (CPU QZ at 0.1-8 s per seed point) â€” amortize by overlapping seeding with cache probing, keep seed counts at ~4-12 per family, and reuse previous-sweep results as seeds in iterative workflows (neutral-curve refinement)

---

# pyMack GPU Engine â€” Software Integration Design

Target repo: `C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST` (package `pymack`, hatchling build, numpy/scipy only today).

Design invariant: **the validated CPU path is never re-implemented and never behaviorally changed.** The GPU engine consumes the CPU assembly functions as oracles (via probing) and the CPU QZ solvers as seeders/fallbacks. The 37-case `verdict.json` harness is the acceptance gate.

---

## 1. Package layout

Keep it **inside** the `pymack` package (one distribution, one import root, one version), as `pymack/gpu/`. A separate `pymack_gpu` distro would fork versioning and break the "verification drivers import pymack only" property for the certification re-runs.

```
pymack/
  sweep.py                 # NEW top-level facade: backend-dispatching batch API (works WITHOUT cupy)
  gpu/
    __init__.py            # import guard; is_available(), require(), GpuNotAvailable
    backend.py             # device/dtype policy, stream + pinned-pool management, env knobs
    affine.py              # AffineOperatorCache + ParameterBasis + probe/verify (PURE NUMPY)
    assemblers.py          # adapters binding affine.py to the existing CPU assembly functions
    batch.py               # TileScheduler: VRAM-adaptive tiling, double-buffered streams, determinism
    temporal.py            # batched two-sided RQI engine on (A(Î±), B(Î±)) â€” eigenvalue c
    spatial.py             # batched bordered-Newton engine on T(a)=C0+aC1+aÂ²C2 â€” eigenvalue Î±
    wavefront.py           # grid continuation: seeding, frontier propagation, consensus re-seed, CPU QZ fallback queue
    refine.py              # complex64â†’complex128 iterative refinement + residual certification
    diagnostics.py         # batched mode filters: edge-decay ratio, QEP/GEVP residual, c-band
    api.py                 # solve_temporal_sweep / solve_spatial_sweep / SweepResult dataclasses
    kernels/
      __init__.py          # registry: get_ops(precision) -> BatchedLinalgOps
      cupy_ops.py          # cuBLAS getrfBatched/getrsBatched via cupy, einsum-as-GEMM contraction
      raw/                 # .cu sources for RawKernels (bordered solve fuse, residual, pivoted scaling)
```

### Optional-dependency handling
- `pyproject.toml` extras: `gpu = ["cupy-cuda12x>=13"]` (document `cupy-cuda11x` alternative in README; do not pin both). `dev` gains nothing GPU-specific.
- `pymack/gpu/__init__.py` does `try: import cupy except ImportError: cupy=None` and defines:
  ```python
  def is_available() -> bool     # cupy importable AND cupy.cuda.runtime.getDeviceCount() > 0
  def require() -> None          # raise GpuNotAvailable with install hint
  ```
  No module under `pymack/gpu/` except `affine.py` may be imported without cupy â€” **`affine.py` is pure numpy by design** (it is also the CPU fallback assembler and must be unit-testable on GPU-less CI).
- `pymack/__init__.py` gains only `from . import sweep` (numpy-safe); it never imports `pymack.gpu` at import time. `pymack.sweep` lazily imports `pymack.gpu.api` when `backend in ('gpu','auto')` and a device exists.

### The one permitted CPU-side change
The spatial QEP assembly is already probe-able (`pymack.solver._assemble_spatial_qep`, line 116, returns BC-applied `C0,C1,C2,y`), and the 3D temporal assembly is public (`assemble_temporal_compressible_3d_evp` line 611 + `apply_wall_bc_3d` / `apply_dirichlet_freestream_bc_3d`). The two **2D temporal** solvers assemble inline and immediately call `linalg.eig`:
- `pymack/solver.py:414 solve_temporal_compressible` (Mack enthalpy form)
- `pymack/temporal_solver.py:37 solve_temporal_2d` (Ozgen form)

Extract each body into `_assemble_temporal_2d_evp(bf, y, D1, D2, alpha, Re, Ma, Pr, gamma, wall_bc, lambda_mu_ratio) -> (A, B)` (respectively `_assemble_temporal_ozgen_2d_evp`) called by the unchanged public function. This is a **pure code motion** â€” no arithmetic reordering â€” certified by a regression test that asserts `np.array_equal` (bitwise) of `(A, B)` before/after the refactor at several parameter points (capture the pre-refactor matrices once into `validation/data/`). If even this is unacceptable, `affine.py` ships a fallback `capture_eig_hook()` context manager that temporarily swaps `scipy.linalg.eig` inside the target module to record `(A,B)` and raise a sentinel; the extraction is strongly preferred (the hook is fragile and documented as such).

---

## 2. AffineOperatorCache (`pymack/gpu/affine.py`) â€” the no-physics-duplication cornerstone

### Math
For a fixed *structural key* (profile, N, y_max, L, wall_bc, length_scale, Ma, Pr, gamma, lambda_mu_ratio), each assembled matrix is exactly a polynomial in the sweep scalars. Verified from the source:
- Spatial (`_assemble_spatial_qep`): `C0(Ï‰, r) = Kâ‚ + Ï‰Â·K_Ï‰ + rÂ·K_r` with `r = 1/Re`; `C1(r) = Kâ‚' + rÂ·K_r'`; `C2(r) = rÂ·K''` (BC rows are parameter-constant â†’ absorbed into the constant term).
- Temporal 2D/3D: `A(Î±, r) âˆˆ span{1, Î±, Î±Â², r, Î±r, Î±Â²r}`, `B(Î±) = Î±Â·BÌƒ` (BC rows of A constant, of B zero). For 3D, Î² joins the basis: `{1, Î±, Î², Î±Â², Î±Î², Î²Â², r, Î±r, Î²r, Î±Â²r, Î±Î²r, Î²Â²r}` (â‰¤ 12 terms).

The design does **not** hardcode these lists: extraction is generic over a declared monomial basis; terms that are absent come out as (numerically) zero K_m and are pruned by a Frobenius threshold, and terms that are missing are caught by self-verification.

### API
```python
@dataclass(frozen=True)
class ParameterBasis:
    params: tuple[str, ...]                 # e.g. ('alpha', 'invRe')
    monomials: tuple[dict[str, int], ...]   # e.g. ({}, {'alpha':1}, {'alpha':2}, {'invRe':1}, ...)
    def design_matrix(self, points: np.ndarray) -> np.ndarray   # (M, B) Vandermonde, column-scaled

class AffineOperatorCache:
    @classmethod
    def from_probe(cls, assemble, basis, probe_box, *, n_verify=2, rtol=1e-12,
                   oversample=1, seed=0) -> "AffineOperatorCache":
        """assemble: Callable[..., tuple[np.ndarray, ...]] taking ONLY the basis params
        as kwargs (everything structural pre-bound via functools.partial).
        probe_box: {param: (lo, hi)} â€” the sweep's actual parameter ranges."""
    terms: dict[str, np.ndarray]      # matrix-name -> (B_kept, nn, nn) complex128 stacked K_m
    kept:  dict[str, list[int]]       # which monomials survived pruning, per matrix
    def evaluate(self, **scalars) -> tuple[np.ndarray, ...]        # CPU reference contraction
    def scalars(self, points: dict[str, np.ndarray]) -> np.ndarray # (batch, B) monomial values
    def verify_at(self, **scalars) -> float                        # rel. Fro residual vs direct assembly
    def fingerprint(self) -> str                                   # sha256 of key + K tensors (for caching/provenance)
```

### Probe-point selection and the one-shot Vandermonde solve
- M = len(monomials) Ã— oversample + n_verify probe tuples. Points are **tensor-free scattered**: for each parameter, take Chebyshev points of the needed 1-D degree mapped into `[lo, hi]` of the *actual sweep box* (conditioning ~1), then combine by a small scrambled low-discrepancy pick (seeded, deterministic). All probe values are real floats fed straight to the CPU assembler.
- Stack results: `Y[name] : (M, nn*nn)` (row = flattened matrix at probe m). Build `V : (M, B)` with column scaling `V[:, j] /= max|V[:, j]|`. Then **one** `np.linalg.lstsq(V, Y)` per matrix name recovers all `nnÂ²` entry-wise polynomial coefficient vectors simultaneously (this is the "tiny per-entry Vandermonde done as one dense solve" â€” cost: M â‰ˆ 8â€“14 CPU assemblies at 2â€“5 ms each + one (MÃ—B)\(MÃ—nnÂ²) lstsq, total < 100 ms).
- Prune: drop monomial m if `||K_m||_F < 1e-10 Â· max_m ||K_m||_F` after unscaling.

### Self-verification and automatic fallback
1. **Held-out check** (mandatory, at construction): assemble at the `n_verify` reserved random tuples (inside the box, not probe points); require `â€–Î£ s_m K_m âˆ’ directâ€–_F / â€–directâ€–_F < rtol (1e-12)` for every matrix. Typical pass â‰ˆ 1e-14â€“1e-15 in complex128.
2. **Runtime spot check** (cheap insurance, on by default): each sweep re-verifies one random grid tuple before launching.
3. On failure â†’ raise `AffineExtractionError`; `TileScheduler` catches it and switches that sweep to `PerPointAssemblySource`: a CPU thread pool calls the *same* `assemble` partial per grid point, writes into pinned staging buffers, and the batched device solve proceeds unchanged (slower H2D, identical numerics contract). The result metadata records `operator_source: 'affine' | 'per_point_cpu'`.

### `pymack/gpu/assemblers.py` â€” the probe adapters (only file that knows CPU internals)
```python
def spatial_qep_assembler(profile, *, N, y_max, L, wall_bc, length_scale, Ma, Pr, gamma, lambda_mu_ratio):
    part = functools.partial(pymack.solver._assemble_spatial_qep, profile, ..., N=N, ...)
    def assemble(omega, invRe): C0, C1, C2, y = part(omega=omega, Re=1.0/invRe); return {'C0':C0,'C1':C1,'C2':C2}
    return assemble, SPATIAL_BASIS, y_grid
def temporal_2d_assembler(...)      # -> {'A':A, 'B':B}, wraps the newly extracted _assemble_temporal_2d_evp
def temporal_ozgen_2d_assembler(...)# wraps temporal_solver._assemble_temporal_ozgen_2d_evp
def temporal_3d_assembler(...)      # wraps assemble_temporal_compressible_3d_evp + apply_wall_bc_3d + apply_dirichlet_freestream_bc_3d
```
Cache key / memo: `(adapter_name, profile_fingerprint, N, y_max, L, wall_bc, length_scale, Ma, Pr, gamma, lambda_mu_ratio, basis_id)` where `profile_fingerprint` hashes the arrays returned by `sample_baseflow(profile, y, length_scale)` â€” this is the *actual* input to assembly, so any profile mutation invalidates correctly. K-tensor footprint at N=128 (nn=516): 6Ã—516Â²Ã—16 B â‰ˆ 26 MB fp64 â€” keep both fp64 and fp32 device copies.

Note on complex Î±: `solver.py`'s temporal assembler accepts complex Î± natively; probing at real points still recovers the exact polynomial coefficients, so the affine form evaluates at complex Î± for free â€” this is what powers the spatial engine's analytic continuation and `dA/dÎ± = K_Î± + 2Î±K_{Î±Â²} + r(...)` **analytic derivatives**.

---

## 3. Public API and driver adoption

**Decision: new batch API (`pymack.sweep`), not a `backend=` flag on existing functions.** Reasons: (a) the GPU engine returns *tracked modes*, not full spectra â€” bolting it under `solve_temporal_2d` would silently change semantics and endanger the validated path; (b) sweep results need structured provenance (converged masks, residuals, seeds, tile size) that single-point signatures can't carry; (c) `pymack.sweep` gets a pure-CPU implementation (loop + ProcessPool over the existing solvers), so drivers can adopt it unconditionally and CPU-only machines keep working â€” `backend='auto'` picks GPU when present.

```python
# pymack/sweep.py  (numpy-safe facade)
def temporal_sweep(profile, alphas, Res, *, Ma=None, N=128, y_max=None, L=None,
                   wall_bc='isothermal', length_scale='L_star', Pr=0.72, gamma=1.4,
                   lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
                   operator='mack_2d',                  # 'mack_2d' | 'ozgen_2d' | 'mack_3d' (beta=...)
                   families=(CBand(0.8, 1.05, label='Mack'),),  # tracked mode families
                   seeds='auto',                        # 'auto' = QZ at coarse corners; or [(ia, ir, c0), ...]
                   backend='auto', precision='mixed',   # 'mixed' | 'fp64'
                   tile_size='auto', return_eigenvectors=False) -> TemporalSweepResult

def spatial_sweep(profile, omegas, Res, *, ..., families=(CBand(0.8, 0.995),),
                  backend='auto', precision='mixed', ...) -> SpatialSweepResult

@dataclass
class TemporalSweepResult:            # per family f, shape (n_alpha, n_Re)
    c: np.ndarray            # complex128
    omega_i: np.ndarray      # alpha * Im(c)
    converged: np.ndarray    # bool
    residual: np.ndarray     # certified fp64 GEVP residual
    edge_ratio: np.ndarray   # freestream-decay diagnostic
    seed_map: np.ndarray     # int provenance: which seed/neighbor chain produced each point
    meta: dict               # backend, precision, tile_size, operator_source, affine fingerprint,
                             # cupy/driver versions, wall time â€” everything needed for reproducibility
    def to_csv(path), to_npz(path)
```
`CBand` carries the c-window + optional target used for seeding and for the per-point acceptance filter (the batched analog of `classify_most_unstable` / the Mazhong band filter).

### Before/after â€” Ozgen c_i grid (`scripts/make_ozgen_fig3_overlay.py::compute_panel`, lines 146â€“192)
Before (per point, ~2.2 s Ã— 720): the double loop over `re_values Ã— alpha_values` calling `solve_temporal_2d(...)` then `classify_most_unstable(evals)`.
After (~10-line diff):
```python
from pymack.sweep import temporal_sweep, CBand
res = temporal_sweep(profile, alpha_values, re_values, Ma=Ma, N=N, y_max=y_max,
                     length_scale='L_star', operator='ozgen_2d',
                     families=(CBand(0.0, TS_CR_MAX, label='TS'),
                               CBand(MACK_CR_MIN, MACK_CR_MAX, label='Mack')),
                     backend='auto')
stack_ci = np.stack([f.c.imag for f in res.families])         # (2, n_a, n_r)
pick = np.nanargmax(np.where(np.abs(stack_ci) < CI_ABS_MAX, stack_ci, -np.inf), axis=0)
ci_grid  = np.take_along_axis(stack_ci, pick[None], 0)[0]
cr_grid  = np.take_along_axis(np.stack([f.c.real for f in res.families]), pick[None], 0)[0]
family_grid = np.array(['TS', 'Mack'])[pick]
```
The CSV writer and plotting are untouched (same arrays). CPU behavior available via `backend='cpu'` for A/B checks.

### Before/after â€” Mazhong trace (`verification/second_mode/mazhong2003_m4p5/trace_mazhong_curves.py`)
Before: resumable per-node CSV, `growth()` calls `solve_spatial(..., n_modes=25)` + c-band filter per node.
After: one call per mode family, then dump the whole grid (sub-second â†’ resume logic becomes a no-op but the CSV schema is preserved for the downstream contour extractor):
```python
res = spatial_sweep(prof, g['omega'], g['R'], Ma=MA, Pr=PR, gamma=GAMMA, N=N, y_max=Y_MAX,
                    wall_bc=WALL_BC, length_scale='L_star', lambda_mu_ratio=LAMBDA_MU,
                    families=(CBand(g['c_lo'], g['c_hi']),), backend='auto')
for (i, R), (j, om) in itertools.product(enumerate(g['R']), enumerate(g['omega'])):
    val = -res.alpha[j, i].imag if res.converged[j, i] else float('nan')
    w.writerow([mode, f"{R:.1f}", f"{om:.5f}", '' if not np.isfinite(val) else f"{val:.6e}"])
```
Determinism guarantee (Â§4) keeps mixed old/new CSV rows consistent.

Existing high-level workflows (`pymack.analysis.temporal_growth_map`, `spatial_growth_map`) get thin `backend='cpu'` keyword forwarding to `pymack.sweep` in a **later** PR, once certification passes â€” not in the first landing.

---

## 4. Engines, scheduler, memory, determinism

### Temporal engine (`gpu/temporal.py`) â€” batched two-sided RQI on (A(Î±), B(Î±))
Per tile of P grid points, state `(c_k, x_k, y_k)`:
1. Materialize `M_k = A âˆ’ c_k B` for all P points in one GEMM: `S(P, B_terms) @ K_flat(B_terms, nnÂ²)` where the c-shift just adds `âˆ’c_kÂ·s_B` scalars to the contraction row (B is affine too). complex64.
2. `getrfBatched` on `M_k`; `getrsBatched` for right solve `M_k x = B x_{kâˆ’1}` and transposed solve for the left vector.
3. Two-sided Rayleigh update `c_{k+1} = (y*Ax)/(y*Bx)` accumulated in complex128 (fp64 K-tensor apply via the affine form â€” cheap: 6 GEMV-like contractions per point).
4. Converged when fp64 residual `â€–(Aâˆ’cB)xâ€–/â€–xâ€– < tol`; typical 2â€“4 cubic-converging iterations from a neighbor predictor.
Diverged/out-of-band points â†’ flagged, handed to `wavefront.py`.

### Spatial engine (`gpu/spatial.py`) â€” bordered Newton / implicit determinant on T(a), size 4n+1
No 8n companion. Spenceâ€“Poulton bordered system per point: with fixed normalization vectors (b, d) chosen once from the seed eigenvector,
`[[T(a), b], [dá´´, 0]] [x; g] = [0; 1]` defines `g(a)`; `g(a)=0` iff a is an eigenvalue. Newton: `g'(a) = âˆ’dá´´ x' ...` obtained from one extra solve with RHS `âˆ’T'(a)x`, `T'(a) = C1 + 2a C2` **analytic** from the cache. Implementation: LU of T(a) (batched, 4n) + Schur-complement bordering (two `getrs` per iteration) rather than forming the (4n+1) matrix â€” keeps cuBLAS batch shapes uniform. Multi-family: independent state per family; multi-root safety inside a band via 2â€“3 distinct seeds per family at the seed points, deduplicated after convergence.

### Wavefront continuation (`gpu/wavefront.py`)
- **Seeding:** CPU QZ (`solve_spatial` / `solve_temporal_*` â€” the existing functions, unmodified) at a coarse set of grid points ('auto': the 4 corners + coarse subsample every ~8 nodes along the low-R edge), filtered by the family's CBand + edge-decay. Runs in a background thread pool while the affine cache is probed.
- **Propagation:** frontier = unsolved points adjacent (4-neighborhood in index space) to â‰¥1 converged point; predictor = distance-weighted linear extrapolation from converged neighbors (the grid-parallel generalization of `continue_temporal_mode_3d`'s `c_target` chain). Whole frontier levels are batched; anti-diagonal levels give O(n_a+n_R) sequential steps each of width O(min(n_a, n_R)) â€” 720-node Ozgen grid â‰ˆ 60 batched steps.
- **Failure handling:** non-converged or filter-rejected points stay unsolved for later frontier passes (re-seeded from a different neighbor consensus: median of â‰¥3 converged neighbors); after 2 passes, the survivors go to an async CPU QZ fallback queue (ProcessPool, existing solvers) whose results are merged and can re-open the frontier. Guarantees: every point either converges with a certified residual, or is marked `converged=False` with a reason code (`'diverged' | 'filtered' | 'qz_empty'`) â€” never a silent wrong branch.

### TileScheduler (`gpu/batch.py`)
- Working-set estimate per point (temporal, mixed precision): fp32 matrix + LU copy `2Â·(nnÂ²Â·8 B)` + vectors + fp64 residual scratch `nnÂ²Â·16 B` amortized per tile â†’ â‰ˆ 6.5 MB/point at N=128; tile = `clamp(floor(free_vramÂ·0.7 / per_point), 8, frontier_size)`, so 8 GB VRAM â‡’ ~800-point tiles (whole Ozgen frontier in one tile), N=256 (nn=1028) â‡’ ~85-point tiles.
- **Determinism contract:** results are bitwise-stable across runs given (GPU model, driver, CuPy version, precision, **tile_size**). Mechanisms: per-matrix batched cuBLAS routines do not mix data across batch entries; no atomics or split reductions in RawKernels (fixed-order tree reductions); `tile_size='auto'` resolves ONCE per sweep, is recorded in `result.meta` and embedded as a CSV/NPZ header comment so resumed/incremental runs pin the same value (`tile_size=meta['tile_size']`). A `determinism_check()` helper (runs a tile twice, asserts equality) is part of the certification run.
- **Streams/pinning:** two CUDA streams double-buffer: stream A computes tile k while stream B contracts scalarsâ†’matrices for tile k+1 and stages seeds via a pinned-memory pool (`cupy.cuda.alloc_pinned_memory` wrapper in `backend.py`). H2D traffic is scalars + seeds only when the affine source is active (K tensors uploaded once); D2H is eigenvalues/diagnostics (KBs). In `per_point_cpu` fallback mode the pinned staging carries whole matrices and the CPU assembly thread pool is the third pipeline stage.

### Mixed precision (`gpu/refine.py`)
- Factorizations and inner solves in complex64 with pre-LU row/column equilibration (RawKernel; equilibration factors saved for the refinement). Residuals, Rayleigh quotients, and eigenvalue updates in complex128 via fp64 affine apply.
- Refinement = 1â€“3 steps of Newton/RQI in fp64 arithmetic using the fp32 LU as the solver (classic mixed-precision iterative refinement); accept when fp64 relative residual < 1e-10 (temporal GEVP) / 1e-9 (spatial QEP, matching `_filter_with_residual`'s normalization). Points where fp32 LU signals singularity/overflow (large growth factors at Nâ‰³200) automatically re-run in full fp64 within the same tile machinery (`precision='fp64'` per point).

---

## 5. Testing + certification

### Unit tests (`validation/`, pytest markers `gpu`, existing `slow`)
- `test_gpu_affine.py` (**no GPU needed** â€” pure numpy): extraction exactness for all four adapters at Nâˆˆ{31,64,128} vs direct assembly at held-out points (`rtol 1e-12`, expect ~1e-14); basis pruning correctness (C2 has no Re-free term); deliberate non-affine parameter (vary `y_max`) must trip `AffineExtractionError`.
- `test_gpu_assembly_extraction.py` (no GPU): bitwise equality of extracted 2D assemblers vs recorded pre-refactor `(A,B)` fixtures.
- `test_gpu_engines.py` (marker `gpu`, `pytest.importorskip('cupy')` + device check): single-point RQI vs `scipy.linalg.eig` selected mode: |Î”c| < 1e-10 (fp64) / 1e-6 (mixed, post-refinement) on the Malik-1990 anchor cases already used by `validation/test_malik1990_*_anchor.py`; bordered-Newton vs `solve_spatial` shift-invert on the Mach-6 neutral case; determinism (two runs, `array_equal`); tile-size independence of *converged values* to 1e-12 (bitwise only at fixed tile).
- `test_gpu_wavefront.py` (no GPU): frontier ordering, consensus re-seed, and fallback-queue logic against a `MockEngine` returning scripted results.

### Certification harness (`verification/gpu_certification/`)
- `run_certification.py`: for each registered compute driver (the `_compute`/`compute_*`/`trace_*` scripts), re-run with `PYMACK_SWEEP_BACKEND=gpu` into a mirrored output tree, then run the **unchanged** `compare_*` judges and `build_success_matrix.py`, and diff each `verdict.json` against the committed CPU one. This requires the drivers to have adopted `pymack.sweep` first (Phase 3 below) â€” the env var is read by `pymack.sweep` as the `backend='auto'` default override, so certification exercises exactly the code users run.
- Pass criteria: verdict **class equal** for all 37 cases; `metrics` numeric drift: eigenvalue-anchor metrics â‰¤ 1e-5 relative; neutral-curve branch locations (e.g. Mazhong `branch_I_R_pymack`) â‰¤ 0.1 % relative; growth-rate map medians â‰¤ 1e-4 absolute in c_i/âˆ’Î±_i. Emits `CERTIFICATION_MATRIX.md` alongside `SUCCESS_MATRIX.md`.
- `bench_sweeps.py` (in `scripts/`): timed pairs on the three real workloads â€” Ozgen 720-node grid, Mack fig10.4 M10 station, Mach-6 eN 3,649-solve database. **Fair CPU baseline = the existing `ProcessPoolExecutor` path** (`scripts/compute_mach6_growth_nfactor.py` pattern) at full core count, not the serial loop. Report points/s, wall clock, speedup vs serial AND vs multi-core, and precision deltas; writes JSON for the paper's tables.

### CI story (GPU-less runner)
Default `pytest` run executes everything unmarked (affine cache, extraction fixtures, wavefront logic, sweep CPU backend) â€” full logic coverage without a device. `gpu`-marked tests skip cleanly via importorskip + device-count check. Optional later: a self-hosted RTX runner job that runs `-m gpu` + a 3-case mini-certification (Malik case 4, Mazhong, Ozgen M4).

---

## 6. Roadmap hooks

- **C++/CUDA extension core:** all device linear algebra flows through one protocol in `kernels/__init__.py`:
  ```python
  class BatchedLinalgOps(Protocol):
      def contract(self, K, S, out): ...            # scalars -> batched matrices
      def lu_factor(self, A_batch): ...             # in-place, returns pivots
      def lu_solve(self, lu, piv, B_batch, trans): ...
      def residual(self, K64, S64, X, lam): ...     # fp64 affine residual
      def equilibrate(self, A_batch): ...
  def get_ops(precision: str) -> BatchedLinalgOps   # env PYMACK_GPU_OPS='cupy' (default) | 'ext'
  ```
  The future `pymack-gpu-core` compiled wheel implements the same protocol (discovered via entry point `pymack.gpu_ops`); engines, scheduler, wavefront, and API never change. K tensors and scalar tables are already contiguous row-major arrays with documented layouts (`docs/gpu_layouts.md`), so the extension consumes them zero-copy.
- **Engine #2 (batched shooting):** `api.py` dispatches on `engine='dense_rqi' | 'dense_newton' | 'shooting'` through a `SweepEngine` protocol (`prepare(cache_or_profile, grid) -> state`, `solve_tile(state, frontier_batch) -> TileResult`); `wavefront.py` and `TileScheduler` are engine-agnostic (they see eigenvalue predictors and converged flags, never matrices). The shooting engine will bypass `affine.py` and reuse `mack_shooting.py`'s first-order matrices as its own probe targets.
- **JAX/differentiable fork:** `affine.py` is pure numpy and side-effect-free â€” K tensors + monomial bases port to JAX directly; the analytic `dT/dÎ±`, `dA/dÏ‰`, `dA/d(1/R)` from the affine form are exactly the ingredients for implicit-function-theorem gradients of eigenvalues w.r.t. sweep parameters. Keep `AffineOperatorCache` free of cupy imports and keep engines stateless between tiles to preserve this option.

---

## Implementation phases
1. **P1 (CPU-only, mergeable alone):** 2D assembly extraction + bitwise regression fixtures; `affine.py` + `assemblers.py` + tests; `pymack.sweep` with CPU backend (ProcessPool) and the structured result types.
2. **P2 (GPU core):** `backend.py`, `kernels/cupy_ops.py`, `batch.py`, `temporal.py` + `refine.py`; temporal certification on Ozgen M2â€“M6 grids; benchmarks.
3. **P3 (spatial + adoption):** `spatial.py` bordered Newton, wavefront QZ-fallback loop; migrate the ~6 sweep-shaped drivers to `pymack.sweep` (each a â‰¤15-line diff, CPU-verified first); full 37-case certification + `CERTIFICATION_MATRIX.md`.
4. **P4 (paper + roadmap):** benchmark suite for JCP/CPC tables, `docs/gpu_layouts.md`, ops-protocol freeze, mini-certification CI job.

### Critical Files for Implementation
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/solver.py (probe targets `_assemble_spatial_qep`, `assemble_temporal_compressible_3d_evp`, the 2D temporal extraction at line 414, QZ seeders/fallbacks, `continue_temporal_mode_3d` semantics to generalize)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/temporal_solver.py (Ozgen-form 2D assembly to extract; used by the Ozgen verification grids)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/equations.py (`assemble_compressible_matrices` â€” the physics oracle behind the spatial QEP cache)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/pymack/__init__.py (facade wiring for `pymack.sweep`; layering conventions to follow)
- C:/Users/merts/OneDrive/MasaÃ¼stÃ¼/MS_LST/verification/build_success_matrix.py (+ `verification/_compare_lib.py`) â€” the solver-agnostic verdict harness the GPU certification gate builds on
