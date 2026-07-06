# `pymack.sweep` — batch parameter sweeps

`pymack.sweep` is the public facade for sweep-shaped workloads: growth-rate
maps, neutral-curve grids, and certification runs over a grid of
`(alpha, Re)` or `(omega, Re)` points. Instead of looping over the per-point
solvers by hand, call `temporal_sweep` / `spatial_sweep` once per grid and get
back a structured result carrying values, convergence masks, certified FP64
residuals, freestream-decay diagnostics, provenance codes, and reproducibility
metadata.

**GPU is an upgrade, not a requirement.** `import pymack` never imports CuPy,
and `import pymack.sweep` never imports CuPy or `pymack.gpu` (see
`test_import_is_numpy_safe` in `validation/test_sweep_cpu_backend.py`). Every
sweep in this document runs on the CPU backend, which is available
unconditionally. A GPU-native batched temporal engine exists (see
[`docs/gpu/PLAN.md`](gpu/PLAN.md)) and plugs into this same facade through
`backend='gpu'` without changing any of the contracts described here. It is
experimental and not release-certified; **no performance numbers for it are
published yet**. Temporal sweeps dispatch to it lazily when a CUDA device and
the optional CuPy stack are available (with an affine-verification gate and an
automatic CPU fallback); spatial sweeps with `backend='gpu'` still raise
`NotImplementedError` cleanly.

## Quickstart

```python
import pymack as pm
from pymack.scales import delta_star_over_lstar
from pymack.sweep import CBand, temporal_sweep

if __name__ == '__main__':          # required: the CPU backend uses a
    boundary_layer = pm.flat_plate(Ma=6.0)  # spawn-context ProcessPoolExecutor
    y_max = 10.0 * delta_star_over_lstar(boundary_layer)  # match pm.temporal_mode's
                                                           # domain-height default

    result = temporal_sweep(
        boundary_layer,
        alphas=[0.10, 0.14, 0.18],
        Res=[3000.0, 5500.0, 8000.0],
        Ma=6.0, N=100, y_max=y_max, operator='ozgen_2d',
        families=(CBand(0.885, 0.96, label='Mack'),),
        backend='cpu',
    )
    family = result.families[0]
    print(family.omega_i)      # temporal growth rate grid, NaN where not converged
    print(result.meta['backend'], result.meta['wall_time_s'])
    result.to_npz('sweep.npz')  # or .to_csv('sweep.csv')
```

`y_max` is passed explicitly above because `temporal_sweep`'s own
`y_max=None` default (a fixed height tuned for the deployed driver it
reproduces bit-for-bit, see *Determinism* below) is not the
boundary-layer-scaled default `pm.temporal_mode` uses; passing the same
`10 * delta*/L*` height used by the per-point facade keeps the two
comparable. The narrow `CBand(0.885, 0.96)` window matters too: it isolates
the single continuous second-mode branch from a second, less-unstable branch
that also lives in this profile's alpha range — see *Mode families* above and
`examples/04_parameter_sweep.py` for the full discussion and a worked
stability map.

Runnable, GPU-less end-to-end example:
[`examples/04_parameter_sweep.py`](../examples/04_parameter_sweep.py).

## Why the `if __name__ == '__main__':` guard

The CPU backend dispatches through `concurrent.futures.ProcessPoolExecutor`
with an explicit `spawn` context (Windows-safe, and identical behavior across
Windows/macOS/Linux). Any script that calls a sweep at import time — i.e. not
inside `if __name__ == '__main__':` — will re-execute the whole module in
every spawned worker on Windows/macOS. Pass `cpu_workers=1` to run serially
in-process instead (identical numerics, no pool, no guard needed).

## API reference

### `temporal_sweep(profile, alphas, Res, *, Ma=None, N=128, y_max=None, L=None, wall_bc='isothermal', length_scale='L_star', Pr=0.72, gamma=1.4, lambda_mu_ratio=..., beta=0.0, operator='mack_2d', families=(CBand(0.8, 1.05, label='Mack'),), seeds='auto', backend='auto', precision='mixed', tile_size='auto', return_eigenvectors=False, cpu_workers=None, cpu_blas_threads=None) -> TemporalSweepResult`

Solves `A phi = c B phi` at every point of the `alphas x Res` grid with the
selected `operator`, and per `CBand` family returns the most unstable admitted
mode — the batched analog of the deployed per-point drivers.

`operator` selects the underlying per-point solver, called exactly as the
deployed drivers call it:

| `operator` | Per-point solver |
|---|---|
| `'mack_2d'` (default) | `pymack.solver.solve_temporal_compressible` |
| `'ozgen_2d'` | `pymack.temporal_solver.solve_temporal_2d` (no `lambda_mu_ratio`) |
| `'mack_3d'` | `pymack.solver.solve_temporal_compressible_3d` at the given `beta` |

### `spatial_sweep(profile, omegas, Res, *, Ma=None, N=128, y_max=None, L=None, wall_bc='isothermal', length_scale='L_star', Pr=0.72, gamma=1.4, lambda_mu_ratio=..., operator='mack_qep', families=(CBand(0.8, 0.995, label='Mack'),), n_modes=25, alpha_i_abs_max=inf, seeds='auto', backend='auto', precision='mixed', tile_size='auto', return_eigenvectors=False, cpu_workers=None, cpu_blas_threads=None) -> SpatialSweepResult`

Solves the spatial QEP via `pymack.solver.solve_spatial` (shift-invert
companion) once per `(point, family)` at the family's phase-speed target
(`CBand.target`, default the window midpoint), then applies the deployed
Ma & Zhong band filter: `cr_min < omega/Re(alpha) < cr_max`,
`|Im(alpha)| < alpha_i_abs_max`, `Re(alpha) > 0`, most unstable admitted mode
(`argmin Im(alpha)`).

### `CBand(cr_min, cr_max, ci_abs_max=inf, label='', target=None)`

One tracked mode family, named by its phase-speed window. Temporal selection
mirrors the deployed `classify_most_unstable` filters: admit eigenvalues with
`cr_min < Re(c) < cr_max` and `|Im(c)| < ci_abs_max`, return the most unstable
admitted mode (`argmax Im(c)`), or an honest "no discrete mode here" when the
window is empty. `target` (spatial only) is the shift-invert seed phase speed;
pass the deployed driver's guess for exact numerical adoption.

Every family result also carries `mode_index`: the solver-order index of the
selected eigenvalue in the per-point spectrum (`-1` where nothing was
selected). This preserves the deployed classifier's *global* argmax tie-break
— when two families select bitwise-equal growth, the tie is broken by the
smaller `mode_index` (earlier in solver order), not by family order.

### Result objects

`TemporalSweepResult` (`alphas`, `Res`, `families`, `meta`) and
`SpatialSweepResult` (`omegas`, `Res`, `families`, `meta`) hold a tuple of
per-family grids:

- `TemporalFamilyResult`: `band`, `c` (complex phase speed, NaN where not
  converged), `omega_i` (`alpha * Im(c)`, the temporal growth rate),
  `converged` (bool), `residual`, `edge_ratio`, `seed_map`, `mode_index`,
  optional `eigenvectors`.
- `SpatialFamilyResult`: `band`, `alpha` (complex wavenumber), `sigma`
  (`-Im(alpha)`, the spatial growth rate), and the same
  `converged`/`residual`/`edge_ratio`/`seed_map`/`mode_index`/`eigenvectors`
  fields.

`residual` is a certified, independent FP64 relative residual: the operator is
re-assembled with the exact deployed assembly code and the selected
eigenpair is checked against it (never fed back into mode selection). Its
exact normalization is recorded verbatim in `result.meta['residual_definition']`.
`edge_ratio` is the freestream-decay diagnostic (mean amplitude over the
`n_edge=4` freestream-side collocation points, divided by the global peak);
its definition string is `result.meta['edge_ratio_definition']`.

Serialization: `result.to_csv(path)` (one row per family/grid-point, floats
written with `repr` for exact round-trip, meta embedded as a JSON comment
header) and `result.to_npz(path)` (all grids plus meta in one `.npz`).
`pymack.sweep.load_csv(path)` / `pymack.sweep.load_npz(path)` (or the
`TemporalSweepResult.from_npz` / `SpatialSweepResult.from_npz` classmethods)
load them back.

## The `seed_map` / provenance contract

Every grid point, per family, carries an integer provenance code in
`seed_map`. Codes `<= 0` are **universal**, identical across every present and
future backend:

| Code | Constant | Meaning |
|---|---|---|
| `0` | `SEED_QZ_FULL_SPECTRUM` | value selected from a full-spectrum CPU QZ solve |
| `-1` | `SEED_NO_ADMISSIBLE_MODE` | solver succeeded; the family window held no admissible mode (an honest miss, not an error) |
| `-2` | `SEED_SOLVER_FAILED` | per-point record construction raised; the point is contained (never kills the sweep), and the exception `repr` is recorded in `meta['errors']` |

Positive integers are **reserved** for backend-specific seed-chain /
provenance ids — for example a future GPU wavefront-continuation engine's
neighbor-chain ids. The CPU backend never emits a positive code today. Every
backend publishes its complete code legend in `meta['seed_codes']` so a
serialized artifact is self-describing without consulting this document; the
legend dict always carries the `'0'`, `'-1'`, `'-2'`, and `'reserved_positive'`
keys (see `_REQUIRED_META_KEYS` / `test_meta_completeness` in
`validation/test_sweep_cpu_backend.py`).

`errors` in `meta` is a list of `{'i', 'j', 'family', 'error'}` dicts, one per
`-2` point, sorted by `(i, j, family)` — a failed point is diagnosable
straight from the serialized artifact, no rerun required. A per-point failure
(or a post-solve exception while building one family's record) degrades
exactly that `(point, family)` to `-2`; it never aborts the sweep, and a
`BaseException` (e.g. `KeyboardInterrupt`, `MemoryError`) still propagates by
design.

## Determinism and the `meta` contract

`result.meta` is a plain, JSON-serializable dict (it is embedded verbatim as
the CSV comment header and the `.npz` `meta_json` entry). Every backend must
populate:

`api`, `kind`, `schema_version`, `backend`, `backend_requested`,
`env_backend_override`, `operator`, `operator_source`, `precision`,
`precision_requested`, `tile_size`, `tile_size_requested`, `seeds`,
`cpu_workers`, `families`, `solver_kwargs`, `grid`, `return_eigenvectors`,
`seed_codes`, `n_failed_points`, `errors`, `residual_definition`,
`edge_ratio_definition`, `versions` (`pymack`, `numpy`, `scipy`, `python`),
`platform`, `timestamp_utc`, `wall_time_s`.

Notes specific to the CPU backend (the only implemented backend today):

- `backend` is always `'cpu'`; `backend_requested` records what the caller
  asked for (`'auto'`, `'cpu'`, or the now-rejected `'gpu'`) and
  `env_backend_override` records the resolved value of the
  `PYMACK_SWEEP_BACKEND` environment variable, if any (see *Backend
  resolution* below).
- `precision` is always `'fp64'` — the CPU path is always a full-spectrum
  FP64 QZ solve; `precision_requested` echoes the caller's `precision=`
  argument (`'mixed'` or `'fp64'`), which is validated but does not change
  CPU values.
- `tile_size` is always `None` (the CPU backend is tile-free, per-point
  exact); `tile_size_requested` echoes the caller's argument.
- `seeds` is validated (`'auto'` or an explicit sequence) and recorded, but
  does not affect CPU values: every point is an independent, unseeded QZ
  solve.
- `operator_source` is `'per_point_cpu_qz'`, naming the reference computation
  the values are certified against.

**An `AffineOperatorCache` fingerprint (`pymack.gpu.affine.AffineOperatorCache.fingerprint()`,
a sha256 over the structural cache key, basis, and extracted operator terms)
is part of the GPU engine's internal cache identity, not part of the CPU
sweep's `meta` today.** When the GPU sweep engine (`pymack.gpu.api`, plan
slices 07+) lands, its `meta` is expected to publish that fingerprint under
`meta['affine_fingerprint']` alongside the fields above, so a GPU artifact is
traceable to the exact affine decomposition that produced it; this document
will be updated when that field ships. Do not rely on a GPU-only key existing
in CPU meta.

### Bitwise identity to the deployed per-point loop

The CPU backend is **bitwise-identical** to the deployed per-point loop it
replaces: same solver calls, same selection filters, same tie-breaks, right
down to the bit pattern of the returned eigenvalues. This is proven, not
asserted — see `validation/test_sweep_cpu_backend.py`
(`test_sweep_equals_point_loop_production_height`,
`test_sweep_equals_point_loop_second_height`,
`test_driver_sweep_engine_equals_point_engine`), which run the sweep facade
and a fresh direct per-point loop side by side and assert bit-equality of
every converged value, every provenance code, and every family label. The
CPU backend is also the certification reference the (in-development) GPU
engine's results are checked against — see `docs/gpu/PLAN.md`.

**No GPU performance numbers are published in this documentation, the
README, or any example.** The GPU temporal engine is experimental and not
release-certified; its speedup, when certified against the CPU backend above,
will be published separately.

## Backend resolution

`backend='cpu'` always uses the `ProcessPoolExecutor` CPU path described
above. `backend='gpu'` dispatches temporal sweeps to the optional GPU engine
(`pymack.gpu.api.solve_temporal_sweep`) when a usable device is present, and
raises a clean `NotImplementedError` when none is; spatial sweeps on
`backend='gpu'` raise `NotImplementedError` until the spatial engine lands.

`backend='auto'` resolves in this order:

1. An explicit `backend=` argument other than `'auto'` always wins outright
   (this rule applies before `'auto'` resolution is even considered).
2. If `'auto'` was passed: the `PYMACK_SWEEP_BACKEND` environment variable
   (`'cpu'` or `'gpu'`), if set, wins.
3. Otherwise: GPU if a usable GPU sweep engine is importable and reports a
   device (`pymack.gpu.is_available()` true and `pymack.gpu.api` exposes a
   callable `solve_temporal_sweep`) — which is never true today, since that
   engine has not landed — else CPU.

Every failure mode in step 3 (no `pymack.gpu` package, no CuPy, no device, no
engine module) makes `'auto'` resolve to CPU silently; the probe is never
called at import time, so `import pymack.sweep` still never touches CuPy.

## Parallelism

`cpu_workers` sizes the worker pool (default: `PYMACK_SWEEP_CPU_WORKERS` env
var, else `os.cpu_count()`, capped at 61 on Windows — `ProcessPoolExecutor`
raises `ValueError` above that on Windows' `WaitForMultipleObjects` limit).
`cpu_workers=1` runs serially in the calling process: identical numerics, no
pool overhead, and no `if __name__ == '__main__':` guard required.

## Many-core CPU performance (BLAS pinning)

On hosts with many logical cores the default `cpu_workers` (often 61 on
Windows) + unpinned BLAS inside each worker can produce heavy oversubscription
(threads x workers). A baseline-fairness measurement recorded 1372 s unpinned
vs ~149 s with workers pinned to 1 BLAS thread each (9x) on the N=128 720-node
Ozgen overlay workload.

`cpu_blas_threads=1` (or env var `PYMACK_SWEEP_CPU_BLAS_THREADS=1`) is the
supported opt-in. It pins each worker via the standard env vars
(OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS) inherited at spawn time. Parent
mutation is exception-safe and restored.

**Honesty contract**: this is a *different floating-point path*. BLAS thread
count can alter last-bit rounding. Do **not** claim bitwise identity. The
default (unset) path remains bitwise-identical; committed fixtures stay green.

Semantic identity is guaranteed by test: on small grids (e.g. 6x4 nodes,
N=31) the selected c (and None-status) agree to 1e-9 relative.

Meta records:
- `cpu_blas_threads`: the requested value (None = default unpinned)
- `cpu_blas_threads_effective`: the count observed *inside* a worker (via
  threadpoolctl when present, else env fallback)

Use the pinned path for throughput on many-core hardware; use default for
exact bitwise reproducibility.

The measurement script `scripts/gpu_bench/measure_cpu_parallel_overlay_n128.py`
accepts `--blas-threads 1` and writes a suffixed artifact carrying the fields
(workload, workers, effective BLAS threads, wall, cores). See the committed
`cpu_parallel_overlay_n128_blas1t.json` (or run the script).

## Validation

The CPU backend's tests live in `validation/test_sweep_cpu_backend.py` and
run in the default (`-m "not slow"`) CI suite — no GPU, no CuPy, every
platform in the CI matrix. They cover: bitwise equivalence to the direct
per-point loop at two domain heights and two operators, the committed-CSV
drift attribution, the cross-family tie-break, environment/argument backend
precedence, `backend='gpu'` dispatch (or a clean availability error when no
usable GPU engine is present), per-point and
post-solve failure containment, `meta` completeness, CSV/NPZ round-trips, the
Windows worker cap, and that importing `pymack.sweep` never imports `cupy` or
`pymack.gpu`.
