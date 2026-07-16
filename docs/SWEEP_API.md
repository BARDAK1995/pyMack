# `pymack.sweep` — batch parameter sweeps

`pymack.sweep` is the public facade for sweep-shaped workloads: growth-rate
maps, neutral-curve grids, and certification runs over a grid of
`(alpha, Re)` or `(omega, Re)` points. Instead of looping over the per-point
solvers by hand, call `temporal_sweep` / `spatial_sweep` once per grid and get
back a structured result carrying values, convergence masks, certified FP64
residuals, freestream-decay diagnostics, provenance codes, and reproducibility
metadata.

This public snapshot is CPU-only. Every sweep in this document uses the
validated process-pool implementation. `backend='auto'` and `backend='cpu'`
select it; `backend='gpu'` is retained only as a compatibility request that
raises a clean `NotImplementedError` explaining the public scope.

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

Runnable CPU end-to-end example:
[`examples/04_parameter_sweep.py`](../examples/04_parameter_sweep.py).

## Why the `if __name__ == '__main__':` guard

The CPU backend dispatches through `concurrent.futures.ProcessPoolExecutor`
with an explicit `spawn` context (Windows-safe, and identical behavior across
Windows/macOS/Linux). Any script that calls a sweep at import time — i.e. not
inside `if __name__ == '__main__':` — will re-execute the whole module in
every spawned worker on Windows/macOS. Pass `cpu_workers=1` to run serially
in-process instead (identical numerics, no pool, no guard needed).

## API reference

### `temporal_sweep(profile, alphas, Res, *, Ma=None, N=128, y_max=None, L=None, wall_bc='isothermal', length_scale='L_star', Pr=0.72, gamma=1.4, lambda_mu_ratio=..., beta=0.0, operator='mack_2d', families=(CBand(0.8, 1.05, label='Mack'),), seeds='auto', backend='auto', precision='mixed', tile_size='auto', return_eigenvectors=False, cpu_workers=None, cpu_blas_threads=None, cpu_eigenvalues_only=False) -> TemporalSweepResult`

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

`cpu_eigenvalues_only=True` is an opt-in CPU optimization for the two 2-D
operators. QZ computes eigenvalues only, mode selection is unchanged, and one
deterministic inverse solve reconstructs the selected vector for the existing
residual and edge-ratio diagnostics. It is rejected for the 3-D leakage-filter
path and non-CPU backends. The default is `False` and does not alter the
historical solver call or its bit pattern.

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

Serialization: `result.save(path)` dispatches by the `.csv` or `.npz` suffix;
an unsupported suffix raises `ValueError`. `result.to_csv(path)` writes one
row per family/grid-point, floats with `repr` for exact round-trip, and meta in
a JSON comment header. `result.to_npz(path)` writes all grids plus meta in one
`.npz`. NPZ also preserves nested NumPy arrays in optional metadata; CSV
records explicit shape/dtype omission markers instead of failing or pretending
those arrays were serialized.
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

Positive integers are **reserved** for future backend-specific seed-chain /
provenance ids. The CPU backend never emits a positive code today. Every
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

Notes specific to the CPU backend:

- `backend` is always `'cpu'`; `backend_requested` records what the caller
  asked for (`'auto'` or `'cpu'`) and
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

### Bitwise identity to the deployed per-point loop

The CPU backend is **bitwise-identical** to the deployed per-point loop it
replaces: same solver calls, same selection filters, same tie-breaks, right
down to the bit pattern of the returned eigenvalues. This is proven, not
asserted — see `validation/test_sweep_cpu_backend.py`
(`test_sweep_equals_point_loop_production_height`,
`test_sweep_equals_point_loop_second_height`,
`test_driver_sweep_engine_equals_point_engine`), which run the sweep facade
and a fresh direct per-point loop side by side and assert bit-equality of
every converged value, every provenance code, and every family label.

## Backend resolution

`backend='cpu'` always uses the `ProcessPoolExecutor` CPU path described
above. `backend='gpu'` always raises `NotImplementedError` in this CPU-only
public snapshot.

`backend='auto'` resolves in this order:

1. An explicit `backend=` argument other than `'auto'` always wins outright
   (this rule applies before `'auto'` resolution is even considered).
2. If `'auto'` was passed: the `PYMACK_SWEEP_BACKEND` environment variable
   (`'cpu'` or `'gpu'`), if set, wins. The latter value reaches the same clean
   unsupported-backend error.
3. Otherwise: CPU, always.

## Parallelism

`cpu_workers` sizes the worker pool (default: `PYMACK_SWEEP_CPU_WORKERS` env
var, else `os.cpu_count()`, capped at 61 on Windows — `ProcessPoolExecutor`
raises `ValueError` above that on Windows' `WaitForMultipleObjects` limit).
`cpu_workers=1` runs serially in the calling process: identical numerics, no
pool overhead, and no `if __name__ == '__main__':` guard required.

The same limit applies to completeness-count R4 CPU-QZ escalations. On Windows,
that escalation pool is additionally bounded by live system commit headroom
queried with `GlobalMemoryStatusEx` (a 1 GiB system reserve and 512 MiB per
spawned SciPy worker). If a worker or its pool fails, only the affected points
are rerun serially in the caller; pool failures are never converted into failed
point-family records.


## Fast CPU sweeps

On many-core hosts, one multithreaded BLAS instance inside every worker causes
severe oversubscription. The supported opt-in is `cpu_blas_threads=1` (or
`PYMACK_SWEEP_CPU_BLAS_THREADS=1`); it pins the standard OMP/OpenBLAS/MKL/
NumExpr variables inherited by spawned workers. Parent mutation is
exception-safe and restored. `cpu_workers` is capped at 61 on Windows because
`ProcessPoolExecutor` uses `WaitForMultipleObjects`; more workers are not a
valid Windows configuration.

The two deployed recipes are:

```powershell
python scripts/make_ozgen_fig3_overlay.py --quality production --panels 2 --engine sweep --workers 24 --blas-threads 1 --eigenvalues-only --verify-against-committed
python verification/compute_mack_fig10_4.py --mach 10 --point-parallel --workers 61 --blas-threads 1 --verify-against-committed
```

The second command's point scheduler parallelizes the 1,224 coarse solves and
then the exact 81-point refine windows. Its default invocation remains the
historical serial loop. The Ozgen driver likewise forwards tuning arguments
only when the user supplies the new flags.

### Measured anchors and scope

All values below are fresh-process, lock-isolated measurements on the same
64-logical-core Windows host. They are workload-specific observations, not
portable promises:

| Workload and path | Workers | Wall (s) | Identity authority | Artifact |
|---|---:|---:|---|---|
| Ozgen M=2, N=128, serial historical loop | 1 | 1598.6827 | committed production grid | `verification/mixed_mode/ozgen_fig3/_compute/ozgen_M2.json` |
| Ozgen M=2, full QZ floor | 24 | 108.9757 | 720/720 committed rows | `docs/benchmarks/cpu_floor_sweep.json` |
| Ozgen M=2, values only + one inverse solve | 24 | 46.7037 | 720/720 committed rows | `docs/benchmarks/cpu_floor_sweep.json` |
| Mack Fig. 10.4 M=10, historical serial | 1 | 9717.0784 | committed nine-station curve | `docs/benchmarks/cpu_fig10_4_m10_serial.json` |
| Mack Fig. 10.4 M=10, point parallel | 61 | 1110.5274 | identical nine-station verdict | `docs/benchmarks/cpu_fig10_4_m10_pointparallel_61w.json` |

The completed worker sweep found that more workers are not monotonically
faster: full-QZ wall was 192.2689, 109.7500,
108.9757, 123.1585, 137.5265, and 153.3321 s at 8, 16, 24, 32, 48, and 61
workers. Its best full-QZ point was 24 workers at 108.9757 s; the matching
values-only-plus-inverse point was 46.7037 s with 720/720 rows matched. That
evidence is committed in `docs/benchmarks/cpu_floor_sweep.json`.

This flattening and reversal is a memory-bandwidth/process-overhead ceiling,
not a correctness failure. Benchmark the target machine; do not assume 61 is
optimal. The earlier 8.75--10.7x user wins compare tuned deployed drivers with
their honest serial historical baselines. They do not imply linear scaling or
a universal speedup.

**Reproducibility boundary:** BLAS pinning and eigenvalues-only QZ are explicit
different floating-point paths, so no bitwise claim is made for either. The
full 720-row production verdict is nevertheless identical at the committed
artifact's `1e-9` precision authority, and the Mack curve is exact-zero. With
both options unset, default-invocation fixtures and per-point equivalence tests
remain byte-identical.

Meta records `cpu_blas_threads`, the effective value observed inside a worker,
and (only when opted in) the `cpu_eigenvalues_only` algorithm description. Use
the default path for exact historical reproduction and the opt-in recipes for
measured throughput.

## Validation

The CPU backend's tests live in `validation/test_sweep_cpu_backend.py`. They
cover bitwise equivalence to the direct
per-point loop at two domain heights and two operators, the committed-CSV
drift attribution, the cross-family tie-break, environment/argument backend
precedence, the clean CPU-only error for `backend='gpu'`, per-point and
post-solve failure containment, `meta` completeness, CSV/NPZ round-trips, and
the Windows worker cap.
