"""Batch sweep facade: grid-parallel temporal/spatial LST sweeps.

This module is the public seam for sweep-shaped workloads (growth-rate maps,
neutral-curve grids, certification runs).  Drivers call
:func:`temporal_sweep` / :func:`spatial_sweep` once per grid instead of
looping over the per-point solvers, and receive a structured
:class:`TemporalSweepResult` / :class:`SpatialSweepResult` carrying values,
convergence masks, certified FP64 residuals, freestream-decay diagnostics,
provenance codes, and reproducibility metadata.

Backends
--------
``backend='cpu'``
    A :class:`~concurrent.futures.ProcessPoolExecutor` (explicit ``spawn``
    context, Windows-safe; as with all :mod:`multiprocessing` use, scripts
    that call a sweep at import time must guard the call with
    ``if __name__ == '__main__':``) over the EXISTING per-point solvers
    (:func:`pymack.solver.solve_temporal_compressible`,
    :func:`pymack.temporal_solver.solve_temporal_2d`,
    :func:`pymack.solver.solve_temporal_compressible_3d`,
    :func:`pymack.solver.solve_spatial`) with the deployed per-point
    selection semantics.  Usable unconditionally; this is the reference
    backend that GPU results are certified against.
``backend='gpu'``
    Lazy dispatch to ``pymack.gpu.api.solve_temporal_sweep`` when the optional
    GPU engine is importable and a CUDA device is visible.  Spatial GPU sweeps
    remain unsupported until the spatial-engine slice lands.  No GPU code is
    imported by this module at import time.
``backend='auto'``
    Resolution order: the ``PYMACK_SWEEP_BACKEND`` environment variable if
    set (values ``cpu``/``gpu``), else GPU when a usable GPU sweep engine is
    importable and reports a device, else CPU.  An explicit ``backend=``
    argument other than ``'auto'`` always wins over the environment.

Import safety
-------------
Importing :mod:`pymack.sweep` never imports ``cupy`` or :mod:`pymack.gpu`.
GPU availability is probed lazily (inside ``backend='auto'`` resolution) and
failure of that probe silently selects the CPU backend.

Mode families
-------------
A :class:`CBand` names one tracked mode family through its phase-speed
window.  Temporal selection is the batched analog of the deployed
``classify_most_unstable`` filters: admit eigenvalues with
``cr_min < Re(c) < cr_max`` and ``|Im(c)| < ci_abs_max``, return the most
unstable admitted mode (argmax ``Im(c)``), or a non-converged point when the
window is empty (an honest "no discrete mode here", not an error).  Spatial
selection mirrors the deployed Ma & Zhong band filter: admit modes with
``cr_min < omega/Re(alpha) < cr_max``, ``|Im(alpha)| < alpha_i_abs_max`` and
``Re(alpha) > 0``, return the most unstable admitted mode (argmin
``Im(alpha)``).

Every family result carries ``mode_index``: the solver-order index of the
selected eigenvalue in the per-point spectrum (``-1`` where no mode was
selected).  This preserves the deployed *global* argmax tie-break -- when
two families' selected modes have bitwise-equal growth, the deployed
``classify_most_unstable`` keeps the one earliest in solver order, so the
cross-family combiner must break ties by the smaller ``mode_index``, not by
family order.  Verdict identity depends on this (slice 13 runs thousands of
solves).

seed_map / provenance legend (per-backend contract)
---------------------------------------------------
``seed_map`` codes ``<= 0`` are UNIVERSAL provenance codes, identical across
every backend: ``0`` = value from a full-spectrum QZ solve
(:data:`SEED_QZ_FULL_SPECTRUM`), ``-1`` = solver succeeded but the family
window held no admissible mode (:data:`SEED_NO_ADMISSIBLE_MODE`, an honest
miss), ``-2`` = the per-point record construction raised and the point was
contained (:data:`SEED_SOLVER_FAILED`; the exception ``repr`` is recorded in
``meta['errors']``).  POSITIVE integers are RESERVED for backend-specific
seed-chain / wavefront-provenance ids (e.g. the future GPU wavefront's
neighbor-chain ids).  Every backend MUST publish its full code legend in
``meta['seed_codes']`` so an artifact is self-describing.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import scipy

from .equations import DEFAULT_LAMBDA_MU_RATIO
from .scales import sample_baseflow
from .spectral import chebyshev_D, physical_derivatives
from .solver import (
    _assemble_spatial_qep,
    _assemble_temporal_2d_evp,
    apply_dirichlet_freestream_bc_3d,
    apply_wall_bc_3d,
    assemble_temporal_compressible_3d_evp,
    solve_spatial,
    solve_temporal_compressible,
    solve_temporal_compressible_3d,
)
from .temporal_solver import _assemble_temporal_ozgen_2d_evp, solve_temporal_2d

__all__ = [
    'CBand',
    'TemporalFamilyResult',
    'SpatialFamilyResult',
    'TemporalSweepResult',
    'SpatialSweepResult',
    'temporal_sweep',
    'spatial_sweep',
    'load_npz',
    'load_csv',
    'SEED_QZ_FULL_SPECTRUM',
    'SEED_NO_ADMISSIBLE_MODE',
    'SEED_SOLVER_FAILED',
]

_BACKEND_ENV_VAR = 'PYMACK_SWEEP_BACKEND'
_CPU_WORKERS_ENV_VAR = 'PYMACK_SWEEP_CPU_WORKERS'
_CPU_BLAS_THREADS_ENV_VAR = 'PYMACK_SWEEP_CPU_BLAS_THREADS'
_BLAS_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')

_TEMPORAL_OPERATORS = ('mack_2d', 'ozgen_2d', 'mack_3d')
_SPATIAL_OPERATORS = ('mack_qep',)

# Seed / provenance codes (per grid point, per family).
# Contract: codes <= 0 are UNIVERSAL across backends; positive ints are
# RESERVED for backend-specific seed-chain / wavefront ids. Every backend
# publishes its full legend in meta['seed_codes'].
SEED_QZ_FULL_SPECTRUM = 0    # value selected from a full-spectrum CPU QZ solve
SEED_NO_ADMISSIBLE_MODE = -1  # solver succeeded; family window empty (honest miss)
SEED_SOLVER_FAILED = -2       # per-point record construction raised (see meta['errors'])

_SEED_CODE_MEANINGS = {
    str(SEED_QZ_FULL_SPECTRUM): 'full-spectrum CPU QZ at this point',
    str(SEED_NO_ADMISSIBLE_MODE): 'no admissible mode in the family window',
    str(SEED_SOLVER_FAILED):
        'per-point record construction raised; exception repr in meta["errors"]',
    'reserved_positive':
        'positive ints are reserved for backend-specific seed-chain ids '
        '(none emitted by the CPU backend)',
}

_RESIDUAL_DEFINITION = (
    'temporal: ||(A - c B) x||_2 / ||x||_2 / ((||A||_F + |c| ||B||_F) / nn); '
    'spatial: ||(C0 + a C1 + a^2 C2) phi||_2 / ||phi||_2 / '
    '((||C0||_F + |a| ||C1||_F + |a|^2 ||C2||_F) / nn)  '
    '(the deployed _filter_with_residual normalization); '
    'operators re-assembled in FP64 by the exact deployed assembly code.'
)
_EDGE_RATIO_DEFINITION = (
    'mean over the top n_edge=4 collocation points (freestream side, index 0) '
    'of max-over-state-blocks |phi|, divided by the global peak of the same '
    'profile amplitude (pymack.api._edge_ratio generalized to any block '
    'count); small values indicate a freestream-decaying discrete mode.'
)


# ---------------------------------------------------------------------------
# Mode families
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CBand:
    """One tracked mode family, named by its phase-speed window.

    Parameters
    ----------
    cr_min, cr_max : float
        Open interval for the phase speed ``c_r`` (temporal: ``Re(c)``;
        spatial: ``omega / Re(alpha)``).  Infinite bounds are allowed for
        temporal families.
    ci_abs_max : float
        Temporal admission cap ``|Im(c)| < ci_abs_max`` (ignored by the
        spatial selection, which uses the sweep-level ``alpha_i_abs_max``).
    label : str
        Family name carried into results and file headers.
    target : complex, optional
        Phase-speed target used for seeding (spatial CPU backend: the
        shift-invert target is ``omega / target``; default is the window
        midpoint).  Pass the deployed driver's guess for exact adoption.
    """

    cr_min: float
    cr_max: float
    ci_abs_max: float = float('inf')
    label: str = ''
    target: complex | None = None

    def as_meta(self) -> dict:
        return {
            'cr_min': float(self.cr_min),
            'cr_max': float(self.cr_max),
            'ci_abs_max': float(self.ci_abs_max),
            'label': str(self.label),
            'target': (None if self.target is None
                       else [float(np.real(self.target)),
                             float(np.imag(self.target))]),
        }

    @classmethod
    def from_meta(cls, d: dict) -> 'CBand':
        target = d.get('target')
        return cls(
            cr_min=float(d['cr_min']),
            cr_max=float(d['cr_max']),
            ci_abs_max=float(d.get('ci_abs_max', float('inf'))),
            label=str(d.get('label', '')),
            target=(None if target is None
                    else complex(target[0], target[1])),
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalFamilyResult:
    """Per-family grids of a temporal sweep; arrays are ``(n_alpha, n_Re)``."""

    band: CBand
    c: np.ndarray            # complex128; NaN+NaNj where not converged
    omega_i: np.ndarray      # alpha * Im(c) (temporal growth rate)
    converged: np.ndarray    # bool
    residual: np.ndarray     # certified FP64 relative residual (NaN if not converged)
    edge_ratio: np.ndarray   # freestream-decay diagnostic (NaN if not converged)
    seed_map: np.ndarray     # int provenance codes (SEED_*)
    mode_index: np.ndarray   # solver-order index of the selected mode (-1 if none)
    eigenvectors: np.ndarray | None = None  # (n_alpha, n_Re, nn) if requested

    @property
    def label(self) -> str:
        return self.band.label


@dataclass
class SpatialFamilyResult:
    """Per-family grids of a spatial sweep; arrays are ``(n_omega, n_Re)``."""

    band: CBand
    alpha: np.ndarray        # complex128; NaN+NaNj where not converged
    sigma: np.ndarray        # -Im(alpha) (spatial growth rate)
    converged: np.ndarray
    residual: np.ndarray
    edge_ratio: np.ndarray
    seed_map: np.ndarray
    mode_index: np.ndarray   # solver-order index of the selected mode (-1 if none)
    eigenvectors: np.ndarray | None = None

    @property
    def label(self) -> str:
        return self.band.label


class _SweepResultBase:
    """Shared CSV/NPZ serialization; subclasses define the axis/value schema."""

    # Overridden by subclasses: (x attribute name, x column name,
    # value attribute, value column stem, growth attribute/column)
    _X_ATTR = 'x'
    _X_COL = 'x'
    _VALUE_ATTR = 'value'
    _VALUE_COL = 'value'
    _GROWTH_ATTR = 'growth'

    def _x_values(self) -> np.ndarray:
        return getattr(self, self._X_ATTR)

    def to_csv(self, path) -> None:
        """Write one row per (family, grid point) with the meta as headers.

        Floats are written with ``repr`` (shortest round-trip), so parsing
        the CSV back recovers the exact doubles.  Eigenvectors are not
        embedded; use :meth:`to_npz` for those.
        """
        xs = self._x_values()
        Res = self.Res
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            handle.write(f'# pymack.sweep {type(self).__name__} v1\n')
            handle.write('# meta = ' + json.dumps(self.meta) + '\n')
            writer = csv.writer(handle)
            writer.writerow([
                'family', 'family_label', self._X_COL, 'Re',
                f'{self._VALUE_COL}_re', f'{self._VALUE_COL}_im',
                self._GROWTH_ATTR, 'converged', 'residual', 'edge_ratio',
                'seed', 'mode_index',
            ])
            for k, fam in enumerate(self.families):
                value = getattr(fam, self._VALUE_ATTR)
                growth = getattr(fam, self._GROWTH_ATTR)
                for i in range(len(xs)):
                    for j in range(len(Res)):
                        writer.writerow([
                            k, fam.band.label,
                            repr(float(xs[i])), repr(float(Res[j])),
                            repr(float(value[i, j].real)),
                            repr(float(value[i, j].imag)),
                            repr(float(growth[i, j])),
                            int(fam.converged[i, j]),
                            repr(float(fam.residual[i, j])),
                            repr(float(fam.edge_ratio[i, j])),
                            int(fam.seed_map[i, j]),
                            int(fam.mode_index[i, j]),
                        ])

    def to_npz(self, path) -> None:
        """Write all grids plus the meta dict (as JSON) to one ``.npz``."""
        payload = {
            'meta_json': np.array(json.dumps(self.meta)),
            self._X_ATTR: self._x_values(),
            'Res': self.Res,
        }
        for k, fam in enumerate(self.families):
            payload[f'family{k}_{self._VALUE_ATTR}'] = getattr(
                fam, self._VALUE_ATTR)
            payload[f'family{k}_{self._GROWTH_ATTR}'] = getattr(
                fam, self._GROWTH_ATTR)
            payload[f'family{k}_converged'] = fam.converged
            payload[f'family{k}_residual'] = fam.residual
            payload[f'family{k}_edge_ratio'] = fam.edge_ratio
            payload[f'family{k}_seed_map'] = fam.seed_map
            payload[f'family{k}_mode_index'] = fam.mode_index
            if fam.eigenvectors is not None:
                payload[f'family{k}_eigenvectors'] = fam.eigenvectors
        np.savez(path, **payload)

    @classmethod
    def from_npz(cls, path) -> '_SweepResultBase':
        result = load_npz(path)
        if not isinstance(result, cls):
            raise TypeError(
                f'{path!r} holds a {type(result).__name__}, not a '
                f'{cls.__name__}')
        return result


@dataclass
class TemporalSweepResult(_SweepResultBase):
    """Structured result of :func:`temporal_sweep`."""

    alphas: np.ndarray
    Res: np.ndarray
    families: tuple[TemporalFamilyResult, ...]
    meta: dict = field(default_factory=dict)

    _X_ATTR = 'alphas'
    _X_COL = 'alpha'
    _VALUE_ATTR = 'c'
    _VALUE_COL = 'c'
    _GROWTH_ATTR = 'omega_i'


@dataclass
class SpatialSweepResult(_SweepResultBase):
    """Structured result of :func:`spatial_sweep`."""

    omegas: np.ndarray
    Res: np.ndarray
    families: tuple[SpatialFamilyResult, ...]
    meta: dict = field(default_factory=dict)

    _X_ATTR = 'omegas'
    _X_COL = 'omega'
    _VALUE_ATTR = 'alpha'
    _VALUE_COL = 'alpha'
    _GROWTH_ATTR = 'sigma'


def load_npz(path):
    """Load a :meth:`to_npz` file back into the matching result class."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta_json']))
        kind = meta.get('kind')
        if kind == 'temporal':
            cls, fam_cls = TemporalSweepResult, TemporalFamilyResult
        elif kind == 'spatial':
            cls, fam_cls = SpatialSweepResult, SpatialFamilyResult
        else:
            raise ValueError(f'unrecognized sweep kind {kind!r} in {path!r}')
        x = np.array(data[cls._X_ATTR])
        Res = np.array(data['Res'])
        families = []
        for k, band_meta in enumerate(meta.get('families', [])):
            vec_key = f'family{k}_eigenvectors'
            families.append(fam_cls(
                band=CBand.from_meta(band_meta),
                **{cls._VALUE_ATTR: np.array(data[f'family{k}_{cls._VALUE_ATTR}']),
                   cls._GROWTH_ATTR: np.array(
                       data[f'family{k}_{cls._GROWTH_ATTR}'])},
                converged=np.array(data[f'family{k}_converged']),
                residual=np.array(data[f'family{k}_residual']),
                edge_ratio=np.array(data[f'family{k}_edge_ratio']),
                seed_map=np.array(data[f'family{k}_seed_map']),
                mode_index=np.array(data[f'family{k}_mode_index']),
                eigenvectors=(np.array(data[vec_key])
                              if vec_key in data.files else None),
            ))
    kwargs = {cls._X_ATTR: x}
    return cls(Res=Res, families=tuple(families), meta=meta, **kwargs)


def load_csv(path):
    """Parse a :meth:`to_csv` file; returns ``(meta, rows)``.

    ``rows`` is a list of dicts with numeric fields converted back to the
    exact doubles that were written (``repr`` round-trip).
    """
    meta = None
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        header = None
        for line in handle:
            if line.startswith('#'):
                stripped = line[1:].strip()
                if stripped.startswith('meta = '):
                    meta = json.loads(stripped[len('meta = '):])
                continue
            for record in csv.reader([line]):
                if header is None:
                    header = record
                    continue
                if not record:
                    continue
                row = dict(zip(header, record))
                row['family'] = int(row['family'])
                row['converged'] = bool(int(row['converged']))
                row['seed'] = int(row['seed'])
                row['mode_index'] = int(row['mode_index'])
                for key in header[2:]:
                    if key in ('converged', 'seed', 'mode_index',
                               'family_label'):
                        continue
                    row[key] = float(row[key])
                rows.append(row)
    if meta is None:
        raise ValueError(f'no "# meta = ..." header found in {path!r}')
    return meta, rows


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _gpu_backend_available() -> bool:
    """True only when a usable GPU sweep engine exists (device + engine API).

    Deliberately conservative: :mod:`pymack.gpu` may exist (or not) without
    an engine; the sweep engines arrive with plan slices 07+ and register by
    exposing ``pymack.gpu.api.solve_temporal_sweep``.  Every failure mode
    (no package, no cupy, no device, no engine) makes ``backend='auto'``
    resolve to CPU.  Never called at import time.
    """
    try:
        import importlib

        gpu = importlib.import_module('pymack.gpu')
        if not bool(gpu.is_available()):
            return False
        api = importlib.import_module('pymack.gpu.api')
    except Exception:
        return False
    return callable(getattr(api, 'solve_temporal_sweep', None))


def _resolve_backend(backend):
    """Return ``(resolved, requested, env_override)``; raise for GPU/invalid.

    Precedence: an explicit ``backend`` argument other than ``'auto'`` wins;
    ``'auto'`` defers to ``PYMACK_SWEEP_BACKEND`` when set, else GPU when a
    usable engine is available, else CPU.
    """
    requested = backend
    env_raw = os.environ.get(_BACKEND_ENV_VAR)
    env_override = env_raw.strip().lower() if env_raw else None
    if backend == 'auto' and env_override:
        backend = env_override
    if backend == 'auto':
        backend = 'gpu' if _gpu_backend_available() else 'cpu'
    if backend == 'cpu':
        return 'cpu', requested, env_override
    if backend == 'gpu':
        if _gpu_backend_available():
            return 'gpu', requested, env_override
        raise NotImplementedError(
            "pymack.sweep: backend='gpu' requested but no usable GPU temporal "
            "sweep engine is available. Use backend='cpu' (ProcessPool over "
            "the validated per-point solvers) or install/enable the optional "
            "GPU stack.")
    raise ValueError(
        f"backend must be 'auto', 'cpu', or 'gpu'; got {requested!r} "
        f'(environment {_BACKEND_ENV_VAR}={env_raw!r})')


def _resolve_cpu_workers(cpu_workers, n_tasks):
    if cpu_workers is None:
        env = os.environ.get(_CPU_WORKERS_ENV_VAR)
        cpu_workers = int(env) if env else (os.cpu_count() or 1)
    cpu_workers = int(cpu_workers)
    if cpu_workers < 1:
        raise ValueError(f'cpu_workers must be >= 1, got {cpu_workers}')
    cpu_workers = min(cpu_workers, n_tasks)
    if sys.platform == 'win32':
        # ProcessPoolExecutor raises ValueError above 61 workers on Windows
        # (WaitForMultipleObjects limit); machines with >61 logical CPUs
        # would otherwise crash on the cpu_workers=None default.
        cpu_workers = min(cpu_workers, 61)
    return max(1, cpu_workers)


def _resolve_cpu_blas_threads(cpu_blas_threads):
    """Return int or None.

    None (or unset) means the historical default unpinned path (bitwise
    fixtures unchanged). Positive int pins worker BLAS threads (opt-in
    different FP path; semantic identity only).
    """
    if cpu_blas_threads is None:
        env = os.environ.get(_CPU_BLAS_THREADS_ENV_VAR)
        if env and env.strip():
            try:
                cpu_blas_threads = int(env)
            except ValueError:
                raise ValueError(
                    f'{_CPU_BLAS_THREADS_ENV_VAR}={env!r} is not an int')
    if cpu_blas_threads is not None:
        cpu_blas_threads = int(cpu_blas_threads)
        if cpu_blas_threads < 1:
            raise ValueError(f'cpu_blas_threads must be >= 1, got {cpu_blas_threads}')
    return cpu_blas_threads


def _get_current_blas_threads():
    """Observe effective BLAS thread count from *inside* the current process.

    Graceful, no hard dependency: try threadpoolctl first (if installed);
    fall back to the env var we set (or any of the standard pinning vars);
    never raises. Used to record the *actual* per-worker count in meta.
    """
    # Preferred: threadpoolctl if available (exact live view of loaded libs)
    try:
        import threadpoolctl
        infos = threadpoolctl.threadpool_info()
        nums = []
        for info in infos:
            api = (info.get('internal_api') or info.get('user_api') or '').lower()
            fpath = (info.get('filepath') or '').lower()
            if ('blas' in api or 'openblas' in api or 'mkl' in api or
                    'blas' in fpath or 'mkl' in fpath or 'openblas' in fpath):
                nt = info.get('num_threads')
                if nt is not None:
                    nums.append(int(nt))
        if nums:
            return min(nums)  # conservative effective for our workers
    except Exception:
        pass
    # Fallback: the env we (or caller) set for pinning
    for var in _BLAS_THREAD_VARS:
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_common(*, families, precision, tile_size, seeds, Ma, kind):
    if Ma is None:
        raise TypeError(f"{kind}_sweep() missing required keyword 'Ma'")
    families = tuple(families)
    if not families:
        raise ValueError('families must contain at least one CBand')
    for band in families:
        if not isinstance(band, CBand):
            raise TypeError(f'families entries must be CBand, got {band!r}')
        if not band.cr_min < band.cr_max:
            raise ValueError(f'CBand requires cr_min < cr_max, got {band}')
        if not band.ci_abs_max > 0.0:
            raise ValueError(f'CBand requires ci_abs_max > 0, got {band}')
    if precision not in ('mixed', 'fp64'):
        raise ValueError(f"precision must be 'mixed' or 'fp64', got {precision!r}")
    if tile_size != 'auto':
        if not (isinstance(tile_size, (int, np.integer)) and tile_size >= 1):
            raise ValueError(
                f"tile_size must be 'auto' or a positive int, got {tile_size!r}")
    if seeds != 'auto' and not isinstance(seeds, (list, tuple)):
        raise ValueError(f"seeds must be 'auto' or a sequence, got {seeds!r}")
    return families


def _as_grid_axis(values, name):
    axis = np.atleast_1d(np.asarray(values, dtype=float))
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f'{name} must be a non-empty 1-D array of floats')
    return axis


# ---------------------------------------------------------------------------
# Per-point selection (the batched analog of the deployed filters)
# ---------------------------------------------------------------------------

def _select_temporal_mode(eigenvalues, band):
    """Index of the most unstable admitted mode, or None (empty window).

    Exact per-family restriction of the deployed classification (e.g.
    ``classify_most_unstable`` in scripts/make_ozgen_fig3_overlay.py):
    strict inequalities, first-maximum tie-break in solver order.
    """
    c = np.asarray(eigenvalues)
    if c.size == 0:
        return None
    admissible = (
        (np.abs(c.imag) < band.ci_abs_max)
        & (c.real > band.cr_min)
        & (c.real < band.cr_max)
    )
    idx = np.flatnonzero(admissible)
    if idx.size == 0:
        return None
    return int(idx[np.argmax(c.imag[idx])])


def _select_spatial_mode(alphas, omega, band, alpha_i_abs_max):
    """Index of the most unstable admitted spatial mode, or None.

    Deployed Ma & Zhong band-filter semantics (trace_mazhong_curves.growth):
    ``c = omega / Re(alpha)`` in the open band, ``|Im(alpha)|`` capped,
    downstream (``Re(alpha) > 0``), pick ``argmin Im(alpha)``.
    """
    a = np.asarray(alphas)
    if a.size == 0:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        c = omega / a.real
    admissible = (
        (c > band.cr_min)
        & (c < band.cr_max)
        & (np.abs(a.imag) < alpha_i_abs_max)
        & (a.real > 0.0)
    )
    idx = np.flatnonzero(admissible)
    if idx.size == 0:
        return None
    return int(idx[np.argmin(a.imag[idx])])


def _edge_ratio(phi, n, n_edge=4):
    """Freestream-to-peak amplitude ratio of a stacked eigenvector.

    Same diagnostic as ``pymack.api._edge_ratio`` generalized to any number
    of state blocks (4 for the 2-D operators and the spatial QEP, 5 for the
    3-D operator).  Collocation index 0 is the freestream boundary.
    """
    phi = np.asarray(phi)
    n_blocks = phi.shape[0] // n
    fields = np.abs(phi[:n_blocks * n].reshape(n_blocks, n))
    profile_amp = fields.max(axis=0)
    peak = float(profile_amp.max())
    if peak == 0.0:
        return float('inf')
    return float(profile_amp[:n_edge].mean() / peak)


# ---------------------------------------------------------------------------
# CPU backend: per-point work (runs inside pool workers; module-level and
# picklable by reference -- required for Windows spawn semantics)
# ---------------------------------------------------------------------------

_WORKER_PAYLOAD = None


def _worker_init(payload):
    """Pool initializer: stash the (heavy) payload once per worker process.

    Also (re)asserts BLAS thread env inside the fresh child (belt-and-suspenders
    with parent-side set before spawn) and ensures numpy is imported under the
    pinned env in this worker.
    """
    global _WORKER_PAYLOAD
    _WORKER_PAYLOAD = payload
    blas = payload.get('cpu_blas_threads')
    if blas is not None:
        n = str(int(blas))
        for var in _BLAS_THREAD_VARS:
            os.environ[var] = n
    # Import numpy under the (possibly pinned) env. Harmless if already loaded
    # by bootstrap; guarantees config sees our vars on this child.
    try:
        import numpy as np  # noqa: F401
        _ = np.__version__
    except Exception:
        pass


def _solve_task(task):
    """Pool task entry point: solve one grid point against the worker payload.

    Returns also the BLAS thread count *observed from inside this worker*.
    """
    i, j, x, Re = task
    records = _point_records(_WORKER_PAYLOAD, x, Re)
    blas_n = _get_current_blas_threads() if (_WORKER_PAYLOAD or {}).get('cpu_blas_threads') is not None else None
    return i, j, records, blas_n


def _point_records(payload, x, Re):
    if payload['kind'] == 'temporal':
        return _temporal_point_records(payload, x, Re)
    return _spatial_point_records(payload, x, Re)


def _failed_record(exc):
    return {'status': SEED_SOLVER_FAILED, 'error': repr(exc)}


def _solve_temporal_point(payload, alpha, Re):
    """Call the EXISTING per-point solver exactly as the deployed drivers do."""
    kw = payload['solver_kwargs']
    op = payload['operator']
    prof = payload['profile']
    if op == 'ozgen_2d':
        return solve_temporal_2d(
            prof, float(alpha), float(Re), kw['Ma'], kw['Pr'], kw['gamma'],
            N=kw['N'], y_max=kw['y_max'], L=kw['L'], wall_bc=kw['wall_bc'],
            length_scale=kw['length_scale'],
        )
    if op == 'mack_2d':
        return solve_temporal_compressible(
            prof, float(alpha), float(Re), kw['Ma'], kw['Pr'], kw['gamma'],
            N=kw['N'], y_max=kw['y_max'], L=kw['L'], wall_bc=kw['wall_bc'],
            length_scale=kw['length_scale'],
            lambda_mu_ratio=kw['lambda_mu_ratio'],
        )
    if op == 'mack_3d':
        return solve_temporal_compressible_3d(
            prof, float(alpha), kw['beta'], float(Re), kw['Ma'], kw['Pr'],
            kw['gamma'], N=kw['N'], y_max=kw['y_max'], L=kw['L'],
            wall_bc=kw['wall_bc'], length_scale=kw['length_scale'],
            lambda_mu_ratio=kw['lambda_mu_ratio'],
        )
    raise ValueError(f'unknown temporal operator {op!r}')


def _assemble_temporal_operator(payload, alpha, Re):
    """Re-assemble (A, B) with the exact deployed assembly code (FP64).

    Deterministic replica of the assembly performed inside the per-point
    solver, used to certify the returned eigenpairs with an independent
    residual.  Never feeds back into mode selection.
    """
    kw = payload['solver_kwargs']
    op = payload['operator']
    prof = payload['profile']
    Ma, N, L = kw['Ma'], kw['N'], kw['L']
    if op in ('mack_2d', 'ozgen_2d'):
        y_max = kw['y_max']
        if y_max is None:
            y_max = 6.0 if Ma > 2.0 else 12.0
        D_eta = chebyshev_D(N)
        y, D1, D2 = physical_derivatives(D_eta, y_max, N, L)
        bf = sample_baseflow(prof, y, kw['length_scale'])
        if op == 'ozgen_2d':
            return _assemble_temporal_ozgen_2d_evp(
                bf, y, D1, D2, alpha, Re, Ma, kw['Pr'], kw['gamma'],
                wall_bc=kw['wall_bc'])
        return _assemble_temporal_2d_evp(
            bf, y, D1, D2, alpha, Re, Ma, kw['Pr'], kw['gamma'],
            wall_bc=kw['wall_bc'], lambda_mu_ratio=kw['lambda_mu_ratio'])
    A, B, _y, D1, n, _a, _b, _bf = assemble_temporal_compressible_3d_evp(
        prof, alpha, kw['beta'], Re, Ma, kw['Pr'], kw['gamma'],
        N=N, y_max=kw['y_max'], L=L, length_scale=kw['length_scale'],
        lambda_mu_ratio=kw['lambda_mu_ratio'])
    apply_wall_bc_3d(A, B, D1, n, wall_bc=kw['wall_bc'])
    apply_dirichlet_freestream_bc_3d(A, B, n)
    return A, B


def _temporal_point_records(payload, alpha, Re):
    families = payload['families']
    # Shared solve: a failure here fails every family at this point (they all
    # read the same spectrum). Exception -> contained -2; BaseException (e.g.
    # KeyboardInterrupt, MemoryError) still propagates by design.
    try:
        evals, evecs, y = _solve_temporal_point(payload, alpha, Re)
    except Exception as exc:
        return [_failed_record(exc) for _ in families]
    n = len(y)

    # Shared FP64 residual operator (best-effort; only assembled when at
    # least one family selects a mode). Contained separately: a residual-
    # operator failure only voids residual certification (NaN), not the point.
    A = B = None
    norm_A = norm_B = nn = None
    try:
        need_operator = any(
            _select_temporal_mode(evals, band) is not None
            for band in families)
    except Exception:
        need_operator = True
    if need_operator:
        try:
            A, B = _assemble_temporal_operator(payload, alpha, Re)
            norm_A = np.linalg.norm(A, 'fro')
            norm_B = np.linalg.norm(B, 'fro')
            nn = A.shape[0]
        except Exception:
            A = B = None

    records = []
    for band in families:
        # Item 3: the WHOLE per-family record construction (selection +
        # value extraction + residual + edge_ratio) is contained, so a
        # post-solve shape/NaN surprise degrades one family, never the sweep.
        try:
            sel = _select_temporal_mode(evals, band)
            if sel is None:
                records.append({'status': SEED_NO_ADMISSIBLE_MODE})
                continue
            c = complex(evals[sel])
            phi = np.asarray(evecs[:, sel])
            record = {
                'status': SEED_QZ_FULL_SPECTRUM,
                'value': c,
                'mode_index': int(sel),
                'edge_ratio': _edge_ratio(phi, n),
            }
            if A is not None:
                resid = (np.linalg.norm(A @ phi - c * (B @ phi))
                         / np.linalg.norm(phi))
                scale = (norm_A + abs(c) * norm_B) / nn
                record['residual'] = float(resid / max(scale, 1e-300))
            else:
                record['residual'] = float('nan')
            if payload['return_eigenvectors']:
                record['vec'] = phi
            records.append(record)
        except Exception as exc:
            records.append(_failed_record(exc))
    return records


def _spatial_point_records(payload, omega, Re):
    kw = payload['solver_kwargs']
    prof = payload['profile']
    families = payload['families']

    C = None
    norms = nn = None
    assembly_failed = False
    records = []
    for band in families:
        # Item 3: the WHOLE per-family record construction (solve + selection
        # + residual + edge_ratio) is contained; one family's failure never
        # kills the point or the sweep. BaseException still propagates.
        try:
            c_target = band.target
            if c_target is None:
                c_target = 0.5 * (band.cr_min + band.cr_max)
            target_alpha = float(omega) / c_target + 0j
            alphas, modes, _y = solve_spatial(
                prof, float(omega), float(Re), kw['Ma'], kw['Pr'],
                kw['gamma'], N=kw['N'], y_max=kw['y_max'], L=kw['L'],
                wall_bc=kw['wall_bc'], target_alpha=target_alpha,
                n_modes=kw['n_modes'], length_scale=kw['length_scale'],
                lambda_mu_ratio=kw['lambda_mu_ratio'],
            )
            sel = _select_spatial_mode(alphas, float(omega), band,
                                       kw['alpha_i_abs_max'])
            if sel is None:
                records.append({'status': SEED_NO_ADMISSIBLE_MODE})
                continue
            a = complex(alphas[sel])
            phi = np.asarray(modes[:, sel])
            if C is None and not assembly_failed:
                try:
                    C0, C1, C2, _y2 = _assemble_spatial_qep(
                        prof, float(omega), float(Re), kw['Ma'], kw['Pr'],
                        kw['gamma'], kw['N'], kw['y_max'], kw['L'],
                        kw['wall_bc'], kw['length_scale'],
                        kw['lambda_mu_ratio'])
                    C = (C0, C1, C2)
                    norms = tuple(np.linalg.norm(M, 'fro') for M in C)
                    nn = C0.shape[0]
                except Exception:
                    assembly_failed = True  # skip residual certification
            record = {
                'status': SEED_QZ_FULL_SPECTRUM,
                'value': a,
                'mode_index': int(sel),
                'edge_ratio': _edge_ratio(phi, phi.shape[0] // 4),
            }
            if C is not None:
                C0, C1, C2 = C
                resid = (np.linalg.norm((C0 + a * C1 + a * a * C2) @ phi)
                         / np.linalg.norm(phi))
                mat_scale = (norms[0] + abs(a) * norms[1]
                             + abs(a) ** 2 * norms[2]) / nn
                record['residual'] = float(resid / max(mat_scale, 1e-300))
            else:
                record['residual'] = float('nan')
            if payload['return_eigenvectors']:
                record['vec'] = phi
            records.append(record)
        except Exception as exc:
            records.append(_failed_record(exc))
    return records


# ---------------------------------------------------------------------------
# CPU backend: grid runner
# ---------------------------------------------------------------------------

def _run_cpu_grid(payload, xs, Res, cpu_workers):
    """Solve every grid point; returns ({(i, j): records}, n_workers, eff_blas).

    ``n_workers == 1`` runs serially in-process (identical numerics, no pool
    overhead); otherwise a spawn-context ProcessPool is used so behavior is
    identical across Windows/macOS/Linux.  Per-point solver failures are
    recorded (SEED_SOLVER_FAILED), never raised, so one bad point cannot
    kill a long sweep.

    When payload['cpu_blas_threads'] is not None, parent mutates the standard
    BLAS thread env vars *before* spawn (children inherit at birth) and restores
    in finally (exception-safe). The observed count is read *from inside* a
    worker via _get_current_blas_threads and returned as eff_blas (recorded in
    result.meta).
    """
    tasks = [
        (i, j, float(xs[i]), float(Res[j]))
        for i in range(len(xs))
        for j in range(len(Res))
    ]
    n_workers = _resolve_cpu_workers(cpu_workers, len(tasks))
    blas_threads = payload.get('cpu_blas_threads')
    out = {}
    if n_workers == 1:
        for i, j, x, Re in tasks:
            out[(i, j)] = _point_records(payload, x, Re)
        eff = _get_current_blas_threads()
        # For serial (in-process) + explicit pin: we report the requested as
        # the documented intent; live _get may reflect pre-existing parent
        # state (see traps). Pin verification uses parallel workers.
        if blas_threads is not None and eff is None:
            eff = blas_threads
        return out, n_workers, eff
    ctx = multiprocessing.get_context('spawn')
    env_pinned = {}
    if blas_threads is not None:
        for var in _BLAS_THREAD_VARS:
            env_pinned[var] = os.environ.get(var)
            os.environ[var] = str(int(blas_threads))
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(payload,),
        ) as pool:
            futures = [pool.submit(_solve_task, task) for task in tasks]
            observed = None
            for future in as_completed(futures):
                i, j, records, blas_n = future.result()
                out[(i, j)] = records
                if observed is None and blas_n is not None:
                    observed = blas_n
            eff = observed if observed is not None else blas_threads
            return out, n_workers, eff
    finally:
        for var, old in env_pinned.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


def _collect_family_grids(results, shape, families, return_vecs):
    """Turn the {(i, j): records} map into per-family dense grids.

    Returns ``(grids, errors)`` where ``errors`` is a compact list of
    ``{'i', 'j', 'family', 'error'}`` dicts (item 2: the worker's exception
    repr for every ``-2`` point, so a failed point is diagnosable straight
    from the serialized artifact).
    """
    grids = []
    errors = []
    for k, band in enumerate(families):
        value = np.full(shape, complex(np.nan, np.nan), dtype=complex)
        converged = np.zeros(shape, dtype=bool)
        residual = np.full(shape, np.nan, dtype=float)
        edge_ratio = np.full(shape, np.nan, dtype=float)
        seed_map = np.full(shape, SEED_NO_ADMISSIBLE_MODE, dtype=np.int32)
        mode_index = np.full(shape, -1, dtype=np.int32)
        vecs = None
        for (i, j), records in results.items():
            record = records[k]
            seed_map[i, j] = record['status']
            if record['status'] == SEED_SOLVER_FAILED:
                errors.append({
                    'i': int(i), 'j': int(j), 'family': int(k),
                    'error': record.get('error', ''),
                })
                continue
            if record['status'] != SEED_QZ_FULL_SPECTRUM and record['status'] <= 0:
                continue
            value[i, j] = record['value']
            converged[i, j] = True
            residual[i, j] = record['residual']
            edge_ratio[i, j] = record['edge_ratio']
            mode_index[i, j] = record['mode_index']
            if return_vecs and 'vec' in record:
                if vecs is None:
                    vecs = np.full(
                        shape + (record['vec'].shape[0],),
                        complex(np.nan, np.nan), dtype=complex)
                vecs[i, j, :] = record['vec']
        grids.append({
            'value': value,
            'converged': converged,
            'residual': residual,
            'edge_ratio': edge_ratio,
            'seed_map': seed_map,
            'mode_index': mode_index,
            'eigenvectors': vecs,
        })
    errors.sort(key=lambda e: (e['i'], e['j'], e['family']))
    return grids, errors


def _build_meta(*, kind, api_name, operator, backend_requested, env_override,
                precision, tile_size, seeds, n_workers, families,
                solver_kwargs, xs, x_name, Res, return_eigenvectors,
                wall_time_s, errors, cpu_blas_threads=None,
                cpu_blas_threads_effective=None):
    from . import __version__ as pymack_version

    solver_meta = {}
    for key, value in solver_kwargs.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        solver_meta[key] = value
    return {
        'api': api_name,
        'kind': kind,
        'schema_version': 1,
        'backend': 'cpu',
        'backend_requested': backend_requested,
        'env_backend_override': env_override,
        'operator': operator,
        'operator_source': 'per_point_cpu_qz',
        'precision': 'fp64',  # effective: the CPU QZ path is always FP64
        'precision_requested': precision,
        'tile_size': None,    # CPU backend is tile-free (per-point exact)
        'tile_size_requested': tile_size if tile_size == 'auto' else int(tile_size),
        'seeds': seeds if seeds == 'auto' else list(map(str, seeds)),
        'cpu_workers': int(n_workers),
        'cpu_blas_threads': cpu_blas_threads,
        'cpu_blas_threads_effective': cpu_blas_threads_effective,
        'families': [band.as_meta() for band in families],
        'solver_kwargs': solver_meta,
        'grid': {
            x_name: {'n': int(len(xs)), 'min': float(xs.min()),
                     'max': float(xs.max())},
            'Re': {'n': int(len(Res)), 'min': float(Res.min()),
                   'max': float(Res.max())},
        },
        'return_eigenvectors': bool(return_eigenvectors),
        'seed_codes': dict(_SEED_CODE_MEANINGS),
        'n_failed_points': len(errors),
        'errors': list(errors),
        'residual_definition': _RESIDUAL_DEFINITION,
        'edge_ratio_definition': _EDGE_RATIO_DEFINITION,
        'versions': {
            'pymack': pymack_version,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'python': platform.python_version(),
        },
        'platform': platform.platform(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'wall_time_s': float(wall_time_s),
    }


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

def temporal_sweep(profile, alphas, Res, *,
                   Ma=None, N=128, y_max=None, L=None,
                   wall_bc='isothermal', length_scale='L_star',
                   Pr=0.72, gamma=1.4,
                   lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
                   beta=0.0,
                   operator='mack_2d',
                   families=(CBand(0.8, 1.05, label='Mack'),),
                   seeds='auto',
                   backend='auto', precision='mixed',
                   tile_size='auto', return_eigenvectors=False,
                   cpu_workers=None, cpu_blas_threads=None) -> TemporalSweepResult:
    """Temporal LST sweep over an ``alphas x Res`` grid.

    Solves ``A phi = c B phi`` at every grid point with the selected
    ``operator`` ('mack_2d': :func:`pymack.solver.solve_temporal_compressible`;
    'ozgen_2d': :func:`pymack.temporal_solver.solve_temporal_2d`;
    'mack_3d': :func:`pymack.solver.solve_temporal_compressible_3d` at the
    given ``beta``) and, per :class:`CBand` family, returns the most unstable
    admitted mode -- the batched equivalent of the deployed per-point loops.

    On the CPU backend (the only implemented backend today) ``seeds``,
    ``precision`` and ``tile_size`` are validated and recorded in
    ``result.meta`` but do not affect values: every point is an independent
    full-spectrum FP64 QZ solve, bitwise-identical to the per-point loop.
    ``cpu_workers`` (or ``PYMACK_SWEEP_CPU_WORKERS``) sizes the process pool;
    1 runs serially in-process. ``cpu_blas_threads`` (or
    ``PYMACK_SWEEP_CPU_BLAS_THREADS``) is the opt-in BLAS pinning for
    many-core throughput (see function docs; default path is bitwise-stable).
    ``cpu_blas_threads`` (or ``PYMACK_SWEEP_CPU_BLAS_THREADS``) is an *opt-in*
    for many-core hosts: when set, each worker pins its BLAS to that thread
    count (via inherited env at spawn). This is a different floating-point
    path (BLAS thread count affects last-bit rounding); default (None) is
    bitwise-identical to historical and keeps all fixtures green. When
    enabled, meta records the requested and the count *observed inside a
    worker*.
    """
    xs = _as_grid_axis(alphas, 'alphas')
    Res = _as_grid_axis(Res, 'Res')
    if operator not in _TEMPORAL_OPERATORS:
        raise ValueError(
            f'operator must be one of {_TEMPORAL_OPERATORS}, got {operator!r}')
    if operator == 'ozgen_2d' and lambda_mu_ratio != DEFAULT_LAMBDA_MU_RATIO:
        raise ValueError(
            "operator='ozgen_2d' (solve_temporal_2d) has no lambda_mu_ratio "
            'parameter; leave it at the default or use a Mack-form operator')
    if operator != 'mack_3d' and beta != 0.0:
        raise ValueError("beta is only meaningful for operator='mack_3d'")
    families = _validate_common(
        families=families, precision=precision, tile_size=tile_size,
        seeds=seeds, Ma=Ma, kind='temporal')
    _resolved, requested, env_override = _resolve_backend(backend)
    if _resolved == 'gpu':
        import importlib

        api = importlib.import_module('pymack.gpu.api')
        return api.solve_temporal_sweep(
            profile, xs, Res, Ma=Ma, N=N, y_max=y_max, L=L,
            wall_bc=wall_bc, length_scale=length_scale, Pr=Pr, gamma=gamma,
            lambda_mu_ratio=lambda_mu_ratio, beta=beta, operator=operator,
            families=families, seeds=seeds, precision=precision,
            tile_size=tile_size, return_eigenvectors=return_eigenvectors,
            cpu_workers=cpu_workers, backend_requested=requested,
            env_backend_override=env_override)

    solver_kwargs = {
        'Ma': float(Ma), 'N': int(N),
        'y_max': None if y_max is None else float(y_max),
        'L': None if L is None else float(L),
        'wall_bc': wall_bc, 'length_scale': length_scale,
        'Pr': float(Pr), 'gamma': float(gamma),
        'lambda_mu_ratio': float(lambda_mu_ratio),
        'beta': float(beta),
    }
    cpu_blas = _resolve_cpu_blas_threads(cpu_blas_threads)
    payload = {
        'kind': 'temporal',
        'profile': profile,
        'operator': operator,
        'families': families,
        'solver_kwargs': solver_kwargs,
        'return_eigenvectors': bool(return_eigenvectors),
        'cpu_blas_threads': cpu_blas,
    }
    t0 = time.perf_counter()
    results, n_workers, eff_blas = _run_cpu_grid(payload, xs, Res, cpu_workers)
    wall_time_s = time.perf_counter() - t0

    shape = (len(xs), len(Res))
    grids, errors = _collect_family_grids(
        results, shape, families, return_eigenvectors)

    meta = _build_meta(
        kind='temporal', api_name='pymack.sweep.temporal_sweep',
        operator=operator, backend_requested=requested,
        env_override=env_override, precision=precision, tile_size=tile_size,
        seeds=seeds, n_workers=n_workers, families=families,
        solver_kwargs=solver_kwargs, xs=xs, x_name='alpha', Res=Res,
        return_eigenvectors=return_eigenvectors, wall_time_s=wall_time_s,
        errors=errors, cpu_blas_threads=cpu_blas,
        cpu_blas_threads_effective=eff_blas)

    family_results = tuple(
        TemporalFamilyResult(
            band=band,
            c=grid['value'],
            omega_i=xs[:, None] * grid['value'].imag,
            converged=grid['converged'],
            residual=grid['residual'],
            edge_ratio=grid['edge_ratio'],
            seed_map=grid['seed_map'],
            mode_index=grid['mode_index'],
            eigenvectors=grid['eigenvectors'],
        )
        for band, grid in zip(families, grids)
    )
    return TemporalSweepResult(
        alphas=xs, Res=Res, families=family_results, meta=meta)


def spatial_sweep(profile, omegas, Res, *,
                  Ma=None, N=128, y_max=None, L=None,
                  wall_bc='isothermal', length_scale='L_star',
                  Pr=0.72, gamma=1.4,
                  lambda_mu_ratio=DEFAULT_LAMBDA_MU_RATIO,
                  operator='mack_qep',
                  families=(CBand(0.8, 0.995, label='Mack'),),
                  n_modes=25, alpha_i_abs_max=float('inf'),
                  seeds='auto',
                  backend='auto', precision='mixed',
                  tile_size='auto', return_eigenvectors=False,
                  cpu_workers=None, cpu_blas_threads=None) -> SpatialSweepResult:
    """Spatial LST sweep over an ``omegas x Res`` grid.

    Solves the spatial QEP via :func:`pymack.solver.solve_spatial`
    (shift-invert companion) once per (point, family) with the family's
    phase-speed target (``CBand.target``, default the window midpoint --
    pass the deployed driver's guess for exact adoption), then applies the
    deployed band filter: ``cr_min < omega/Re(alpha) < cr_max``,
    ``|Im(alpha)| < alpha_i_abs_max``, ``Re(alpha) > 0``, most unstable
    admitted mode (argmin ``Im(alpha)``).  Backend/parallelism semantics are
    identical to :func:`temporal_sweep` (including the opt-in
    ``cpu_blas_threads`` for pinned-BLAS workers on many-core hosts).
    """
    xs = _as_grid_axis(omegas, 'omegas')
    Res = _as_grid_axis(Res, 'Res')
    if operator not in _SPATIAL_OPERATORS:
        raise ValueError(
            f'operator must be one of {_SPATIAL_OPERATORS}, got {operator!r}')
    families = _validate_common(
        families=families, precision=precision, tile_size=tile_size,
        seeds=seeds, Ma=Ma, kind='spatial')
    for band in families:
        target = band.target
        if target is None:
            if not (np.isfinite(band.cr_min) and np.isfinite(band.cr_max)):
                raise ValueError(
                    'spatial families need finite (cr_min, cr_max) or an '
                    f'explicit target for shift-invert seeding, got {band}')
            target = 0.5 * (band.cr_min + band.cr_max)
        if target == 0:
            raise ValueError(
                f'spatial family target must be non-zero, got {band}')
    if not (isinstance(n_modes, (int, np.integer)) and n_modes >= 1):
        raise ValueError(f'n_modes must be a positive int, got {n_modes!r}')
    _resolved, requested, env_override = _resolve_backend(backend)
    if _resolved == 'gpu':
        raise NotImplementedError(
            "pymack.sweep: backend='gpu' for spatial_sweep is not implemented "
            "until the GPU spatial-engine slice lands; use backend='cpu'.")

    solver_kwargs = {
        'Ma': float(Ma), 'N': int(N),
        'y_max': None if y_max is None else float(y_max),
        'L': None if L is None else float(L),
        'wall_bc': wall_bc, 'length_scale': length_scale,
        'Pr': float(Pr), 'gamma': float(gamma),
        'lambda_mu_ratio': float(lambda_mu_ratio),
        'n_modes': int(n_modes),
        'alpha_i_abs_max': float(alpha_i_abs_max),
    }
    cpu_blas = _resolve_cpu_blas_threads(cpu_blas_threads)
    payload = {
        'kind': 'spatial',
        'profile': profile,
        'operator': operator,
        'families': families,
        'solver_kwargs': solver_kwargs,
        'return_eigenvectors': bool(return_eigenvectors),
        'cpu_blas_threads': cpu_blas,
    }
    t0 = time.perf_counter()
    results, n_workers, eff_blas = _run_cpu_grid(payload, xs, Res, cpu_workers)
    wall_time_s = time.perf_counter() - t0

    shape = (len(xs), len(Res))
    grids, errors = _collect_family_grids(
        results, shape, families, return_eigenvectors)

    meta = _build_meta(
        kind='spatial', api_name='pymack.sweep.spatial_sweep',
        operator=operator, backend_requested=requested,
        env_override=env_override, precision=precision, tile_size=tile_size,
        seeds=seeds, n_workers=n_workers, families=families,
        solver_kwargs=solver_kwargs, xs=xs, x_name='omega', Res=Res,
        return_eigenvectors=return_eigenvectors, wall_time_s=wall_time_s,
        errors=errors, cpu_blas_threads=cpu_blas,
        cpu_blas_threads_effective=eff_blas)

    family_results = tuple(
        SpatialFamilyResult(
            band=band,
            alpha=grid['value'],
            sigma=-grid['value'].imag,
            converged=grid['converged'],
            residual=grid['residual'],
            edge_ratio=grid['edge_ratio'],
            seed_map=grid['seed_map'],
            mode_index=grid['mode_index'],
            eigenvectors=grid['eigenvectors'],
        )
        for band, grid in zip(families, grids)
    )
    return SpatialSweepResult(
        omegas=xs, Res=Res, families=family_results, meta=meta)
