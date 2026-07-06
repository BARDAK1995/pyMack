"""Affine operator extraction by probing the validated CPU assemblers.

PURE NUMPY BY DESIGN.  This module never imports CuPy; it is the CPU
reference/fallback assembler of the GPU engine and must run on GPU-less CI.
(design_architecture.md section 2; measured bases in design_redteam.md.)

For a fixed structural key (profile, N, y_max, L, wall_bc, length_scale, Ma,
Pr, gamma, lambda_mu_ratio), every assembled stability matrix is exactly a
finite linear combination of scalar basis functions of the sweep parameters:

    M(p) = sum_j  m_j(p) * K_j,          m_j = declared basis functions,

with constant BC rows absorbed into the constant term.  The K_j are recovered
by assembling at a few well-conditioned probe tuples and solving one dense
least-squares system across all matrix entries simultaneously.  Nothing about
the physics is re-implemented: the CPU assembler is the oracle.

The basis is declared, never hardcoded per formulation: terms that are absent
come out as numerically zero K_j and are pruned by a Frobenius threshold;
terms that are missing from the declaration are caught by the mandatory
held-out self-verification gate (relative residual <= 1e-12), which raises
:class:`AffineExtractionError` so a silent basis error is impossible.

Non-monomial symbols are supported as basis features -- e.g. ``cos(psi)`` and
``sin(psi)`` for the 3D wave-aligned operator, which is NOT polynomial in
(alpha, beta) (measured probe residual 6.1e-4) but is exactly affine in the
(k, cos psi, sin psi) x {1, 1/Re} basis (measured 9.1e-14).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AffineExtractionError",
    "ParameterBasis",
    "AffineOperatorCache",
]


class AffineExtractionError(RuntimeError):
    """Raised when probed matrices are not affine in the declared basis.

    Callers fall back to per-point CPU assembly + batched device solve
    (identical results, slower); they must never ignore this error.
    """


# --------------------------------------------------------------------------
# Basis symbols: a symbol is either a parameter name (identity feature) or
# "cos(param)" / "sin(param)".  Each symbol may carry an integer power >= 1.
# --------------------------------------------------------------------------

_TRIG_FUNCS = {
    "cos": (np.cos, lambda x: -np.sin(x)),
    "sin": (np.sin, np.cos),
}


def _parse_symbol(symbol, params):
    """Return (func_name_or_None, param) for a basis symbol string."""
    if symbol in params:
        return None, symbol
    if symbol.endswith(")"):
        for func in _TRIG_FUNCS:
            prefix = func + "("
            if symbol.startswith(prefix):
                param = symbol[len(prefix):-1]
                if param in params:
                    return func, param
    raise ValueError(
        f"basis symbol {symbol!r} is neither a declared parameter "
        f"{tuple(params)} nor cos(p)/sin(p) of one"
    )


def _feature_value(func, values):
    if func is None:
        return values
    return _TRIG_FUNCS[func][0](values)


def _feature_derivative(func, values):
    if func is None:
        return np.ones_like(values)
    return _TRIG_FUNCS[func][1](values)


@dataclass(frozen=True)
class ParameterBasis:
    """A declared scalar basis over named sweep parameters.

    Parameters
    ----------
    params : tuple of str
        Underlying probe parameters, e.g. ``('k', 'psi', 'invRe')``.  Probe
        points and :meth:`design_matrix` columns are expressed in these.
    monomials : tuple
        One entry per basis function: a tuple of ``(symbol, power)`` pairs
        (canonically sorted).  The empty tuple is the constant term.  Use
        :meth:`from_monomials` to build from ``{symbol: power}`` dicts.
    """

    params: tuple
    monomials: tuple

    def __post_init__(self):
        params = tuple(self.params)
        if len(set(params)) != len(params):
            raise ValueError("duplicate parameter names")
        canon = []
        seen = set()
        for mono in self.monomials:
            mono = tuple(sorted((str(s), int(e)) for s, e in mono))
            for sym, exp in mono:
                _parse_symbol(sym, params)
                if exp < 1:
                    raise ValueError(f"exponent must be >= 1, got {sym}^{exp}")
            if mono in seen:
                raise ValueError(f"duplicate monomial {mono}")
            seen.add(mono)
            canon.append(mono)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "monomials", tuple(canon))

    @classmethod
    def from_monomials(cls, params, monomials):
        """Build from an iterable of ``{symbol: power}`` mappings."""
        return cls(
            params=tuple(params),
            monomials=tuple(tuple(m.items()) for m in monomials),
        )

    @property
    def n_terms(self):
        return len(self.monomials)

    def labels(self):
        """Human-readable name per monomial, e.g. ``'k*cos(psi)*invRe'``."""
        out = []
        for mono in self.monomials:
            if not mono:
                out.append("1")
                continue
            parts = [
                sym if exp == 1 else f"{sym}^{exp}" for sym, exp in mono
            ]
            out.append("*".join(parts))
        return tuple(out)

    # -- evaluation ---------------------------------------------------------

    def _points_array(self, points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[1] != len(self.params):
            raise ValueError(
                f"points must have {len(self.params)} columns "
                f"({self.params}), got shape {pts.shape}"
            )
        return pts

    def evaluate_monomials(self, points):
        """Return the (M, B) matrix of monomial values (no column scaling)."""
        pts = self._points_array(points)
        cols = np.ones((pts.shape[0], self.n_terms), dtype=float)
        for j, mono in enumerate(self.monomials):
            for sym, exp in mono:
                func, param = _parse_symbol(sym, self.params)
                vals = _feature_value(func, pts[:, self.params.index(param)])
                cols[:, j] *= vals**exp
        return cols

    def evaluate_monomial_derivatives(self, points, param):
        """Return the (M, B) matrix of d(monomial)/d(param) values."""
        if param not in self.params:
            raise ValueError(f"unknown parameter {param!r}")
        pts = self._points_array(points)
        out = np.zeros((pts.shape[0], self.n_terms), dtype=float)
        for j, mono in enumerate(self.monomials):
            # product rule over the factors that depend on `param`
            for i, (sym_i, exp_i) in enumerate(mono):
                func_i, param_i = _parse_symbol(sym_i, self.params)
                if param_i != param:
                    continue
                x = pts[:, self.params.index(param)]
                fval = _feature_value(func_i, x)
                term = exp_i * fval ** (exp_i - 1) * _feature_derivative(func_i, x)
                for k, (sym_k, exp_k) in enumerate(mono):
                    if k == i:
                        continue
                    func_k, param_k = _parse_symbol(sym_k, self.params)
                    xk = pts[:, self.params.index(param_k)]
                    term = term * _feature_value(func_k, xk) ** exp_k
                out[:, j] += term
        return out

    def design_matrix(self, points):
        """Column-scaled Vandermonde design matrix.

        Returns ``(V_scaled, col_scale)`` with ``V_scaled[:, j] =
        V[:, j] / col_scale[j]`` and ``col_scale[j] = max|V[:, j]|`` (1 for an
        all-zero column).  Coefficients fitted against ``V_scaled`` must be
        divided by ``col_scale`` to recover the unscaled expansion.
        """
        V = self.evaluate_monomials(points)
        col_scale = np.max(np.abs(V), axis=0)
        col_scale[col_scale == 0.0] = 1.0
        return V / col_scale, col_scale

    # -- probe placement ----------------------------------------------------

    def univariate_factor_counts(self):
        """Distinct univariate basis functions per parameter (incl. constant).

        A parameter appearing as {p, p^2} needs 3 distinct probe values; one
        appearing as {cos(p), sin(p)} needs 3 (the constant counts once).
        This is the minimum 1-D node count for an identifiable tensor probe.
        """
        counts = {}
        for p in self.params:
            factors = set()
            has_trig = False
            for mono in self.monomials:
                for sym, exp in mono:
                    func, param = _parse_symbol(sym, self.params)
                    if param == p:
                        factors.add((func, exp))
                        has_trig = has_trig or func is not None
            counts[p] = (len(factors) + 1, has_trig)
        return counts


# --------------------------------------------------------------------------
# Probe-point selection
# --------------------------------------------------------------------------


def _chebyshev_gauss_nodes(n, lo, hi):
    """n Chebyshev-Gauss points mapped into [lo, hi] (conditioning ~1)."""
    if n == 1:
        return np.array([0.5 * (lo + hi)])
    i = np.arange(n)
    x = np.cos(np.pi * (2 * i + 1) / (2 * n))  # open nodes in (-1, 1)
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sort(x)

_MAX_TENSOR_PROBES = 64


def _probe_points(basis, probe_box, oversample, rng):
    """Chebyshev-placed probe tuples over the actual sweep box.

    Per-parameter 1-D Chebyshev-Gauss nodes of the minimum identifiable
    degree (+1 node for parameters probed through trig features, whose
    narrow-window {1, cos, sin} columns are the worst conditioned), combined
    as a full tensor product while small (<= 64 probes; every production
    family needs <= 32).  Larger bases fall back to a seeded random subset of
    the tensor grid of size ``oversample * n_terms`` (the architecture doc's
    scattered pick), which none of the current families hits.
    """
    counts = basis.univariate_factor_counts()
    axes = []
    for p in basis.params:
        lo, hi = probe_box[p]
        n_min, has_trig = counts[p]
        n_nodes = n_min + (oversample - 1) + (1 if has_trig else 0)
        axes.append(_chebyshev_gauss_nodes(n_nodes, lo, hi))
    grids = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)
    if pts.shape[0] > _MAX_TENSOR_PROBES:
        n_keep = max(oversample * basis.n_terms, basis.n_terms + 2)
        idx = rng.choice(pts.shape[0], size=min(n_keep, pts.shape[0]),
                         replace=False)
        pts = pts[np.sort(idx)]
    return pts


# --------------------------------------------------------------------------
# Two-sided power-of-2 equilibration (exact in binary floating point)
# --------------------------------------------------------------------------


def _power2_equilibration(S, n_iter=10):
    """Ruiz-style two-sided max equilibration with power-of-2 scalings.

    ``S`` is a non-negative magnitude matrix.  Returns ``(row_scale,
    col_scale)``, both exact powers of two, such that ``diag(row) @ S @
    diag(col)`` has row/column maxima near 1.  Powers of two make the scaling
    multiplication exact in binary FP: results are bitwise comparable through
    the scaling.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("equilibration expects a square magnitude matrix")
    W = S.copy()
    row = np.ones(S.shape[0])
    col = np.ones(S.shape[1])
    for _ in range(n_iter):
        rmax = W.max(axis=1)
        cmax = W.max(axis=0)
        if (np.all((rmax == 0) | ((rmax >= 0.5) & (rmax <= 2.0)))
                and np.all((cmax == 0) | ((cmax >= 0.5) & (cmax <= 2.0)))):
            break
        rs = np.where(rmax > 0, np.exp2(np.round(-0.5 * np.log2(rmax))), 1.0)
        cs = np.where(cmax > 0, np.exp2(np.round(-0.5 * np.log2(cmax))), 1.0)
        W = rs[:, None] * W * cs[None, :]
        row *= rs
        col *= cs
    return row, col


def _is_power_of_two(x):
    x = np.asarray(x, dtype=float)
    mant, _ = np.frexp(x)
    return np.all(x > 0) and np.all(mant == 0.5)


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def _canonical_key_bytes(key):
    if key is None:
        return b"<no key>"
    return json.dumps(key, sort_keys=True, default=repr).encode("utf-8")


class AffineOperatorCache:
    """Constant K-tensors for a family of parameter-affine operators.

    Build with :meth:`from_probe`; never construct directly.  Public
    attributes (all read-only by convention):

    ``basis``            the :class:`ParameterBasis`
    ``names``            matrix names in assembler order, e.g. ('A', 'B')
    ``terms``            name -> (B_kept, nn, nn) complex128 stacked K_j
    ``kept``             name -> list of surviving monomial indices
    ``term_norms``       name -> (B,) Frobenius norms of ALL fitted K_j
    ``term_contributions``  name -> (B,) norms times monomial box magnitude
                         (the pruning metric)
    ``diagnostics``      2-norm condition numbers of the column-scaled probe
                         design matrix (``design_cond_full`` and per-name
                         ``design_cond_kept`` after pruning/refit), recorded
                         for the device engines' provenance metadata
    ``key``              structural cache key (hashed into the fingerprint)
    ``probe_points``     (M, P) probe tuples used for the fit
    ``heldout_points``   (n_verify, P) reserved verification tuples
    ``heldout_residuals``  name -> list of relative Frobenius residuals
    ``max_heldout_residual``  the gate value checked against ``rtol``
    ``row_scale, col_scale``  power-of-2 equilibration vectors (see below)
    ``timings``          seconds spent in probe / fit / verify stages
    """

    def __init__(self, **state):
        self.__dict__.update(state)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_probe(cls, assemble, basis, probe_box, *, n_verify=2, rtol=1e-12,
                   oversample=1, seed=0, key=None, prune_rtol=1e-10):
        """Extract K-tensors by probing ``assemble`` over ``probe_box``.

        Parameters
        ----------
        assemble : callable
            Takes ONLY the basis parameters as keyword arguments (everything
            structural pre-bound via ``functools.partial`` or a closure) and
            returns a dict of equally-shaped square complex matrices.
        basis : ParameterBasis
            The declared scalar basis.
        probe_box : dict
            ``{param: (lo, hi)}`` -- the sweep's ACTUAL parameter ranges.
        n_verify : int
            Held-out random tuples for the mandatory self-verification gate.
        rtol : float
            Held-out relative Frobenius residual gate (default 1e-12).
        oversample : int
            Extra per-parameter Chebyshev nodes beyond the identifiable
            minimum.
        seed : int
            Seed for the held-out tuples (and the scattered pick, if used).
        key : jsonable, optional
            Structural cache key; hashed into :meth:`fingerprint`.
        prune_rtol : float
            Drop monomial j when its box-scale contribution
            ``||K_j||_F * s_j`` (with ``s_j`` the design-matrix column scale)
            falls below ``prune_rtol`` times the largest contribution.

        Raises
        ------
        AffineExtractionError
            If any held-out residual exceeds ``rtol`` -- the operator family
            is not affine in the declared basis over this box (or a
            structural parameter leaked into the probe parameters).
        """
        if set(probe_box) != set(basis.params):
            raise ValueError(
                f"probe_box keys {sorted(probe_box)} must equal basis params "
                f"{sorted(basis.params)}"
            )
        for p, (lo, hi) in probe_box.items():
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"invalid probe range for {p!r}: {(lo, hi)}")
        if n_verify < 1:
            raise ValueError("n_verify must be >= 1 (the gate is mandatory)")

        rng = np.random.default_rng(seed)
        probe_pts = _probe_points(basis, probe_box, oversample, rng)
        if probe_pts.shape[0] < basis.n_terms:
            raise ValueError(
                f"{probe_pts.shape[0]} probes cannot identify "
                f"{basis.n_terms} basis terms"
            )
        lo = np.array([probe_box[p][0] for p in basis.params])
        hi = np.array([probe_box[p][1] for p in basis.params])
        heldout_pts = lo + (hi - lo) * rng.uniform(
            0.05, 0.95, size=(n_verify, len(basis.params))
        )

        def _call(point):
            kwargs = {p: float(v) for p, v in zip(basis.params, point)}
            result = assemble(**kwargs)
            if not isinstance(result, dict) or not result:
                raise TypeError("assemble must return a non-empty dict")
            return {k: np.asarray(v) for k, v in result.items()}

        timings = {}
        t0 = time.perf_counter()
        probe_results = [_call(pt) for pt in probe_pts]
        heldout_results = [_call(pt) for pt in heldout_pts]
        timings["probe_assembly_s"] = time.perf_counter() - t0

        names = tuple(probe_results[0].keys())
        nn = probe_results[0][names[0]].shape[0]
        for res in probe_results + heldout_results:
            if tuple(res.keys()) != names:
                raise ValueError("assemble returned inconsistent matrix names")
            for name in names:
                if res[name].shape != (nn, nn):
                    raise ValueError(
                        f"matrix {name!r} shape {res[name].shape} != {(nn, nn)}"
                    )

        # One dense least-squares fit per matrix name, all entries at once.
        t0 = time.perf_counter()
        V_scaled, col_scale = basis.design_matrix(probe_pts)
        terms = {}
        kept = {}
        term_norms = {}
        term_contributions = {}
        for name in names:
            Y = np.stack(
                [res[name].ravel() for res in probe_results]
            ).astype(np.complex128, copy=False)
            coeff_scaled, *_ = np.linalg.lstsq(V_scaled, Y, rcond=None)
            coeff = coeff_scaled / col_scale[:, None]
            K = coeff.reshape(basis.n_terms, nn, nn)
            norms = np.linalg.norm(K.reshape(basis.n_terms, -1), axis=1)
            # Frobenius pruning in CONTRIBUTION units: ||K_j||_F times the
            # monomial's magnitude over the box (the design-matrix column
            # scale).  Pruning on unscaled ||K_j||_F alone misclassifies
            # least-squares noise on tiny-magnitude columns as signal
            # (measured: invRe*k^2-flavored junk survived in the fig10.4 M10
            # family's B, whose true form is exactly alpha*B1).
            contributions = norms * col_scale
            keep = np.flatnonzero(
                contributions >= prune_rtol * contributions.max()
            )
            # Refit on the surviving columns only: dropping the nearly-null
            # tiny columns improves the conditioning of the design and
            # tightens the recovered K tensors.
            coeff_kept, *_ = np.linalg.lstsq(
                V_scaled[:, keep], Y, rcond=None
            )
            K_kept = (
                coeff_kept / col_scale[keep, None]
            ).reshape(len(keep), nn, nn)
            terms[name] = np.ascontiguousarray(K_kept)
            kept[name] = [int(j) for j in keep]
            term_norms[name] = norms
            term_contributions[name] = contributions
        timings["fit_s"] = time.perf_counter() - t0
        diagnostics = {
            "design_cond_full": float(np.linalg.cond(V_scaled)),
            "design_cond_kept": {
                name: float(np.linalg.cond(V_scaled[:, kept[name]]))
                for name in names
            },
        }

        self = cls(
            basis=basis,
            names=names,
            nn=nn,
            terms=terms,
            kept=kept,
            term_norms=term_norms,
            term_contributions=term_contributions,
            diagnostics=diagnostics,
            key=key,
            rtol=float(rtol),
            prune_rtol=float(prune_rtol),
            seed=int(seed),
            probe_box={p: (float(l), float(h)) for p, (l, h) in probe_box.items()},
            probe_points=probe_pts,
            heldout_points=heldout_pts,
            heldout_residuals=None,
            max_heldout_residual=None,
            row_scale=None,
            col_scale=None,
            timings=timings,
            _assemble=assemble,
        )

        # Mandatory held-out gate: reconstruction vs direct assembly.
        t0 = time.perf_counter()
        residuals = {name: [] for name in names}
        for pt, direct in zip(heldout_pts, heldout_results):
            model = self.evaluate(
                **{p: float(v) for p, v in zip(basis.params, pt)}
            )
            for name in names:
                ref = np.linalg.norm(direct[name])
                err = np.linalg.norm(model[name] - direct[name])
                residuals[name].append(float(err / ref) if ref > 0 else float(err))
        self.heldout_residuals = residuals
        self.max_heldout_residual = max(
            max(v) for v in residuals.values()
        )
        timings["verify_s"] = time.perf_counter() - t0
        if self.max_heldout_residual > rtol:
            worst = {
                name: float(max(v)) for name, v in residuals.items()
            }
            raise AffineExtractionError(
                f"held-out self-verification failed: max relative residual "
                f"{self.max_heldout_residual:.3e} > rtol {rtol:.1e} "
                f"(per-matrix worst: {worst}; key={key!r}). The operator is "
                f"not affine in the declared basis over this box, or a "
                f"structural parameter (grid, y_max, profile) varied with "
                f"the probe parameters. Fall back to per-point CPU assembly."
            )

        # Equilibration at the sweep-central point (exact powers of two).
        t0 = time.perf_counter()
        center = {
            p: 0.5 * (probe_box[p][0] + probe_box[p][1]) for p in basis.params
        }
        central = self.evaluate(**center)
        magnitude = np.zeros((nn, nn), dtype=float)
        for name in names:
            magnitude += np.abs(central[name])
        row, col = _power2_equilibration(magnitude)
        if not (_is_power_of_two(row) and _is_power_of_two(col)):
            raise AssertionError("equilibration scalings must be powers of 2")
        self.row_scale = row
        self.col_scale = col
        self.center = center
        timings["equilibration_s"] = time.perf_counter() - t0
        return self

    # -- evaluation -----------------------------------------------------------

    def _point(self, scalars):
        if set(scalars) != set(self.basis.params):
            raise ValueError(
                f"expected scalars {self.basis.params}, got {sorted(scalars)}"
            )
        return np.array([[float(scalars[p]) for p in self.basis.params]])

    def evaluate(self, **scalars):
        """CPU reference contraction: dict name -> assembled (nn, nn) matrix."""
        weights = self.basis.evaluate_monomials(self._point(scalars))[0]
        out = {}
        for name in self.names:
            w = weights[self.kept[name]]
            out[name] = np.tensordot(w, self.terms[name], axes=(0, 0))
        return out

    def evaluate_derivative(self, param, **scalars):
        """Analytic d(matrix)/d(param) from the basis, per matrix name.

        For the temporal families this is e.g. dA/dalpha = K_alpha +
        2*alpha*K_alpha^2 + invRe*(...); for trig features the chain rule
        d cos(psi)/d psi = -sin(psi) applies.  These derivatives power
        eigenvalue continuation and implicit-function-theorem gradients.
        """
        dweights = self.basis.evaluate_monomial_derivatives(
            self._point(scalars), param
        )[0]
        out = {}
        for name in self.names:
            w = dweights[self.kept[name]]
            out[name] = np.tensordot(w, self.terms[name], axes=(0, 0))
        return out

    def scalars(self, points):
        """Monomial values for a batch of sweep points.

        ``points`` maps every basis parameter to a broadcastable array;
        returns the (batch, B) matrix over the FULL declared basis (device
        engines column-select with ``kept[name]``).
        """
        if set(points) != set(self.basis.params):
            raise ValueError(
                f"expected points for {self.basis.params}, got {sorted(points)}"
            )
        arrays = np.broadcast_arrays(
            *[np.asarray(points[p], dtype=float) for p in self.basis.params]
        )
        batch_shape = arrays[0].shape
        flat = np.stack([a.ravel() for a in arrays], axis=1)
        vals = self.basis.evaluate_monomials(flat)
        return vals.reshape(batch_shape + (self.basis.n_terms,))

    def verify_at(self, **scalars):
        """Relative Frobenius residual vs direct assembly (max over names)."""
        direct = self._assemble(
            **{p: float(scalars[p]) for p in self.basis.params}
        )
        model = self.evaluate(**scalars)
        worst = 0.0
        for name in self.names:
            ref = np.linalg.norm(np.asarray(direct[name]))
            err = np.linalg.norm(model[name] - np.asarray(direct[name]))
            worst = max(worst, float(err / ref) if ref > 0 else float(err))
        return worst

    # -- equilibration ---------------------------------------------------------

    @property
    def equilibration(self):
        """(row_scale, col_scale): two-sided power-of-2 scaling vectors.

        Computed once at the sweep-central point from ``sum_name |M_name|``
        so one scaling pair serves the whole pencil: scaling every matrix of
        the family by the SAME ``diag(row) @ M @ diag(col)`` leaves
        generalized/polynomial eigenvalues unchanged, and powers of two make
        the transformation exact (bitwise reversible) in binary FP.
        """
        return self.row_scale, self.col_scale

    def apply_equilibration(self, matrices):
        """Return ``{name: diag(row) @ M @ diag(col)}`` (exact in binary FP)."""
        return {
            name: self.row_scale[:, None] * np.asarray(M) * self.col_scale[None, :]
            for name, M in matrices.items()
        }

    def remove_equilibration(self, matrices):
        """Exact inverse of :meth:`apply_equilibration` (bitwise round-trip)."""
        inv_row = 1.0 / self.row_scale  # exact: reciprocals of powers of two
        inv_col = 1.0 / self.col_scale
        return {
            name: inv_row[:, None] * np.asarray(M) * inv_col[None, :]
            for name, M in matrices.items()
        }

    # -- bookkeeping -----------------------------------------------------------

    def kept_labels(self, name):
        labels = self.basis.labels()
        return tuple(labels[j] for j in self.kept[name])

    def pruned_labels(self, name):
        labels = self.basis.labels()
        kept = set(self.kept[name])
        return tuple(
            labels[j] for j in range(self.basis.n_terms) if j not in kept
        )

    def fingerprint(self):
        """sha256 over key, validated probe box, basis, kept indices, tensors."""
        h = hashlib.sha256()
        h.update(_canonical_key_bytes(self.key))
        h.update(_canonical_key_bytes(self.probe_box))
        h.update(repr(self.basis).encode("utf-8"))
        h.update(repr([(n, self.kept[n]) for n in self.names]).encode("utf-8"))
        for name in self.names:
            arr = np.ascontiguousarray(self.terms[name])
            h.update(name.encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(arr.tobytes())
        h.update(np.ascontiguousarray(self.row_scale).tobytes())
        h.update(np.ascontiguousarray(self.col_scale).tobytes())
        return h.hexdigest()
