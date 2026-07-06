"""Mixed-precision refinement and per-lane z128 promotion.

The device engines factor the equilibrated shifted operator ``M(z) = A - z B``
in the base precision (complex64) and drive the FP64 matrix-free residual down
by iterative refinement.  Where the base precision cannot reach the target
(near-singular shift, pivot growth, or a c64-singular factorization), the
affected lanes are promoted to complex128 and re-solved -- per lane, so a single
badly-conditioned node never forces the whole tile to the slow precision.

Design (validated end-to-end before use):

* ``AffinePencil`` holds the equilibrated K-tensors of the ``(A, B)`` pencil on
  the device in both c64 and c128.  The shifted operator ``M(z) = A - zB`` is
  assembled in ONE GEMM by merging the two term stacks with per-lane scalars
  ``[s_A, -z s_B]`` (:meth:`AffinePencil.assemble`); the FP64 residual applies
  the same merged affine form matrix-free (:meth:`AffinePencil.residual`), so no
  per-lane FP64 dense matrix is ever built (B is singular -- nothing forms
  ``B^{-1}A``; the residual certifies every candidate).

* Iterative refinement in :func:`refine_solve`: factor once in the base
  precision, then repeatedly solve the FP64 residual for a correction.  Because
  the residual is formed in FP64 while the factorization stays c64, the
  attainable relative residual is set by ``eps_fp64 * kappa_eq`` (measured on
  real anchors: equilibrated shift kappa ~1e3-2e5, floor well under 1e-10).

* Promotion policy in :func:`solve_pencil`: a lane promotes to z128 on ANY of
  (i) getrf ``info != 0`` (c64-singular), (ii) refinement stall/divergence
  (best relative residual > tol), or (iii) pivot growth above
  ``config.promote_growth``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AffinePencil",
    "RefineResult",
    "SolveResult",
    "refine_solve",
    "solve_pencil",
    "estimate_kappa_eq",
]


# ---------------------------------------------------------------------------
# The equilibrated (A - z B) pencil on the device
# ---------------------------------------------------------------------------


class AffinePencil:
    """Equilibrated pencil ``M(z) = A - z B`` from an :class:`AffineOperatorCache`.

    The cache's baked power-of-2 ``row_scale`` (``Dr``) / ``col_scale`` (``Dc``)
    are folded into every K-tensor once (exact in binary FP), so the assembled
    operator is ``M_eq(z) = Dr (A - z B) Dc`` and the shift stays exact.

    EQUILIBRATION SCALING CONTRACT (the surface is equilibrated-in /
    equilibrated-out).  :meth:`assemble`, :meth:`residual`, and
    :func:`solve_pencil` all operate on ``M_eq``.  Solving the PHYSICAL system
    ``(A - z B) x_phys = b_phys`` therefore requires scaling across the surface:

        solve ``M_eq y = Dr b_phys``  (``y = Dc^{-1} x_phys``),  then
        ``x_phys = Dc y``.

    Passing a physical RHS straight into the solve would converge to the WRONG
    system SILENTLY (the refinement residual on ``M_eq`` still goes to zero).
    Use :meth:`scale_rhs` on the RHS and :meth:`unscale_solution` on the result;
    the raw-operator residual is then correct.  Slice 08's polisher (whose
    left/right RHS are ``B x_prev`` etc. in physical units) MUST route through
    these two helpers.
    """

    def __init__(self, ops, cache, a_name="A", b_name="B"):
        self.ops = ops
        cp = ops.cp
        self.cache = cache
        self.basis = cache.basis
        self.n = int(cache.nn)
        self.a_name = a_name
        self.b_name = b_name
        row, col = cache.equilibration

        def eq(K):
            return row[:, None] * np.asarray(K) * col[None, :]

        KA = eq(cache.terms[a_name])           # (kA, n, n) complex128
        KB = eq(cache.terms[b_name])           # (kB, n, n) complex128
        merged = np.concatenate([KA, KB], axis=0)
        self.kA = int(KA.shape[0])
        self.kB = int(KB.shape[0])
        self.terms_c128 = cp.asarray(np.ascontiguousarray(merged))
        self.terms_c64 = cp.asarray(
            np.ascontiguousarray(merged.astype(np.complex64))
        )
        self.keptA = np.asarray(cache.kept[a_name], dtype=int)
        self.keptB = np.asarray(cache.kept[b_name], dtype=int)
        # device copies of the baked scalings for the equilibration helpers
        self.row_scale = cp.asarray(np.asarray(row, dtype=float))
        self.col_scale = cp.asarray(np.asarray(col, dtype=float))

    # -- equilibration scaling across the solve surface --------------------

    def scale_rhs(self, b_phys):
        """Map a PHYSICAL RHS into equilibrated space: ``Dr @ b_phys``.

        Pass the result to :func:`solve_pencil`; the solve is
        equilibrated-in / equilibrated-out (see the class contract).
        """
        cp = self.ops.cp
        b = cp.asarray(b_phys)
        if b.ndim == 2:
            return b * self.row_scale[None, :]
        return b * self.row_scale[None, :, None]

    def unscale_solution(self, y):
        """Map an equilibrated solution back to PHYSICAL units: ``Dc @ y``."""
        cp = self.ops.cp
        y = cp.asarray(y)
        if y.ndim == 2:
            return y * self.col_scale[None, :]
        return y * self.col_scale[None, :, None]

    def scale_adjoint_rhs(self, b_phys):
        """Map a PHYSICAL adjoint RHS for ``M^H y = b`` into equil space."""
        cp = self.ops.cp
        b = cp.asarray(b_phys)
        if b.ndim == 2:
            return b * self.col_scale[None, :]
        return b * self.col_scale[None, :, None]

    def unscale_adjoint_solution(self, y):
        """Map an equilibrated adjoint solution back to PHYSICAL units."""
        cp = self.ops.cp
        y = cp.asarray(y)
        if y.ndim == 2:
            return y * self.row_scale[None, :]
        return y * self.row_scale[None, :, None]

    # -- scalar assembly (tiny; built on the host, uploaded once) ----------

    def merged_scalars(self, points, z):
        """Per-lane scalars ``[s_A, -z s_B]`` for the merged term stack.

        ``points`` maps each basis parameter to a ``(batch,)`` array; ``z`` is
        the ``(batch,)`` complex shift.  Returns a device ``(batch, kA + kB)``
        complex128 array.
        """
        cp = self.ops.cp
        params = self.basis.params
        pts = np.stack(
            [np.asarray(points[p], dtype=float).ravel() for p in params],
            axis=1,
        )
        w = self.basis.evaluate_monomials(pts)          # (batch, n_terms_full)
        z = np.asarray(z, dtype=complex).ravel()
        sA = w[:, self.keptA].astype(np.complex128)
        sB = w[:, self.keptB].astype(np.complex128)
        merged = np.concatenate([sA, -z[:, None] * sB], axis=1)
        return cp.asarray(merged)

    def assemble(self, scalars, dtype):
        """Assemble ``M(z)`` in ``dtype`` (one GEMM over the merged terms)."""
        cp = self.ops.cp
        terms = self.terms_c64 if dtype == cp.complex64 else self.terms_c128
        return self.ops.contract(scalars.astype(dtype, copy=False), terms)

    def residual(self, scalars, x, b):
        """FP64 matrix-free residual ``b - M(z) x`` (no dense FP64 M)."""
        cp = self.ops.cp
        return self.ops.residual(
            scalars.astype(cp.complex128, copy=False), self.terms_c128, x, b
        )


# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------


@dataclass
class RefineResult:
    x: object            # (batch, n, nrhs) complex128 solution
    rel: object          # (batch,) final relative FP64 residual per lane
    best: object         # (batch,) best relative residual reached per lane
    iters: int           # refinement iterations performed
    history: list        # max relative residual per iteration
    converged: object    # (batch,) bool, best <= tol


def _rel_residual(cp, r, bnorm):
    rn = cp.linalg.norm(r.reshape(r.shape[0], -1), axis=1)
    return rn / bnorm


def refine_solve(ops, lu, pencil, scalars, b, *, tol, max_iter):
    """Mixed-precision iterative refinement reusing factorization ``lu``.

    Solves ``M(z) x = b`` where ``lu`` factors ``M(z)`` in its (base) precision;
    the residual and solution accumulate in FP64.  Returns a :class:`RefineResult`.
    """
    cp = ops.cp
    b = cp.asarray(b)
    squeeze = b.ndim == 2
    if squeeze:
        b = b[:, :, None]
    b64 = b.astype(cp.complex128)
    bnorm = cp.linalg.norm(b64.reshape(b64.shape[0], -1), axis=1)
    bnorm = cp.where(bnorm == 0, cp.asarray(1.0), bnorm)

    x = ops.lu_solve(lu, b, trans="N").astype(cp.complex128)
    if x.ndim == 2:
        x = x[:, :, None]
    best = cp.full(b.shape[0], cp.inf, dtype=cp.float64)
    rel = None
    history = []
    # per-lane freeze: a lane stops updating once it reaches tol, so its result
    # depends only on its own convergence path -- not on tile-mates (removes the
    # global-break coupling; each lane's answer is stable at fixed tile size).
    active = cp.ones(b.shape[0], dtype=bool)
    for it in range(max_iter):
        r = pencil.residual(scalars, x, b64)
        rel = _rel_residual(cp, r, bnorm)
        best = cp.minimum(best, rel)
        history.append(float(cp.max(rel)))
        active = active & (rel > tol)
        if not bool(cp.any(active)):
            break
        d = ops.lu_solve(lu, r, trans="N")
        if d.ndim == 2:
            d = d[:, :, None]
        x = x + cp.where(active[:, None, None], d.astype(cp.complex128), 0)
    converged = best <= tol
    if squeeze:
        x = x[:, :, 0]
    return RefineResult(x=x, rel=rel, best=best, iters=len(history),
                        history=history, converged=converged)


# ---------------------------------------------------------------------------
# Promotion policy
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Result of :func:`solve_pencil` (all arrays lead with the lane axis).

    ``rel`` is the relative FP64 residual at loop exit.  For a NON-converged
    lane this is measured BEFORE that iteration's correction is applied, so the
    promotion decision keys off ``best`` (the minimum reached), never ``rel``;
    after a promotion, ``rel`` is overwritten with the converged z128 value.
    """

    x: object            # (batch, n[, nrhs]) complex128 EQUILIBRATED solution
    rel: object          # (batch,) relative FP64 residual at loop exit
    best: object         # (batch,) best relative residual reached
    precision: object    # (batch,) '<U10' array of 'complex64' / 'complex128'
    promoted: object     # (batch,) bool
    trigger: object      # (batch,) '<U16' reason ('' if not promoted)
    growth: object       # (batch,) c64 pivot-growth factor
    base_iters: int
    promote_iters: int
    converged: object    # (batch,) bool


def _pivot_growth(cp, lu, M):
    lu_max = cp.max(cp.abs(lu.lu).reshape(lu.batch, -1), axis=1)
    m_max = cp.max(cp.abs(M).reshape(M.shape[0], -1), axis=1)
    m_max = cp.where(m_max == 0, cp.asarray(1.0), m_max)
    return lu_max / m_max


def solve_pencil(ops, pencil, points, z, b, config):
    """Factor + refine ``M_eq(z) x = b`` in c64, promoting stalled lanes to z128.

    ``points`` maps basis parameters to ``(batch,)`` arrays; ``z`` is the
    ``(batch,)`` complex shift; ``b`` is ``(batch, n[, nrhs])``.  Returns a
    :class:`SolveResult` carrying per-lane precision, promotion triggers, and
    certified FP64 residuals.

    EQUILIBRATED-IN / EQUILIBRATED-OUT: ``b`` and the returned ``x`` live in the
    equilibrated space ``M_eq = Dr (A - zB) Dc``.  For a physical RHS route it
    through :meth:`AffinePencil.scale_rhs` first and the solution through
    :meth:`AffinePencil.unscale_solution` (see the :class:`AffinePencil`
    contract) or the raw-operator residual will be wrong.
    """
    cp = ops.cp
    base_dtype = (cp.complex64 if config.base_dtype == "complex64"
                  else cp.complex128)
    tol = config.refine_tol
    max_iter = config.max_refine

    scalars = pencil.merged_scalars(points, z)
    batch = scalars.shape[0]

    M = pencil.assemble(scalars, base_dtype)
    lu = ops.lu_factor(M)
    singular = cp.asarray(lu.info != 0)
    growth = _pivot_growth(cp, lu, M)

    rr = refine_solve(ops, lu, pencil, scalars, b, tol=tol, max_iter=max_iter)
    x = rr.x
    best = rr.best
    rel = rr.rel

    precision = np.array(["complex64"] * batch, dtype="<U10")
    trigger = np.array([""] * batch, dtype="<U16")
    promoted = np.zeros(batch, dtype=bool)

    if config.promote and base_dtype == cp.complex64:
        sing_h = cp.asnumpy(singular)
        stall_h = cp.asnumpy(best > tol)
        growth_h = cp.asnumpy(growth) > config.promote_growth
        want = sing_h | stall_h | growth_h
        trigger[sing_h] = "getrf_info"
        trigger[~sing_h & growth_h] = "pivot_growth"
        trigger[~sing_h & ~growth_h & stall_h] = "stall"
        promoted = want
    promote_iters = 0
    if bool(promoted.any()):
        idx = np.flatnonzero(promoted)
        idx_d = cp.asarray(idx)
        sub_points = {p: np.asarray(points[p], dtype=float).ravel()[idx]
                      for p in pencil.basis.params}
        sub_z = np.asarray(z, dtype=complex).ravel()[idx]
        sub_scalars = pencil.merged_scalars(sub_points, sub_z)
        Mz = pencil.assemble(sub_scalars, cp.complex128)
        luz = ops.lu_factor(Mz)
        b_d = cp.asarray(b)
        sub_b = b_d[idx_d]
        rz = refine_solve(ops, luz, pencil, sub_scalars, sub_b,
                          tol=tol, max_iter=max_iter)
        promote_iters = rz.iters
        # scatter z128 results back
        x[idx_d] = rz.x
        best[idx_d] = rz.best
        rel[idx_d] = rz.rel
        precision[idx] = "complex128"

    converged = best <= tol
    return SolveResult(
        x=x, rel=rel, best=best, precision=precision, promoted=promoted,
        trigger=trigger, growth=growth, base_iters=rr.iters,
        promote_iters=promote_iters, converged=converged,
    )


# ---------------------------------------------------------------------------
# Per-family equilibrated-conditioning estimate (box-corner caveat)
# ---------------------------------------------------------------------------


def estimate_kappa_eq(cache, *, shift=0.5 + 0.3j, a_name="A", b_name="B",
                      alpha_for_qep=None, max_corners=8):
    """Estimate equilibrated conditioning at box CENTER and box CORNERS.

    Honours the slice-05 caveat: the baked equilibration is computed at the box
    CENTER, so it is not guaranteed optimal across the box.  This re-estimates
    ``kappa_eq`` at the box corners to bound the worst case a sweep will hit.

    For a pencil family (names include ``A`` and ``B``) the operator is
    ``A - shift * B``; for a spatial QEP family (``C0, C1, C2``) it is
    ``C0 + alpha C1 + alpha^2 C2`` at ``alpha_for_qep`` (defaulting to a
    box-typical value).  Returns a dict with ``center``, ``corner_max``, the
    per-corner values, and provenance.
    """
    row, col = cache.equilibration
    names = set(cache.names)
    box = cache.probe_box
    params = list(cache.basis.params)

    def operator(point):
        mats = cache.evaluate(**point)
        if {"A", "B"} <= names:
            M = mats[a_name] - shift * mats[b_name]
        elif {"C0", "C1", "C2"} <= names:
            a = alpha_for_qep
            if a is None:
                a = 0.5 * (min(box[params[0]]) + max(box[params[0]]))
            M = mats["C0"] + a * mats["C1"] + a * a * mats["C2"]
        else:
            # fall back to the first matrix name
            M = mats[cache.names[0]]
        return row[:, None] * np.asarray(M) * col[None, :]

    def kappa(point):
        return float(np.linalg.cond(operator(point)))

    center = dict(cache.center)
    k_center = kappa(center)

    # box corners (capped at max_corners)
    corner_pts = []
    n_corners = 2 ** len(params)
    for mask in range(min(n_corners, max_corners)):
        pt = {}
        for bit, p in enumerate(params):
            lo, hi = box[p]
            pt[p] = lo if (mask >> bit) & 1 else hi
        corner_pts.append(pt)
    corners = {}
    for pt in corner_pts:
        label = tuple(round(pt[p], 6) for p in params)
        corners[label] = kappa(pt)
    corner_max = max(corners.values()) if corners else k_center
    return {
        "center": k_center,
        "corner_max": corner_max,
        "corners": corners,
        "baked_at": "center",
        "shift": complex(shift) if {"A", "B"} <= names else None,
        "n_corners_sampled": len(corners),
        "note": (
            "baked equilibration is centred; corner_max bounds the worst "
            "conditioning a sweep hits (slice-05 caveat)."
        ),
    }
