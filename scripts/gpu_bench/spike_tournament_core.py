"""Numerical kernels for the slice-02 method tournament (spec pymack-gpu).

Shared by ``spike_method_tournament.py``.  Pure numerics, no physics:

* ``EmulatedSolver``  -- complex64 LU on two-sided power-of-2 equilibrated
  matrices (the slice's c64 emulation) + one FP64 iterative-refinement pass
  per solve, with per-solve FP64 residual logging and log-det phase access.
* ``beyn_contour``    -- Beyn contour projection (order K >= 1, block-Hankel)
  over a rectangle (Gauss-Legendre nodes per edge, never across corners) or a
  circle (periodic trapezoid -- optimal there, no corners), with scaled
  monomials, singular-gap logging and rank-saturation flags.
* ``winding_number``  -- independent eigenvalue count from LU log-det phase
  increments along the contour, with adaptive midpoint insertion.
* ``two_sided_rqi``   -- two-sided Rayleigh-quotient iteration on (A - zB).
* ``bordered_newton`` -- Spence-Poulton implicit-determinant Newton on the
  quadratic T(alpha).
* ``CostLedger``      -- honest factorization / solve accounting.

Everything that factors a matrix goes through the ledger.  All moment /
Rayleigh-quotient / residual accumulation is complex128.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla

EPS32 = float(np.finfo(np.float32).eps)


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------
# LU-equivalent model (pre-registered, see tournament report "cost_model"):
#   unit = one complex LU factorization at the family's primary pencil
#   dimension m (flops ~ (2/3) m^3 complex).
#   LU(d)      = (d/m)^3
#   QZ(d)      = 75 (d/m)^3   (values + left/right vectors, ~50 d^3 complex
#                              flops vs (2/3) d^3 for LU)
#   ARPACK(d)  = 8 (d/m)^3 + iteration solves (shift-invert: one dense LU of
#                the companion pencil dominates; Arnoldi solves counted as
#                triangular solves)
# Triangular solves and refinement GEMMs are NOT factorizations; they are
# accounted separately in ``solve_flops`` (complex flops) and reported, so the
# D1 rule (iii) "LU count" is scored on factorizations per its letter while
# the full flops picture stays visible.
QZ_LU_FACTOR = 75.0
# ARPACK shift-invert: recorded at the COMPANION dimension; the dominant
# factorization is one dense LU there ((d/m)^3 LU-equivalents follows from
# the dimension alone).  Arnoldi triangular solves are not counted -- this
# slightly UNDERSTATES the incumbent's spatial spine cost (favors the
# incumbent; disclosed in the report).
ARPACK_LU_FACTOR = 1.0


class CostLedger:
    """Counts factorizations by (kind, dim) plus solve flops."""

    def __init__(self):
        self.fact = {}          # (kind, dim) -> count
        self.solve_flops = 0.0  # complex flops of triangular solves + refine

    def add_fact(self, kind, dim, n=1):
        key = (str(kind), int(dim))
        self.fact[key] = self.fact.get(key, 0) + int(n)

    def add_solve(self, dim, n_rhs, refined=True):
        # forward+back substitution ~ 2 d^2 k complex flops; one refinement
        # adds a GEMM residual (d^2 k) and a second substitution pair.
        per = 2.0 * dim * dim * n_rhs
        self.solve_flops += per * (2.0 if refined else 1.0)
        if refined:
            self.solve_flops += 1.0 * dim * dim * n_rhs

    def merge(self, other):
        for key, n in other.fact.items():
            self.fact[key] = self.fact.get(key, 0) + n
        self.solve_flops += other.solve_flops

    def lu_equivalents(self, base_dim):
        base = float(base_dim) ** 3
        total = 0.0
        for (kind, dim), n in self.fact.items():
            r = (float(dim) ** 3) / base
            if kind in ("lu", "lu64"):
                total += n * r
            elif kind == "qz":
                total += n * r * QZ_LU_FACTOR
            elif kind == "arpack":
                total += n * r * ARPACK_LU_FACTOR
            else:
                raise ValueError(f"unknown factorization kind {kind}")
        return total

    def fact_flops(self):
        """Factorization cost in complex-flop units ((2/3) d^3 per LU),
        dimension-consistent across families -- the D1(iii) aggregate.
        lu64 promotions carry equal flops (device WALL cost differs; the
        lu64 fraction is reported so slice 08 can weigh it)."""
        total = 0.0
        for (kind, dim), n in self.fact.items():
            d3 = float(dim) ** 3
            if kind in ("lu", "lu64"):
                total += n * (2.0 / 3.0) * d3
            elif kind == "qz":
                total += n * (2.0 / 3.0) * d3 * QZ_LU_FACTOR
            elif kind == "arpack":
                total += n * (2.0 / 3.0) * d3 * ARPACK_LU_FACTOR
            else:
                raise ValueError(f"unknown factorization kind {kind}")
        return total

    def n_lu64(self):
        return sum(n for (k, _d), n in self.fact.items() if k == "lu64")

    def summary(self, base_dim):
        return {
            "factorizations": {f"{k}@{d}": n
                               for (k, d), n in sorted(self.fact.items())},
            "lu_equivalents": self.lu_equivalents(base_dim),
            "fact_complex_flops": float(self.fact_flops()),
            "solve_complex_flops": float(self.solve_flops),
            "base_dim": int(base_dim),
        }


# ---------------------------------------------------------------------------
# Equilibration + c64-emulated LU
# ---------------------------------------------------------------------------
def pow2_scalings(M):
    """Two-sided power-of-2 equilibration vectors (rows then columns,
    max-abs) -- identical algorithm to the corpus builder's
    ``_equilibrate_pow2``, returned as vectors."""
    a = np.abs(M)
    rmax = a.max(axis=1)
    dr = np.ones_like(rmax)
    nz = rmax > 0.0
    dr[nz] = 2.0 ** (-np.round(np.log2(rmax[nz])))
    a = a * dr[:, None]
    cmax = a.max(axis=0)
    dc = np.ones_like(cmax)
    nz = cmax > 0.0
    dc[nz] = 2.0 ** (-np.round(np.log2(cmax[nz])))
    return dr, dc


class EmulatedSolver:
    """complex64 LU of the equilibrated matrix + 1 FP64 refinement per solve.

    Emulates the GPU engine's cgetrfBatched path on CPU: the factorization is
    float32-precision on D_r M D_c (power-of-2 scalings, exact in binary FP),
    every solve gets one FP64 matrix-residual refinement, and the FP64
    relative residual of each solve is logged.
    """

    def __init__(self, M, dr, dc, ledger, kind_dim=None):
        self.m = M.shape[0]
        self.dr = dr
        self.dc = dc
        self.M_eq = (M * dr[:, None]) * dc[None, :]          # complex128 copy
        self.lu, self.piv = sla.lu_factor(
            self.M_eq.astype(np.complex64), check_finite=False)
        ledger.add_fact("lu", self.m if kind_dim is None else kind_dim)
        self._ledger = ledger
        self.solve_residuals = []
        d = np.abs(np.diag(self.lu))
        self.singular_flag = bool((not np.all(np.isfinite(self.lu))) or
                                  (d.min() == 0.0))

    def solve(self, rhs, refine=1, log=True):
        """Solve M x = rhs (rhs: (m,) or (m,k) complex128)."""
        b = np.atleast_2d(rhs.T).T if rhs.ndim == 1 else rhs
        b_eq = b * self.dr[:, None]
        y = sla.lu_solve((self.lu, self.piv), b_eq.astype(np.complex64),
                         check_finite=False).astype(np.complex128)
        for _ in range(refine):
            r = b_eq - self.M_eq @ y
            y = y + sla.lu_solve((self.lu, self.piv),
                                 r.astype(np.complex64),
                                 check_finite=False).astype(np.complex128)
        self._ledger.add_solve(self.m, b.shape[1], refined=refine > 0)
        if log:
            r = b_eq - self.M_eq @ y
            num = np.linalg.norm(r)
            den = np.linalg.norm(b_eq)
            self.solve_residuals.append(float(num / den) if den > 0 else 0.0)
        x = y * self.dc[:, None]
        return x[:, 0] if rhs.ndim == 1 else x

    def logdet_phase(self):
        """Phase of det(M_eq) from the c64 LU diagonal + pivot parity.

        det(M) differs from det(M_eq) by det(D_r) det(D_c) > 0 (constant per
        cell), so phase INCREMENTS along a contour are unaffected."""
        diag = np.diag(self.lu).astype(np.complex128)
        if np.any(diag == 0):
            return math.nan
        phase = float(np.sum(np.angle(diag)))
        swaps = int(np.count_nonzero(self.piv != np.arange(self.m)))
        return phase + math.pi * (swaps % 2)


class FP64Solver:
    """Plain complex128 LU -- the design-E promotion rung (zgetrfBatched on
    device) for the few lanes/candidates whose c64-refined accuracy stalls.
    Counted separately as ``lu64`` in the ledger."""

    def __init__(self, M, ledger):
        self.m = M.shape[0]
        self.lu, self.piv = sla.lu_factor(M, check_finite=False)
        ledger.add_fact("lu64", self.m)
        self._ledger = ledger
        d = np.abs(np.diag(self.lu))
        self.singular_flag = bool((not np.all(np.isfinite(self.lu))) or
                                  (d.min() == 0.0))

    def solve(self, rhs, refine=0, log=False):
        b = np.atleast_2d(rhs.T).T if rhs.ndim == 1 else rhs
        y = sla.lu_solve((self.lu, self.piv), b, check_finite=False)
        self._ledger.add_solve(self.m, b.shape[1], refined=False)
        return y[:, 0] if rhs.ndim == 1 else y

    def solve_herm(self, rhs):
        b = np.atleast_2d(rhs.T).T if rhs.ndim == 1 else rhs
        y = sla.lu_solve((self.lu, self.piv), b, trans=2, check_finite=False)
        self._ledger.add_solve(self.m, b.shape[1], refined=False)
        return y[:, 0] if rhs.ndim == 1 else y


# ---------------------------------------------------------------------------
# Contour node generation
# ---------------------------------------------------------------------------
def rect_nodes(rect, n_per_edge):
    """Counterclockwise Gauss-Legendre nodes per rectangle edge.

    Returns (z, w) with w = GL weight * d z /(2 pi i) so that
    sum_j w_j f(z_j) ~ (1/2 pi i) contour-integral f dz.
    Nodes are in path order (bottom, right, top, left) for winding use.
    """
    (x0, x1), (y0, y1) = rect["real"], rect["imag"]
    xi, wq = np.polynomial.legendre.leggauss(n_per_edge)
    corners = [complex(x0, y0), complex(x1, y0), complex(x1, y1),
               complex(x0, y1)]
    zs, ws = [], []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        zs.append(mid + half * xi)
        ws.append(wq * half / (2.0j * math.pi))
    return np.concatenate(zs), np.concatenate(ws)


def circle_nodes(center, radius, n):
    """Periodic-trapezoid nodes on a circle (no corners -> trapezoid is the
    spectrally-accurate rule there; the GL-per-edge mandate is for
    rectangles)."""
    th = 2.0 * math.pi * np.arange(n) / n
    z = center + radius * np.exp(1j * th)
    # (1/2 pi i) * integral f dz = (1/n) sum f(z_j) * r e^{i th_j}
    w = radius * np.exp(1j * th) / n
    return z, w


# ---------------------------------------------------------------------------
# Beyn contour projection (order K block-Hankel; K=1 is classic Beyn-1)
# ---------------------------------------------------------------------------
def beyn_contour(nodes, weights, make_solver, rhs_matrix, L, K, center, scale,
                 extract_tol=1e-8, strong_tol=1e-3, m=None,
                 saturation_slack=2, reuse_cache=None):
    """Contour moments + block-Hankel rank reveal + small eigenproblem.

    Parameters
    ----------
    nodes, weights : contour quadrature (weights already carry dz/(2 pi i)).
    make_solver    : z -> EmulatedSolver for the pencil at z (LU counted by
                     the caller's ledger through the solver).
    rhs_matrix     : (m, L) right-hand block (B V for a linear pencil so that
                     residues live on eigenvector directions; V itself for a
                     quadratic T(z)).
    L, K           : probe count and moment order (capacity K*L).
    center, scale  : scaled monomial phi(z) = (z - center)/scale (moment
                     conditioning; eigenvalues returned unscaled).
    reuse_cache    : optional dict z -> (phase, Y, res) shared across
                     sub-boxes so shared-edge factorizations are counted once.

    Returns dict with candidate values/vectors, singular values, rank info,
    per-node solve residuals, and per-node log-det phases (winding reuse).
    """
    m = rhs_matrix.shape[0] if m is None else m
    n_mom = 2 * K
    moments = [np.zeros((m, L), dtype=np.complex128) for _ in range(n_mom)]
    node_res = []
    phases = []
    for z, w in zip(nodes, weights):
        key = complex(z)
        if reuse_cache is not None and key in reuse_cache:
            phase, y, res = reuse_cache[key]
        else:
            solver = make_solver(z)
            y = solver.solve(rhs_matrix, refine=1)
            phase = solver.logdet_phase()
            res = (solver.solve_residuals[-1]
                   if solver.solve_residuals else math.nan)
            del solver
            if reuse_cache is not None:
                reuse_cache[key] = (phase, y, res)
        phases.append(phase)
        node_res.append(res)
        phi = (z - center) / scale
        f = w
        for p in range(n_mom):
            moments[p] += f * y
            f = f * phi
    # Block-Hankel stacking (Beyn 2012, Algorithm 2; K=1 reduces to classic).
    bh0 = np.vstack([np.hstack([moments[i + j] for j in range(K)])
                     for i in range(K)])
    bh1 = np.vstack([np.hstack([moments[i + j + 1] for j in range(K)])
                     for i in range(K)])
    U, s, Vh = np.linalg.svd(bh0, full_matrices=False)
    s0 = s[0] if s.size and s[0] > 0 else 1.0
    rel = s / s0
    cap = min(bh0.shape)
    # Rank policy (measured on the corpus): near-contour eigenvalues inflate
    # sigma_0 and append a smooth leakage tail, so a fixed relative threshold
    # neither counts nor saturates correctly.  Extraction is GENEROUS
    # (junk Ritz values die at the polish+FP64-certification stage);
    # saturation looks only at STRONG directions (capacity truly exhausted).
    rank = int(min(np.count_nonzero(rel > extract_tol), cap - 1))
    strong_rank = int(np.count_nonzero(rel > strong_tol))
    saturated = bool(strong_rank >= cap - saturation_slack)
    gap = float(rel[rank - 1] / rel[rank]) if 0 < rank < s.size else math.inf
    out = {
        "singular_values_rel": rel.tolist(),
        "rank": rank,
        "strong_rank": strong_rank,
        "rank_gap": gap,
        "rank_saturated": saturated,
        "capacity": int(cap),
        "node_solve_residuals": node_res,
        "phases": phases,
        "values": np.zeros(0, dtype=np.complex128),
        "vectors": np.zeros((m, 0), dtype=np.complex128),
    }
    if rank == 0:
        return out
    Uk = U[:, :rank]
    Vk = Vh[:rank, :].conj().T
    Mk = Uk.conj().T @ bh1 @ Vk / s[:rank]
    mu, sv = np.linalg.eig(Mk)
    vals = center + scale * mu
    vecs = Uk[:m, :] @ sv
    nv = np.linalg.norm(vecs, axis=0)
    nv[nv == 0] = 1.0
    out["values"] = vals
    out["vectors"] = vecs / nv[None, :]
    return out


# ---------------------------------------------------------------------------
# Winding-number certificate
# ---------------------------------------------------------------------------
def winding_number(nodes, phases, make_phase, max_extra, gap_tol=0.5 * math.pi):
    """Winding count of det along the closed node path.

    ``phases[j]`` = log-det phase at nodes[j] (path order).  Where adjacent
    increments exceed ``gap_tol`` the midpoint is evaluated recursively via
    ``make_phase(z)`` (one extra LU each, counted by the caller's ledger)
    until resolved or ``max_extra`` insertions are spent.

    Returns (winding:int|None, certified:bool, n_extra:int).
    """
    pts = [(complex(z), float(p)) for z, p in zip(nodes, phases)]
    pts.append(pts[0])
    total = 0.0
    n_extra = 0
    certified = True

    def refine(a, b, depth):
        nonlocal total, n_extra, certified
        d = _wrap(b[1] - a[1])
        if abs(d) <= gap_tol:
            total += d
            return
        if n_extra >= max_extra or depth > 12:
            certified = False
            total += d
            return
        zm = 0.5 * (a[0] + b[0])
        pm = make_phase(zm)
        n_extra += 1
        if not math.isfinite(pm):
            certified = False
            total += d
            return
        refine(a, (zm, pm), depth + 1)
        refine((zm, pm), b, depth + 1)

    for a, b in zip(pts[:-1], pts[1:]):
        if not (math.isfinite(a[1]) and math.isfinite(b[1])):
            certified = False
            continue
        refine(a, b, 0)
    w = total / (2.0 * math.pi)
    wi = int(round(w))
    if abs(w - wi) > 0.15:
        certified = False
    return (wi if certified else None), certified, n_extra


def _wrap(d):
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


# ---------------------------------------------------------------------------
# Two-sided RQI on (A - z B)
# ---------------------------------------------------------------------------
def pencil_residual(A, B, z, x, normA=None, normB=None):
    nx = np.linalg.norm(x, np.inf)
    if nx == 0 or not np.isfinite(nx):
        return math.inf
    if normA is None:
        normA = np.linalg.norm(A, np.inf)
    if normB is None:
        normB = np.linalg.norm(B, np.inf)
    return float(np.linalg.norm(A @ x - z * (B @ x), np.inf)
                 / ((normA + abs(z) * normB) * nx))


def two_sided_rqi(A, B, z0, x0, w0, ledger, normA, normB,
                  max_iter=8, res_tol=1e-11, cert_tol=1e-8,
                  breakdown_tol=1e-10, n_refine=3, use_fp64=False):
    """Two-sided RQI with c64-emulated solves.  Returns a result dict.

    Per iteration: 1 c64 LU of (A - rho B) [equilibrated] + 2 refined solves
    (right and left, same factorization) + FP64 Rayleigh quotient + residual.
    Polish/tracking solves use n_refine FP64 refinement passes (measured:
    3 passes reach ~1e-12 eigenvalue accuracy vs ~1e-7 at 1 pass; the
    factorization stays complex64 -- this is the design's promotion-free
    refinement loop, applied identically in both contenders).
    """
    x = x0 / np.linalg.norm(x0)
    w = (w0 if w0 is not None else x0)
    w = w / np.linalg.norm(w)
    rho = complex(z0)
    n_it = 0
    breakdown = False
    last_step = math.inf
    res = pencil_residual(A, B, rho, x, normA, normB)
    # NOTE: no residual-based entry exit.  The RELATIVE pencil residual can
    # pass 1e-11 while the eigenvalue is still 1e-5 off (|dz| <= res * scale
    # * kappa_c with scale = ||A||+|z|||B|| ~ 1e4 here) -- measured on the
    # Ozgen Hankel candidates.  Convergence is judged on the VALUE step.
    for _ in range(max_iter):
        n_it += 1
        M = A - rho * B
        if use_fp64:
            solver = FP64Solver(M, ledger)
        else:
            dr, dc = pow2_scalings(M)
            solver = EmulatedSolver(M, dr, dc, ledger)
        bx = B @ x
        bw = B.conj().T @ w
        if solver.singular_flag:
            rho = rho * (1.0 + 1e-10) + 1e-14
            continue
        x_new = solver.solve(bx, refine=n_refine, log=False)
        # Left solve M^H w = B^H w through the SAME factorization
        # (lu_solve trans=2), with its own FP64 refinement.
        w_new = (solver.solve_herm(bw) if use_fp64
                 else _solve_herm(solver, bw, n_refine=n_refine))
        nx = np.linalg.norm(x_new)
        nw = np.linalg.norm(w_new)
        if not (np.isfinite(nx) and np.isfinite(nw)) or nx == 0 or nw == 0:
            breakdown = True
            break
        x = x_new / nx
        w = w_new / nw
        denom = w.conj() @ (B @ x)
        numer = w.conj() @ (A @ x)
        if abs(denom) < breakdown_tol * np.linalg.norm(w) * \
                np.linalg.norm(B @ x):
            breakdown = True
            break
        rho_new = complex(numer / denom)
        last_step = abs(rho_new - rho)
        rho = rho_new
        res = pencil_residual(A, B, rho, x, normA, normB)
        if last_step < 1e-13 * max(1.0, abs(rho)):
            break
    # 3e-10: comfortably below the 1e-9 recall tolerance, above the
    # c64-refined RQI value-noise floor (else the FP64 promotion rung fires
    # on nearly every lane for nothing -- measured)
    value_converged = last_step < 3e-10 * max(1.0, abs(rho))
    return {"value": rho, "vector": x, "left": w, "residual": res,
            "last_step": float(last_step),
            "iterations": n_it,
            "converged": (res < cert_tol) and value_converged
            and not breakdown,
            "breakdown": breakdown}


def _solve_herm(solver, rhs, n_refine=1):
    """Solve M^H y = rhs through the existing c64 LU of M_eq (trans='C'),
    with FP64 refinement.  Scalings: M^H y = b <=> M_eq^H (D_r^-1 y) =
    D_c b (D real diagonal), so rows scale by dc and output by dr."""
    b = np.atleast_2d(rhs.T).T if rhs.ndim == 1 else rhs
    b_eq = b * solver.dc[:, None]
    y = sla.lu_solve((solver.lu, solver.piv), b_eq.astype(np.complex64),
                     trans=2, check_finite=False).astype(np.complex128)
    for _ in range(n_refine):
        r = b_eq - solver.M_eq.conj().T @ y
        y = y + sla.lu_solve((solver.lu, solver.piv),
                             r.astype(np.complex64),
                             trans=2, check_finite=False
                             ).astype(np.complex128)
    solver._ledger.add_solve(solver.m, b.shape[1], refined=True)
    out = y * solver.dr[:, None]
    return out[:, 0] if rhs.ndim == 1 else out


# ---------------------------------------------------------------------------
# Bordered implicit-determinant Newton on quadratic T(alpha)
# ---------------------------------------------------------------------------
def qep_residual(C0, C1, C2, a, x, norms=None):
    nx = np.linalg.norm(x, np.inf)
    if nx == 0 or not np.isfinite(nx):
        return math.inf
    if norms is None:
        norms = (np.linalg.norm(C0, np.inf), np.linalg.norm(C1, np.inf),
                 np.linalg.norm(C2, np.inf))
    n0, n1, n2 = norms
    t = (C0 + a * C1 + (a * a) * C2) @ x
    return float(np.linalg.norm(t, np.inf)
                 / ((n0 + abs(a) * n1 + abs(a) ** 2 * n2) * nx))


def bordered_newton(C0, C1, C2, a0, x0, ledger, norms,
                    max_iter=8, step_tol=1e-13, cert_tol=1e-8, n_refine=3,
                    use_fp64=False):
    """Spence-Poulton implicit-determinant Newton for T(a) x = 0.

    Borders: c = seed vector, b = T'(a0) c (nonzero left-null component for a
    simple root).  Per iteration: 1 c64 LU of the bordered matrix + 2 refined
    solves; FP64 residual certification at the end.  n_refine as in
    two_sided_rqi (measured: 3 passes reach ~4e-12 eigenvalue accuracy).
    """
    m = C0.shape[0]
    a = complex(a0)
    x = x0 / np.linalg.norm(x0)
    n_it = 0
    breakdown = False

    last_step = math.inf

    def _sweep(a, x, iters):
        """One Newton sweep with borders frozen from the entry vector."""
        nonlocal n_it, breakdown, last_step
        c = x / np.linalg.norm(x)
        b = (C1 + 2.0 * a * C2) @ c
        nb = np.linalg.norm(b)
        if nb == 0 or not np.isfinite(nb):
            b = np.ones(m, dtype=np.complex128)
            nb = np.linalg.norm(b)
        b = b / nb
        for _ in range(iters):
            n_it += 1
            T = C0 + a * C1 + (a * a) * C2
            F = np.zeros((m + 1, m + 1), dtype=np.complex128)
            F[:m, :m] = T
            F[:m, m] = b
            F[m, :m] = c.conj()
            if use_fp64:
                solver = FP64Solver(F, ledger)
            else:
                dr, dc = pow2_scalings(F)
                solver = EmulatedSolver(F, dr, dc, ledger)
            if solver.singular_flag:
                breakdown = True
                break
            rhs = np.zeros((m + 1, 1), dtype=np.complex128)
            rhs[m, 0] = 1.0
            sol0 = solver.solve(rhs, refine=n_refine, log=False)[:, 0]
            x, g = sol0[:m], sol0[m]
            Tp = (C1 + 2.0 * a * C2) @ x
            rhs2 = np.zeros(m + 1, dtype=np.complex128)
            rhs2[:m] = -Tp
            sol1 = solver.solve(rhs2, refine=n_refine, log=False)
            gp = sol1[m]
            if not np.isfinite(gp) or gp == 0:
                breakdown = True
                break
            da = -g / gp
            a = a + da
            last_step = abs(da)
            if abs(da) < step_tol * max(1.0, abs(a)):
                break
        return a, x

    a, x = _sweep(a, x, max_iter)
    nx = np.linalg.norm(x)
    if nx > 0 and np.isfinite(nx):
        x = x / nx
    res = qep_residual(C0, C1, C2, a, x, norms)
    # Border refresh: rebuilding (b, c) from the converged vector removes the
    # conditioning floor a poor seed border imposes on g (measured: ~1e-7 ->
    # ~1e-11 eigenvalue accuracy from noisy Beyn starts), then 1-3 cheap
    # Newton steps re-polish.  Skipped on breakdown.
    if not breakdown and np.isfinite(res) and res < 1e-2:
        a, x = _sweep(a, x, 3)
        nx = np.linalg.norm(x)
        if nx > 0 and np.isfinite(nx):
            x = x / nx
        res = qep_residual(C0, C1, C2, a, x, norms)
    value_converged = last_step < 3e-10 * max(1.0, abs(a))
    return {"value": a, "vector": x, "residual": res,
            "last_step": float(last_step),
            "iterations": n_it,
            "converged": (res < cert_tol) and value_converged
            and not breakdown,
            "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def dedupe(values, vectors, residuals, tol=1e-8):
    """Cluster candidates within tol*max(1,|z|); keep best-residual member."""
    order = np.argsort(np.asarray(residuals))
    kept = []
    for i in order:
        z = values[i]
        if any(abs(z - values[j]) <= tol * max(1.0, abs(values[j]))
               for j in kept):
            continue
        kept.append(i)
    kept.sort()
    return kept


def match_sets(found, truth, tol=1e-9):
    """For each truth value, nearest found distance and pass flag
    (|d| <= tol * max(1, |truth|))."""
    found = np.asarray(found, dtype=complex)
    rows = []
    for t in np.asarray(truth, dtype=complex):
        if found.size:
            d = float(np.min(np.abs(found - t)))
        else:
            d = math.inf
        rows.append({"truth": [float(t.real), float(t.imag)],
                     "distance": d,
                     "recalled": bool(d <= tol * max(1.0, abs(t)))})
    return rows
