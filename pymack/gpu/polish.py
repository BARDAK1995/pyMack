"""Polishing helpers for contour-produced temporal candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PolishResult", "batched_two_sided_rqi", "dense_two_sided_rqi"]


@dataclass
class PolishResult:
    value: complex
    vector: np.ndarray
    residual: float
    iterations: int
    converged: bool
    bv_norm: float = float("nan")
    edge_ratio: float = float("nan")


def _residual(A, B, z, x):
    nx = np.linalg.norm(x)
    if nx == 0.0 or not np.isfinite(nx):
        return float("inf")
    scale = np.linalg.norm(A, ord=np.inf) + abs(z) * np.linalg.norm(B, ord=np.inf)
    return float(np.linalg.norm(A @ x - z * (B @ x), ord=np.inf) / max(scale * nx, 1e-300))


def _edge_ratio_host(phi, n, n_edge=4):
    phi = np.asarray(phi)
    n_blocks = phi.shape[0] // n
    fields = np.abs(phi[:n_blocks * n].reshape(n_blocks, n))
    profile_amp = fields.max(axis=0)
    peak = float(profile_amp.max())
    if peak == 0.0:
        return float("inf")
    return float(profile_amp[:n_edge].mean() / peak)


def _batch_residual(cp, A, B, z, x):
    Ax = cp.matmul(A, x[:, :, None])[:, :, 0]
    Bx = cp.matmul(B, x[:, :, None])[:, :, 0]
    r = Ax - z[:, None] * Bx
    nx = cp.linalg.norm(x, axis=1)
    scale = cp.linalg.norm(A, ord=cp.inf) + cp.abs(z) * cp.linalg.norm(B, ord=cp.inf)
    return cp.linalg.norm(r, ord=cp.inf, axis=1) / cp.maximum(scale * nx, 1.0e-300)


def _normalize(cp, x):
    nrm = cp.linalg.norm(x, axis=1)
    return x / cp.maximum(nrm[:, None], 1.0e-300)


def batched_two_sided_rqi(ops, pencil, point, A, B, values, vectors, config,
                          *, max_iter=2, tol=1.0e-11, n_physical=None):
    """Run two-sided RQI for one sweep point on the device.

    ``values`` and ``vectors`` are host arrays from contour projection.  The
    shifted matrix is assembled through :class:`~pymack.gpu.refine.AffinePencil`
    and factored with cuBLAS in complex128.  Right solves use
    ``AffinePencil.scale_rhs`` / ``unscale_solution``; left solves use the
    adjoint scaling helpers and ``trans='C'``.
    """
    vals = np.asarray(values, dtype=np.complex128).ravel()
    vecs = np.asarray(vectors, dtype=np.complex128)
    if vals.size == 0:
        return []
    if vecs.ndim != 2 or vecs.shape[1] != vals.size:
        raise ValueError("vectors must have shape (n, n_values)")

    cp = ops.cp
    batch = vals.size
    points = {
        p: np.full(batch, float(point[p]), dtype=float)
        for p in pencil.basis.params
    }
    z = cp.asarray(vals)
    x = cp.asarray(vecs.T)
    x = _normalize(cp, x)
    w = x.copy()
    A_d = cp.asarray(np.asarray(A, dtype=np.complex128))
    B_d = cp.asarray(np.asarray(B, dtype=np.complex128))
    n = int(n_physical) if n_physical is not None else max(1, A.shape[0] // 4)
    converged = cp.zeros(batch, dtype=bool)
    residual = cp.full(batch, cp.inf, dtype=cp.float64)
    iterations = 0

    for it in range(1, int(max_iter) + 1):
        scalars = pencil.merged_scalars(points, cp.asnumpy(z))
        M = pencil.assemble(scalars, cp.complex128)
        lu = ops.lu_factor(M)

        bx = cp.matmul(B_d, x[:, :, None])[:, :, 0]
        rhs_x = pencil.scale_rhs(bx)
        y = ops.lu_solve(lu, rhs_x, trans="N").astype(cp.complex128)
        x_new = pencil.unscale_solution(y)

        bw = cp.matmul(B_d.conj().T, w[:, :, None])[:, :, 0]
        rhs_w = pencil.scale_adjoint_rhs(bw)
        q = ops.lu_solve(lu, rhs_w, trans="C").astype(cp.complex128)
        w_new = pencil.unscale_adjoint_solution(q)

        x = _normalize(cp, x_new)
        w = _normalize(cp, w_new)

        Ax = cp.matmul(A_d, x[:, :, None])[:, :, 0]
        Bx = cp.matmul(B_d, x[:, :, None])[:, :, 0]
        numer = cp.sum(cp.conj(w) * Ax, axis=1)
        denom = cp.sum(cp.conj(w) * Bx, axis=1)
        ok = cp.abs(denom) > 1.0e-300
        z_next = numer / denom
        z = cp.where(ok, z_next, z)
        residual = _batch_residual(cp, A_d, B_d, z, x)
        converged = ok & (residual <= tol)
        iterations = it
        if bool(cp.all(converged)):
            break

    bv = cp.matmul(B_d, x[:, :, None])[:, :, 0]
    bv_den = cp.linalg.norm(B_d, ord=cp.inf) * cp.maximum(cp.linalg.norm(x, axis=1), 1.0e-300)
    bv_norm = cp.linalg.norm(bv, ord=cp.inf, axis=1) / cp.maximum(bv_den, 1.0e-300)
    z_h = cp.asnumpy(z)
    x_h = cp.asnumpy(x)
    res_h = cp.asnumpy(residual)
    conv_h = cp.asnumpy(converged)
    bv_h = cp.asnumpy(bv_norm)
    return [
        PolishResult(
            value=complex(z_h[i]),
            vector=x_h[i].copy(),
            residual=float(res_h[i]),
            iterations=int(iterations),
            converged=bool(conv_h[i]),
            bv_norm=float(bv_h[i]),
            edge_ratio=_edge_ratio_host(x_h[i], n),
        )
        for i in range(batch)
    ]


def dense_two_sided_rqi(A, B, z0, x0, *, max_iter=2, tol=1.0e-11):
    """Small dense two-sided RQI reference polisher.

    The device engine uses the same mathematical step through slice-07 batched
    solves.  This CPU helper is kept for tests, fallback diagnostics, and
    benchmark bake-off bookkeeping.
    """
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    x = np.asarray(x0, dtype=np.complex128)
    x = x / max(np.linalg.norm(x), 1e-300)
    w = x.copy()
    z = complex(z0)
    for it in range(1, int(max_iter) + 1):
        M = A - z * B
        try:
            bx = B @ x
            bw = B.conj().T @ w
            x = np.linalg.solve(M, bx)
            w = np.linalg.solve(M.conj().T, bw)
        except np.linalg.LinAlgError:
            return PolishResult(z, x, _residual(A, B, z, x), it, False)
        x = x / max(np.linalg.norm(x), 1e-300)
        w = w / max(np.linalg.norm(w), 1e-300)
        denom = w.conj() @ (B @ x)
        if abs(denom) < 1e-300:
            return PolishResult(z, x, _residual(A, B, z, x), it, False)
        z = complex((w.conj() @ (A @ x)) / denom)
        res = _residual(A, B, z, x)
        if res <= tol:
            return PolishResult(z, x, res, it, True)
    res = _residual(A, B, z, x)
    return PolishResult(z, x, res, int(max_iter), res <= tol)
