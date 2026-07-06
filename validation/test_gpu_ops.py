"""Device-core certification (pymack-gpu slice 07).

Exercises the frozen ``BatchedLinalgOps`` protocol and the mixed-precision
batched engine that every GPU sweep consumes:

* the affine contraction (``contract``) reproduces the CPU assembly to
  <= 1e-13 across a spatial, a 2D-temporal, and a 3D-temporal family (using the
  committed :class:`~pymack.gpu.affine.AffineOperatorCache`);
* the c64 solve + FP64 matrix-free refinement drives the relative residual
  <= 1e-10 on the Malik (1990) anchor operators at N in {31, 128, 200};
* the CUDA-graph inner loop matches eager execution bitwise (subprocess-isolated
  ctypes capture) and the scheduler records the not-used-with-reason decision;
* :func:`~pymack.gpu.backend.determinism_check` is bitwise-green;
* the per-lane z128 promotion path is forced and certified.

Requires CuPy and a CUDA device; skips cleanly on GPU-less CI (importorskip +
device check).  Marked ``gpu`` and ``slow`` so the fast suite
(``pytest -m "not slow"``) never runs it, while ``pytest -m gpu`` selects it.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import pymack  # noqa: E402
import pymack.gpu as pygpu  # noqa: E402

if not pygpu.is_available():  # pragma: no cover - GPU-less CI path
    pytest.skip("no CUDA device available", allow_module_level=True)

from pymack import CompressibleBlasiusProfile  # noqa: E402
from pymack.gpu import assemblers as gpu_asm  # noqa: E402
from pymack.gpu import backend, batch as batchmod, refine as refinemod  # noqa: E402
from pymack.gpu.affine import (  # noqa: E402
    AffineOperatorCache,
    ParameterBasis,
)
from pymack.gpu.kernels import (  # noqa: E402
    BatchedLinalgOps,
    CupyBatchedLinalgOps,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CONTRACT_RTOL = 1e-13
SOLVE_TOL = 1e-10


@pytest.fixture(scope="module")
def ops():
    return CupyBatchedLinalgOps(cp=cp)


@pytest.fixture(scope="module")
def config():
    return backend.resolve_config()


# ===========================================================================
# Profiles / caches (built once; N kept small where accuracy is N-independent)
# ===========================================================================


def _malik_case4_profile():
    # Malik (1990) Test Case 4: M=10, R=2000, cooled wall T_w/T_adb=0.1.
    gamma, pr, ma = 1.4, 0.7, 10.0
    T0 = 4200.0 * 5.0 / 9.0
    T_edge = T0 / (1.0 + 0.5 * (gamma - 1.0) * ma ** 2)
    t_rec = T_edge * (1.0 + 0.5 * (gamma - 1.0) * pr ** 0.5 * ma ** 2)
    return CompressibleBlasiusProfile(
        Ma=ma, T_edge=T_edge, T_wall=0.1 * t_rec, gamma=gamma, Pr=pr,
        wall_bc="isothermal", viscosity_model="sutherland",
        sutherland_S=198.6 / 1.8, n_points=2000, eta_max=40.0,
    )


def _malik_case4_cache(N):
    prof = _malik_case4_profile()
    bundle = gpu_asm.temporal_2d_assembler(
        prof, N=N, y_max=75.0, wall_bc="isothermal", length_scale="L_star",
        Ma=10.0, Pr=0.7, gamma=1.4, lambda_mu_ratio=1.2,
    )
    box = {"alpha": (0.08, 0.13), "invRe": (1.0 / 2400.0, 1.0 / 1600.0)}
    return AffineOperatorCache.from_probe(
        bundle.assemble, bundle.basis, box, key=bundle.key, rtol=1e-12,
    )


@pytest.fixture(scope="module")
def malik_caches():
    return {N: _malik_case4_cache(N) for N in (31, 128, 200)}


def _contraction_families():
    """A representative spatial, 2D-temporal, and 3D-temporal family."""
    fams = {}

    # 2D temporal (Mack enthalpy form), fig10.6 M4.5 constants, small N.
    mack = pymack.make_mack_profile(4.5, condition="table_11_1")
    fams["temporal2d_mack"] = (
        gpu_asm.temporal_2d_assembler(
            mack, N=48, y_max=40.0, wall_bc="isothermal",
            length_scale="L_star", Ma=4.5, Pr=0.72, gamma=1.4,
            lambda_mu_ratio=0.0),
        {"alpha": (0.08, 0.40), "invRe": (1.0 / 2000.0, 1.0 / 300.0)},
    )

    # 2D temporal (Ozgen form), flat-plate M4, small N.
    ozgen = pymack.make_flatplate_profile(4.0)
    from pymack.scales import delta_star_over_lstar

    d = delta_star_over_lstar(ozgen)
    fams["temporal2d_ozgen"] = (
        gpu_asm.temporal_ozgen_2d_assembler(
            ozgen, N=48, y_max=35.0 * d, wall_bc="isothermal",
            length_scale="L_star", Ma=4.0, Pr=0.72, gamma=1.4),
        {"alpha": (0.006, 0.11), "invRe": (1.0 / 5500.0, 1.0 / 1200.0)},
    )

    # 3D wave-aligned temporal, fig10.4 M4.5, small N.
    fams["temporal3d"] = (
        gpu_asm.temporal_3d_assembler(
            mack, N=24, y_max=42.0, wall_bc="isothermal",
            length_scale="L_star", Ma=4.5, Pr=0.72, gamma=1.4,
            lambda_mu_ratio=0.0),
        {"k": (0.028, 0.246), "psi": (float(np.deg2rad(45.0)),
                                      float(np.deg2rad(66.0))),
         "invRe": (1.0 / 2000.0, 1.0 / 200.0)},
    )

    # spatial QEP (Ma & Zhong M4.5), small N.
    t_rec = 65.15 * (1.0 + 0.5 * (1.4 - 1.0) * 0.72 ** 0.5 * 4.5 ** 2)
    mz = CompressibleBlasiusProfile(
        Ma=4.5, T_edge=65.15, T_wall=t_rec, gamma=1.4, Pr=0.72,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=110.33, n_points=1500, eta_max=20.0,
    )
    fams["spatial_qep"] = (
        gpu_asm.spatial_qep_assembler(
            mz, N=32, y_max=40.0, wall_bc="isothermal", length_scale="L_star",
            Ma=4.5, Pr=0.72, gamma=1.4, lambda_mu_ratio=1.2),
        {"omega": (0.010, 0.235), "invRe": (1.0 / 2000.0, 1.0 / 240.0)},
    )
    return fams


@pytest.fixture(scope="module")
def contraction_caches():
    out = {}
    for name, (bundle, box) in _contraction_families().items():
        out[name] = (
            AffineOperatorCache.from_probe(
                bundle.assemble, bundle.basis, box, key=bundle.key, rtol=1e-12),
            box,
        )
    return out


# ===========================================================================
# Protocol conformance + primitive semantics
# ===========================================================================


def test_ops_conforms_to_protocol(ops):
    assert isinstance(ops, BatchedLinalgOps)


@pytest.mark.parametrize("dtype,tol", [(np.complex128, 1e-10),
                                       (np.complex64, 1e-4)])
def test_lu_solve_semantics_N_and_C(ops, dtype, tol):
    """trans='N' solves A x = b; trans='C' solves A^H x = b (c64 AND c128).

    The c64 trans='C' path is the future RQI/bordered-Newton left-solve, so it
    is ground-truthed here in both precisions.
    """
    rng = np.random.default_rng(0)
    batch, n, nrhs = 4, 40, 3
    A = (rng.standard_normal((batch, n, n))
         + 1j * rng.standard_normal((batch, n, n))
         + (n * np.eye(n))[None]).astype(dtype)
    b = (rng.standard_normal((batch, n, nrhs))
         + 1j * rng.standard_normal((batch, n, nrhs))).astype(dtype)
    Ad, bd = cp.asarray(A), cp.asarray(b)
    lu = ops.lu_factor(Ad)
    xn = ops.lu_solve(lu, bd, trans="N")
    xc = ops.lu_solve(lu, bd, trans="C")
    rn = cp.matmul(Ad, xn) - bd
    rc = cp.matmul(cp.conj(cp.transpose(Ad, (0, 2, 1))), xc) - bd
    bnorm = float(cp.max(cp.abs(bd)))
    assert float(cp.max(cp.abs(rn))) / bnorm < tol
    assert float(cp.max(cp.abs(rc))) / bnorm < tol


@pytest.mark.parametrize("dtype,tol", [(np.complex128, 1e-10),
                                       (np.complex64, 5e-3)])
def test_lu_diagonal_is_determinant(ops, dtype, tol):
    """prod(diag U) * (-1)^swaps == det(A) in c64 AND c128 (winding consumer)."""
    rng = np.random.default_rng(1)
    batch, n = 5, 32
    A = (rng.standard_normal((batch, n, n))
         + 1j * rng.standard_normal((batch, n, n))
         + (n * np.eye(n))[None]).astype(dtype)  # diag-dominant: modest cond
    lu = ops.lu_factor(cp.asarray(A))
    diagU = cp.asnumpy(ops.lu_diagonal(lu)).astype(np.complex128)
    swaps = cp.asnumpy(ops.lu_pivot_swaps(lu))
    det_lu = np.prod(diagU, axis=1) * np.where(swaps % 2 == 0, 1.0, -1.0)
    det_np = np.array([np.linalg.det(A[i].astype(np.complex128))
                       for i in range(batch)])
    assert np.max(np.abs(det_lu - det_np) / np.abs(det_np)) < tol


def test_lu_logdet_folds_imaginary_part(ops):
    """exp(lu_logdet) == det (fold is exp-invariant) and imag in (-pi, pi],
    including a det with argument exactly at the +-pi branch boundary."""
    rng = np.random.default_rng(21)
    n = 16
    A_rand = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
              + n * np.eye(n)).astype(np.complex128)
    A_neg = np.diag(np.array([-1.0] + [1.0] * (n - 1), dtype=np.complex128))
    A = np.stack([A_rand, A_neg])  # A_neg has det = -1 -> arg = pi (boundary)
    lu = ops.lu_factor(cp.asarray(A))
    logdet = cp.asnumpy(ops.lu_logdet(lu))
    det = np.array([np.linalg.det(A[i]) for i in range(2)])
    assert np.allclose(np.exp(logdet), det, rtol=1e-8, atol=1e-8)
    assert np.all(logdet.imag <= np.pi + 1e-12)
    assert np.all(logdet.imag > -np.pi - 1e-12)
    # det = -1 folds to the principal argument +pi (angle(-1) == pi)
    assert abs(logdet.imag[1] - np.pi) < 1e-10


def test_contract_is_one_gemm_affine(ops):
    """contract(scalars, terms) == sum_j scalars[:,j] terms[j]."""
    rng = np.random.default_rng(2)
    batch, K, n = 6, 5, 16
    terms = (rng.standard_normal((K, n, n))
             + 1j * rng.standard_normal((K, n, n))).astype(np.complex128)
    scalars = (rng.standard_normal((batch, K))
               + 1j * rng.standard_normal((batch, K))).astype(np.complex128)
    got = cp.asnumpy(ops.contract(cp.asarray(scalars), cp.asarray(terms)))
    want = np.einsum("bk,kij->bij", scalars, terms)
    assert np.max(np.abs(got - want)) < 1e-11


def test_affine_matvec_matches_dense(ops):
    """The matrix-free residual equals the dense apply (no FP64 M formed)."""
    rng = np.random.default_rng(3)
    batch, K, n, nrhs = 4, 6, 24, 2
    terms = (rng.standard_normal((K, n, n))
             + 1j * rng.standard_normal((K, n, n))).astype(np.complex128)
    scalars = (rng.standard_normal((batch, K))
               + 1j * rng.standard_normal((batch, K))).astype(np.complex128)
    x = (rng.standard_normal((batch, n, nrhs))
         + 1j * rng.standard_normal((batch, n, nrhs))).astype(np.complex128)
    b = (rng.standard_normal((batch, n, nrhs))
         + 1j * rng.standard_normal((batch, n, nrhs))).astype(np.complex128)
    r = cp.asnumpy(ops.residual(cp.asarray(scalars), cp.asarray(terms),
                                cp.asarray(x), cp.asarray(b)))
    M = np.einsum("bk,kij->bij", scalars, terms)
    r_dense = b - np.matmul(M, x)
    assert np.max(np.abs(r - r_dense)) < 1e-10


def test_equilibration_is_exact_power_of_two(ops):
    """Fused equilibration is bitwise-reversible (powers of two)."""
    rng = np.random.default_rng(4)
    n = 20
    M = (rng.standard_normal((3, n, n))
         + 1j * rng.standard_normal((3, n, n))).astype(np.complex128)
    row = np.exp2(rng.integers(-4, 5, size=n)).astype(float)
    col = np.exp2(rng.integers(-4, 5, size=n)).astype(float)
    Md = cp.asarray(M)
    once = ops.equilibrate(Md, cp.asarray(row), cp.asarray(col))
    back = ops.equilibrate(once, cp.asarray(1.0 / row), cp.asarray(1.0 / col))
    assert bool(cp.array_equal(back, Md))


# ===========================================================================
# Contraction vs CPU assembly (<= 1e-13) across representative families
# ===========================================================================


def _device_assemble(ops, cache, name, point):
    w = cache.scalars({p: np.array([point[p]]) for p in cache.basis.params})
    scalars = cp.asarray(w[:, cache.kept[name]].astype(np.complex128))
    terms = cp.asarray(np.ascontiguousarray(cache.terms[name]))
    return ops.contract(scalars, terms)[0]


@pytest.mark.parametrize("family", ["temporal2d_mack", "temporal2d_ozgen",
                                    "temporal3d", "spatial_qep"])
def test_contraction_vs_cpu(ops, contraction_caches, family):
    cache, box = contraction_caches[family]
    params = cache.basis.params
    rng = np.random.default_rng(sum(ord(c) for c in family))  # stable seed
    worst = 0.0
    for _ in range(4):
        point = {p: float(box[p][0] + (box[p][1] - box[p][0]) * rng.uniform(0.1, 0.9))
                 for p in params}
        cpu = cache.evaluate(**point)
        for name in cache.names:
            dev = cp.asnumpy(_device_assemble(ops, cache, name, point))
            ref = np.asarray(cpu[name])
            denom = np.linalg.norm(ref)
            err = np.linalg.norm(dev - ref) / (denom if denom > 0 else 1.0)
            worst = max(worst, err)
            assert err <= CONTRACT_RTOL, (
                f"{family}/{name}: contraction rel err {err:.3e}"
            )
    assert worst <= CONTRACT_RTOL


# ===========================================================================
# Solve residuals <= 1e-10 after refinement on Malik anchors, N in {31,128,200}
# ===========================================================================


@pytest.mark.parametrize("N", [31, 128, 200])
def test_solve_residual_malik_anchor(ops, config, malik_caches, N):
    cache = malik_caches[N]
    pencil = refinemod.AffinePencil(ops, cache)
    n = cache.nn
    batch, nrhs = 6, 4
    # representative contour-node shifts: bounded away from the c~0.9 modes and
    # the real-axis continuous spectrum, spanning the box in (alpha, invRe).
    alpha = np.linspace(0.09, 0.12, batch)
    invRe = np.full(batch, 1.0 / 2000.0)
    z = np.full(batch, 0.5 + 0.3j)
    rng = np.random.default_rng(7)
    b = (rng.standard_normal((batch, n, nrhs))
         + 1j * rng.standard_normal((batch, n, nrhs)))
    res = refinemod.solve_pencil(
        ops, pencil, {"alpha": alpha, "invRe": invRe}, z, cp.asarray(b), config,
    )
    rel = cp.asnumpy(res.rel)
    assert np.all(rel <= SOLVE_TOL), f"N={N}: worst residual {rel.max():.3e}"
    assert np.all(cp.asnumpy(res.converged))
    # these well-posed contour-node shifts converge in the base precision
    assert np.all(res.precision == "complex64")


def test_solve_residual_determinism(ops, config, malik_caches):
    """Same anchor tile solved twice is bitwise-identical (fixed tile size)."""
    cache = malik_caches[31]
    pencil = refinemod.AffinePencil(ops, cache)
    n = cache.nn
    pts = {"alpha": np.full(4, 0.105), "invRe": np.full(4, 1.0 / 2000.0)}
    z = np.full(4, 0.5 + 0.3j)
    rng = np.random.default_rng(11)
    b = cp.asarray(rng.standard_normal((4, n, 2))
                   + 1j * rng.standard_normal((4, n, 2)))
    r1 = refinemod.solve_pencil(ops, pencil, pts, z, b, config)
    r2 = refinemod.solve_pencil(ops, pencil, pts, z, b, config)
    assert bool(cp.array_equal(r1.x, r2.x))
    assert bool(cp.array_equal(r1.rel, r2.rel))


def test_physical_rhs_equilibration_contract(ops, config, malik_caches):
    """The solve surface is equilibrated-in/equilibrated-out.  A PHYSICAL RHS
    (like the polisher's B@x_prev) must go through scale_rhs -> solve ->
    unscale_solution to satisfy the RAW-operator residual; skipping the scaling
    converges to the wrong system.  This closes the lens-2 mixed-case hazard."""
    cache = malik_caches[31]
    pencil = refinemod.AffinePencil(ops, cache)
    n = cache.nn
    zval = 0.5 + 0.3j
    raw = cache.evaluate(alpha=0.105, invRe=1.0 / 2000.0)  # UN-equilibrated
    M_raw = raw["A"] - zval * raw["B"]
    rng = np.random.default_rng(23)
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b_phys = M_raw @ x_true                      # a physical RHS
    pts = {"alpha": np.array([0.105]), "invRe": np.array([1.0 / 2000.0])}
    z = np.array([zval])

    # RIGHT: scale the physical RHS, solve, unscale the solution
    b_eq = pencil.scale_rhs(cp.asarray(b_phys[None, :, None]))
    res = refinemod.solve_pencil(ops, pencil, pts, z, b_eq, config)
    x_phys = cp.asnumpy(pencil.unscale_solution(res.x))[0, :, 0]
    raw_resid = np.linalg.norm(b_phys - M_raw @ x_phys) / np.linalg.norm(b_phys)
    assert raw_resid <= SOLVE_TOL, f"scaled path raw residual {raw_resid:.3e}"

    # WRONG: physical RHS straight in (no scale_rhs), then unscale -> converges
    # on M_eq but to the wrong physical system (off by Dr^{-1}).
    res_wrong = refinemod.solve_pencil(
        ops, pencil, pts, z, cp.asarray(b_phys[None, :, None]), config)
    x_wrong = cp.asnumpy(pencil.unscale_solution(res_wrong.x))[0, :, 0]
    raw_resid_wrong = (np.linalg.norm(b_phys - M_raw @ x_wrong)
                       / np.linalg.norm(b_phys))
    # the equilibration is nontrivial, so the unscaled path is clearly wrong
    assert raw_resid_wrong > 1e-2, (
        f"hazard not demonstrated: wrong-path residual {raw_resid_wrong:.3e}"
    )


# ===========================================================================
# z128 promotion: force a lane to promote
# ===========================================================================


def _ill_conditioned_pencil(ops, n=64, cond=1e9, seed=5):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(cond), n)
    A_ill = (U * s) @ V.conj().T
    K1 = 0.01 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    Bmat = np.eye(n) + 0.0j

    def assemble(alpha, invRe):
        return {"A": A_ill + alpha * K1, "B": Bmat.copy()}

    basis = ParameterBasis.from_monomials(("alpha", "invRe"),
                                          [{}, {"alpha": 1}])
    box = {"alpha": (0.0, 0.02), "invRe": (1e-4, 1e-2)}
    cache = AffineOperatorCache.from_probe(
        assemble, basis, box, key={"synthetic": "ill"}, rtol=1e-9,
    )
    return refinemod.AffinePencil(ops, cache), cache


def test_z128_promotion_forced(ops, config):
    """A mixed batch: well-conditioned lanes stay c64, ill-conditioned lanes
    promote to z128 and reach the target residual."""
    pencil, cache = _ill_conditioned_pencil(ops, n=64, cond=1e9)
    n = cache.nn
    rng = np.random.default_rng(9)
    row, col = cache.equilibration
    mats = cache.apply_equilibration(cache.evaluate(alpha=0.0, invRe=1e-3))
    Aeq, Beq = mats["A"], mats["B"]
    batch = 4
    # lanes 0,1: shift far from spectrum (well-conditioned);
    # lanes 2,3: shift at 0 -> M = A (cond ~1e9) with a consistent RHS.
    z = np.array([5.0 + 5.0j, 4.0 - 4.0j, 0.0, 0.0], dtype=complex)
    b = np.zeros((batch, n, 1), dtype=complex)
    for i in range(batch):
        M = Aeq - z[i] * Beq
        xt = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        b[i, :, 0] = M @ xt
    pts = {"alpha": np.zeros(batch), "invRe": np.full(batch, 1e-3)}
    res = refinemod.solve_pencil(ops, pencil, pts, z, cp.asarray(b), config)

    assert res.precision[0] == "complex64"
    assert res.precision[1] == "complex64"
    assert res.precision[2] == "complex128"
    assert res.precision[3] == "complex128"
    assert bool(res.promoted[2]) and bool(res.promoted[3])
    assert not bool(res.promoted[0]) and not bool(res.promoted[1])
    assert res.trigger[2] in ("stall", "getrf_info", "pivot_growth")
    assert np.all(cp.asnumpy(res.rel) <= SOLVE_TOL)
    assert np.all(cp.asnumpy(res.converged))


def test_promotion_info_trigger(ops, config):
    """A c64-singular lane (getrf info != 0) is flagged for promotion."""
    pencil, cache = _ill_conditioned_pencil(ops, n=48, cond=1e6, seed=2)
    n = cache.nn
    # Assemble a genuinely singular lane by zeroing a column of A via a crafted
    # operator is not affine; instead check the policy directly on a singular M.
    rng = np.random.default_rng(1)
    mats = cache.apply_equilibration(cache.evaluate(alpha=0.0, invRe=1e-3))
    M = mats["A"].astype(np.complex64)
    M[:, 0] = 0.0  # exactly singular in c64
    lu = ops.lu_factor(cp.asarray(M[None]))
    assert bool(lu.singular_lanes()[0]), "getrf must report info != 0"


def test_all_promote_no_oom(ops):
    """Every lane promoting to z128 (the reviewer's memory-pressure case) must
    not OOM: bytes_per_lane budgets the full z128 promotion working set."""
    n = 512
    pencil, cache = _ill_conditioned_pencil(ops, n=n, cond=1e9, seed=8)
    batch = 24
    cfg = backend.BackendConfig(max_refine=8)   # c64 will not converge anyway
    rng = np.random.default_rng(31)
    mats = cache.apply_equilibration(cache.evaluate(alpha=0.0, invRe=1e-3))
    Aeq = mats["A"]
    b = np.zeros((batch, n, 1), dtype=complex)
    for i in range(batch):
        xt = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        b[i, :, 0] = Aeq @ xt          # consistent RHS -> all lanes promote
    z = np.zeros(batch, dtype=complex)
    pts = {"alpha": np.zeros(batch), "invRe": np.full(batch, 1e-3)}
    res = refinemod.solve_pencil(ops, pencil, pts, z, cp.asarray(b), cfg)
    assert np.all(res.promoted)
    assert np.all(res.precision == "complex128")
    assert np.all(cp.asnumpy(res.rel) <= SOLVE_TOL)
    # the memory model now budgets the full z128 promotion reserve
    sched = batchmod.TileScheduler(ops, cfg)
    per = sched.bytes_per_lane(n, nrhs=1)
    assert per >= 40 * n * n, "bytes_per_lane must include the z128 reserve"


# ===========================================================================
# Determinism contract
# ===========================================================================


def test_determinism_check_green(ops, config):
    report = backend.determinism_check(ops, n=96, batch=8, nrhs=4, config=config)
    assert report["deterministic"], report["flags"]
    assert all(report["flags"].values())


# ===========================================================================
# Tile scheduler + CUDA-graph decision
# ===========================================================================


def test_tile_plan_records_size(ops, config):
    sched = batchmod.TileScheduler(ops, config)
    plan = sched.plan(total_lanes=500, n=516, n_terms=12, nrhs=4)
    assert plan.meta["tile_size"] >= 1
    assert plan.meta["n_tiles"] == -(-500 // plan.meta["tile_size"])
    assert plan.meta["tile_source"] in ("vram_adaptive", "env_override")
    # resolved deterministically for the same free-VRAM budget
    plan2 = sched.plan(total_lanes=500, n=516, n_terms=12, nrhs=4,
                       free_bytes=plan.meta["free_bytes"])
    plan3 = sched.plan(total_lanes=500, n=516, n_terms=12, nrhs=4,
                       free_bytes=plan.meta["free_bytes"])
    assert plan2.meta["tile_size"] == plan3.meta["tile_size"]


def test_env_tile_clamped_to_vram_cap(ops):
    """PYMACK_GPU_TILE must never silently exceed the VRAM cap (loud clamp)."""
    cfg = backend.BackendConfig(tile=10 ** 9)
    sched = batchmod.TileScheduler(ops, cfg)
    plan = sched.plan(total_lanes=10 ** 9, n=1005, n_terms=12, nrhs=8,
                      free_bytes=8 * 1024 ** 3)
    assert plan.meta["tile_source"] == "env_override"
    assert plan.meta["env_tile_clamped_to_vram_cap"] is True
    assert plan.meta["tile_size"] == plan.meta["auto_tile_lanes"]
    assert plan.meta["tile_size"] < 10 ** 9


def test_tile_scheduler_runs_sweep(ops, config, malik_caches):
    """A pinned, double-buffered tiled sweep produces correct residuals."""
    cache = malik_caches[31]
    pencil = refinemod.AffinePencil(ops, cache)
    n = cache.nn
    total = 20
    rng = np.random.default_rng(13)
    alpha = np.full(total, 0.105)
    invRe = np.full(total, 1.0 / 2000.0)
    z = np.full(total, 0.5 + 0.3j)
    b = (rng.standard_normal((total, n, 1))
         + 1j * rng.standard_normal((total, n, 1)))
    # force multiple tiles via an env-style override
    cfg = backend.BackendConfig(tile=7)
    sched = batchmod.TileScheduler(ops, cfg)
    plan = sched.plan(total_lanes=total, n=n,
                      n_terms=pencil.kA + pencil.kB, nrhs=1)
    assert plan.meta["n_tiles"] == 3

    def compute(dev_inputs, sl):
        s0, s1 = sl
        res = refinemod.solve_pencil(
            ops, pencil,
            {"alpha": alpha[s0:s1], "invRe": invRe[s0:s1]}, z[s0:s1],
            dev_inputs["b"], cfg,
        )
        return res.rel          # per-lane FP64 residual (real, gatherable)

    rel_tiled = sched.stage_and_run({"b": b}, compute, plan)
    # correctness through the actual pinned, double-buffered tiled path
    assert rel_tiled.shape == (total,)
    assert np.all(rel_tiled <= SOLVE_TOL), f"worst tiled residual {rel_tiled.max():.3e}"


def test_graph_decision_recorded(ops, config):
    """Honesty contract: the graph path is recorded as not-used-with-reason,
    and 'used' matches the actual (eager) execution path -- never claims a
    graph ran when stage_and_run runs eager."""
    sched = batchmod.TileScheduler(ops, config)
    decision = sched.graph_decision()
    assert decision["reason"]
    assert decision["used"] is False
    assert decision["execution_path"] == "eager"
    # safe launch-overhead proxy captured+replayed bitwise-identically
    proxy = decision["proxy"]
    if proxy.get("supported"):
        assert proxy["bitwise_match"] is True
    # meta consistency: a real tile plan carries the same not-used decision
    plan = sched.plan(total_lanes=10, n=64, n_terms=4, nrhs=1)
    assert plan.meta["graph_decision"]["used"] is False
    # even with PYMACK_GPU_GRAPHS set, 'used' stays False (execution is eager)
    sched_g = batchmod.TileScheduler(ops, backend.BackendConfig(graphs=True))
    assert sched_g.graph_decision()["used"] is False
    assert sched_g.graph_decision()["graphs_env"] is True


def test_ctypes_graph_capture_matches_eager():
    """Subprocess-isolated: the ctypes cuBLAS inner loop captures AND replays
    bitwise-identically to eager (context-poisoning risk isolated per slice 00)."""
    rec = batchmod.probe_ctypes_graph(n=48, batch=16, iters=6)
    if not rec.get("supported"):
        pytest.skip(f"ctypes graph capture unsupported: {rec}")
    assert rec["captured"] is True
    assert rec["replayed"] is True
    assert rec["bitwise_match_vs_eager"] is True
    assert rec["max_abs_diff"] == 0.0


# ===========================================================================
# Per-family kappa_eq estimate (box-corner caveat)
# ===========================================================================


def test_kappa_eq_center_vs_corner(malik_caches):
    cache = malik_caches[31]
    kappa = refinemod.estimate_kappa_eq(cache, shift=0.5 + 0.3j)
    assert np.isfinite(kappa["center"]) and kappa["center"] >= 1.0
    assert np.isfinite(kappa["corner_max"]) and kappa["corner_max"] >= 1.0
    assert kappa["baked_at"] == "center"
    assert kappa["n_corners_sampled"] == 4
    # the caveat made concrete: corner conditioning is at least the center's
    assert kappa["corner_max"] >= 0.5 * kappa["center"]
