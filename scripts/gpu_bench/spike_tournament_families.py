"""Family adapters + method runners for the slice-02 method tournament.

Binds BOTH contenders to the committed production assemblers and the verbatim
CPU filter stack (via ``build_truth_set``, which itself imports the production
modules).  No physics is reimplemented here:

* assembly     -- ``_assemble_temporal_ozgen_2d_evp`` (slice-04 extraction),
  ``assemble_temporal_compressible_3d_evp`` + BC appliers,
  ``_assemble_spatial_qep``, ``pymack.dense.quadratic_matrices``.
* filters      -- ``discrete_mode._decaying_candidates`` and the corpus
  builder's ``derive_*_verdict`` transcriptions (cross-checked verbatim by
  slice 01 against the production drivers).
* Ma&Zhong verdicts follow the amended D1 ``verdict_basis`` scoping: the
  production basis is a shift-invert window of the 25 eigenvalues nearest the
  target, so the contour method reproduces THAT mechanism (target-centred
  disk, 25 nearest certified candidates, verbatim band filter) instead of
  judging box content; the box content is still reported honestly.

Pre-registered scoring (recorded in the report):
* truth modes per cell = the production candidate set the verdict reduction
  (argmax / argmin / two-domain match) runs over -- decaying candidates for
  Ozgen, band-filtered eigenvalues for Mack fig10.4, in-band production
  window candidates for Ma&Zhong, ``candidate_indices`` survivors for the
  Mach-6 dense family.  A secondary all-in-box-QZ recall diagnostic is also
  reported.
* a truth mode is recalled iff a certified candidate lies within
  1e-9 * max(1, |truth|) after polish.
"""
from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORPUS_DEFAULT = REPO / "verification" / "gpu_certification" / "hard_cells"
for _p in (str(HERE), str(CORPUS_DEFAULT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_truth_set as bts  # noqa: E402  (adds production dirs to path)
import spike_tournament_core as core  # noqa: E402
from pymack.dense import (  # noqa: E402
    candidate_indices,
    omega_from_frequency,
    prepare_dense_case,
    quadratic_matrices,
    solve_spatial_evp,
)
from pymack.scales import delta_star_over_lstar, sample_baseflow  # noqa: E402
from pymack.spectral import chebyshev_D, physical_derivatives  # noqa: E402
from pymack.temporal_solver import (  # noqa: E402
    _assemble_temporal_ozgen_2d_evp,
)

# ---------------------------------------------------------------------------
# Pre-registered tournament configuration (all deviations from census.json
# are measured-justified; see "census" vs "contour" blocks + notes).
# ---------------------------------------------------------------------------
CFG = {
    "recall_match_tol": 1.0e-9,
    "cert_residual": 1.0e-8,
    "dedupe_tol": 1.0e-8,
    "spurious_tol": 1.0e-6,
    # Rank policy (measured, see report): extraction generous, saturation
    # from STRONG directions only (near-contour leakage tails otherwise
    # fake-saturate every dense-field contour).
    "extract_tol": 1.0e-8,
    "strong_tol": 1.0e-3,
    "bv_floor": 1.0e-10,
    "polish_steps": 3,
    "polish_escalation_steps": 5,
    # polish escalates while the FP64 residual exceeds this target: a 1e-8
    # residual can still leave ~1e-7 eigenvalue error on clustered modes
    # (measured on hc_001/hc_040); the 1e-9 recall tolerance needs ~1e-11
    "polish_residual_target": 1.0e-11,
    "max_resplit_events": 2,
    # winding-from-LU-diagonals: measured to alias undetectably at
    # production dims (det phase velocity is set by ALL ~800 eigenvalues:
    # ~100 rad between adjacent nodes at m=804; adaptive bisection
    # certifies wrong counts).  Enabled only for small pencils.
    "winding_max_dim": 300,
    # c64 emulation: node solves = 1 FP64 refinement pass (contract);
    # polish/tracking solves = 3 passes (measured: 1 pass leaves ~3e-7
    # eigenvalue error on damped Ma&Zhong modes, 3 passes reach ~4e-12;
    # factorizations stay complex64; applied identically to both methods).
    "node_solve_refine": 1,
    "polish_solve_refine": 3,
    "track_max_iter": 8,
    "bundle_cap": 8,
    "spine_param_step": 0.02,
    "perturbations": [0.8, 1.0, 1.2],
    "winding_max_rank": 40,
    "winding_extra_budget_factor": 4,
    "families": {
        "ozgen_first_pair": {
            "census": {"N_q": 32, "L": 112, "margin": 0.04, "collar": 0.02},
            "contour": {
                "nq_edge": 12, "K": 1, "L": 120,
                "guard": {"left": 0.02, "right": 0.005, "imag": 0.02},
                "collar": 0.02,
                "variants": {
                    "single": {"L": 120, "K": 1, "n_split": 1},
                    "split4": {"L": 64, "K": 1, "n_split": 4},
                    "hankel": {"L": 48, "K": 3, "n_split": 1},
                },
            },
            "notes": ("guard: census margin 0.04 would cross the c~1 "
                      "vorticity/entropy accumulation (measured: guard "
                      "counts 168-408 vs L=112); right edge capped at "
                      "+0.005 (measured leak +0..+2), other sides at the "
                      "census collar 0.02.  L=120 vs census 112: measured "
                      "guard content reaches 111."),
        },
        "ozgen_second_pair": {
            "census": {"N_q": 32, "L": 76, "margin": 0.04, "collar": 0.02},
            "contour": {
                "nq_edge": 12, "K": 1, "L": 88,
                "guard": {"left": 0.02, "right": 0.005, "imag": 0.02},
                "collar": 0.02,
                "variants": {
                    "single": {"L": 88, "K": 1, "n_split": 1},
                    "split4": {"L": 56, "K": 1, "n_split": 4},
                    "hankel": {"L": 40, "K": 3, "n_split": 1},
                },
            },
            "notes": ("box right edge 0.99 sits 0.01 from the c=1 "
                      "accumulation: +0.02 pulls in 153-187 modes, +0.005 "
                      "pulls 1-6 (measured); L=88 vs census 76 for rank "
                      "headroom (guard content ~74)."),
        },
        "mack_fig10_4_m10_3d": {
            "census": {"N_q": 32, "L": 40, "margin": 0.02576021438541192,
                       "collar": 0.01288010719270596},
            "contour": {
                "nq_edge": 12, "K": 1, "L": 112,
                "guard": {"left": 0.02576021438541192,
                          "right": 0.02576021438541192,
                          "imag": 0.02576021438541192},
                "collar": 0.01288010719270596,
                "variants": {"single": {"L": 112, "K": 1, "n_split": 1}},
            },
            "notes": ("census L=40 replaced by 112 (nq 8->12/edge) by "
                      "measurement: the production phys filter admits modes "
                      "down to c_i=-0.526 and the c~1 vorticity/entropy "
                      "continuum leaks through the right guard edge, "
                      "inflating the Beyn extract-rank to ~63 on top of the "
                      "<=36 true finite eigenvalues.  L=64 forced clip-driven "
                      "resplits that recovered recall but made the +-20% "
                      "in-box count resplit-geometry-sensitive (hc_034); "
                      "L=112 enumerates the full guard in one box (no clip, "
                      "no resplit) -> recall AND count-stability."),
        },
        "mazhong_first_spatial": {
            "census": {"N_q": 24, "L": 20, "margin": 0.0032113762442823727,
                       "collar": 0.0016056881221411864},
            "contour": {"disk_nq": 64, "disk_L": 112, "disk_L_max": 112,
                        "window_n": 25},
            "notes": ("verdict basis is the production shift-invert window "
                      "(25 nearest the target): contour reproduces the "
                      "mechanism with a target-centred disk (25-nearest is "
                      "exact for a centred disk; verified against stored "
                      "prod_candidates to ~1e-10).  Census N_q=24/L=20 "
                      "replaced by 64/96-112 by measurement: the target "
                      "sits in a dense continuum cluster (78 eigenvalues "
                      "within r=0.0152 at hc_043; 107 within r=0.116 at "
                      "hc_040), so the window disk must enumerate the "
                      "cluster.  Radius ladder driven by CERTIFIED "
                      "candidate counts, not rank (leakage tails make rank "
                      "counting unreliable in dense fields)."),
        },
        "mazhong_second_spatial": {
            "census": {"N_q": 24, "L": 28, "margin": 0.0010740757336691498,
                       "collar": 0.0005370378668345749},
            "contour": {"disk_nq": 64, "disk_L": 112, "disk_L_max": 112,
                        "window_n": 25},
            "notes": "same window mechanism as mazhong_first_spatial.",
        },
        "mach6_spatial_dense": {
            "census": {"N_q": 16, "L": 12, "margin": 0.00015307635688069783,
                       "collar": 7.653817844034891e-05},
            "contour": {
                "nq_edge": 8, "K": 1, "L": 56,
                "guard": {"left": 0.00015307635688069783,
                          "right": 0.00015307635688069783,
                          "imag": 0.00015307635688069783,
                          "target_aspect": 2.0},
                "collar": 7.653817844034891e-05,
                "variants": {"single": {"L": 56, "K": 1, "n_split": 1}},
            },
            "notes": ("census N_q=16/L=12 replaced by measurement: the "
                      "production candidate box has ~26-76:1 aspect ratio "
                      "(width 0.003-0.016, height 0.075-0.44) and the "
                      "quadrature cannot resolve it (rank saturation, zero "
                      "certified candidates).  Guard pads the real sides to "
                      "aspect<=2; measured padded-guard content 30-45 -> "
                      "L=56, 8 GL nodes/edge."),
        },
    },
}


def _seed_for(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _cpair(z):
    return [float(np.real(z)), float(np.imag(z))]


# ---------------------------------------------------------------------------
# Assembly (committed production assemblers only)
# ---------------------------------------------------------------------------
def assemble_ozgen(params, ymf, alpha):
    prof = bts.dm._profile(float(params["Ma"]))
    dstar = float(delta_star_over_lstar(prof))
    N = int(params["N"])
    d_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(d_eta, float(ymf) * dstar, N, None)
    bf = sample_baseflow(prof, y, str(params["length_scale"]))
    A, B = _assemble_temporal_ozgen_2d_evp(
        bf, y, D1, D2, float(alpha), float(params["Re"]),
        float(params["Ma"]), 0.72, 1.4, wall_bc=str(params["wall_bc"]))
    return A, B, y


def assemble_mack(params, alpha, beta):
    key = round(float(params["Ma"]), 4)
    if key not in bts._MACK_PROFILE_CACHE:
        bts._MACK_PROFILE_CACHE[key] = bts.mk.make_profile(key)
    prof = bts._MACK_PROFILE_CACHE[key]
    A, B, y, D1, n, _al, _be, _bf = bts.assemble_temporal_compressible_3d_evp(
        prof, float(alpha), float(beta), float(params["R"]),
        float(params["Ma"]), float(params["Pr"]), float(params["gamma"]),
        N=int(params["N"]), y_max=float(params["y_max"]),
        length_scale="L_star",
        lambda_mu_ratio=float(params["lambda_mu_ratio"]))
    bts.apply_wall_bc_3d(A, B, D1, n)
    bts.apply_dirichlet_freestream_bc_3d(A, B, n)
    return A, B, y


def assemble_mazhong(params, omega):
    if "profile" not in bts._MAZHONG_PROFILE_CACHE:
        bts._MAZHONG_PROFILE_CACHE["profile"] = bts.mzc.build_profile()
    prof = bts._MAZHONG_PROFILE_CACHE["profile"]
    C0, C1, C2, y = bts._assemble_spatial_qep(
        prof, float(omega), float(params["R"]), float(params["Ma"]),
        float(params["Pr"]), float(params["gamma"]), int(params["N"]),
        float(params["y_max"]), None, str(params["wall_bc"]), "L_star",
        float(params["lambda_mu_ratio"]))
    return C0, C1, C2, y


def _mach6_case(params):
    import json as _json
    key = _json.dumps([params["gas"], params["base_cfg"], params["lst_cfg"]],
                      sort_keys=True)
    if key not in bts._MACH6_CASE_CACHE:
        gas = bts.DenseGasModel(**params["gas"])
        base_cfg = bts.DenseBaseFlowConfig(**params["base_cfg"])
        lst_cfg = bts.DenseLSTConfig(**params["lst_cfg"])
        _base, y, D, base_grid = prepare_dense_case(gas, base_cfg, lst_cfg)
        bts._MACH6_CASE_CACHE[key] = (gas, lst_cfg, y, D, base_grid)
    return bts._MACH6_CASE_CACHE[key]


def assemble_mach6(params, omega, R):
    gas, lst_cfg, y, D, base_grid = _mach6_case(params)
    A0, A1, A2 = quadratic_matrices(float(omega), float(R), y, D, base_grid,
                                    gas, lst_cfg)
    return A0, A1, A2, y


# ---------------------------------------------------------------------------
# Pencil operations (linear pencil / quadratic T)
# ---------------------------------------------------------------------------
class PencilOps:
    """A - z B."""

    kind = "pencil"

    def __init__(self, A, B):
        self.A, self.B = A, B
        self.m = A.shape[0]
        self.normA = float(np.linalg.norm(A, np.inf))
        self.normB = float(np.linalg.norm(B, np.inf))

    def matrix(self, z):
        return self.A - z * self.B

    def scalings(self, z):
        return core.pow2_scalings(self.matrix(z))

    def factory(self, dr, dc, ledger):
        return lambda z: core.EmulatedSolver(self.matrix(z), dr, dc, ledger)

    def rhs(self, V):
        return self.B @ V

    def residual(self, z, x):
        return core.pencil_residual(self.A, self.B, z, x,
                                    self.normA, self.normB)

    def polish(self, z0, x0, ledger, steps, cert_tol, use_fp64=False):
        return core.two_sided_rqi(self.A, self.B, z0, x0, None, ledger,
                                  self.normA, self.normB, max_iter=steps,
                                  cert_tol=cert_tol, use_fp64=use_fp64)

    def bv_ratio(self, x):
        nx = np.linalg.norm(x)
        if nx == 0:
            return 0.0
        return float(np.linalg.norm(self.B @ x) / (self.normB * nx))


class QepOps:
    """T(z) = C0 + z C1 + z^2 C2."""

    kind = "qep"

    def __init__(self, C0, C1, C2):
        self.C0, self.C1, self.C2 = C0, C1, C2
        self.m = C0.shape[0]
        self.norms = (float(np.linalg.norm(C0, np.inf)),
                      float(np.linalg.norm(C1, np.inf)),
                      float(np.linalg.norm(C2, np.inf)))

    def matrix(self, z):
        return self.C0 + z * self.C1 + (z * z) * self.C2

    def scalings(self, z):
        return core.pow2_scalings(self.matrix(z))

    def factory(self, dr, dc, ledger):
        return lambda z: core.EmulatedSolver(self.matrix(z), dr, dc, ledger)

    def rhs(self, V):
        return V

    def residual(self, z, x):
        return core.qep_residual(self.C0, self.C1, self.C2, z, x, self.norms)

    def polish(self, z0, x0, ledger, steps, cert_tol, use_fp64=False):
        return core.bordered_newton(self.C0, self.C1, self.C2, z0, x0,
                                    ledger, self.norms, max_iter=steps,
                                    cert_tol=cert_tol, use_fp64=use_fp64)

    def bv_ratio(self, x):
        return 1.0  # no linear-pencil infinite-mode leakage channel


# ---------------------------------------------------------------------------
# Truth sets (production-candidate basis, from the verified NPZ truth)
# ---------------------------------------------------------------------------
def ozgen_decaying(vals, vecs, y, params):
    """Verbatim per-mode application of discrete_mode._decaying_candidates."""
    keep, fs = [], []
    for k in range(len(vals)):
        out = bts.dm._decaying_candidates(
            np.asarray([vals[k]]), vecs[:, [k]], np.asarray(y, dtype=float),
            ci_abs_max=float(params["ci_abs_max"]),
            cr_band=tuple(params["cr_band"]),
            fs_thresh=float(params["fs_thresh"]))
        keep.append(bool(out))
        fs.append(float(out[0][1]) if out else math.nan)
    return np.asarray(keep, dtype=bool), np.asarray(fs)


def mack_band_mask(vals, params):
    v = np.asarray(vals, dtype=complex)
    fin = np.isfinite(v)
    phys = (fin & (v.real > float(params["phys_cr"][0]))
            & (v.real < float(params["phys_cr"][1]))
            & (np.abs(v.imag) < float(params["phys_ci_abs"])))
    lo, hi = params["cr_band"]
    return phys & (v.real > float(lo)) & (v.real < float(hi)) \
        & (v.imag < float(params["ci_cap"]))


def mazhong_band_mask(vals, params, omega=None):
    v = np.asarray(vals, dtype=complex)
    om = float(params["omega"] if omega is None else omega)
    fin = np.isfinite(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = om / v.real
    return (fin & (c > float(params["c_lo"])) & (c < float(params["c_hi"]))
            & (np.abs(v.imag) < float(params["ai_cap"])) & (v.real > 0.0))


def truth_sets(entry, spectra):
    """Per-spectrum production-candidate truth (values + stored
    eigenvectors, for FP64 truth refinement) + in-box QZ values."""
    kind, p = entry["kind"], entry["params"]
    prod, prod_vecs, inbox = [], [], []
    for s in spectra:
        raw = np.asarray(s["raw_eigenvalues"], dtype=complex)
        inbox.append(raw[bts.in_box_mask(entry, raw)])
        idx = np.asarray(s["candidate_indices"], dtype=int)
        vec = np.asarray(s["candidate_vectors"], dtype=complex)
        if kind == "ozgen_pair":
            ev = np.asarray(s["prod_eigenvalues"], dtype=complex)[idx]
            keep, _fs = ozgen_decaying(ev, vec, s["y"], p)
            prod.append(ev[keep])
            prod_vecs.append(vec[:, keep])
            continue
        if kind == "mack_3d":
            vals = raw[mack_band_mask(raw, p)]
        elif kind == "mazhong_spatial":
            pc = np.asarray(s["prod_candidates"], dtype=complex)
            vals = pc[mazhong_band_mask(pc, p)]
        elif kind == "mach6_dense":
            cfg = bts.DenseLSTConfig(**p["lst_cfg"])
            vals = raw[candidate_indices(raw, float(p["omega_L"]), cfg)]
        else:
            raise ValueError(kind)
        # map each truth value to its stored eigenvector (all production
        # candidates are in-box, and in-box vectors are stored)
        stored_vals = (np.asarray(s["prod_eigenvalues"],
                                  dtype=complex)[idx]
                       if kind == "ozgen_pair" else raw[idx])
        cols = np.zeros((vec.shape[0], len(vals)), dtype=complex)
        for j, v in enumerate(vals):
            if idx.size:
                k = int(np.argmin(np.abs(stored_vals - v)))
                if abs(stored_vals[k] - v) <= 1e-6 * max(1.0, abs(v)):
                    cols[:, j] = vec[:, k]
        prod.append(np.asarray(vals, dtype=complex))
        prod_vecs.append(cols)
    return prod, prod_vecs, inbox


def assembly_certificate(entry, spectra, ops_list):
    """FP64 residual of stored truth pairs against the freshly assembled
    operators -- proves the tournament factors the SAME physics."""
    worst = 0.0
    for s, ops in zip(spectra, ops_list):
        idx = np.asarray(s["candidate_indices"], dtype=int)
        if idx.size == 0:
            continue
        if entry["kind"] == "ozgen_pair":
            vals = np.asarray(s["prod_eigenvalues"], dtype=complex)[idx]
        else:
            vals = np.asarray(s["raw_eigenvalues"], dtype=complex)[idx]
        vecs = np.asarray(s["candidate_vectors"], dtype=complex)
        for k in range(min(idx.size, 8)):
            worst = max(worst, ops.residual(vals[k], vecs[:, k]))
    return float(worst)


# ---------------------------------------------------------------------------
# Verdicts (verbatim filter stack) + identity scoring
# ---------------------------------------------------------------------------
def verdict_from_candidates(entry, cand_per_spectrum, ys):
    kind, p = entry["kind"], entry["params"]
    if kind == "ozgen_pair":
        spectra = []
        for (vals, vecs), y in zip(cand_per_spectrum, ys):
            spectra.append({
                "prod_eigenvalues": np.asarray(vals, dtype=complex),
                "candidate_indices": np.arange(len(vals)),
                "candidate_vectors": np.asarray(vecs, dtype=complex),
                "y": np.asarray(y, dtype=float),
            })
        return bts.derive_ozgen_verdict(spectra, p)
    vals = np.asarray(cand_per_spectrum[0][0], dtype=complex)
    if kind == "mack_3d":
        return bts.derive_mack_verdict(vals, p)
    if kind == "mazhong_spatial":
        return bts.derive_mazhong_verdict(vals, p)
    if kind == "mach6_dense":
        return bts.derive_mach6_verdict(vals, p)
    raise ValueError(kind)


def verdict_identity(entry, mine, tol=1e-9):
    stored = entry["verdict"]
    kind = entry["kind"]
    fields = {"status": stored["status"] == mine["status"]}
    if stored["status"] == "discrete_mode" and mine["status"] == \
            "discrete_mode":
        zs = complex(*stored["selected"])
        zm = complex(*mine["selected"])
        fields["selected"] = bool(abs(zs - zm) <= tol * max(1.0, abs(zs)))
        if kind == "ozgen_pair":
            fields["n_match"] = stored["n_match"] == mine["n_match"]
            fields["selected_fs"] = bool(
                abs(stored["selected_fs"] - mine["selected_fs"]) <= 1e-5)
        elif kind == "mack_3d":
            fields["n_band"] = stored["n_band"] == mine["n_band"]
        elif kind == "mazhong_spatial":
            fields["n_band"] = stored["n_band"] == mine["n_band"]
            fields["growth"] = bool(
                abs(stored["growth"] - mine["growth"])
                <= tol * max(1.0, abs(stored["growth"])))
        elif kind == "mach6_dense":
            fields["n_filtered"] = stored["n_filtered"] == mine["n_filtered"]
    identical = all(fields.values())
    decision_keys = [k for k in ("status", "selected", "growth")
                     if k in fields]
    return {"fields": fields, "identical": bool(identical),
            "identical_decision_only": bool(
                all(fields[k] for k in decision_keys)),
            "mine": _json_verdict(mine)}


def _json_verdict(v):
    out = {}
    for k, val in v.items():
        if isinstance(val, (list, tuple)):
            out[k] = [None if x is None else float(x) for x in val]
        elif val is None or isinstance(val, (bool, int, str)):
            out[k] = val
        else:
            out[k] = float(val) if math.isfinite(float(val)) else None
    return out


# ---------------------------------------------------------------------------
# Contour enumeration pipeline (rectangles)
# ---------------------------------------------------------------------------
def _guard_rect(box, guard, factor):
    (x0, x1), (y0, y1) = box["real"], box["imag"]
    left, right, gim = guard["left"], guard["right"], guard["imag"]
    ta = guard.get("target_aspect")
    if ta:
        pad = max(0.5 * ((y1 - y0) / float(ta) - (x1 - x0)), 0.0)
        left, right = max(left, pad), max(right, pad)
    return {"real": [x0 - factor * left, x1 + factor * right],
            "imag": [y0 - factor * gim, y1 + factor * gim]}


def _split_rect(rect, n):
    """Split along the LONGER axis (splitting the short axis of a thin box
    makes the quadrature worse, measured on the Mach-6 family)."""
    (x0, x1), (y0, y1) = rect["real"], rect["imag"]
    if (x1 - x0) >= (y1 - y0):
        xs = np.linspace(x0, x1, n + 1)
        return [{"real": [float(xs[i]), float(xs[i + 1])], "imag": [y0, y1]}
                for i in range(n)]
    ys = np.linspace(y0, y1, n + 1)
    return [{"real": [x0, x1], "imag": [float(ys[i]), float(ys[i + 1])]}
            for i in range(n)]


def _in_rect(z, rect, pad=0.0):
    return (rect["real"][0] - pad < z.real < rect["real"][1] + pad
            and rect["imag"][0] - pad < z.imag < rect["imag"][1] + pad)


def _shrunk_count_and_set(values, box, collar):
    inner = {"real": [box["real"][0] + collar, box["real"][1] - collar],
             "imag": [box["imag"][0] + collar, box["imag"][1] - collar]}
    sel = [complex(z) for z in values if _in_rect(z, inner)]
    return len(sel), sel


def contour_rect_spectrum(entry, ops, fam_cfg, variant_name, variant,
                          perturb, seed, want_winding, cfg=CFG):
    """One Beyn enumeration (optionally sub-box split) + polish + certify."""
    ledger = core.CostLedger()
    box = bts.box_for_cell(entry)
    guard = _guard_rect(box, fam_cfg["guard"], perturb)
    n_split = int(variant.get("n_split", 1))
    boxes = _split_rect(guard, n_split) if n_split > 1 else [guard]
    L, K = int(variant["L"]), int(variant["K"])
    collar = float(fam_cfg["collar"])
    nq_edge = int(fam_cfg["nq_edge"])

    rng = np.random.default_rng(seed)
    V = (rng.standard_normal((ops.m, L))
         + 1j * rng.standard_normal((ops.m, L)))
    RHS = ops.rhs(V)
    zc = complex(0.5 * (guard["real"][0] + guard["real"][1]),
                 0.5 * (guard["imag"][0] + guard["imag"][1]))
    dr, dc = ops.scalings(zc)
    factory = ops.factory(dr, dc, ledger)
    cache = {}

    raw_cands, raw_vecs, diags = [], [], []
    queue = list(boxes)
    n_resplit = 0
    while queue:
        rect = queue.pop(0)
        nodes, weights = core.rect_nodes(rect, nq_edge)
        sc = 0.5 * abs(complex(rect["real"][1] - rect["real"][0],
                               rect["imag"][1] - rect["imag"][0]))
        cc = complex(0.5 * (rect["real"][0] + rect["real"][1]),
                     0.5 * (rect["imag"][0] + rect["imag"][1]))
        res = core.beyn_contour(nodes, weights, factory, RHS, L, K, cc, sc,
                                extract_tol=cfg["extract_tol"],
                                strong_tol=cfg["strong_tol"], m=ops.m,
                                reuse_cache=cache)
        clipped = res["rank"] >= res["capacity"] - 1
        if (res["rank_saturated"] or clipped) and \
                n_resplit < cfg["max_resplit_events"]:
            # capacity exhausted (strong saturation) or extraction clipped
            # (leakage walls: c~1 cluster / damped continuum): concentrate
            # capacity by splitting along the long axis
            n_resplit += 1
            queue.extend(_split_rect(rect, 2))
            diags.append({"rect": rect, "rank": res["rank"],
                          "resplit": True})
            continue
        wind = {"winding": None, "certified": False, "skipped": True,
                "n_extra": 0}
        if want_winding and ops.m <= cfg["winding_max_dim"] \
                and res["rank"] <= cfg["winding_max_rank"]:
            budget = cfg["winding_extra_budget_factor"] * len(nodes)
            w, ok, n_extra = core.winding_number(
                nodes, res["phases"],
                lambda z: factory(z).logdet_phase(), budget)
            wind = {"winding": w, "certified": bool(ok), "skipped": False,
                    "n_extra": int(n_extra)}
        nres = [r for r in res["node_solve_residuals"] if math.isfinite(r)]
        diags.append({
            "rect": rect, "rank": int(res["rank"]),
            "strong_rank": int(res["strong_rank"]),
            "rank_gap": (None if not math.isfinite(res["rank_gap"])
                         else float(res["rank_gap"])),
            "rank_saturated": bool(res["rank_saturated"]),
            "capacity": res["capacity"],
            "sv_tail_rel": [float(x) for x in
                            res["singular_values_rel"][-3:]],
            "node_res_max": (max(nres) if nres else None),
            "node_res_med": (float(np.median(nres)) if nres else None),
            "winding": wind,
        })
        for i, z in enumerate(res["values"]):
            if _in_rect(z, rect, pad=collar):
                raw_cands.append(complex(z))
                raw_vecs.append(res["vectors"][:, i])

    certified, pstats = _polish_and_certify(ops, raw_cands, raw_vecs,
                                            ledger, cfg)
    inguard = [c for c in certified if _in_rect(c["value"], guard, pad=0.0)]
    n_int, int_set = _shrunk_count_and_set(
        [c["value"] for c in inguard], box, collar)
    return {
        "ledger": ledger,
        "certified": inguard,
        "diag": diags,
        "polish_stats": pstats,
        "interior_count": n_int,
        "interior_set": int_set,
        "n_raw_candidates": len(raw_cands),
    }


def _polish_and_certify(ops, raw_cands, raw_vecs, ledger, cfg):
    """Polish every screened candidate, certify by FP64 residual, apply the
    ||Bv|| floor (temporal), dedupe.  Returns (certified_list, stats)."""
    out, vals, resids = [], [], []
    n_unconv = n_leak = n_promoted = 0
    target = cfg["polish_residual_target"]

    def stalled(r):
        return (not r["breakdown"]) and (
            r["residual"] > target or not r["converged"])

    for z0, x0 in zip(raw_cands, raw_vecs):
        r = ops.polish(z0, x0, ledger, cfg["polish_steps"],
                       cfg["cert_residual"])
        n_esc = 0
        if stalled(r):
            r = ops.polish(r["value"], r["vector"], ledger,
                           cfg["polish_escalation_steps"],
                           cfg["cert_residual"])
            n_esc = 1
        if stalled(r):
            # design-E promotion rung: FP64 factorization for the stalled
            # few (counted as lu64; fraction reported)
            r = ops.polish(r["value"], r["vector"], ledger, 3,
                           cfg["cert_residual"], use_fp64=True)
            n_promoted += 1
        if not r["converged"]:
            n_unconv += 1
            out.append(None)
            vals.append(complex(z0))
            resids.append(math.inf)
            continue
        if ops.kind == "pencil" and ops.bv_ratio(r["vector"]) < \
                cfg["bv_floor"]:
            n_leak += 1
            out.append(None)
            vals.append(complex(r["value"]))
            resids.append(math.inf)
            continue
        out.append({"value": complex(r["value"]), "vector": r["vector"],
                    "residual": float(r["residual"]),
                    "iterations": int(r["iterations"]), "escalated": n_esc})
        vals.append(complex(r["value"]))
        resids.append(float(r["residual"]))
    good = [i for i, o in enumerate(out) if o is not None]
    stats = {"n_polished": len(raw_cands), "n_unconverged": n_unconv,
             "n_infinite_leakage": n_leak, "n_promoted_fp64": n_promoted,
             "n_certified": 0, "n_dedup_removed": 0}
    if not good:
        return [], stats
    keep = core.dedupe(np.asarray([vals[i] for i in good]),
                       None, [resids[i] for i in good],
                       tol=cfg["dedupe_tol"])
    certified = [out[good[i]] for i in keep]
    stats["n_certified"] = len(certified)
    stats["n_dedup_removed"] = len(good) - len(keep)
    return certified, stats


# ---------------------------------------------------------------------------
# Contour: Ma&Zhong window mechanism (target-centred disk)
# ---------------------------------------------------------------------------
def contour_mazhong_disk(entry, ops, fam_cfg, radius_factor, seed,
                         r_fixed=None, L_fixed=None, cfg=CFG):
    """Window-mechanism contour: target-centred disk, radius ladder driven
    by CERTIFIED candidate counts (rank counting is unreliable in the
    measured dense continuum fields), then 25-nearest certified.

    ``r_fixed``/``L_fixed`` (perturbation runs) skip the ladder and
    enumerate at ``r_fixed * radius_factor`` so the search cost is not
    triple-counted in the stability apparatus.
    """
    ledger = core.CostLedger()
    p = entry["params"]
    om = float(p["omega"])
    a0 = complex(om / float(p["c_guess"]))
    a_lo, a_hi = om / float(p["c_hi"]), om / float(p["c_lo"])
    r_band = max(abs(a0 - a_lo), abs(a0 - a_hi))
    nq = int(fam_cfg["disk_nq"])
    window_n = int(fam_cfg["window_n"])
    dr, dc = ops.scalings(a0)
    factory = ops.factory(dr, dc, ledger)

    def enumerate_raw(r, L, attempt):
        rng = np.random.default_rng(_seed_for(seed, attempt, L))
        V = (rng.standard_normal((ops.m, L))
             + 1j * rng.standard_normal((ops.m, L)))
        rhs = ops.rhs(V)
        nodes, weights = core.circle_nodes(a0, r, nq)
        res = core.beyn_contour(nodes, weights, factory, rhs, L, 1, a0, r,
                                extract_tol=cfg["extract_tol"],
                                strong_tol=cfg["strong_tol"], m=ops.m)
        raw_c, raw_v = [], []
        for i, z in enumerate(res["values"]):
            if abs(z - a0) <= 1.15 * r:
                raw_c.append(complex(z))
                raw_v.append(res["vectors"][:, i])
        return res, raw_c, raw_v

    def polish_raw(r, raw_c, raw_v):
        certified, pstats = _polish_and_certify(ops, raw_c, raw_v, ledger,
                                                cfg)
        cert_in = [c for c in certified if abs(c["value"] - a0) <= r]
        cert_in.sort(key=lambda c: abs(c["value"] - a0))
        return cert_in, pstats

    def enumerate_at(r, L, attempt):
        res, raw_c, raw_v = enumerate_raw(r, L, attempt)
        cert_in, pstats = polish_raw(r, raw_c, raw_v)
        return res, cert_in, pstats

    trials = []
    if r_fixed is None:
        # LARGEST-CLEAN-RADIUS search (measured structure): the production
        # 25-nearest-target window has radius r25 that sits JUST BELOW a dense
        # continuum arc (~25 finite QEP eigenvalues within r25; the field
        # jumps to ~80-110 across the arc, which clips a finite-L Beyn disk).
        # All in-band candidates lie inside r25, so the disk must reach the
        # arc boundary from below.  We bracket [r_lo clean | r_hi clipped]
        # and drive r_lo UP to that boundary; enumerating at the largest clean
        # radius provably contains the full window (hence every band member).
        L = int(fam_cfg["disk_L_max"])
        r = r_band
        r_lo = r_hi = None
        best_clean = None       # (r, L, res, cert_in, pstats) at largest r_lo
        last_clipped = None
        for attempt in range(8):
            res, raw_c, raw_v = enumerate_raw(r, L, attempt)
            clipped = res["rank"] >= res["capacity"] - 2
            if clipped:
                trials.append({"radius": float(r), "L": int(L),
                               "n_certified": None, "rank": int(res["rank"]),
                               "strong_rank": int(res["strong_rank"]),
                               "clipped": True})
                last_clipped = (r, L, res, raw_c, raw_v)
                r_hi = r
                r = math.sqrt(r_lo * r_hi) if r_lo else 0.8 * r
            else:
                cert_in, pstats = polish_raw(r, raw_c, raw_v)
                trials.append({"radius": float(r), "L": int(L),
                               "n_certified": len(cert_in),
                               "rank": int(res["rank"]),
                               "strong_rank": int(res["strong_rank"]),
                               "clipped": False,
                               "n_unconverged": pstats["n_unconverged"]})
                if r_lo is None or r > r_lo:
                    r_lo = r
                    best_clean = (r, L, res, cert_in, pstats)
                r = math.sqrt(r_lo * r_hi) if r_hi else 1.25 * r
            if r_lo is not None and r_hi is not None and r_hi / r_lo < 1.04:
                break
        if best_clean is None:
            # never found a clean radius: enumerate the smallest clipped disk
            # (honest partial window; flagged via window_complete downstream)
            r, L, res, raw_c, raw_v = last_clipped
            cert_in, pstats = polish_raw(r, raw_c, raw_v)
            best_clean = (r, L, res, cert_in, pstats)
        r_ref, L_use, res, certified, pstats = best_clean
        r_final = r_ref
    else:
        r_ref = float(r_fixed)
        L_use = int(L_fixed if L_fixed else fam_cfg["disk_L"])
        r_final = r_ref * radius_factor
        res, certified, pstats = enumerate_at(r_final, L_use, 0)
        trials.append({"radius": float(r_final), "L": int(L_use),
                       "n_certified": len(certified),
                       "strong_rank": int(res["strong_rank"]),
                       "saturated": bool(res["rank_saturated"])})

    wind = {"winding": None, "certified": False, "skipped": True,
            "n_extra": 0}
    if ops.m <= cfg["winding_max_dim"] and \
            len(certified) <= cfg["winding_max_rank"]:
        nodes, _w = core.circle_nodes(a0, r_final, nq)
        budget = cfg["winding_extra_budget_factor"] * nq
        w, ok, n_extra = core.winding_number(
            nodes, res["phases"], lambda z: factory(z).logdet_phase(),
            budget)
        wind = {"winding": w, "certified": bool(ok), "skipped": False,
                "n_extra": int(n_extra)}

    window = certified[:window_n]
    nres = [x for x in res["node_solve_residuals"] if math.isfinite(x)]
    # Stability quantity (criterion iv) for the window mechanism = the
    # certified BAND-MEMBER set drawn from the 25-nearest window -- the
    # verdict-relevant candidates, the honest analog of the rectangle
    # families' in-box count.  Band members lie within r25 (< 0.8 r_ref, well
    # inside every +-20% perturbed disk), so their set is invariant; a raw
    # geometric interior count is not, because a +20% disk grazes the dense
    # continuum arc that sits just past r25.
    win_vals = np.asarray([c["value"] for c in window], dtype=complex)
    band = (mazhong_band_mask(win_vals, p) if win_vals.size
            else np.zeros(0, dtype=bool))
    interior = [complex(z) for z in win_vals[band]] if win_vals.size else []
    return {
        "ledger": ledger,
        "certified": certified,
        "window": window,
        "window_complete": len(certified) >= window_n,
        "radius": float(r_final),
        "radius_ref": float(r_ref),
        "L_used": int(L_use),
        "trials": trials,
        "polish_stats": pstats,
        "diag": [{"rank": int(res["rank"]),
                  "strong_rank": int(res["strong_rank"]),
                  "rank_gap": (None if not math.isfinite(res["rank_gap"])
                               else float(res["rank_gap"])),
                  "rank_saturated": bool(res["rank_saturated"]),
                  "node_res_max": (max(nres) if nres else None),
                  "node_res_med": (float(np.median(nres)) if nres else None),
                  "winding": wind}],
        "interior_set": [complex(z) for z in interior],
        "interior_count": len(interior),
    }


# ---------------------------------------------------------------------------
# Recall scoring
# ---------------------------------------------------------------------------
def score_recall(found_per_spectrum, truth_per_spectrum, tol):
    rows, n_ok, n_tot = [], 0, 0
    for i, truth in enumerate(truth_per_spectrum):
        found = found_per_spectrum[i] if i < len(found_per_spectrum) else []
        m = core.match_sets(found, truth, tol)
        for r in m:
            r["spectrum"] = i
        rows.extend(m)
        n_ok += sum(1 for r in m if r["recalled"])
        n_tot += len(m)
    return {"n_truth": n_tot, "n_recalled": n_ok,
            "recall": (1.0 if n_tot == 0 else n_ok / n_tot),
            "rows": rows}


def spurious_check(entry, certified_values_per_spectrum, inbox_per_spectrum,
                   tol):
    """Certified in-box candidates matching NO in-box QZ eigenvalue."""
    bad = []
    for i, vals in enumerate(certified_values_per_spectrum):
        vals = np.asarray(vals, dtype=complex)
        if vals.size == 0:
            continue
        mask = bts.in_box_mask(entry, vals)
        ref = np.asarray(inbox_per_spectrum[i], dtype=complex)
        for z in vals[mask]:
            d = float(np.min(np.abs(ref - z))) if ref.size else math.inf
            if d > tol * max(1.0, abs(z)):
                bad.append({"spectrum": i, "value": _cpair(z),
                            "distance_to_qz": d})
    return bad
