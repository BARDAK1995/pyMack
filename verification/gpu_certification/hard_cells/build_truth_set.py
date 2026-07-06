"""Build the pyMack-GPU hard-cell truth corpus (spec slice 01).

Stratified corpus (53 cells across 11 strata) with full-spectrum QZ truth,
production-filter verdicts, and a candidate-set census.  Read-only against
``pymack/`` and the existing verification drivers; writes only inside this
directory.

Regenerability: the committed ``truth_manifest.json`` is the frozen cell
list.  By default the builder reconstructs every cell from the manifest's
embedded params, so a clean checkout can rebuild (or census-rerun) without
the gitignored selection CSVs.  Live reselection from the CSVs requires
``--force-reselect`` and refuses to proceed if any hc_### params would
change relative to the manifest.

Verdict bases: each manifest cell carries a ``verdict_basis`` field.  Most
families judge the full filtered QZ spectrum; the Ma&Zhong family judges
the production shift-invert candidate set (25 modes nearest the target), so
a box can honestly contain in-box QZ eigenvalues while the production
verdict is "no discrete mode" (hc_043; flagged
``box_content_contradicts_verdict``).

Design rules (enforced here, audited by ``test_truth_set.py``):

* Verdicts are computed through the EXISTING production filter code:
  - Ozgen pairs   : ``discrete_mode._decaying_candidates`` + the two-domain
    match step transcribed from ``discrete_mode.discrete_mode`` and
    cross-checked verbatim against ``discrete_mode.discrete_mode`` per cell
    (bitwise, same process, same profile cache).
  - Mack fig10.4  : the band filter transcribed from
    ``compute_mack_fig10_4.first_mode_growth`` and cross-checked verbatim
    against that function per cell.
  - Ma&Zhong      : the phase-band filter transcribed from
    ``trace_mazhong_curves.growth`` applied to the production
    ``solve_spatial`` candidate set, cross-checked verbatim against
    ``trace_mazhong_curves.growth`` per cell.
  - Mach-6 dense  : ``pymack.dense.candidate_indices`` + ``_select_seed``
    called directly (no transcription), fingerprinted against the production
    growth CSV (``n_filtered_candidates`` equality + tracked-alpha
    containment in the stored spectrum).
* The Ozgen 2-D pencil (A, B) is captured by intercepting the production
  solver's ``linalg.eig`` call at runtime -- no re-implemented assembly.
* NPZ files (gitignored) carry: the full raw QZ spectrum, eigenvectors for
  in-box + near-box candidates only, filter outcomes, verdict, kappa audit,
  and the cross-check record.  ``truth_manifest.json`` + ``census.json`` are
  the committed artifacts.
* Census per cell: exact-production-filter in-box counts, distance-to-box-
  boundary distribution, and kappa_2 of the relevant pencil at 16
  Gauss-Legendre nodes on the inflated box boundary, raw and after two-sided
  power-of-2 equilibration.  Pencils: temporal families use (A - c B);
  spatial families use the quadratic T(alpha) (the object a contour engine
  would factor).

Usage
-----
    python verification/gpu_certification/hard_cells/build_truth_set.py \
        --census-gate "l_max<=48,node_kappa_eq_p95<=1e7"
"""
from __future__ import annotations

# Single-thread BLAS before numpy import (workers are process-parallel).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_DIR = HERE / "data"
MANIFEST_PATH = HERE / "truth_manifest.json"
CENSUS_PATH = HERE / "census.json"

_OZGEN_DIR = REPO / "verification" / "mixed_mode" / "ozgen_fig3" / "_refdigitize"
_MAZHONG_DIR = REPO / "verification" / "second_mode" / "mazhong2003_m4p5"
_MACH6_CSV = (
    REPO / "chapters" / "ozgen_kircali_2008" / "results"
    / "aps_dimensional_production" / "spatial_fixed_frequency_growth_curves.csv"
)
for _p in (str(REPO), str(_OZGEN_DIR), str(_MAZHONG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import discrete_mode as dm  # noqa: E402  (Ozgen production extractor)
import trace_mazhong_curves as mztr  # noqa: E402  (Ma&Zhong production bands)
import compute_mazhong_m4p5 as mzc  # noqa: E402  (Ma&Zhong case constants)
import verification.compute_mack_fig10_4 as mk  # noqa: E402  (fig10.4 driver)

import pymack.temporal_solver as pts  # noqa: E402
from pymack.dense import (  # noqa: E402
    DenseBaseFlowConfig,
    DenseGasModel,
    DenseLSTConfig,
    _select_seed,
    candidate_indices,
    omega_from_frequency,
    prepare_dense_case,
    quadratic_matrices,
    solve_spatial_evp,
)
from pymack.scales import delta_star_over_lstar  # noqa: E402
from pymack.solver import (  # noqa: E402
    _assemble_spatial_qep,
    _spatial_companion_matrices,
    apply_dirichlet_freestream_bc_3d,
    apply_wall_bc_3d,
    assemble_temporal_compressible_3d_evp,
    solve_spatial,
)

# ---------------------------------------------------------------------------
# Frozen per-family constants (mirrors of the production drivers, recorded in
# every cell's params so the NPZ is self-describing; cross-checked at build).
# ---------------------------------------------------------------------------
OZGEN_FIRST = {
    "N": 200,                     # build_firstmode_grid.py
    "cr_band": [0.05, 0.97],      # discrete_mode.py defaults
    "ci_abs_max": 0.05,
    "fs_thresh": 0.06,
    "match_tol": 4.0e-3,
}
OZGEN_SECOND = {
    "N": 180,                     # build_secondmode_grid.py
    "cr_band": [0.4, 0.99],
    "ci_abs_max": 0.05,
    "fs_thresh": 0.06,
    "match_tol": 4.0e-3,
}
OZGEN_YMF_FIRST = {2: (35.0, 45.0), 3: (35.0, 45.0), 4: (35.0, 45.0),
                   6: (35.0, 45.0), 7: (28.0, 37.0), 8: (23.0, 31.0),
                   10: (17.0, 22.0)}      # build_firstmode_grid.YMF_BY_MACH
OZGEN_YMF_SECOND = (8.0, 12.0)            # build_secondmode_grid.YMF

MACK_M10 = {
    "Ma": 10.0,
    "N": int(mk.N_BY_MACH[10.0]),
    "y_max": float(mk.Y_MAX_BY_MACH[10.0]),
    "Pr": float(mk.PR),
    "gamma": float(mk.GAMMA),
    "cr_band": [float(mk.CR_LO), float(mk.CR_HI)],
    "ci_cap": float(mk.CI_CAP),
    "phys_cr": [-0.5, 1.5],
    "phys_ci_abs": 0.5,
    "lambda_mu_ratio": 0.0,
}
# Mode-death strip: fixed wave angle inside the production PSI grid (42:3:63),
# alpha sweeping through the documented collapse (~0.030 peak, dead by ~0.040+).
MACK_STRIP_PSI = 57.0
MACK_STRIP_ALPHAS = [0.010, 0.015, 0.020, 0.0275, 0.0325, 0.0375, 0.0425, 0.0475]
MACK_STRIP_R = 1500.0

MAZHONG = {
    "Ma": float(mzc.MA),
    "Pr": float(mzc.PR),
    "gamma": float(mzc.GAMMA),
    "N": int(mzc.N),
    "y_max": float(mzc.Y_MAX),
    "wall_bc": str(mzc.WALL_BC),
    "lambda_mu_ratio": float(mzc.LAMBDA_MU),
    "ai_cap": 0.06,               # trace_mazhong_curves.growth
    "n_modes": 25,
}

MACH6_GAS = {"gamma": 1.4, "prandtl": 0.72, "viscosity_law": "power",
             "mu_power": 0.74, "sutherland_S_K": 111.0, "T_edge_K": 52.0}
MACH6_BASE = {"mach_edge": 5.85, "Tw_Te": 300.0 / 52.0, "eta_max": 16.0,
              "eta_nodes": 80, "bvp_tol": 1.0e-4, "adiabatic": False}
MACH6_LST = {"ny": 31, "y_max": 30.0, "c_min": 0.90, "c_max": 0.97,
             "c_target": 0.86, "c_target_half_width": 0.08,
             "max_abs_alpha": 8.0, "max_abs_ai": 0.4, "max_ai_over_ar": 1.0}

# 4 Gauss-Legendre nodes per rectangle edge x 4 edges = 16 boundary nodes
KAPPA_NODES_PER_EDGE = 4
KAPPA_MARGIN_REL = 0.10           # per-axis inflation for the kappa contour
# Eigenvector storage collar: 1% per axis beyond the box.  The verdict only
# ever needs IN-box eigenvectors (the value-box test precedes the fs filter);
# near-box vectors are a diagnostic convenience.  Larger collars drag in the
# dense c~1 vorticity/entropy branch (hundreds of columns) for nothing.
STORAGE_COLLAR_REL = 0.01

# What candidate set the production verdict is computed over, per family kind.
# Slice-02 scoring must read this instead of assuming box content == verdict
# basis: for the shift-invert basis, box emptiness and verdict emptiness are
# distinct facts (see hc_043).
VERDICT_BASIS = {
    "ozgen_pair": "full_spectrum_filtered(two_domain_match+fs_decay)",
    "mack_3d": "full_spectrum_filtered",
    "mazhong_spatial": "prod_candidates(shift_invert n_modes=25)",
    "mach6_dense": "full_spectrum_filtered",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _pair(z) -> list:
    return [float(np.real(z)), float(np.imag(z))]


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _inflate(rect, rel):
    (x0, x1), (y0, y1) = rect["real"], rect["imag"]
    mx, my = rel * (x1 - x0), rel * (y1 - y0)
    return {"real": [x0 - mx, x1 + mx], "imag": [y0 - my, y1 + my]}


def _in_rect(values, rect):
    v = np.asarray(values, dtype=complex)
    return (np.isfinite(v)
            & (v.real > rect["real"][0]) & (v.real < rect["real"][1])
            & (v.imag > rect["imag"][0]) & (v.imag < rect["imag"][1]))


def _distance_to_rect_boundary(values, rect):
    """Distance to the nearest rectangle edge, for values inside the rect."""
    v = np.asarray(values, dtype=complex)
    dx = np.minimum(v.real - rect["real"][0], rect["real"][1] - v.real)
    dy = np.minimum(v.imag - rect["imag"][0], rect["imag"][1] - v.imag)
    return np.minimum(dx, dy)


def _boundary_nodes(rect, n_per_edge=KAPPA_NODES_PER_EDGE):
    """Gauss-Legendre nodes per rectangle edge (never across corners)."""
    (x0, x1), (y0, y1) = rect["real"], rect["imag"]
    xi, _w = np.polynomial.legendre.leggauss(n_per_edge)
    xs = 0.5 * (x1 - x0) * xi + 0.5 * (x1 + x0)
    ys = 0.5 * (y1 - y0) * xi + 0.5 * (y1 + y0)
    nodes = [complex(x, y0) for x in xs] + [complex(x, y1) for x in xs]
    nodes += [complex(x0, y) for y in ys] + [complex(x1, y) for y in ys]
    return nodes


def _equilibrate_pow2(M):
    """Two-sided power-of-2 equilibration (rows then columns, max-abs)."""
    out = np.array(M, copy=True)
    rn = np.max(np.abs(out), axis=1)
    rs = np.ones_like(rn)
    nz = rn > 0.0
    rs[nz] = 2.0 ** (-np.round(np.log2(rn[nz])))
    out *= rs[:, None]
    cn = np.max(np.abs(out), axis=0)
    cs = np.ones_like(cn)
    nz = cn > 0.0
    cs[nz] = 2.0 ** (-np.round(np.log2(cn[nz])))
    out *= cs[None, :]
    return out


def _kappa2(M):
    try:
        sv = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return math.inf
    if sv[-1] == 0.0:
        return math.inf
    return float(sv[0] / sv[-1])


def _kappa_audit(eval_matrix, rect, margin_rel=KAPPA_MARGIN_REL):
    """kappa_2 at 16 GL nodes on the inflated box boundary, raw + equilibrated.

    ``eval_matrix(z)`` returns the pencil evaluated at z.
    """
    nodes = _boundary_nodes(_inflate(rect, margin_rel))
    raw, eq = [], []
    for z in nodes:
        M = eval_matrix(z)
        raw.append(_kappa2(M))
        eq.append(_kappa2(_equilibrate_pow2(M)))
    return {"nodes": [_pair(z) for z in nodes],
            "margin_rel": float(margin_rel),
            "raw": raw, "equilibrated": eq}


class _EigCapture:
    """Runtime interception of a module's ``linalg`` so the exact production
    pencil (A, B) and the raw QZ output are captured without re-implementing
    any assembly."""

    def __init__(self, real_linalg):
        self._real = real_linalg
        self.calls = []

    def eig(self, A, B=None, **kw):
        out = self._real.eig(A, B, **kw)
        self.calls.append({"A": A, "B": B, "out": out})
        return out

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Box definitions (exact production value-filters + bounding rectangles)
# ---------------------------------------------------------------------------
def box_for_cell(cell):
    """Bounding rectangle of the production verdict region (contour target)."""
    p = cell["params"]
    kind = cell["kind"]
    if kind == "ozgen_pair":
        return {"plane": "c",
                "real": [float(p["cr_band"][0]), float(p["cr_band"][1])],
                "imag": [-float(p["ci_abs_max"]), float(p["ci_abs_max"])]}
    if kind == "mack_3d":
        return {"plane": "c",
                "real": [float(p["cr_band"][0]), float(p["cr_band"][1])],
                "imag": [-float(p["phys_ci_abs"]), float(p["ci_cap"])]}
    if kind == "mazhong_spatial":
        om = float(p["omega"])
        return {"plane": "alpha",
                "real": [om / float(p["c_hi"]), om / float(p["c_lo"])],
                "imag": [-float(p["ai_cap"]), float(p["ai_cap"])]}
    if kind == "mach6_dense":
        om = float(p["omega_L"])
        lst = p["lst_cfg"]
        ar_lo, ar_hi = om / float(lst["c_max"]), om / float(lst["c_min"])
        ai_cap = min(float(lst["max_abs_ai"]),
                     float(lst["max_ai_over_ar"]) * ar_hi)
        return {"plane": "alpha", "real": [ar_lo, ar_hi],
                "imag": [-ai_cap, ai_cap],
                "note": "bounding rectangle of the candidate_indices region"}
    raise ValueError(f"unknown kind {kind}")


def in_box_mask(cell, values):
    """EXACT production value-filter membership per family."""
    p = cell["params"]
    kind = cell["kind"]
    v = np.asarray(values, dtype=complex)
    fin = np.isfinite(v)
    if kind == "ozgen_pair":
        lo, hi = p["cr_band"]
        return fin & (np.abs(v.imag) < float(p["ci_abs_max"])) \
            & (v.real > float(lo)) & (v.real < float(hi))
    if kind == "mack_3d":
        lo, hi = p["cr_band"]
        return (fin & (v.real > float(lo)) & (v.real < float(hi))
                & (v.imag < float(p["ci_cap"]))
                & (np.abs(v.imag) < float(p["phys_ci_abs"])))
    if kind == "mazhong_spatial":
        om = float(p["omega"])
        with np.errstate(divide="ignore", invalid="ignore"):
            c = om / v.real
        return (fin & (c > float(p["c_lo"])) & (c < float(p["c_hi"]))
                & (np.abs(v.imag) < float(p["ai_cap"])) & (v.real > 0.0))
    if kind == "mach6_dense":
        cfg = DenseLSTConfig(**p["lst_cfg"])
        mask = np.zeros(v.shape, dtype=bool)
        mask[candidate_indices(v, float(p["omega_L"]), cfg)] = True
        return mask
    raise ValueError(f"unknown kind {kind}")


# ---------------------------------------------------------------------------
# Verdict derivation (shared verbatim by build and test)
# ---------------------------------------------------------------------------
def derive_ozgen_verdict(spectra, params):
    """Two-domain discrete-mode verdict.

    The per-mode filter is the EXISTING ``discrete_mode._decaying_candidates``;
    the match step transcribes ``discrete_mode.discrete_mode`` (cross-checked
    verbatim against it at build time).
    """
    cands = []
    for spec in spectra:
        ev = np.asarray(spec["prod_eigenvalues"], dtype=complex)
        idx = np.asarray(spec["candidate_indices"], dtype=int)
        vec = np.asarray(spec["candidate_vectors"], dtype=complex)
        cands.append(dm._decaying_candidates(
            ev[idx], vec, np.asarray(spec["y"], dtype=float),
            ci_abs_max=float(params["ci_abs_max"]),
            cr_band=tuple(params["cr_band"]),
            fs_thresh=float(params["fs_thresh"]),
        ))
    n_decaying = [len(c) for c in cands]
    a, b = cands
    if not a or not b:
        return {"status": "no_discrete_mode", "selected": None,
                "n_decaying": n_decaying, "n_match": 0,
                "min_match_distance": None}
    bc = np.array([c for c, _fs in b])
    matched = []
    min_dist = math.inf
    for c, fs in a:
        d = float(np.min(np.abs(c - bc)))
        min_dist = min(min_dist, d)
        if d < float(params["match_tol"]):
            matched.append((c, fs))
    if not matched:
        return {"status": "no_discrete_mode", "selected": None,
                "n_decaying": n_decaying, "n_match": 0,
                "min_match_distance": min_dist}
    matched.sort(key=lambda r: -r[0].imag)
    c, fs = matched[0]
    return {"status": "discrete_mode", "selected": _pair(c),
            "selected_fs": float(fs), "n_decaying": n_decaying,
            "n_match": len(matched), "min_match_distance": min_dist}


def derive_mack_verdict(raw_values, params):
    """Band-filter argmax verdict, transcribed from
    ``compute_mack_fig10_4.first_mode_growth`` (cross-checked verbatim)."""
    c = np.asarray(raw_values, dtype=complex)
    c = c[np.isfinite(c)]
    phys = ((c.real > float(params["phys_cr"][0]))
            & (c.real < float(params["phys_cr"][1]))
            & (np.abs(c.imag) < float(params["phys_ci_abs"])))
    c = c[phys]
    if c.size == 0:
        return {"status": "no_discrete_mode", "selected": None, "n_band": 0}
    lo, hi = params["cr_band"]
    band = (c.real > float(lo)) & (c.real < float(hi)) \
        & (c.imag < float(params["ci_cap"]))
    if not np.any(band):
        return {"status": "no_discrete_mode", "selected": None, "n_band": 0}
    cb = c[band]
    best = cb[int(np.argmax(cb.imag))]
    return {"status": "discrete_mode", "selected": _pair(best),
            "omega_i": float(params["alpha"]) * float(best.imag),
            "n_band": int(np.count_nonzero(band))}


def derive_mazhong_verdict(prod_candidates, params):
    """Phase-band argmin(alpha_i) verdict, transcribed from
    ``trace_mazhong_curves.growth`` (cross-checked verbatim); operates on the
    production ``solve_spatial`` candidate set exactly as the driver does."""
    alphas = np.asarray(prod_candidates, dtype=complex)
    if alphas.size == 0:
        return {"status": "no_discrete_mode", "selected": None, "n_band": 0}
    om = float(params["omega"])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = om / alphas.real
    m = ((c > float(params["c_lo"])) & (c < float(params["c_hi"]))
         & (np.abs(alphas.imag) < float(params["ai_cap"]))
         & (alphas.real > 0.0))
    cand = alphas[m]
    if cand.size == 0:
        return {"status": "no_discrete_mode", "selected": None, "n_band": 0}
    best = cand[int(np.argmin(cand.imag))]
    return {"status": "discrete_mode", "selected": _pair(best),
            "growth": float(-best.imag),
            "phase_speed": float(om / best.real),
            "n_band": int(np.count_nonzero(m))}


def derive_mach6_verdict(raw_values, params):
    """Existing production filter + seed selection from ``pymack.dense``
    (``candidate_indices`` + ``_select_seed``), called directly."""
    cfg = DenseLSTConfig(**params["lst_cfg"])
    vals = np.asarray(raw_values, dtype=complex)
    om = float(params["omega_L"])
    idx = candidate_indices(vals, om, cfg)
    if idx.size == 0:
        return {"status": "no_discrete_mode", "selected": None,
                "n_filtered": 0}
    seed = _select_seed(vals, om, cfg)
    best = vals[int(seed)]
    return {"status": "discrete_mode", "selected": _pair(best),
            "growth": float(-best.imag),
            "phase_speed": float(om / best.real),
            "n_filtered": int(idx.size)}


def derive_verdict(cell, spectra):
    kind = cell["kind"]
    if kind == "ozgen_pair":
        return derive_ozgen_verdict(spectra, cell["params"])
    if kind == "mack_3d":
        return derive_mack_verdict(spectra[0]["raw_eigenvalues"], cell["params"])
    if kind == "mazhong_spatial":
        return derive_mazhong_verdict(spectra[0]["prod_candidates"],
                                      cell["params"])
    if kind == "mach6_dense":
        return derive_mach6_verdict(spectra[0]["raw_eigenvalues"],
                                    cell["params"])
    raise ValueError(f"unknown kind {kind}")


# ---------------------------------------------------------------------------
# Per-family solvers (full QZ truth + kappa + verbatim cross-check)
# ---------------------------------------------------------------------------
_MACK_PROFILE_CACHE = {}
_MAZHONG_PROFILE_CACHE = {}
_MACH6_CASE_CACHE = {}


def _storage_indices(cell, values):
    """Indices whose eigenvectors are stored: in-box + near-box (inflated
    bounding rectangle), always a superset of the exact in-box set."""
    rect = _inflate(box_for_cell(cell), STORAGE_COLLAR_REL)
    near = _in_rect(values, rect)
    near |= in_box_mask(cell, values)
    return np.flatnonzero(near)


def _solve_ozgen(cell):
    p = cell["params"]
    prof = dm._profile(float(p["Ma"]))
    dstar = float(delta_star_over_lstar(prof))
    spectra, kappa = [], []
    for ymf in p["ymf_pair"]:
        cap = _EigCapture(sla)
        old = pts.linalg
        pts.linalg = cap
        try:
            ev, vec, y = pts.solve_temporal_2d(
                prof, float(p["alpha"]), float(p["Re"]), float(p["Ma"]),
                N=int(p["N"]), y_max=float(ymf) * dstar, length_scale="L_star",
            )
        finally:
            pts.linalg = old
        if len(cap.calls) != 1:
            raise RuntimeError(
                f"{cell['id']}: expected 1 eig call, got {len(cap.calls)}")
        A, B = cap.calls[0]["A"], cap.calls[0]["B"]
        raw_vals = np.asarray(cap.calls[0]["out"][0], dtype=complex)
        idx = _storage_indices(cell, ev)
        spectra.append({
            "name": f"ymf_{float(ymf):g}",
            "raw_eigenvalues": raw_vals,
            "prod_eigenvalues": np.asarray(ev, dtype=complex),
            "candidate_indices": idx,
            "candidate_vectors": np.asarray(vec[:, idx], dtype=complex),
            "y": np.asarray(y, dtype=float),
        })
        kap = _kappa_audit(lambda z: A - z * B, box_for_cell(cell))
        kap.update({"spectrum": f"ymf_{float(ymf):g}", "pencil": "A - c B",
                    "dim": int(A.shape[0])})
        kappa.append(kap)
    verdict = derive_verdict(cell, spectra)

    ref = dm.discrete_mode(
        float(p["Ma"]), float(p["Re"]), float(p["alpha"]), N=int(p["N"]),
        ymf_pair=tuple(float(v) for v in p["ymf_pair"]),
        ci_abs_max=float(p["ci_abs_max"]), cr_band=tuple(p["cr_band"]),
        fs_thresh=float(p["fs_thresh"]), match_tol=float(p["match_tol"]),
    )
    if ref is None:
        ok = verdict["status"] == "no_discrete_mode"
        detail = {"ref": None}
    else:
        sel = verdict.get("selected")
        ok = (verdict["status"] == "discrete_mode" and sel is not None
              and abs(complex(*sel) - complex(ref["c_r"], ref["c_i"])) <= 1e-12
              and abs(verdict["selected_fs"] - ref["fs"]) <= 1e-12
              and verdict["n_match"] == ref["n_match"])
        detail = {"ref": ref}
    crosscheck = {"reference": "discrete_mode.discrete_mode (verbatim)",
                  "match": bool(ok), **detail}
    if not ok:
        raise RuntimeError(f"{cell['id']}: verdict != discrete_mode() "
                           f"({verdict} vs {ref})")
    return spectra, verdict, kappa, crosscheck


def _solve_mack(cell):
    p = cell["params"]
    key = round(float(p["Ma"]), 4)
    if key not in _MACK_PROFILE_CACHE:
        _MACK_PROFILE_CACHE[key] = mk.make_profile(key)
    prof = _MACK_PROFILE_CACHE[key]
    A, B, y, D1, n, _al, _be, _bf = assemble_temporal_compressible_3d_evp(
        prof, float(p["alpha"]), float(p["beta"]), float(p["R"]),
        float(p["Ma"]), float(p["Pr"]), float(p["gamma"]),
        N=int(p["N"]), y_max=float(p["y_max"]), length_scale="L_star",
        lambda_mu_ratio=float(p["lambda_mu_ratio"]),
    )
    apply_wall_bc_3d(A, B, D1, n)
    apply_dirichlet_freestream_bc_3d(A, B, n)
    raw_vals, raw_vecs = sla.eig(A.copy(), B.copy(), check_finite=False)
    raw_vals = np.asarray(raw_vals, dtype=complex)
    idx = _storage_indices(cell, raw_vals)
    spectra = [{
        "name": "m10_3d",
        "raw_eigenvalues": raw_vals,
        "candidate_indices": idx,
        "candidate_vectors": np.asarray(raw_vecs[:, idx], dtype=complex),
        "y": np.asarray(y, dtype=float),
    }]
    verdict = derive_verdict(cell, spectra)
    kap = _kappa_audit(lambda z: A - z * B, box_for_cell(cell))
    kap.update({"spectrum": "m10_3d", "pencil": "A - c B",
                "dim": int(A.shape[0])})

    oi, cref = mk.first_mode_growth(
        prof, float(p["alpha"]), float(p["beta"]), float(p["R"]),
        float(p["Ma"]), N=int(p["N"]), y_max=float(p["y_max"]),
    )
    if cref is None:
        ok = verdict["status"] == "no_discrete_mode"
        detail = {"ref": None}
        dist = None
    else:
        sel = verdict.get("selected")
        dist = None if sel is None else abs(complex(*sel) - cref)
        ok = (verdict["status"] == "discrete_mode" and dist is not None
              and dist <= 1e-9)
        detail = {"ref": {"omega_i": float(oi), "c": _pair(cref)}}
    crosscheck = {
        "reference": "compute_mack_fig10_4.first_mode_growth (verbatim)",
        "match": bool(ok),
        "selected_c_distance": None if dist is None else float(dist),
        **detail}
    if not ok:
        raise RuntimeError(f"{cell['id']}: verdict != first_mode_growth() "
                           f"({verdict} vs {detail})")
    return spectra, verdict, [kap], crosscheck


def _solve_mazhong(cell):
    p = cell["params"]
    if "profile" not in _MAZHONG_PROFILE_CACHE:
        _MAZHONG_PROFILE_CACHE["profile"] = mzc.build_profile()
    prof = _MAZHONG_PROFILE_CACHE["profile"]
    om, R = float(p["omega"]), float(p["R"])

    # Production candidate set (shift-invert around the production target).
    prod_alphas, _prod_modes, _y0 = solve_spatial(
        prof, om, R, float(p["Ma"]), float(p["Pr"]), float(p["gamma"]),
        N=int(p["N"]), y_max=float(p["y_max"]), wall_bc=str(p["wall_bc"]),
        target_alpha=om / float(p["c_guess"]) + 0j, n_modes=int(p["n_modes"]),
        length_scale="L_star", lambda_mu_ratio=float(p["lambda_mu_ratio"]),
    )

    # Full-spectrum QZ truth on the companion pencil.
    C0, C1, C2, y = _assemble_spatial_qep(
        prof, om, R, float(p["Ma"]), float(p["Pr"]), float(p["gamma"]),
        int(p["N"]), float(p["y_max"]), None, str(p["wall_bc"]),
        "L_star", float(p["lambda_mu_ratio"]),
    )
    LL, RR = _spatial_companion_matrices(C0, C1, C2)
    raw_vals, raw_vecs = sla.eig(LL, RR, check_finite=False)
    raw_vals = np.asarray(raw_vals, dtype=complex)
    nn = C0.shape[0]
    idx = _storage_indices(cell, raw_vals)
    spectra = [{
        "name": f"mazhong_{p['mode']}",
        "raw_eigenvalues": raw_vals,
        "prod_candidates": np.asarray(prod_alphas, dtype=complex),
        "candidate_indices": idx,
        "candidate_vectors": np.asarray(raw_vecs[nn:, idx], dtype=complex),
        "y": np.asarray(y, dtype=float),
    }]
    verdict = derive_verdict(cell, spectra)
    kap = _kappa_audit(lambda z: C0 + z * C1 + (z * z) * C2,
                       box_for_cell(cell))
    kap.update({"spectrum": f"mazhong_{p['mode']}",
                "pencil": "T(alpha) = C0 + alpha C1 + alpha^2 C2",
                "dim": int(C0.shape[0])})

    ref = mztr.growth(prof, R, om, float(p["c_lo"]), float(p["c_hi"]))
    mine = verdict.get("growth", math.nan) \
        if verdict["status"] == "discrete_mode" else math.nan
    both_nan = (not math.isfinite(mine)) and (not np.isfinite(ref))
    ok = both_nan or (math.isfinite(mine) and np.isfinite(ref)
                      and abs(mine - float(ref)) <= 1e-12)
    containment = None
    if verdict["status"] == "discrete_mode":
        fin = raw_vals[np.isfinite(raw_vals)]
        containment = float(np.min(np.abs(fin - complex(*verdict["selected"]))))
        ok = ok and containment <= 1e-6
    crosscheck = {"reference": "trace_mazhong_curves.growth (verbatim)",
                  "match": bool(ok),
                  "ref_growth": _finite_or_none(ref),
                  "selected_to_full_spectrum_distance": containment}
    if not ok:
        raise RuntimeError(f"{cell['id']}: verdict != growth() "
                           f"({mine} vs {ref}, containment={containment})")
    return spectra, verdict, [kap], crosscheck


def _solve_mach6(cell):
    p = cell["params"]
    key = json.dumps([p["gas"], p["base_cfg"], p["lst_cfg"]], sort_keys=True)
    if key not in _MACH6_CASE_CACHE:
        gas = DenseGasModel(**p["gas"])
        base_cfg = DenseBaseFlowConfig(**p["base_cfg"])
        lst_cfg = DenseLSTConfig(**p["lst_cfg"])
        _base, y, D, base_grid = prepare_dense_case(gas, base_cfg, lst_cfg)
        _MACH6_CASE_CACHE[key] = (gas, lst_cfg, y, D, base_grid)
    gas, lst_cfg, y, D, base_grid = _MACH6_CASE_CACHE[key]
    om = omega_from_frequency(float(p["freq_parameter"]), float(p["R_L"]),
                              "mack")
    assert abs(om - float(p["omega_L"])) <= 1e-12 * max(1.0, abs(om))
    raw_vals, raw_vecs = solve_spatial_evp(om, float(p["R_L"]), y, D,
                                           base_grid, gas, lst_cfg)
    raw_vals = np.asarray(raw_vals, dtype=complex)
    idx = _storage_indices(cell, raw_vals)
    spectra = [{
        "name": "mach6_dense_qep",
        "raw_eigenvalues": raw_vals,
        "candidate_indices": idx,
        "candidate_vectors": np.asarray(raw_vecs[:, idx], dtype=complex),
        "y": np.asarray(y, dtype=float),
    }]
    verdict = derive_verdict(cell, spectra)

    # T(alpha) for the kappa audit (same production assembly path).
    A0, A1, A2 = quadratic_matrices(om, float(p["R_L"]), y, D, base_grid,
                                    gas, lst_cfg)
    kap = _kappa_audit(lambda z: A0 + z * A1 + (z * z) * A2,
                       box_for_cell(cell))
    kap.update({"spectrum": "mach6_dense_qep",
                "pencil": "T(alpha) = A0 + alpha A1 + alpha^2 A2",
                "dim": int(A0.shape[0])})

    # Fingerprint against the production growth CSV row.
    n_filtered = verdict.get("n_filtered", 0)
    ok = int(n_filtered) == int(p["csv_n_filtered"])
    containment = None
    if p.get("csv_alpha") is not None and math.isfinite(p["csv_alpha"][0]):
        tgt = complex(p["csv_alpha"][0], p["csv_alpha"][1])
        fin = raw_vals[np.isfinite(raw_vals)]
        containment = float(np.min(np.abs(fin - tgt)))
        ok = ok and containment <= 1e-9
    crosscheck = {
        "reference": "aps_dimensional_production growth CSV fingerprint",
        "match": bool(ok),
        "csv_n_filtered": int(p["csv_n_filtered"]),
        "csv_alpha_to_spectrum_distance": containment}
    if not ok:
        raise RuntimeError(f"{cell['id']}: CSV fingerprint mismatch "
                           f"(n_filtered {n_filtered} vs {p['csv_n_filtered']},"
                           f" containment={containment})")
    return spectra, verdict, [kap], crosscheck


_SOLVERS = {"ozgen_pair": _solve_ozgen, "mack_3d": _solve_mack,
            "mazhong_spatial": _solve_mazhong, "mach6_dense": _solve_mach6}


# ---------------------------------------------------------------------------
# Corpus selection (deterministic, from the committed production artifacts)
# ---------------------------------------------------------------------------
def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _ozgen_cell(row, family, stratum, base):
    mach = int(round(float(row["Ma"])))
    if family == "ozgen_first_pair":
        ymf = OZGEN_YMF_FIRST.get(mach, (35.0, 45.0))
    else:
        ymf = OZGEN_YMF_SECOND
    return {
        "family": family, "kind": "ozgen_pair", "stratum": stratum,
        "params": {
            "Ma": float(row["Ma"]), "Re": float(row["Re"]),
            "alpha": float(row["alpha"]), "ymf_pair": [float(v) for v in ymf],
            "length_scale": "L_star", "wall_bc": "isothermal",
            **{k: base[k] for k in
               ("N", "cr_band", "ci_abs_max", "fs_thresh", "match_tol")},
            "grid_resolved": int(float(row["resolved"])),
            "grid_c_i": _finite_or_none(_fnum(row["c_i"])),
            "grid_fs": _finite_or_none(_fnum(row["fs"])),
        },
    }


def _take(rows, count, key, used, per_ma=2):
    out, ma_count = [], {}
    for row in sorted(rows, key=key):
        sig = (row["Ma"], row["Re"], row["alpha"])
        ma = row["Ma"]
        if sig in used or ma_count.get(ma, 0) >= per_ma:
            continue
        used.add(sig)
        ma_count[ma] = ma_count.get(ma, 0) + 1
        out.append(row)
        if len(out) >= count:
            break
    return out


def _resolve_frontier(rows):
    """(resolved, unresolved) row pairs adjacent in alpha at fixed (Ma, Re)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["Ma"], r["Re"]), []).append(r)
    pairs = []
    for grp in groups.values():
        grp.sort(key=lambda r: float(r["alpha"]))
        for a, b in zip(grp[:-1], grp[1:]):
            ra, rb = int(float(a["resolved"])), int(float(b["resolved"]))
            if ra == 1 and rb == 0:
                pairs.append((a, b))
            elif ra == 0 and rb == 1:
                pairs.append((b, a))
    # nearest-to-threshold first: smallest |c_i| on the resolved side
    pairs.sort(key=lambda ab: (abs(_fnum(ab[0]["c_i"]))
                               if math.isfinite(_fnum(ab[0]["c_i"]))
                               else math.inf,
                               ab[0]["Ma"], float(ab[0]["Re"]),
                               float(ab[0]["alpha"])))
    return pairs


def load_cells_from_manifest():
    """Frozen cell list from the committed manifest (no CSVs needed)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [{k: c[k] for k in ("id", "family", "kind", "stratum", "params")}
            for c in manifest["cells"]]


def get_cells(force_reselect=False):
    """Return (cells, source).  Default: the frozen manifest definitions.

    Live reselection reads the (gitignored) production grid CSVs and is
    allowed only with ``force_reselect``; it refuses to proceed if any
    existing hc_### would silently change params.
    """
    if MANIFEST_PATH.exists() and not force_reselect:
        return load_cells_from_manifest(), "manifest"
    cells = select_cells_live()
    if MANIFEST_PATH.exists():
        frozen = {c["id"]: c["params"] for c in load_cells_from_manifest()}
        changed = [
            c["id"] for c in cells
            if c["id"] in frozen
            and json.dumps(c["params"], sort_keys=True)
            != json.dumps(frozen[c["id"]], sort_keys=True)
        ]
        if changed:
            raise RuntimeError(
                "--force-reselect would change params of frozen cells "
                f"{changed}; delete truth_manifest.json and the stale "
                "data/*.npz deliberately if a new corpus is intended.")
    return cells, "live_reselection"


def select_cells_live():
    """Stratified selection from the production grid CSVs (gitignored)."""
    missing = [str(p) for p in
               (_OZGEN_DIR / "firstmode_grid.csv",
                _OZGEN_DIR / "secondmode_grid.csv", _MACH6_CSV)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "live reselection needs the production grid CSVs (gitignored, "
            f"not in a clean checkout): {missing}; the default manifest-"
            "based path does not need them.")
    cells = []

    # --- Ozgen first-mode family -----------------------------------------
    fm = _read_csv(_OZGEN_DIR / "firstmode_grid.csv")
    fm_res = [r for r in fm if int(float(r["resolved"])) == 1]
    used = set()

    for r in _take(fm_res, 6, lambda r: (abs(_fnum(r["c_i"])), r["Ma"],
                                         float(r["Re"]), float(r["alpha"])),
                   used):
        cells.append(_ozgen_cell(r, "ozgen_first_pair", "lobe_edge",
                                 OZGEN_FIRST))
    for r in _take(fm_res, 4,
                   lambda r: (abs(_fnum(r["fs"]) - OZGEN_FIRST["fs_thresh"]),
                              r["Ma"], float(r["Re"]), float(r["alpha"])),
                   used):
        cells.append(_ozgen_cell(r, "ozgen_first_pair", "fs_boundary",
                                 OZGEN_FIRST))

    frontier = _resolve_frontier(fm)
    used_res, used_unres = set(), set()
    n_trans, n_nodisc = 0, 0
    for res_row, unres_row in frontier:
        sig_r = (res_row["Ma"], res_row["Re"], res_row["alpha"])
        sig_u = (unres_row["Ma"], unres_row["Re"], unres_row["alpha"])
        if n_trans < 4 and sig_r not in used and sig_r not in used_res:
            used_res.add(sig_r)
            used.add(sig_r)
            cells.append(_ozgen_cell(res_row, "ozgen_first_pair",
                                     "resolve_transition", OZGEN_FIRST))
            n_trans += 1
        if n_nodisc < 6 and sig_u not in used_unres:
            used_unres.add(sig_u)
            cells.append(_ozgen_cell(unres_row, "ozgen_first_pair",
                                     "no_discrete_frontier", OZGEN_FIRST))
            n_nodisc += 1
        if n_trans >= 4 and n_nodisc >= 6:
            break

    # --- Ozgen second-mode family (all rows resolved in the grid) ---------
    sm = _read_csv(_OZGEN_DIR / "secondmode_grid.csv")
    sm_res = [r for r in sm if int(float(r["resolved"])) == 1]
    used2 = set()
    for r in _take(sm_res, 3, lambda r: (abs(_fnum(r["c_i"])), r["Ma"],
                                         float(r["Re"]), float(r["alpha"])),
                   used2):
        cells.append(_ozgen_cell(r, "ozgen_second_pair", "second_lobe_edge",
                                 OZGEN_SECOND))

    # --- first/second-mode coalescence (M6, alpha window 0.115-0.135) -----
    f6 = sorted((r for r in fm_res if r["Ma"] == "6"
                 and float(r["alpha"]) >= 0.115),
                key=lambda r: (-float(r["alpha"]), float(r["Re"])))
    picks = []
    if f6:
        picks.append(f6[0])
        far = max(f6, key=lambda r: abs(float(r["Re"]) - float(f6[0]["Re"])))
        if far is not f6[0]:
            picks.append(far)
    for r in picks[:2]:
        cells.append(_ozgen_cell(r, "ozgen_first_pair", "coalescence_m6",
                                 OZGEN_FIRST))
    s6 = sorted((r for r in sm_res if r["Ma"] == "6"
                 and float(r["alpha"]) <= 0.135),
                key=lambda r: (float(r["alpha"]), float(r["Re"])))
    picks = []
    if s6:
        picks.append(s6[0])
        far = max(s6, key=lambda r: abs(float(r["Re"]) - float(s6[0]["Re"])))
        if far is not s6[0]:
            picks.append(far)
    for r in picks[:2]:
        cells.append(_ozgen_cell(r, "ozgen_second_pair", "coalescence_m6",
                                 OZGEN_SECOND))

    # --- Mack fig10.4 M10 mode-death alpha strip ---------------------------
    for alpha in MACK_STRIP_ALPHAS:
        cells.append({
            "family": "mack_fig10_4_m10_3d", "kind": "mack_3d",
            "stratum": "m10_mode_death_alpha_strip",
            "params": {
                **MACK_M10,
                "R": MACK_STRIP_R, "alpha": float(alpha),
                "psi_deg": MACK_STRIP_PSI,
                "beta": float(alpha * math.tan(math.radians(MACK_STRIP_PSI))),
            },
        })

    # --- Ma & Zhong band edges + no-candidate cells ------------------------
    mz_rows = _read_csv(_MAZHONG_DIR / "mazhong_curve_grid.csv")
    for mode, count in (("second", 4), ("first", 3)):
        band = mztr.MODES[mode]
        finite = [r for r in mz_rows if r["mode"] == mode
                  and math.isfinite(_fnum(r["neg_alpha_i"]))]
        picked, r_count = [], {}
        for r in sorted(finite, key=lambda r: (abs(_fnum(r["neg_alpha_i"])),
                                               float(r["R"]),
                                               float(r["omega"]))):
            if r_count.get(r["R"], 0) >= 2:
                continue
            r_count[r["R"]] = r_count.get(r["R"], 0) + 1
            picked.append(r)
            if len(picked) >= count:
                break
        for r in picked:
            cells.append({
                "family": f"mazhong_{mode}_spatial", "kind": "mazhong_spatial",
                "stratum": "mazhong_band_edge",
                "params": {
                    **MAZHONG, "mode": mode,
                    "R": float(r["R"]), "omega": float(r["omega"]),
                    "c_lo": float(band["c_lo"]), "c_hi": float(band["c_hi"]),
                    "c_guess": 0.90 if float(band["c_hi"]) > 0.8 else 0.55,
                    "grid_neg_alpha_i":
                        _finite_or_none(_fnum(r["neg_alpha_i"])),
                },
            })
    # no-candidate rows adjacent (same mode, same R, neighboring omega) to a
    # finite row -- honest "no discrete mode" spatial cells.
    empties = []
    by_mode_r = {}
    for r in mz_rows:
        by_mode_r.setdefault((r["mode"], r["R"]), []).append(r)
    for (mode, _R), grp in sorted(by_mode_r.items()):
        grp.sort(key=lambda r: float(r["omega"]))
        for i, r in enumerate(grp):
            if math.isfinite(_fnum(r["neg_alpha_i"])):
                continue
            neigh = [g for j, g in enumerate(grp) if abs(j - i) == 1
                     and math.isfinite(_fnum(g["neg_alpha_i"]))]
            if neigh:
                empties.append(r)
    seen_modes = set()
    for r in sorted(empties, key=lambda r: (r["mode"], float(r["R"]),
                                            float(r["omega"]))):
        if r["mode"] in seen_modes:
            continue
        seen_modes.add(r["mode"])
        band = mztr.MODES[r["mode"]]
        cells.append({
            "family": f"mazhong_{r['mode']}_spatial", "kind": "mazhong_spatial",
            "stratum": "mazhong_no_candidate",
            "params": {
                **MAZHONG, "mode": r["mode"],
                "R": float(r["R"]), "omega": float(r["omega"]),
                "c_lo": float(band["c_lo"]), "c_hi": float(band["c_hi"]),
                "c_guess": 0.90 if float(band["c_hi"]) > 0.8 else 0.55,
                "grid_neg_alpha_i": None,
            },
        })

    # --- Mach-6 eN dimensional production (dense QEP) ----------------------
    m6 = _read_csv(_MACH6_CSV)
    for r in m6:
        r["_sig"] = abs(_fnum(r["sigma_L"]))
        r["_nf"] = int(float(r["n_filtered_candidates"]))
    used6 = set()

    def _mach6_cell(r, stratum):
        key = (r["freq_parameter"], r["R_L"])
        if key in used6:
            return
        used6.add(key)
        ar, ai = _fnum(r["alpha_r_L"]), _fnum(r["alpha_i_L"])
        cells.append({
            "family": "mach6_spatial_dense", "kind": "mach6_dense",
            "stratum": stratum,
            "params": {
                "freq_parameter": float(r["freq_parameter"]),
                "R_L": float(r["R_L"]), "omega_L": float(r["omega_L"]),
                "gas": MACH6_GAS, "base_cfg": MACH6_BASE, "lst_cfg": MACH6_LST,
                "csv_n_filtered": int(r["_nf"]),
                "csv_alpha": ([ar, ai] if math.isfinite(ar) else None),
                "csv_sigma_L": _finite_or_none(_fnum(r["sigma_L"])),
            },
        })

    ok_rows = [r for r in m6 if r["status"] == "ok"
               and math.isfinite(r["_sig"])]
    seen_freq = set()
    n_taken = 0
    for r in sorted(ok_rows, key=lambda r: (r["_sig"], float(r["R_L"]))):
        if r["freq_parameter"] in seen_freq:
            continue
        seen_freq.add(r["freq_parameter"])
        _mach6_cell(r, "en_near_neutral")
        n_taken += 1
        if n_taken >= 6:
            break
    nf_lo = min(r["_nf"] for r in m6)
    nf_hi = max(r["_nf"] for r in m6)
    for nf_val, want in ((nf_lo, 2), (nf_hi, 2)):
        sub = [r for r in m6 if r["_nf"] == nf_val]
        seen_freq = set()
        n_taken = 0
        for r in sorted(sub, key=lambda r: (r["_sig"], float(r["R_L"]))):
            if (r["freq_parameter"] in seen_freq
                    or (r["freq_parameter"], r["R_L"]) in used6):
                continue
            seen_freq.add(r["freq_parameter"])
            _mach6_cell(r, "candidate_count_edge")
            n_taken += 1
            if n_taken >= want:
                break

    for i, cell in enumerate(cells, start=1):
        cell["id"] = f"hc_{i:03d}"
    return cells


# ---------------------------------------------------------------------------
# NPZ round trip
# ---------------------------------------------------------------------------
def save_npz(path: Path, cell, spectra, verdict, kappa, crosscheck):
    payload = {
        "cell_json": json.dumps(cell, sort_keys=True),
        "verdict_json": json.dumps(verdict, sort_keys=True, allow_nan=True),
        "kappa_json": json.dumps(kappa, sort_keys=True, allow_nan=True),
        "crosscheck_json": json.dumps(crosscheck, sort_keys=True,
                                      allow_nan=True),
        "n_spectra": np.int64(len(spectra)),
    }
    for i, s in enumerate(spectra):
        payload[f"name_{i}"] = s["name"]
        payload[f"raw_eigenvalues_{i}"] = np.asarray(s["raw_eigenvalues"],
                                                     dtype=np.complex128)
        payload[f"candidate_indices_{i}"] = np.asarray(s["candidate_indices"],
                                                       dtype=np.int64)
        payload[f"candidate_vectors_{i}"] = np.asarray(s["candidate_vectors"],
                                                       dtype=np.complex128)
        payload[f"y_{i}"] = np.asarray(s["y"], dtype=float)
        if "prod_eigenvalues" in s:
            payload[f"prod_eigenvalues_{i}"] = np.asarray(
                s["prod_eigenvalues"], dtype=np.complex128)
        if "prod_candidates" in s:
            payload[f"prod_candidates_{i}"] = np.asarray(
                s["prod_candidates"], dtype=np.complex128)
    np.savez_compressed(path, **payload)


def load_npz(path: Path):
    z = np.load(path, allow_pickle=False)
    cell = json.loads(str(z["cell_json"]))
    verdict = json.loads(str(z["verdict_json"]))
    kappa = json.loads(str(z["kappa_json"]))
    crosscheck = json.loads(str(z["crosscheck_json"]))
    spectra = []
    for i in range(int(z["n_spectra"])):
        s = {"name": str(z[f"name_{i}"]),
             "raw_eigenvalues": z[f"raw_eigenvalues_{i}"],
             "candidate_indices": z[f"candidate_indices_{i}"],
             "candidate_vectors": z[f"candidate_vectors_{i}"],
             "y": z[f"y_{i}"]}
        if f"prod_eigenvalues_{i}" in z:
            s["prod_eigenvalues"] = z[f"prod_eigenvalues_{i}"]
        if f"prod_candidates_{i}" in z:
            s["prod_candidates"] = z[f"prod_candidates_{i}"]
        spectra.append(s)
    return cell, spectra, verdict, kappa, crosscheck


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------
def census_row_for(cell, spectra, kappa):
    rect = box_for_cell(cell)
    per_spec = []
    for s in spectra:
        vals = np.asarray(s["raw_eigenvalues"], dtype=complex)
        mask = in_box_mask(cell, vals)
        dists = _distance_to_rect_boundary(vals[mask], rect)
        per_spec.append({
            "spectrum": str(s["name"]),
            "n_raw": int(vals.size),
            "n_finite": int(np.count_nonzero(np.isfinite(vals))),
            "in_box_count": int(np.count_nonzero(mask)),
            "distance_to_box_boundary": [float(d) for d in dists],
            "n_stored_vectors": int(np.asarray(s["candidate_indices"]).size),
        })
    raw = [v for k in kappa for v in k["raw"]]
    eq = [v for k in kappa for v in k["equilibrated"]]
    return {
        "id": cell["id"], "family": cell["family"],
        "stratum": cell["stratum"],
        "box": rect, "per_spectrum": per_spec,
        "node_kappa_raw": [_finite_or_none(v) for v in raw],
        "node_kappa_equilibrated": [_finite_or_none(v) for v in eq],
        "kappa_pencil": kappa[0].get("pencil"),
        "kappa_dim": kappa[0].get("dim"),
    }


def _pctl(values, q):
    clean = [float(v) for v in values
             if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def _kappa_stats(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "p50": _pctl(values, 50), "p95": _pctl(values, 95),
        "max_finite": (max(finite) if finite else None),
        "n_nonfinite": int(len(values) - len(finite)),
    }


def recommend(rows):
    """Measured (N_q, L, margin, collar) per family from the census rows."""
    spec_counts = [ps["in_box_count"] for row in rows
                   for ps in row["per_spectrum"]]
    dists = [d for row in rows for ps in row["per_spectrum"]
             for d in ps["distance_to_box_boundary"]]
    eq = [v for row in rows for v in row["node_kappa_equilibrated"]]
    raw = [v for row in rows for v in row["node_kappa_raw"]]
    max_count = max(spec_counts) if spec_counts else 0
    eq_p95 = _pctl(eq, 95)
    eq_p95_v = math.inf if eq_p95 is None else eq_p95
    if max_count <= 8 and eq_p95_v <= 1.0e7:
        nq = 16
    elif max_count <= 24 and eq_p95_v <= 1.0e9:
        nq = 24
    else:
        nq = 32
    L = max(8, 4 * math.ceil((max_count + 4) / 4))
    positive = [d for d in dists if d > 0.0]
    p10 = _pctl(positive, 10)
    collar = float(min(2.0e-2, max(1.0e-5,
                                   p10 if p10 is not None else 5.0e-3)))
    margin = float(max(2.0e-5, 2.0 * collar))
    return {
        "N_q": int(nq), "L": int(L), "margin": margin, "collar": collar,
        "in_box_count_min": int(min(spec_counts)) if spec_counts else 0,
        "in_box_count_median": (float(median(spec_counts))
                                if spec_counts else 0.0),
        "in_box_count_max": int(max_count),
        "distance_p10": p10,
        "node_kappa_raw": _kappa_stats(raw),
        "node_kappa_equilibrated": _kappa_stats(eq),
    }


def npz_arrays_sha256(path: Path) -> str:
    """Deterministic digest over every stored array (name, dtype, shape,
    bytes) -- catches verdict-preserving NPZ corruption."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for name in sorted(z.files):
            a = np.ascontiguousarray(z[name])
            h.update(name.encode("utf-8"))
            h.update(str(a.dtype).encode("utf-8"))
            h.update(str(a.shape).encode("utf-8"))
            h.update(a.tobytes())
    return h.hexdigest()


def _git_provenance():
    def _run(*args):
        return subprocess.run(["git", *args], cwd=str(REPO), timeout=15,
                              capture_output=True, text=True).stdout.strip()
    try:
        sha = _run("rev-parse", "HEAD")
        dirty = bool(_run("status", "--porcelain"))
        return {"git_sha": sha or None, "git_dirty": dirty}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------
def _npz_name(cell):
    return f"{cell['id']}_{cell['family']}.npz"


def solve_one(cell):
    """Worker entry: solve, audit, save NPZ; return JSON-safe summaries."""
    t0 = time.perf_counter()
    spectra, verdict, kappa, crosscheck = _SOLVERS[cell["kind"]](cell)
    save_npz(DATA_DIR / _npz_name(cell), cell, spectra, verdict, kappa,
             crosscheck)
    row = census_row_for(cell, spectra, kappa)
    return {
        "cell": cell, "verdict": verdict, "crosscheck": crosscheck,
        "census_row": row, "runtime_s": float(time.perf_counter() - t0),
    }


def _cell_annotations(cell, verdict, census_row):
    """Reviewer-mandated per-cell manifest annotations."""
    out = {"verdict_basis": VERDICT_BASIS[cell["kind"]]}
    total_in_box = sum(ps["in_box_count"] for ps in census_row["per_spectrum"])
    if cell["kind"] == "mazhong_spatial":
        out["box_content_contradicts_verdict"] = bool(
            verdict["status"] == "no_discrete_mode" and total_in_box > 0)
    if cell["kind"] == "mach6_dense":
        sel = verdict.get("selected")
        csv_alpha = cell["params"].get("csv_alpha")
        out["seed_vs_tracked_alpha_distance"] = (
            float(abs(complex(*sel) - complex(*csv_alpha)))
            if sel is not None and csv_alpha is not None else None)
    return out


def build(force=False, workers=1, census_gate=None, force_reselect=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cells, cell_source = get_cells(force_reselect=force_reselect)
    started = time.perf_counter()

    todo, results = [], {}
    for cell in cells:
        path = DATA_DIR / _npz_name(cell)
        if path.exists() and not force:
            stored_cell, spectra, verdict, kappa, crosscheck = load_npz(path)
            if json.dumps(stored_cell.get("params"), sort_keys=True) != \
                    json.dumps(cell["params"], sort_keys=True):
                todo.append(cell)
                continue
            results[cell["id"]] = {
                "cell": cell, "verdict": verdict, "crosscheck": crosscheck,
                "census_row": census_row_for(cell, spectra, kappa),
                "runtime_s": 0.0,
            }
        else:
            todo.append(cell)

    n_done = len(results)
    print(f"corpus: {len(cells)} cells ({n_done} cached, {len(todo)} to "
          f"solve, workers={workers})", flush=True)
    if todo:
        if workers <= 1:
            for cell in todo:
                res = solve_one(cell)
                results[cell["id"]] = res
                n_done += 1
                print(f"  [{n_done:02d}/{len(cells)}] {cell['id']} "
                      f"{cell['family']}/{cell['stratum']} "
                      f"{res['verdict']['status']} "
                      f"({res['runtime_s']:.1f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(solve_one, cell): cell for cell in todo}
                for fut in as_completed(futs):
                    cell = futs[fut]
                    res = fut.result()
                    results[cell["id"]] = res
                    n_done += 1
                    print(f"  [{n_done:02d}/{len(cells)}] {cell['id']} "
                          f"{cell['family']}/{cell['stratum']} "
                          f"{res['verdict']['status']} "
                          f"({res['runtime_s']:.1f}s)", flush=True)

    census_rows = [results[c["id"]]["census_row"] for c in cells]
    families = {}
    for row in census_rows:
        families.setdefault(row["family"], []).append(row)
    family_summary = {fam: {"cell_count": len(rows), **recommend(rows)}
                      for fam, rows in families.items()}
    for c in cells:
        fam = c["family"]
        family_summary[fam].setdefault("n_no_discrete_mode_cells", 0)
        if results[c["id"]]["verdict"]["status"] == "no_discrete_mode":
            family_summary[fam]["n_no_discrete_mode_cells"] += 1
        family_summary[fam].setdefault(
            "verdict_basis", VERDICT_BASIS[c["kind"]])
    mack_note = (
        "measured fact, not an error: this family contains NO empty-box "
        "cells in this corpus because the production band filter retains "
        "damped modes at these parameters; mode death manifests as argmax "
        "handover plus omega_i sign flip along the alpha strip, not as an "
        "empty box.")
    if family_summary.get("mack_fig10_4_m10_3d", {}).get(
            "n_no_discrete_mode_cells") == 0:
        family_summary["mack_fig10_4_m10_3d"]["empty_box_note"] = mack_note
    l_max = max(item["L"] for item in family_summary.values())
    data_size = int(sum(p.stat().st_size for p in DATA_DIR.glob("*.npz")))

    cell_entries = []
    for c in cells:
        row = results[c["id"]]["census_row"]
        verdict = results[c["id"]]["verdict"]
        cell_entries.append({
            **{k: c[k] for k in ("id", "family", "kind", "stratum",
                                 "params")},
            **_cell_annotations(c, verdict, row),
            "npz": (Path("data") / _npz_name(c)).as_posix(),
            "npz_sha256": npz_arrays_sha256(DATA_DIR / _npz_name(c)),
            "box": box_for_cell(c),
            "verdict": verdict,
            "crosscheck": results[c["id"]]["crosscheck"],
        })

    strata = sorted({c["stratum"] for c in cells})
    notes = [
        "verdict_basis declares the candidate set each production verdict "
        "is computed over; box content and verdict emptiness are distinct "
        "facts for shift-invert-basis families.",
        "mack_fig10_4_m10_3d: " + mack_note,
    ]
    for entry in cell_entries:
        if entry.get("box_content_contradicts_verdict"):
            in_box = sum(
                ps["in_box_count"] for ps in
                results[entry["id"]]["census_row"]["per_spectrum"])
            notes.append(
                f"{entry['id']}: production verdict is no_discrete_mode "
                f"(shift-invert basis) while the full QZ spectrum holds "
                f"{in_box} in-box eigenvalues; both sets are stored.")

    manifest = {
        "status": "complete",
        "generated_at_unix": time.time(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            **_git_provenance(),
        },
        "cell_source": cell_source,
        "cell_count": len(cells),
        "n_strata": len(strata),
        "strata": strata,
        "notes": notes,
        "data_size_bytes": data_size,
        "storage_policy": {
            "npz_files_gitignored": True,
            "eigenvectors": "in-box + near-box candidates only",
            "storage_collar_rel": STORAGE_COLLAR_REL,
        },
        "cells": cell_entries,
    }
    census = {
        "status": "complete",
        "cell_count": len(cells),
        "l_max": int(l_max),
        "kappa_policy": {
            "nodes": 16,
            "rule": ("4 Gauss-Legendre nodes per rectangle edge "
                     "(never across corners)"),
            "contour": (f"production box inflated {KAPPA_MARGIN_REL:.0%} "
                        "per axis"),
            "equilibration": ("two-sided power-of-2 (rows then columns, "
                              "max-abs)"),
            "pencils": {
                "ozgen_pair": "A - c B (captured production assembly)",
                "mack_3d": "A - c B (production assembly + BCs)",
                "mazhong_spatial": "T(alpha) = C0 + alpha C1 + alpha^2 C2",
                "mach6_dense": "T(alpha) = A0 + alpha A1 + alpha^2 A2",
            },
        },
        "families": family_summary,
        "cells": census_rows,
        "runtime_s": float(time.perf_counter() - started),
        "data_size_bytes": data_size,
    }
    _json_dump(MANIFEST_PATH, manifest)
    _json_dump(CENSUS_PATH, census)

    if census_gate:
        _print_gate(census, census_gate)
    return manifest, census


def _print_gate(census, gate):
    for clause in (part.strip() for part in str(gate).split(",")
                   if part.strip()):
        if clause.startswith("l_max<="):
            limit = int(float(clause.split("<=", 1)[1]))
            val = int(census["l_max"])
            print(f"gate {clause}: {'ok' if val <= limit else 'EXCEEDED'} "
                  f"(l_max={val})", flush=True)
        elif clause.startswith("node_kappa_eq_p95<="):
            limit = float(clause.split("<=", 1)[1])
            worst = max((fam["node_kappa_equilibrated"]["p95"]
                         if fam["node_kappa_equilibrated"]["p95"] is not None
                         else math.inf)
                        for fam in census["families"].values())
            print(f"gate {clause}: {'ok' if worst <= limit else 'EXCEEDED'} "
                  f"(max family eq p95={worst:.6g})", flush=True)
        else:
            print(f"gate {clause}: unknown advisory clause", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="recompute cells whose NPZ already exists")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--census-gate", default=None,
                        help='advisory gates, e.g. '
                             '"l_max<=48,node_kappa_eq_p95<=1e7"')
    parser.add_argument("--force-reselect", action="store_true",
                        help="reselect cells from the production grid CSVs "
                             "instead of the frozen manifest definitions")
    args = parser.parse_args(argv)
    manifest, census = build(force=args.force, workers=args.workers,
                             census_gate=args.census_gate,
                             force_reselect=args.force_reselect)
    print(f"cell_count={manifest['cell_count']}")
    print(f"l_max={census['l_max']}")
    print(f"data_size_bytes={census['data_size_bytes']}")
    for fam, item in sorted(census["families"].items()):
        print(f"family {fam}: cells={item['cell_count']} "
              f"in_box[min/med/max]={item['in_box_count_min']}/"
              f"{item['in_box_count_median']:g}/{item['in_box_count_max']} "
              f"N_q={item['N_q']} L={item['L']} "
              f"margin={item['margin']:.4g} collar={item['collar']:.4g}")
    print(f"manifest={MANIFEST_PATH}")
    print(f"census={CENSUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
