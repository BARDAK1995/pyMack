"""Robust discrete-mode extractor for the Ozgen 2D temporal EVP.

The original neutral-curve generator picked the most-unstable eigenvalue inside
a fixed phase-speed band (c_r<0.45 TS, 0.88<c_r<0.99 Mack).  That band is wrong
for the first mode: at moderate Mach the genuine first mode sits at c_r~0.5-0.6
(excluded), while continuous-spectrum junk at c_r~0.2-0.4 (admitted) flooded the
map.  See the diagnosis in this folder.

This replaces the c_r band with a PHYSICAL discriminant: a discrete boundary-layer
eigenmode has an eigenfunction that DECAYS at the freestream (top of the domain),
whereas continuous-spectrum modes do not.  We additionally require the discrete
mode to be y_max-STATIONARY (its c agrees between two domain heights), since the
slow-acoustic continuous spectrum (c_r <= 1-1/Ma) overlaps the first mode in c_r
and only the true discrete mode is domain-insensitive.

Returns the discrete mode of interest (most-unstable decaying, y_max-stable mode
in a broad physical band), or None when only continuous-spectrum modes exist
there (an honest "cannot isolate" rather than a fake value).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pymack import make_flatplate_profile  # noqa: E402
from pymack.temporal_solver import solve_temporal_2d  # noqa: E402
from pymack.scales import delta_star_over_lstar  # noqa: E402

_PROFILE_CACHE: dict[float, object] = {}


def _profile(Ma: float):
    if Ma not in _PROFILE_CACHE:
        _PROFILE_CACHE[Ma] = make_flatplate_profile(Ma)
    return _PROFILE_CACHE[Ma]


def _decaying_candidates(ev, vec, y, *, ci_abs_max, cr_band, fs_thresh):
    """Return list of (c, fs_amp) for modes that decay at the freestream."""
    n = len(y)
    ntop = max(2, n // 10)            # top ~10% of domain = freestream region
    out = []
    for k in range(len(ev)):
        c = ev[k]
        if not (abs(c.imag) < ci_abs_max and cr_band[0] < c.real < cr_band[1]):
            continue
        u = vec[0:n, k]
        v = vec[n:2 * n, k]
        T = vec[2 * n:3 * n, k]
        amp = np.abs(u) + np.abs(v) + np.abs(T)
        mx = amp.max()
        if mx <= 0:
            continue
        fs = amp[:ntop].max() / mx     # index 0 = freestream; small => decays
        if fs < fs_thresh:
            out.append((complex(c), float(fs)))
    return out


def discrete_mode(
    Ma, Re, alpha, *,
    N=240,
    ymf_pair=(35.0, 45.0),
    ci_abs_max=0.05,
    cr_band=(0.05, 0.97),
    fs_thresh=0.06,
    match_tol=4e-3,
):
    """Most-unstable discrete (decaying, y_max-stable) mode at (Re, alpha).

    Returns dict(c_r, c_i, fs, n_match) or None if only continuous-spectrum
    modes exist (cannot isolate a discrete BL mode here).
    """
    prof = _profile(float(Ma))
    d = delta_star_over_lstar(prof)
    cand_by_ymf = []
    for ymf in ymf_pair:
        ev, vec, y = solve_temporal_2d(
            prof, float(alpha), float(Re), float(Ma),
            N=N, y_max=ymf * d, length_scale="L_star",
        )
        cand_by_ymf.append(
            _decaying_candidates(ev, vec, y, ci_abs_max=ci_abs_max,
                                 cr_band=cr_band, fs_thresh=fs_thresh)
        )
    a, b = cand_by_ymf
    if not a or not b:
        return None
    bc = np.array([c for c, _ in b])
    # y_max-stable: a candidate from the taller domain whose c matches one in
    # the shorter domain (discrete modes are domain-insensitive).
    matched = []
    for c, fs in a:
        if np.min(np.abs(c - bc)) < match_tol:
            matched.append((c, fs))
    if not matched:
        return None
    matched.sort(key=lambda r: -r[0].imag)
    c, fs = matched[0]
    return {"c_r": float(c.real), "c_i": float(c.imag),
            "fs": float(fs), "n_match": len(matched)}


if __name__ == "__main__":
    # Validation: sample pyMack's discrete first mode along/inside Ozgen's M2
    # unstable lobe and report agreement (sign of c_i) + coverage.
    import csv

    HERE = Path(__file__).resolve().parent
    ref = HERE.parent.parent / "reference_data/digitized/ozgen_fig3_M2_neutral_v2.csv"
    # fall back to repo-root reference_data
    ref = REPO / "reference_data/digitized/ozgen_fig3_M2_neutral_v2.csv"
    upper, lower = [], []
    with open(ref) as f:
        for row in csv.DictReader(f):
            (upper if row["lobe"] == "upper" else lower).append(
                (float(row["Re"]), float(row["alpha"]))
            )
    upper.sort(); lower.sort()
    up = np.array(upper); lo = np.array(lower)

    # sample mid-lobe alpha (between lower and upper branch) at a set of Re
    print("M2 discrete first mode vs Ozgen lobe (mid-lobe alpha):")
    print(" Re      alpha_mid  pyMack-discrete-mode        verdict")
    res_re = np.array([450, 600, 800, 1100, 1500, 2000, 2800, 4000])
    for Re in res_re:
        a_up = np.interp(Re, up[:, 0], up[:, 1])
        a_lo = np.interp(Re, lo[:, 0], lo[:, 1])
        a_mid = 0.5 * (a_up + a_lo)
        m = discrete_mode(2.0, float(Re), float(a_mid), N=240)
        if m is None:
            print(f" {Re:5.0f}  {a_mid:.4f}   NO discrete mode (CS-contaminated)")
        else:
            verdict = "UNSTABLE (agrees: Ozgen unstable here)" if m["c_i"] > 0 else "stable (disagrees)"
            print(f" {Re:5.0f}  {a_mid:.4f}   c={m['c_r']:.3f}{m['c_i']:+.4f}i fs={m['fs']:.3f}  {verdict}")
