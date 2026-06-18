"""pyMack sharp-cone (Mangler) second-mode N-factor vs Sivasubramanian & Fasel
(2015) Mach-6 sharp 7-deg cone (BAM6QT).

Edge state from a Taylor-Maccoll solve (scratch_taylor_maccoll.py), cross-
checked against the BAM6QT design review (arxiv:2509.10411: Me~5.5 cone /
TM Me~5.6 for a 5-deg cone; our 7-deg gives Me=5.36 -- a larger cone -> lower
edge Mach, physically consistent and validating the TM solver).

POST-SHOCK CONE EDGE (Taylor-Maccoll, 7-deg, M_inf=6, T_inf=52.4 K):
    M_e = 5.356, T_e = 63.78 K, U_e = 857.5 m/s, nu_e = 6.0628e-5 m2/s,
    Re_unit_e = 1.414e7 /m, Tw/Te = 4.704 (Tw = 300 K isothermal).

Method mirrors validation/test_malik1990_case6_anchor.py:
  CompressibleBlasiusProfile (Sutherland, isothermal wall) + solve_spatial,
  length_scale='L_star', lambda_mu_ratio=1.2, second-mode (c_r~0.9) band.
Cone bookkeeping via pymack.cone: s_mm -> R_eq=sqrt(Re_s/3); N=int 6 sigma_L dR_eq.
"""
import os
os.environ.setdefault("PYMACK_NO_BANNER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json
from pathlib import Path
import numpy as np

REPO = Path("C:/Users/merts/OneDrive/Masaüstü/MS_LST")
sys.path.insert(0, str(REPO))

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_spatial
from pymack.scales import DimensionalEdgeState, frequency_khz_to_F, F_to_frequency_khz
from pymack import cone

# --- Post-shock cone edge state (Taylor-Maccoll) ---------------------------
M_E   = 5.356
T_E   = 63.78      # K
U_E   = 857.5      # m/s
NU_E  = 6.0628e-5  # m2/s
RE_UNIT_E = U_E / NU_E
TW_OVER_TE = 300.0 / T_E
GAMMA = 1.4
PR = 0.72
SUTH_S_K = 110.4

edge = DimensionalEdgeState(U_e=U_E, nu_e=NU_E, T_e=T_E, M_e=M_E, gamma=GAMMA,
                            gas="air", unit_reynolds_per_m=RE_UNIT_E)

# --- Mean flow (isothermal Tw=300 K, Sutherland), like the Malik anchor ----
profile = CompressibleBlasiusProfile(
    Ma=M_E, T_edge=T_E, T_wall=300.0, gamma=GAMMA, Pr=PR,
    wall_bc="isothermal", viscosity_model="sutherland",
    sutherland_S=SUTH_S_K, n_points=3000, eta_max=20.0,
)

# --- Surface-distance window along the cone ray ----------------------------
# S&F (2015) cone region of strong second-mode growth is roughly s ~ 0.2-0.5 m.
# Map to R_eq = sqrt(Re_s/3).
s_mm = np.linspace(120.0, 520.0, 28)        # surface distance along ray
R_eq = np.array([float(cone.cone_s_mm_to_R_eq(s, edge)) for s in s_mm])
print("s_mm range", s_mm[0], s_mm[-1], "-> R_eq range", R_eq[0], R_eq[-1])

# Frequencies to scan (kHz). Benchmark most-amplified band ~220-300 kHz.
freqs_khz = np.array([150., 180., 210., 240., 270., 300., 330., 360., 400.])

N_COLLOC = 90
Y_MAX = 40.0

def second_mode_sigma(omega_L, R):
    """Return sigma_L = -Im(alpha_L) for the discrete second mode, or nan."""
    try:
        alphas, _m, _y = solve_spatial(
            profile, float(omega_L), float(R), M_E, PR, GAMMA,
            N=N_COLLOC, y_max=Y_MAX, wall_bc="isothermal",
            target_alpha=omega_L / 0.90, n_modes=12,
            length_scale="L_star", lambda_mu_ratio=1.2,
        )
    except Exception as exc:  # noqa: BLE001
        return np.nan, np.nan
    c = omega_L / alphas.real
    band = (c > 0.80) & (c < 0.985) & (np.abs(alphas.imag) < 0.05) & (alphas.real > 0)
    cand = alphas[band]
    if cand.size == 0:
        return np.nan, np.nan
    # most amplified (most negative alpha_i) among second-mode candidates
    best = cand[np.argmin(cand.imag)]
    return float(-best.imag), float(omega_L / best.real)

results = {}
for f_khz in freqs_khz:
    F = float(frequency_khz_to_F(f_khz, edge))
    sig = np.full(R_eq.size, np.nan)
    cr = np.full(R_eq.size, np.nan)
    for i, R in enumerate(R_eq):
        omega_L = F * R          # omega_L = F * R_eq (since F=omega nu/U^2, R=U L*/nu)
        s, c = second_mode_sigma(omega_L, R)
        sig[i] = s
        cr[i] = c
    # cone N-factor integral (clip negative = transition-envelope convention)
    finite = np.isfinite(sig)
    if finite.sum() >= 2:
        Rf = R_eq[finite]; sf = sig[finite]
        order = np.argsort(Rf)
        out = cone.cone_n_factor(sf[order], Rf[order], clip_negative=True)
        Npeak = float(out["N"][-1])
    else:
        Npeak = float("nan")
    nfin = int(finite.sum())
    results[float(f_khz)] = {"F": F, "N_peak": Npeak, "n_finite": nfin,
                             "sigma_max": float(np.nanmax(sig)) if nfin else float("nan"),
                             "cr_med": float(np.nanmedian(cr)) if nfin else float("nan")}
    print(f"f={f_khz:6.1f} kHz  F={F:.4e}  n_finite={nfin:2d}/{R_eq.size}  "
          f"sigma_max={results[float(f_khz)]['sigma_max']:.4e}  N_peak={Npeak:.3f}  "
          f"c_r~{results[float(f_khz)]['cr_med']:.3f}")

# Peak-N frequency
valid = {f: d for f, d in results.items() if np.isfinite(d["N_peak"])}
if valid:
    f_peak = max(valid, key=lambda f: valid[f]["N_peak"])
    print(f"\nPEAK-N frequency = {f_peak:.1f} kHz  N_peak = {valid[f_peak]['N_peak']:.3f}")
out = {"edge": edge.to_dict(), "Tw_over_Te": TW_OVER_TE, "s_mm": s_mm.tolist(),
       "R_eq": R_eq.tolist(), "results": results,
       "f_peak_khz": (f_peak if valid else None),
       "N_peak": (valid[f_peak]["N_peak"] if valid else None)}
Path("scratch_cone_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote scratch_cone_result.json")
