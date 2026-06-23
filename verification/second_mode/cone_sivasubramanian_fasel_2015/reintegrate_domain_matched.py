"""Domain-matched N-factor re-integration for the Sivasubramanian-Fasel
Mach-6 sharp 7-deg cone (BAM6QT), anchored to the OG thesis Fig. 5.1.

Sivasubramanian (2012) PhD thesis, Fig. 5.1 (PDF page 136), states verbatim:
  - The most amplified axisymmetric wave (kc=0) reaches N ~= 9.5.
  - That wave has F = 1.071e-5  (f* ~= 210 kHz).
  - N is defined from the LOWER NEUTRAL POINT: N = ln(A/A_0), A_0 at the
    lower neutral point.
  - Fig. 5.1 x* (axial distance, m) spans ~0.30 -> ~0.59 m; the curve emerges
    from its lower neutral point near x* ~= 0.30-0.31 m and peaks (N~9.5) at the
    domain edge x* ~= 0.58-0.59 m (the computational outflow x*_L = 0.6 m, i.e.
    in the buffer region beyond the 0.5 m physical cone).

Geometry (thesis Ch.3, p.52 / Fig.4.1, p.57):
  sharp cone, half-angle theta_c = 7 deg, length 0.5 m, nose r = 0.05 mm.
  BAM6QT: M_inf=6, T0=430 K, T_inf=52.4 K, Re/m ~ 10.5e6 (thesis) /11.2e6 (review).

x* nondimensionalization: x* is the DIMENSIONAL AXIAL distance in METERS along
the cone axis (values quoted as 0.138 m, 0.250 m, 0.58 m, etc.).  pyMack works
in SURFACE distance s along the ray; for a sharp cone s = x*/cos(theta_c).

This script:
  (1) maps the thesis x*-domain to pyMack's surface s and R_eq=sqrt(Re_s/3)
      using the SAME post-shock Taylor-Maccoll edge state as compute_cone_sf2015,
  (2) re-runs pyMack's spatial second-mode solver over that matched physical
      extent, starting from the lower neutral point (clip_negative absorbs the
      pre-growth region exactly like the thesis A_0 reference),
  (3) sweeps frequency to find pyMack's own peak-N frequency, and
  (4) reports pyMack domain-matched peak N vs thesis 9.5 and most-amplified
      frequency vs 210 kHz.  Honest: if pyMack falls short of 9.5, it is stated.
"""
import os
os.environ.setdefault("PYMACK_NO_BANNER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, math
from pathlib import Path
import numpy as np

REPO = Path("C:/Users/merts/OneDrive/Masaüstü/MS_LST")
sys.path.insert(0, str(REPO))

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_spatial
from pymack.scales import DimensionalEdgeState, frequency_khz_to_F, F_to_frequency_khz
from pymack import cone

HERE = Path(__file__).resolve().parent

# --- Post-shock cone edge state (Taylor-Maccoll), identical to compute_cone_sf2015 ---
M_E   = 5.356
T_E   = 63.78      # K
U_E   = 857.5      # m/s
NU_E  = 6.0628e-5  # m2/s
RE_UNIT_E = U_E / NU_E
GAMMA = 1.4
PR = 0.72
SUTH_S_K = 110.4
THETA_C_DEG = 7.0
COS_TC = math.cos(math.radians(THETA_C_DEG))

edge = DimensionalEdgeState(U_e=U_E, nu_e=NU_E, T_e=T_E, M_e=M_E, gamma=GAMMA,
                            gas="air", unit_reynolds_per_m=RE_UNIT_E)

profile = CompressibleBlasiusProfile(
    Ma=M_E, T_edge=T_E, T_wall=300.0, gamma=GAMMA, Pr=PR,
    wall_bc="isothermal", viscosity_model="sutherland",
    sutherland_S=SUTH_S_K, n_points=3000, eta_max=20.0,
)

# --- Thesis Fig. 5.1 physical domain (AXIAL x*, meters) ---
# Lower neutral point of the F=1.071e-5 (kc=0) curve ~ 0.30 m; we start a hair
# below it so the lower neutral point falls strictly inside the swept window and
# clip_negative anchors N=0 at the true neutral point (the thesis A_0).
X_STAR_START_M = 0.30      # axial; just below the bold-curve lower neutral point
X_STAR_END_M   = 0.59      # axial; thesis domain edge where N peaks ~9.5
# surface distance s = x*/cos(theta_c)
s_start_mm = 1000.0 * X_STAR_START_M / COS_TC
s_end_mm   = 1000.0 * X_STAR_END_M / COS_TC

N_STATIONS = 60            # fine grid over the matched window
s_mm = np.linspace(s_start_mm, s_end_mm, N_STATIONS)
R_eq = np.array([float(cone.cone_s_mm_to_R_eq(s, edge)) for s in s_mm])

print(f"theta_c = {THETA_C_DEG} deg  cos = {COS_TC:.5f}")
print(f"thesis x* axial   : {X_STAR_START_M:.3f} -> {X_STAR_END_M:.3f} m")
print(f"surface s         : {s_start_mm:.2f} -> {s_end_mm:.2f} mm")
print(f"R_eq=sqrt(Re_s/3) : {R_eq[0]:.1f} -> {R_eq[-1]:.1f}")
print(f"Re_unit_edge      : {RE_UNIT_E:.4e} /m   (freestream review 11.2e6/m)")

# Frequency sweep (kHz). Fine around 210 to locate pyMack's own peak-N freq.
freqs_khz = np.array([170., 180., 190., 200., 205., 210., 215., 220.,
                      230., 240., 250., 260., 280., 300.])

N_COLLOC = 90
Y_MAX = 40.0

def second_mode_sigma(omega_L, R):
    try:
        alphas, _m, _y = solve_spatial(
            profile, float(omega_L), float(R), M_E, PR, GAMMA,
            N=N_COLLOC, y_max=Y_MAX, wall_bc="isothermal",
            target_alpha=omega_L / 0.90, n_modes=12,
            length_scale="L_star", lambda_mu_ratio=1.2,
        )
    except Exception:
        return np.nan, np.nan
    c = omega_L / alphas.real
    band = (c > 0.80) & (c < 0.985) & (np.abs(alphas.imag) < 0.05) & (alphas.real > 0)
    cand = alphas[band]
    if cand.size == 0:
        return np.nan, np.nan
    best = cand[np.argmin(cand.imag)]
    return float(-best.imag), float(omega_L / best.real)

results = {}
sigma_at_210 = None
for f_khz in freqs_khz:
    F = float(frequency_khz_to_F(f_khz, edge))
    sig = np.full(R_eq.size, np.nan)
    cr  = np.full(R_eq.size, np.nan)
    for i, R in enumerate(R_eq):
        omega_L = F * R
        s, c = second_mode_sigma(omega_L, R)
        sig[i] = s; cr[i] = c
    finite = np.isfinite(sig)
    if finite.sum() >= 2:
        Rf = R_eq[finite]; sf = sig[finite]
        order = np.argsort(Rf)
        out = cone.cone_n_factor(sf[order], Rf[order], clip_negative=True)
        Npeak = float(out["N"][-1])
        Ncurve = out["N"][order.argsort()] if False else out["N"]
        Rcurve = Rf[order]
    else:
        Npeak = float("nan"); Ncurve = None; Rcurve = None
    results[float(f_khz)] = {
        "F": F, "N_peak": Npeak, "n_finite": int(finite.sum()),
        "sigma_max": float(np.nanmax(sig)) if finite.sum() else float("nan"),
        "cr_med": float(np.nanmedian(cr)) if finite.sum() else float("nan"),
        "sigma_L": sig.tolist(),
        "N_curve": (Ncurve.tolist() if Ncurve is not None else None),
        "R_curve": (Rcurve.tolist() if Rcurve is not None else None),
    }
    if abs(f_khz - 210.0) < 1e-6:
        sigma_at_210 = sig.copy()
    print(f"f={f_khz:6.1f} kHz  F={F:.4e}  nfin={int(finite.sum()):2d}/{R_eq.size}  "
          f"sigma_max={results[float(f_khz)]['sigma_max']:.4e}  N_peak={Npeak:.3f}  "
          f"c_r~{results[float(f_khz)]['cr_med']:.3f}")

valid = {f: d for f, d in results.items() if np.isfinite(d["N_peak"])}
f_peak = max(valid, key=lambda f: valid[f]["N_peak"]) if valid else None
print()
if f_peak is not None:
    print(f"pyMack DOMAIN-MATCHED peak-N freq = {f_peak:.1f} kHz   N_peak = {valid[f_peak]['N_peak']:.3f}")
print(f"pyMack N at 210 kHz (thesis primary) = {results[210.0]['N_peak']:.3f}")
print(f"thesis Fig 5.1 peak N ~= 9.5 at f ~= 210 kHz (F=1.071e-5)")

out = {
    "edge": edge.to_dict(),
    "theta_c_deg": THETA_C_DEG, "cos_theta_c": COS_TC,
    "x_star_axial_m": [X_STAR_START_M, X_STAR_END_M],
    "s_surface_mm": [s_start_mm, s_end_mm],
    "R_eq_window": [R_eq[0], R_eq[-1]],
    "s_mm": s_mm.tolist(), "R_eq": R_eq.tolist(),
    "freqs_khz": freqs_khz.tolist(),
    "results": results,
    "f_peak_khz": f_peak,
    "N_peak_domain_matched": (valid[f_peak]["N_peak"] if f_peak is not None else None),
    "N_at_210khz": results[210.0]["N_peak"],
    "thesis_peak_N": 9.5, "thesis_peak_f_khz": 210.0, "thesis_F": 1.071e-5,
}
(HERE / "domain_matched_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote domain_matched_result.json")
