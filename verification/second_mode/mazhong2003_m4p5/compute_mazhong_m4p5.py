"""Compute pyMack second-mode neutral branch points for the Ma & Zhong (2003)
Mach 4.5 flat-plate case at fixed F = 2.2e-4.

Reference (verified June 2026 from the original PDF,
http://cfdhost.seas.ucla.edu/papers/2003/Zhong-JFM-2003a.pdf):

    Ma, Y. & Zhong, X. (2003) "Receptivity of a supersonic boundary layer over
    a flat plate. Part 1." J. Fluid Mech. 488, 31-78.

    Flow conditions (== Kendall 1975 Mach 4.5 flat-plate experiment):
        M_inf = 4.5,  T*_inf = 65.15 K,  p*_inf = 728.4381557 Pa,  Pr = 0.72,
        Re*_inf (unit) = 7.2e6 /m,  Sutherland viscosity, air, gamma = 1.4.
        Steady base flow uses the ADIABATIC wall BC.
        F = omega* mu*_inf / (rho*_inf u*_inf^2) = omega* nu*_inf / U*_inf^2,
        and omega = R F  with  R = sqrt(Re_x) = u*_inf L* / nu*_inf,
        L* = sqrt(nu*_inf x* / u*_inf)  (eq. 8-11).

    Quote (sec. 6): "The second-mode unstable region for a fixed frequency of
    F = 2.2e-4 predicted by the LST is located in an interval from R = 806
    (branch I neutral stability point) to 999.6 (branch II neutral stability
    point)."  Fig. 15 neutral curve uses the ISOTHERMAL temperature-disturbance
    boundary condition (T'|_wall = 0): the paper states "the isothermal wall
    boundary condition is used for temperature perturbations" for the neutral
    curve (see MaZhong_2003_notes.md). The MEAN base flow is still adiabatic
    (insulated plate); only the disturbance T' BC is isothermal.

This script: builds the adiabatic CompressibleBlasiusProfile, sweeps R at fixed
F (omega = R*F), tracks the second mode (-alpha_i vs R), and brackets+refines
the two neutral crossings.  Length scale = L* (Mack/Malik convention), same as
validation/test_malik1990_case6_anchor.py for this exact M=4.5 base flow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pymack import CompressibleBlasiusProfile
from pymack.solver import solve_spatial

# --- Confirmed Ma & Zhong (2003) Mach 4.5 conditions ----------------------
MA = 4.5
T_EDGE_K = 65.15            # T*_inf (edge == freestream on a flat plate)
PR = 0.72
GAMMA = 1.4
SUTHERLAND_S_K = 110.33     # air Sutherland constant (110.4 also standard)
F = 2.2e-4                  # fixed dimensionless frequency

REF_BRANCH_I = 806.0
REF_BRANCH_II = 999.6

N = 130
Y_MAX = 40.0
LAMBDA_MU = 1.2             # Mack 2nd-viscosity convention (Malik mapping)
WALL_BC = "isothermal"      # disturbance temperature BC for Fig.15 (T'|_wall=0, per Ma & Zhong text)

HERE = Path(__file__).resolve().parent


def build_profile():
    # adiabatic base flow; T_wall guess = recovery temperature
    t_rec = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
    return CompressibleBlasiusProfile(
        Ma=MA, T_edge=T_EDGE_K, T_wall=t_rec, gamma=GAMMA, Pr=PR,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K, n_points=3000, eta_max=20.0,
    )


def second_mode_growth(profile, R, wall_bc=WALL_BC):
    """Return -alpha_i (spatial growth) of the second/Mack mode at this R.

    omega = R*F. Second-mode phase speed c ~ 1 - 1/M ~ 0.778, so the Mack root
    sits near alpha_r ~ omega/c.  We grab the full near-target set and select
    the genuine 2-D Mack mode (c in [0.85, 0.99] band like the Malik anchor;
    second mode of M=4.5 has c ~ 0.90-0.97 across this R range).
    """
    omega = R * F
    c_guess = 1.0 - 1.0 / MA
    alphas, _modes, _y = solve_spatial(
        profile, omega, R, MA, PR, GAMMA,
        N=N, y_max=Y_MAX, wall_bc=wall_bc,
        target_alpha=omega / c_guess + 0j, n_modes=25,
        length_scale="L_star", lambda_mu_ratio=LAMBDA_MU,
    )
    if alphas.size == 0:
        return np.nan, np.nan, np.nan
    c = omega / alphas.real
    # Mack / second-mode candidates: subsonic-relative phase speed, near-neutral
    cand_mask = (c > 0.80) & (c < 0.995) & (np.abs(alphas.imag) < 0.05) \
        & (alphas.real > 0.0)
    cand = alphas[cand_mask]
    if cand.size == 0:
        # fall back to least-damped physical mode with c in a wider band
        cand_mask = (c > 0.5) & (c < 1.05) & (alphas.real > 0.0)
        cand = alphas[cand_mask]
        if cand.size == 0:
            return np.nan, np.nan, np.nan
    # second mode = the one whose -alpha_i is largest (most unstable / least
    # damped) among the high-phase-speed candidates -> tracks the Mack band
    alpha = cand[np.argmin(cand.imag)]   # smallest imag = largest growth
    return -alpha.imag, float(alpha.real), float(omega / alpha.real)


def refine_crossing(profile, R_lo, R_hi, wall_bc=WALL_BC, tol=0.3, itmax=40):
    """Bisection on -alpha_i(R) = 0 between a stable and an unstable R."""
    g_lo = second_mode_growth(profile, R_lo, wall_bc)[0]
    g_hi = second_mode_growth(profile, R_hi, wall_bc)[0]
    if not (np.isfinite(g_lo) and np.isfinite(g_hi)) or g_lo * g_hi > 0:
        return np.nan
    for _ in range(itmax):
        R_mid = 0.5 * (R_lo + R_hi)
        g_mid = second_mode_growth(profile, R_mid, wall_bc)[0]
        if not np.isfinite(g_mid):
            return np.nan
        if abs(R_hi - R_lo) < tol:
            return R_mid
        if g_lo * g_mid <= 0:
            R_hi, g_hi = R_mid, g_mid
        else:
            R_lo, g_lo = R_mid, g_mid
    return 0.5 * (R_lo + R_hi)


def main():
    prof = build_profile()
    wall = prof(0.0)
    print(f"adiabatic wall T/T_edge = {wall['T']:.3f} "
          f"(recovery ~ {1 + 0.5*(GAMMA-1)*PR**0.5*MA**2:.3f})")

    # --- coarse sweep bracketing 700-1100 ---
    Rs = np.arange(700.0, 1101.0, 20.0)
    rows = []
    print("\n  R       omega      -alpha_i      alpha_r     c")
    for R in Rs:
        gi, ar, c = second_mode_growth(prof, R)
        rows.append((R, R * F, gi, ar, c))
        print(f"{R:7.1f}  {R*F:.5f}  {gi:+.6e}  {ar:9.5f}  {c:.4f}")

    arr = np.array([(r[0], r[2]) for r in rows], float)
    Rg, gi = arr[:, 0], arr[:, 1]
    good = np.isfinite(gi)
    Rg, gi = Rg[good], gi[good]

    # sign changes (stable<0 -> unstable>0 = branch I; unstable -> stable = branch II)
    crossings = []
    for i in range(len(Rg) - 1):
        if gi[i] * gi[i + 1] < 0:
            kind = "I" if gi[i] < 0 else "II"
            R_star = refine_crossing(prof, Rg[i], Rg[i + 1])
            crossings.append((kind, R_star, Rg[i], Rg[i + 1]))
            print(f"\n  crossing {kind}: bracket [{Rg[i]:.0f},{Rg[i+1]:.0f}] "
                  f"-> R = {R_star:.2f}")

    branch_I = next((c[1] for c in crossings if c[0] == "I"), np.nan)
    branch_II = next((c[1] for c in crossings if c[0] == "II"), np.nan)
    n_unstable_bands = sum(1 for i in range(len(gi) - 1) if gi[i] < 0 < gi[i + 1])

    out = {
        "F": F, "N": N, "lambda_mu_ratio": LAMBDA_MU, "wall_bc_disturbance": WALL_BC,
        "branch_I_R_pymack": None if np.isnan(branch_I) else round(branch_I, 2),
        "branch_II_R_pymack": None if np.isnan(branch_II) else round(branch_II, 2),
        "ref_branch_I": REF_BRANCH_I, "ref_branch_II": REF_BRANCH_II,
        "n_neutral_crossings": len(crossings),
        "n_unstable_band_entries": int(n_unstable_bands),
        "sweep": [{"R": r[0], "omega": r[1], "neg_alpha_i": r[2],
                   "alpha_r": r[3], "c": r[4]} for r in rows],
    }
    (HERE / "pymack_growth_sweep.json").write_text(json.dumps(out, indent=2))
    print("\n=== RESULT ===")
    print(f"Branch I  pyMack R = {branch_I:.2f}  (ref 806)   "
          f"rel-err {abs(branch_I-806)/806*100:.2f}%")
    print(f"Branch II pyMack R = {branch_II:.2f}  (ref 999.6) "
          f"rel-err {abs(branch_II-999.6)/999.6*100:.2f}%")
    print(f"crossings={len(crossings)}  unstable-band-entries={n_unstable_bands}")
    return out


if __name__ == "__main__":
    main()
