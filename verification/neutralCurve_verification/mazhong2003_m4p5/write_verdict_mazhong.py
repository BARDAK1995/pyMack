"""Write verdict.json for the Ma & Zhong (2003) Mach 4.5 second-mode neutral
branch verification (case mazhong2003_m4p5).

Compares pyMack's two second-mode neutral-branch Reynolds numbers (R = sqrt(Re_x))
at fixed F = 2.2e-4 against the values printed in Ma & Zhong (2003) JFM 488:31-78:
Branch I (lower) R = 806, Branch II (upper) R = 999.6.

Headline = classify_relative on max(branch_I_rel_err, branch_II_rel_err).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # verification/
from _compare_lib import classify_relative, write_verdict  # noqa: E402

# pyMack results (adiabatic temperature-disturbance BC == Fig.15 neutral curve;
# N=130, length_scale=L*, lambda_mu_ratio=1.2; N=160 agrees to <0.2 in R).
BRANCH_I_PYMACK = 831.0
BRANCH_II_PYMACK = 1033.4
BRANCH_I_REF = 806.0
BRANCH_II_REF = 999.6

# Isothermal-disturbance-BC sensitivity (recorded for honesty): I=786.5, II=1010.0
ISO_I, ISO_II = 786.5, 1010.0


def main():
    bI_err = abs(BRANCH_I_PYMACK - BRANCH_I_REF) / BRANCH_I_REF
    bII_err = abs(BRANCH_II_PYMACK - BRANCH_II_REF) / BRANCH_II_REF
    headline = max(bI_err, bII_err)
    topology_ok = True  # single closed second-mode band, exactly 2 neutral points
    verdict = classify_relative(headline, topology_ok)

    iso_I_err = abs(ISO_I - BRANCH_I_REF) / BRANCH_I_REF
    iso_II_err = abs(ISO_II - BRANCH_II_REF) / BRANCH_II_REF

    reason = (
        f"At fixed F=2.2e-4, pyMack (adiabatic disturbance BC, == Fig.15) places "
        f"the second-mode Branch I (lower) neutral point at R={BRANCH_I_PYMACK:.1f} "
        f"vs Ma & Zhong R=806 ({bI_err*100:.2f}%) and Branch II (upper) at "
        f"R={BRANCH_II_PYMACK:.1f} vs R=999.6 ({bII_err*100:.2f}%). Topology matches: "
        f"a single closed unstable band bounded by exactly two neutral points. "
        f"Both branch errors are below the 5% digitization-floor threshold "
        f"(headline = max = {headline*100:.2f}%) -> agrees. N-converged (N=130 vs "
        f"N=160 differ <0.2 in R). The isothermal disturbance BC gives I={ISO_I:.1f} "
        f"({iso_I_err*100:.2f}%), II={ISO_II:.1f} ({iso_II_err*100:.2f}%) -- also "
        f"within 5%, confirming robustness to the temperature-perturbation BC choice."
    )

    v = {
        "case_id": "mazhong2003_m4p5",
        "category": "neutral_curve",
        "source": "Ma & Zhong (2003) JFM 488:31-78, Part 1, sec. 6 / Fig. 15",
        "conditions": {
            "Ma": 4.5,
            "gas": "air",
            "wall": "adiabatic",
            "psi_deg": 0,
            "formulation": "spatial second mode (Mack), F fixed",
            "transport": "Sutherland viscosity (S=110.33 K), constant Pr=0.72",
            "gamma": 1.4,
            "T_edge_K": 65.15,
            "p_inf_Pa": 728.4381557,
            "unit_Re_per_m": 7.2e6,
            "F": 2.2e-4,
            "length_scale": "L* = sqrt(nu_e x / U_e); R = sqrt(Re_x); omega = R*F",
            "disturbance_temperature_BC": "adiabatic (Fig.15 neutral curve)",
            "N": 130,
            "lambda_mu_ratio": 1.2,
        },
        "quantity": (
            "second-mode neutral-branch Reynolds numbers R=sqrt(Re_x) at fixed "
            "F=2.2e-4: Branch I (lower) and Branch II (upper)"
        ),
        "metrics": {
            "branch_I_R_pymack": round(BRANCH_I_PYMACK, 2),
            "branch_I_R_ref": BRANCH_I_REF,
            "branch_I_rel_err": round(bI_err, 5),
            "branch_II_R_pymack": round(BRANCH_II_PYMACK, 2),
            "branch_II_R_ref": BRANCH_II_REF,
            "branch_II_rel_err": round(bII_err, 5),
            "F": 2.2e-4,
            "topology_ok": topology_ok,
            "headline_max_rel_err": round(headline, 5),
            "iso_BC_branch_I_R_pymack": ISO_I,
            "iso_BC_branch_I_rel_err": round(iso_I_err, 5),
            "iso_BC_branch_II_R_pymack": ISO_II,
            "iso_BC_branch_II_rel_err": round(iso_II_err, 5),
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "generated": "new",
        "artifacts": {
            "pymack": "verification/neutralCurve_verification/mazhong2003_m4p5/"
                      "pymack_growth_sweep.json",
            "reference": None,
            "overlay": None,
        },
        "pymack_provenance": (
            "verification/neutralCurve_verification/mazhong2003_m4p5/"
            "compute_mazhong_m4p5.py (this session, 2026-06). CompressibleBlasius "
            "adiabatic M=4.5 base flow (Sutherland, S=110.33 K, Pr=0.72, gamma=1.4) "
            "at T_edge=65.15 K; solve_spatial with length_scale='L_star', "
            "lambda_mu_ratio=1.2, N=130, y_max=40; fixed F=2.2e-4 (omega=R*F); "
            "coarse R-sweep 700-1100 then bisection on -alpha_i=0 for the two "
            "second-mode neutral crossings. Same M=4.5 base-flow setup as the "
            "CI-gated Malik (1990) Case-6 anchor (validation/"
            "test_malik1990_case6_anchor.py). Reference values read directly from "
            "the Ma & Zhong (2003) PDF (sec. 6: 'interval from R = 806 (branch I "
            "neutral stability point) to 999.6 (branch II neutral stability "
            "point)')."
        ),
    }

    path = write_verdict(HERE, v)
    print(f"Branch I  : pyMack {BRANCH_I_PYMACK:.1f} vs 806   -> {bI_err*100:.2f}%")
    print(f"Branch II : pyMack {BRANCH_II_PYMACK:.1f} vs 999.6 -> {bII_err*100:.2f}%")
    print(f"headline max rel-err = {headline*100:.2f}%  topology_ok={topology_ok}")
    print(f"VERDICT = {verdict}")
    print(f"written -> {path}")
    return v


if __name__ == "__main__":
    main()
